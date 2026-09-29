from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
from beartype import beartype
from transformers import BertConfig, BertModel

from lead.plant.plant_config import PlantConfig
from lead.plant.plant_variables import PlantVariables

logger = logging.getLogger(__name__)


class PlantBackbone(nn.Module):
    """BERT-based object token encoder.

    Encodes object tokens (vehicles, pedestrians, traffic lights, etc.),
    route waypoints, speed limit, and optional HD map / ego-speed into a
    sequence of context tokens for downstream planning decoders.
    """

    # 0: padding, 1: vehicle, 2: pedestrian, 3: static,
    # 4: stop_sign, 5: traffic_light, 6: emergency vehicle
    OBJECT_TYPES = 7
    NUM_ATTRIBUTES = 6  # x, y, yaw, speed/id, extent_x, extent_y

    @beartype
    def __init__(self, device: torch.device, config: PlantConfig):
        super().__init__()
        self.device = device
        self.config = config

        # BERT transformer
        bert_config = BertConfig.from_pretrained(config.plant_hf_checkpoint)
        self.n_embd = bert_config.hidden_size
        self.model = BertModel(config=bert_config)
        # We feed custom embeddings, not word tokens
        self.model.embeddings.word_embeddings = None
        self.model.pooler = None

        # Per-type object token embeddings
        self.tok_emb = nn.ModuleList(
            nn.Linear(self.NUM_ATTRIBUTES, self.n_embd)
            for _ in range(self.OBJECT_TYPES)
        )

        # Route embedding: num_route_points x 2 coords -> hidden_size
        self.route_emb = nn.Linear(config.plant_num_route_points * 2, self.n_embd)

        # Speed limit embedding: one slot per category in PlantVariables.speed_cats
        self.speed_emb = nn.Embedding(len(PlantVariables.speed_cats), self.n_embd)

        # Optional HD map encoder
        if config.plant_input_hdmap:
            self.hdmap_encoder = timm.create_model(
                "resnet18",
                pretrained=True,
                num_classes=self.n_embd,
            )

        # Optional ego speed encoder
        if config.plant_input_ego_speed:
            self.ego_speed_emb = nn.Linear(1, self.n_embd)

        # Dropout
        if config.plant_use_dropout:
            self.drop = nn.Dropout(config.plant_embd_pdrop)

        # Projection from BERT hidden_size to planning decoder token dim
        self.project = nn.Linear(self.n_embd, config.transfuser_token_dim)

        self.apply(self._init_weights)

        logger.info(
            "PlantBackbone parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear | nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @beartype
    def forward(self, data: dict) -> torch.Tensor:
        """Encode object tokens into context tokens for the planning decoder.

        Args:
            data: Batch dict with keys ``idxs``, ``x_objs``,
                ``route_original``, ``speed_limit``, and optionally
                ``hdmap``, ``input_ego_speed``.

        Returns:
            Context tokens of shape ``[bs, seq_len, transfuser_token_dim]``.
        """
        batch_idxs = data["idxs"].to(self.device, non_blocking=True)
        x_batch_objs = data["x_objs"].to(self.device, non_blocking=True)
        route_batch = data["route_original"].to(self.device, non_blocking=True)
        speed_limit_batch = data["speed_limit"].to(self.device, non_blocking=True)

        # Embed each object token by its type.
        # Allocate in the autocast dtype so index_put matches emb_layer output.
        embed_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else x_batch_objs.dtype
        )
        embedding = torch.zeros(
            (*x_batch_objs.shape[:-1], self.n_embd),
            device=x_batch_objs.device,
            dtype=embed_dtype,
        )
        for i, emb_layer in enumerate(self.tok_emb):
            mask = x_batch_objs[..., 0] == i
            if mask.any():
                # First element is class index, remaining are attributes
                embedding[mask] = emb_layer(x_batch_objs[mask, 1:])

        # Restore batch shape from flat object tensor via index mapping
        embedding = embedding[batch_idxs]

        # Prepend route token
        route_tok = self.route_emb(route_batch.flatten(1))[:, None]
        embedding = torch.cat((route_tok, embedding), dim=1)

        # Prepend speed limit token
        speed_tok = self.speed_emb(speed_limit_batch)[:, None]
        embedding = torch.cat((speed_tok, embedding), dim=1)

        # Optional ego speed token
        if self.config.plant_input_ego_speed:
            ego_speed = data["input_ego_speed"].to(self.device, non_blocking=True)
            ego_speed_tok = self.ego_speed_emb(ego_speed[:, None])
            ego_speed_tok = ego_speed_tok[:, None]
            embedding = torch.cat((ego_speed_tok, embedding), dim=1)

        # Optional HD map token
        if self.config.plant_input_hdmap:
            hdmap = data["hdmap"].to(self.device, non_blocking=True)
            hdmap_tok = self.hdmap_encoder(hdmap)[:, None]
            embedding = torch.cat((hdmap_tok, embedding), dim=1)

        # Dropout
        if self.config.plant_use_dropout:
            embedding = self.drop(embedding)

        # BERT encoder
        output = self.model(inputs_embeds=embedding)
        context_tokens = output.last_hidden_state

        # Project to planning decoder dimension
        context_tokens = self.project(context_tokens)

        return context_tokens
