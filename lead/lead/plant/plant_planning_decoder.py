from __future__ import annotations

import jaxtyping as jt
import torch
from beartype import beartype

from lead.tfv6.planning_decoder import PlanningDecoder, decode_two_hot
from lead.training.config_training import TrainingConfig


class PlantPlanningDecoder(PlanningDecoder):
    """PlanningDecoder that accepts pre-computed context tokens."""

    @beartype
    def __init__(self, config: TrainingConfig, device: torch.device):
        super().__init__(input_bev_channels=1, config=config, device=device)
        del self.planning_context_encoder

    @beartype
    def forward(
        self,
        context_tokens: jt.Float[torch.Tensor, "bs seq_len dim"],
        data: dict,
        log: dict,
    ) -> tuple[
        jt.Float[torch.Tensor, "B n_checkpoints 2"] | None,
        jt.Float[torch.Tensor, "B n_waypoints 2"] | None,
        jt.Float[torch.Tensor, "B speed_classes"] | None,
        jt.Float[torch.Tensor, " B"] | None,
    ]:
        self.kv = context_tokens
        bs = context_tokens.shape[0]

        queries = self.transformer_decoder(self.query.repeat(bs, 1, 1), context_tokens)

        # Split queries
        query_idx = 0
        route = waypoints = target_speed_dist = target_speed_scalar = None

        if self.config.predict_spatial_path:
            route_queries = queries[
                :,
                query_idx : query_idx + self.config.num_route_points_prediction,
            ]
            route = torch.cumsum(self.route_decoder(route_queries), 1)
            query_idx += self.config.num_route_points_prediction

        if self.config.predict_temporal_spatial_waypoints:
            wp_queries = queries[
                :,
                query_idx : query_idx + self.config.num_way_points_prediction,
            ]
            waypoints = torch.cumsum(self.wp_decoder(wp_queries), 1)
            query_idx += self.config.num_way_points_prediction

        if self.config.predict_target_speed:
            target_speed_query = queries[:, query_idx]
            target_speed_dist = self.target_speed_decoder(target_speed_query)

            with torch.amp.autocast(device_type="cuda", enabled=False):
                target_speed_softmax = torch.softmax(target_speed_dist.float(), dim=-1)
                target_speed_scalar = decode_two_hot(
                    target_speed_softmax,
                    self.config.target_speed_classes,
                    self.device,
                )

        return route, waypoints, target_speed_dist, target_speed_scalar
