# Copyright 2026 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Literal, Optional, List

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import logging
from mmdet.models import HEADS
from mmdet.models.builder import build_head, build_loss
from mmdet.models.losses import accuracy
from timm.models.layers import Mlp
from einops import rearrange
from .unidrivevla_vlm_qwenvl3 import Qwen3VLWithExpertModel
from torch.nn.utils.rnn import pad_sequence
from .flex_attention_opt import build_blockmask_unidrive
from .constants import (
    B2D_SYSTEM_PROMPT,
    B2D_USER_PROMPT_TEMPLATE,
    B2D_VIEW_TOKENS,
    TARGET_SENSOR_ORDER,
    OPENPI_ATTENTION_MASK_VALUE,
    DEFAULT_PERM_INDICES,
)
from .utils import (
    make_att_2d_masks,
    sample_beta,
    create_sinusoidal_pos_embedding,
)
from .modules import DenseDepthNet
from projects.mmdet3d_plugin.losses.collision_loss import CollisionLoss
from projects.mmdet3d_plugin.losses.plan_map_loss import GTMapBoundLoss, GTMapDirectionLoss

from projects.mmdet3d_plugin.models.detection3d.target import SparseBox3DTarget
from projects.mmdet3d_plugin.models.detection3d.losses import SparseBox3DLoss
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import SparseBox3DEncoder
from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget
from projects.mmdet3d_plugin.models.map.loss import SparseLineLoss
from projects.mmdet3d_plugin.models.map.map_blocks import SparsePoint3DEncoder
from projects.mmdet3d_plugin.models.map.decoder import SparsePoint3DDecoder
from projects.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *


@dataclass
class DrivingBatch:
    images: torch.Tensor
    image_masks: Dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    command: Optional[torch.Tensor]
    ego_status: Optional[torch.Tensor]
    target_point: Optional[torch.Tensor] = None
    view_token_ids: Optional[torch.Tensor] = None


class QwenConfig:
    def __init__(self, head_dim, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:
    num_hidden_layers = int(variant.split('_')[-1][:-1])
    if variant.startswith("qwen3_vl_8b"):
        return QwenConfig(
            head_dim=128,
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3_vl"):
        return QwenConfig(
            head_dim=128,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3_8b_expert"):
        return QwenConfig(
            head_dim=128,
            hidden_size=4096,
            intermediate_size=2048,
            num_attention_heads=32,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3"):
        return QwenConfig(
            head_dim=128,
            hidden_size=1024,
            intermediate_size=2048,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


@HEADS.register_module()
class QwenVL3APlanningHead(nn.Module):
    def __init__(
        self,
        pretrained_path,
        action_dim: int = 2,
        action_horizon: int = 6,
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        num_sample_steps: int = 10,
        enable_knowledge_insulation: bool = False,
        ar_loss_weight: float = 1.0,
        train_vlm: bool = False,
        loss_planning: Optional[dict] = None,
        collision_loss_weight: float = 0.0,
        map_bound_loss_weight: float = 0.0,
        map_dir_loss_weight: float = 0.0,
        map_bound_dis_thresh: float = 1.0,
        map_dir_dis_thresh: float = 2.0,
        x_min: float = -13.97,
        x_max: float = 11.77,
        y_min: float = -2.02,
        y_max: float = 55.79,
        attn_implementation: Literal["eager", "sdpa", "flex"] = "flex",
        unified_decoder_cfg: dict = None,
        with_depth_supervision: bool = False,
        depth_loss_weight: float = 0.2,
        num_depth_bins: int = 80,
        depth_range: tuple = (1.0, 60.0),
        depth_supervision_source: Literal["input", "output"] = "input",
        feature_source: Literal["raw", "deepstack"] = "deepstack",
        feat_grad: Optional[bool] = None,
        vlm_variant: Literal["2b", "8b"] = "2b",
        lora_cfg: Optional[dict] = None,
        inference_seed: Optional[int] = None,
        driving_deepstack: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.min_period = min_period
        self.max_period = max_period
        self.num_sample_steps = num_sample_steps

        self.enable_knowledge_insulation = enable_knowledge_insulation
        self.ar_loss_weight = ar_loss_weight
        self.train_vlm = train_vlm
        self.collision_loss_weight = collision_loss_weight
        self.collision_loss_fn = CollisionLoss(weight=1.0) if collision_loss_weight > 0 else None

        self.map_bound_loss_weight = map_bound_loss_weight
        self.map_bound_loss_fn = (
            GTMapBoundLoss(dis_thresh=map_bound_dis_thresh, weight=1.0)
            if map_bound_loss_weight > 0 else None
        )
        self.map_dir_loss_weight = map_dir_loss_weight
        self.map_dir_loss_fn = (
            GTMapDirectionLoss(dis_thresh=map_dir_dis_thresh, weight=1.0)
            if map_dir_loss_weight > 0 else None
        )

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        self.loss_planning = build_loss(loss_planning) if loss_planning is not None else None
        self.inference_seed = inference_seed

        if vlm_variant == "8b":
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_8b_36l')
            perception_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
            action_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
        else:
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_28l')
            perception_expert_cfg = get_qwen_config('qwen3_28l')
            action_expert_cfg = get_qwen_config('qwen3_28l')

        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
            qwen3_vl_cfg,
            perception_expert_cfg,
            action_expert_cfg,
            pretrained_path,
            precision=dtype,
            train_vlm=train_vlm,
            lora_cfg=lora_cfg,
        )

        self.attn_implementation = attn_implementation
        self.qwen3_vl_with_expert._vla_attn_impl = self.attn_implementation

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg must be provided via config.")

        self.embed_dims = unified_decoder_cfg.get("embed_dims", 256)
        self.vlm_hidden_size = perception_expert_cfg.hidden_size

        self.num_det_queries = unified_decoder_cfg.get("det_instance_bank", {}).get("num_anchor", 900)
        self.num_map_queries = unified_decoder_cfg.get("map_instance_bank", {}).get("num_anchor", 100)

        self.with_motion = "motion" in unified_decoder_cfg.get("task_select", [])

        if self.with_motion:
            self.num_motion_queries = unified_decoder_cfg.get("motion_instance_bank", {}).get("num_anchor", 900)
            self.motion_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
            self.motion_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        else:
            self.num_motion_queries = 0
            self.motion_proj_up = None
            self.motion_proj_down = None

        self.ego_status_dim = unified_decoder_cfg.get("ego_refine_layer", {}).get("status_dims", 10)

        self.det_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.map_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.det_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        self.map_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        self.ego_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.ego_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)

        action_hidden = action_expert_cfg.hidden_size
        vision_hidden_size = self.qwen3_vl_with_expert.qwen3_vl.config.vision_config.hidden_size
        self.feature_source = feature_source

        if feat_grad is None:
            raise ValueError("feat_grad must be provided via config.")
        self.feat_grad = bool(feat_grad)

        if self.feature_source == "raw":
            proj_input_dim = vision_hidden_size
            num_proj_layers = 4
        elif self.feature_source == "deepstack":
            proj_input_dim = vision_hidden_size * 2
            num_proj_layers = 3
        else:
            raise ValueError(f"Unknown feature_source: {feature_source}")

        self.feature_map_proj = nn.ModuleList([
            nn.Linear(proj_input_dim, self.embed_dims)
            for _ in range(num_proj_layers)
        ])

        self.fusion_weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.embed_dims, num_proj_layers),
                nn.Softmax(dim=-1)
            ) for _ in range(num_proj_layers)
        ])

        self.with_depth_supervision = with_depth_supervision
        self.depth_loss_weight = depth_loss_weight
        self.num_depth_bins = num_depth_bins
        self.depth_range = depth_range
        self.depth_supervision_source = depth_supervision_source

        if self.with_depth_supervision:
            self.depth_net = DenseDepthNet(
                embed_dims=self.embed_dims,
                in_channels=qwen3_vl_cfg.hidden_size,
                num_depth_layers=1,
                equal_focal=100,
                max_depth=60,
                loss_weight=depth_loss_weight,
            )

        self.unified_decoder = build_head(unified_decoder_cfg)

        self.action_in_proj = nn.Linear(action_dim, action_expert_cfg.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_cfg.hidden_size, action_dim)

        status_in_features = 6 + self.ego_status_dim
        self.status_mlp = Mlp(
            in_features=status_in_features,
            hidden_features=action_expert_cfg.hidden_size,
            out_features=action_expert_cfg.hidden_size,
            norm_layer=nn.LayerNorm,
        )

        self.action_time_mlp_in = nn.Linear(2 * action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)

        if dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif dtype == "float32":
            target_dtype = torch.float32
        else:
            target_dtype = torch.float16

        self.action_in_proj.to(target_dtype)
        self.status_mlp.to(target_dtype)
        self.action_time_mlp_in.to(target_dtype)
        self.action_time_mlp_out.to(target_dtype)
        self.det_proj_up.to(target_dtype)
        self.det_proj.to(target_dtype)
        self.map_proj_up.to(target_dtype)
        self.map_proj.to(target_dtype)
        self.ego_proj_up.to(target_dtype)
        self.ego_proj_down.to(target_dtype)
        if self.motion_proj_up is not None:
            self.motion_proj_up.to(target_dtype)
        if self.motion_proj_down is not None:
            self.motion_proj_down.to(target_dtype)
        self.unified_decoder.to(target_dtype)
        self.fusion_weight_generators.to(target_dtype)
        self.feature_map_proj.to(target_dtype)
        if hasattr(self.qwen3_vl_with_expert.qwen3_vl, 'lm_head'):
            self.qwen3_vl_with_expert.qwen3_vl.lm_head.requires_grad_(False)

        self.gradient_checkpointing_enable()
        self.gradient_checkpointing_enabled = True

        self.driving_deepstack = driving_deepstack

        self.view_token_str_list = B2D_VIEW_TOKENS
        self.view_token_ids = None

    def _get_view_token_ids(self, device):
        if self.view_token_ids is None:
            tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
            ids = [tokenizer.convert_tokens_to_ids(t) for t in self.view_token_str_list]
            self.view_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        return self.view_token_ids.to(device)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing = True
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing = True
        self.qwen3_vl_with_expert.qwen3_action_expert.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing = False
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing = False
        self.qwen3_vl_with_expert.qwen3_action_expert.gradient_checkpointing = False

    def merge_and_save_lora(self, save_dir=None):
        self.qwen3_vl_with_expert.merge_lora()

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device, generator=None):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device, generator=generator)

    def sample_time(self, bsize, device):
        t = sample_beta(self.time_beta_alpha, self.time_beta_beta, bsize, device)
        t = t * 0.999 + 0.001
        return t.to(dtype=torch.float32, device=device)

    def norm_delta(self, delta_meter: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor([-0.0222, 2.0249], device=delta_meter.device, dtype=delta_meter.dtype)
        std = torch.tensor([0.6720, 1.8586], device=delta_meter.device, dtype=delta_meter.dtype)
        return (delta_meter - mu) / (std + 1e-6)

    def denorm_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor([-0.0222, 2.0249], device=delta_norm.device, dtype=delta_norm.dtype)
        std = torch.tensor([0.6720, 1.8586], device=delta_norm.device, dtype=delta_norm.dtype)
        return delta_norm * (std + 1e-6) + mu

    def embed_perception(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stage1_outs: dict,
    ):
        query_select = self.unified_decoder.query_select

        motion_token_256 = stage1_outs.get('motion_token', None)

        proj_dtype = self.det_proj_up.weight.dtype

        perception_parts_embs = []
        perception_parts_pad = []
        perception_parts_att = []
        perception_lengths = {'det': 0, 'map': 0, 'ego': 0, 'motion': 0}

        if 'det' in query_select and len(stage1_outs.get('det_predictions', [])) > 0:
            det_feat = stage1_outs['det_instance_feature']
            det_anchor = stage1_outs['det_predictions'][-1]
            det_embs = self.det_proj_up(det_feat.to(dtype=proj_dtype))
            anchor_embed_256 = self.unified_decoder.det_anchor_encoder(det_anchor)
            anchor_embed_vlm = self.det_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            det_embs = (det_embs + anchor_embed_vlm).to(dtype)
            det_pad_masks = torch.ones((batch_size, self.num_det_queries), dtype=torch.bool, device=device)
            det_att_masks = torch.zeros((batch_size, self.num_det_queries), dtype=torch.bool, device=device)
            perception_parts_embs.append(det_embs)
            perception_parts_pad.append(det_pad_masks)
            perception_parts_att.append(det_att_masks)
            perception_lengths['det'] = det_embs.shape[1]

        if 'map' in query_select and len(stage1_outs.get('map_predictions', [])) > 0:
            map_feat = stage1_outs['map_instance_feature']
            map_anchor = stage1_outs['map_predictions'][-1]
            map_embs = self.map_proj_up(map_feat.to(dtype=proj_dtype))
            anchor_embed_out = self.unified_decoder.map_anchor_encoder(map_anchor)
            anchor_embed_256 = anchor_embed_out[0] if isinstance(anchor_embed_out, (tuple, list)) else anchor_embed_out
            anchor_embed_vlm = self.map_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            map_embs = (map_embs + anchor_embed_vlm).to(dtype)
            map_pad_masks = torch.ones((batch_size, self.num_map_queries), dtype=torch.bool, device=device)
            map_att_masks = torch.zeros((batch_size, self.num_map_queries), dtype=torch.bool, device=device)
            perception_parts_embs.append(map_embs)
            perception_parts_pad.append(map_pad_masks)
            perception_parts_att.append(map_att_masks)
            perception_lengths['map'] = map_embs.shape[1]

        if 'ego' in query_select:
            ego_feat = stage1_outs['ego_instance_feature']
            ego_anchor = stage1_outs['ego_anchor']
            ego_embs = self.ego_proj_up(ego_feat.to(dtype=proj_dtype))
            if hasattr(self.unified_decoder, 'ego_anchor_encoder') and ego_anchor is not None:
                ego_anchor_embed_256 = self.unified_decoder.ego_anchor_encoder(ego_anchor)
                ego_anchor_embed_vlm = self.ego_proj_up(ego_anchor_embed_256.to(dtype=proj_dtype))
                ego_embs = (ego_embs + ego_anchor_embed_vlm).to(dtype)
            else:
                ego_embs = ego_embs.to(dtype)
            ego_pad_masks = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            ego_att_masks = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            perception_parts_embs.append(ego_embs)
            perception_parts_pad.append(ego_pad_masks)
            perception_parts_att.append(ego_att_masks)
            perception_lengths['ego'] = ego_embs.shape[1]

        if motion_token_256 is not None:
            motion_token_vlm = self.motion_proj_up(motion_token_256.to(dtype=proj_dtype)).to(dtype)
            motion_pad_masks = torch.ones((batch_size, motion_token_vlm.shape[1]), dtype=torch.bool, device=device)
            motion_att_masks = torch.zeros((batch_size, motion_token_vlm.shape[1]), dtype=torch.bool, device=device)
            perception_parts_embs.append(motion_token_vlm)
            perception_parts_pad.append(motion_pad_masks)
            perception_parts_att.append(motion_att_masks)
            perception_lengths['motion'] = motion_token_vlm.shape[1]

        if perception_parts_embs:
            perception_embs = torch.cat(perception_parts_embs, dim=1)
            perception_pad_masks = torch.cat(perception_parts_pad, dim=1)
            perception_att_masks = torch.cat(perception_parts_att, dim=1)
        else:
            perception_embs = torch.empty((batch_size, 0, self.vlm_hidden_size), dtype=dtype, device=device)
            perception_pad_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
            perception_att_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)

        return perception_embs, perception_pad_masks, perception_att_masks, perception_lengths

    def project_and_reshape_features(
        self,
        source_features,
        bsz: int,
        all_image_grids,
        feature_source: str,
    ):
        feature_maps = []

        if source_features is None:
            return feature_maps

        if not isinstance(source_features, list):
            source_features = [source_features]

        projected_features = []
        for i, feat in enumerate(source_features):
            if i < len(self.feature_map_proj):
                feat = feat.to(self.feature_map_proj[i].weight.dtype)
                feat_proj = self.feature_map_proj[i](feat)
                projected_features.append(feat_proj)
            else:
                projected_features.append(feat)

        if all_image_grids is not None and len(all_image_grids) > 0:
            h_grid = int(all_image_grids[0, 1].item())
            w_grid = int(all_image_grids[0, 2].item())
            num_views = 6

            for ds_feat in projected_features:
                if ds_feat.dim() != 2:
                    continue

                feat_reshaped = None

                if feature_source == "raw":
                    merge_size = 2
                    h_block = h_grid // merge_size
                    w_block = w_grid // merge_size
                    expected_tokens = bsz * num_views * h_grid * w_grid

                    if ds_feat.shape[0] == expected_tokens:
                        try:
                            feat_vis = ds_feat.view(bsz, num_views, h_block, w_block, merge_size, merge_size, -1)
                            feat_vis = feat_vis.permute(0, 1, 2, 4, 3, 5, 6)
                            feat_reshaped = feat_vis.reshape(bsz, num_views, h_grid, w_grid, -1).permute(0, 1, 4, 2, 3).contiguous()
                        except Exception:
                            feat_reshaped = None

                elif feature_source == "deepstack":
                    merge_size = 2
                    h_ds = h_grid // merge_size
                    w_ds = w_grid // merge_size
                    expected_tokens = bsz * num_views * h_ds * w_ds

                    if ds_feat.shape[0] == expected_tokens:
                        try:
                            feat_reshaped = ds_feat.view(bsz, num_views, h_ds, w_ds, -1).permute(0, 1, 4, 2, 3).contiguous()
                        except Exception:
                            feat_reshaped = None

                if feat_reshaped is not None:
                    feature_maps.append(feat_reshaped)

        if len(feature_maps) > 0:
            feature_maps = self.adaptive_feature_fusion(feature_maps)

        return feature_maps

    def adaptive_feature_fusion(self, feature_maps):
        if len(feature_maps) <= 1:
            return feature_maps

        B, N, C, H, W = feature_maps[0].shape
        fused_maps = []

        for i, feat in enumerate(feature_maps):
            feat_flat = feat.view(B * N, C, H, W)
            fusion_weights = self.fusion_weight_generators[i](feat_flat)
            weights = fusion_weights.view(B, N, len(feature_maps), 1, 1, 1)

            current_fused = 0
            for j in range(len(feature_maps)):
                other_feat = feature_maps[j]
                w = weights[:, :, j]
                if other_feat.shape[-2:] != (H, W):
                    ref = F.interpolate(
                        other_feat.flatten(0, 1), size=(H, W), mode='bilinear'
                    ).view(B, N, C, H, W)
                else:
                    ref = other_feat
                current_fused = current_fused + ref * w

            fused_maps.append(current_fused)

        return fused_maps

    def compute_depth_loss(
        self,
        prefix_embs: torch.Tensor,
        prefix_out: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        all_image_grids: torch.Tensor,
        gt_depth: torch.Tensor,
        focal: Optional[torch.Tensor],
        bsz: int,
    ) -> torch.Tensor:
        if not self.with_depth_supervision or gt_depth is None:
            return torch.tensor(0.0, device=prefix_embs.device)

        feat_for_depth = None
        depth_spatial_shape = None

        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        image_mask = (prefix_input_ids == image_token_id)

        if image_mask.any():
            if self.depth_supervision_source == "input":
                feat_for_depth = prefix_embs[image_mask]
            elif self.depth_supervision_source == "output":
                feat_for_depth = prefix_out[image_mask]

            if feat_for_depth is not None and all_image_grids is not None and len(all_image_grids) > 0:
                h_grid = int(all_image_grids[0, 1].item())
                w_grid = int(all_image_grids[0, 2].item())
                merge_size = 2
                h_ds, w_ds = h_grid // merge_size, w_grid // merge_size
                expected_tokens = bsz * 6 * h_ds * w_ds

                if feat_for_depth.shape[0] == expected_tokens:
                    feat_for_depth = feat_for_depth.view(bsz * 6, h_ds, w_ds, -1).permute(0, 3, 1, 2)
                    depth_spatial_shape = (h_ds, w_ds)
                else:
                    feat_for_depth = None

        if feat_for_depth is not None and depth_spatial_shape is not None:
            gt_depth = gt_depth.to(feat_for_depth.device)
            num_feat_images = feat_for_depth.shape[0]
            gt_depth_reshaped = rearrange(gt_depth, 'b n h w -> (b n) 1 h w')
            if num_feat_images < gt_depth_reshaped.shape[0]:
                gt_depth_reshaped = gt_depth_reshaped[:num_feat_images]

            H_feat, W_feat = feat_for_depth.shape[-2:]
            gt_depth_resized = F.interpolate(
                gt_depth_reshaped, size=(H_feat, W_feat), mode='nearest'
            ).squeeze(1)

            focal_flat = focal.reshape(-1) if focal is not None else None
            if focal_flat is not None and num_feat_images < focal_flat.shape[0]:
                focal_flat = focal_flat[:num_feat_images]

            return self.depth_net(feat_for_depth, focal=focal_flat, gt_depths=gt_depth_resized)

        return torch.tensor(0.0, device=prefix_embs.device)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )
        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def _build_driving_batch(
        self,
        img: torch.Tensor,
        command=None,
        ego_status=None,
        target_point=None,
    ) -> DrivingBatch:
        device = img.device if img is not None else torch.device("cuda")
        b = int(img.shape[0]) if torch.is_tensor(img) else 1

        permute_indices = [0, 1, 2, 4, 5, 3]
        images = img[:, permute_indices]

        image_masks = {f"cam{i}": torch.ones((b,), device=device, dtype=torch.bool) for i in range(6)}
        view_token_ids = self._get_view_token_ids(device)

        if command is not None:
            if not torch.is_tensor(command):
                try:
                    command = torch.stack(command)
                except:
                    command = torch.tensor(command)
            command = command.to(device)
            cmd_idx = command.view(-1).long()
            idx_list = cmd_idx.tolist()
        else:
            idx_list = [2] * b

        idx2cmd = {
            0: "TURN LEFT",
            1: "TURN RIGHT",
            2: "GO STRAIGHT",
            3: "LANE FOLLOW",
            4: "CHANGE LANE LEFT",
            5: "CHANGE LANE RIGHT",
        }
        nav_cmd_texts = [idx2cmd.get(i, "GO STRAIGHT") for i in idx_list]

        ego_status_np = ego_status.detach().cpu().numpy() if ego_status is not None else np.zeros((b, 6))

        if not hasattr(self.qwen3_vl_with_expert, "processor") or self.qwen3_vl_with_expert.processor is None:
            raise RuntimeError("QwenVLAPlanningHead expects `self.qwen3_vl_with_expert.processor`")

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        system_ids = tokenizer.encode("system", add_special_tokens=False)
        user_ids = tokenizer.encode("user", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        sys_content_ids = tokenizer.encode(B2D_SYSTEM_PROMPT, add_special_tokens=False)
        sys_part = [im_start_id] + system_ids + [nl_id] + sys_content_ids + [im_end_id, nl_id]

        user_start_part = [im_start_id] + user_ids + [nl_id]
        user_end_assistant_start_part = [im_end_id, nl_id, im_start_id] + assistant_ids + [nl_id]

        input_ids_list = []
        attention_mask_list = []

        for i in range(b):
            speed = ego_status_np[i, 0]
            acc_x = ego_status_np[i, 1]
            acc_y = ego_status_np[i, 2]

            user_prompt_text = B2D_USER_PROMPT_TEMPLATE.format(
                nav_cmd=nav_cmd_texts[i],
                speed=speed,
                acc_x=acc_x,
                acc_y=acc_y,
            )
            user_content_ids = tokenizer.encode(user_prompt_text, add_special_tokens=False)
            full_ids = sys_part + user_start_part + user_content_ids + user_end_assistant_start_part

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        tokenized_prompt = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        tokenized_prompt_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        return DrivingBatch(
            images=images,
            image_masks=image_masks,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            command=command,
            ego_status=ego_status,
            target_point=target_point,
            view_token_ids=view_token_ids,
        )

    def embed_prefix(self, batch: DrivingBatch):
        images_tensor = batch.images
        device = self.action_in_proj.weight.device

        image_features, feature_lens, all_image_grids, deepstack_features, raw_features = self.qwen3_vl_with_expert.embed_image_tensor(images_tensor)

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        vision_start_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_start_token_id
        vision_end_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_end_token_id
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        nl_id = self.qwen3_vl_with_expert.processor.tokenizer.encode("\n", add_special_tokens=False)[0]

        bs = images_tensor.shape[0]
        num_views_per_sample = 6
        view_token_ids = self._get_view_token_ids(device)

        prefix_input_ids_list = []
        FIXED_PREFIX_MAX_LEN = 3670

        for b_idx in range(bs):
            sample_input_ids = []
            for v_idx in range(num_views_per_sample):
                img_len = feature_lens[b_idx * num_views_per_sample + v_idx]
                ids = [view_token_ids[v_idx].item(), nl_id, vision_start_id] + \
                    [image_token_id] * img_len + \
                    [vision_end_id, nl_id]
                sample_input_ids.extend(ids)

            prompt_mask = batch.tokenized_prompt_mask[b_idx].bool()
            prompt_ids = batch.tokenized_prompt[b_idx][prompt_mask].tolist()
            sample_input_ids.extend(prompt_ids)
            prefix_input_ids_list.append(torch.tensor(sample_input_ids, dtype=torch.long, device=device))

        prefix_input_ids = pad_sequence(prefix_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)

        curr_len = prefix_input_ids.shape[1]
        if curr_len < FIXED_PREFIX_MAX_LEN:
            pad_len = FIXED_PREFIX_MAX_LEN - curr_len
            prefix_input_ids = F.pad(prefix_input_ids, (0, pad_len), value=tokenizer.pad_token_id)
        elif curr_len > FIXED_PREFIX_MAX_LEN:
            print(f"[WARNING] Prefix length {curr_len} > {FIXED_PREFIX_MAX_LEN}. Truncating.")
            prefix_input_ids = prefix_input_ids[:, :FIXED_PREFIX_MAX_LEN]
            truncated_mask = (prefix_input_ids == image_token_id)
            valid_img_tokens = truncated_mask.sum().item()
            if valid_img_tokens < image_features.shape[0]:
                image_features = image_features[:valid_img_tokens]

        prefix_pad_masks = (prefix_input_ids != tokenizer.pad_token_id)
        input_embeds = self.qwen3_vl_with_expert.qwen3_vl.get_input_embeddings()(prefix_input_ids)
        image_mask = (prefix_input_ids == image_token_id)

        if image_mask.sum() != image_features.shape[0]:
            target_count = image_mask.sum()
            current_count = image_features.shape[0]
            if current_count > target_count:
                image_features = image_features[:target_count]
            else:
                raise ValueError(f"Visual features mismatch! Feat: {current_count}, Tokens: {target_count}")

        input_embeds = input_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features.to(input_embeds.dtype))
        prefix_att_marks = torch.ones_like(prefix_pad_masks, dtype=torch.long)

        return input_embeds, prefix_pad_masks, prefix_att_marks, all_image_grids, prefix_input_ids, deepstack_features, raw_features

    def _maybe_get_status_features(
        self,
        batch: DrivingBatch,
        ego_status_pred: Optional[torch.Tensor],
        *,
        use_gt: bool,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = batch.command.device
        cmd_onehot = batch.command.to(device=device, dtype=dtype)
        ego = batch.ego_status.to(device=device, dtype=dtype)
        return torch.cat([cmd_onehot, ego], dim=-1)

    def embed_suffix(
        self,
        batch: DrivingBatch,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        ego_status_pred: Optional[torch.Tensor] = None,
        use_gt_status: bool = False,
    ):
        device = actions.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        B = actions.shape[0]

        status_input = self._maybe_get_status_features(batch, ego_status_pred, use_gt=use_gt_status, dtype=dtype)
        status_emb = self._apply_checkpoint(self.status_mlp, status_input)
        status_emb = status_emb.unsqueeze(1)

        num_status_tokens = 1

        if actions.dtype != dtype:
            actions = actions.to(dtype=dtype)

        action_emb = self._apply_checkpoint(self.action_in_proj, actions)

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, self.min_period, self.max_period, device=device
        )
        time_emb_expanded = time_emb.unsqueeze(1).expand_as(action_emb)
        fused_input = torch.cat([action_emb, time_emb_expanded], dim=-1).to(dtype)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, fused_input)
        suffix_emb = torch.cat([status_emb, action_time_emb], dim=1)

        suffix_len = suffix_emb.shape[1]
        pad_masks = torch.ones((B, suffix_len), dtype=torch.bool, device=device)

        att_marks = [1] * num_status_tokens + [0] * (suffix_len - num_status_tokens)
        att_marks = torch.tensor(att_marks, dtype=torch.bool, device=device)
        att_marks = att_marks[None, :].expand(B, att_marks.numel())

        return suffix_emb, pad_masks, att_marks

    def get_position_ids(self, input_ids, image_grid_thw, pad_masks):
        attention_mask = pad_masks.long()
        position_ids, rope_deltas = self.qwen3_vl_with_expert.qwen3_vl.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def prepare_for_deformable_aggregation(self, feature_maps):
        if not feature_maps:
            return []
        return feature_maps_format(feature_maps)

    def forward_train(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        ar_batch=None,
        **kwargs,
    ):
        if gt_ego_fut_trajs is None:
            raise ValueError("gt_ego_fut_trajs is required")

        if gt_ego_fut_trajs.dtype == torch.float64:
            gt_ego_fut_trajs = gt_ego_fut_trajs.float()

        permute_indices = [0, 1, 2, 4, 5, 3]
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]

        def _permute_metas_per_camera_fields(img_metas):
            if not isinstance(img_metas, list):
                return img_metas
            out = []
            for m in img_metas:
                if not isinstance(m, dict):
                    out.append(m)
                    continue
                m2 = dict(m)
                for k, v in list(m2.items()):
                    if isinstance(v, list) and len(v) == 6:
                        m2[k] = [v[i] for i in permute_indices]
                cams = m2.get("cams", None)
                if isinstance(cams, dict) and len(cams) == 6:
                    ordered_keys = list(cams.keys())
                    if all(k in cams for k in TARGET_SENSOR_ORDER):
                        ordered_keys = TARGET_SENSOR_ORDER
                    m2["cams"] = {k: cams[k] for k in [ordered_keys[i] for i in permute_indices]}
                out.append(m2)
            return out

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = _permute_metas_per_camera_fields(kwargs.get("img_metas"))

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            target_point=kwargs.get('target_point'),
        )

        if isinstance(gt_agent_fut_trajs, list):
            gt_agent_fut_trajs = [t.to(device=img.device) if torch.is_tensor(t) else torch.tensor(t, device=img.device) for t in gt_agent_fut_trajs]
        if isinstance(gt_agent_fut_masks, list):
            gt_agent_fut_masks = [t.to(device=img.device) if torch.is_tensor(t) else torch.tensor(t, device=img.device) for t in gt_agent_fut_masks]

        actions = gt_ego_fut_trajs
        if torch.is_tensor(actions) and actions.dim() == 4 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
        actions = self.norm_delta(actions)

        bsz = actions.shape[0]

        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features = self.embed_prefix(batch)

        if self.driving_deepstack and deepstack_features is not None:
            image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
            _visual_pos_masks = (prefix_input_ids == image_token_id)
            _ds_embeds = deepstack_features
        else:
            _visual_pos_masks = None
            _ds_embeds = None

        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(source_features, bsz, all_image_grids, self.feature_source)
        feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
        if not self.feat_grad:
            feature_maps_daf = [x.detach() for x in feature_maps_daf]

        head_device = actions.device
        head_param_dtype = next(self.unified_decoder.parameters()).dtype
        perception_metas = {
            'img_metas': kwargs.get('img_metas'),
            'timestamp': timestamp,
            'projection_mat': projection_mat.to(device=head_device, dtype=head_param_dtype),
            'image_wh': image_wh.to(device=head_device, dtype=head_param_dtype),
        }

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)
        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz, actions.device, prefix_embs.dtype, stage1_outs
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch, x_t, time, ego_status_pred=None, use_gt_status=True,
        )

        if self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            perception_embs = perception_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        prefix_pos_ids, rope_deltas = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        perception_len = perception_embs.shape[1]
        perception_range = torch.arange(1, perception_len + 1, device=actions.device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        if perception_len > 0:
            max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values
        else:
            max_perception_pos = max_prefix_pos
        perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

        suffix_len = suffix_embs.shape[1]
        suffix_range = torch.arange(1, suffix_len + 1, device=actions.device).view(1, -1).expand(bsz, -1)
        suffix_pos_ids_1d = max_perception_pos + suffix_range
        suffix_pos_ids_3d = torch.stack([suffix_pos_ids_1d] * 3, dim=0)

        position_ids = torch.cat([prefix_pos_ids, perception_pos_ids_3d, suffix_pos_ids_3d], dim=2)

        prefix_len = prefix_embs.shape[1]
        det_len = perception_lengths['det']
        map_len = perception_lengths['map']
        ego_len = perception_lengths['ego']
        motion_len = perception_lengths['motion']

        att_mask_input = None
        q_len_rounded = None

        if self.attn_implementation == "flex" and not self.enable_knowledge_insulation:
            block_mask, q_len_rounded = build_blockmask_unidrive(
                bsz=bsz,
                hq=self.qwen3_vl_with_expert.qwen3_vl.config.text_config.num_attention_heads,
                prefix_len=prefix_len,
                perception_len=perception_len,
                suffix_len=suffix_len,
                device=actions.device,
                compile_blockmask=True,
            )
            att_mask_input = block_mask
        else:
            pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, perception_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            att_mask_input = self._prepare_attention_masks_4d(att_2d_masks)

        total_loss = 0.0
        stats = {}
        ar_loss = 0.0
        perception_out = None

        if self.enable_knowledge_insulation:
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            prefix_pos_ids_sliced = position_ids[..., :prefix_len]

            (prefix_out, _), past_key_values, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=prefix_att_4d,
                position_ids=prefix_pos_ids_sliced,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            lm_logits = self.qwen3_vl_with_expert.qwen3_vl.lm_head(prefix_out)
            text_len = batch.tokenized_prompt.shape[1]
            text_logits = lm_logits[:, -text_len:, :]
            text_labels = batch.tokenized_prompt.clone()
            text_labels.masked_fill_(~batch.tokenized_prompt_mask.bool(), -100)
            shift_logits = text_logits[..., :-1, :].contiguous()
            shift_labels = text_labels[..., 1:].contiguous()

            ar_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            stats["loss_ar"] = ar_loss

            detached_past_kv = [(layer_kv[0].detach(), layer_kv[1].detach()) for layer_kv in past_key_values]

            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_mixed = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            suffix_att_4d = self._prepare_attention_masks_4d(full_att_2d_mixed)
            suffix_pos_ids_sliced = position_ids[..., prefix_len:]

            (_, suffix_out), _, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=suffix_att_4d,
                position_ids=suffix_pos_ids_sliced,
                past_key_values=detached_past_kv,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
            )
        else:
            def forward_func(prefix_embs, perception_embs, suffix_embs, att_mask, position_ids, _unused, q_len_rnd, ds_embeds, vis_pos_masks):
                outputs, _, middle_layer_outputs = self.qwen3_vl_with_expert.forward(
                    attention_mask=att_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, perception_embs, suffix_embs],
                    use_cache=False,
                    return_middle_layers=None,
                    q_len_rounded=q_len_rnd,
                    deepstack_visual_embeds=ds_embeds,
                    visual_pos_masks=vis_pos_masks,
                )
                return outputs, middle_layer_outputs

            (outputs_embeds, _) = self._apply_checkpoint(
                forward_func,
                prefix_embs,
                perception_embs,
                suffix_embs,
                att_mask_input,
                position_ids,
                None,
                q_len_rounded,
                _ds_embeds,
                _visual_pos_masks,
            )
            prefix_out, perception_out, suffix_out = outputs_embeds

        loss_depth_val = self.compute_depth_loss(
            prefix_embs, prefix_out, prefix_input_ids, all_image_grids, gt_depth, focal, bsz
        )

        d0 = 0
        d1 = d0 + det_len
        m0 = d1
        m1 = m0 + map_len
        e0 = m1
        e1 = e0 + ego_len
        t0 = e1
        t1 = t0 + motion_len

        if perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]

            proj_dtype = self.det_proj.weight.dtype
            target_dtype = self.det_proj.weight.dtype
            motion_token_256 = stage1_outs.get('motion_token', None)
            ego_feat_stage1 = stage1_outs['ego_instance_feature']

            det_feat_fused = self.det_proj(det_out_vlm.to(proj_dtype)).to(torch.float32)
            map_feat_fused = self.map_proj(map_out_vlm.to(proj_dtype)).to(torch.float32)
            ego_feat_fused = (
                self.ego_proj_down(ego_out.to(proj_dtype)).to(torch.float32)
                if ego_out is not None
                else ego_feat_stage1.to(torch.float32)
            )
            motion_feat_fused = (
                self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                if motion_token_256 is not None
                else None
            )

            vlm_enhanced = {
                'det_feat': det_feat_fused.to(target_dtype),
                'map_feat': map_feat_fused.to(target_dtype),
                'ego_feat': ego_feat_fused.to(target_dtype),
            }
            if motion_feat_fused is not None:
                vlm_enhanced['motion_feat'] = motion_feat_fused.to(target_dtype)

            stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)

            gt_map_labels_val = gt_map_labels.data if hasattr(gt_map_labels, 'data') else gt_map_labels
            gt_map_pts_val = gt_map_pts.data if hasattr(gt_map_pts, 'data') else gt_map_pts
            gt_boxes_list = None
            gt_labels_list = None
            if gt_bboxes_3d is not None:
                gt_boxes_list = gt_bboxes_3d.data if hasattr(gt_bboxes_3d, 'data') else gt_bboxes_3d
                gt_boxes_list = [x.tensor.to(actions.device) if hasattr(x, 'tensor') else x.to(actions.device) for x in gt_boxes_list]
            if gt_labels_3d is not None:
                gt_labels_list = gt_labels_3d.data if hasattr(gt_labels_3d, 'data') else gt_labels_3d
                gt_labels_list = [x.to(actions.device) for x in gt_labels_list]

            perception_data = {
                'gt_bboxes_3d': gt_boxes_list,
                'gt_labels_3d': gt_labels_list,
                'gt_map_labels': gt_map_labels_val,
                'gt_map_pts': gt_map_pts_val,
                'ego_status': ego_status,
                'ego_status_mask': kwargs.get('ego_status_mask', None),
                'gt_agent_fut_trajs': gt_agent_fut_trajs,
                'gt_agent_fut_masks': gt_agent_fut_masks,
            }
            perception_losses = self.unified_decoder.loss(stage1_outs, stage2_outs, perception_data)
            stats.update(perception_losses)

        suffix_out_float = suffix_out[:, -self.action_horizon:].to(dtype=torch.float32)
        model_out = self._apply_checkpoint(self.action_out_proj, suffix_out_float)

        t_exp = time[:, None, None].to(dtype=torch.float32)
        pred_x_gt = x_t.to(dtype=torch.float32) - t_exp * model_out
        gt_x = actions

        if self.loss_planning is not None:
            planning_loss = self.loss_planning(
                u_t, model_out, time,
                gt_ego_fut_masks=gt_ego_fut_masks,
                pred_x0=pred_x_gt,
                gt_x0=gt_x,
            )
        else:
            planning_loss = torch.tensor(0.0, device=model_out.device)

        stats["loss_planning"] = planning_loss

        _pred_abs = None
        if pred_x_gt is not None and (
            self.collision_loss_fn is not None or
            self.map_bound_loss_fn is not None or
            self.map_dir_loss_fn is not None
        ):
            _pred_deltas = self.denorm_delta(pred_x_gt[..., :2].detach().float())
            _pred_abs = torch.cumsum(_pred_deltas, dim=1)

        if self.collision_loss_fn is not None and gt_bboxes_3d is not None and _pred_abs is not None:
            loss_collision = self.collision_loss_fn(_pred_abs, gt_bboxes_3d, gt_ego_fut_masks)
            stats["loss_collision"] = self.collision_loss_weight * loss_collision

        if self.map_bound_loss_fn is not None and gt_map_pts is not None and _pred_abs is not None:
            _gt_map_pts = gt_map_pts.data if hasattr(gt_map_pts, 'data') else gt_map_pts
            _gt_map_labels = gt_map_labels.data if hasattr(gt_map_labels, 'data') else gt_map_labels
            loss_map_bound = self.map_bound_loss_fn(_pred_abs, _gt_map_pts, _gt_map_labels, gt_ego_fut_masks)
            stats["loss_map_bound"] = self.map_bound_loss_weight * loss_map_bound

        if self.map_dir_loss_fn is not None and gt_map_pts is not None and _pred_abs is not None:
            _gt_map_pts = gt_map_pts.data if hasattr(gt_map_pts, 'data') else gt_map_pts
            _gt_map_labels = gt_map_labels.data if hasattr(gt_map_labels, 'data') else gt_map_labels
            loss_map_dir = self.map_dir_loss_fn(_pred_abs, _gt_map_pts, _gt_map_labels, gt_ego_fut_masks)
            stats["loss_map_dir"] = self.map_dir_loss_weight * loss_map_dir

        loss_motion = stats.get("loss_motion", perception_embs.sum() * 0.0)

        ar_loss_total = torch.tensor(0.0, device=planning_loss.device)
        if self.train_vlm and ar_batch is not None:
            ar_outputs = self.forward_ar_batch(ar_batch)
            ar_loss_total = ar_outputs['loss_ar']
            stats['loss_vlm_raw'] = ar_outputs['loss_vlm_raw']
            stats['ar_planning_ratio'] = (ar_outputs['loss_vlm_raw'] / (planning_loss.detach() + 1e-6)).item()
        else:
            stats['loss_vlm_raw'] = torch.tensor(0.0, device=planning_loss.device)
            stats['ar_planning_ratio'] = 0.0
        if self.train_vlm:
            stats['loss_ar'] = ar_loss_total

        if self.enable_knowledge_insulation:
            total_loss = self.ar_loss_weight * ar_loss + planning_loss + loss_motion
        else:
            total_loss = planning_loss + loss_motion
        total_loss = total_loss + ar_loss_total

        total_loss = total_loss + loss_depth_val
        stats["loss_depth"] = loss_depth_val

        for geo_key in ("loss_collision", "loss_map_bound", "loss_map_dir"):
            if geo_key in stats:
                total_loss = total_loss + stats[geo_key]

        stats["loss"] = total_loss

        fixed_keys = ["loss_planning", "loss", "loss_ego_status", "loss_motion"]
        zero_val = total_loss.sum() * 0.0
        for k in fixed_keys:
            stats.setdefault(k, zero_val)

        return {"losses": stats}

    @torch.no_grad()
    def forward_test(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        permute_indices = [0, 1, 2, 4, 5, 3]
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]

        def _permute_metas_per_camera_fields(img_metas):
            if not isinstance(img_metas, list):
                return img_metas
            out = []
            for m in img_metas:
                if not isinstance(m, dict):
                    out.append(m)
                    continue
                m2 = dict(m)
                for k, v in list(m2.items()):
                    if isinstance(v, list) and len(v) == 6:
                        m2[k] = [v[i] for i in permute_indices]
                cams = m2.get("cams", None)
                if isinstance(cams, dict) and len(cams) == 6:
                    ordered_keys = list(cams.keys())
                    if all(k in cams for k in TARGET_SENSOR_ORDER):
                        ordered_keys = TARGET_SENSOR_ORDER
                    m2["cams"] = {k: cams[k] for k in [ordered_keys[i] for i in permute_indices]}
                out.append(m2)
            return out

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = _permute_metas_per_camera_fields(kwargs.get("img_metas"))

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            target_point=kwargs.get('target_point'),
        )
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        num_steps = int(self.num_sample_steps if num_steps is None else num_steps)

        if noise is None:
            if self.inference_seed is not None:
                _gen = torch.Generator(device=device).manual_seed(self.inference_seed)
                noise = self.sample_noise((bsz, self.action_horizon, self.action_dim), device, generator=_gen)
            else:
                noise = self.sample_noise((bsz, self.action_horizon, self.action_dim), device)

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features = self.embed_prefix(batch)

        if prefix_embs.dtype != dtype:
            prefix_embs = prefix_embs.to(dtype)

        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(source_features, bsz, all_image_grids, self.feature_source)
        feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
        if not self.feat_grad:
            feature_maps_daf = [x.detach() for x in feature_maps_daf]

        head_dtype = next(self.unified_decoder.parameters()).dtype
        head_device = device

        perception_metas = {
            'img_metas': kwargs.get('img_metas'),
            'timestamp': timestamp,
            'projection_mat': projection_mat.to(device=head_device, dtype=head_dtype) if projection_mat is not None else None,
            'image_wh': image_wh.to(device=head_device, dtype=head_dtype) if image_wh is not None else None,
        }

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)
        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz, device, dtype, stage1_outs
        )

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d)

        prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "eager"

        _ds_embeds_test = deepstack_features if (self.driving_deepstack and deepstack_features is not None) else None
        _vis_masks_test = (prefix_input_ids == self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id) if _ds_embeds_test is not None else None

        _, past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
            deepstack_visual_embeds=_ds_embeds_test,
            visual_pos_masks=_vis_masks_test,
        )

        perception_len = perception_embs.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, perception_len, prefix_len)
        perception_att_2d = make_att_2d_masks(perception_pad_masks, perception_att_masks)
        perception_full_att_2d = torch.cat([prefix_pad_2d, perception_att_2d], dim=2)
        perception_full_att_2d_4d = self._prepare_attention_masks_4d(perception_full_att_2d)

        perception_range = torch.arange(1, perception_len + 1, device=device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

        self.qwen3_vl_with_expert.qwen3_perception_expert.config._attn_implementation = "eager"

        (_, perception_out, _), past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=perception_full_att_2d_4d,
            position_ids=perception_pos_ids_3d,
            past_key_values=past_key_values,
            inputs_embeds=[None, perception_embs, None],
            use_cache=True,
        )

        det_len = perception_lengths['det']
        map_len = perception_lengths['map']
        ego_len = perception_lengths['ego']
        motion_len = perception_lengths['motion']

        d0 = 0
        d1 = d0 + det_len
        m0 = d1
        m1 = m0 + map_len
        e0 = m1
        e1 = e0 + ego_len
        t0 = e1
        t1 = t0 + motion_len

        if perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]

            proj_dtype = self.det_proj.weight.dtype
            target_dtype = self.det_proj.weight.dtype
            motion_token_256 = stage1_outs.get('motion_token', None)
            ego_feat_stage1 = stage1_outs['ego_instance_feature']

            det_feat_fused = self.det_proj(det_out_vlm.to(proj_dtype)).to(torch.float32)
            map_feat_fused = self.map_proj(map_out_vlm.to(proj_dtype)).to(torch.float32)
            ego_feat_fused = (
                self.ego_proj_down(ego_out.to(proj_dtype)).to(torch.float32)
                if ego_out is not None
                else ego_feat_stage1.to(torch.float32)
            )
            motion_feat_fused = (
                self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                if motion_token_256 is not None
                else None
            )

            vlm_enhanced = {
                'det_feat': det_feat_fused.to(target_dtype),
                'map_feat': map_feat_fused.to(target_dtype),
                'ego_feat': ego_feat_fused.to(target_dtype),
            }
            if motion_feat_fused is not None:
                vlm_enhanced['motion_feat'] = motion_feat_fused.to(target_dtype)

            stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)
            det_result, map_result = self.unified_decoder.post_process(stage2_outs)
        else:
            ego_out = None
            det_result = None
            map_result = None
            stage2_outs = None

        cached_pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks], dim=1)
        dt_val = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)

        x_t = noise
        time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)

        if stage2_outs and 'ego_status_list' in stage2_outs and len(stage2_outs['ego_status_list']) > 0:
            ego_status_pred = stage2_outs['ego_status_list'][-1].squeeze(1).to(torch.float32)
        else:
            ego_status_pred = x_t.new_zeros((bsz, self.ego_status_dim), dtype=torch.float32)

        max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values

        while time_tensor >= -dt_val / 2:
            expanded_time = time_tensor.expand(bsz)
            v_t = self._denoise_step(
                batch, cached_pad_masks, past_key_values, x_t.to(dtype), expanded_time.to(dtype), max_perception_pos, ego_status_pred=ego_status_pred,
            )
            x_t = x_t + dt_val * v_t
            time_tensor += dt_val

        traj_pred_meter_deltas = self.denorm_delta(x_t)
        zeros = torch.zeros((bsz, 1, 2), device=device, dtype=x_t.dtype)
        traj_pred_points_meter = zeros + torch.cumsum(traj_pred_meter_deltas, dim=1)

        return {
            "traj": traj_pred_points_meter,
            "det": det_result,
            "map": map_result,
        }

    def _denoise_step(self, batch, cached_pad_masks, past_key_values, x_t, timestep, max_cached_position_ids, *, ego_status_pred: Optional[torch.Tensor] = None):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch,
            x_t,
            timestep,
            ego_status_pred=ego_status_pred,
            use_gt_status=False,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = cached_pad_masks.shape[0]
        cached_len = cached_pad_masks.shape[1]

        cached_pad_2d_masks = cached_pad_masks[:, None, :].expand(batch_size, suffix_len, cached_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([cached_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        suffix_range = torch.arange(1, suffix_len + 1, device=max_cached_position_ids.device).view(1, -1).expand(batch_size, -1)
        suffix_pos_ids_1d = max_cached_position_ids + suffix_range

        position_ids = torch.stack([suffix_pos_ids_1d, suffix_pos_ids_1d, suffix_pos_ids_1d], dim=0)

        self.qwen3_vl_with_expert.qwen3_action_expert.config._attn_implementation = "eager"

        outputs_embeds, _, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.action_horizon:].to(dtype=torch.float32)
        model_out = self.action_out_proj(suffix_out)
        return model_out

    def forward_ar_batch(self, ar_batch):
        input_ids = ar_batch['ar_input_ids']
        labels = ar_batch['ar_labels']
        device = input_ids.device
        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        raw_pv = ar_batch.get('ar_pixel_values', None)
        pixel_values_tensor = None
        if raw_pv is not None:
            flat = []
            for item in raw_pv:
                if isinstance(item, (list, tuple)):
                    for t in item:
                        flat.append(t.to(device))
                else:
                    flat.append(item.to(device))
            if flat:
                pixel_values_tensor = torch.cat(flat, dim=0)

        raw_thw = ar_batch.get('ar_image_grid_thw', None)
        flat_image_grid_thw = None
        if raw_thw is not None:
            if isinstance(raw_thw, (list, tuple)):
                parts = [t.to(device) for t in raw_thw if t is not None]
                if parts:
                    flat_image_grid_thw = torch.cat(parts, dim=0)
            elif torch.is_tensor(raw_thw):
                if raw_thw.dim() == 3:
                    flat_image_grid_thw = raw_thw.reshape(-1, 3).to(device)
                else:
                    flat_image_grid_thw = raw_thw.to(device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        position_ids, _ = self.qwen3_vl_with_expert.vlm_base.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=flat_image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "flash_attention_2"
        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.qwen3_vl_with_expert.qwen3_vl(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values_tensor,
                image_grid_thw=flat_image_grid_thw,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )

        loss_vlm = outputs.loss

        return dict(
            loss_ar=self.ar_loss_weight * loss_vlm,
            loss_vlm_raw=loss_vlm.detach(),
        )
