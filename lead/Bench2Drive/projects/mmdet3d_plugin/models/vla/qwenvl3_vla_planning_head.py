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
from string import Formatter
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


def _prompt_format_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }


def _render_user_prompt(
    template: str,
    nav_cmd: str,
    ego_status_row,
    strict_user_prompt_fields: Optional[tuple[str, ...]] = None,
) -> str:
    if strict_user_prompt_fields is not None:
        return template.format(nav_cmd=nav_cmd)
    return template.format(
        nav_cmd=nav_cmd,
        speed=ego_status_row[0],
        acc_x=ego_status_row[1],
        acc_y=ego_status_row[2],
    )


@dataclass(frozen=True)
class PlanningTokenLayout:
    status: slice
    route: slice
    waypoint: slice
    speed: int
    flow: slice
    total_tokens: int

    @classmethod
    def build(
        cls,
        *,
        num_status_tokens: int,
        num_route_queries: int,
        num_waypoint_queries: int,
        num_flow_tokens: int,
    ) -> "PlanningTokenLayout":
        status = slice(0, num_status_tokens)
        route = slice(status.stop, status.stop + num_route_queries)
        waypoint = slice(route.stop, route.stop + num_waypoint_queries)
        speed = waypoint.stop
        flow = slice(speed + 1, speed + 1 + num_flow_tokens)
        return cls(
            status=status,
            route=route,
            waypoint=waypoint,
            speed=speed,
            flow=flow,
            total_tokens=flow.stop,
        )

@dataclass(frozen=True)
class HiPADPlanningTokenLayout:
    status: slice
    temporal: slice
    spatial: slice
    flow: slice
    total_tokens: int

    @classmethod
    def build(
        cls,
        *,
        num_status_tokens: int,
        num_temporal_queries: int,
        num_spatial_queries: int,
        num_flow_tokens: int,
    ) -> "HiPADPlanningTokenLayout":
        status = slice(0, num_status_tokens)
        temporal = slice(status.stop, status.stop + num_temporal_queries)
        spatial = slice(temporal.stop, temporal.stop + num_spatial_queries)
        flow = slice(spatial.stop, spatial.stop + num_flow_tokens)
        return cls(
            status=status,
            temporal=temporal,
            spatial=spatial,
            flow=flow,
            total_tokens=flow.stop,
        )


def _split_action_expert_planning_tokens(
    suffix_out: torch.Tensor,
    layout: PlanningTokenLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if suffix_out.shape[1] != layout.total_tokens:
        raise RuntimeError(
            "Action expert returned an unexpected number of suffix tokens: "
            f"expected {layout.total_tokens}, got {suffix_out.shape[1]}"
        )
    route_tokens = suffix_out[:, layout.route]
    waypoint_tokens = suffix_out[:, layout.waypoint]
    speed_token = suffix_out[:, layout.speed]
    flow_tokens = suffix_out[:, layout.flow]
    return route_tokens, waypoint_tokens, speed_token, flow_tokens


@dataclass
class DrivingBatch:
    images: torch.Tensor
    image_masks: Dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    command: Optional[torch.Tensor]
    ego_status: Optional[torch.Tensor]
    target_points: Optional[torch.Tensor] = None
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
        action_delta_mu: Optional[List[float]] = None,
        action_delta_std: Optional[List[float]] = None,
        aux_route_loss_weight: float = 0.0,
        aux_target_speed_loss_weight: float = 0.0,
        target_speed_classes: Optional[List[float]] = None,
        route_points_prediction: int = 10,
        planning_query_layout: Literal[
            "perception_shared", "action_expert", "action_expert_hipad"
        ] = "perception_shared",
        planning_status_layout: Literal["legacy_fused", "tfv6_5token"] = "legacy_fused",
        use_deterministic_waypoint_queries: bool = False,
        deterministic_waypoint_loss_weight: float = 0.0,
        temporal_waypoint_queries: Optional[int] = None,
        inject_navigation_into_ego: bool = True,
        spatial_waypoint_queries: int = 6,
        spatial_waypoint_loss_weight: float = 0.0,
        velocity_normalization: float = 25.0,
        prefix_max_len: Optional[int] = 3670,
        target_points_normalization_constants: Optional[List[float]] = None,
        waypoint_hz: float = 2.0,
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
        peft_cfg: Optional[dict] = None,
        visual_lora_cfg: Optional[dict] = None,
        visual_wasp_cfg: Optional[dict] = None,
        visual_projector_wasp_cfg: Optional[dict] = None,
        inference_seed: Optional[int] = None,
        driving_deepstack: bool = False,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        strict_user_prompt_fields: Optional[List[str]] = None,
        view_tokens: Optional[List[str]] = None,
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
        self.aux_route_loss_weight = float(aux_route_loss_weight)
        self.aux_target_speed_loss_weight = float(aux_target_speed_loss_weight)
        self.route_points_prediction = int(route_points_prediction)
        self.planning_query_layout = str(planning_query_layout)
        if self.planning_query_layout not in {
            "perception_shared",
            "action_expert",
            "action_expert_hipad",
            "hipad_plan_query",
        }:
            raise ValueError(
                "planning_query_layout must be 'perception_shared', 'action_expert', "
                "'action_expert_hipad', or 'hipad_plan_query', "
                f"got {self.planning_query_layout!r}"
            )
        self.planning_status_layout = str(planning_status_layout)
        if self.planning_status_layout not in {"legacy_fused", "tfv6_5token"}:
            raise ValueError(
                "planning_status_layout must be 'legacy_fused' or 'tfv6_5token', "
                f"got {self.planning_status_layout!r}"
            )
        self.num_status_tokens = (
            5 if self.planning_status_layout == "tfv6_5token" else 1
        )
        self.use_deterministic_waypoint_queries = bool(
            use_deterministic_waypoint_queries
        )
        self.deterministic_waypoint_loss_weight = float(
            deterministic_waypoint_loss_weight
        )
        self.temporal_waypoint_queries = int(
            action_horizon
            if temporal_waypoint_queries is None
            else temporal_waypoint_queries
        )
        if self.temporal_waypoint_queries <= 1:
            raise ValueError("temporal_waypoint_queries must be greater than 1")
        self.spatial_waypoint_queries = int(spatial_waypoint_queries)
        self.spatial_waypoint_loss_weight = float(spatial_waypoint_loss_weight)
        if self.spatial_waypoint_queries <= 1:
            raise ValueError(
                "spatial_waypoint_queries must be greater than 1, got "
                f"{self.spatial_waypoint_queries}"
            )
        self.inject_navigation_into_ego = bool(inject_navigation_into_ego)
        self.velocity_normalization = float(velocity_normalization)
        if self.velocity_normalization <= 0:
            raise ValueError(
                "velocity_normalization must be positive, got "
                f"{self.velocity_normalization}"
            )
        self.prefix_max_len = (
            None if prefix_max_len is None else int(prefix_max_len)
        )
        if self.prefix_max_len is not None and self.prefix_max_len <= 0:
            raise ValueError(
                f"prefix_max_len must be positive or None, got {self.prefix_max_len}"
            )
        self.waypoint_hz = float(waypoint_hz)
        if self.waypoint_hz <= 0:
            raise ValueError(f"waypoint_hz must be positive, got {self.waypoint_hz}")
        if target_points_normalization_constants is None:
            target_points_normalization_constants = [200.0, 50.0]
        if len(target_points_normalization_constants) != 2:
            raise ValueError(
                "target_points_normalization_constants must contain [forward, lateral] scales"
            )
        self.register_buffer(
            "target_points_normalization_constants",
            torch.tensor(target_points_normalization_constants, dtype=torch.float32).view(1, 1, 2),
            persistent=False,
        )
        if target_speed_classes is None:
            target_speed_classes = [
                0.0,
                4.0,
                8.0,
                10.0,
                13.88888888,
                16.0,
                17.77777777,
                20.0,
            ]
        self.register_buffer(
            "target_speed_classes",
            torch.tensor(target_speed_classes, dtype=torch.float32),
            persistent=False,
        )
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

        if action_delta_mu is None:
            action_delta_mu = [-0.0222, 2.0249]
        if action_delta_std is None:
            action_delta_std = [0.6720, 1.8586]
        if len(action_delta_mu) != action_dim or len(action_delta_std) != action_dim:
            raise ValueError(
                "action_delta_mu/action_delta_std must match action_dim "
                f"({action_dim}), got {len(action_delta_mu)} and {len(action_delta_std)}"
            )
        self.register_buffer(
            "action_delta_mu",
            torch.tensor(action_delta_mu, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_delta_std",
            torch.tensor(action_delta_std, dtype=torch.float32),
            persistent=False,
        )

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
            peft_cfg=peft_cfg,
            visual_lora_cfg=visual_lora_cfg,
            visual_wasp_cfg=visual_wasp_cfg,
            visual_projector_wasp_cfg=visual_projector_wasp_cfg,
        )

        self._wasp_loss_weight = 0.0
        if peft_cfg is not None:
            peft_type = str(peft_cfg.get("type", "lora")).lower()
            if peft_type == "wasp":
                self._wasp_loss_weight = float(
                    peft_cfg.get("wasp_loss_weight", peft_cfg.get("beta", 0.01))
                )

        self._visual_wasp_loss_weight = 0.0
        if visual_wasp_cfg is not None:
            visual_wasp_type = str(
                visual_wasp_cfg.get("type", "wasp")
            ).lower()
            if visual_wasp_type != "wasp":
                raise ValueError(
                    "visual_wasp_cfg only supports type='wasp'; "
                    f"got {visual_wasp_type!r}."
                )
            self._visual_wasp_loss_weight = float(
                visual_wasp_cfg.get(
                    "wasp_loss_weight", visual_wasp_cfg.get("beta", 0.01)
                )
            )

        self._visual_projector_wasp_loss_weight = 0.0
        if visual_projector_wasp_cfg is not None:
            projector_wasp_type = str(
                visual_projector_wasp_cfg.get("type", "wasp")
            ).lower()
            if projector_wasp_type != "wasp":
                raise ValueError(
                    "visual_projector_wasp_cfg only supports type='wasp'; "
                    f"got {projector_wasp_type!r}."
                )
            self._visual_projector_wasp_loss_weight = float(
                visual_projector_wasp_cfg.get(
                    "wasp_loss_weight",
                    visual_projector_wasp_cfg.get("beta", 0.01),
                )
            )

        self._compute_wasp_regularization_this_step = True
        self._wasp_regularization_step_scale = 1.0

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

        if dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif dtype == "float32":
            target_dtype = torch.float32
        else:
            target_dtype = torch.float16

        self.action_in_proj = nn.Linear(action_dim, action_expert_cfg.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_cfg.hidden_size, action_dim)
        self.planning_token_layout = None
        self.spatial_waypoint_query = None
        self.spatial_waypoint_out_proj = None
        if self.planning_query_layout == "action_expert":
            self.route_query = nn.Parameter(
                torch.empty(
                    1,
                    self.route_points_prediction,
                    action_hidden,
                    dtype=target_dtype,
                )
            )
            if self.use_deterministic_waypoint_queries:
                self.deterministic_waypoint_query = nn.Parameter(
                    torch.empty(
                        1,
                        self.action_horizon,
                        action_hidden,
                        dtype=target_dtype,
                    )
                )
            else:
                self.deterministic_waypoint_query = None
            self.target_speed_query = nn.Parameter(
                torch.empty(1, 1, action_hidden, dtype=target_dtype)
            )
            nn.init.uniform_(self.route_query)
            if self.deterministic_waypoint_query is not None:
                nn.init.uniform_(self.deterministic_waypoint_query)
            nn.init.uniform_(self.target_speed_query)
            self.planning_token_layout = PlanningTokenLayout.build(
                num_status_tokens=self.num_status_tokens,
                num_route_queries=self.route_points_prediction,
                num_waypoint_queries=(
                    self.action_horizon
                    if self.use_deterministic_waypoint_queries
                    else 0
                ),
                num_flow_tokens=self.action_horizon,
            )
            self.route_out_proj = nn.Linear(action_hidden, 2)
            self.deterministic_waypoint_out_proj = (
                nn.Linear(action_hidden, 2)
                if self.use_deterministic_waypoint_queries
                else None
            )
            self.target_speed_out_proj = nn.Sequential(
                nn.Linear(action_hidden, action_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(action_hidden, len(target_speed_classes)),
            )
            self.aux_route_head = None
            self.aux_target_speed_head = None
        elif self.planning_query_layout == "action_expert_hipad":
            if not self.use_deterministic_waypoint_queries:
                raise ValueError(
                    "action_expert_hipad requires "
                    "use_deterministic_waypoint_queries=True"
                )
            self.route_query = None
            self.target_speed_query = None
            # Keep the latest LEAD checkpoint's 8-point parameter shape; only
            # the configured temporal prefix is exposed as planning tokens.
            self.deterministic_waypoint_query = nn.Parameter(
                torch.empty(
                    1,
                    self.action_horizon,
                    action_hidden,
                    dtype=target_dtype,
                )
            )
            self.spatial_waypoint_query = nn.Parameter(
                torch.empty(
                    1,
                    self.spatial_waypoint_queries,
                    action_hidden,
                    dtype=target_dtype,
                )
            )
            nn.init.uniform_(self.deterministic_waypoint_query)
            nn.init.uniform_(self.spatial_waypoint_query)
            self.planning_token_layout = HiPADPlanningTokenLayout.build(
                num_status_tokens=self.num_status_tokens,
                num_temporal_queries=self.temporal_waypoint_queries,
                num_spatial_queries=self.spatial_waypoint_queries,
                num_flow_tokens=self.action_horizon,
            )
            self.route_out_proj = None
            self.target_speed_out_proj = None
            self.deterministic_waypoint_out_proj = nn.Linear(action_hidden, 2)
            self.spatial_waypoint_out_proj = nn.Linear(action_hidden, 2)
            self.aux_route_head = None
            self.aux_target_speed_head = None
        elif self.planning_query_layout == "hipad_plan_query":
            self.route_query = None
            self.deterministic_waypoint_query = None
            self.spatial_waypoint_query = None
            self.target_speed_query = None
            self.route_out_proj = None
            self.deterministic_waypoint_out_proj = None
            self.spatial_waypoint_out_proj = None
            self.target_speed_out_proj = None
            self.aux_route_head = None
            self.aux_target_speed_head = None
            self.planning_token_layout = HiPADPlanningTokenLayout.build(
                num_status_tokens=self.num_status_tokens,
                num_temporal_queries=0,
                num_spatial_queries=0,
                num_flow_tokens=self.action_horizon,
            )
        else:
            self.route_query = None
            self.deterministic_waypoint_query = None
            self.target_speed_query = None
            self.route_out_proj = None
            self.deterministic_waypoint_out_proj = None
            self.target_speed_out_proj = None
            self.aux_route_head = nn.Linear(
                self.vlm_hidden_size,
                self.route_points_prediction * 2,
            )
            self.aux_target_speed_head = nn.Sequential(
                nn.Linear(self.vlm_hidden_size, self.vlm_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.vlm_hidden_size, len(target_speed_classes)),
            )
        self.target_point_encoder = None
        if self.inject_navigation_into_ego:
            self.target_point_encoder = nn.Sequential(
                nn.Linear(6, self.embed_dims),
                nn.GELU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

        self.status_mlp = None
        self.velocity_encoder = None
        self.command_encoder = None
        self.tp_encoder = None
        self.status_position_embedding = None
        if self.planning_status_layout == "tfv6_5token":
            self.velocity_encoder = nn.Linear(1, action_hidden)
            self.command_encoder = nn.Linear(6, action_hidden)
            self.tp_encoder = nn.Linear(2, action_hidden)
            self.status_position_embedding = nn.Parameter(
                torch.empty(
                    1,
                    self.num_status_tokens,
                    action_hidden,
                    dtype=target_dtype,
                )
            )
            nn.init.uniform_(self.status_position_embedding)
        else:
            status_in_features = 6 + self.ego_status_dim
            self.status_mlp = Mlp(
                in_features=status_in_features,
                hidden_features=action_expert_cfg.hidden_size,
                out_features=action_expert_cfg.hidden_size,
                norm_layer=nn.LayerNorm,
            )

        self.action_time_mlp_in = nn.Linear(2 * action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)

        self.action_in_proj.to(target_dtype)
        if self.route_out_proj is not None:
            self.route_out_proj.to(target_dtype)
            if self.deterministic_waypoint_out_proj is not None:
                self.deterministic_waypoint_out_proj.to(target_dtype)
            self.target_speed_out_proj.to(target_dtype)
        elif self.planning_query_layout == "action_expert_hipad":
            self.deterministic_waypoint_out_proj.to(target_dtype)
            self.spatial_waypoint_out_proj.to(target_dtype)
        elif self.planning_query_layout == "hipad_plan_query":
            pass
        else:
            self.aux_route_head.to(target_dtype)
            self.aux_target_speed_head.to(target_dtype)
        if self.target_point_encoder is not None:
            self.target_point_encoder.to(target_dtype)
        if self.status_mlp is not None:
            self.status_mlp.to(target_dtype)
        if self.velocity_encoder is not None:
            self.velocity_encoder.to(target_dtype)
            self.command_encoder.to(target_dtype)
            self.tp_encoder.to(target_dtype)
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

        self.system_prompt = system_prompt or B2D_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or B2D_USER_PROMPT_TEMPLATE
        self.strict_user_prompt_fields = (
            None
            if strict_user_prompt_fields is None
            else tuple(strict_user_prompt_fields)
        )
        if self.strict_user_prompt_fields is not None:
            expected_fields = set(self.strict_user_prompt_fields)
            if expected_fields != {"nav_cmd"}:
                raise ValueError(
                    "strict_user_prompt_fields only permits ['nav_cmd']; "
                    f"got {sorted(expected_fields)}"
                )
            actual_fields = _prompt_format_fields(self.user_prompt_template)
            if actual_fields != expected_fields:
                raise ValueError(
                    "user_prompt_template fields do not match the strict allowlist: "
                    f"expected {sorted(expected_fields)}, got {sorted(actual_fields)}"
                )
        self.view_token_str_list = view_tokens or B2D_VIEW_TOKENS
        self.view_token_ids = None

    def _get_view_token_ids(self, device):
        if self.view_token_ids is None:
            tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
            ids = [tokenizer.convert_tokens_to_ids(t) for t in self.view_token_str_list]
            self.view_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        return self.view_token_ids.to(device)

    @staticmethod
    def _camera_permute_indices(num_cams: int):
        if num_cams >= 6:
            return [0, 1, 2, 4, 5, 3]
        return list(range(num_cams))

    def _permute_camera_tensor(self, tensor, permute_indices):
        if tensor is None:
            return None
        if not torch.is_tensor(tensor) or tensor.dim() < 2:
            return tensor
        return tensor[:, permute_indices]

    def _permute_metas_per_camera_fields(self, img_metas, permute_indices):
        if not isinstance(img_metas, list):
            return img_metas
        out = []
        num_cams = len(permute_indices)
        for m in img_metas:
            if not isinstance(m, dict):
                out.append(m)
                continue
            m2 = dict(m)
            for k, v in list(m2.items()):
                if isinstance(v, list) and len(v) >= num_cams:
                    m2[k] = [v[i] for i in permute_indices]
            cams = m2.get("cams", None)
            if isinstance(cams, dict) and len(cams) >= num_cams:
                ordered_keys = list(cams.keys())
                if all(k in cams for k in TARGET_SENSOR_ORDER[:num_cams]):
                    ordered_keys = TARGET_SENSOR_ORDER
                m2["cams"] = {k: cams[k] for k in [ordered_keys[i] for i in permute_indices]}
            out.append(m2)
        return out

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

    def set_wasp_accumulation_state(self, compute: bool, scale: float = 1.0):
        self._compute_wasp_regularization_this_step = bool(compute)
        self._wasp_regularization_step_scale = float(scale) if compute else 0.0

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
        mu = self.action_delta_mu.to(device=delta_meter.device, dtype=delta_meter.dtype)
        std = self.action_delta_std.to(device=delta_meter.device, dtype=delta_meter.dtype)
        return (delta_meter - mu) / (std + 1e-6)

    def denorm_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        mu = self.action_delta_mu.to(device=delta_norm.device, dtype=delta_norm.dtype)
        std = self.action_delta_std.to(device=delta_norm.device, dtype=delta_norm.dtype)
        return delta_norm * (std + 1e-6) + mu

    def _encode_target_speed_two_hot(
        self,
        target_speed: torch.Tensor,
        brake: Optional[torch.Tensor],
    ) -> torch.Tensor:
        classes = self.target_speed_classes.to(
            device=target_speed.device,
            dtype=target_speed.dtype,
        )
        scalar = target_speed.reshape(-1).clamp(
            min=float(classes[0]),
            max=float(classes[-1]),
        )
        labels = torch.zeros(
            (scalar.shape[0], classes.numel()),
            device=target_speed.device,
            dtype=target_speed.dtype,
        )
        brake_mask = (
            brake.reshape(-1).to(device=target_speed.device, dtype=torch.bool)
            if brake is not None
            else torch.zeros_like(scalar, dtype=torch.bool)
        )
        labels[brake_mask, 0] = 1.0

        non_brake = ~brake_mask
        if non_brake.any():
            values = scalar[non_brake]
            upper = torch.searchsorted(classes, values, right=False).clamp(
                max=classes.numel() - 1
            )
            lower = torch.clamp(upper - 1, min=0)
            denominator = (classes[upper] - classes[lower]).clamp(min=1e-6)
            upper_weight = ((values - classes[lower]) / denominator).clamp(0.0, 1.0)
            rows = torch.nonzero(non_brake, as_tuple=False).reshape(-1)
            labels[rows, lower] = 1.0 - upper_weight
            labels[rows, upper] += upper_weight

            low_mask = values <= classes[0]
            high_mask = values >= classes[-1]
            labels[rows[low_mask]] = 0.0
            labels[rows[low_mask], 0] = 1.0
            labels[rows[high_mask]] = 0.0
            labels[rows[high_mask], -1] = 1.0
        return labels

    @staticmethod
    def _unwrap_aux_target(value):
        return value.data if hasattr(value, "data") else value

    def _decode_perception_planning(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dtype = self.aux_route_head.weight.dtype
        context = context.to(dtype=head_dtype)
        route_deltas = self.aux_route_head(context).float().view(
            -1,
            self.route_points_prediction,
            2,
        )
        pred_route = torch.cumsum(route_deltas, dim=1)
        speed_logits = self.aux_target_speed_head(context).float()
        return pred_route, speed_logits

    def _decode_action_expert_planning(
        self,
        suffix_out: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        if suffix_out.shape[1] != self.planning_token_layout.total_tokens:
            raise RuntimeError(
                "Action expert returned an unexpected number of suffix tokens: "
                f"expected {self.planning_token_layout.total_tokens}, "
                f"got {suffix_out.shape[1]}"
            )
        pred_route, pred_waypoints, speed_logits = (
            self._decode_action_expert_stable_planning(
                suffix_out[:, : self.planning_token_layout.flow.start]
            )
        )
        flow_tokens = suffix_out[:, self.planning_token_layout.flow]
        return pred_route, pred_waypoints, speed_logits, flow_tokens

    def _decode_action_expert_stable_planning(
        self,
        stable_out: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        layout = self.planning_token_layout
        expected_tokens = layout.flow.start
        if stable_out.shape[1] != expected_tokens:
            raise RuntimeError(
                "Action expert returned an unexpected number of stable planning "
                f"tokens: expected {expected_tokens}, got {stable_out.shape[1]}"
            )
        route_tokens = stable_out[:, layout.route]
        waypoint_tokens = stable_out[:, layout.waypoint]
        speed_token = stable_out[:, layout.speed]

        route_dtype = self.route_out_proj.weight.dtype
        route_deltas = self.route_out_proj(
            route_tokens.to(dtype=route_dtype)
        ).float()
        pred_route = torch.cumsum(route_deltas, dim=1)
        pred_waypoints = None
        if self.deterministic_waypoint_out_proj is not None:
            waypoint_deltas = self.deterministic_waypoint_out_proj(
                waypoint_tokens.to(dtype=route_dtype)
            ).float()
            pred_waypoints = torch.cumsum(waypoint_deltas, dim=1)
        speed_logits = self.target_speed_out_proj(
            speed_token.to(dtype=route_dtype)
        ).float()
        return pred_route, pred_waypoints, speed_logits

    def _decode_action_expert_hipad_planning(
        self,
        suffix_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = self.planning_token_layout
        if suffix_out.shape[1] != layout.total_tokens:
            raise RuntimeError(
                "HiP-AD action expert returned an unexpected number of suffix "
                f"tokens: expected {layout.total_tokens}, got {suffix_out.shape[1]}"
            )
        pred_temporal, pred_spatial = self._decode_action_expert_hipad_stable(
            suffix_out[:, : layout.flow.start]
        )
        return pred_temporal, pred_spatial, suffix_out[:, layout.flow]

    def _decode_action_expert_hipad_stable(
        self,
        stable_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = self.planning_token_layout
        if stable_out.shape[1] != layout.flow.start:
            raise RuntimeError(
                "HiP-AD action expert returned an unexpected number of stable "
                f"tokens: expected {layout.flow.start}, got {stable_out.shape[1]}"
            )
        output_dtype = self.deterministic_waypoint_out_proj.weight.dtype
        temporal_deltas = self.deterministic_waypoint_out_proj(
            stable_out[:, layout.temporal].to(dtype=output_dtype)
        ).float()
        spatial_deltas = self.spatial_waypoint_out_proj(
            stable_out[:, layout.spatial].to(dtype=output_dtype)
        ).float()
        return (
            torch.cumsum(temporal_deltas, dim=1),
            torch.cumsum(spatial_deltas, dim=1),
        )

    def _decode_target_speed(
        self,
        speed_logits: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if speed_logits is None:
            return None, None
        speed_distribution = torch.softmax(speed_logits.float(), dim=-1)
        speed_classes = self.target_speed_classes.to(
            device=speed_distribution.device,
            dtype=speed_distribution.dtype,
        )
        speed_scalar = torch.sum(
            speed_distribution * speed_classes.unsqueeze(0),
            dim=-1,
        )
        return speed_distribution, speed_scalar

    def _compute_aux_planning_losses(
        self,
        pred_route: Optional[torch.Tensor],
        pred_waypoints: Optional[torch.Tensor],
        speed_logits: Optional[torch.Tensor],
        kwargs: dict,
        gt_ego_fut_trajs: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ) -> dict:
        losses = {}
        if pred_route is None and pred_waypoints is None and speed_logits is None:
            return losses

        gt_route = self._unwrap_aux_target(kwargs.get("gt_route"))
        if (
            self.aux_route_loss_weight > 0
            and pred_route is not None
            and torch.is_tensor(gt_route)
        ):
            gt_route = gt_route.to(device=pred_route.device, dtype=torch.float32)
            if gt_route.dim() == 4 and gt_route.shape[1] == 1:
                gt_route = gt_route.squeeze(1)
            target_route = gt_route[:, : self.route_points_prediction]
            route_ade = F.l1_loss(pred_route, target_route)
            route_fde = F.l1_loss(pred_route[:, -1, :], target_route[:, -1, :])
            losses["loss_spatial_route"] = (
                self.aux_route_loss_weight * (route_ade + route_fde)
            )

        if (
            self.deterministic_waypoint_loss_weight > 0
            and pred_waypoints is not None
            and torch.is_tensor(gt_ego_fut_trajs)
        ):
            gt_waypoint_deltas = gt_ego_fut_trajs.to(
                device=pred_waypoints.device,
                dtype=torch.float32,
            )
            if gt_waypoint_deltas.dim() == 4 and gt_waypoint_deltas.shape[1] == 1:
                gt_waypoint_deltas = gt_waypoint_deltas.squeeze(1)
            gt_waypoints_absolute = torch.cumsum(gt_waypoint_deltas, dim=1)
            gt_waypoints_absolute = gt_waypoints_absolute[:, : self.action_horizon]
            losses["loss_spatio_temporal_waypoints"] = (
                self.deterministic_waypoint_loss_weight
                * F.l1_loss(pred_waypoints, gt_waypoints_absolute)
            )

        gt_target_speed = self._unwrap_aux_target(kwargs.get("gt_target_speed"))
        gt_brake = self._unwrap_aux_target(kwargs.get("gt_brake"))
        if (
            self.aux_target_speed_loss_weight > 0
            and speed_logits is not None
            and torch.is_tensor(gt_target_speed)
        ):
            gt_target_speed = gt_target_speed.to(
                device=speed_logits.device,
                dtype=torch.float32,
            )
            if torch.is_tensor(gt_brake):
                gt_brake = gt_brake.to(device=speed_logits.device, dtype=torch.float32)
            else:
                gt_brake = None
            speed_target = self._encode_target_speed_two_hot(
                gt_target_speed,
                gt_brake,
            )
            losses["loss_target_speed"] = self.aux_target_speed_loss_weight * F.cross_entropy(
                speed_logits.float(),
                speed_target,
            )

            with torch.no_grad():
                speed_distribution = torch.softmax(speed_logits.float(), dim=-1)
                speed_classes = self.target_speed_classes.to(
                    device=speed_logits.device,
                    dtype=speed_distribution.dtype,
                )
                pred_speed = torch.sum(
                    speed_distribution * speed_classes.unsqueeze(0), dim=-1
                )
                target_speed = torch.sum(
                    speed_target * speed_classes.unsqueeze(0), dim=-1
                )
                pred_centered = pred_speed - pred_speed.mean()
                target_centered = target_speed - target_speed.mean()
                correlation_denom = torch.sqrt(
                    torch.sum(pred_centered.square())
                    * torch.sum(target_centered.square())
                )
                correlation = torch.where(
                    correlation_denom > 1e-6,
                    torch.sum(pred_centered * target_centered)
                    / correlation_denom.clamp_min(1e-6),
                    correlation_denom.new_zeros(()),
                )
                losses["metric_target_speed_error"] = torch.mean(
                    torch.abs(pred_speed - target_speed)
                )
                losses["metric_target_speed_correlation"] = correlation
                losses["metric_target_speed_stop_probability"] = (
                    speed_distribution[:, 0].mean()
                )
                losses["metric_target_speed_hard_stop_rate"] = (
                    (speed_distribution[:, 0] > 0.9).float().mean()
                )
                gt_hard_stop = speed_target[:, 0] > 0.9
                pred_hard_stop = speed_distribution[:, 0] > 0.9
                committed_stop = pred_speed < 0.01
                override_band = pred_hard_stop & ~committed_stop
                losses["metric_gt_hard_stop_rate"] = gt_hard_stop.float().mean()
                losses["metric_speed_committed_stop_rate"] = (
                    committed_stop.float().mean()
                )
                losses["metric_speed_override_band_rate"] = (
                    override_band.float().mean()
                )

                ego_status_value = self._unwrap_aux_target(ego_status)
                if torch.is_tensor(ego_status_value):
                    ego_speed = ego_status_value.to(
                        device=speed_logits.device,
                        dtype=torch.float32,
                    ).reshape(speed_logits.shape[0], -1)[:, 0]
                    low_speed = ego_speed.abs() < 0.5
                    departure = low_speed & ~gt_hard_stop

                    def masked_rate(value, mask):
                        mask_float = mask.float()
                        count = mask_float.sum()
                        return (value.float() * mask_float).sum() / count.clamp_min(1.0)

                    losses["metric_low_speed_sample_rate"] = low_speed.float().mean()
                    losses["metric_departure_sample_rate"] = departure.float().mean()
                    losses["metric_gt_hard_stop_rate_at_low_speed"] = masked_rate(
                        gt_hard_stop, low_speed
                    )
                    losses["metric_pred_hard_stop_rate_at_low_speed"] = masked_rate(
                        pred_hard_stop, low_speed
                    )
                    losses["metric_false_hard_stop_rate_at_departure"] = masked_rate(
                        pred_hard_stop, departure
                    )
        return losses

    @staticmethod
    def _masked_absolute_waypoint_l1(
        prediction: torch.Tensor,
        target_deltas: torch.Tensor,
        target_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target_deltas = target_deltas.to(
            device=prediction.device, dtype=torch.float32
        )
        if target_deltas.dim() == 4 and target_deltas.shape[1] == 1:
            target_deltas = target_deltas.squeeze(1)
        target_deltas = target_deltas[:, : prediction.shape[1]]
        if target_deltas.shape != prediction.shape:
            raise ValueError(
                "Planning target shape does not match prediction: "
                f"target={tuple(target_deltas.shape)} "
                f"prediction={tuple(prediction.shape)}"
            )
        target_absolute = torch.cumsum(target_deltas, dim=1)

        if target_masks is None:
            mask = torch.ones(
                prediction.shape[:2],
                device=prediction.device,
                dtype=torch.float32,
            )
        else:
            mask = target_masks.to(
                device=prediction.device, dtype=torch.float32
            )
            if mask.dim() == 3 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            mask = mask[:, : prediction.shape[1]]
            if mask.shape != prediction.shape[:2]:
                raise ValueError(
                    "Planning mask shape does not match prediction: "
                    f"mask={tuple(mask.shape)} "
                    f"prediction={tuple(prediction.shape)}"
                )
        mask = mask.clamp(0.0, 1.0).unsqueeze(-1)
        error = torch.abs(prediction.float() - target_absolute) * mask
        return error.sum() / (mask.sum() * prediction.shape[-1]).clamp_min(1.0)

    def _compute_hipad_planning_losses(
        self,
        pred_temporal: torch.Tensor,
        pred_spatial: torch.Tensor,
        _flow_gt_deltas: torch.Tensor,
        _flow_gt_masks: Optional[torch.Tensor],
        kwargs: dict,
    ) -> dict:
        losses = {}
        gt_temporal_deltas = self._unwrap_aux_target(
            kwargs.get("gt_ego_fut_trajs_5hz")
        )
        gt_temporal_masks = self._unwrap_aux_target(
            kwargs.get("gt_ego_fut_masks_5hz")
        )
        if self.deterministic_waypoint_loss_weight > 0:
            if not torch.is_tensor(gt_temporal_deltas):
                raise ValueError(
                    "action_expert_hipad requires gt_ego_fut_trajs_5hz"
                )
            losses["loss_plan_speed_5hz"] = (
                self.deterministic_waypoint_loss_weight
                * self._masked_absolute_waypoint_l1(
                    pred_temporal,
                    gt_temporal_deltas,
                    gt_temporal_masks,
                )
            )

        gt_spatial_deltas = self._unwrap_aux_target(
            kwargs.get("gt_ego_spat_trajs_2m")
        )
        gt_spatial_masks = self._unwrap_aux_target(
            kwargs.get("gt_ego_spat_masks_2m")
        )
        if self.spatial_waypoint_loss_weight > 0:
            if not torch.is_tensor(gt_spatial_deltas):
                raise ValueError(
                    "action_expert_hipad requires gt_ego_spat_trajs_2m"
                )
            losses["loss_plan_spat_2m"] = (
                self.spatial_waypoint_loss_weight
                * self._masked_absolute_waypoint_l1(
                    pred_spatial,
                    gt_spatial_deltas,
                    gt_spatial_masks,
                )
            )
            if torch.is_tensor(gt_spatial_masks):
                spatial_mask = gt_spatial_masks.to(
                    device=pred_spatial.device, dtype=torch.float32
                )
                if spatial_mask.dim() == 3 and spatial_mask.shape[1] == 1:
                    spatial_mask = spatial_mask.squeeze(1)
                losses["metric_plan_spat_2m_valid_rate"] = (
                    spatial_mask.float().mean()
                )
                losses["metric_plan_spat_2m_full_rate"] = (
                    spatial_mask.bool().all(dim=-1).float().mean()
                )
        return losses

    def embed_perception(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stage1_outs: dict,
        target_points: Optional[torch.Tensor] = None,
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
            if self.inject_navigation_into_ego and target_points is not None:
                target_points = target_points.to(device=device, dtype=torch.float32)
                normalization = self.target_points_normalization_constants.to(
                    device=device,
                    dtype=target_points.dtype,
                )
                normalized_target_points = (target_points / normalization).reshape(batch_size, 6)
                nav_dtype = self.target_point_encoder[0].weight.dtype
                navigation_feature = self.target_point_encoder(
                    normalized_target_points.to(dtype=nav_dtype)
                ).unsqueeze(1)
                ego_feat = ego_feat.to(dtype=navigation_feature.dtype) + navigation_feature
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
            # Start one bidirectional perception segment after the causal prefix.
            if perception_att_masks.shape[1] > 0:
                perception_att_masks[:, 0] = True
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
            num_views = max(1, all_image_grids.shape[0] // max(1, bsz))

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
                num_views = max(1, all_image_grids.shape[0] // max(1, bsz))
                expected_tokens = bsz * num_views * h_ds * w_ds

                if feat_for_depth.shape[0] == expected_tokens:
                    feat_for_depth = feat_for_depth.view(bsz * num_views, h_ds, w_ds, -1).permute(0, 3, 1, 2)
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

    def _hipad_plan_internal_coords(self, value):
        """Convert LEAD (forward, lateral) plans to Sparse4D (lateral, forward)."""
        if value is None or self.planning_query_layout != "hipad_plan_query":
            return value
        if not torch.is_tensor(value) or value.shape[-1] != 2:
            raise ValueError(
                "HiP-AD planning coordinates must be a tensor with last dimension 2"
            )
        return value[..., [1, 0]]

    def _build_driving_batch(
        self,
        img: torch.Tensor,
        command=None,
        ego_status=None,
        target_point_previous=None,
        target_point=None,
        target_point_next=None,
    ) -> DrivingBatch:
        device = img.device if img is not None else torch.device("cuda")
        b = int(img.shape[0]) if torch.is_tensor(img) else 1

        num_cams = int(img.shape[1])
        permute_indices = self._camera_permute_indices(num_cams)
        images = img[:, permute_indices]

        image_masks = {f"cam{i}": torch.ones((b,), device=device, dtype=torch.bool) for i in range(len(permute_indices))}
        view_token_ids = self._get_view_token_ids(device)[:len(permute_indices)]

        if command is not None:
            if not torch.is_tensor(command):
                try:
                    command = torch.stack(command)
                except:
                    command = torch.tensor(command)
            command = command.to(device)
            if command.dim() > 1:
                cmd_idx = command.argmax(dim=-1).long().view(-1)
            else:
                cmd_idx = command.view(-1).long()
            if cmd_idx.numel() < b:
                cmd_idx = F.pad(cmd_idx, (0, b - cmd_idx.numel()), value=2)
            elif cmd_idx.numel() > b:
                cmd_idx = cmd_idx[:b]
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

        ego_status_np = None
        if self.strict_user_prompt_fields is None:
            ego_status_np = (
                ego_status.detach().cpu().numpy()
                if ego_status is not None
                else np.zeros((b, 6))
            )

        target_points = None
        raw_target_points = [target_point_previous, target_point, target_point_next]
        if all(value is not None for value in raw_target_points):
            prepared_target_points = []
            for value in raw_target_points:
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                value = value.to(device=device, dtype=torch.float32)
                while value.dim() > 2 and value.shape[1] == 1:
                    value = value.squeeze(1)
                prepared_target_points.append(value.reshape(b, -1)[:, :2])
            target_points = torch.stack(prepared_target_points, dim=1)

        if not hasattr(self.qwen3_vl_with_expert, "processor") or self.qwen3_vl_with_expert.processor is None:
            raise RuntimeError("QwenVLAPlanningHead expects `self.qwen3_vl_with_expert.processor`")

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        system_ids = tokenizer.encode("system", add_special_tokens=False)
        user_ids = tokenizer.encode("user", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        sys_content_ids = tokenizer.encode(self.system_prompt, add_special_tokens=False)
        sys_part = [im_start_id] + system_ids + [nl_id] + sys_content_ids + [im_end_id, nl_id]

        user_start_part = [im_start_id] + user_ids + [nl_id]
        user_end_assistant_start_part = [im_end_id, nl_id, im_start_id] + assistant_ids + [nl_id]

        input_ids_list = []
        attention_mask_list = []

        for i in range(b):
            user_prompt_text = _render_user_prompt(
                self.user_prompt_template,
                nav_cmd_texts[i],
                None if ego_status_np is None else ego_status_np[i],
                self.strict_user_prompt_fields,
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
            target_points=target_points,
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
        num_views_per_sample = int(images_tensor.shape[1])
        view_token_ids = self._get_view_token_ids(device)[:num_views_per_sample]

        prefix_input_ids_list = []

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
        if self.prefix_max_len is not None:
            if curr_len > self.prefix_max_len:
                raise ValueError(
                    "Prefix length exceeds configured prefix_max_len; refusing to "
                    f"truncate image or prompt tokens: {curr_len} > {self.prefix_max_len}"
                )
            if curr_len < self.prefix_max_len:
                pad_len = self.prefix_max_len - curr_len
                prefix_input_ids = F.pad(
                    prefix_input_ids,
                    (0, pad_len),
                    value=tokenizer.pad_token_id,
                )

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

    def _build_status_tokens(
        self,
        batch: DrivingBatch,
        *,
        dtype: torch.dtype,
        ego_status_pred: Optional[torch.Tensor],
        use_gt: bool,
    ) -> torch.Tensor:
        if self.planning_status_layout == "legacy_fused":
            status_input = self._maybe_get_status_features(
                batch,
                ego_status_pred,
                use_gt=use_gt,
                dtype=dtype,
            )
            status_emb = self._apply_checkpoint(self.status_mlp, status_input)
            return status_emb.unsqueeze(1)

        if batch.command is None:
            raise ValueError("TFv6 status tokens require a navigation command")
        command = batch.command
        if command.dim() == 1 or (command.dim() == 2 and command.shape[-1] == 1):
            command_indices = command.reshape(-1).long()
            command_onehot = F.one_hot(command_indices, num_classes=6)
        elif command.dim() == 2 and command.shape[-1] == 6:
            command_onehot = command
        else:
            raise ValueError(
                f"Expected command indices or [B, 6] one-hot commands, got {tuple(command.shape)}"
            )

        if batch.ego_status is None:
            raise ValueError("TFv6 status tokens require ego_status")
        if batch.target_points is None or batch.target_points.shape[1:] != (3, 2):
            raise ValueError(
                "TFv6 status tokens require [previous, current, next] target points"
            )

        encoder_dtype = self.velocity_encoder.weight.dtype
        device = self.velocity_encoder.weight.device
        velocity = batch.ego_status[:, :1].to(
            device=device,
            dtype=encoder_dtype,
        )
        command_onehot = command_onehot.to(device=device, dtype=encoder_dtype)
        target_points = batch.target_points.to(device=device, dtype=encoder_dtype)
        normalization = self.target_points_normalization_constants.to(
            device=device,
            dtype=encoder_dtype,
        )
        target_point_tokens = self.tp_encoder(target_points / normalization)

        status_tokens = torch.cat(
            [
                self.velocity_encoder(velocity / self.velocity_normalization).unsqueeze(1),
                self.command_encoder(command_onehot).unsqueeze(1),
                target_point_tokens[:, 1:2],
                target_point_tokens[:, 0:1],
                target_point_tokens[:, 2:3],
            ],
            dim=1,
        )
        return status_tokens.to(dtype=dtype) + self.status_position_embedding.to(
            device=device,
            dtype=dtype,
        )

    def _embed_stable_planning_tokens(
        self,
        batch: DrivingBatch,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        ego_status_pred: Optional[torch.Tensor] = None,
        use_gt_status: bool = False,
    ) -> torch.Tensor:
        status_emb = self._build_status_tokens(
            batch,
            dtype=dtype,
            ego_status_pred=ego_status_pred,
            use_gt=use_gt_status,
        )
        stable_parts = [status_emb]
        if self.planning_query_layout == "action_expert":
            stable_parts.append(
                self.route_query.expand(batch_size, -1, -1).to(
                    device=device, dtype=dtype
                )
            )
            if self.deterministic_waypoint_query is not None:
                stable_parts.append(
                    self.deterministic_waypoint_query.expand(
                        batch_size, -1, -1
                    ).to(device=device, dtype=dtype)
                )
            stable_parts.append(
                self.target_speed_query.expand(batch_size, -1, -1).to(
                    device=device, dtype=dtype
                )
            )
        elif self.planning_query_layout == "action_expert_hipad":
            stable_parts.append(
                self.deterministic_waypoint_query[:, : self.temporal_waypoint_queries].expand(
                    batch_size, -1, -1
                ).to(device=device, dtype=dtype)
            )
            stable_parts.append(
                self.spatial_waypoint_query.expand(
                    batch_size, -1, -1
                ).to(device=device, dtype=dtype)
            )
        return torch.cat(stable_parts, dim=1)

    def _embed_flow_tokens(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = actions.device
        if dtype is None:
            dtype = (
                self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0]
                .self_attn.q_proj.weight.dtype
            )
        if actions.dtype != dtype:
            actions = actions.to(dtype=dtype)
        action_emb = self._apply_checkpoint(self.action_in_proj, actions)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            self.min_period,
            self.max_period,
            device=device,
        )
        fused_input = torch.cat(
            [action_emb, time_emb.unsqueeze(1).expand_as(action_emb)],
            dim=-1,
        ).to(dtype)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        return self._apply_checkpoint(mlp_func, fused_input)

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

        stable_emb = self._embed_stable_planning_tokens(
            batch,
            batch_size=B,
            device=device,
            dtype=dtype,
            ego_status_pred=ego_status_pred,
            use_gt_status=use_gt_status,
        )
        flow_emb = self._embed_flow_tokens(actions, timestep, dtype=dtype)
        suffix_emb = torch.cat([stable_emb, flow_emb], dim=1)

        suffix_len = suffix_emb.shape[1]
        if (
            self.planning_token_layout is not None
            and suffix_len != self.planning_token_layout.total_tokens
        ):
            raise RuntimeError(
                f"Expected {self.planning_token_layout.total_tokens} suffix tokens, "
                f"got {suffix_len}"
            )
        pad_masks = torch.ones((B, suffix_len), dtype=torch.bool, device=device)

        # Stable planning reads status, and flow reads stable planning. Stable
        # route/waypoint/speed tokens must never read noisy flow tokens.
        att_marks = torch.zeros(suffix_len, dtype=torch.bool, device=device)
        if self.planning_query_layout == "action_expert":
            layout = self.planning_token_layout
            att_marks[layout.status.start] = True
            att_marks[layout.route.start] = True
            att_marks[layout.flow.start] = True
        elif self.planning_query_layout == "action_expert_hipad":
            layout = self.planning_token_layout
            att_marks[layout.status.start] = True
            att_marks[layout.temporal.start] = True
            att_marks[layout.spatial.start] = True
            att_marks[layout.flow.start] = True
        elif self.planning_query_layout == "hipad_plan_query":
            layout = self.planning_token_layout
            att_marks[layout.status.start] = True
            att_marks[layout.flow.start] = True
        else:
            att_marks[0] = True
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

        num_cams = int(img.shape[1]) if torch.is_tensor(img) and img.dim() >= 2 else 6
        permute_indices = self._camera_permute_indices(num_cams)
        projection_mat = self._permute_camera_tensor(projection_mat, permute_indices)
        image_wh = self._permute_camera_tensor(image_wh, permute_indices)

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = self._permute_metas_per_camera_fields(kwargs.get("img_metas"), permute_indices)

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            target_point_previous=kwargs.get('target_point_previous'),
            target_point=kwargs.get('target_point'),
            target_point_next=kwargs.get('target_point_next'),
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
            'gt_ego_fut_cmd': batch.command,
            'target_point': self._hipad_plan_internal_coords(
                batch.target_points[:, 1]
                if batch.target_points is not None
                else None
            ),
        }

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)
        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz,
            actions.device,
            prefix_embs.dtype,
            stage1_outs,
            target_points=batch.target_points,
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
        pad_masks = torch.cat(
            [prefix_pad_masks, perception_pad_masks, suffix_pad_masks],
            dim=1,
        )
        att_masks = torch.cat(
            [prefix_att_masks, perception_att_masks, suffix_att_masks],
            dim=1,
        )

        if self.attn_implementation == "flex" and not self.enable_knowledge_insulation:
            block_mask, q_len_rounded = build_blockmask_unidrive(
                bsz=bsz,
                hq=self.qwen3_vl_with_expert.qwen3_vl.config.text_config.num_attention_heads,
                prefix_len=prefix_len,
                perception_len=perception_len,
                suffix_len=suffix_len,
                stable_start=(
                    self.planning_token_layout.route.start
                    if self.planning_token_layout is not None
                    else None
                ),
                flow_start=(
                    self.planning_token_layout.flow.start
                    if self.planning_token_layout is not None
                    else None
                ),
                device=actions.device,
                pad_masks=pad_masks,
                compile_blockmask=True,
            )
            att_mask_input = block_mask
        else:
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
                'gt_ego_fut_cmd': gt_ego_fut_cmd,
                'gt_ego_fut_trajs_5hz': self._hipad_plan_internal_coords(
                    kwargs.get('gt_ego_fut_trajs_5hz')
                ),
                'gt_ego_fut_masks_5hz': kwargs.get('gt_ego_fut_masks_5hz'),
                'gt_ego_fut_trajs_2hz': self._hipad_plan_internal_coords(
                    kwargs.get('gt_ego_fut_trajs_2hz')
                ),
                'gt_ego_fut_masks_2hz': kwargs.get('gt_ego_fut_masks_2hz'),
                'gt_ego_spat_trajs_2m': self._hipad_plan_internal_coords(
                    kwargs.get('gt_ego_spat_trajs_2m')
                ),
                'gt_ego_spat_masks_2m': kwargs.get('gt_ego_spat_masks_2m'),
                'gt_ego_spat_trajs_5m': self._hipad_plan_internal_coords(
                    kwargs.get('gt_ego_spat_trajs_5m')
                ),
                'gt_ego_spat_masks_5m': kwargs.get('gt_ego_spat_masks_5m'),
            }
            perception_losses = self.unified_decoder.loss(stage1_outs, stage2_outs, perception_data)
            stats.update(perception_losses)

            if self.planning_query_layout == "perception_shared":
                aux_context = (
                    ego_out.mean(dim=1)
                    if ego_out is not None
                    else perception_out.mean(dim=1)
                )
                pred_route, speed_logits = self._decode_perception_planning(
                    aux_context
                )
                stats.update(
                    self._compute_aux_planning_losses(
                        pred_route,
                        None,
                        speed_logits,
                        kwargs,
                        gt_ego_fut_trajs=gt_ego_fut_trajs,
                        ego_status=ego_status,
                    )
                )

        if self.planning_query_layout == "action_expert":
            pred_route, pred_waypoints, speed_logits, waypoint_suffix_out = (
                self._decode_action_expert_planning(suffix_out)
            )
            stats.update(
                self._compute_aux_planning_losses(
                    pred_route,
                    pred_waypoints,
                    speed_logits,
                    kwargs,
                    gt_ego_fut_trajs=gt_ego_fut_trajs,
                    ego_status=ego_status,
                )
            )
        elif self.planning_query_layout == "action_expert_hipad":
            pred_temporal, pred_spatial, waypoint_suffix_out = (
                self._decode_action_expert_hipad_planning(suffix_out)
            )
            stats.update(
                self._compute_hipad_planning_losses(
                    pred_temporal,
                    pred_spatial,
                    gt_ego_fut_trajs,
                    gt_ego_fut_masks,
                    kwargs,
                )
            )
        elif self.planning_query_layout == "hipad_plan_query":
            waypoint_suffix_out = suffix_out[:, self.planning_token_layout.flow]
        else:
            waypoint_suffix_out = suffix_out[:, -self.action_horizon:]

        suffix_out_float = waypoint_suffix_out.to(dtype=torch.float32)
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
            stats['metric_vlm_raw'] = ar_outputs['loss_vlm_raw']
            stats['ar_planning_ratio'] = (ar_outputs['loss_vlm_raw'] / (planning_loss.detach() + 1e-6)).item()
        else:
            stats['metric_vlm_raw'] = torch.tensor(0.0, device=planning_loss.device)
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

        wasp_step_scale = (
            self._wasp_regularization_step_scale
            if self._compute_wasp_regularization_this_step
            else 0.0
        )
        wasp_zero = planning_loss.detach().new_zeros(())

        if self._wasp_loss_weight > 0:
            if wasp_step_scale > 0:
                from .wasp import compute_wasp_regularization
                language_wasp_raw = compute_wasp_regularization(
                    self.qwen3_vl_with_expert.qwen3_vl.model.language_model
                )
                if language_wasp_raw is None:
                    raise RuntimeError(
                        "LLM peft_cfg uses WASP but no WASPLinear layers exist."
                    )
                stats["loss_wasp"] = (
                    wasp_step_scale * self._wasp_loss_weight * language_wasp_raw
                )
                stats["wasp_raw"] = wasp_step_scale * language_wasp_raw.detach()
            else:
                stats["loss_wasp"] = wasp_zero
                stats["wasp_raw"] = wasp_zero

        if self._visual_wasp_loss_weight > 0:
            if wasp_step_scale > 0:
                visual_wasp_raw = (
                    self.qwen3_vl_with_expert.visual_wasp_regularization()
                )
                if visual_wasp_raw is None:
                    raise RuntimeError(
                        "visual_wasp_cfg is enabled but no encoder WASPLinear "
                        "layers exist."
                    )
                stats["loss_visual_wasp"] = (
                    wasp_step_scale
                    * self._visual_wasp_loss_weight
                    * visual_wasp_raw
                )
                stats["visual_wasp_raw"] = (
                    wasp_step_scale * visual_wasp_raw.detach()
                )
            else:
                stats["loss_visual_wasp"] = wasp_zero
                stats["visual_wasp_raw"] = wasp_zero

        if self._visual_projector_wasp_loss_weight > 0:
            if wasp_step_scale > 0:
                projector_wasp_raw = (
                    self.qwen3_vl_with_expert
                    .visual_projector_wasp_regularization()
                )
                if projector_wasp_raw is None:
                    raise RuntimeError(
                        "visual_projector_wasp_cfg is enabled but no projector "
                        "WASPLinear layers exist."
                    )
                stats["loss_projector_wasp"] = (
                    wasp_step_scale
                    * self._visual_projector_wasp_loss_weight
                    * projector_wasp_raw
                )
                stats["projector_wasp_raw"] = (
                    wasp_step_scale * projector_wasp_raw.detach()
                )
            else:
                stats["loss_projector_wasp"] = wasp_zero
                stats["projector_wasp_raw"] = wasp_zero

        fixed_keys = ["loss_planning", "loss_ego_status", "loss_motion"]
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
        num_cams = int(img.shape[1]) if torch.is_tensor(img) and img.dim() >= 2 else 6
        permute_indices = self._camera_permute_indices(num_cams)
        projection_mat = self._permute_camera_tensor(projection_mat, permute_indices)
        image_wh = self._permute_camera_tensor(image_wh, permute_indices)

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = self._permute_metas_per_camera_fields(kwargs.get("img_metas"), permute_indices)

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            target_point_previous=kwargs.get('target_point_previous'),
            target_point=kwargs.get('target_point'),
            target_point_next=kwargs.get('target_point_next'),
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
            'gt_ego_fut_cmd': batch.command,
            'target_point': self._hipad_plan_internal_coords(
                batch.target_points[:, 1]
                if batch.target_points is not None
                else None
            ),
        }

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)
        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz,
            device,
            dtype,
            stage1_outs,
            target_points=batch.target_points,
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

        aux_route_pred = None
        aux_waypoint_pred = None
        aux_target_speed_logits = None
        aux_target_speed_distribution = None
        aux_spatial_waypoint_pred = None
        aux_target_speed_scalar = None

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
            decoded_plan = self.unified_decoder.decode_plan(
                stage2_outs, {'gt_ego_fut_cmd': batch.command}
            )
            if decoded_plan:
                plan_speed_5hz = torch.stack(
                    [item['plan_speed_5hz'] for item in decoded_plan], dim=0
                ).to(device=device)
                plan_spat_2m = torch.stack(
                    [item['plan_spat_2m'] for item in decoded_plan], dim=0
                ).to(device=device)
                plan_speed_5hz = self._hipad_plan_internal_coords(plan_speed_5hz)
                plan_spat_2m = self._hipad_plan_internal_coords(plan_spat_2m)
            else:
                plan_speed_5hz = None
                plan_spat_2m = None

            if self.planning_query_layout == "perception_shared":
                aux_context = (
                    ego_out.mean(dim=1)
                    if ego_out is not None
                    else perception_out.mean(dim=1)
                )
                aux_route_pred, aux_target_speed_logits = (
                    self._decode_perception_planning(aux_context)
                )
                (
                    aux_target_speed_distribution,
                    aux_target_speed_scalar,
                ) = self._decode_target_speed(aux_target_speed_logits)
        else:
            ego_out = None
            det_result = None
            map_result = None
            stage2_outs = None
            plan_speed_5hz = None
            plan_spat_2m = None

        cached_pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks], dim=1)
        dt_val = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)

        x_t = noise
        time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)

        if stage2_outs and 'ego_status_list' in stage2_outs and len(stage2_outs['ego_status_list']) > 0:
            ego_status_pred = stage2_outs['ego_status_list'][-1].squeeze(1).to(torch.float32)
        else:
            ego_status_pred = x_t.new_zeros((bsz, self.ego_status_dim), dtype=torch.float32)

        max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values

        if self.planning_query_layout in {
            "action_expert",
            "action_expert_hipad",
            "hipad_plan_query",
        }:
            (
                aux_route_pred,
                aux_waypoint_pred,
                aux_spatial_waypoint_pred,
                aux_target_speed_logits,
                aux_target_speed_distribution,
                aux_target_speed_scalar,
                stable_past_key_values,
                stable_pad_masks,
                max_stable_pos,
            ) = self._infer_stable_planning(
                batch,
                cached_pad_masks,
                past_key_values,
                max_perception_pos,
                ego_status_pred=ego_status_pred,
            )
            while time_tensor >= -dt_val / 2:
                expanded_time = time_tensor.expand(bsz)
                v_t = self._denoise_flow_step(
                    stable_pad_masks,
                    stable_past_key_values,
                    x_t.to(dtype),
                    expanded_time.to(dtype),
                    max_stable_pos,
                )
                x_t = x_t + dt_val * v_t
                time_tensor += dt_val
        else:
            while time_tensor >= -dt_val / 2:
                expanded_time = time_tensor.expand(bsz)
                (
                    v_t,
                    step_route,
                    step_waypoints,
                    step_speed_logits,
                    step_speed_distribution,
                    step_speed_scalar,
                ) = self._denoise_step(
                    batch,
                    cached_pad_masks,
                    past_key_values,
                    x_t.to(dtype),
                    expanded_time.to(dtype),
                    max_perception_pos,
                    ego_status_pred=ego_status_pred,
                )
                if step_route is not None:
                    aux_route_pred = step_route
                    aux_waypoint_pred = step_waypoints
                    aux_target_speed_logits = step_speed_logits
                    aux_target_speed_distribution = step_speed_distribution
                    aux_target_speed_scalar = step_speed_scalar
                x_t = x_t + dt_val * v_t
                time_tensor += dt_val

        traj_pred_meter_deltas = self.denorm_delta(x_t)
        zeros = torch.zeros((bsz, 1, 2), device=device, dtype=x_t.dtype)
        traj_pred_points_meter = zeros + torch.cumsum(traj_pred_meter_deltas, dim=1)

        return {
            "traj": traj_pred_points_meter,
            "det": det_result,
            "map": map_result,
            "pred_route": aux_route_pred,
            "pred_waypoints": aux_waypoint_pred,
            "plan_speed_5hz": (
                plan_speed_5hz
                if self.planning_query_layout == "hipad_plan_query"
                else (
                    aux_waypoint_pred
                    if self.planning_query_layout == "action_expert_hipad"
                    else None
                )
            ),
            "plan_spat_2m": (
                plan_spat_2m
                if self.planning_query_layout == "hipad_plan_query"
                else aux_spatial_waypoint_pred
            ),
            "pred_target_speed_logits": aux_target_speed_logits,
            "pred_target_speed_distribution": aux_target_speed_distribution,
            "pred_target_speed_scalar": aux_target_speed_scalar,
        }

    def _infer_stable_planning(
        self,
        batch,
        cached_pad_masks,
        past_key_values,
        max_cached_position_ids,
        *,
        ego_status_pred: Optional[torch.Tensor] = None,
    ):
        batch_size = cached_pad_masks.shape[0]
        device = cached_pad_masks.device
        dtype = (
            self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )
        stable_embs = self._embed_stable_planning_tokens(
            batch,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            ego_status_pred=ego_status_pred,
            use_gt_status=False,
        )

        stable_len = self.planning_token_layout.flow.start
        if stable_embs.shape[1] != stable_len:
            raise RuntimeError(
                f"Expected {stable_len} stable planning tokens, "
                f"got {stable_embs.shape[1]}"
            )
        stable_pad_masks = torch.ones(
            (batch_size, stable_len), dtype=torch.bool, device=device
        )
        stable_att_marks = torch.zeros(
            stable_len, dtype=torch.bool, device=device
        )
        stable_att_marks[self.planning_token_layout.status.start] = True
        if self.planning_query_layout == "action_expert":
            stable_att_marks[self.planning_token_layout.route.start] = True
        elif self.planning_query_layout == "action_expert_hipad":
            stable_att_marks[self.planning_token_layout.temporal.start] = True
            stable_att_marks[self.planning_token_layout.spatial.start] = True
        stable_att_masks = stable_att_marks[None, :].expand(batch_size, -1)
        cached_len = cached_pad_masks.shape[1]

        cached_pad_2d = cached_pad_masks[:, None, :].expand(
            batch_size, stable_len, cached_len
        )
        stable_att_2d = make_att_2d_masks(stable_pad_masks, stable_att_masks)
        full_att_2d = torch.cat([cached_pad_2d, stable_att_2d], dim=2)
        full_att_4d = self._prepare_attention_masks_4d(full_att_2d)

        stable_range = torch.arange(
            1, stable_len + 1, device=device
        ).view(1, -1).expand(batch_size, -1)
        stable_pos_ids_1d = max_cached_position_ids + stable_range
        stable_pos_ids = torch.stack([stable_pos_ids_1d] * 3, dim=0)

        self.qwen3_vl_with_expert.qwen3_action_expert.config._attn_implementation = "eager"
        outputs_embeds, stable_past_key_values, _ = (
            self.qwen3_vl_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=stable_pos_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, None, stable_embs],
                use_cache=True,
            )
        )
        if stable_past_key_values is None:
            raise RuntimeError("Action expert did not return the stable planning KV cache")

        stable_out = outputs_embeds[2]
        pred_spatial = None
        if self.planning_query_layout == "action_expert_hipad":
            pred_waypoints, pred_spatial = (
                self._decode_action_expert_hipad_stable(stable_out)
            )
            pred_route = None
            speed_logits = None
            speed_distribution = None
            speed_scalar = None
        elif self.planning_query_layout == "action_expert":
            pred_route, pred_waypoints, speed_logits = (
                self._decode_action_expert_stable_planning(stable_out)
            )
            speed_distribution, speed_scalar = self._decode_target_speed(
                speed_logits
            )
        else:
            pred_route = None
            pred_waypoints = None
            speed_logits = None
            speed_distribution = None
            speed_scalar = None
        full_stable_pad_masks = torch.cat(
            [cached_pad_masks, stable_pad_masks], dim=1
        )
        max_stable_pos = stable_pos_ids_1d.max(dim=-1, keepdim=True).values
        return (
            pred_route,
            pred_waypoints,
            pred_spatial,
            speed_logits,
            speed_distribution,
            speed_scalar,
            stable_past_key_values,
            full_stable_pad_masks,
            max_stable_pos,
        )

    def _denoise_flow_step(
        self,
        cached_pad_masks,
        past_key_values,
        x_t,
        timestep,
        max_cached_position_ids,
    ):
        flow_embs = self._embed_flow_tokens(
            x_t,
            timestep,
        )
        batch_size, flow_len = flow_embs.shape[:2]
        if flow_len != self.action_horizon:
            raise RuntimeError(
                f"Expected {self.action_horizon} flow tokens, got {flow_len}"
            )
        cached_len = cached_pad_masks.shape[1]
        full_att_2d = torch.ones(
            (batch_size, flow_len, cached_len + flow_len),
            dtype=torch.bool,
            device=flow_embs.device,
        )
        full_att_4d = self._prepare_attention_masks_4d(full_att_2d)

        flow_range = torch.arange(
            1, flow_len + 1, device=flow_embs.device
        ).view(1, -1).expand(batch_size, -1)
        flow_pos_ids_1d = max_cached_position_ids + flow_range
        flow_pos_ids = torch.stack([flow_pos_ids_1d] * 3, dim=0)

        self.qwen3_vl_with_expert.qwen3_action_expert.config._attn_implementation = "eager"
        outputs_embeds, _, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_4d,
            position_ids=flow_pos_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, flow_embs],
            use_cache=False,
        )
        return self.action_out_proj(outputs_embeds[2].to(dtype=torch.float32))

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
        if self.planning_query_layout == "action_expert":
            pred_route, pred_waypoints, speed_logits, waypoint_suffix_out = (
                self._decode_action_expert_planning(suffix_out)
            )
            speed_distribution, speed_scalar = self._decode_target_speed(
                speed_logits
            )
        else:
            pred_route = None
            pred_waypoints = None
            speed_logits = None
            speed_distribution = None
            speed_scalar = None
            waypoint_suffix_out = suffix_out[:, -self.action_horizon:]

        model_out = self.action_out_proj(
            waypoint_suffix_out.to(dtype=torch.float32)
        )
        return model_out, pred_route, pred_waypoints, speed_logits, speed_distribution, speed_scalar

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
