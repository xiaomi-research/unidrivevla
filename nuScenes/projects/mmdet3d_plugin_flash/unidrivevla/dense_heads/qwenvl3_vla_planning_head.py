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
import copy
from dataclasses import dataclass
from typing import Dict, Literal, Optional, List

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
import logging
import deepspeed
from mmdet.models import HEADS
from mmdet.models.builder import build_head, build_loss
from timm.models.layers import Mlp
from einops import rearrange
from .unidrivevla_vlm_qwenvl3 import Qwen3VLWithExpertModel
from torch.nn.utils.rnn import pad_sequence
from .flex_attention_opt import build_blockmask_unidrive
from .constants import (
    NUSCENES_SYSTEM_PROMPT,
    NUSCENES_SYSTEM_PROMPT_FULL,
    NUSCENES_USER_PROMPT_TEMPLATE,
    NUSCENES_VIEW_TOKENS,
    TARGET_SENSOR_ORDER,
    OPENPI_ATTENTION_MASK_VALUE,
    DEFAULT_PERM_INDICES,
    _NAV_CMD_FIXED,
)
from .utils import (
    make_att_2d_masks,
    sample_beta,
    create_sinusoidal_pos_embedding,
    permute_metas_per_camera_fields,
)
from .modules import OccLatentDecoder, DenseDepthNet


from projects.mmdet3d_plugin_flash.losses.collision_loss import CollisionLoss
from projects.mmdet3d_plugin_flash.losses.plan_map_loss import GTMapBoundLoss, GTMapDirectionLoss
from projects.mmdet3d_plugin_flash.models.detection3d.target import SparseBox3DTarget
from projects.mmdet3d_plugin_flash.models.detection3d.losses import SparseBox3DLoss
from projects.mmdet3d_plugin_flash.models.detection3d.detection3d_blocks import SparseBox3DEncoder
from projects.mmdet3d_plugin_flash.models.map.target import SparsePoint3DTarget
from projects.mmdet3d_plugin_flash.models.map.loss import SparseLineLoss
from projects.mmdet3d_plugin_flash.models.map.map_blocks import SparsePoint3DEncoder
from projects.mmdet3d_plugin_flash.models.map.decoder import SparsePoint3DDecoder
from projects.mmdet3d_plugin_flash.ops import feature_maps_format
from projects.mmdet3d_plugin_flash.core.box3d import *

from .unified_perception_decoder import UnifiedPerceptionDecoder

@dataclass
class DrivingBatch:
    images: torch.Tensor
    image_masks: Dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    command: Optional[torch.Tensor]
    ego_status: Optional[torch.Tensor]
    view_token_ids: Optional[torch.Tensor] = None
    traj_answer_ids: Optional[torch.Tensor] = None
    traj_answer_mask: Optional[torch.Tensor] = None
    traj_labels: Optional[torch.Tensor] = None
    prompt_lens: Optional[List[int]] = None

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
        vlm_variant: Literal["2b", "8b"] = "2b",
        action_dim: int = 2,
        action_horizon: int = 6,
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        flow_mode: Literal["flow_matching", "meanflow"] = "flow_matching",
        meanflow_ratio_r_not_equal_t: float = 0.25,
        meanflow_min_dt: float = 1e-3,
        meanflow_time_sampling: Literal["logit_normal", "uniform", "beta"] = "logit_normal",
        meanflow_time_mu: float = -0.4,
        meanflow_time_sigma: float = 1.0,
        meanflow_jvp_create_graph: bool = True,
        enable_meanflow_arm: bool = False,
        meanflow_arm_loss_weight: float = 0.0,
        meanflow_arm_num_layers: int = 3,
        meanflow_arm_num_heads: int = 8,
        meanflow_arm_detach_context: bool = True,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        num_sample_steps: int = 10,
        enable_knowledge_insulation: bool = False,
        ar_loss_weight: float = 0.1,
        train_vlm: bool = False,
        enable_traj_ar: bool = False,
        traj_ar_loss_weight: float = 0.1,
        traj_ar_target: Literal["waypoint", "raw_delta", "norm_delta"] = "waypoint",
        loss_planning: Optional[dict] = None,
        occ_loss_weight: float = 1.0,
        collision_loss_weight: float = 0.0,
        map_bound_loss_weight: float = 0.0,
        map_dir_loss_weight: float = 0.0,
        map_bound_dis_thresh: float = 1.0,
        map_dir_dis_thresh: float = 2.0,
        x_min: float = -13.97,
        x_max: float = 11.77,
        y_min: float = -2.02,
        y_max: float = 55.79,
        occ_aux_loss_weight: float = 1.0,
        occ_aux_layers_1based: Optional[List[int]] = None,
        attn_implementation: Literal["eager", "sdpa", "flex"] = "flex",
        inference_attn_impl: Literal["eager", "sdpa"] = "eager",
        unified_decoder_cfg: dict = None,
        occworld_vae_config: Optional[dict] = None,
        occworld_vae_path: Optional[str] = None,
        with_depth_supervision: bool = False,
        depth_loss_weight: float = 0.2,
        num_depth_bins: int = 80,
        depth_range: tuple = (1.0, 60.0),
        depth_supervision_source: Literal["input", "output"] = "input",
        feature_source: Literal["raw", "deepstack"] = "deepstack",
        feat_grad: Optional[bool] = None,
        use_tau0_pred: bool = False,
        vlm_fusion_cfg: Optional[dict] = None,
        feature_fusion_cfg: Optional[dict] = None,
        vlm_grad_scale: float = 1.0,
        lora_cfg: Optional[dict] = None,
        lora_merge_save_dir: Optional[str] = None,
        image_token_compression_cfg: Optional[dict] = None,
        perception_adanorm_cfg: Optional[dict] = None,
        system_prompt_variant: Literal["short", "full"] = "short",
        flow_distill_cfg: Optional[dict] = None,
        action_training_mode: Literal["flow", "query_supervised"] = "flow",
        action_query_cfg: Optional[dict] = None,
        driving_deepstack: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        if flow_mode not in ("flow_matching", "meanflow"):
            raise ValueError(f"flow_mode must be 'flow_matching' or 'meanflow', got {flow_mode}")
        self.flow_mode = flow_mode
        self.meanflow_ratio_r_not_equal_t = float(meanflow_ratio_r_not_equal_t)
        self.meanflow_min_dt = float(meanflow_min_dt)
        if meanflow_time_sampling not in ("logit_normal", "uniform", "beta"):
            raise ValueError(
                "meanflow_time_sampling must be one of 'logit_normal', 'uniform', or 'beta', "
                f"got {meanflow_time_sampling}"
            )
        self.meanflow_time_sampling = meanflow_time_sampling
        self.meanflow_time_mu = float(meanflow_time_mu)
        self.meanflow_time_sigma = float(meanflow_time_sigma)
        self.meanflow_jvp_create_graph = bool(meanflow_jvp_create_graph)
        self.enable_meanflow_arm = bool(enable_meanflow_arm)
        self.meanflow_arm_loss_weight = float(meanflow_arm_loss_weight)
        self.meanflow_arm_num_layers = int(meanflow_arm_num_layers)
        self.meanflow_arm_num_heads = int(meanflow_arm_num_heads)
        self.meanflow_arm_detach_context = bool(meanflow_arm_detach_context)
        self.min_period = min_period
        self.max_period = max_period
        self.num_sample_steps = num_sample_steps
        if action_training_mode not in ("flow", "query_supervised"):
            raise ValueError(
                "action_training_mode must be one of 'flow' or 'query_supervised', "
                f"got {action_training_mode}"
            )
        self.action_training_mode = action_training_mode
        action_query_cfg = copy.deepcopy(action_query_cfg) or {}
        self.action_query_num_queries = int(action_query_cfg.get("num_queries", 8))
        self.action_query_l1_loss_weight = float(action_query_cfg.get("l1_loss_weight", 1.0))
        self.action_query_l2_loss_weight = float(action_query_cfg.get("l2_loss_weight", 1.0))
        if self.action_query_num_queries <= 0:
            raise ValueError("action_query_cfg.num_queries must be positive")
        if system_prompt_variant not in ("short", "full"):
            raise ValueError(f"system_prompt_variant must be 'short' or 'full', got {system_prompt_variant}")
        self.system_prompt_variant = system_prompt_variant
        flow_distill_cfg = copy.deepcopy(flow_distill_cfg) or {}
        self.flow_distill_enabled = bool(flow_distill_cfg.get("enabled", False))
        self.flow_distill_teacher_steps = int(flow_distill_cfg.get("teacher_num_steps", num_sample_steps))
        self.flow_distill_loss_weight = float(flow_distill_cfg.get("loss_weight", 1.0))
        self.flow_distill_gt_loss_weight = float(flow_distill_cfg.get("gt_loss_weight", 0.0))
        self.flow_distill_fm_loss_weight = float(flow_distill_cfg.get("fm_loss_weight", 0.0))
        self.flow_distill_waypoint_loss_weight = float(flow_distill_cfg.get("waypoint_loss_weight", 0.0))
        self.flow_distill_waypoint_gt_loss_weight = float(flow_distill_cfg.get("waypoint_gt_loss_weight", 0.0))
        self.flow_distill_final_waypoint_loss_weight = float(
            flow_distill_cfg.get("final_waypoint_loss_weight", 0.0)
        )
        self.flow_distill_final_waypoint_gt_loss_weight = float(
            flow_distill_cfg.get("final_waypoint_gt_loss_weight", 0.0)
        )
        self.flow_distill_loss_type = str(flow_distill_cfg.get("loss_type", "mse"))
        self.flow_distill_student_time = float(flow_distill_cfg.get("student_time", 1.0))
        self.flow_distill_freeze_condition_modules = bool(
            flow_distill_cfg.get("freeze_condition_modules", False)
        )
        self.flow_distill_train_action_adapters = bool(flow_distill_cfg.get("train_action_adapters", False))
        self.flow_distill_train_adanorm = bool(flow_distill_cfg.get("train_adanorm", False))
        if self.action_training_mode == "query_supervised" and self.flow_distill_enabled:
            raise ValueError("action_training_mode='query_supervised' does not support flow_distill_cfg.enabled=True")
        if self.flow_distill_enabled:
            if self.flow_mode != "flow_matching":
                raise ValueError("flow_distill_cfg.enabled=True requires flow_mode='flow_matching'")
            if self.flow_distill_teacher_steps <= 0:
                raise ValueError("flow_distill_cfg.teacher_num_steps must be positive")
            if self.flow_distill_loss_type not in ("mse", "l1", "smooth_l1"):
                raise ValueError(
                    "flow_distill_cfg.loss_type must be one of mse/l1/smooth_l1, "
                    f"got {self.flow_distill_loss_type}"
                )

        assert not (enable_knowledge_insulation and not enable_traj_ar), \
            "enable_knowledge_insulation=True requires enable_traj_ar=True"
        self.enable_knowledge_insulation = enable_knowledge_insulation
        self._inference_attn_impl = inference_attn_impl
        self.ar_loss_weight = ar_loss_weight
        self.train_vlm = train_vlm
        self.enable_traj_ar = enable_traj_ar
        self.traj_ar_loss_weight = traj_ar_loss_weight
        assert traj_ar_target in ("waypoint", "raw_delta", "norm_delta"), \
            f"traj_ar_target must be one of waypoint/raw_delta/norm_delta, got {traj_ar_target}"
        self.traj_ar_target = traj_ar_target
        self.occ_loss_weight = occ_loss_weight
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

        self.occ_aux_loss_weight = occ_aux_loss_weight
        if occ_aux_layers_1based is None:
            occ_aux_layers_1based = [4, 14, 24]
        self.occ_aux_layers = [int(x) - 1 for x in occ_aux_layers_1based]

        self.loss_planning = build_loss(loss_planning) if loss_planning is not None else None

        if use_tau0_pred:
            raise ValueError(
                "use_tau0_pred=True is not implemented in QwenVL3APlanningHead; "
                "use the default velocity/MeanFlow velocity target path."
            )
        self.use_tau0_pred = False

        if vlm_variant == "8b":
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_8b_36l')
            perception_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
            action_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
        else:
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_28l')
            perception_expert_cfg = get_qwen_config('qwen3_28l')
            action_expert_cfg = get_qwen_config('qwen3_28l')

        self.lora_merge_save_dir = lora_merge_save_dir

        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
            qwen3_vl_cfg,
            perception_expert_cfg,
            action_expert_cfg,
            pretrained_path,
            precision=dtype,
            train_vlm=train_vlm,
            lora_cfg=lora_cfg,
            image_token_compression_cfg=image_token_compression_cfg,
        )

        self.attn_implementation = attn_implementation
        self.qwen3_vl_with_expert._vla_attn_impl = self.attn_implementation

        if occworld_vae_config is None:
            raise ValueError("occworld_vae_config must be provided via config.")
        if occworld_vae_path is None:
            raise ValueError("occworld_vae_path must be provided via config.")

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg must be provided via config.")

        self.embed_dims = unified_decoder_cfg.get("embed_dims", 256)
        self.vlm_hidden_size = perception_expert_cfg.hidden_size

        self.num_det_queries = unified_decoder_cfg.get("det_instance_bank", {}).get("num_anchor", 900)
        self.num_map_queries = unified_decoder_cfg.get("map_instance_bank", {}).get("num_anchor", 100)
        self.num_occ_queries = 625

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

        self.num_feature_scales = num_proj_layers

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
        self.num_occ_queries = 625
        self.occ_queries = nn.Parameter(torch.randn(1, self.num_occ_queries, self.vlm_hidden_size))

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

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg is required. Legacy det_vla_head/map_vla_head are no longer supported.")

        self.unified_decoder = build_head(unified_decoder_cfg)

        self.occ_decoder = OccLatentDecoder(
            qwen_dim=perception_expert_cfg.hidden_size,
            occworld_vae_config=occworld_vae_config,
            pretrained_vae_path=occworld_vae_path,
        )
        image_compression_enabled = bool(image_token_compression_cfg) and bool(
            image_token_compression_cfg.get("enabled", True)
        ) and image_token_compression_cfg.get("method", "visionzip") != "none"
        if image_compression_enabled and self.with_depth_supervision:
            raise ValueError(
                "with_depth_supervision=True is not supported with VisionZip image token "
                "compression because prefix image tokens are sparse/compressed while the "
                "depth head expects a dense visual grid."
            )
        self.action_in_proj = nn.Linear(action_dim, action_expert_cfg.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_cfg.hidden_size, action_dim)
        self.flow_distill_teacher_action_expert = None
        self.flow_distill_teacher_action_out_proj = None
        self.flow_distill_teacher_action_in_proj = None
        self.flow_distill_teacher_status_mlp = None
        self.flow_distill_teacher_hist_traj_encoder = None
        self.flow_distill_teacher_action_time_mlp_in = None
        self.flow_distill_teacher_action_time_mlp_out = None
        self.flow_distill_teacher_perception_condition_mlp = None
        self.flow_distill_teacher_perception_suffix_norm = None
        self.flow_distill_teacher_perception_suffix_scale_shift = None
        if self.flow_distill_enabled:
            self.flow_distill_teacher_action_expert = copy.deepcopy(
                self.qwen3_vl_with_expert.qwen3_action_expert
            )
            self.flow_distill_teacher_action_out_proj = copy.deepcopy(self.action_out_proj)
            for module in (self.flow_distill_teacher_action_expert, self.flow_distill_teacher_action_out_proj):
                module.requires_grad_(False)
                module.eval()
        self.register_buffer(
            "_flow_distill_teacher_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "_flow_distill_teacher_action_adapters_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "_flow_distill_teacher_adanorm_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        if self.flow_distill_enabled:
            self.register_load_state_dict_pre_hook(
                self._mark_flow_distill_teacher_initialized_from_state_dict
            )

        status_in_features = 3 + self.ego_status_dim

        self.status_mlp = Mlp(
            in_features=status_in_features,
            hidden_features=action_expert_cfg.hidden_size,
            out_features=action_expert_cfg.hidden_size,
            norm_layer=nn.LayerNorm,
        )

        self.hist_traj_steps = 4
        self.hist_traj_dim = 2
        self.hist_traj_encoder = Mlp(
            in_features=self.hist_traj_steps * self.hist_traj_dim,
            hidden_features=action_expert_cfg.hidden_size,
            out_features=action_expert_cfg.hidden_size,
            norm_layer=nn.LayerNorm,
        )

        self.meanflow_time_mlp = None
        if self.flow_mode == "meanflow":
            self.meanflow_time_mlp = nn.Sequential(
                nn.Linear(2 * action_expert_cfg.hidden_size, action_expert_cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(action_expert_cfg.hidden_size, action_expert_cfg.hidden_size),
            )
            nn.init.zeros_(self.meanflow_time_mlp[2].weight)
            nn.init.zeros_(self.meanflow_time_mlp[2].bias)

        self.action_time_mlp_in = nn.Linear(2 * action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)
        self.action_query_tokens = None
        self.action_query_decoder = None
        if self.action_training_mode == "query_supervised":
            self.action_query_tokens = nn.Parameter(
                torch.empty(self.action_query_num_queries, action_expert_cfg.hidden_size)
            )
            nn.init.normal_(self.action_query_tokens, mean=0.0, std=0.02)
            self.action_query_decoder = nn.Sequential(
                nn.LayerNorm(self.action_query_num_queries * action_expert_cfg.hidden_size),
                nn.Linear(
                    self.action_query_num_queries * action_expert_cfg.hidden_size,
                    action_expert_cfg.hidden_size,
                ),
                nn.SiLU(),
                nn.Linear(action_expert_cfg.hidden_size, action_horizon * action_dim),
            )
        if self.flow_distill_enabled and self.flow_distill_train_action_adapters:
            self.flow_distill_teacher_action_in_proj = copy.deepcopy(self.action_in_proj)
            self.flow_distill_teacher_status_mlp = copy.deepcopy(self.status_mlp)
            self.flow_distill_teacher_hist_traj_encoder = copy.deepcopy(self.hist_traj_encoder)
            self.flow_distill_teacher_action_time_mlp_in = copy.deepcopy(self.action_time_mlp_in)
            self.flow_distill_teacher_action_time_mlp_out = copy.deepcopy(self.action_time_mlp_out)
            for module in (
                self.flow_distill_teacher_action_in_proj,
                self.flow_distill_teacher_status_mlp,
                self.flow_distill_teacher_hist_traj_encoder,
                self.flow_distill_teacher_action_time_mlp_in,
                self.flow_distill_teacher_action_time_mlp_out,
            ):
                module.requires_grad_(False)
                module.eval()

        self.meanflow_arm_traj_encoder = None
        self.meanflow_arm_cross_attn = None
        self.meanflow_arm_norm = None
        self.meanflow_arm_recon = None
        if self.enable_meanflow_arm:
            if self.flow_mode != "meanflow":
                raise ValueError("enable_meanflow_arm=True requires flow_mode='meanflow'")
            if self.meanflow_arm_num_heads <= 0:
                raise ValueError("meanflow_arm_num_heads must be positive")
            if action_expert_cfg.hidden_size % self.meanflow_arm_num_heads != 0:
                raise ValueError(
                    "action expert hidden size must be divisible by meanflow_arm_num_heads, "
                    f"got hidden={action_expert_cfg.hidden_size}, heads={self.meanflow_arm_num_heads}"
                )
            arm_dim = action_expert_cfg.hidden_size
            arm_input_dim = self.action_horizon * self.action_dim
            self.meanflow_arm_traj_encoder = nn.Sequential(
                nn.Linear(arm_input_dim, arm_dim, bias=False),
                nn.SiLU(),
                nn.Linear(arm_dim, arm_dim, bias=False),
            )
            self.meanflow_arm_cross_attn = nn.ModuleList([
                nn.ModuleDict({
                    "norm_q": nn.LayerNorm(arm_dim),
                    "norm_mem": nn.LayerNorm(arm_dim),
                    "attn": nn.MultiheadAttention(
                        embed_dim=arm_dim,
                        num_heads=self.meanflow_arm_num_heads,
                        dropout=0.0,
                        batch_first=True,
                    ),
                    "ffn": nn.Sequential(
                        nn.LayerNorm(arm_dim),
                        nn.Linear(arm_dim, arm_dim * 4, bias=False),
                        nn.SiLU(),
                        nn.Linear(arm_dim * 4, arm_dim, bias=False),
                    ),
                })
                for _ in range(self.meanflow_arm_num_layers)
            ])
            self.meanflow_arm_norm = nn.LayerNorm(arm_dim)
            self.meanflow_arm_recon = nn.Sequential(
                nn.Linear(arm_dim, arm_dim, bias=False),
                nn.SiLU(),
                nn.Linear(arm_dim, arm_input_dim, bias=False),
            )
            nn.init.zeros_(self.meanflow_arm_recon[-1].weight)

        self.perception_adanorm_cfg = perception_adanorm_cfg or {}
        self.use_perception_adanorm = bool(self.perception_adanorm_cfg.get("enabled", False))
        self.skip_perception_shared_attention = bool(
            self.perception_adanorm_cfg.get("skip_shared_attention", True)
        )
        self.perception_condition_mlp = None
        self.perception_suffix_norm = None
        self.perception_suffix_scale_shift = None
        if self.use_perception_adanorm:
            num_pooled_groups = 5  # det, map, occ, ego, motion
            self.perception_condition_mlp = nn.Sequential(
                nn.LayerNorm(self.vlm_hidden_size * num_pooled_groups),
                nn.Linear(self.vlm_hidden_size * num_pooled_groups, action_expert_cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(action_expert_cfg.hidden_size, action_expert_cfg.hidden_size),
            )
            self.perception_suffix_norm = nn.LayerNorm(action_expert_cfg.hidden_size, elementwise_affine=False)
            self.perception_suffix_scale_shift = nn.Linear(
                action_expert_cfg.hidden_size, 2 * action_expert_cfg.hidden_size
            )
            nn.init.zeros_(self.perception_suffix_scale_shift.weight)
            nn.init.zeros_(self.perception_suffix_scale_shift.bias)
            if self.flow_distill_enabled and self.flow_distill_train_adanorm:
                self.flow_distill_teacher_perception_condition_mlp = copy.deepcopy(self.perception_condition_mlp)
                self.flow_distill_teacher_perception_suffix_norm = copy.deepcopy(self.perception_suffix_norm)
                self.flow_distill_teacher_perception_suffix_scale_shift = copy.deepcopy(
                    self.perception_suffix_scale_shift
                )
                for module in (
                    self.flow_distill_teacher_perception_condition_mlp,
                    self.flow_distill_teacher_perception_suffix_norm,
                    self.flow_distill_teacher_perception_suffix_scale_shift,
                ):
                    module.requires_grad_(False)
                    module.eval()

        if dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif dtype == "float32":
            target_dtype = torch.float32
        else:
            target_dtype = torch.float16

        self.action_in_proj.to(target_dtype)
        self.status_mlp.to(target_dtype)
        self.hist_traj_encoder.to(target_dtype)
        for module in (
            self.flow_distill_teacher_action_in_proj,
            self.flow_distill_teacher_status_mlp,
            self.flow_distill_teacher_hist_traj_encoder,
        ):
            if module is not None:
                module.to(target_dtype)
        if self.meanflow_time_mlp is not None:
            self.meanflow_time_mlp.to(target_dtype)
        self.action_time_mlp_in.to(target_dtype)
        self.action_time_mlp_out.to(target_dtype)
        if self.action_query_tokens is not None:
            self.action_query_tokens.data = self.action_query_tokens.data.to(dtype=target_dtype)
        if self.action_query_decoder is not None:
            self.action_query_decoder.to(target_dtype)
        for module in (
            self.flow_distill_teacher_action_time_mlp_in,
            self.flow_distill_teacher_action_time_mlp_out,
        ):
            if module is not None:
                module.to(target_dtype)
        if self.meanflow_arm_traj_encoder is not None:
            self.meanflow_arm_traj_encoder.to(target_dtype)
        if self.meanflow_arm_cross_attn is not None:
            self.meanflow_arm_cross_attn.to(target_dtype)
        if self.meanflow_arm_norm is not None:
            self.meanflow_arm_norm.to(target_dtype)
        if self.meanflow_arm_recon is not None:
            self.meanflow_arm_recon.to(target_dtype)
        if self.perception_condition_mlp is not None:
            self.perception_condition_mlp.to(target_dtype)
        if self.perception_suffix_norm is not None:
            self.perception_suffix_norm.to(target_dtype)
        if self.perception_suffix_scale_shift is not None:
            self.perception_suffix_scale_shift.to(target_dtype)
        for module in (
            self.flow_distill_teacher_perception_condition_mlp,
            self.flow_distill_teacher_perception_suffix_norm,
            self.flow_distill_teacher_perception_suffix_scale_shift,
        ):
            if module is not None:
                module.to(target_dtype)

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
        self.feature_map_proj.to(target_dtype)
        self.fusion_weight_generators.to(target_dtype)

        if hasattr(self.qwen3_vl_with_expert.qwen3_vl, 'lm_head'):
            self.qwen3_vl_with_expert.qwen3_vl.lm_head.requires_grad_(False)

        self.gradient_checkpointing_enable()
        self.gradient_checkpointing_enabled = True

        self.vlm_grad_scale = vlm_grad_scale
        self.driving_deepstack = driving_deepstack

        self.view_token_str_list = NUSCENES_VIEW_TOKENS
        self.view_token_ids = None
        self.view_token_id_list = None
        self._prompt_token_cache = {}
        self._last_prefix_image_spans = None
        self._last_prefix_valid_lens = None

        self._cached_block_mask = None
        self._cached_block_mask_key = None
        self._cached_q_len_rounded = None

        self.adaptive_feature_fusion = torch.compile(
            self.adaptive_feature_fusion, mode="default", fullgraph=False, dynamic=True
        )
        self._apply_flow_distill_freeze_policy()

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_flow_distill_freeze_policy()
        return self

    def _get_view_token_ids(self, device):
        if self.view_token_ids is None:
            tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
            ids = []
            for t in self.view_token_str_list:
                tid = tokenizer.convert_tokens_to_ids(t)
                ids.append(tid)
            self.view_token_id_list = ids
            self.view_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        return self.view_token_ids.to(device)

    def _get_prompt_token_parts(self):
        variant = self.system_prompt_variant
        cached = self._prompt_token_cache.get(variant)
        if cached is not None:
            return cached

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        system_ids = tokenizer.encode("system", add_special_tokens=False)
        user_ids = tokenizer.encode("user", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        system_prompt = NUSCENES_SYSTEM_PROMPT_FULL if variant == "full" else NUSCENES_SYSTEM_PROMPT
        sys_content_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
        sys_part = [im_start_id] + system_ids + [nl_id] + sys_content_ids + [im_end_id, nl_id]
        user_start_part = [im_start_id] + user_ids + [nl_id]
        user_end_assistant_start_part = [im_end_id, nl_id, im_start_id] + assistant_ids + [nl_id]

        cached = {
            "im_end_id": im_end_id,
            "nl_id": nl_id,
            "sys_part": sys_part,
            "user_start_part": user_start_part,
            "user_end_assistant_start_part": user_end_assistant_start_part,
        }
        self._prompt_token_cache[variant] = cached
        return cached

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        gc_kwargs = {"use_reentrant": False}
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_perception_expert.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_action_expert.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_perception_expert.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_action_expert.gradient_checkpointing_disable()

    def merge_and_save_lora(self, save_dir: Optional[str] = None) -> None:
        target = save_dir or self.lora_merge_save_dir
        if target is None:
            logging.warning(
                "[LoRA] merge_and_save_lora called but no save_dir configured. "
                "Set lora_merge_save_dir in the config or pass it explicitly."
            )
            return
        self.qwen3_vl_with_expert.merge_and_save_lora(target)

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

    def _set_trainable(self, module: Optional[nn.Module], trainable: bool) -> None:
        if module is not None:
            module.requires_grad_(trainable)

    def _flow_distill_teacher_modules(self):
        return (
            self.flow_distill_teacher_action_expert,
            self.flow_distill_teacher_action_out_proj,
            self.flow_distill_teacher_action_in_proj,
            self.flow_distill_teacher_status_mlp,
            self.flow_distill_teacher_hist_traj_encoder,
            self.flow_distill_teacher_action_time_mlp_in,
            self.flow_distill_teacher_action_time_mlp_out,
            self.flow_distill_teacher_perception_condition_mlp,
            self.flow_distill_teacher_perception_suffix_norm,
            self.flow_distill_teacher_perception_suffix_scale_shift,
        )

    def _freeze_flow_distill_teacher_modules(self) -> None:
        for module in self._flow_distill_teacher_modules():
            if module is None:
                continue
            module.requires_grad_(False)
            module.eval()

    def _sync_flow_distill_teacher_pairs(self, pairs) -> None:
        for teacher_module, student_module in pairs:
            if teacher_module is not None and student_module is not None:
                teacher_module.load_state_dict(student_module.state_dict())

    def _mark_flow_distill_teacher_initialized_from_state_dict(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        if not self.flow_distill_enabled:
            return

        def has_teacher_key(module_name: str) -> bool:
            dotted_prefix = prefix + module_name + "."
            flattened_name = (prefix + module_name).replace(".", "_") + "_"
            ema_flattened_name = "ema_" + flattened_name
            return any(
                key.startswith(dotted_prefix)
                or key.startswith(flattened_name)
                or key.startswith(ema_flattened_name)
                or module_name in key
                for key in state_dict.keys()
            )

        if (
            has_teacher_key("flow_distill_teacher_action_expert")
            and has_teacher_key("flow_distill_teacher_action_out_proj")
        ):
            self._flow_distill_teacher_initialized.fill_(True)
        if self.flow_distill_train_action_adapters and any(
            has_teacher_key(name)
            for name in (
                "flow_distill_teacher_action_in_proj",
                "flow_distill_teacher_status_mlp",
                "flow_distill_teacher_hist_traj_encoder",
                "flow_distill_teacher_action_time_mlp_in",
                "flow_distill_teacher_action_time_mlp_out",
            )
        ):
            self._flow_distill_teacher_action_adapters_initialized.fill_(True)
        if self.flow_distill_train_adanorm and any(
            has_teacher_key(name)
            for name in (
                "flow_distill_teacher_perception_condition_mlp",
                "flow_distill_teacher_perception_suffix_norm",
                "flow_distill_teacher_perception_suffix_scale_shift",
            )
        ):
            self._flow_distill_teacher_adanorm_initialized.fill_(True)

    @staticmethod
    def _first_tensor_device_dtype(module: nn.Module, fallback: torch.Tensor):
        for tensor in module.parameters(recurse=True):
            return tensor.device, tensor.dtype
        for tensor in module.buffers(recurse=True):
            return tensor.device, tensor.dtype
        return fallback.device, fallback.dtype

    def _apply_flow_distill_freeze_policy(self) -> None:
        if not (self.flow_distill_enabled and self.flow_distill_freeze_condition_modules):
            return

        self.requires_grad_(False)

        trainable_modules = [
            self.qwen3_vl_with_expert.qwen3_action_expert,
            self.action_out_proj,
        ]
        if self.flow_distill_train_action_adapters:
            trainable_modules.extend([
                self.action_in_proj,
                self.status_mlp,
                self.hist_traj_encoder,
                self.action_time_mlp_in,
                self.action_time_mlp_out,
            ])
        if self.flow_distill_train_adanorm:
            trainable_modules.extend([
                self.perception_condition_mlp,
                self.perception_suffix_norm,
                self.perception_suffix_scale_shift,
            ])

        for module in trainable_modules:
            self._set_trainable(module, True)

        self._freeze_flow_distill_teacher_modules()

    def _ensure_flow_distill_teacher_initialized(self) -> None:
        if not self.flow_distill_enabled:
            return
        if self.flow_distill_teacher_action_expert is None or self.flow_distill_teacher_action_out_proj is None:
            raise RuntimeError("Flow distillation teacher modules are not initialized")
        action_adapter_pairs = [
            (self.flow_distill_teacher_action_in_proj, self.action_in_proj),
            (self.flow_distill_teacher_status_mlp, self.status_mlp),
            (self.flow_distill_teacher_hist_traj_encoder, self.hist_traj_encoder),
            (self.flow_distill_teacher_action_time_mlp_in, self.action_time_mlp_in),
            (self.flow_distill_teacher_action_time_mlp_out, self.action_time_mlp_out),
        ]
        adanorm_pairs = [
            (self.flow_distill_teacher_perception_condition_mlp, self.perception_condition_mlp),
            (self.flow_distill_teacher_perception_suffix_norm, self.perception_suffix_norm),
            (self.flow_distill_teacher_perception_suffix_scale_shift, self.perception_suffix_scale_shift),
        ]
        teacher_student_pairs = action_adapter_pairs + adanorm_pairs
        if bool(self._flow_distill_teacher_initialized.item()):
            if (
                self.flow_distill_train_action_adapters
                and not bool(self._flow_distill_teacher_action_adapters_initialized.item())
            ):
                self._sync_flow_distill_teacher_pairs(action_adapter_pairs)
                self._flow_distill_teacher_action_adapters_initialized.fill_(True)
            if (
                self.flow_distill_train_adanorm
                and not bool(self._flow_distill_teacher_adanorm_initialized.item())
            ):
                self._sync_flow_distill_teacher_pairs(adanorm_pairs)
                self._flow_distill_teacher_adanorm_initialized.fill_(True)
            self._freeze_flow_distill_teacher_modules()
            return
        self.flow_distill_teacher_action_expert.load_state_dict(
            self.qwen3_vl_with_expert.qwen3_action_expert.state_dict()
        )
        self.flow_distill_teacher_action_out_proj.load_state_dict(self.action_out_proj.state_dict())
        self._sync_flow_distill_teacher_pairs(teacher_student_pairs)
        student_expert_param = next(self.qwen3_vl_with_expert.qwen3_action_expert.parameters())
        self.flow_distill_teacher_action_expert.to(
            device=student_expert_param.device,
            dtype=student_expert_param.dtype,
        )
        self.flow_distill_teacher_action_out_proj.to(
            device=self.action_out_proj.weight.device,
            dtype=self.action_out_proj.weight.dtype,
        )
        for teacher_module, student_module in teacher_student_pairs:
            if teacher_module is not None and student_module is not None:
                device, dtype = self._first_tensor_device_dtype(student_module, student_expert_param)
                teacher_module.to(device=device, dtype=dtype)
        self._freeze_flow_distill_teacher_modules()
        self._flow_distill_teacher_initialized.fill_(True)
        if self.flow_distill_train_action_adapters:
            self._flow_distill_teacher_action_adapters_initialized.fill_(True)
        if self.flow_distill_train_adanorm:
            self._flow_distill_teacher_adanorm_initialized.fill_(True)

    @staticmethod
    def _masked_mean(loss_unreduced: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            m = mask
            if m.dim() == 3 and m.shape[1] == 1:
                m = m.squeeze(1)
            m = m.to(dtype=loss_unreduced.dtype, device=loss_unreduced.device)
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            loss_unreduced = loss_unreduced * m
            denom = m.sum().clamp_min(1.0) * loss_unreduced.shape[-1]
            return loss_unreduced.sum() / denom
        return loss_unreduced.mean()

    def _flow_distill_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.flow_distill_loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.flow_distill_loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        else:
            loss = F.smooth_l1_loss(pred, target, reduction="none")
        return self._masked_mean(loss, mask)

    def _action_query_supervised_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = pred.new_zeros(())
        if self.action_query_l1_loss_weight > 0.0:
            l1 = self._masked_mean(F.l1_loss(pred, target, reduction="none"), mask)
            loss = loss + self.action_query_l1_loss_weight * l1
        if self.action_query_l2_loss_weight > 0.0:
            l2 = self._masked_mean(F.mse_loss(pred, target, reduction="none"), mask)
            loss = loss + self.action_query_l2_loss_weight * l2
        return loss

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    @staticmethod
    def _pool_token_span(tokens: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if tokens is None or end <= start:
            return tokens.new_zeros((tokens.shape[0], tokens.shape[-1]))
        return tokens[:, start:end].mean(dim=1)

    def build_perception_condition(
        self,
        perception_embs: torch.Tensor,
        perception_lengths: Dict[str, int],
        condition_mlp_override: Optional[nn.Module] = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_perception_adanorm:
            return None

        det_len = int(perception_lengths.get('det', 0))
        map_len = int(perception_lengths.get('map', 0))
        occ_len = int(perception_lengths.get('occ', 0))
        ego_len = int(perception_lengths.get('ego', 0))
        motion_len = int(perception_lengths.get('motion', 0))

        d0, d1 = 0, det_len
        m0, m1 = d1, d1 + map_len
        o0, o1 = m1, m1 + occ_len
        e0, e1 = o1, o1 + ego_len
        t0, t1 = e1, e1 + motion_len

        pooled = torch.cat([
            self._pool_token_span(perception_embs, d0, d1),
            self._pool_token_span(perception_embs, m0, m1),
            self._pool_token_span(perception_embs, o0, o1),
            self._pool_token_span(perception_embs, e0, e1),
            self._pool_token_span(perception_embs, t0, t1),
        ], dim=-1)

        condition_mlp = condition_mlp_override or self.perception_condition_mlp
        target_dtype = condition_mlp[1].weight.dtype
        return condition_mlp(pooled.to(dtype=target_dtype))

    def apply_perception_adanorm(
        self,
        suffix_embs: torch.Tensor,
        perception_condition: Optional[torch.Tensor],
        suffix_norm_override: Optional[nn.Module] = None,
        scale_shift_override: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if perception_condition is None or not self.use_perception_adanorm:
            return suffix_embs

        suffix_norm = suffix_norm_override or self.perception_suffix_norm
        scale_shift_layer = scale_shift_override or self.perception_suffix_scale_shift
        target_dtype = scale_shift_layer.weight.dtype
        scale_shift = scale_shift_layer(perception_condition.to(dtype=target_dtype))
        scale, shift = scale_shift.chunk(2, dim=-1)
        normed = suffix_norm(suffix_embs.to(dtype=target_dtype))
        modulated = suffix_embs.to(dtype=target_dtype) + normed * scale[:, None, :] + shift[:, None, :]
        return modulated.to(dtype=suffix_embs.dtype)

    def _apply_meanflow_arm(
        self,
        proposal_x0: torch.Tensor,
        action_context: torch.Tensor,
        perception_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.enable_meanflow_arm:
            return proposal_x0
        if (
            self.meanflow_arm_traj_encoder is None or
            self.meanflow_arm_cross_attn is None or
            self.meanflow_arm_norm is None or
            self.meanflow_arm_recon is None
        ):
            raise RuntimeError("MeanFlow ARM modules are not initialized")

        target_dtype = self.meanflow_arm_recon[-1].weight.dtype
        proposal_flat = proposal_x0.reshape(proposal_x0.shape[0], -1).to(dtype=target_dtype)
        traj_token = self.meanflow_arm_traj_encoder(proposal_flat).unsqueeze(1)

        memory_parts = [action_context.to(dtype=target_dtype)]
        if perception_condition is not None:
            memory_parts.append(perception_condition.to(dtype=target_dtype).unsqueeze(1))
        memory = torch.cat(memory_parts, dim=1)
        if self.meanflow_arm_detach_context:
            memory = memory.detach()

        refined_token = traj_token
        for block in self.meanflow_arm_cross_attn:
            q = block["norm_q"](refined_token)
            mem = block["norm_mem"](memory)
            attn_out, _ = block["attn"](q, mem, mem, need_weights=False)
            refined_token = refined_token + attn_out
            refined_token = refined_token + block["ffn"](refined_token)

        residual = self.meanflow_arm_recon(self.meanflow_arm_norm(refined_token.squeeze(1)))
        residual = residual.reshape_as(proposal_x0).to(dtype=proposal_x0.dtype)
        return proposal_x0 + residual

    @staticmethod
    def _masked_l1(loss_unreduced: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            m = mask
            if m.dim() == 3 and m.shape[1] == 1:
                m = m.squeeze(1)
            m = m.to(dtype=loss_unreduced.dtype, device=loss_unreduced.device)
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            loss_unreduced = loss_unreduced * m
            denom = m.sum().clamp_min(1.0) * loss_unreduced.shape[-1]
            return loss_unreduced.sum() / denom
        return loss_unreduced.mean()

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        t = sample_beta(self.time_beta_alpha, self.time_beta_beta, bsize, device)
        t = t * 0.999 + 0.001
        return t.to(dtype=torch.float32, device=device)

    def sample_meanflow_interval(self, bsize, device):
        if not 0.0 <= self.meanflow_ratio_r_not_equal_t <= 1.0:
            raise ValueError(
                "meanflow_ratio_r_not_equal_t must be in [0, 1], "
                f"got {self.meanflow_ratio_r_not_equal_t}"
            )

        if self.meanflow_time_sampling == "logit_normal":
            t = torch.sigmoid(
                self.meanflow_time_mu + self.meanflow_time_sigma * torch.randn(bsize, device=device)
            )
            r = torch.sigmoid(
                self.meanflow_time_mu + self.meanflow_time_sigma * torch.randn(bsize, device=device)
            )
        elif self.meanflow_time_sampling == "uniform":
            t = torch.rand(bsize, device=device)
            r = torch.rand(bsize, device=device)
        else:
            t = sample_beta(self.time_beta_alpha, self.time_beta_beta, bsize, device)
            r = sample_beta(self.time_beta_alpha, self.time_beta_beta, bsize, device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        if self.meanflow_min_dt > 0:
            t = t.clamp_min(self.meanflow_min_dt)
            too_close = (t - r) < self.meanflow_min_dt
            r = torch.where(too_close, (t - self.meanflow_min_dt).clamp_min(0.0), r)

        if self.meanflow_ratio_r_not_equal_t < 1.0:
            same_time = torch.rand(bsize, device=device) >= self.meanflow_ratio_r_not_equal_t
            r = torch.where(same_time, t, r)

        return r.to(dtype=torch.float32), t.to(dtype=torch.float32)

    def norm_delta(self, delta_meter: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor([0.0233, 2.2707], device=delta_meter.device, dtype=delta_meter.dtype)
        std = torch.tensor([0.3427, 1.8668], device=delta_meter.device, dtype=delta_meter.dtype)
        return (delta_meter - mu) / (std + 1e-6)

    def denorm_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor([0.0233, 2.2707], device=delta_norm.device, dtype=delta_norm.dtype)
        std = torch.tensor([0.3427, 1.8668], device=delta_norm.device, dtype=delta_norm.dtype)
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

        parts_embs = []
        parts_pad  = []
        parts_att  = []
        perception_lengths = {'det': 0, 'map': 0, 'occ': 0, 'ego': 0, 'motion': 0}

        # ── det ──────────────────────────────────────────────────────────────
        if 'det' in query_select and len(stage1_outs.get('det_predictions', [])) > 0:
            det_feat   = stage1_outs['det_instance_feature']
            det_anchor = stage1_outs['det_predictions'][-1]
            det_embs   = self.det_proj_up(det_feat.to(dtype=proj_dtype))
            anchor_embed_256 = self.unified_decoder.det_anchor_encoder(det_anchor)
            anchor_embed_vlm = self.det_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            det_embs = (det_embs + anchor_embed_vlm).to(dtype)
            det_n = det_embs.shape[1]
            parts_embs.append(det_embs)
            parts_pad.append(torch.ones( (batch_size, det_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, det_n), dtype=torch.bool, device=device))
            perception_lengths['det'] = det_n

        # ── map ──────────────────────────────────────────────────────────────
        if 'map' in query_select and len(stage1_outs.get('map_predictions', [])) > 0:
            map_feat   = stage1_outs['map_instance_feature']
            map_anchor = stage1_outs['map_predictions'][-1]
            map_embs   = self.map_proj_up(map_feat.to(dtype=proj_dtype))
            anchor_embed_out = self.unified_decoder.map_anchor_encoder(map_anchor)
            anchor_embed_256 = anchor_embed_out[0] if isinstance(anchor_embed_out, (tuple, list)) else anchor_embed_out
            anchor_embed_vlm = self.map_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            map_embs = (map_embs + anchor_embed_vlm).to(dtype)
            map_n = map_embs.shape[1]
            parts_embs.append(map_embs)
            parts_pad.append(torch.ones( (batch_size, map_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, map_n), dtype=torch.bool, device=device))
            perception_lengths['map'] = map_n

        # ── occ (always included when the module exists) ──────────────────────
        occ_embs = self.occ_queries.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        occ_n = occ_embs.shape[1]
        parts_embs.append(occ_embs)
        parts_pad.append(torch.ones( (batch_size, occ_n), dtype=torch.bool, device=device))
        parts_att.append(torch.zeros((batch_size, occ_n), dtype=torch.bool, device=device))
        perception_lengths['occ'] = occ_n

        # ── ego ──────────────────────────────────────────────────────────────
        if 'ego' in query_select:
            ego_feat   = stage1_outs['ego_instance_feature']
            ego_anchor = stage1_outs.get('ego_anchor', None)
            ego_embs   = self.ego_proj_up(ego_feat.to(dtype=proj_dtype))
            if hasattr(self.unified_decoder, 'ego_anchor_encoder') and ego_anchor is not None:
                ego_anchor_embed_256 = self.unified_decoder.ego_anchor_encoder(ego_anchor)
                ego_anchor_embed_vlm = self.ego_proj_up(ego_anchor_embed_256.to(dtype=proj_dtype))
                ego_embs = (ego_embs + ego_anchor_embed_vlm).to(dtype)
            else:
                ego_embs = ego_embs.to(dtype)
            ego_n = ego_embs.shape[1]
            parts_embs.append(ego_embs)
            parts_pad.append(torch.ones( (batch_size, ego_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, ego_n), dtype=torch.bool, device=device))
            perception_lengths['ego'] = ego_n

        # ── motion ───────────────────────────────────────────────────────────
        if motion_token_256 is not None:
            motion_token_vlm = self.motion_proj_up(motion_token_256.to(dtype=proj_dtype)).to(dtype)
            motion_n = motion_token_vlm.shape[1]
            parts_embs.append(motion_token_vlm)
            parts_pad.append(torch.ones( (batch_size, motion_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, motion_n), dtype=torch.bool, device=device))
            perception_lengths['motion'] = motion_n

        # ── assemble ─────────────────────────────────────────────────────────
        if parts_embs:
            perception_embs      = torch.cat(parts_embs, dim=1)
            perception_pad_masks = torch.cat(parts_pad,  dim=1)
            perception_att_masks = torch.cat(parts_att,  dim=1)
        else:
            perception_embs      = torch.empty((batch_size, 0, self.vlm_hidden_size), dtype=dtype, device=device)
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

                current_fused += ref * w

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
        hist_traj=None,
        gt_trajs=None,
        camera_inputs_permuted: bool = False,
    ) -> DrivingBatch:
        device = img.device if img is not None else torch.device("cuda")
        b = int(img.shape[0]) if torch.is_tensor(img) else 1

        permute_indices = [0, 2, 1, 4, 5, 3]
        images = img if camera_inputs_permuted else img[:, permute_indices]

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

        nav_cmd_texts = [_NAV_CMD_FIXED.get(i, _NAV_CMD_FIXED[2]) for i in idx_list]

        hist_traj_np = hist_traj.detach().cpu().numpy() if hist_traj is not None else None

        if not hasattr(self.qwen3_vl_with_expert, "processor") or self.qwen3_vl_with_expert.processor is None:
            raise RuntimeError("QwenVLAPlanningHead expects `self.qwen3_vl_with_expert.processor`")

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        prompt_parts = self._get_prompt_token_parts()
        sys_part = prompt_parts["sys_part"]
        user_start_part = prompt_parts["user_start_part"]
        user_end_assistant_start_part = prompt_parts["user_end_assistant_start_part"]
        im_end_id = prompt_parts["im_end_id"]
        nl_id = prompt_parts["nl_id"]

        input_ids_list = []
        attention_mask_list = []

        for i in range(b):
            hist_points = hist_traj_np[i] if hist_traj_np is not None else []
            points_str = [f"({pt[0]:+07.2f}, {pt[1]:+07.2f})" for pt in hist_points]
            hist_traj_str = f"[PT_HIST, {', '.join(points_str)}]"

            user_prompt_text = NUSCENES_USER_PROMPT_TEMPLATE.format(
                nav_cmd=nav_cmd_texts[i],
                hist_traj_str=hist_traj_str,
            )
            user_content_ids = tokenizer.encode(user_prompt_text, add_special_tokens=False)

            full_ids = sys_part + user_start_part + user_content_ids + user_end_assistant_start_part

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        tokenized_prompt = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        tokenized_prompt_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        traj_answer_ids = None
        traj_answer_mask = None
        traj_labels = None
        if gt_trajs is not None:
            if self.traj_ar_target == "waypoint":
                gt_trajs_target = torch.cumsum(gt_trajs, dim=1)
            elif self.traj_ar_target == "norm_delta":
                gt_trajs_target = self.norm_delta(gt_trajs)
            else:
                gt_trajs_target = gt_trajs

            gt_trajs_np = gt_trajs_target.detach().cpu().numpy()
            answer_ids_list = []
            for i in range(b):
                pts = gt_trajs_np[i]
                pts_str = ", ".join(f"({p[0]:+07.2f}, {p[1]:+07.2f})" for p in pts)
                answer_text = f"[PT, {pts_str}]"
                answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
                answer_token_ids = answer_token_ids + [im_end_id, nl_id]
                answer_ids_list.append(torch.tensor(answer_token_ids, dtype=torch.long))

            traj_answer_ids = pad_sequence(
                answer_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)

            FIXED_ANSWER_MAX_LEN = 128
            curr_ans_len = traj_answer_ids.shape[1]
            if curr_ans_len < FIXED_ANSWER_MAX_LEN:
                traj_answer_ids = F.pad(
                    traj_answer_ids, (0, FIXED_ANSWER_MAX_LEN - curr_ans_len),
                    value=tokenizer.pad_token_id
                )
            elif curr_ans_len > FIXED_ANSWER_MAX_LEN:
                traj_answer_ids = traj_answer_ids[:, :FIXED_ANSWER_MAX_LEN]

            traj_answer_mask = (traj_answer_ids != tokenizer.pad_token_id).long()
            traj_labels = traj_answer_ids.masked_fill(traj_answer_mask == 0, -100)

        return DrivingBatch(
            images=images,
            image_masks=image_masks,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            command=command,
            ego_status=ego_status,
            view_token_ids=view_token_ids,
            traj_answer_ids=traj_answer_ids,
            traj_answer_mask=traj_answer_mask,
            traj_labels=traj_labels,
        )


    def embed_prefix(self, batch: DrivingBatch):
        images_tensor = batch.images
        device = self.action_in_proj.weight.device

        image_features, feature_lens, all_image_grids, deepstack_features, raw_features = self.qwen3_vl_with_expert.embed_image_tensor(images_tensor)

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        vision_start_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_start_token_id
        vision_end_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_end_token_id
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        nl_id = self._get_prompt_token_parts()["nl_id"]

        bs = images_tensor.shape[0]
        num_views_per_sample = 6
        self._get_view_token_ids(device)
        view_token_id_list = self.view_token_id_list

        prefix_input_ids_list = []
        image_spans = []

        FIXED_PREFIX_MAX_LEN = 3740
        use_dynamic_prefix_len = bool(
            getattr(self.qwen3_vl_with_expert, "_image_token_compression_enabled", lambda: False)()
        )

        for b_idx in range(bs):
            sample_input_ids = []
            sample_spans = []

            for v_idx in range(num_views_per_sample):
                img_len = feature_lens[b_idx * num_views_per_sample + v_idx]
                img_start = len(sample_input_ids) + 3
                ids = [view_token_id_list[v_idx], nl_id, vision_start_id] + \
                    [image_token_id] * img_len + \
                    [vision_end_id, nl_id]
                sample_input_ids.extend(ids)
                sample_spans.append((img_start, img_start + int(img_len)))

            prompt_mask = batch.tokenized_prompt_mask[b_idx].bool()
            prompt_ids = batch.tokenized_prompt[b_idx][prompt_mask].tolist()
            sample_input_ids.extend(prompt_ids)

            prefix_input_ids_list.append(torch.tensor(sample_input_ids, dtype=torch.long, device=device))
            image_spans.append(sample_spans)

        prompt_lens_list = [ids.shape[0] for ids in prefix_input_ids_list]

        if batch.traj_answer_ids is not None:
            full_ids_list = []
            for b_idx in range(bs):
                p_ids = prefix_input_ids_list[b_idx]
                a_ids = batch.traj_answer_ids[b_idx]
                if (not use_dynamic_prefix_len) and p_ids.shape[0] > FIXED_PREFIX_MAX_LEN:
                    removed_image_tokens = (p_ids[FIXED_PREFIX_MAX_LEN:] == image_token_id).sum().item()
                    if removed_image_tokens > 0:
                        raise ValueError(
                            "Prefix truncation would drop image tokens after VisionZip compression. "
                            "This would desynchronize image_features, deepstack_features, and sparse "
                            "position ids; increase FIXED_PREFIX_MAX_LEN or shorten the prompt."
                        )
                    p_ids = p_ids[:FIXED_PREFIX_MAX_LEN]
                    prompt_lens_list[b_idx] = FIXED_PREFIX_MAX_LEN
                full_ids_list.append(torch.cat([p_ids, a_ids], dim=0))

            prefix_input_ids = pad_sequence(
                full_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            if not use_dynamic_prefix_len:
                fixed_total_len = FIXED_PREFIX_MAX_LEN + batch.traj_answer_ids.shape[1]
                curr_len = prefix_input_ids.shape[1]
                if curr_len < fixed_total_len:
                    prefix_input_ids = F.pad(prefix_input_ids, (0, fixed_total_len - curr_len), value=tokenizer.pad_token_id)
                elif curr_len > fixed_total_len:
                    prefix_input_ids = prefix_input_ids[:, :fixed_total_len]
        else:
            prefix_input_ids = pad_sequence(prefix_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            if not use_dynamic_prefix_len:
                curr_len = prefix_input_ids.shape[1]
                if curr_len < FIXED_PREFIX_MAX_LEN:
                    prefix_input_ids = F.pad(prefix_input_ids, (0, FIXED_PREFIX_MAX_LEN - curr_len), value=tokenizer.pad_token_id)
                elif curr_len > FIXED_PREFIX_MAX_LEN:
                    removed_image_tokens = (prefix_input_ids[:, FIXED_PREFIX_MAX_LEN:] == image_token_id).sum().item()
                    if removed_image_tokens > 0:
                        raise ValueError(
                            "Prefix truncation would drop image tokens after VisionZip compression. "
                            "This would desynchronize image_features, deepstack_features, and sparse "
                            "position ids; increase FIXED_PREFIX_MAX_LEN or shorten the prompt."
                        )
                    prefix_input_ids = prefix_input_ids[:, :FIXED_PREFIX_MAX_LEN]

        prompt_only_len = max(prompt_lens_list)

        prefix_pad_masks = (prefix_input_ids != tokenizer.pad_token_id)

        input_embeds = self.qwen3_vl_with_expert.qwen3_vl.get_input_embeddings()(prefix_input_ids)

        image_mask = (prefix_input_ids == image_token_id)

        if image_mask.sum() != image_features.shape[0]:
            target_count = image_mask.sum()
            current_count = image_features.shape[0]
            raise ValueError(
                f"Visual features mismatch after prefix construction! Feat: {current_count}, "
                f"Tokens: {target_count}. Do not truncate image features because sparse position "
                "ids and deepstack features must stay aligned."
            )

        input_embeds = input_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features.to(input_embeds.dtype))

        prefix_att_marks = torch.zeros_like(prefix_pad_masks, dtype=torch.long)
        for b_idx in range(bs):
            p_len = prompt_lens_list[b_idx]
            prefix_att_marks[b_idx, :p_len] = 1
        if batch.traj_answer_ids is not None:
            for b_idx in range(bs):
                p_len = prompt_lens_list[b_idx]
                real_ans_len = batch.traj_answer_mask[b_idx].sum().item()
                prefix_att_marks[b_idx, p_len:p_len + real_ans_len] = 1

        batch.prompt_lens = prompt_lens_list
        self._last_prefix_image_spans = image_spans
        self._last_prefix_valid_lens = prompt_lens_list

        return input_embeds, prefix_pad_masks, prefix_att_marks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len

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

        if use_gt:
            if batch.ego_status is None:
                raise ValueError("ego_status is required during training")
            ego = batch.ego_status.to(device=device, dtype=dtype)
            if ego.shape[-1] >= self.ego_status_dim:
                ego = ego[..., : self.ego_status_dim]
            else:
                pad = cmd_onehot.new_zeros((ego.shape[0], self.ego_status_dim - ego.shape[-1]))
                ego = torch.cat([ego, pad], dim=-1)
            return torch.cat([cmd_onehot, ego], dim=-1)

        if ego_status_pred is None:
            ego_status_pred = cmd_onehot.new_zeros((cmd_onehot.shape[0], self.ego_status_dim))
        return torch.cat([cmd_onehot, ego_status_pred.to(dtype=dtype)], dim=-1)

    def embed_suffix(
        self,
        batch: DrivingBatch,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        interval_start: Optional[torch.Tensor] = None,
        ego_status_pred: Optional[torch.Tensor] = None,
        use_gt_status: bool = False,
        hist_traj: Optional[torch.Tensor] = None,
        action_in_proj_override: Optional[nn.Module] = None,
        status_mlp_override: Optional[nn.Module] = None,
        hist_traj_encoder_override: Optional[nn.Module] = None,
        action_time_mlp_in_override: Optional[nn.Module] = None,
        action_time_mlp_out_override: Optional[nn.Module] = None,
    ):
        device = actions.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        action_in_proj = action_in_proj_override or self.action_in_proj
        status_mlp = status_mlp_override or self.status_mlp
        hist_traj_encoder = hist_traj_encoder_override or self.hist_traj_encoder
        action_time_mlp_in = action_time_mlp_in_override or self.action_time_mlp_in
        action_time_mlp_out = action_time_mlp_out_override or self.action_time_mlp_out

        B = actions.shape[0]

        status_input = self._maybe_get_status_features(
            batch, ego_status_pred, use_gt=use_gt_status, dtype=dtype,
        )
        status_emb = self._apply_checkpoint(status_mlp, status_input)
        status_emb = status_emb.unsqueeze(1)

        if hist_traj is not None:
            if hist_traj.dtype != dtype:
                hist_traj = hist_traj.to(dtype=dtype)
            hist_traj_emb = self._apply_checkpoint(hist_traj_encoder, hist_traj.flatten(1))
            status_emb = torch.cat([hist_traj_emb.unsqueeze(1), status_emb], dim=1)

        num_status_tokens = status_emb.shape[1]

        if actions.dtype != dtype:
            actions = actions.to(dtype=dtype)

        action_emb = self._apply_checkpoint(action_in_proj, actions)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, action_in_proj.out_features, self.min_period, self.max_period, device=device
        )
        if interval_start is not None:
            if self.meanflow_time_mlp is None:
                raise RuntimeError("interval_start was provided but flow_mode is not meanflow")
            start_emb = create_sinusoidal_pos_embedding(
                interval_start, action_in_proj.out_features, self.min_period, self.max_period, device=device
            )
            time_pair = torch.cat([start_emb, time_emb], dim=-1).to(dtype)
            time_emb = time_emb + self.meanflow_time_mlp(time_pair).to(dtype)
        fused_input = torch.cat([action_emb, time_emb.unsqueeze(1).expand_as(action_emb)], dim=-1).to(dtype)

        def mlp_func(x):
            x = action_time_mlp_in(x)
            x = F.silu(x)
            return action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, fused_input)
        suffix_emb = torch.cat([status_emb, action_time_emb], dim=1)

        suffix_len = suffix_emb.shape[1]
        pad_masks = torch.ones((B, suffix_len), dtype=torch.bool, device=device)
        att_marks = [1] * num_status_tokens + [0] * (suffix_len - num_status_tokens)
        att_marks = torch.tensor(att_marks, dtype=torch.bool, device=device)[None, :].expand(B, -1)

        return suffix_emb, pad_masks, att_marks

    def embed_action_queries(
        self,
        batch: DrivingBatch,
        ego_status_pred: Optional[torch.Tensor] = None,
        use_gt_status: bool = False,
        hist_traj: Optional[torch.Tensor] = None,
    ):
        if self.action_query_tokens is None:
            raise RuntimeError("action_training_mode='query_supervised' requires action_query_tokens")
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        B = batch.tokenized_prompt.shape[0]

        status_input = self._maybe_get_status_features(
            batch, ego_status_pred, use_gt=use_gt_status, dtype=dtype,
        )
        status_emb = self._apply_checkpoint(self.status_mlp, status_input).unsqueeze(1)

        if hist_traj is not None:
            if hist_traj.dtype != dtype:
                hist_traj = hist_traj.to(dtype=dtype)
            hist_traj_emb = self._apply_checkpoint(self.hist_traj_encoder, hist_traj.flatten(1))
            status_emb = torch.cat([hist_traj_emb.unsqueeze(1), status_emb], dim=1)

        query_emb = self.action_query_tokens.to(device=device, dtype=dtype)
        query_emb = query_emb.unsqueeze(0).expand(B, -1, -1)
        suffix_emb = torch.cat([status_emb, query_emb], dim=1)

        suffix_len = suffix_emb.shape[1]
        num_status_tokens = status_emb.shape[1]
        pad_masks = torch.ones((B, suffix_len), dtype=torch.bool, device=device)
        att_marks = [1] * num_status_tokens + [0] * (suffix_len - num_status_tokens)
        att_marks = torch.tensor(att_marks, dtype=torch.bool, device=device)[None, :].expand(B, -1)

        return suffix_emb, pad_masks, att_marks

    def get_position_ids(self, input_ids, image_grid_thw, pad_masks):
        sparse_info = getattr(self.qwen3_vl_with_expert, "_last_sparse_image_position_ids", None)
        if sparse_info is not None:
            spans = getattr(self, "_last_prefix_image_spans", None)
            valid_lens = getattr(self, "_last_prefix_valid_lens", None)
            if spans is not None:
                fast_result = self._build_sparse_position_ids_from_spans(
                    input_ids, pad_masks, sparse_info, spans, valid_lens
                )
                if fast_result is not None:
                    return fast_result
            return self._build_sparse_position_ids(input_ids, image_grid_thw, pad_masks, sparse_info)
        attention_mask = pad_masks.long()
        position_ids, rope_deltas = self.qwen3_vl_with_expert.vlm_base.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def _build_sparse_position_ids_from_spans(self, input_ids, pad_masks, sparse_info, image_spans, valid_lens):
        if len(image_spans) != input_ids.shape[0]:
            return None
        if valid_lens is not None and len(valid_lens) != input_ids.shape[0]:
            return None
        expected_images = sum(len(spans) for spans in image_spans)
        if expected_images != len(sparse_info):
            return None

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        position_ids = torch.zeros((3, batch_size, seq_len), dtype=torch.long, device=device)
        rope_deltas = []
        sparse_cursor = 0

        for b_idx, spans in enumerate(image_spans):
            valid_len = int(valid_lens[b_idx]) if valid_lens is not None else int(pad_masks[b_idx].sum().item())
            pos_cursor = 0
            prev_end = 0

            for img_start, img_end in spans:
                if img_start < prev_end or img_end > valid_len:
                    return None
                if img_start > prev_end:
                    text_pos = torch.arange(
                        pos_cursor,
                        pos_cursor + (img_start - prev_end),
                        device=device,
                        dtype=torch.long,
                    )
                    position_ids[:, b_idx, prev_end:img_start] = text_pos.view(1, -1)
                    pos_cursor += img_start - prev_end

                info = sparse_info[sparse_cursor]
                sparse_cursor += 1
                sparse_pos = info["position_ids"] if isinstance(info, dict) else info
                sparse_pos = sparse_pos.to(device=device, dtype=torch.long)
                img_len = img_end - img_start
                if sparse_pos.shape[1] != img_len:
                    return None
                image_pos = sparse_pos + pos_cursor
                position_ids[:, b_idx, img_start:img_end] = image_pos
                if isinstance(info, dict) and "grid_hw" in info:
                    grid_h, grid_w = info["grid_hw"]
                    pos_cursor += max(int(grid_h), int(grid_w))
                else:
                    pos_cursor = int(image_pos.max().item()) + 1
                prev_end = img_end

            if valid_len > prev_end:
                text_pos = torch.arange(
                    pos_cursor,
                    pos_cursor + (valid_len - prev_end),
                    device=device,
                    dtype=torch.long,
                )
                position_ids[:, b_idx, prev_end:valid_len] = text_pos.view(1, -1)
                pos_cursor += valid_len - prev_end

            if valid_len < seq_len:
                position_ids[:, b_idx, valid_len:] = max(pos_cursor - 1, 0)
            rope_deltas.append(torch.tensor([pos_cursor - valid_len], device=device, dtype=torch.long))

        if sparse_cursor != len(sparse_info):
            return None
        return position_ids, torch.stack(rope_deltas, dim=0)

    def _build_sparse_position_ids(self, input_ids, image_grid_thw, pad_masks, sparse_info):
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        vision_start_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_start_token_id
        position_ids = torch.zeros((3, input_ids.shape[0], input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
        rope_deltas = []
        sparse_cursor = 0

        for b_idx in range(input_ids.shape[0]):
            seq = input_ids[b_idx]
            mask = pad_masks[b_idx].bool()
            pos_cursor = 0
            idx = 0
            while idx < seq.shape[0] and mask[idx]:
                token = int(seq[idx].item())
                if token == vision_start_id and idx + 1 < seq.shape[0] and int(seq[idx + 1].item()) == image_token_id:
                    position_ids[:, b_idx, idx] = pos_cursor
                    pos_cursor += 1
                    idx += 1
                    img_start = idx
                    while idx < seq.shape[0] and mask[idx] and int(seq[idx].item()) == image_token_id:
                        idx += 1
                    img_end = idx
                    img_len = img_end - img_start
                    if sparse_cursor >= len(sparse_info):
                        raise ValueError("Sparse image position ids are fewer than image spans in prefix")
                    info = sparse_info[sparse_cursor]
                    sparse_cursor += 1
                    sparse_pos = info["position_ids"] if isinstance(info, dict) else info
                    sparse_pos = sparse_pos.to(device=input_ids.device, dtype=torch.long)
                    if sparse_pos.shape[1] != img_len:
                        raise ValueError(
                            f"Sparse image position length mismatch: got {sparse_pos.shape[1]}, expected {img_len}"
                        )
                    image_pos = sparse_pos + pos_cursor
                    position_ids[:, b_idx, img_start:img_end] = image_pos
                    if isinstance(info, dict) and "grid_hw" in info:
                        grid_h, grid_w = info["grid_hw"]
                        pos_cursor += max(int(grid_h), int(grid_w))
                    else:
                        pos_cursor = int(image_pos.max().item()) + 1
                    continue

                position_ids[:, b_idx, idx] = pos_cursor
                pos_cursor += 1
                idx += 1

            if idx < seq.shape[0]:
                position_ids[:, b_idx, idx:] = max(pos_cursor - 1, 0)
            rope_deltas.append(torch.tensor([pos_cursor - int(mask.sum().item())], device=input_ids.device, dtype=torch.long))

        if sparse_cursor != len(sparse_info):
            raise ValueError(
                f"Unused sparse image position ids: used {sparse_cursor}, available {len(sparse_info)}"
            )
        return position_ids, torch.stack(rope_deltas, dim=0)

    def prepare_for_deformable_aggregation(self, feature_maps):
        if not feature_maps:
            return []
        formatted = feature_maps_format(feature_maps)
        formatted[0] = formatted[0].contiguous().float()
        formatted[1] = formatted[1].contiguous().int()
        formatted[2] = formatted[2].contiguous().int()
        return formatted

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
        gt_occ_dense=None,
        hist_traj=None,
        ar_batch=None,
        **kwargs,
    ):
        if gt_ego_fut_trajs is None:
            raise ValueError("gt_ego_fut_trajs is required")

        permute_indices = [0, 2, 1, 4, 5, 3]
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = permute_metas_per_camera_fields(
                kwargs.get("img_metas"), permute_indices, TARGET_SENSOR_ORDER
            )

        _gt_trajs_raw = gt_ego_fut_trajs
        if torch.is_tensor(_gt_trajs_raw) and _gt_trajs_raw.dim() == 4 and _gt_trajs_raw.shape[1] == 1:
            _gt_trajs_raw = _gt_trajs_raw.squeeze(1)

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            hist_traj=hist_traj,
            gt_trajs=_gt_trajs_raw if self.enable_traj_ar else None,
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

        if self.action_training_mode == "query_supervised":
            noise = actions.new_zeros(actions.shape)
            interval_start = None
            time = actions.new_zeros((actions.shape[0],))
            x_t = actions.new_zeros(actions.shape)
            u_t = actions.new_zeros(actions.shape)
        else:
            noise = self.sample_noise(actions.shape, actions.device)
        if self.action_training_mode == "query_supervised":
            pass
        elif self.flow_mode == "meanflow":
            interval_start, time = self.sample_meanflow_interval(actions.shape[0], actions.device)
        else:
            interval_start = None
            time = self.sample_time(actions.shape[0], actions.device)

        if self.action_training_mode != "query_supervised":
            t = time[:, None, None]
            x_t = t * noise + (1 - t) * actions
            u_t = noise - actions

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = self.embed_prefix(batch)
        perception_image_grids = getattr(self.qwen3_vl_with_expert, "_last_perception_image_grids", all_image_grids)
        perception_deepstack_features = getattr(
            self.qwen3_vl_with_expert, "_last_perception_deepstack_features", deepstack_features
        )
        perception_raw_features = getattr(
            self.qwen3_vl_with_expert, "_last_perception_raw_features", raw_features
        )

        if self.driving_deepstack and deepstack_features is not None:
            image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
            _visual_pos_masks = (prefix_input_ids == image_token_id)
            _ds_embeds = deepstack_features
        else:
            _visual_pos_masks = None
            _ds_embeds = None

        source_features = perception_raw_features if self.feature_source == "raw" else perception_deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, perception_image_grids, self.feature_source
        )
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
        perception_condition = self.build_perception_condition(perception_embs, perception_lengths)
        adanorm_skip_enabled = self.use_perception_adanorm and self.skip_perception_shared_attention

        if self.action_training_mode == "query_supervised":
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_action_queries(
                batch, ego_status_pred=None, use_gt_status=True, hist_traj=hist_traj,
            )
        else:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                batch, x_t, time, interval_start=interval_start,
                ego_status_pred=None, use_gt_status=True, hist_traj=hist_traj,
            )
        suffix_embs = self.apply_perception_adanorm(suffix_embs, perception_condition)

        if self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            if not adanorm_skip_enabled:
                perception_embs = perception_embs.to(dtype=torch.bfloat16)

        if self.vlm_grad_scale != 1.0:
            s = self.vlm_grad_scale
            prefix_embs = prefix_embs * 1.0
            prefix_embs.register_hook(lambda g: g * s)
            if not adanorm_skip_enabled:
                perception_embs = perception_embs * 1.0
                perception_embs.register_hook(lambda g: g * s)
            suffix_embs = suffix_embs * 1.0
            suffix_embs.register_hook(lambda g: g * s)

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
        occ_len = perception_lengths['occ']
        ego_len = perception_lengths['ego']
        motion_len = perception_lengths['motion']

        att_mask_input = None
        q_len_rounded = None
        use_adanorm_skip = adanorm_skip_enabled
        if use_adanorm_skip and self.enable_knowledge_insulation:
            raise RuntimeError(
                "perception_adanorm_cfg.skip_shared_attention=True is not supported with "
                "enable_knowledge_insulation=True in this clean FAST ablation path."
            )
        empty_perception_embs = None

        if use_adanorm_skip:
            empty_perception_embs = perception_embs[:, :0].to(dtype=prefix_embs.dtype)
            empty_perception_pad_masks = perception_pad_masks[:, :0]
            empty_perception_att_masks = perception_att_masks[:, :0]

            suffix_pos_ids_1d = max_prefix_pos + suffix_range
            suffix_pos_ids_3d = torch.stack([suffix_pos_ids_1d] * 3, dim=0)
            position_ids = torch.cat([prefix_pos_ids, suffix_pos_ids_3d], dim=2)

            if self.attn_implementation == "flex":
                _prompt_only_len_key = prompt_only_len if self.enable_traj_ar else -1
                _bm_key = (bsz, prefix_len, 0, suffix_len, _prompt_only_len_key, "adanorm_skip")
                if self._cached_block_mask is None or self._cached_block_mask_key != _bm_key:
                    block_mask, q_len_rounded = build_blockmask_unidrive(
                        bsz=bsz,
                        hq=self.qwen3_vl_with_expert.qwen3_vl.config.text_config.num_attention_heads,
                        prefix_len=prefix_len,
                        perception_len=0,
                        suffix_len=suffix_len,
                        device=actions.device,
                        compile_blockmask=True,
                        prompt_only_len=_prompt_only_len_key,
                    )
                    self._cached_block_mask = block_mask
                    self._cached_q_len_rounded = q_len_rounded
                    self._cached_block_mask_key = _bm_key
                att_mask_input = self._cached_block_mask
                q_len_rounded = self._cached_q_len_rounded
            else:
                pad_masks = torch.cat([prefix_pad_masks, empty_perception_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks, empty_perception_att_masks, suffix_att_masks], dim=1)
                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                if batch.traj_answer_ids is not None:
                    for b_idx in range(bsz):
                        p_len = batch.prompt_lens[b_idx]
                        real_ans_len = int(batch.traj_answer_mask[b_idx].sum().item())
                        att_2d_masks[b_idx, prefix_len:, p_len:p_len + real_ans_len] = False
                att_mask_input = self._prepare_attention_masks_4d(att_2d_masks)
        elif self.attn_implementation == "flex" and not self.enable_knowledge_insulation:
            _prompt_only_len_key = prompt_only_len if self.enable_traj_ar else -1
            _bm_key = (bsz, prefix_len, perception_len, suffix_len, _prompt_only_len_key)
            if self._cached_block_mask is None or self._cached_block_mask_key != _bm_key:
                block_mask, q_len_rounded = build_blockmask_unidrive(
                    bsz=bsz,
                    hq=self.qwen3_vl_with_expert.qwen3_vl.config.text_config.num_attention_heads,
                    prefix_len=prefix_len,
                    perception_len=perception_len,
                    suffix_len=suffix_len,
                    device=actions.device,
                    compile_blockmask=True,
                    prompt_only_len=_prompt_only_len_key,
                )
                self._cached_block_mask = block_mask
                self._cached_q_len_rounded = q_len_rounded
                self._cached_block_mask_key = _bm_key
            block_mask = self._cached_block_mask
            q_len_rounded = self._cached_q_len_rounded
            att_mask_input = block_mask
        else:
            pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, perception_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            if batch.traj_answer_ids is not None:
                for b_idx in range(bsz):
                    p_len = batch.prompt_lens[b_idx]
                    real_ans_len = int(batch.traj_answer_mask[b_idx].sum().item())
                    att_2d_masks[b_idx, prefix_len:, p_len:p_len + real_ans_len] = False
            att_mask_input = self._prepare_attention_masks_4d(att_2d_masks)

        stats = {}
        perception_out = None
        skip_initial_action_forward = (
            (self.flow_mode == "meanflow" or (
                self.flow_distill_enabled and self.flow_distill_fm_loss_weight == 0.0
            ))
            and use_adanorm_skip
            and not self.enable_traj_ar
            and not self.with_depth_supervision
        )

        if skip_initial_action_forward:
            prefix_out = prefix_embs
            suffix_out = None
            perception_out = None
        elif self.enable_knowledge_insulation:
            self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "sdpa"
            self.qwen3_vl_with_expert.qwen3_perception_expert.config._attn_implementation = "sdpa"
            self.qwen3_vl_with_expert.qwen3_action_expert.config._attn_implementation = "sdpa"

            def _to_list_tuple(past_kv):
                if hasattr(past_kv, 'key_cache'):
                    return [(past_kv.key_cache[i], past_kv.value_cache[i])
                            for i in range(len(past_kv.key_cache))]
                return list(past_kv)

            def _to_dynamic_cache(kv_list):
                from transformers import DynamicCache
                cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(kv_list):
                    cache.update(k, v, layer_idx)
                return cache

            def _detach_kv(past_kv):
                kv_list = _to_list_tuple(past_kv)
                detached = [(kv[0].detach(), kv[1].detach()) for kv in kv_list]
                return _to_dynamic_cache(detached)

            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            prefix_pos_ids_sliced = position_ids[..., :prefix_len]

            self.gradient_checkpointing_disable()
            (prefix_out, _, _), past_key_values, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=prefix_att_4d,
                position_ids=prefix_pos_ids_sliced,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
                deepstack_visual_embeds=_ds_embeds,
                visual_pos_masks=_visual_pos_masks,
            )

            kv_list = _to_list_tuple(past_key_values)
            kv_list_sliced = [
                (kv[0][:, :, :prompt_only_len, :].detach().to(torch.float16),
                 kv[1][:, :, :prompt_only_len, :].detach().to(torch.float16))
                for kv in kv_list
            ]
            past_key_values = _to_dynamic_cache(kv_list_sliced)

            valid_prompt_mask = torch.zeros(
                (bsz, prompt_only_len), dtype=torch.bool, device=actions.device
            )
            for b_idx in range(bsz):
                valid_prompt_mask[b_idx, :batch.prompt_lens[b_idx]] = True

            prompt_pad_2d_for_percep = valid_prompt_mask[:, None, :].expand(bsz, perception_len, prompt_only_len)
            percep_att_2d = make_att_2d_masks(perception_pad_masks, perception_att_masks)
            percep_full_att_2d = torch.cat([prompt_pad_2d_for_percep, percep_att_2d], dim=2)
            percep_full_att_4d = self._prepare_attention_masks_4d(percep_full_att_2d)
            percep_pos_ids_sliced = position_ids[..., prefix_len:prefix_len + perception_len]

            (_, perception_out, _), past_key_values, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=percep_full_att_4d,
                position_ids=percep_pos_ids_sliced,
                past_key_values=past_key_values,
                inputs_embeds=[None, perception_embs, None],
                use_cache=True,
            )

            past_key_values = _detach_kv(past_key_values)
            kv_full = _to_list_tuple(past_key_values)
            kv_full_fp16 = [
                (kv[0].to(torch.float16), kv[1].to(torch.float16))
                for kv in kv_full
            ]
            past_key_values = _to_dynamic_cache(kv_full_fp16)

            self.gradient_checkpointing_enable()

            cached_len = prompt_only_len + perception_len
            prompt_percep_pad = torch.cat([valid_prompt_mask, perception_pad_masks], dim=1)
            cached_pad_2d = prompt_percep_pad[:, None, :].expand(bsz, suffix_len, cached_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_mixed = torch.cat([cached_pad_2d, suffix_att_2d], dim=2)
            suffix_att_4d = self._prepare_attention_masks_4d(full_att_2d_mixed)
            suffix_pos_ids_sliced = position_ids[..., prefix_len + perception_len:]

            (_, _, suffix_out), _, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=suffix_att_4d,
                position_ids=suffix_pos_ids_sliced,
                past_key_values=past_key_values,
                inputs_embeds=[None, None, suffix_embs],
                use_cache=False,
            )
        else:
            def forward_func(prefix_embs, perception_embs, suffix_embs, att_mask, position_ids, _unused_return_middle_layers, q_len_rnd, ds_embeds, vis_pos_masks):
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

            (outputs_embeds, _middle_layer_outs_unused) = self._apply_checkpoint(
                forward_func,
                prefix_embs,
                empty_perception_embs if use_adanorm_skip else perception_embs,
                suffix_embs,
                att_mask_input,
                position_ids,
                None,
                q_len_rounded,
                _ds_embeds,
                _visual_pos_masks,
            )
            prefix_out, perception_out, suffix_out = outputs_embeds
            if use_adanorm_skip:
                perception_out = None

        loss_traj_ar = prefix_embs.sum() * 0.0
        if self.enable_traj_ar and self.train_vlm and batch.traj_labels is not None:
            lm_head = self.qwen3_vl_with_expert.qwen3_vl.lm_head
            answer_len = batch.traj_labels.shape[1]
            loss_sum = prefix_embs.sum() * 0.0
            for i in range(bsz):
                p_len = batch.prompt_lens[i]
                pred_hidden_i = prefix_out[i, p_len - 1: p_len - 1 + answer_len, :]
                pred_logits_i = lm_head(pred_hidden_i.to(lm_head.weight.dtype))
                loss_sum = loss_sum + F.cross_entropy(
                    pred_logits_i,
                    batch.traj_labels[i],
                    ignore_index=-100,
                )
            loss_traj_ar = (loss_sum / bsz) * self.traj_ar_loss_weight

        loss_depth_val = self.compute_depth_loss(
            prefix_embs, prefix_out, prefix_input_ids, all_image_grids, gt_depth, focal, bsz
        )

        d0 = 0
        d1 = d0 + det_len
        m0 = d1
        m1 = m0 + map_len
        o0 = m1
        o1 = o0 + occ_len
        e0 = o1
        e1 = e0 + ego_len
        t0 = e1
        t1 = t0 + motion_len
        distill_freeze_condition_path = self.flow_distill_enabled and self.flow_distill_freeze_condition_modules

        if self.use_perception_adanorm and self.skip_perception_shared_attention:
            if not distill_freeze_condition_path:
                target_dtype = self.det_proj.weight.dtype
                vlm_enhanced = {
                    'det_feat': stage1_outs['det_instance_feature'].to(target_dtype),
                    'map_feat': stage1_outs['map_instance_feature'].to(target_dtype),
                    'ego_feat': stage1_outs['ego_instance_feature'].to(target_dtype),
                }

                if self.with_motion and 'motion_token' in stage1_outs:
                    vlm_enhanced['motion_feat'] = stage1_outs['motion_token'].to(target_dtype)

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
                    'gt_agent_fut_trajs': gt_agent_fut_trajs,
                    'gt_agent_fut_masks': gt_agent_fut_masks,
                }
                perception_losses = self.unified_decoder.loss(stage1_outs, stage2_outs, perception_data)
                stats.update(perception_losses)
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=actions.device, dtype=torch.float32)
            ego_out = None
        elif perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            occ_out = perception_out[:, o0:o1]
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]

            target_dtype = self.det_proj.weight.dtype
            proj_dtype = self.det_proj.weight.dtype
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
                'gt_agent_fut_trajs': gt_agent_fut_trajs,
                'gt_agent_fut_masks': gt_agent_fut_masks,
            }
            if not distill_freeze_condition_path:
                perception_losses = self.unified_decoder.loss(stage1_outs, stage2_outs, perception_data)
                stats.update(perception_losses)
        else:
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=actions.device, dtype=torch.float32)
            ego_out = None

        if self.action_training_mode == "query_supervised":
            if suffix_out is None:
                raise RuntimeError("query_supervised action mode requires suffix_out from action expert")
            query_hidden = suffix_out[:, -self.action_query_num_queries:].to(dtype=self.action_query_decoder[0].weight.dtype)
            pred_x0 = self.action_query_decoder(query_hidden.flatten(1))
            pred_x0 = pred_x0.view(bsz, self.action_horizon, self.action_dim).to(dtype=torch.float32)
            loss_model_out = None
            pred_x_gt = pred_x0
            pred_x0_for_loss = pred_x0
            pred_x0_for_aux = pred_x0
        elif self.flow_mode == "meanflow":
            loss_model_out = None
            pred_x_gt = None
            pred_x0_for_loss = None
            pred_x0_for_aux = None
        elif self.flow_distill_enabled and self.flow_distill_fm_loss_weight == 0.0:
            loss_model_out = x_t.new_zeros(x_t.shape, dtype=torch.float32)
            pred_x_gt = None
            pred_x0_for_loss = None
            pred_x0_for_aux = None
        else:
            suffix_out_float = suffix_out[:, -self.action_horizon:].to(dtype=torch.float32)
            model_out = self._apply_checkpoint(self.action_out_proj, suffix_out_float)

            t_exp = time[:, None, None].to(dtype=torch.float32)
            loss_model_out = model_out
            pred_x_gt = x_t.to(dtype=torch.float32) - t_exp * model_out
            pred_x0_for_loss = pred_x_gt
            pred_x0_for_aux = pred_x_gt

        if self.flow_distill_enabled:
            if self.flow_mode != "flow_matching":
                raise RuntimeError("Flow distillation is only supported with flow_mode='flow_matching'")
            if self.enable_knowledge_insulation:
                raise RuntimeError("Flow distillation is not supported with enable_knowledge_insulation=True")
            if self.flow_distill_teacher_steps <= 0:
                raise RuntimeError("flow_distill_teacher_steps must be positive")
            self._ensure_flow_distill_teacher_initialized()

            def _build_flow_distill_cache():
                prefix_att_2d_distill = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_att_4d_distill = self._prepare_attention_masks_4d(prefix_att_2d_distill).to(prefix_embs.dtype)

                (_, _, _), distill_past_key_values, _ = self.qwen3_vl_with_expert.forward(
                    attention_mask=prefix_att_4d_distill,
                    position_ids=prefix_pos_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None, None],
                    use_cache=True,
                    deepstack_visual_embeds=_ds_embeds,
                    visual_pos_masks=_visual_pos_masks,
                )

                if self.use_perception_adanorm and self.skip_perception_shared_attention:
                    return distill_past_key_values

                prefix_pad_2d_distill = prefix_pad_masks[:, None, :].expand(bsz, perception_len, prefix_len)
                perception_att_2d_distill = make_att_2d_masks(perception_pad_masks, perception_att_masks)
                perception_full_att_2d_distill = torch.cat(
                    [prefix_pad_2d_distill, perception_att_2d_distill],
                    dim=2,
                )
                perception_full_att_4d_distill = self._prepare_attention_masks_4d(
                    perception_full_att_2d_distill
                ).to(prefix_embs.dtype)
                perception_pos_ids_distill = position_ids[..., prefix_len:prefix_len + perception_len]

                (_, _, _), distill_past_key_values, _ = self.qwen3_vl_with_expert.forward(
                    attention_mask=perception_full_att_4d_distill,
                    position_ids=perception_pos_ids_distill,
                    past_key_values=distill_past_key_values,
                    inputs_embeds=[None, perception_embs, None],
                    use_cache=True,
                )
                return distill_past_key_values

            was_checkpointing = self.gradient_checkpointing_enabled
            try:
                if was_checkpointing:
                    self.gradient_checkpointing_disable()
                if self.flow_distill_freeze_condition_modules:
                    with torch.no_grad():
                        distill_past_key_values = _build_flow_distill_cache()
                else:
                    distill_past_key_values = _build_flow_distill_cache()
            finally:
                if was_checkpointing:
                    self.gradient_checkpointing_enable()

            if self.use_perception_adanorm and self.skip_perception_shared_attention:
                cached_pad_masks_distill = prefix_pad_masks
                max_cached_pos_distill = max_prefix_pos
            else:
                cached_pad_masks_distill = torch.cat([prefix_pad_masks, perception_pad_masks], dim=1)
                max_cached_pos_distill = max_perception_pos

            teacher_perception_condition = perception_condition
            if self.flow_distill_teacher_perception_condition_mlp is not None:
                with torch.no_grad():
                    teacher_perception_condition = self.build_perception_condition(
                        perception_embs,
                        perception_lengths,
                        condition_mlp_override=self.flow_distill_teacher_perception_condition_mlp,
                    )

            bsz_distill = actions.shape[0]
            teacher_x = noise.to(dtype=torch.float32)
            teacher_dt = -1.0 / float(self.flow_distill_teacher_steps)
            teacher_time = torch.tensor(1.0, dtype=torch.float32, device=actions.device)
            was_checkpointing = self.gradient_checkpointing_enabled
            try:
                if was_checkpointing:
                    self.gradient_checkpointing_disable()
                self.flow_distill_teacher_action_expert.eval()
                self.flow_distill_teacher_action_out_proj.eval()
                with torch.no_grad():
                    while teacher_time >= (-teacher_dt / 2.0):
                        expanded_teacher_time = teacher_time.expand(bsz_distill)
                        teacher_v = self._denoise_step(
                            batch,
                            cached_pad_masks_distill,
                            distill_past_key_values,
                            teacher_x.to(dtype=prefix_embs.dtype),
                            expanded_teacher_time.to(dtype=prefix_embs.dtype),
                            max_cached_pos_distill,
                            ego_status_pred=None,
                            use_gt_status=True,
                            hist_traj=hist_traj,
                            perception_condition=teacher_perception_condition,
                            action_expert_override=self.flow_distill_teacher_action_expert,
                            action_out_proj_override=self.flow_distill_teacher_action_out_proj,
                            action_in_proj_override=self.flow_distill_teacher_action_in_proj,
                            status_mlp_override=self.flow_distill_teacher_status_mlp,
                            hist_traj_encoder_override=self.flow_distill_teacher_hist_traj_encoder,
                            action_time_mlp_in_override=self.flow_distill_teacher_action_time_mlp_in,
                            action_time_mlp_out_override=self.flow_distill_teacher_action_time_mlp_out,
                            perception_suffix_norm_override=self.flow_distill_teacher_perception_suffix_norm,
                            perception_suffix_scale_shift_override=self.flow_distill_teacher_perception_suffix_scale_shift,
                        )
                        teacher_x = teacher_x + teacher_dt * teacher_v.to(dtype=torch.float32)
                        teacher_time = teacher_time + teacher_dt
            finally:
                if was_checkpointing:
                    self.gradient_checkpointing_enable()

            student_time = torch.full(
                (bsz_distill,),
                self.flow_distill_student_time,
                dtype=torch.float32,
                device=actions.device,
            )
            student_v = self._denoise_step(
                batch,
                cached_pad_masks_distill,
                distill_past_key_values,
                noise.to(dtype=prefix_embs.dtype),
                student_time.to(dtype=prefix_embs.dtype),
                max_cached_pos_distill,
                ego_status_pred=None,
                use_gt_status=True,
                hist_traj=hist_traj,
                perception_condition=perception_condition,
            )
            student_x0 = noise.to(dtype=torch.float32) - self.flow_distill_student_time * student_v.to(dtype=torch.float32)
            teacher_x0 = teacher_x.detach()
            distill_loss = self._flow_distill_loss(student_x0, teacher_x0, gt_ego_fut_masks)
            stats["loss_flow_distill"] = self.flow_distill_loss_weight * distill_loss
            stats["flow_distill_teacher_l1"] = F.l1_loss(student_x0.detach(), teacher_x0, reduction="mean")
            if self.flow_distill_gt_loss_weight > 0.0:
                gt_anchor_loss = self._flow_distill_loss(student_x0, actions.to(dtype=torch.float32), gt_ego_fut_masks)
                stats["loss_flow_distill_gt"] = self.flow_distill_gt_loss_weight * gt_anchor_loss
            if (
                self.flow_distill_waypoint_loss_weight > 0.0
                or self.flow_distill_waypoint_gt_loss_weight > 0.0
                or self.flow_distill_final_waypoint_loss_weight > 0.0
                or self.flow_distill_final_waypoint_gt_loss_weight > 0.0
            ):
                student_wpts = torch.cumsum(
                    self.denorm_delta(student_x0[..., :2].to(dtype=torch.float32)),
                    dim=1,
                )
                teacher_wpts = torch.cumsum(
                    self.denorm_delta(teacher_x0[..., :2].to(dtype=torch.float32)),
                    dim=1,
                ).detach()
                gt_wpts = torch.cumsum(
                    self.denorm_delta(actions[..., :2].to(dtype=torch.float32)),
                    dim=1,
                )
                if self.flow_distill_waypoint_loss_weight > 0.0:
                    waypoint_distill_loss = self._flow_distill_loss(
                        student_wpts, teacher_wpts, gt_ego_fut_masks
                    )
                    stats["loss_flow_distill_waypoint"] = (
                        self.flow_distill_waypoint_loss_weight * waypoint_distill_loss
                    )
                if self.flow_distill_waypoint_gt_loss_weight > 0.0:
                    waypoint_gt_loss = self._flow_distill_loss(
                        student_wpts, gt_wpts, gt_ego_fut_masks
                    )
                    stats["loss_flow_distill_waypoint_gt"] = (
                        self.flow_distill_waypoint_gt_loss_weight * waypoint_gt_loss
                    )
                if self.flow_distill_final_waypoint_loss_weight > 0.0:
                    final_mask = gt_ego_fut_masks[:, -1:] if gt_ego_fut_masks is not None else None
                    final_waypoint_distill_loss = self._flow_distill_loss(
                        student_wpts[:, -1:], teacher_wpts[:, -1:], final_mask
                    )
                    stats["loss_flow_distill_final_waypoint"] = (
                        self.flow_distill_final_waypoint_loss_weight * final_waypoint_distill_loss
                    )
                if self.flow_distill_final_waypoint_gt_loss_weight > 0.0:
                    final_mask = gt_ego_fut_masks[:, -1:] if gt_ego_fut_masks is not None else None
                    final_waypoint_gt_loss = self._flow_distill_loss(
                        student_wpts[:, -1:], gt_wpts[:, -1:], final_mask
                    )
                    stats["loss_flow_distill_final_waypoint_gt"] = (
                        self.flow_distill_final_waypoint_gt_loss_weight * final_waypoint_gt_loss
                    )
            if pred_x0_for_aux is None:
                pred_x0_for_aux = student_x0

        if self.action_training_mode != "query_supervised" and self.flow_mode == "meanflow":
            if interval_start is None:
                raise RuntimeError("flow_mode='meanflow' requires interval_start")
            if self.enable_knowledge_insulation:
                raise RuntimeError("flow_mode='meanflow' is not supported with enable_knowledge_insulation=True")
            if not use_adanorm_skip:
                raise RuntimeError(
                    "flow_mode='meanflow' in this FAST path requires "
                    "perception_adanorm_cfg.enabled=True and skip_shared_attention=True"
                )

            flow_velocity = u_t.to(dtype=torch.float32)
            r_for_jvp = interval_start.to(dtype=torch.float32)
            t_for_jvp = time.to(dtype=torch.float32)
            x_for_jvp = x_t.to(dtype=torch.float32)

            def meanflow_forward(z_t_arg, r_arg, t_arg, return_context=False):
                suffix_embs_mf, _, _ = self.embed_suffix(
                    batch,
                    z_t_arg,
                    t_arg,
                    interval_start=r_arg,
                    ego_status_pred=None,
                    use_gt_status=True,
                    hist_traj=hist_traj,
                )
                suffix_embs_mf = self.apply_perception_adanorm(suffix_embs_mf, perception_condition)
                if self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                    suffix_embs_mf = suffix_embs_mf.to(dtype=torch.bfloat16)

                outputs_embeds_mf, _, _ = self.qwen3_vl_with_expert.forward(
                    attention_mask=att_mask_input,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[
                        prefix_embs,
                        empty_perception_embs if use_adanorm_skip else perception_embs,
                        suffix_embs_mf,
                    ],
                    use_cache=False,
                    return_middle_layers=None,
                    q_len_rounded=q_len_rounded,
                    deepstack_visual_embeds=_ds_embeds,
                    visual_pos_masks=_visual_pos_masks,
                )
                suffix_out_mf = outputs_embeds_mf[2][:, -self.action_horizon:].to(dtype=torch.float32)
                u_mf = self.action_out_proj(suffix_out_mf)
                if return_context:
                    return u_mf, suffix_out_mf
                return u_mf

            def meanflow_u_func(z_t_arg, r_arg, t_arg):
                return meanflow_forward(z_t_arg, r_arg, t_arg, return_context=False)

            jvp_inputs = (x_for_jvp, r_for_jvp, t_for_jvp)
            jvp_tangents = (
                flow_velocity,
                torch.zeros_like(r_for_jvp),
                torch.ones_like(t_for_jvp),
            )
            if self.meanflow_jvp_create_graph:
                meanflow_u, dudt = torch.autograd.functional.jvp(
                    meanflow_u_func,
                    jvp_inputs,
                    jvp_tangents,
                    create_graph=True,
                    strict=False,
                )
            else:
                # MeanFlow uses the JVP correction as a stop-gradient target term.
                # Compute u with gradients, then compute only the detached JVP
                # correction in no_grad. This is a memory-saving approximation
                # that keeps batch4 feasible on the current VLM path.
                meanflow_u = meanflow_u_func(x_for_jvp, r_for_jvp, t_for_jvp)
                was_checkpointing = self.gradient_checkpointing_enabled
                try:
                    if was_checkpointing:
                        self.gradient_checkpointing_disable()
                    with torch.no_grad():
                        with sdpa_kernel(SDPBackend.MATH):
                            _, dudt = torch.func.jvp(
                                meanflow_u_func,
                                tuple(arg.detach() for arg in jvp_inputs),
                                tuple(arg.detach() for arg in jvp_tangents),
                                strict=False,
                            )
                finally:
                    if was_checkpointing:
                        self.gradient_checkpointing_enable()

            dt_exp = (time - interval_start).clamp_min(0.0)[:, None, None].to(dtype=torch.float32)
            loss_model_out = meanflow_u.to(dtype=torch.float32) + dt_exp * dudt.to(dtype=torch.float32).detach()
            u_t = flow_velocity
            if self.enable_meanflow_arm and self.meanflow_arm_loss_weight > 0.0:
                endpoint_r = torch.zeros_like(t_for_jvp)
                endpoint_t = torch.ones_like(t_for_jvp)
                was_checkpointing = self.gradient_checkpointing_enabled
                try:
                    if was_checkpointing:
                        self.gradient_checkpointing_disable()
                    with torch.no_grad():
                        endpoint_u, endpoint_context = meanflow_forward(
                            noise.to(dtype=torch.float32),
                            endpoint_r,
                            endpoint_t,
                            return_context=True,
                        )
                        endpoint_proposal = noise.to(dtype=torch.float32) - endpoint_u.to(dtype=torch.float32)
                finally:
                    if was_checkpointing:
                        self.gradient_checkpointing_enable()

                arm_pred_x0 = self._apply_meanflow_arm(
                    endpoint_proposal.detach(),
                    endpoint_context.detach(),
                    perception_condition.detach() if perception_condition is not None else None,
                )
                arm_pred_wpts = torch.cumsum(self.denorm_delta(arm_pred_x0.to(dtype=torch.float32)), dim=1)
                arm_gt_wpts = torch.cumsum(
                    self.denorm_delta(actions.to(dtype=torch.float32)).detach(),
                    dim=1,
                )
                meanflow_arm_l1 = self._masked_l1(
                    F.l1_loss(arm_pred_wpts, arm_gt_wpts, reduction="none"),
                    gt_ego_fut_masks,
                )
                stats["meanflow_arm_l1"] = meanflow_arm_l1.detach()
                stats["loss_meanflow_arm"] = self.meanflow_arm_loss_weight * meanflow_arm_l1

        if self.action_training_mode == "query_supervised":
            planning_loss = self._action_query_supervised_loss(
                pred_x0_for_loss,
                actions.to(dtype=torch.float32),
                gt_ego_fut_masks,
            )
            stats["action_query_supervised"] = planning_loss.detach()
        elif self.flow_distill_enabled and self.flow_distill_fm_loss_weight == 0.0:
            planning_loss = actions.new_zeros(())
        elif self.loss_planning is not None:
            planning_loss = self.loss_planning(
                loss_model_out, u_t, time,
                gt_ego_fut_masks=gt_ego_fut_masks,
                pred_x0=pred_x0_for_loss,
                gt_x0=actions,
            )
        else:
            planning_loss = torch.tensor(0.0, device=actions.device)
        if self.flow_distill_enabled:
            stats["loss_flow_matching_anchor"] = planning_loss.detach()
            if self.flow_distill_fm_loss_weight != 0.0:
                planning_loss = planning_loss * self.flow_distill_fm_loss_weight

        stats["loss_planning"] = planning_loss

        _pred_abs = None
        if pred_x0_for_aux is not None and (
            self.collision_loss_fn is not None or
            self.map_bound_loss_fn is not None or
            self.map_dir_loss_fn is not None
        ):
            _pred_deltas = self.denorm_delta(pred_x0_for_aux[..., :2].float())
            _pred_abs = torch.cumsum(_pred_deltas, dim=1)

        if self.collision_loss_fn is not None and gt_bboxes_3d is not None and _pred_abs is not None:
            loss_collision = self.collision_loss_fn(_pred_abs, gt_bboxes_3d, gt_ego_fut_masks)
            stats["loss_collision"] = self.collision_loss_weight * loss_collision

        if self.map_bound_loss_fn is not None and gt_map_pts is not None and _pred_abs is not None:
            _gt_map_pts = gt_map_pts.data if hasattr(gt_map_pts, 'data') else gt_map_pts
            _gt_map_labels = gt_map_labels.data if hasattr(gt_map_labels, 'data') else gt_map_labels
            loss_map_bound = self.map_bound_loss_fn(
                _pred_abs, _gt_map_pts, _gt_map_labels, gt_ego_fut_masks)
            stats["loss_map_bound"] = self.map_bound_loss_weight * loss_map_bound

        if self.map_dir_loss_fn is not None and gt_map_pts is not None and _pred_abs is not None:
            _gt_map_pts = gt_map_pts.data if hasattr(gt_map_pts, 'data') else gt_map_pts
            _gt_map_labels = gt_map_labels.data if hasattr(gt_map_labels, 'data') else gt_map_labels
            loss_map_dir = self.map_dir_loss_fn(
                _pred_abs, _gt_map_pts, _gt_map_labels, gt_ego_fut_masks)
            stats["loss_map_dir"] = self.map_dir_loss_weight * loss_map_dir

        if distill_freeze_condition_path:
            loss_occ = actions.new_zeros(())
        elif gt_occ_dense is not None and perception_out is not None:
            occ_target = gt_occ_dense.permute(0, 3, 1, 2).long()
            occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            loss_occ = F.cross_entropy(occ_logits, occ_target)
        else:
            loss_occ = perception_embs.sum() * 0.0
        stats["loss_occ"] = loss_occ

        if distill_freeze_condition_path:
            loss_motion = actions.new_zeros(())
        else:
            loss_motion = stats.get("loss_motion", perception_embs.sum() * 0.0)

        ar_loss_total = torch.tensor(0.0, device=planning_loss.device)
        if self.train_vlm:
            if ar_batch is not None:
                if getattr(self.qwen3_vl_with_expert, "_image_token_compression_enabled")():
                    raise ValueError(
                        "ar_batch auxiliary VLM loss is not supported with VisionZip image token "
                        "compression in this path. forward_ar_batch() calls the base qwen3_vl "
                        "forward and would use uncompressed full image tokens, which would make "
                        "the auxiliary AR objective inconsistent with the compressed planning prefix."
                    )
                ar_outputs = self.forward_ar_batch(ar_batch)
                ar_loss_total = ar_outputs['loss_ar']
                stats['loss_vlm_raw'] = ar_outputs['loss_vlm_raw']
                stats['ar_planning_ratio'] = (ar_outputs['loss_vlm_raw'] / (planning_loss.detach() + 1e-6)).item()
            else:
                stats['loss_vlm_raw'] = torch.tensor(0.0, device=planning_loss.device)
                stats['ar_planning_ratio'] = 0.0

        stats["loss_planning"] = planning_loss
        stats["loss_occ"] = self.occ_loss_weight * loss_occ
        stats["loss_depth"] = loss_depth_val

        stats.setdefault("loss_motion", loss_motion)

        if self.train_vlm:
            stats["loss_ar"] = ar_loss_total

        if self.enable_traj_ar and self.train_vlm:
            stats["loss_traj_ar"] = loss_traj_ar

        stats.pop('loss_perception', None)
        stats.pop('loss', None)

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
        hist_traj=None,
        use_gt_ego_status: bool = False,
        **kwargs,
    ):
        permute_indices = [0, 2, 1, 4, 5, 3]
        camera_inputs_permuted = bool(kwargs.pop("camera_inputs_permuted", False))
        prebuilt_batch = kwargs.pop("prebuilt_batch", None)
        prebuilt_prefix_outputs = kwargs.pop("prebuilt_prefix_outputs", None)
        prebuilt_feature_maps_daf = kwargs.pop("prebuilt_feature_maps_daf", None)
        prebuilt_stage1_outs = kwargs.pop("prebuilt_stage1_outs", None)
        prebuilt_perception_outputs = kwargs.pop("prebuilt_perception_outputs", None)
        prebuilt_perception_condition = kwargs.pop("prebuilt_perception_condition", None)
        prebuilt_prefix_pos_ids = kwargs.pop("prebuilt_prefix_pos_ids", None)
        prebuilt_max_prefix_pos = kwargs.pop("prebuilt_max_prefix_pos", None)
        profile_core_compute_only = bool(kwargs.pop("profile_core_compute_only", False))
        if not camera_inputs_permuted:
            if projection_mat is not None:
                projection_mat = projection_mat[:, permute_indices]
            if image_wh is not None:
                image_wh = image_wh[:, permute_indices]

            if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
                kwargs["img_metas"] = permute_metas_per_camera_fields(
                    kwargs.get("img_metas"), permute_indices, TARGET_SENSOR_ORDER
                )

        if prebuilt_batch is None:
            batch = self._build_driving_batch(
                img=img,
                command=gt_ego_fut_cmd,
                ego_status=ego_status,
                hist_traj=hist_traj,
                camera_inputs_permuted=camera_inputs_permuted,
            )
        else:
            batch = prebuilt_batch
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        num_steps = int(self.num_sample_steps if num_steps is None else num_steps)

        if noise is None and self.action_training_mode == "query_supervised":
            noise = torch.zeros((bsz, self.action_horizon, self.action_dim), dtype=torch.float32, device=device)
        elif noise is None:
            noise = self.sample_noise((bsz, self.action_horizon, self.action_dim), device)

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        if prebuilt_prefix_outputs is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = self.embed_prefix(batch)
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = prebuilt_prefix_outputs
        perception_image_grids = getattr(self.qwen3_vl_with_expert, "_last_perception_image_grids", all_image_grids)
        perception_deepstack_features = getattr(
            self.qwen3_vl_with_expert, "_last_perception_deepstack_features", deepstack_features
        )
        perception_raw_features = getattr(
            self.qwen3_vl_with_expert, "_last_perception_raw_features", raw_features
        )

        if prefix_embs.dtype != dtype:
            prefix_embs = prefix_embs.to(dtype)

        if prebuilt_feature_maps_daf is None:
            source_features = perception_raw_features if self.feature_source == "raw" else perception_deepstack_features
            feature_maps = self.project_and_reshape_features(
                source_features, bsz, perception_image_grids, self.feature_source
            )

            feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
            if not self.feat_grad:
                feature_maps_daf = [x.detach() for x in feature_maps_daf]
        else:
            feature_maps_daf = prebuilt_feature_maps_daf

        head_dtype = next(self.unified_decoder.parameters()).dtype
        head_device = device

        perception_metas = {
            'img_metas': kwargs.get('img_metas'),
            'timestamp': timestamp,
            'projection_mat': projection_mat.to(device=head_device, dtype=head_dtype) if projection_mat is not None else None,
            'image_wh': image_wh.to(device=head_device, dtype=head_dtype) if image_wh is not None else None,
        }

        if prebuilt_stage1_outs is None:
            stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)
        else:
            stage1_outs = prebuilt_stage1_outs

        if prebuilt_perception_outputs is None:
            perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
                bsz, device, dtype, stage1_outs
            )
        else:
            perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = prebuilt_perception_outputs
        if prebuilt_perception_condition is None:
            perception_condition = self.build_perception_condition(perception_embs, perception_lengths)
        else:
            perception_condition = prebuilt_perception_condition

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d).to(dtype)

        if prebuilt_prefix_pos_ids is None:
            prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        else:
            prefix_pos_ids = prebuilt_prefix_pos_ids
        if prebuilt_max_prefix_pos is None:
            max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values
        else:
            max_prefix_pos = prebuilt_max_prefix_pos

        _ds_embeds = deepstack_features if self.driving_deepstack and deepstack_features is not None else None
        _vis_masks = (
            (prefix_input_ids == self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id)
            if _ds_embeds is not None else None
        )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = self._inference_attn_impl

        _, past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
            deepstack_visual_embeds=_ds_embeds,
            visual_pos_masks=_vis_masks,
        )

        perception_len = perception_embs.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        perception_range = torch.arange(1, perception_len + 1, device=device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        perception_out = None
        if not (self.use_perception_adanorm and self.skip_perception_shared_attention):
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, perception_len, prefix_len)
            perception_att_2d = make_att_2d_masks(perception_pad_masks, perception_att_masks)
            perception_full_att_2d = torch.cat([prefix_pad_2d, perception_att_2d], dim=2)
            perception_full_att_2d_4d = self._prepare_attention_masks_4d(perception_full_att_2d).to(dtype)
            perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

            self.qwen3_vl_with_expert.qwen3_perception_expert.config._attn_implementation = self._inference_attn_impl

            (_, perception_out, _), past_key_values, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=perception_full_att_2d_4d,
                position_ids=perception_pos_ids_3d,
                past_key_values=past_key_values,
                inputs_embeds=[None, perception_embs, None],
                use_cache=True,
            )

        det_len = perception_lengths['det']
        map_len = perception_lengths['map']
        occ_len = perception_lengths['occ']
        ego_len = perception_lengths['ego']
        motion_len = perception_lengths['motion']

        d0 = 0
        d1 = d0 + det_len
        m0 = d1
        m1 = m0 + map_len
        o0 = m1
        o1 = o0 + occ_len
        e0 = o1
        e1 = e0 + ego_len
        t0 = e1
        t1 = t0 + motion_len

        stage2_outs = None

        if self.use_perception_adanorm and self.skip_perception_shared_attention:
            target_dtype = self.det_proj.weight.dtype
            vlm_enhanced = {
                'det_feat': stage1_outs['det_instance_feature'].to(target_dtype),
                'map_feat': stage1_outs['map_instance_feature'].to(target_dtype),
                'ego_feat': stage1_outs['ego_instance_feature'].to(target_dtype),
            }
            if self.with_motion and 'motion_token' in stage1_outs:
                vlm_enhanced['motion_feat'] = stage1_outs['motion_token'].to(target_dtype)

            stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)
            if profile_core_compute_only:
                det_result, map_result = None, None
            else:
                det_result, map_result = self.unified_decoder.post_process(stage2_outs)
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=device, dtype=torch.float32)
            ego_out = None
        elif perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            occ_out = perception_out[:, o0:o1]
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]

            target_dtype = self.det_proj.weight.dtype
            proj_dtype = self.det_proj.weight.dtype
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
            if profile_core_compute_only:
                det_result, map_result = None, None
            else:
                det_result, map_result = self.unified_decoder.post_process(stage2_outs)
        else:
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=device, dtype=torch.float32)
            ego_out = None
            det_result = None
            map_result = None

        if self.use_perception_adanorm and self.skip_perception_shared_attention:
            cached_pad_masks = prefix_pad_masks
        else:
            cached_pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks], dim=1)
        dt_val = -1.0 / num_steps
        dt_val = torch.tensor(dt_val, dtype=torch.float32, device=device)

        x_t = noise
        time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)

        if use_gt_ego_status:
            ego_status_pred = None
        elif stage2_outs and 'ego_status_list' in stage2_outs and len(stage2_outs['ego_status_list']) > 0:
            ego_status_pred = stage2_outs['ego_status_list'][-1].squeeze(1).to(torch.float32)
        else:
            ego_status_pred = x_t.new_zeros((bsz, self.ego_status_dim), dtype=torch.float32)
        if stage2_outs and 'ego_status_list' in stage2_outs and len(stage2_outs['ego_status_list']) > 0:
            ego_status_eval = stage2_outs['ego_status_list'][-1].squeeze(1).to(torch.float32)
        else:
            ego_status_eval = x_t.new_zeros((bsz, self.ego_status_dim), dtype=torch.float32)

        if perception_len > 0:
            max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values
        else:
            max_perception_pos = max_prefix_pos
        max_denoise_pos = (
            max_prefix_pos
            if self.use_perception_adanorm and self.skip_perception_shared_attention
            else max_perception_pos
        )

        if self.action_training_mode == "query_supervised":
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_action_queries(
                batch,
                ego_status_pred=ego_status_pred,
                use_gt_status=use_gt_ego_status,
                hist_traj=hist_traj,
            )
            suffix_embs = self.apply_perception_adanorm(suffix_embs, perception_condition)
            if suffix_embs.dtype != dtype:
                suffix_embs = suffix_embs.to(dtype)

            suffix_len = suffix_pad_masks.shape[1]
            cached_len = cached_pad_masks.shape[1]
            cached_pad_2d = cached_pad_masks[:, None, :].expand(bsz, suffix_len, cached_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([cached_pad_2d, suffix_att_2d], dim=2)
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d).to(dtype)
            suffix_range = torch.arange(1, suffix_len + 1, device=device).view(1, -1).expand(bsz, -1)
            suffix_pos_ids_1d = max_denoise_pos + suffix_range
            suffix_pos_ids = torch.stack([suffix_pos_ids_1d] * 3, dim=0)

            self.qwen3_vl_with_expert.qwen3_action_expert.config._attn_implementation = self._inference_attn_impl
            outputs_embeds, _, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=suffix_pos_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, None, suffix_embs],
                use_cache=False,
            )
            query_hidden = outputs_embeds[2][:, -self.action_query_num_queries:]
            query_hidden = query_hidden.to(dtype=self.action_query_decoder[0].weight.dtype)
            x_t = self.action_query_decoder(query_hidden.flatten(1))
            x_t = x_t.view(bsz, self.action_horizon, self.action_dim).to(dtype=torch.float32)
        elif self.flow_mode == "meanflow":
            interval_start = torch.zeros((bsz,), dtype=torch.float32, device=device)
            expanded_time = torch.ones((bsz,), dtype=torch.float32, device=device)
            denoise_out = self._denoise_step(
                batch, cached_pad_masks, past_key_values, x_t.to(dtype), expanded_time.to(dtype),
                max_denoise_pos, ego_status_pred=ego_status_pred, use_gt_status=use_gt_ego_status,
                hist_traj=hist_traj, perception_condition=perception_condition,
                interval_start=interval_start.to(dtype),
                return_context=self.enable_meanflow_arm,
            )
            if self.enable_meanflow_arm:
                v_t, action_context = denoise_out
            else:
                v_t = denoise_out
            step = (expanded_time - interval_start).view(bsz, 1, 1).to(dtype=torch.float32)
            x_t = x_t - step * v_t
            if self.enable_meanflow_arm:
                x_t = self._apply_meanflow_arm(x_t, action_context, perception_condition)
        else:
            while time_tensor >= -dt_val / 2:
                expanded_time = time_tensor.expand(bsz)
                v_t = self._denoise_step(
                    batch, cached_pad_masks, past_key_values, x_t.to(dtype), expanded_time.to(dtype),
                    max_denoise_pos, ego_status_pred=ego_status_pred, use_gt_status=use_gt_ego_status,
                    hist_traj=hist_traj, perception_condition=perception_condition,
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
            "ego_status_pred": ego_status_eval,
        }

    @torch.no_grad()
    def prepare_partial_action_graph_state(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        hist_traj=None,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        use_gt_ego_status: bool = False,
        **kwargs,
    ) -> Dict[str, object]:
        """Prepare graph-unsafe full-output context outside a partial action CUDA graph."""
        permute_indices = [0, 2, 1, 4, 5, 3]
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]
        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = permute_metas_per_camera_fields(
                kwargs.get("img_metas"), permute_indices, TARGET_SENSOR_ORDER
            )

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            hist_traj=hist_traj,
            camera_inputs_permuted=False,
        )
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        num_steps = int(self.num_sample_steps if num_steps is None else num_steps)

        if noise is None:
            noise = self.sample_noise((bsz, self.action_horizon, self.action_dim), device)

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"
        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, _ = self.embed_prefix(batch)
        perception_image_grids = getattr(self.qwen3_vl_with_expert, "_last_perception_image_grids", all_image_grids)
        perception_deepstack_features = getattr(
            self.qwen3_vl_with_expert, "_last_perception_deepstack_features", deepstack_features
        )
        perception_raw_features = getattr(
            self.qwen3_vl_with_expert, "_last_perception_raw_features", raw_features
        )

        if prefix_embs.dtype != dtype:
            prefix_embs = prefix_embs.to(dtype)

        source_features = perception_raw_features if self.feature_source == "raw" else perception_deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, perception_image_grids, self.feature_source
        )
        feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
        if not self.feat_grad:
            feature_maps_daf = [x.detach() for x in feature_maps_daf]

        head_dtype = next(self.unified_decoder.parameters()).dtype
        perception_metas = {
            'img_metas': kwargs.get('img_metas'),
            'timestamp': timestamp,
            'projection_mat': projection_mat.to(device=device, dtype=head_dtype) if projection_mat is not None else None,
            'image_wh': image_wh.to(device=device, dtype=head_dtype) if image_wh is not None else None,
        }

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)
        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz, device, dtype, stage1_outs
        )
        perception_condition = self.build_perception_condition(perception_embs, perception_lengths)

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d).to(dtype)
        prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        deepstack_embeds = deepstack_features if self.driving_deepstack and deepstack_features is not None else None
        visual_pos_masks = (
            (prefix_input_ids == self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id)
            if deepstack_embeds is not None else None
        )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = self._inference_attn_impl
        _, past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
            deepstack_visual_embeds=deepstack_embeds,
            visual_pos_masks=visual_pos_masks,
        )

        perception_len = perception_embs.shape[1]
        perception_range = torch.arange(1, perception_len + 1, device=device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values if perception_len > 0 else max_prefix_pos
        max_denoise_pos = (
            max_prefix_pos
            if self.use_perception_adanorm and self.skip_perception_shared_attention
            else max_perception_pos
        )

        if self.use_perception_adanorm and self.skip_perception_shared_attention:
            target_dtype = self.det_proj.weight.dtype
            vlm_enhanced = {
                'det_feat': stage1_outs['det_instance_feature'].to(target_dtype),
                'map_feat': stage1_outs['map_instance_feature'].to(target_dtype),
                'ego_feat': stage1_outs['ego_instance_feature'].to(target_dtype),
            }
            if self.with_motion and 'motion_token' in stage1_outs:
                vlm_enhanced['motion_feat'] = stage1_outs['motion_token'].to(target_dtype)
            stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)
            det_result, map_result = self.unified_decoder.post_process(stage2_outs)
            cached_pad_masks = prefix_pad_masks
        else:
            # The current H20 fast path uses skip_shared_attention=True. Keep a
            # complete fallback so this helper remains behaviorally aligned.
            cached_pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks], dim=1)
            stage2_outs = None
            det_result = None
            map_result = None

        if use_gt_ego_status:
            ego_status_pred = None
        elif stage2_outs and 'ego_status_list' in stage2_outs and len(stage2_outs['ego_status_list']) > 0:
            ego_status_pred = stage2_outs['ego_status_list'][-1].squeeze(1).to(torch.float32)
        else:
            ego_status_pred = noise.new_zeros((bsz, self.ego_status_dim), dtype=torch.float32)

        return {
            "batch": batch,
            "cached_pad_masks": cached_pad_masks,
            "past_key_values": past_key_values,
            "x_t": noise,
            "timestep": torch.ones((bsz,), dtype=dtype, device=device),
            "dt": torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device),
            "max_denoise_pos": max_denoise_pos,
            "ego_status_pred": ego_status_pred,
            "use_gt_ego_status": use_gt_ego_status,
            "hist_traj": hist_traj,
            "perception_condition": perception_condition,
            "denorm_mu": noise.new_tensor([0.0233, 2.2707]),
            "denorm_std": noise.new_tensor([0.3427, 1.8668]),
            "det": det_result,
            "map": map_result,
        }

    @torch.no_grad()
    def run_partial_action_graph(self, state: Dict[str, object]) -> torch.Tensor:
        x_t = state["x_t"]
        v_t = self._denoise_step(
            state["batch"],
            state["cached_pad_masks"],
            state["past_key_values"],
            x_t.to(state["timestep"].dtype),
            state["timestep"],
            state["max_denoise_pos"],
            ego_status_pred=state["ego_status_pred"],
            use_gt_status=bool(state["use_gt_ego_status"]),
            hist_traj=state["hist_traj"],
            perception_condition=state["perception_condition"],
        )
        x_0 = x_t + state["dt"] * v_t
        traj_pred_meter_deltas = x_0 * (state["denorm_std"] + 1e-6) + state["denorm_mu"]
        zeros = torch.zeros((x_0.shape[0], 1, 2), device=x_0.device, dtype=x_0.dtype)
        return zeros + torch.cumsum(traj_pred_meter_deltas, dim=1)

    def _denoise_step(
        self,
        batch,
        cached_pad_masks,
        past_key_values,
        x_t,
        timestep,
        max_cached_position_ids,
        *,
        ego_status_pred: Optional[torch.Tensor] = None,
        use_gt_status: bool = False,
        hist_traj=None,
        perception_condition: Optional[torch.Tensor] = None,
        interval_start: Optional[torch.Tensor] = None,
        return_context: bool = False,
        action_expert_override: Optional[nn.Module] = None,
        action_out_proj_override: Optional[nn.Module] = None,
        action_in_proj_override: Optional[nn.Module] = None,
        status_mlp_override: Optional[nn.Module] = None,
        hist_traj_encoder_override: Optional[nn.Module] = None,
        action_time_mlp_in_override: Optional[nn.Module] = None,
        action_time_mlp_out_override: Optional[nn.Module] = None,
        perception_suffix_norm_override: Optional[nn.Module] = None,
        perception_suffix_scale_shift_override: Optional[nn.Module] = None,
    ):

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch,
            x_t,
            timestep,
            interval_start=interval_start,
            ego_status_pred=ego_status_pred,
            use_gt_status=use_gt_status,
            hist_traj=hist_traj,
            action_in_proj_override=action_in_proj_override,
            status_mlp_override=status_mlp_override,
            hist_traj_encoder_override=hist_traj_encoder_override,
            action_time_mlp_in_override=action_time_mlp_in_override,
            action_time_mlp_out_override=action_time_mlp_out_override,
        )
        suffix_embs = self.apply_perception_adanorm(
            suffix_embs,
            perception_condition,
            suffix_norm_override=perception_suffix_norm_override,
            scale_shift_override=perception_suffix_scale_shift_override,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = cached_pad_masks.shape[0]
        cached_len = cached_pad_masks.shape[1]

        cached_pad_2d_masks = cached_pad_masks[:, None, :].expand(batch_size, suffix_len, cached_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([cached_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks).to(x_t.dtype)

        suffix_range = torch.arange(1, suffix_len + 1, device=max_cached_position_ids.device).view(1, -1).expand(batch_size, -1)
        suffix_pos_ids_1d = max_cached_position_ids + suffix_range

        position_ids = torch.stack([
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
        ], dim=0)

        action_expert = action_expert_override or self.qwen3_vl_with_expert.qwen3_action_expert
        action_expert.config._attn_implementation = self._inference_attn_impl

        outputs_embeds, _, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
            action_expert_override=action_expert_override,
        )

        suffix_out = outputs_embeds[2]

        suffix_out = suffix_out[:, -self.action_horizon :].to(dtype=torch.float32)

        action_out_proj = action_out_proj_override or self.action_out_proj
        model_out = action_out_proj(suffix_out)

        if return_context:
            return model_out, suffix_out
        return model_out

    def forward_ar_batch(self, ar_batch):
        input_ids = ar_batch['ar_input_ids']
        labels    = ar_batch['ar_labels']
        device    = input_ids.device
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

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "flash_attention_2"
        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"
        loss_vlm = outputs.loss

        return dict(
            loss_ar=self.ar_loss_weight * loss_vlm,
            loss_vlm_raw=loss_vlm.detach(),
        )
