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

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional, List

import numpy as np
import torch
from torch import nn
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
from .canbus_prompt import build_driving_user_prompt


from projects.mmdet3d_plugin.losses.collision_loss import CollisionLoss
from projects.mmdet3d_plugin.losses.plan_map_loss import GTMapBoundLoss, GTMapDirectionLoss
from projects.mmdet3d_plugin.models.detection3d.target import SparseBox3DTarget
from projects.mmdet3d_plugin.models.detection3d.losses import SparseBox3DLoss
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import SparseBox3DEncoder
from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget
from projects.mmdet3d_plugin.models.map.loss import SparseLineLoss
from projects.mmdet3d_plugin.models.map.map_blocks import SparsePoint3DEncoder
from projects.mmdet3d_plugin.models.map.decoder import SparsePoint3DDecoder
from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *

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
    prompt_id_lists: Optional[List[List[int]]] = None
    traj_answer_ids: Optional[torch.Tensor] = None
    traj_answer_mask: Optional[torch.Tensor] = None
    traj_labels: Optional[torch.Tensor] = None
    traj_answer_lens: Optional[List[int]] = None
    coc_texts: Optional[List[str]] = None
    coc_override_texts: Optional[List[str]] = None
    coc_answer_ids: Optional[torch.Tensor] = None
    coc_answer_mask: Optional[torch.Tensor] = None
    coc_labels: Optional[torch.Tensor] = None
    coc_answer_lens: Optional[List[int]] = None
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
        min_period: float = 4e-3,
        max_period: float = 4.0,
        num_sample_steps: int = 10,
        enable_knowledge_insulation: bool = False,
        ar_loss_weight: float = 0.1,
        train_vlm: bool = False,
        enable_traj_ar: bool = False,
        shared_decoder: bool = False,
        traj_ar_loss_weight: float = 0.1,
        traj_ar_target: Literal["waypoint", "raw_delta", "norm_delta"] = "waypoint",
        coc_jsonl_path: Optional[str] = None,
        enable_coc: bool = False,
        enable_coc_policy_prefix: bool = False,
        enable_coc_context: bool = False,
        coc_context_format: Literal["cot", "nav_cot", "nav_cot_plan"] = "nav_cot",
        coc_context_dropout: float = 0.0,
        coc_planning_loss_weight: float = 0.0,
        enable_coc_language_output: bool = False,
        coc_language_target_format: Literal["cot", "nav_cot"] = "cot",
        coc_language_loss_weight: float = 0.0,
        enable_coc_generate: bool = False,
        coc_generate_max_new_tokens: int = 128,
        coc_eval_source: Literal["pred", "gt", "none"] = "pred",
        coc_eval_injection: Literal["answer", "context", "none"] = "answer",
        driving_prefix_max_len: int = 3840,
        include_ego_status_prompt: bool = False,
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
        peft_cfg: Optional[dict] = None,
        visual_lora_cfg: Optional[dict] = None,
        visual_wasp_cfg: Optional[dict] = None,
        visual_projector_wasp_cfg: Optional[dict] = None,
        visual_gradient_checkpointing: bool = True,
        joint_transformer_checkpoint: bool = True,
        lora_merge_save_dir: Optional[str] = None,
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

        assert not (enable_knowledge_insulation and not enable_traj_ar), \
            "enable_knowledge_insulation=True requires enable_traj_ar=True"
        self.enable_knowledge_insulation = enable_knowledge_insulation
        self._inference_attn_impl = inference_attn_impl
        self.ar_loss_weight = ar_loss_weight
        self.train_vlm = train_vlm
        self.shared_decoder = bool(shared_decoder)
        self.enable_traj_ar = enable_traj_ar
        self.traj_ar_loss_weight = traj_ar_loss_weight
        assert traj_ar_target in ("waypoint", "raw_delta", "norm_delta"), \
            f"traj_ar_target must be one of waypoint/raw_delta/norm_delta, got {traj_ar_target}"
        self.traj_ar_target = traj_ar_target
        assert coc_context_format in ("cot", "nav_cot", "nav_cot_plan"), \
            f"coc_context_format must be one of cot/nav_cot/nav_cot_plan, got {coc_context_format}"
        assert coc_language_target_format in ("cot", "nav_cot"), \
            f"coc_language_target_format must be one of cot/nav_cot, got {coc_language_target_format}"
        assert coc_eval_source in ("pred", "gt", "none"), \
            f"coc_eval_source must be one of pred/gt/none, got {coc_eval_source}"
        assert coc_eval_injection in ("answer", "context", "none"), \
            f"coc_eval_injection must be one of answer/context/none, got {coc_eval_injection}"
        enable_coc = bool(enable_coc)
        assert not (enable_coc and enable_coc_language_output and enable_traj_ar), \
            "enable_coc_language_output uses the assistant answer slot and cannot be enabled with enable_traj_ar"
        self.enable_coc = enable_coc
        self.enable_coc_policy_prefix = enable_coc and bool(enable_coc_policy_prefix)
        self.enable_coc_context = enable_coc and (bool(enable_coc_context) or self.enable_coc_policy_prefix)
        self.coc_context_format = coc_context_format
        self.coc_context_dropout = float(coc_context_dropout) if enable_coc else 0.0
        self.coc_planning_loss_weight = float(coc_planning_loss_weight) if enable_coc else 0.0
        if self.coc_planning_loss_weight != 0.0:
            raise NotImplementedError(
                "coc_planning_loss_weight is reserved for future pred_xyz planning supervision; "
                "current CoC training uses language loss plus the original planning loss."
            )
        self.enable_coc_language_output = enable_coc and bool(enable_coc_language_output)
        self.coc_language_target_format = coc_language_target_format
        self.coc_language_loss_weight = float(coc_language_loss_weight) if enable_coc else 0.0
        self.enable_coc_generate = enable_coc and bool(enable_coc_generate)
        self.coc_generate_max_new_tokens = int(coc_generate_max_new_tokens)
        self.coc_eval_source = coc_eval_source if enable_coc else "none"
        self.coc_eval_injection = coc_eval_injection if enable_coc else "none"
        self.include_ego_status_prompt = bool(include_ego_status_prompt)
        self.driving_prefix_max_len = int(driving_prefix_max_len)
        if self.driving_prefix_max_len <= 0:
            raise ValueError(f"driving_prefix_max_len must be positive, got {driving_prefix_max_len}")
        self._token_ids_cache: Dict[str, tuple] = {}
        self.coc_by_token = self._load_coc_jsonl(coc_jsonl_path) if enable_coc and coc_jsonl_path else {}
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

        self.use_tau0_pred = False

        if vlm_variant == "8b":
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_8b_36l')
            perception_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
            action_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
        else:
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_28l')
            perception_expert_cfg = get_qwen_config('qwen3_28l')
            action_expert_cfg = get_qwen_config('qwen3_28l')

        if self.shared_decoder:
            perception_expert_cfg = qwen3_vl_cfg
            action_expert_cfg = qwen3_vl_cfg

        self.lora_merge_save_dir = lora_merge_save_dir

        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
            qwen3_vl_cfg,
            perception_expert_cfg,
            action_expert_cfg,
            pretrained_path,
            precision=dtype,
            train_vlm=train_vlm,
            lora_cfg=lora_cfg,
            shared_decoder=self.shared_decoder,
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
                    peft_cfg.get(
                        "wasp_loss_weight", peft_cfg.get("beta", 0.01)
                    )
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

        if (
            attn_implementation == "flex"
            and os.environ.get("UNIDRIVE_DISABLE_COMPILED_FLEX", "0").lower() in ("1", "true", "yes", "on")
        ):
            attn_implementation = "sdpa"

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
        self.action_in_proj = nn.Linear(action_dim, action_expert_cfg.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_cfg.hidden_size, action_dim)

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
        self.hist_traj_encoder.to(target_dtype)
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
        self.feature_map_proj.to(target_dtype)
        self.fusion_weight_generators.to(target_dtype)

        if hasattr(self.qwen3_vl_with_expert.qwen3_vl, 'lm_head'):
            self.qwen3_vl_with_expert.qwen3_vl.lm_head.requires_grad_(False)

        self.visual_gradient_checkpointing_enabled = bool(
            visual_gradient_checkpointing
        )
        self.gradient_checkpointing_enable()
        self.joint_transformer_checkpoint = bool(joint_transformer_checkpoint)
        self.gradient_checkpointing_enabled = True

        self.vlm_grad_scale = vlm_grad_scale
        self.driving_deepstack = driving_deepstack

        self.view_token_str_list = NUSCENES_VIEW_TOKENS
        self.view_token_ids = None
        self.view_token_id_list = None

        self._cached_block_mask = None
        self._cached_block_mask_key = None
        self._cached_q_len_rounded = None

        self.adaptive_feature_fusion = torch.compile(
            self.adaptive_feature_fusion, mode="default", fullgraph=False, dynamic=True
        )

    def _get_view_token_id_list(self) -> List[int]:
        if self.view_token_id_list is None:
            tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
            self.view_token_id_list = [
                tokenizer.convert_tokens_to_ids(token)
                for token in self.view_token_str_list
            ]
        return self.view_token_id_list

    def _get_view_token_ids(self, device):
        if self.view_token_ids is None:
            self.view_token_ids = torch.tensor(
                self._get_view_token_id_list(), dtype=torch.long, device=device
            )
        return self.view_token_ids.to(device)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        gc_kwargs = {"use_reentrant": False}
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        visual = self.qwen3_vl_with_expert.qwen3_vl.visual
        if self.visual_gradient_checkpointing_enabled:
            visual.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs
            )
        else:
            visual.gradient_checkpointing_disable()
        if not self.shared_decoder:
            self.qwen3_vl_with_expert.perception_decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs
            )
            self.qwen3_vl_with_expert.action_decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs
            )

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_disable()
        if not self.shared_decoder:
            self.qwen3_vl_with_expert.perception_decoder.gradient_checkpointing_disable()
            self.qwen3_vl_with_expert.action_decoder.gradient_checkpointing_disable()

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

    def set_wasp_accumulation_state(self, compute: bool, scale: float = 1.0):
        self._compute_wasp_regularization_this_step = bool(compute)
        self._wasp_regularization_step_scale = float(scale) if compute else 0.0

    def _encode_text_cached(self, tokenizer, text: str) -> List[int]:
        token_ids = self._token_ids_cache.get(text)
        if token_ids is None:
            token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
            self._token_ids_cache[text] = token_ids
        return list(token_ids)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        t = sample_beta(self.time_beta_alpha, self.time_beta_beta, bsize, device)
        t = t * 0.999 + 0.001
        return t.to(dtype=torch.float32, device=device)

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
            grid_hw = self.qwen3_vl_with_expert._last_image_grid_hw
            if grid_hw is None:
                h_grid = int(all_image_grids[0, 1].item())
                w_grid = int(all_image_grids[0, 2].item())
            else:
                h_grid, w_grid = grid_hw
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

    def _load_coc_jsonl(self, jsonl_path: str) -> Dict[str, dict]:
        coc_by_token = {}
        loaded = 0
        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                token = item.get("sample_token") or item.get("token")
                if not token:
                    skipped += 1
                    continue
                coc_by_token[token] = dict(
                    nav_text=item.get("nav_text") or "",
                    nav_command=item.get("nav_command") or "",
                    cot=item.get("cot") or "",
                )
                loaded += 1
        print(f"[COC] Loaded {loaded} annotations from {jsonl_path}; skipped {skipped}")
        return coc_by_token

    def _extract_sample_tokens(self, img_metas, batch_size: int) -> List[Optional[str]]:
        def unwrap(x):
            if hasattr(x, "data"):
                return unwrap(x.data)
            if isinstance(x, (list, tuple)) and len(x) == 1:
                inner = unwrap(x[0])
                if isinstance(inner, (list, tuple)):
                    return inner
                return [inner]
            return x

        metas = unwrap(img_metas)
        if isinstance(metas, dict):
            metas = [metas]
        if not isinstance(metas, (list, tuple)):
            return [None] * batch_size

        out = []
        for meta in metas:
            if hasattr(meta, "data"):
                meta = unwrap(meta)
            if isinstance(meta, (list, tuple)) and meta:
                meta = meta[0]
            if isinstance(meta, dict):
                out.append(meta.get("token") or meta.get("sample_idx"))
            else:
                out.append(None)
        if len(out) < batch_size:
            out.extend([None] * (batch_size - len(out)))
        return out[:batch_size]

    def _get_coc_records(self, sample_tokens: Optional[List[Optional[str]]], batch_size: int) -> List[Optional[dict]]:
        if not self.coc_by_token or not sample_tokens:
            return [None] * batch_size
        return [self.coc_by_token.get(tok) if tok is not None else None for tok in sample_tokens[:batch_size]]

    def _format_coc_context(self, record: Optional[dict]) -> str:
        if not record:
            return ""
        parts = []
        if self.coc_context_format in ("nav_cot", "nav_cot_plan"):
            nav = record.get("nav_text") or record.get("nav_command") or ""
            if nav:
                parts.append(f"Navigation hint: {nav}")
        cot = record.get("cot") or ""
        if cot:
            parts.append(f"Chain-of-causation reasoning: {cot}")
        if not parts:
            return ""
        return "4. Chain-of-causation context:\n   - " + "\n   - ".join(parts)

    def _format_coc_context_from_text(self, text: Optional[str]) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        return f"4. Chain-of-causation context:\n   - Chain-of-causation reasoning: {text}"

    def _format_coc_language_target(self, record: Optional[dict]) -> str:
        if not record:
            return ""
        cot = (record.get("cot") or "").strip()
        nav = (record.get("nav_text") or record.get("nav_command") or "").strip()
        if self.coc_language_target_format == "nav_cot" and nav and cot:
            return f"Navigation: {nav}\nReasoning: {cot}"
        if self.coc_language_target_format == "nav_cot" and nav:
            return f"Navigation: {nav}"
        return cot

    def _decode_coc_from_slots(
        self,
        batch: DrivingBatch,
        prefix_embs: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        all_image_grids,
        deepstack_features,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        max_new_tokens = int(max_new_tokens or self.coc_generate_max_new_tokens)
        if max_new_tokens <= 0:
            return [""] * int(prefix_embs.shape[0])

        device = prefix_input_ids.device
        bsz = int(prefix_input_ids.shape[0])
        prefix_pad_masks = prefix_input_ids != tokenizer.pad_token_id
        prefix_att_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.long)
        for b_idx, p_len in enumerate(batch.prompt_lens):
            prefix_att_masks[b_idx, : int(p_len)] = 1
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d).to(prefix_embs.dtype)
        prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        ds_embeds = deepstack_features if self.driving_deepstack and deepstack_features is not None else None
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        vis_masks = (prefix_input_ids == image_token_id) if ds_embeds is not None else None

        lm_head = self.qwen3_vl_with_expert.qwen3_vl.lm_head
        embed_tokens = self.qwen3_vl_with_expert.qwen3_vl.get_input_embeddings()
        im_end_ids = self._encode_text_cached(tokenizer, "<|im_end|>")
        stop_ids = set(im_end_ids)
        if tokenizer.eos_token_id is not None:
            stop_ids.add(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            stop_ids.add(tokenizer.pad_token_id)

        generated = [[] for _ in range(bsz)]
        finished = torch.zeros((bsz,), dtype=torch.bool, device=device)
        old_impl = self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation
        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = self._inference_attn_impl
        try:
            prefix_outputs, past_key_values, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=prefix_att_4d,
                position_ids=prefix_pos_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
                deepstack_visual_embeds=ds_embeds,
                visual_pos_masks=vis_masks,
            )
            prefix_hidden = prefix_outputs[0]
            last_indices = torch.tensor(
                [max(int(p_len) - 1, 0) for p_len in batch.prompt_lens],
                dtype=torch.long,
                device=device,
            ).clamp(max=max(int(prefix_hidden.shape[1]) - 1, 0))
            next_hidden = prefix_hidden[torch.arange(bsz, device=device), last_indices]
            next_logits = lm_head(next_hidden.to(lm_head.weight.dtype))
            next_token = next_logits.argmax(dim=-1)

            cache_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1).values + 1
            generated_att_mask = torch.zeros((bsz, 0), dtype=torch.bool, device=device)
            for _ in range(max_new_tokens):
                step_tokens = next_token.masked_fill(finished, tokenizer.pad_token_id)
                was_finished = finished.clone()
                for b_idx, tok in enumerate(step_tokens.detach().cpu().tolist()):
                    if finished[b_idx]:
                        continue
                    if tok in stop_ids:
                        finished[b_idx] = True
                    else:
                        generated[b_idx].append(tok)
                if bool(finished.all().item()):
                    break

                step_embs = embed_tokens(step_tokens[:, None]).to(prefix_embs.dtype)
                step_pos_1d = cache_pos[:, None]
                step_pos_ids = torch.stack([step_pos_1d] * 3, dim=0)
                step_att_2d = torch.cat([prefix_pad_masks, generated_att_mask], dim=1)[:, None, :]
                self_att = torch.ones((bsz, 1, 1), dtype=torch.bool, device=device)
                step_att_2d = torch.cat([step_att_2d, self_att], dim=-1)
                step_att_mask = self._prepare_attention_masks_4d(step_att_2d).to(prefix_embs.dtype)
                step_outputs, past_key_values, _ = self.qwen3_vl_with_expert.forward(
                    attention_mask=step_att_mask,
                    position_ids=step_pos_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[step_embs, None, None],
                    use_cache=True,
                )
                step_hidden = step_outputs[0][:, -1]
                step_logits = lm_head(step_hidden.to(lm_head.weight.dtype))
                next_token = step_logits.argmax(dim=-1)
                cache_pos = cache_pos + 1
                generated_att_mask = torch.cat([generated_att_mask, (~was_finished)[:, None]], dim=1)
        finally:
            self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = old_impl

        return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]

    def _build_driving_batch(
        self,
        img: torch.Tensor,
        command=None,
        ego_status=None,
        gt_ego_lcf_feat=None,
        gt_ego_his_trajs=None,
        hist_traj=None,
        gt_trajs=None,
        coc_records: Optional[List[Optional[dict]]] = None,
        include_coc_answer_tokens: bool = True,
        include_coc_context: bool = True,
        force_coc_language_prompt: bool = False,
        coc_override_texts: Optional[List[str]] = None,
    ) -> DrivingBatch:
        device = img.device if img is not None else torch.device("cuda")
        b = int(img.shape[0]) if torch.is_tensor(img) else 1

        permute_indices = [0, 2, 1, 4, 5, 3]
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
            if command.dim() > 1 and command.shape[-1] == 3:
                cmd_idx = command.argmax(dim=-1).view(-1).long()
                command = command.to(dtype=torch.float32)
            else:
                cmd_idx = command.view(-1).long()
                command = F.one_hot(cmd_idx, num_classes=3).to(device=device, dtype=torch.float32)

            idx_list = cmd_idx.tolist()
        else:
            idx_list = [2] * b

        nav_cmd_texts = [_NAV_CMD_FIXED.get(i, _NAV_CMD_FIXED[2]) for i in idx_list]

        if hist_traj is None:
            hist_traj_np = np.zeros((b, self.hist_traj_steps, self.hist_traj_dim), dtype=np.float32)
        elif torch.is_tensor(hist_traj):
            hist_traj_np = hist_traj.detach().cpu().numpy()
        else:
            hist_traj_np = np.asarray(hist_traj, dtype=np.float32)
        if hist_traj_np.ndim == 2:
            hist_traj_np = hist_traj_np[None]

        if not hasattr(self.qwen3_vl_with_expert, "processor") or self.qwen3_vl_with_expert.processor is None:
            raise RuntimeError("QwenVLAPlanningHead expects `self.qwen3_vl_with_expert.processor`")

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        im_start_id = self._encode_text_cached(tokenizer, "<|im_start|>")[0]
        im_end_id = self._encode_text_cached(tokenizer, "<|im_end|>")[0]
        nl_id = self._encode_text_cached(tokenizer, "\n")[0]

        system_ids = self._encode_text_cached(tokenizer, "system")
        user_ids = self._encode_text_cached(tokenizer, "user")
        assistant_ids = self._encode_text_cached(tokenizer, "assistant")

        sys_content_ids = self._encode_text_cached(tokenizer, NUSCENES_SYSTEM_PROMPT)
        sys_part = [im_start_id] + system_ids + [nl_id] + sys_content_ids + [im_end_id, nl_id]

        user_start_part = [im_start_id] + user_ids + [nl_id]

        user_end_assistant_start_part = [im_end_id, nl_id, im_start_id] + assistant_ids + [nl_id]

        input_ids_list = []
        prompt_id_lists = []
        attention_mask_list = []

        def _batch_item(value, batch_idx):
            if value is None:
                return None
            if torch.is_tensor(value):
                value = value.detach().to(dtype=torch.float32).cpu().numpy()
            else:
                value = np.asarray(value, dtype=np.float32)
            return value[batch_idx] if value.ndim > 1 else value

        for i in range(b):


            coc_record_i = coc_records[i] if coc_records is not None and i < len(coc_records) else None
            coc_override_i = coc_override_texts[i] if coc_override_texts is not None and i < len(coc_override_texts) else None
            has_coc_language_target = bool(coc_override_i or self._format_coc_language_target(coc_record_i))
            use_coc_language_prompt = (
                force_coc_language_prompt
                or (
                    include_coc_answer_tokens
                    and self.enable_coc_language_output
                    and has_coc_language_target
                )
            )
            user_prompt_text = build_driving_user_prompt(
                nav_cmd=nav_cmd_texts[i],
                hist_traj=hist_traj_np[i],
                trajectory_prompt_template=NUSCENES_USER_PROMPT_TEMPLATE,
                use_coc_language_prompt=use_coc_language_prompt,
                include_ego_status_prompt=self.include_ego_status_prompt,
                gt_ego_lcf_feat=_batch_item(gt_ego_lcf_feat, i),
                gt_ego_his_trajs=_batch_item(gt_ego_his_trajs, i),
                ego_status=_batch_item(ego_status, i),
            )

            coc_text = ""
            if include_coc_context and (self.enable_coc_context or coc_override_texts is not None):
                if coc_override_texts is not None and i < len(coc_override_texts):
                    coc_text = self._format_coc_context_from_text(coc_override_texts[i])
                elif coc_records is not None:
                    coc_text = self._format_coc_context(coc_record_i)
                if coc_text and self.training and self.coc_context_dropout > 0:
                    if torch.rand((), device=device).item() < self.coc_context_dropout:
                        coc_text = ""
                if coc_text:
                    user_prompt_text = user_prompt_text.replace(
                        "\n\nOutput requirements:",
                        f"\n{coc_text}\n\nOutput requirements:",
                    )
            user_content_ids = self._encode_text_cached(tokenizer, user_prompt_text)

            full_ids = sys_part + user_start_part + user_content_ids + user_end_assistant_start_part

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            prompt_id_lists.append(full_ids)
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        tokenized_prompt = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        tokenized_prompt_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        traj_answer_ids = None
        traj_answer_mask = None
        traj_labels = None
        traj_answer_lens = None
        if gt_trajs is not None:
            traj_answer_lens = []
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
                answer_token_ids = self._encode_text_cached(tokenizer, answer_text)
                answer_token_ids = answer_token_ids + [im_end_id, nl_id]
                traj_answer_lens.append(
                    sum(
                        token_id != tokenizer.pad_token_id
                        for token_id in answer_token_ids[:128]
                    )
                )
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

        coc_answer_ids = None
        coc_answer_mask = None
        coc_labels = None
        coc_answer_lens = None
        if (
            include_coc_answer_tokens
            and self.enable_coc_language_output
            and (coc_records is not None or coc_override_texts is not None)
        ):
            coc_answer_ids_list = []
            coc_answer_lens = []
            coc_loss_masks = []
            for i in range(b):
                if coc_override_texts is not None and i < len(coc_override_texts):
                    target_text = (coc_override_texts[i] or "").strip()
                else:
                    target_text = self._format_coc_language_target(coc_records[i] if i < len(coc_records) else None)
                answer_token_ids = self._encode_text_cached(tokenizer, target_text) if target_text else []
                answer_token_ids = answer_token_ids + [im_end_id, nl_id] if answer_token_ids else []
                coc_answer_lens.append(min(len(answer_token_ids), 128))
                coc_answer_ids_list.append(torch.tensor(answer_token_ids, dtype=torch.long))
                coc_loss_masks.append(torch.ones(len(answer_token_ids), dtype=torch.long))

            coc_answer_ids = pad_sequence(
                coc_answer_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            coc_answer_mask = pad_sequence(
                coc_loss_masks, batch_first=True, padding_value=0
            ).to(device)

            FIXED_COC_ANSWER_LEN = 128
            curr_coc_len = coc_answer_ids.shape[1]
            if curr_coc_len < FIXED_COC_ANSWER_LEN:
                pad_len = FIXED_COC_ANSWER_LEN - curr_coc_len
                coc_answer_ids = F.pad(coc_answer_ids, (0, pad_len), value=tokenizer.pad_token_id)
                coc_answer_mask = F.pad(coc_answer_mask, (0, pad_len), value=0)
            elif curr_coc_len > FIXED_COC_ANSWER_LEN:
                coc_answer_ids = coc_answer_ids[:, :FIXED_COC_ANSWER_LEN]
                coc_answer_mask = coc_answer_mask[:, :FIXED_COC_ANSWER_LEN]

            coc_labels = coc_answer_ids.masked_fill(coc_answer_mask == 0, -100)

        return DrivingBatch(
            images=images,
            image_masks=image_masks,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            command=command,
            ego_status=ego_status,
            view_token_ids=view_token_ids,
            prompt_id_lists=prompt_id_lists,
            traj_answer_ids=traj_answer_ids,
            traj_answer_mask=traj_answer_mask,
            traj_labels=traj_labels,
            traj_answer_lens=traj_answer_lens,
            coc_texts=[self._format_coc_context(r) for r in coc_records] if coc_records is not None else None,
            coc_override_texts=coc_override_texts,
            coc_answer_ids=coc_answer_ids,
            coc_answer_mask=coc_answer_mask,
            coc_labels=coc_labels,
            coc_answer_lens=coc_answer_lens,
        )


    def embed_prefix(self, batch: DrivingBatch):
        images_tensor = batch.images
        device = self.action_in_proj.weight.device

        image_features, feature_lens, all_image_grids, deepstack_features, raw_features = self.qwen3_vl_with_expert.embed_image_tensor(images_tensor)

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        vision_start_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_start_token_id
        vision_end_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_end_token_id
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        nl_id = self._encode_text_cached(tokenizer, "\n")[0]

        bs = images_tensor.shape[0]
        num_views_per_sample = 6
        view_token_id_list = self._get_view_token_id_list()

        prefix_input_ids_list = []
        vision_prefix_lens = []

        fixed_prefix_max_len = self.driving_prefix_max_len

        for b_idx in range(bs):
            sample_input_ids = []

            for v_idx in range(num_views_per_sample):
                img_len = feature_lens[b_idx * num_views_per_sample + v_idx]
                ids = [view_token_id_list[v_idx], nl_id, vision_start_id] + \
                    [image_token_id] * img_len + \
                    [vision_end_id, nl_id]
                sample_input_ids.extend(ids)

            vision_prefix_lens.append(len(sample_input_ids))
            if batch.prompt_id_lists is not None:
                prompt_ids = batch.prompt_id_lists[b_idx]
            else:
                prompt_mask = batch.tokenized_prompt_mask[b_idx].bool()
                prompt_ids = batch.tokenized_prompt[b_idx][prompt_mask].tolist()
            sample_input_ids.extend(prompt_ids)

            prefix_input_ids_list.append(torch.tensor(sample_input_ids, dtype=torch.long, device=device))

        prompt_lens_list = [ids.shape[0] for ids in prefix_input_ids_list]

        answer_ids = batch.traj_answer_ids if batch.traj_answer_ids is not None else batch.coc_answer_ids
        answer_mask = batch.traj_answer_mask if batch.traj_answer_mask is not None else batch.coc_answer_mask

        answer_lens = batch.traj_answer_lens if batch.traj_answer_ids is not None else batch.coc_answer_lens
        fixed_total_len = fixed_prefix_max_len + (
            answer_ids.shape[1] if answer_ids is not None else 0
        )

        def _trim_prefix_ids_keep_prompt_tail(p_ids: torch.Tensor, max_len: int, vision_prefix_len: int) -> torch.Tensor:
            if p_ids.shape[0] <= max_len:
                return p_ids
            if vision_prefix_len < 0 or vision_prefix_len > int(p_ids.shape[0]):
                return p_ids[-max_len:]
            if vision_prefix_len >= max_len:
                return p_ids[:max_len]
            tail_budget = max_len - vision_prefix_len
            text_tail = p_ids[vision_prefix_len:][-tail_budget:]
            return torch.cat([p_ids[:vision_prefix_len], text_tail], dim=0)

        if answer_ids is not None:
            full_ids_list = []
            for b_idx in range(bs):
                p_ids = prefix_input_ids_list[b_idx]
                a_ids = answer_ids[b_idx]
                if p_ids.shape[0] > fixed_prefix_max_len:
                    p_ids = _trim_prefix_ids_keep_prompt_tail(
                        p_ids,
                        fixed_prefix_max_len,
                        vision_prefix_lens[b_idx],
                    )
                    prompt_lens_list[b_idx] = int(p_ids.shape[0])
                full_ids_list.append(torch.cat([p_ids, a_ids], dim=0))

            prefix_input_ids = pad_sequence(
                full_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            curr_len = prefix_input_ids.shape[1]
            if curr_len < fixed_total_len:
                prefix_input_ids = F.pad(prefix_input_ids, (0, fixed_total_len - curr_len), value=tokenizer.pad_token_id)
            elif curr_len > fixed_total_len:
                prefix_input_ids = prefix_input_ids[:, :fixed_total_len]
        else:
            prefix_input_ids = pad_sequence(prefix_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            curr_len = prefix_input_ids.shape[1]
            if curr_len < fixed_prefix_max_len:
                prefix_input_ids = F.pad(prefix_input_ids, (0, fixed_prefix_max_len - curr_len), value=tokenizer.pad_token_id)
            elif curr_len > fixed_prefix_max_len:
                trimmed_ids_list = []
                for b_idx, p_ids in enumerate(prefix_input_ids_list):
                    p_ids = _trim_prefix_ids_keep_prompt_tail(
                        p_ids,
                        fixed_prefix_max_len,
                        vision_prefix_lens[b_idx],
                    )
                    trimmed_ids_list.append(p_ids)
                    prompt_lens_list[b_idx] = int(p_ids.shape[0])
                prefix_input_ids = pad_sequence(
                    trimmed_ids_list,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                )
                if prefix_input_ids.shape[1] < fixed_prefix_max_len:
                    prefix_input_ids = F.pad(
                        prefix_input_ids,
                        (0, fixed_prefix_max_len - prefix_input_ids.shape[1]),
                        value=tokenizer.pad_token_id,
                    )
                truncated_mask = (prefix_input_ids == image_token_id)
                valid_img_tokens = truncated_mask.sum().item()
                if valid_img_tokens < image_features.shape[0]:
                    image_features = image_features[:valid_img_tokens]

        prompt_only_len = max(prompt_lens_list)

        prefix_pad_masks = (prefix_input_ids != tokenizer.pad_token_id)

        input_embeds = self.qwen3_vl_with_expert.qwen3_vl.get_input_embeddings()(prefix_input_ids)

        image_mask = (prefix_input_ids == image_token_id)

        target_count = int(image_mask.sum().item())
        if target_count != image_features.shape[0]:
            current_count = image_features.shape[0]
            if current_count > target_count:
                image_features = image_features[:target_count]
            else:
                raise ValueError(f"Visual features mismatch! Feat: {current_count}, Tokens: {target_count}")

        input_embeds = input_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features.to(input_embeds.dtype))

        prefix_att_marks = torch.zeros_like(prefix_pad_masks, dtype=torch.long)
        for b_idx in range(bs):
            p_len = prompt_lens_list[b_idx]
            prefix_att_marks[b_idx, :p_len] = 1
        if answer_ids is not None:
            if answer_lens is None:
                answer_lens = answer_mask.sum(dim=1).tolist()
            for b_idx in range(bs):
                p_len = prompt_lens_list[b_idx]
                real_ans_len = int(answer_lens[b_idx])
                prefix_att_marks[b_idx, p_len:p_len + real_ans_len] = 1

        batch.prompt_lens = prompt_lens_list

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
        ego_status_pred: Optional[torch.Tensor] = None,
        use_gt_status: bool = False,
        hist_traj: Optional[torch.Tensor] = None,
    ):
        device = actions.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype

        B = actions.shape[0]

        status_input = self._maybe_get_status_features(
            batch, ego_status_pred, use_gt=use_gt_status, dtype=dtype,
        )
        status_emb = self.status_mlp(status_input)
        status_emb = status_emb.unsqueeze(1)

        if hist_traj is not None:
            if not torch.is_tensor(hist_traj):
                hist_traj = torch.as_tensor(hist_traj, device=device)
            else:
                hist_traj = hist_traj.to(device=device)
            if hist_traj.dtype != dtype:
                hist_traj = hist_traj.to(dtype=dtype)
            hist_traj_emb = self.hist_traj_encoder(hist_traj.flatten(1))
            status_emb = torch.cat([hist_traj_emb.unsqueeze(1), status_emb], dim=1)

        num_status_tokens = status_emb.shape[1]

        if actions.dtype != dtype:
            actions = actions.to(dtype=dtype)

        action_emb = self.action_in_proj(actions)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, self.min_period, self.max_period, device=device
        )
        fused_input = torch.cat([action_emb, time_emb.unsqueeze(1).expand_as(action_emb)], dim=-1).to(dtype)

        def mlp_func(x):
            x = self.action_time_mlp_in(x)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = mlp_func(fused_input)
        suffix_emb = torch.cat([status_emb, action_time_emb], dim=1)

        suffix_len = suffix_emb.shape[1]
        pad_masks = torch.ones((B, suffix_len), dtype=torch.bool, device=device)
        att_marks = [1] * num_status_tokens + [0] * (suffix_len - num_status_tokens)
        att_marks = torch.tensor(att_marks, dtype=torch.bool, device=device)[None, :].expand(B, -1)

        return suffix_emb, pad_masks, att_marks

    def get_position_ids(self, input_ids, image_grid_thw, pad_masks):
        attention_mask = pad_masks.long()
        position_ids, rope_deltas = self.qwen3_vl_with_expert.vlm_base.model.get_rope_index(
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

    def _compute_coc_language_loss(
        self,
        batch: DrivingBatch,
        prefix_out: torch.Tensor,
        zero_ref: torch.Tensor,
    ):
        loss_coc_language = zero_ref.sum() * 0.0
        if batch.coc_labels is None or batch.coc_answer_mask is None:
            valid_ratio = torch.tensor(0.0, device=zero_ref.device)
            return loss_coc_language, valid_ratio

        lm_head = self.qwen3_vl_with_expert.qwen3_vl.lm_head
        answer_len = batch.coc_labels.shape[1]
        valid_mask = batch.coc_answer_mask[:, :answer_len].bool()
        valid_samples = valid_mask.any(dim=1)
        valid_ratio = valid_samples.to(dtype=torch.float32).mean()
        sample_indices, answer_offsets = valid_mask.nonzero(as_tuple=True)
        if sample_indices.numel() == 0:
            return loss_coc_language, valid_ratio

        prompt_lens = torch.as_tensor(
            batch.prompt_lens,
            device=prefix_out.device,
            dtype=torch.long,
        )
        hidden_positions = prompt_lens[sample_indices] - 1 + answer_offsets
        valid_hidden = prefix_out[sample_indices, hidden_positions]
        valid_labels = batch.coc_labels[sample_indices, answer_offsets]

        valid_logits = lm_head(valid_hidden.to(lm_head.weight.dtype))
        token_losses = F.cross_entropy(valid_logits, valid_labels, reduction="none")

        per_sample_loss_sum = token_losses.new_zeros(prefix_out.shape[0])
        per_sample_loss_sum.scatter_add_(0, sample_indices, token_losses)
        per_sample_token_count = valid_mask.sum(dim=1).to(token_losses.dtype)
        per_sample_loss = per_sample_loss_sum / per_sample_token_count.clamp_min(1)
        loss_coc_language = (
            per_sample_loss[valid_samples].mean() * self.coc_language_loss_weight
        )
        return loss_coc_language, valid_ratio

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
        gt_ego_lcf_feat=None,
        gt_ego_his_trajs=None,
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

        coc_records = None
        if self.coc_by_token:
            sample_tokens = self._extract_sample_tokens(kwargs.get("img_metas"), int(img.shape[0]))
            coc_records = self._get_coc_records(sample_tokens, int(img.shape[0]))
        coc_match_count = sum(1 for r in (coc_records or []) if r is not None)

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            gt_ego_lcf_feat=gt_ego_lcf_feat,
            gt_ego_his_trajs=gt_ego_his_trajs,
            hist_traj=hist_traj,
            gt_trajs=_gt_trajs_raw if self.enable_traj_ar else None,
            coc_records=coc_records,
            include_coc_answer_tokens=self.enable_coc_language_output,
            include_coc_context=self.enable_coc_policy_prefix,
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

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = self.embed_prefix(batch)

        if self.driving_deepstack and deepstack_features is not None:
            image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
            _visual_pos_masks = (prefix_input_ids == image_token_id)
            _ds_embeds = deepstack_features
        else:
            _visual_pos_masks = None
            _ds_embeds = None

        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, all_image_grids, self.feature_source
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

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch, x_t, time, ego_status_pred=None, use_gt_status=True, hist_traj=hist_traj,
        )

        if self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            perception_embs = perception_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        if self.vlm_grad_scale != 1.0:
            s = self.vlm_grad_scale
            prefix_embs = prefix_embs * 1.0
            prefix_embs.register_hook(lambda g: g * s)
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

        if (
            self.attn_implementation == "flex"
            and not self.enable_knowledge_insulation
            and batch.coc_answer_ids is None
        ):
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
                traj_answer_lens = batch.traj_answer_lens
                if traj_answer_lens is None:
                    traj_answer_lens = batch.traj_answer_mask.sum(dim=1).tolist()
                for b_idx in range(bsz):
                    p_len = batch.prompt_lens[b_idx]
                    real_ans_len = int(traj_answer_lens[b_idx])
                    att_2d_masks[b_idx, prefix_len:, p_len:p_len + real_ans_len] = False
            att_mask_input = self._prepare_attention_masks_4d(att_2d_masks)
            del att_2d_masks, pad_masks, att_masks

        stats = {}
        if self.enable_coc:
            stats["coc_match_ratio"] = torch.tensor(
                float(coc_match_count) / max(float(bsz), 1.0),
                device=actions.device,
            )
            if batch.coc_answer_mask is not None:
                stats["coc_language_valid_ratio"] = (
                    batch.coc_answer_mask.sum(dim=1) > 0
                ).to(dtype=torch.float32).mean()
            stats["coc_policy_prefix_enabled"] = torch.tensor(
                1.0 if self.enable_coc_policy_prefix else 0.0,
                device=actions.device,
            )
        perception_out = None

        if self.enable_knowledge_insulation:
            self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "sdpa"
            self.qwen3_vl_with_expert.perception_decoder.config._attn_implementation = "sdpa"
            self.qwen3_vl_with_expert.action_decoder.config._attn_implementation = "sdpa"

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

            forward_args = (
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
            if self.joint_transformer_checkpoint:
                outputs_embeds, _middle_layer_outs_unused = self._apply_checkpoint(
                    forward_func, *forward_args
                )
            else:
                outputs_embeds, _middle_layer_outs_unused = forward_func(
                    *forward_args
                )
            prefix_out, perception_out, suffix_out = outputs_embeds

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

        loss_coc_language = prefix_embs.sum() * 0.0
        if self.enable_coc_language_output and self.train_vlm and batch.coc_labels is not None:
            loss_coc_language, coc_language_valid_ratio = self._compute_coc_language_loss(
                batch, prefix_out, prefix_embs
            )
            stats["coc_language_valid_ratio"] = coc_language_valid_ratio

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

        if perception_out is not None:
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
            perception_losses = self.unified_decoder.loss(stage1_outs, stage2_outs, perception_data)
            stats.update(perception_losses)
        else:
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=actions.device, dtype=torch.float32)
            ego_out = None

        suffix_out_float = suffix_out[:, -self.action_horizon:].to(dtype=torch.float32)
        model_out = self.action_out_proj(suffix_out_float)

        t_exp = time[:, None, None].to(dtype=torch.float32)
        pred_x_gt = x_t.to(dtype=torch.float32) - t_exp * model_out

        if self.loss_planning is not None:
            _pred_x0_norm = x_t.to(dtype=torch.float32) - t_exp * model_out
            planning_loss = self.loss_planning(
                model_out, u_t, time,
                gt_ego_fut_masks=gt_ego_fut_masks,
                pred_x0=_pred_x0_norm,
                gt_x0=actions,
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
            _pred_deltas = self.denorm_delta(pred_x_gt[..., :2].float())
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

        if gt_occ_dense is not None and perception_out is not None:
            occ_target = gt_occ_dense.permute(0, 3, 1, 2).long()
            occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            loss_occ = F.cross_entropy(occ_logits, occ_target)
        else:
            loss_occ = perception_embs.sum() * 0.0
        stats["loss_occ"] = loss_occ

        loss_motion = stats.get("loss_motion", perception_embs.sum() * 0.0)

        ar_loss_total = torch.tensor(0.0, device=planning_loss.device)
        if self.train_vlm:
            if ar_batch is not None:
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

        if self.enable_coc_language_output and self.train_vlm:
            stats["loss_coc_language"] = loss_coc_language

        stats.pop('loss_perception', None)
        stats.pop('loss', None)

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
        gt_ego_lcf_feat=None,
        gt_ego_his_trajs=None,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        hist_traj=None,
        use_gt_ego_status: bool = False,
        **kwargs,
    ):
        permute_indices = [0, 2, 1, 4, 5, 3]
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = permute_metas_per_camera_fields(
                kwargs.get("img_metas"), permute_indices, TARGET_SENSOR_ORDER
            )

        sample_tokens = None
        coc_records = None
        if self.coc_by_token:
            sample_tokens = self._extract_sample_tokens(kwargs.get("img_metas"), int(img.shape[0]))
            coc_records = self._get_coc_records(sample_tokens, int(img.shape[0]))

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            gt_ego_lcf_feat=gt_ego_lcf_feat,
            gt_ego_his_trajs=gt_ego_his_trajs,
            hist_traj=hist_traj,
            coc_records=coc_records,
            include_coc_answer_tokens=False,
            include_coc_context=False,
        )
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        num_steps = int(self.num_sample_steps if num_steps is None else num_steps)

        if noise is None:
            noise = self.sample_noise((bsz, self.action_horizon, self.action_dim), device)

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = self.embed_prefix(batch)
        coc_pred_text = None
        coc_gt_text = None
        should_generate_coc = self.enable_coc_generate and self.coc_eval_source == "pred"
        if should_generate_coc:
            coc_batch = self._build_driving_batch(
                img=img,
                command=gt_ego_fut_cmd,
                ego_status=ego_status,
            gt_ego_lcf_feat=gt_ego_lcf_feat,
            gt_ego_his_trajs=gt_ego_his_trajs,
                hist_traj=hist_traj,
                coc_records=coc_records,
                include_coc_answer_tokens=False,
                include_coc_context=False,
                force_coc_language_prompt=True,
            )
            (
                coc_prefix_embs,
                _coc_prefix_pad_masks,
                _coc_prefix_att_masks,
                coc_all_image_grids,
                coc_prefix_input_ids,
                coc_deepstack_features,
                _coc_raw_features,
                _coc_prompt_only_len,
            ) = self.embed_prefix(coc_batch)
            coc_pred_text = self._decode_coc_from_slots(
                coc_batch,
                coc_prefix_embs,
                coc_prefix_input_ids,
                coc_all_image_grids,
                coc_deepstack_features,
                max_new_tokens=self.coc_generate_max_new_tokens,
            )
        if self.enable_coc and self.coc_eval_source in ("pred", "gt"):
            if self.coc_eval_injection == "context":
                coc_gt_text = [
                    self._format_coc_context(r) if r is not None else ""
                    for r in (coc_records or [None] * int(img.shape[0]))
                ]
            else:
                coc_gt_text = [
                    self._format_coc_language_target(r) if r is not None else ""
                    for r in (coc_records or [None] * int(img.shape[0]))
                ]

        coc_condition_text = None
        if self.coc_eval_source == "pred" and coc_pred_text is not None:
            coc_condition_text = coc_pred_text
        elif self.coc_eval_source == "gt" and coc_gt_text is not None:
            coc_condition_text = coc_gt_text

        if coc_condition_text is not None and self.coc_eval_injection != "none":
            use_gt_record_context = (
                self.coc_eval_source == "gt"
                and self.coc_eval_injection == "context"
            )
            batch = self._build_driving_batch(
                img=img,
                command=gt_ego_fut_cmd,
                ego_status=ego_status,
            gt_ego_lcf_feat=gt_ego_lcf_feat,
            gt_ego_his_trajs=gt_ego_his_trajs,
                hist_traj=hist_traj,
                coc_records=coc_records,
                include_coc_answer_tokens=self.coc_eval_injection == "answer",
                include_coc_context=self.coc_eval_injection == "context",
                coc_override_texts=None if use_gt_record_context else coc_condition_text,
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = self.embed_prefix(batch)

        if prefix_embs.dtype != dtype:
            prefix_embs = prefix_embs.to(dtype)

        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, all_image_grids, self.feature_source
        )

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
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d).to(dtype)

        prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

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

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, perception_len, prefix_len)
        perception_att_2d = make_att_2d_masks(perception_pad_masks, perception_att_masks)
        perception_full_att_2d = torch.cat([prefix_pad_2d, perception_att_2d], dim=2)

        perception_full_att_2d_4d = self._prepare_attention_masks_4d(perception_full_att_2d).to(dtype)

        perception_range = torch.arange(1, perception_len + 1, device=device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

        self.qwen3_vl_with_expert.perception_decoder.config._attn_implementation = self._inference_attn_impl

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

        if perception_out is not None:
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
            det_result, map_result = self.unified_decoder.post_process(stage2_outs)
        else:
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=device, dtype=torch.float32)
            ego_out = None
            det_result = None
            map_result = None

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
        ego_status_eval = ego_status_pred.detach().cpu() if torch.is_tensor(ego_status_pred) else None

        if perception_len > 0:
            max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values
        else:
            max_perception_pos = max_prefix_pos

        while time_tensor >= -dt_val / 2:
            expanded_time = time_tensor.expand(bsz)
            v_t = self._denoise_step(
                batch, cached_pad_masks, past_key_values, x_t.to(dtype), expanded_time.to(dtype),
                max_perception_pos, ego_status_pred=ego_status_pred, use_gt_status=use_gt_ego_status, hist_traj=hist_traj,
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
            "coc_pred_text": coc_pred_text,
            "coc_gt_text": coc_gt_text,
            "coc_sample_tokens": sample_tokens,
            "ego_status_pred": ego_status_eval,
        }

    def _denoise_step(self, batch, cached_pad_masks, past_key_values, x_t, timestep, max_cached_position_ids, *, ego_status_pred: Optional[torch.Tensor] = None, use_gt_status: bool = False, hist_traj=None):

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch,
            x_t,
            timestep,
            ego_status_pred=ego_status_pred,
            use_gt_status=use_gt_status,
            hist_traj=hist_traj,
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

        self.qwen3_vl_with_expert.action_decoder.config._attn_implementation = self._inference_attn_impl

        outputs_embeds, _, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]

        suffix_out = suffix_out[:, -self.action_horizon :].to(dtype=torch.float32)

        model_out = self.action_out_proj(suffix_out)

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
