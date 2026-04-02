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
import deepspeed
from mmdet.models import HEADS
from mmdet.models.builder import build_head, build_loss
from mmdet.models.losses import accuracy
from timm.models.layers import Mlp
from einops import rearrange
from .unidrivevla_vlm_qwenvl3_single_decoder import Qwen3VLSingleDecoderModel
from torch.nn.utils.rnn import pad_sequence
from .flex_attention_opt import build_blockmask_unidrive
from .constants import (
    NUSCENES_SYSTEM_PROMPT,
    NUSCENES_USER_PROMPT_TEMPLATE,
    NUSCENES_VIEW_TOKENS,
    TARGET_SENSOR_ORDER,
    OPENPI_ATTENTION_MASK_VALUE,
    DEFAULT_PERM_INDICES,
)
from .utils import (
    make_att_2d_masks,
    sample_beta,
    create_sinusoidal_pos_embedding,
    permute_metas_per_camera_fields,
)
from .modules import OccLatentDecoder, DenseDepthNet


from projects.mmdet3d_plugin.models.detection3d.target import SparseBox3DTarget
from projects.mmdet3d_plugin.models.detection3d.losses import SparseBox3DLoss
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import SparseBox3DEncoder
from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget
from projects.mmdet3d_plugin.models.map.loss import SparseLineLoss
from projects.mmdet3d_plugin.models.map.map_blocks import SparsePoint3DEncoder
from projects.mmdet3d_plugin.models.map.decoder import SparsePoint3DDecoder
from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *

# Unified perception decoder
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
class QwenVL3ASingleDecoderPlanningHead(nn.Module):
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
        vlm_variant: Literal["2b", "8b"] = "2b",
        lora_cfg: Optional[dict] = None,
        loss_planning: Optional[dict] = None,
        occ_loss_weight: float = 1.0,
        collision_loss_weight: float = 1.0,
        x_min: float = -13.97,
        x_max: float = 11.77,
        y_min: float = -2.02,
        y_max: float = 55.79,
        occ_aux_loss_weight: float = 1.0,
        occ_aux_layers_1based: Optional[List[int]] = None,
        attn_implementation: Literal["eager", "sdpa", "flex"] = "sdpa",
        unified_decoder_cfg: dict = None,  # Required, no longer optional
        occworld_vae_config: Optional[dict] = None,
        occworld_vae_path: Optional[str] = None,
        with_depth_supervision: bool = False,
        depth_loss_weight: float = 0.2,
        num_depth_bins: int = 80,
        depth_range: tuple = (1.0, 60.0),
        depth_supervision_source: Literal["input", "output"] = "input",
        feature_source: Literal["raw", "deepstack"] = "deepstack",
        feat_grad: Optional[bool] = None,
        # CMA-Flow parameters
        use_cma_flow: bool = False,
        cma_flow_prior_path: Optional[str] = None,
        cma_flow_loss_type: str = "wta",  # "wta" or "soft"
        cma_flow_top_k: int = 1,  # Number of modes to use in inference
        # HDP (Hybrid Diffusion Policy) parameters
        use_tau0_pred: bool = False,  # Predict clean trajectory τ₀ instead of velocity field v_t
        use_hdp_hybrid_loss: bool = False,  # Use HDP hybrid loss (velocity + waypoint)
        hdp_window_size: int = 4,  # Gradient detach window size for integral
        hdp_omega: float = 0.1,  # Waypoint loss weight
        # VLM Fusion configuration
        vlm_fusion_cfg: Optional[dict] = None,  # {'type': 'mlp' or 'cross_attention', 'num_heads': 8, ...}
        # Feature Fusion configuration
        feature_fusion_cfg: Optional[dict] = None,  # {'type': 'none' or 'fpn', 'fpn_scales': [1, 0.5, 0.25], ...}
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
        self.occ_loss_weight = occ_loss_weight
        self.collision_loss_weight = collision_loss_weight

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        self.occ_aux_loss_weight = occ_aux_loss_weight
        if occ_aux_layers_1based is None:
            occ_aux_layers_1based = [4, 14, 24]
        self.occ_aux_layers = [int(x) - 1 for x in occ_aux_layers_1based]

        self.loss_planning = build_loss(loss_planning) if loss_planning is not None else None

        from projects.mmdet3d_plugin.losses.collision_loss import CollisionLoss
        self.collision_loss_fn = CollisionLoss(weight=1.0) if collision_loss_weight > 0 else None

        self.use_cma_flow = use_cma_flow
        self.cma_flow_loss_type = cma_flow_loss_type
        self.cma_flow_top_k = cma_flow_top_k
        self.cma_flow_prior = None
        self._cma_flow_cmd_verified = False  # Runtime verification flag

        # HDP parameters
        self.use_tau0_pred = use_tau0_pred
        self.use_hdp_hybrid_loss = use_hdp_hybrid_loss
        self.hdp_window_size = hdp_window_size
        self.hdp_omega = hdp_omega

        if self.use_tau0_pred:
            print(f"[HDP] Using τ₀-Prediction: Network predicts clean trajectory directly")
        if self.use_hdp_hybrid_loss:
            print(f"[HDP] Using Hybrid Loss: velocity + {self.hdp_omega} * waypoint (window={self.hdp_window_size})")

        # action_expert_cfg is needed for traj_cls_token initialization
        if vlm_variant == "8b":
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_8b_36l')
            occ_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
            action_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
        else:
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_28l')
            occ_expert_cfg = get_qwen_config('qwen3_28l')
            action_expert_cfg = get_qwen_config('qwen3_28l')

        if self.use_cma_flow:
            if cma_flow_prior_path is None:
                raise ValueError("cma_flow_prior_path must be provided when use_cma_flow=True")
            self._load_cma_flow_prior(cma_flow_prior_path)

            # cma_flow_top_k is a config parameter (6), but actual K is from prior (8)
            # Build trajectory classification token and head
            # cls_token attends to all K modes and outputs K scores
            action_hidden = action_expert_cfg.hidden_size
            K_per_cmd = self.cma_flow_prior['num_clusters_per_cmd']  # ✅ 从prior读取实际的K
            self.traj_cls_token = nn.Parameter(torch.randn(1, 1, action_hidden))
            self.traj_cls_head = nn.Sequential(
                nn.LayerNorm(action_hidden),
                nn.Linear(action_hidden, action_hidden // 2),
                nn.ReLU(),
                nn.Linear(action_hidden // 2, K_per_cmd)  # ✅ 输出实际的K个分数
            )

            print(f"[CMA-Flow] Enabled with {self.cma_flow_prior['total_clusters']} modes")
            print(f"[CMA-Flow] Loss type: {self.cma_flow_loss_type} (trajectory-level WTA)")
            print(f"[CMA-Flow] Trajectory classifier: cls_token → {K_per_cmd} scores")
            print(f"[CMA-Flow] ✅ Classifier head output dim matches prior K={K_per_cmd}")
        else:
            self.traj_cls_token = None
            self.traj_cls_head = None
            print("[Flow Matching] Using single-mode flow matching")


        # Single Decoder: Only use VLM, no separate experts
        self.qwen3_vl_with_expert = Qwen3VLSingleDecoderModel(
            qwen3_vl_cfg,
            pretrained_path,
            precision=dtype,
            train_vlm=train_vlm,
            lora_cfg=lora_cfg,
        )

        self.attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.qwen3_vl.config._attn_implementation = self.attn_implementation

        if occworld_vae_config is None:
            raise ValueError("occworld_vae_config must be provided via config.")
        if occworld_vae_path is None:
            raise ValueError("occworld_vae_path must be provided via config.")

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg must be provided via config.")

        self.embed_dims = unified_decoder_cfg.get("embed_dims", 256)
        # Single decoder: all tokens must be in the VLM's hidden dim, not the expert's dim
        self.vlm_hidden_size = qwen3_vl_cfg.hidden_size

        # Retrieve query counts from unified_decoder_cfg instance banks
        self.num_det_queries = unified_decoder_cfg.get("det_instance_bank", {}).get("num_anchor", 900)
        self.num_map_queries = unified_decoder_cfg.get("map_instance_bank", {}).get("num_anchor", 100)
        self.num_occ_queries = 625

        # Check if motion is enabled in unified_decoder_cfg
        self.with_motion = "motion" in unified_decoder_cfg.get("task_select", [])

        if self.with_motion:
            # Retrieve motion parameters from unified_decoder_cfg
            self.num_motion_queries = unified_decoder_cfg.get("motion_instance_bank", {}).get("num_anchor", 900)
            self.motion_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
            self.motion_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        else:
            # When motion is disabled, set these to None
            self.num_motion_queries = 0
            self.motion_proj_up = None
            self.motion_proj_down = None

        # Ego status dimension (VLM output dimension for velocity, acceleration, etc.)
        # Can be overridden in unified_decoder_cfg['ego_instance_bank']['status_dims']
        self.ego_status_dim = unified_decoder_cfg.get("ego_refine_layer", {}).get("status_dims", 10)

        self.det_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.map_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)

        self.det_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        self.map_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)

        # Ego projection layers (for unified decoder)
        self.ego_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.ego_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)

        # ===== VLM Feature Fusion Configuration =====
        # Configurable fusion type: 'mlp', 'cross_attention', or 'direct'
        # - 'mlp': Concat(stage1, vlm) -> MLP -> residual (original)
        # - 'cross_attention': stage1 as Q, vlm as K/V -> cross-attn -> residual
        # - 'direct': Directly use VLM output, no fusion (stage2_input = vlm_feat)
        self.vlm_fusion_type = vlm_fusion_cfg.get('type', 'mlp') if vlm_fusion_cfg is not None else 'mlp'

        # ✅ FIX 2: Add gating mechanism for VLM feature fusion
        # Initialize to 0 so model starts with pure visual features (preserving base mAP)
        # and gradually learns how much VLM information to incorporate
        # ✅ Lightweight MLP Fusion (Decoupled Tracking & Refinement Architecture)
        # Replace scalar gates with zero-initialized residual MLPs to avoid gradient cutoff
        # while protecting geometric features from VLM's high variance
        def build_fusion_mlp(embed_dims):
            """
            Build a lightweight fusion MLP: Concat(feat_stage1, vlm_feat) → residual delta
            Structure: Linear(2*embed_dims, embed_dims) → LN → ReLU → Linear(embed_dims, embed_dims)
            """
            mlp = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
            )
            # CRITICAL: Zero-initialize the last linear layer to protect 3D geometry
            # This ensures initial output is ~0, so VLM doesn't destroy precise localization
            # But unlike scalar gate=0, gradients can still flow through!
            nn.init.normal_(mlp[-1].weight, std=0.01)
            nn.init.zeros_(mlp[-1].bias)
            return mlp

        def build_cross_attention_fusion(embed_dims, num_heads=8):
            """
            Build cross-attention based fusion module.
            Stage1 features as Query, VLM features as Key/Value.
            Returns a module that outputs residual delta.
            """
            class CrossAttentionFusion(nn.Module):
                def __init__(self, embed_dims, num_heads):
                    super().__init__()
                    self.cross_attn = nn.MultiheadAttention(
                        embed_dim=embed_dims,
                        num_heads=num_heads,
                        dropout=0.1,
                        batch_first=True
                    )
                    self.norm1 = nn.LayerNorm(embed_dims)
                    self.norm2 = nn.LayerNorm(embed_dims)
                    self.ffn = nn.Sequential(
                        nn.Linear(embed_dims, embed_dims * 4),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Linear(embed_dims * 4, embed_dims),
                        nn.Dropout(0.1),
                    )
                    # Zero-initialize output projection for stability
                    nn.init.normal_(self.ffn[-2].weight, std=0.01)
                    nn.init.zeros_(self.ffn[-2].bias)

                def forward(self, stage1_feat, vlm_feat):
                    """
                    Args:
                        stage1_feat: [B, N, D] - Stage-1 decoder features
                        vlm_feat: [B, N, D] - VLM-enhanced features
                    Returns:
                        delta: [B, N, D] - Residual to add to stage1_feat
                    """
                    # Cross-attention: stage1 as Q, vlm as K/V
                    attn_out, _ = self.cross_attn(
                        query=stage1_feat,
                        key=vlm_feat,
                        value=vlm_feat
                    )
                    attn_out = self.norm1(attn_out)

                    # FFN
                    ffn_out = self.ffn(attn_out)
                    delta = self.norm2(ffn_out)

                    return delta

            return CrossAttentionFusion(embed_dims, num_heads)

        # Build fusion modules based on config
        if self.vlm_fusion_type == 'cross_attention':
            fusion_num_heads = vlm_fusion_cfg.get('num_heads', 8) if vlm_fusion_cfg is not None else 8
            self.det_fusion = build_cross_attention_fusion(self.embed_dims, fusion_num_heads)
            self.map_fusion = build_cross_attention_fusion(self.embed_dims, fusion_num_heads)
            self.ego_fusion = build_cross_attention_fusion(self.embed_dims, fusion_num_heads)
            if self.with_motion:
                self.motion_fusion = build_cross_attention_fusion(self.embed_dims, fusion_num_heads)
            else:
                self.motion_fusion = None
            print(f"[VLM Fusion] Using Cross-Attention fusion with {fusion_num_heads} heads")
        elif self.vlm_fusion_type == 'direct':
            # Direct replacement: no fusion modules needed
            self.det_fusion = None
            self.map_fusion = None
            self.ego_fusion = None
            self.motion_fusion = None
            print(f"[VLM Fusion] Using Direct replacement (VLM output only, no fusion)")
        else:  # Default: mlp
            self.det_fusion = build_fusion_mlp(self.embed_dims)
            self.map_fusion = build_fusion_mlp(self.embed_dims)
            self.ego_fusion = build_fusion_mlp(self.embed_dims)
            if self.with_motion:
                self.motion_fusion = build_fusion_mlp(self.embed_dims)
            else:
                self.motion_fusion = None
            print(f"[VLM Fusion] Using MLP fusion (original)")

        # Prevents cls loss from dominating or being dominated by flow loss
        # Typical values: 0.05-0.2, can be tuned or warmup scheduled
        self.loss_traj_cls_weight = 0.1  # Default: 10% of total loss

        # Vision encoder hidden size (2B: 1024, 8B: 1152)
        vision_hidden_size = self.qwen3_vl_with_expert.qwen3_vl.config.vision_config.hidden_size
        self.feature_source = feature_source

        if feat_grad is None:
            raise ValueError("feat_grad must be provided via config.")
        self.feat_grad = bool(feat_grad)

        if self.feature_source == "raw":
            # raw mode: 3 raw features from deepstack layers + 1 raw feature from second-to-last layer
            # Total: 4 raw features
            proj_input_dim = vision_hidden_size
            num_proj_layers = 4
            print(f"[Feature Config] Using RAW features: 3 deepstack layers + 1 second-to-last layer = 4 features")
        elif self.feature_source == "deepstack":
            # deepstack mode: 3 deepstack features (merged 2x2)
            proj_input_dim = vision_hidden_size * 2  # Merged 2x2
            num_proj_layers = 3
            print(f"[Feature Config] Using DEEPSTACK features: 3 merged features")
        else:
            raise ValueError(f"Unknown feature_source: {feature_source}")

        self.num_feature_scales = num_proj_layers

        # Feature projection layers
        self.feature_map_proj = nn.ModuleList([
            nn.Linear(proj_input_dim, self.embed_dims)
            for _ in range(num_proj_layers)
        ])

        # Multi-scale feature fusion module - learns adaptive weights for different scales
        # This helps balance information from different resolution levels
        self.use_adaptive_fusion = True  # Can be configurable
        if self.use_adaptive_fusion:
            self.fusion_weight_generators = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(self.embed_dims, num_proj_layers),
                    nn.Softmax(dim=-1)
                ) for _ in range(num_proj_layers)
            ])
            print(f"[Feature Fusion] Enabled adaptive multi-scale fusion with {num_proj_layers} scales")
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

        # Build unified decoder (required)
        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg is required. Legacy det_vla_head/map_vla_head are no longer supported.")

        self.unified_decoder = build_head(unified_decoder_cfg)


        self.occ_decoder = OccLatentDecoder(
            qwen_dim=qwen3_vl_cfg.hidden_size,
            occworld_vae_config=occworld_vae_config,
            pretrained_vae_path=occworld_vae_path,
        )
        self.action_in_proj = nn.Linear(action_dim, qwen3_vl_cfg.hidden_size)
        self.action_out_proj = nn.Linear(qwen3_vl_cfg.hidden_size, action_dim)

        # Mode classifier for CMA-Flow (will be initialized properly if use_cma_flow=True)
        self.mode_classifier = None

        status_in_features = 3 + self.ego_status_dim

        self.status_mlp = Mlp(
            in_features=status_in_features,
            hidden_features=qwen3_vl_cfg.hidden_size,
            out_features=qwen3_vl_cfg.hidden_size,
            norm_layer=nn.LayerNorm,
        )

        self.hist_traj_steps = 4
        self.hist_traj_dim = 2
        self.hist_traj_encoder = Mlp(
            in_features=self.hist_traj_steps * self.hist_traj_dim,
            hidden_features=qwen3_vl_cfg.hidden_size,
            out_features=qwen3_vl_cfg.hidden_size,
            norm_layer=nn.LayerNorm,
        )


        self.action_time_mlp_in = nn.Linear(2 * qwen3_vl_cfg.hidden_size, qwen3_vl_cfg.hidden_size)
        self.action_time_mlp_out = nn.Linear(qwen3_vl_cfg.hidden_size, qwen3_vl_cfg.hidden_size)


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

        # Projection layers for perception tasks
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

        # Convert feature fusion module to target dtype

        self.fusion_weight_generators.to(target_dtype)
        if hasattr(self.qwen3_vl_with_expert.qwen3_vl, 'lm_head'):
            print("[Config] Freezing LM Head (requires_grad=False) to fix ZeRO-3 unused param error.")
            self.qwen3_vl_with_expert.qwen3_vl.lm_head.requires_grad_(False)

        # self.gradient_checkpointing_enabled = False
        # self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing = True
        # self.qwen3_vl_with_expert.qwen3_action_expert.gradient_checkpointing = True
        self.gradient_checkpointing_enable()
        self.gradient_checkpointing_enabled = True

        self.view_token_str_list = NUSCENES_VIEW_TOKENS
        self.view_token_ids = None

    def _get_view_token_ids(self, device):
        if self.view_token_ids is None:
            tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
            ids = []
            for t in self.view_token_str_list:
                tid = tokenizer.convert_tokens_to_ids(t)
                ids.append(tid)
            self.view_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        return self.view_token_ids.to(device)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        # use_reentrant=False: safe with ZeRO Stage 1/2.
        # ZeRO Stage 2 only does reduce_scatter on gradients (no allgather), so
        # the unpack_hook used by use_reentrant=False does NOT trigger NCCL collectives.
        # use_reentrant=True causes double gradient reduction with ZeRO Stage 2 and
        # must NOT be used.
        gc_kwargs = {"use_reentrant": False}
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        logging.info("Enabled gradient checkpointing (use_reentrant=False) for Single Decoder model")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_disable()
        logging.info("Disabled gradient checkpointing for Single Decoder model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

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

    def _detached_integral(self, v: torch.Tensor, W: int, dt: float = 1.0) -> torch.Tensor:
        """
        Implementation of Algorithm 1 from HDP paper: Detached Integral.

        Args:
            v: velocity/delta of future trajectory [..., T, D]
            W: gradient detach window size
            dt: time interval (default 1.0 since v is already per-timestep delta)

        Returns:
            wpt: waypoint trajectories with detached integral [..., T, D]
        """
        # wpt_sg = torch.cumsum(v.detach(), dim=-2) * dt
        wpt_sg = torch.cumsum(v.detach(), dim=-2) * dt

        # shift_sg = torch.roll(wpt_sg, shifts=W, dims=-2)
        shift_sg = torch.roll(wpt_sg, shifts=W, dims=-2)
        if shift_sg.shape[-2] > W:
            shift_sg[..., :W, :] = 0.0

        # wpt = torch.cumsum(v, dim=-2) * dt
        wpt = torch.cumsum(v, dim=-2) * dt

        # shift = torch.roll(wpt, shifts=W, dims=-2)
        shift = torch.roll(wpt, shifts=W, dims=-2)
        if shift.shape[-2] > W:
            shift[..., :W, :] = 0.0

        return wpt + shift_sg - shift

    def compute_hdp_hybrid_loss(self, pred_v: torch.Tensor, gt_v: torch.Tensor) -> torch.Tensor:
        """
        Compute the HDP Hybrid Loss combining velocity and waypoint supervision.

        Args:
            pred_v: predicted deltas/velocities [..., T, D]
            gt_v: ground truth deltas/velocities [..., T, D]

        Returns:
            Scalar loss combining velocity and waypoint terms
        """
        # Velocity loss (L2)
        l_v = F.mse_loss(pred_v, gt_v)

        # Waypoint loss (L2 on detached integral)
        pred_wpt = self._detached_integral(pred_v, self.hdp_window_size, dt=1.0)
        gt_wpt = torch.cumsum(gt_v, dim=-2) * 1.0
        l_wpt = F.mse_loss(pred_wpt, gt_wpt)

        return l_v + self.hdp_omega * l_wpt

    def _load_cma_flow_prior(self, prior_path: str):
        """Load CMA-Flow prior from pickle file."""
        import pickle
        print(f"[CMA-Flow] Loading prior from {prior_path}")
        with open(prior_path, 'rb') as f:
            self.cma_flow_prior = pickle.load(f)

        # Validate prior structure
        required_keys = ['anchors', 'cholesky_L', 'cmd_to_cluster_idx', 'total_clusters', 'normalized']
        for key in required_keys:
            if key not in self.cma_flow_prior:
                raise ValueError(f"Invalid CMA-Flow prior: missing key '{key}'")

        # Convert to torch tensors
        self.cma_flow_prior['anchors'] = torch.from_numpy(self.cma_flow_prior['anchors']).float()
        self.cma_flow_prior['cholesky_L'] = torch.from_numpy(self.cma_flow_prior['cholesky_L']).float()

        print(f"[CMA-Flow] Prior loaded:")
        print(f"  Total clusters: {self.cma_flow_prior['total_clusters']}")
        print(f"  Clusters per command: {self.cma_flow_prior['num_clusters_per_cmd']}")
        print(f"  Trajectory shape: ({self.cma_flow_prior['traj_timesteps']}, {self.cma_flow_prior['traj_dim']})")
        print(f"  Normalized: {self.cma_flow_prior['normalized']}")
        print(f"  Command names: {self.cma_flow_prior['cmd_names']}")
        print(f"  Command mapping: 0={self.cma_flow_prior['cmd_names'][0]}, "
              f"1={self.cma_flow_prior['cmd_names'][1]}, "
              f"2={self.cma_flow_prior['cmd_names'][2]}")
        print(f"  WARNING: Ensure gt_ego_fut_cmd.argmax() matches this mapping!")

        anchors_torch = self.cma_flow_prior['anchors']
        print(f"\n[Prior Quality Diagnosis]")
        print(f"  Anchors shape: {anchors_torch.shape}")
        print(f"  Anchors range: [{anchors_torch.min().item():.3f}, {anchors_torch.max().item():.3f}]")
        anchors_mean = anchors_torch.mean(dim=(0, 1))
        anchors_std = anchors_torch.std(dim=(0, 1))
        print(f"  Anchors mean (x, y): [{anchors_mean[0].item():.3f}, {anchors_mean[1].item():.3f}]")
        print(f"  Anchors std (x, y): [{anchors_std[0].item():.3f}, {anchors_std[1].item():.3f}]")

        if self.cma_flow_prior['normalized']:
            print(f"  Expected: mean≈[0, 0], std≈[1, 1] for normalized data")
            if abs(anchors_mean[0]) > 0.5 or abs(anchors_mean[1]) > 0.5:
                print(f"  ⚠️  WARNING: Anchors mean deviates significantly from 0!")
            if abs(anchors_std[0] - 1.0) > 0.3 or abs(anchors_std[1] - 1.0) > 0.3:
                print(f"  ⚠️  WARNING: Anchors std deviates significantly from 1!")
            if abs(anchors_mean[0]) < 0.3 and abs(anchors_mean[1]) < 0.3 and abs(anchors_std[0] - 1.0) < 0.2 and abs(anchors_std[1] - 1.0) < 0.2:
                print(f"  ✅ Prior quality looks good (normalized)")
        else:
            print(f"  Prior is NOT normalized (raw meter space)")

    @torch.no_grad()
    def _sample_from_cma_prior(self, cmd: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Sample initial trajectories from CMA-Flow prior.

        Args:
            cmd: Command tensor [B], values in {0, 1, 2}
                 MUST match prior's cmd_names order:
                 0='right', 1='left', 2='straight' (as built by build_cma_flow_prior.py)
            device: Target device
            dtype: Target dtype

        Returns:
            x_0: Sampled trajectories [B, K_per_cmd, T, D] in normalized delta space
        """
        if self.cma_flow_prior is None:
            raise RuntimeError("CMA-Flow prior not loaded")

        bsz = cmd.shape[0]
        K_per_cmd = self.cma_flow_prior['num_clusters_per_cmd']
        T = self.cma_flow_prior['traj_timesteps']
        D = self.cma_flow_prior['traj_dim']
        TD = T * D

        # Move prior tensors to device
        anchors = self.cma_flow_prior['anchors'].to(device=device, dtype=torch.float32)  # [total_K, T, D]
        cholesky_L = self.cma_flow_prior['cholesky_L'].to(device=device, dtype=torch.float32)  # [total_K, TD, TD]

        # Sample for each command in the batch
        samples = []
        for b in range(bsz):
            cmd_idx = int(cmd[b].item())
            start_idx, end_idx = self.cma_flow_prior['cmd_to_cluster_idx'][cmd_idx]

            # Get anchors and Cholesky factors for this command
            cmd_anchors = anchors[start_idx:end_idx]  # [K_per_cmd, T, D]
            cmd_cholesky = cholesky_L[start_idx:end_idx]  # [K_per_cmd, TD, TD]

            # Sample white noise: ε ~ N(0, I)
            epsilon = torch.randn(K_per_cmd, TD, device=device, dtype=torch.float32)  # [K_per_cmd, TD]

            # Apply Cholesky: L_k ε
            # For each mode k: correlated_noise[k] = cholesky[k] @ epsilon[k]
            correlated_noise = torch.bmm(
                cmd_cholesky,  # [K_per_cmd, TD, TD]
                epsilon.unsqueeze(-1)  # [K_per_cmd, TD, 1]
            ).squeeze(-1)  # [K_per_cmd, TD]

            # Add anchors: x_0 = μ_k + L_k ε
            anchors_flat = cmd_anchors.reshape(K_per_cmd, TD)  # [K_per_cmd, TD]
            samples_flat = anchors_flat + correlated_noise  # [K_per_cmd, TD]

            # Reshape to [K_per_cmd, T, D]
            samples_b = samples_flat.reshape(K_per_cmd, T, D)
            samples.append(samples_b)

        # Stack batch: [B, K_per_cmd, T, D]
        x_0 = torch.stack(samples, dim=0).to(dtype=dtype)

        return x_0

    def _create_cma_flow_suffix_mask(
        self,
        batch_size: int,
        num_status_tokens: int,
        K: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block diagonal attention mask for CMA-Flow suffix tokens.

        Suffix layout: [status_tokens, mode_0, mode_1, ..., mode_K-1, cls_token]
        - Status tokens: all suffix tokens can attend
        - Mode tokens: each mode only attends within its own block
        - Cls token: attends to all tokens (global view)

        Args:
            batch_size: B
            num_status_tokens: number of status tokens (e.g., 1 or 2 with hist_traj)
            K: number of modes
            T: trajectory timesteps
            device: torch device

        Returns:
            mask: [B, 1, suffix_len, suffix_len] where suffix_len = num_status_tokens + K*T + 1
                  Values: True = can attend, False = cannot attend (bool tensor)
        """
        suffix_len = num_status_tokens + K * T + 1
        mask = torch.zeros(batch_size, 1, suffix_len, suffix_len, dtype=torch.bool, device=device)

        # 1. Status tokens: all suffix tokens can attend to status tokens
        mask[:, :, :, :num_status_tokens] = True

        # 2. Mode tokens: block diagonal structure
        for k in range(K):
            start = num_status_tokens + k * T
            end = num_status_tokens + (k + 1) * T

            mask[:, :, start:end, start:end] = True

        mask[:, :, -1, :] = True

        for k in range(K):
            start = num_status_tokens + k * T
            end = num_status_tokens + (k + 1) * T
            mask[:, :, start:end, -1] = False

        return mask

    def embed_perception(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stage1_outs: dict,
    ):
        """
        Unified perception embedding from unified decoder outputs.

        Args:
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
            stage1_outs: Output from unified_decoder.forward_stage1() containing:
                - det_instance_feature: (B, num_det, 256)
                - det_predictions: list of anchors, use [-1] for latest
                - map_instance_feature: (B, num_map, 256)
                - map_predictions: list of anchors, use [-1] for latest
                - ego_instance_feature: (B, 1, 256)
                - ego_anchor: (B, 1, 11)
                - motion_token: (B, num_det, 256) or None

        Returns:
            Tuple of:
                - perception_embs: Concatenated embeddings (B, total_len, C)
                - perception_pad_masks: Concatenated pad masks (B, total_len)
                - perception_att_masks: Concatenated attention masks (B, total_len)
                - perception_lengths: Dict with keys 'det', 'map', 'occ', 'ego', 'motion'
                    containing the length of each task's embeddings
        """
        # Extract features from stage1_outs
        det_feat = stage1_outs['det_instance_feature']
        det_anchor = stage1_outs['det_predictions'][-1]
        map_feat = stage1_outs['map_instance_feature']
        map_anchor = stage1_outs['map_predictions'][-1]
        ego_feat = stage1_outs['ego_instance_feature']
        ego_anchor = stage1_outs['ego_anchor']
        motion_token_256 = stage1_outs.get('motion_token', None)

        proj_dtype = self.det_proj_up.weight.dtype

        det_embs = self.det_proj_up(det_feat.to(dtype=proj_dtype))
        anchor_embed_256 = self.unified_decoder.det_anchor_encoder(det_anchor)
        anchor_embed_vlm = self.det_proj_up(anchor_embed_256.to(dtype=proj_dtype))
        det_embs = (det_embs + anchor_embed_vlm).to(dtype)

        det_pad_masks = torch.ones((batch_size, self.num_det_queries), dtype=torch.bool, device=device)
        det_att_masks = torch.zeros((batch_size, self.num_det_queries), dtype=torch.bool, device=device)

        map_embs = self.map_proj_up(map_feat.to(dtype=proj_dtype))
        anchor_embed_out = self.unified_decoder.map_anchor_encoder(map_anchor)
        anchor_embed_256 = anchor_embed_out[0] if isinstance(anchor_embed_out, (tuple, list)) else anchor_embed_out
        anchor_embed_vlm = self.map_proj_up(anchor_embed_256.to(dtype=proj_dtype))
        map_embs = (map_embs + anchor_embed_vlm).to(dtype)

        map_pad_masks = torch.ones((batch_size, self.num_map_queries), dtype=torch.bool, device=device)
        map_att_masks = torch.zeros((batch_size, self.num_map_queries), dtype=torch.bool, device=device)

        occ_embs = self.occ_queries.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        occ_pad_masks = torch.ones((batch_size, self.num_occ_queries), dtype=torch.bool, device=device)
        occ_att_masks = torch.zeros((batch_size, self.num_occ_queries), dtype=torch.bool, device=device)

        # Ego feature from unified decoder (from front-view feature map)
        ego_embs = self.ego_proj_up(ego_feat.to(dtype=proj_dtype))

        # Add ego anchor embedding (ego vehicle box)
        if hasattr(self.unified_decoder, 'ego_anchor_encoder') and ego_anchor is not None:
            ego_anchor_embed_256 = self.unified_decoder.ego_anchor_encoder(ego_anchor)
            ego_anchor_embed_vlm = self.ego_proj_up(ego_anchor_embed_256.to(dtype=proj_dtype))
            ego_embs = (ego_embs + ego_anchor_embed_vlm).to(dtype)
        else:
            ego_embs = ego_embs.to(dtype)

        ego_pad_masks = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        ego_att_masks = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

        if motion_token_256 is not None:
            motion_token_vlm = self.motion_proj_up(motion_token_256.to(dtype=proj_dtype)).to(dtype)
            motion_pad_masks = torch.ones((batch_size, motion_token_vlm.shape[1]), dtype=torch.bool, device=device)
            motion_att_masks = torch.zeros((batch_size, motion_token_vlm.shape[1]), dtype=torch.bool, device=device)
        else:
            # No motion in Stage1, create empty tensors (no dummy tokens)
            # This saves ~900 tokens in sequence length
            motion_token_vlm = torch.empty((batch_size, 0, self.vlm_hidden_size), dtype=dtype, device=device)
            motion_pad_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
            motion_att_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)

        # Pack all embeddings and masks
        perception_embs = torch.cat([det_embs, map_embs, occ_embs, ego_embs, motion_token_vlm], dim=1)
        perception_pad_masks = torch.cat([det_pad_masks, map_pad_masks, occ_pad_masks, ego_pad_masks, motion_pad_masks], dim=1)
        perception_att_masks = torch.cat([det_att_masks, map_att_masks, occ_att_masks, ego_att_masks, motion_att_masks], dim=1)

        # Store lengths for later unpacking
        perception_lengths = {
            'det': det_embs.shape[1],
            'map': map_embs.shape[1],
            'occ': occ_embs.shape[1],
            'ego': ego_embs.shape[1] if ego_embs is not None else 0,
            'motion': motion_token_vlm.shape[1],
        }

        return perception_embs, perception_pad_masks, perception_att_masks, perception_lengths

    def project_and_reshape_features(
        self,
        source_features,
        bsz: int,
        all_image_grids,
        feature_source: str,
    ):
        """
        Project and reshape VLM features to 4D feature maps for deformable aggregation.

        Args:
            source_features: Raw or deepstack features from VLM encoder
                - Can be a single tensor or list of tensors
                - Shape: (N_tokens, C) where N_tokens = bsz * num_views * h * w
            bsz: Batch size
            all_image_grids: Image grid information (batch_size, 3) containing [num_patches, h_grid, w_grid]
            feature_source: "raw" or "deepstack"
                - "raw": Features before vision merge, need to unmerge 2x2 blocks
                - "deepstack": Features after vision merge, already downsampled by 2x

        Returns:
            List of 4D feature maps with shape (bsz, num_views, C, H, W)

        Feature reshaping logic:
            - raw: (bsz * 6 * h_grid * w_grid, C) → (bsz, 6, C, h_grid, w_grid)
                   with 2x2 block unmerging
            - deepstack: (bsz * 6 * h_ds * w_ds, C) → (bsz, 6, C, h_ds, w_ds)
                   where h_ds = h_grid // 2, w_ds = w_grid // 2
        """
        feature_maps = []

        if source_features is None:
            return feature_maps

        # Ensure source_features is a list
        if not isinstance(source_features, list):
            source_features = [source_features]

        # Step 1: Project features through projection layers
        projected_features = []
        for i, feat in enumerate(source_features):
            if i < len(self.feature_map_proj):
                feat = feat.to(self.feature_map_proj[i].weight.dtype)
                feat_proj = self.feature_map_proj[i](feat)
                projected_features.append(feat_proj)
            else:
                projected_features.append(feat)

        # Step 2: Reshape 2D tokens to 4D feature maps
        if all_image_grids is not None and len(all_image_grids) > 0:
            h_grid = int(all_image_grids[0, 1].item())
            w_grid = int(all_image_grids[0, 2].item())
            num_views = 6

            for ds_feat in projected_features:
                # Skip if not 2D (already reshaped or wrong format)
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

        # Apply adaptive multi-scale feature fusion if enabled
        if self.use_adaptive_fusion and len(feature_maps) > 0:
            feature_maps = self.adaptive_feature_fusion(feature_maps)

        return feature_maps

    def adaptive_feature_fusion(self, feature_maps):
        if len(feature_maps) <= 1:
            return feature_maps

        B, N, C, H, W = feature_maps[0].shape
        fused_maps = []

        for i, feat in enumerate(feature_maps):
            feat_flat = feat.view(B * N, C, H, W)

            # ✅ 核心改动：使用专属 generator 算出针对第 i 层输出的权重
            # fusion_weights 形状为 (B*N, num_scales)
            w_dtype = next(self.fusion_weight_generators[i].parameters()).dtype
            fusion_weights = self.fusion_weight_generators[i](feat_flat.to(w_dtype))
            
            # 将权重展开以适配广播计算
            # weight_for_this_scale 形状: (B, N, num_scales, 1, 1, 1)
            weights = fusion_weights.view(B, N, len(feature_maps), 1, 1, 1)

            # 执行加权聚合
            current_fused = 0
            for j in range(len(feature_maps)):
                other_feat = feature_maps[j]
                w = weights[:, :, j] # 取出第 j 层 input 对第 i 层 output 的贡献度

                # 尺寸检查（对齐到当前输出层 i 的 H, W）
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
        """
        Compute depth supervision loss.

        Args:
            prefix_embs: Prefix embeddings (input to VLM), shape (B, prefix_len, C)
            prefix_out: Prefix output (output from VLM), shape (B, prefix_len, C)
            prefix_input_ids: Input token IDs to identify image tokens
            all_image_grids: Image grid information (batch_size, 3) containing [num_patches, h_grid, w_grid]
            gt_depth: Ground truth depth maps, shape (B, num_views, H, W)
            focal: Camera focal lengths, shape (B, num_views) or None
            bsz: Batch size

        Returns:
            Depth supervision loss (scalar tensor)
        """
        if not self.with_depth_supervision or gt_depth is None:
            return torch.tensor(0.0, device=prefix_embs.device)

        feat_for_depth = None
        depth_spatial_shape = None

        # Extract image token features
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        image_mask = (prefix_input_ids == image_token_id)

        if image_mask.any():
            # Select feature source (input or output embeddings)
            if self.depth_supervision_source == "input":
                feat_for_depth = prefix_embs[image_mask]
            elif self.depth_supervision_source == "output":
                feat_for_depth = prefix_out[image_mask]

            # Reshape features to 4D (B*N, C, H, W)
            if feat_for_depth is not None and all_image_grids is not None and len(all_image_grids) > 0:
                h_grid = int(all_image_grids[0, 1].item())
                w_grid = int(all_image_grids[0, 2].item())
                merge_size = 2
                h_ds, w_ds = h_grid // merge_size, w_grid // merge_size
                expected_tokens = bsz * 6 * h_ds * w_ds

                if feat_for_depth.shape[0] == expected_tokens:
                    # Reshape from (N_tokens, C) to (B*6, C, H, W)
                    feat_for_depth = feat_for_depth.view(bsz * 6, h_ds, w_ds, -1).permute(0, 3, 1, 2)
                    depth_spatial_shape = (h_ds, w_ds)
                else:
                    feat_for_depth = None

        # Compute depth loss if features are valid
        if feat_for_depth is not None and depth_spatial_shape is not None:
            gt_depth = gt_depth.to(feat_for_depth.device)
            num_feat_images = feat_for_depth.shape[0]


            gt_depth_reshaped = rearrange(gt_depth, 'b n h w -> (b n) 1 h w')
            if num_feat_images < gt_depth_reshaped.shape[0]:
                gt_depth_reshaped = gt_depth_reshaped[:num_feat_images]

            # Resize ground truth to match feature spatial size
            H_feat, W_feat = feat_for_depth.shape[-2:]
            gt_depth_resized = F.interpolate(
                gt_depth_reshaped, size=(H_feat, W_feat), mode='nearest'
            ).squeeze(1)

            # Prepare focal length
            focal_flat = focal.reshape(-1) if focal is not None else None
            if focal_flat is not None and num_feat_images < focal_flat.shape[0]:
                focal_flat = focal_flat[:num_feat_images]

            # Compute depth loss
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
            cmd_idx = command.view(-1).long()
            idx_list = cmd_idx.tolist()
        else:
            idx_list = [2] * b

        idx2cmd = {
            0: "TURN RIGHT",
            1: "TURN LEFT",
            2: "GO STRAIGHT",
        }
        nav_cmd_texts = [idx2cmd.get(i, "GO STRAIGHT") for i in idx_list]

        hist_traj_np = hist_traj.detach().cpu().numpy()

        if not hasattr(self.qwen3_vl_with_expert, "processor") or self.qwen3_vl_with_expert.processor is None:
            raise RuntimeError("QwenVLAPlanningHead expects `self.qwen3_vl_with_expert.processor`")

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        system_ids = tokenizer.encode("system", add_special_tokens=False)
        user_ids = tokenizer.encode("user", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        sys_content_ids = tokenizer.encode(NUSCENES_SYSTEM_PROMPT, add_special_tokens=False)
        sys_part = [im_start_id] + system_ids + [nl_id] + sys_content_ids + [im_end_id, nl_id]

        user_start_part = [im_start_id] + user_ids + [nl_id]

        user_end_assistant_start_part = [im_end_id, nl_id, im_start_id] + assistant_ids + [nl_id]

        input_ids_list = []
        attention_mask_list = []

        for i in range(b):
            points_str = [f"({pt[0]:.2f}, {pt[1]:.2f})" for pt in hist_traj_np[i]]
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

        return DrivingBatch(
            images=images,
            image_masks=image_masks,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            command=command,
            ego_status=ego_status,
            view_token_ids=view_token_ids
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

        FIXED_PREFIX_MAX_LEN = 3720

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
            print(f"[WARNING] Prefix length {curr_len} > {FIXED_PREFIX_MAX_LEN}. Truncating to avoid crash.")
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
        del image_features

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
        """
        Embed suffix tokens for trajectory prediction.

        Args:
            actions: [B, T, D] for single-mode or [B, K, T, D] for CMA-Flow
            timestep: [B]

        Returns:
            suffix_emb: [B, suffix_len, hidden]
                - Single-mode: suffix_len = status_tokens + T
                - CMA-Flow: suffix_len = status_tokens + K*T + 1 (cls_token)
            pad_masks: [B, suffix_len]
            att_marks: [B, suffix_len]
        """
        device = actions.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype

        # Check if CMA-Flow mode
        is_cma_flow = (actions.dim() == 4)  # [B, K, T, D]

        if is_cma_flow:
            B, K, T, D = actions.shape
        else:
            B = actions.shape[0]
            T = actions.shape[1]

        # === Status embedding (shared across all modes) ===
        status_input = self._maybe_get_status_features(
            batch,
            ego_status_pred,
            use_gt=use_gt_status,
            dtype=dtype,
        )
        status_emb = self._apply_checkpoint(self.status_mlp, status_input)
        status_emb = status_emb.unsqueeze(1)  # [B, 1, hidden]

        if hist_traj is not None:
            if hist_traj.dtype != dtype:
                hist_traj = hist_traj.to(dtype=dtype)
            hist_traj_flat = hist_traj.flatten(1)
            hist_traj_emb = self._apply_checkpoint(self.hist_traj_encoder, hist_traj_flat)
            hist_traj_emb = hist_traj_emb.unsqueeze(1)  # [B, 1, hidden]
            status_emb = torch.cat([hist_traj_emb, status_emb], dim=1)  # [B, 2, hidden]

        num_status_tokens = status_emb.shape[1]

        # === Action embedding ===
        if is_cma_flow:

            if actions.dtype != dtype:
                actions = actions.to(dtype=dtype)

            actions_flat = actions.view(B * K, T, D)  # [B*K, T, D]
            action_emb_flat = self._apply_checkpoint(self.action_in_proj, actions_flat)  # [B*K, T, hidden]
            action_emb = action_emb_flat.view(B, K, T, -1)  # [B, K, T, hidden]

            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.action_in_proj.out_features, self.min_period, self.max_period, device=device
            )  # [B, hidden]
            time_emb_expanded = time_emb.unsqueeze(1).unsqueeze(2).expand_as(action_emb)  # [B, K, T, hidden]

            fused_input = torch.cat([action_emb, time_emb_expanded], dim=-1).to(dtype)  # [B, K, T, 2*hidden]

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            fused_flat = fused_input.view(B * K, T, -1)  # [B*K, T, 2*hidden]
            action_time_emb_flat = self._apply_checkpoint(mlp_func, fused_flat)  # [B*K, T, hidden]
            action_time_emb_all = action_time_emb_flat.view(B, K, T, -1)  # [B, K, T, hidden]

            mode_embeddings_all = action_time_emb_all.view(B, K * T, -1)  # [B, K*T, hidden]

            cls_token_base = self.traj_cls_token.expand(B, -1, -1).to(dtype)  # [B, 1, hidden]
            status_context = status_emb[:, 0:1, :]  # [B, 1, hidden] - first status token
            cls_token = cls_token_base + status_context  # Give classifier contextual hint

            suffix_emb = torch.cat([status_emb, mode_embeddings_all, cls_token], dim=1)  # [B, num_status_tokens + K*T + 1, hidden]

        else:
            if actions.dtype != dtype:
                actions = actions.to(dtype=dtype)

            action_emb = self._apply_checkpoint(self.action_in_proj, actions)  # [B, T, hidden]

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

            suffix_emb = torch.cat([status_emb, action_time_emb], dim=1)  # [B, num_status_tokens + T, hidden]

        suffix_len = suffix_emb.shape[1]
        pad_masks = torch.ones((B, suffix_len), dtype=torch.bool, device=device)

        # att_marks: first status token(s) can be attended by all
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
        gt_occ_dense=None,
        hist_traj=None,
        ar_batch=None,  # AR Cotraining: JSONL data batch
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

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            hist_traj=hist_traj,
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

        if self.use_cma_flow:

            cmd = gt_ego_fut_cmd.argmax(dim=-1) if gt_ego_fut_cmd.dim() > 1 else gt_ego_fut_cmd

            if not self._cma_flow_cmd_verified:
                unique_cmds, counts = torch.unique(cmd, return_counts=True)
                print(f"\n[CMA-Flow] First batch command distribution:")
                for c, count in zip(unique_cmds.tolist(), counts.tolist()):
                    cmd_name = self.cma_flow_prior['cmd_names'][c] if c < len(self.cma_flow_prior['cmd_names']) else 'unknown'
                    print(f"  cmd={c} ({cmd_name:8s}): {count} samples")
                print(f"  Expected: 0=right, 1=left, 2=straight")
                print(f"  If 'left' has most samples, mapping is correct ✅")
                self._cma_flow_cmd_verified = True

            x_0_modes = self._sample_from_cma_prior(cmd, actions.device, actions.dtype)  # [B, K, T, D] - Already normalized in prior
            K_per_cmd = x_0_modes.shape[1]

            time = self.sample_time(bsz, actions.device)  # [B]

            t_expanded = time[:, None, None, None]  # [B, 1, 1, 1]
            actions_expanded = actions[:, None, :, :]  # [B, 1, T, D]

            # Align with single-mode: t=1→prior, t=0→GT
            x_t_modes = t_expanded * x_0_modes + (1 - t_expanded) * actions_expanded  # [B, K, T, D]
            u_t_modes = x_0_modes - actions_expanded  # [B, K, T, D]

            anchor_dist = torch.norm(x_0_modes - actions_expanded, dim=(-2, -1))  # [B, K]
            k_star = torch.argmin(anchor_dist, dim=1)  # [B] - stable WTA target!
        else:
            # Single-mode Flow Matching: Sample from random noise
            noise = self.sample_noise(actions.shape, actions.device)
            time = self.sample_time(actions.shape[0], actions.device)

            t = time[:, None, None]
            x_t = t * noise + (1 - t) * actions
            u_t = noise - actions

            # Single mode: set K=1 for consistency
            K_per_cmd = 1
            x_t_modes = x_t.unsqueeze(1)  # [B, 1, T, D]
            u_t_modes = u_t.unsqueeze(1)  # [B, 1, T, D]

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features = self.embed_prefix(batch)

        # Prepare feature maps for unified decoder
        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, all_image_grids, self.feature_source
        )
        feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
        if not self.feat_grad:
            feature_maps_daf = [x.detach() for x in feature_maps_daf]
        # Free large vision feature tensors — no longer needed after feature_maps_daf is built
        del deepstack_features, raw_features, source_features, feature_maps

        # Prepare perception metadata
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

        # CMA-Flow: embed K modes in sequence dimension
        if self.use_cma_flow:

            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                batch, x_t_modes, time, ego_status_pred=None, use_gt_status=True, hist_traj=hist_traj,
            )
        else:
            # Single-mode: [B, num_status_tokens + T, hidden]
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                batch, x_t, time, ego_status_pred=None, use_gt_status=True, hist_traj=hist_traj,
            )

        if self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            perception_embs = perception_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Prefix position IDs
        prefix_pos_ids, rope_deltas = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        # Perception position IDs
        perception_len = perception_embs.shape[1]
        perception_range = torch.arange(1, perception_len + 1, device=actions.device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values
        perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

        # Suffix position IDs
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

        if self.attn_implementation == "flex" and not self.enable_knowledge_insulation:
            block_mask, q_len_rounded = build_blockmask_unidrive(
                bsz=bsz,
                hq=self.qwen3_vl_with_expert.qwen3_vl.config.text_config.num_attention_heads,
                prefix_len=prefix_len,
                occ_len=perception_len,
                suffix_len=suffix_len,
                device=actions.device,
                compile_blockmask=True,
                prompt_only_len=-1,
            )
            att_mask_input = block_mask
        else:
            pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, perception_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

            if self.use_cma_flow:
                num_status_tokens = 2 if hist_traj is not None else 1
                K = K_per_cmd  # from earlier CMA-Flow sampling

                suffix_block_mask = self._create_cma_flow_suffix_mask(
                    batch_size=bsz,
                    num_status_tokens=num_status_tokens,
                    K=K,
                    T=self.action_horizon,
                    device=actions.device,
                )  # [B, 1, suffix_len, suffix_len]

                total_len = att_2d_masks.shape[1]
                suffix_start = prefix_len + perception_len

                att_2d_masks[:, suffix_start:, suffix_start:] = suffix_block_mask.squeeze(1)

            att_mask_input = self._prepare_attention_masks_4d(att_2d_masks)

        total_loss = 0.0
        stats = {}
        ar_loss = 0.0
        perception_out = None

        if self.enable_knowledge_insulation:
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            prefix_pos_ids_sliced = position_ids[..., :prefix_len]
            
            # Single Decoder: Process prefix only
            prefix_out, past_key_values, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=prefix_att_4d,
                position_ids=prefix_pos_ids_sliced,
                past_key_values=None,
                inputs_embeds=prefix_embs,
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
                ignore_index=-100
            )
            stats["loss_ar"] = ar_loss

            detached_past_kv = []
            for layer_kv in past_key_values:
                detached_past_kv.append((layer_kv[0].detach(), layer_kv[1].detach()))
            
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_mixed = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            suffix_att_4d = self._prepare_attention_masks_4d(full_att_2d_mixed)
            suffix_pos_ids_sliced = position_ids[..., prefix_len:]
            
            # Single Decoder: Process suffix with KV cache from prefix
            suffix_out, _, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=suffix_att_4d,
                position_ids=suffix_pos_ids_sliced,
                past_key_values=detached_past_kv,
                inputs_embeds=suffix_embs,
                use_cache=False,
            )
        else:
            def forward_func(prefix_embs, perception_embs, suffix_embs, att_mask, position_ids, _unused_return_middle_layers, q_len_rnd):
                # Single Decoder: Concatenate all embeddings and process together
                all_embs = torch.cat([prefix_embs, perception_embs, suffix_embs], dim=1)
                # Store lengths before freeing references
                prefix_len = prefix_embs.shape[1]
                perception_len = perception_embs.shape[1]
                suffix_len = suffix_embs.shape[1]
                del prefix_embs, perception_embs, suffix_embs

                output, _, middle_layer_outputs = self.qwen3_vl_with_expert.forward(
                    attention_mask=att_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=all_embs,
                    use_cache=False,
                    return_middle_layers=None,
                    q_len_rounded=q_len_rnd
                )
                del all_embs

                # Split output back to prefix/perception/suffix
                prefix_out = output[:, :prefix_len]
                perception_out = output[:, prefix_len:prefix_len+perception_len]
                suffix_out = output[:, prefix_len+perception_len:prefix_len+perception_len+suffix_len]
                del output

                return [prefix_out, perception_out, suffix_out], middle_layer_outputs

            (outputs_embeds, _middle_layer_outs_unused) = self._apply_checkpoint(
                forward_func,
                prefix_embs,
                perception_embs,
                suffix_embs,
                att_mask_input,
                position_ids,
                None,
                q_len_rounded
            )
            prefix_out, perception_out, suffix_out = outputs_embeds

        loss_depth_val = self.compute_depth_loss(
            prefix_embs, prefix_out, prefix_input_ids, all_image_grids, gt_depth, focal, bsz
        )
        # prefix_embs and prefix_out are no longer needed after depth loss
        del prefix_embs, prefix_out

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
            del perception_out  # slices already extracted, free the large tensor
            del perception_embs  # no longer needed after slicing

            target_dtype = self.det_proj.weight.dtype

            # Extract Stage1 features for residual connection
            det_feat_stage1 = stage1_outs['det_instance_feature']
            map_feat_stage1 = stage1_outs['map_instance_feature']
            ego_feat_stage1 = stage1_outs['ego_instance_feature']
            motion_token_256 = stage1_outs.get('motion_token', None)

            # Get projection layer dtype
            proj_dtype = self.det_proj.weight.dtype

            # Det fusion
            det_vlm_proj = self.det_proj(det_out_vlm.to(proj_dtype)).to(torch.float32)
            if self.vlm_fusion_type == 'direct':
                # Direct replacement: use VLM output only, ignore stage1
                det_feat_fused = det_vlm_proj
            elif self.vlm_fusion_type == 'cross_attention':
                # Cross-attention fusion: stage1 as Q, vlm as K/V
                det_delta = self.det_fusion(det_feat_stage1.to(torch.float32), det_vlm_proj)
                det_feat_fused = det_feat_stage1.to(torch.float32) + det_delta
            else:
                # MLP fusion: concat and residual
                det_concat = torch.cat([det_feat_stage1.to(torch.float32), det_vlm_proj], dim=-1)
                det_delta = self.det_fusion(det_concat)
                det_feat_fused = det_feat_stage1.to(torch.float32) + det_delta

            # Map fusion
            map_vlm_proj = self.map_proj(map_out_vlm.to(proj_dtype)).to(torch.float32)
            if self.vlm_fusion_type == 'direct':
                map_feat_fused = map_vlm_proj
            elif self.vlm_fusion_type == 'cross_attention':
                map_delta = self.map_fusion(map_feat_stage1.to(torch.float32), map_vlm_proj)
                map_feat_fused = map_feat_stage1.to(torch.float32) + map_delta
            else:
                map_concat = torch.cat([map_feat_stage1.to(torch.float32), map_vlm_proj], dim=-1)
                map_delta = self.map_fusion(map_concat)
                map_feat_fused = map_feat_stage1.to(torch.float32) + map_delta

            # Ego fusion
            if ego_out is not None:
                ego_vlm_proj = self.ego_proj_down(ego_out.to(proj_dtype)).to(torch.float32)
                if self.vlm_fusion_type == 'direct':
                    ego_feat_fused = ego_vlm_proj
                elif self.vlm_fusion_type == 'cross_attention':
                    ego_delta = self.ego_fusion(ego_feat_stage1.to(torch.float32), ego_vlm_proj)
                    ego_feat_fused = ego_feat_stage1.to(torch.float32) + ego_delta
                else:
                    ego_concat = torch.cat([ego_feat_stage1.to(torch.float32), ego_vlm_proj], dim=-1)
                    ego_delta = self.ego_fusion(ego_concat)
                    ego_feat_fused = ego_feat_stage1.to(torch.float32) + ego_delta
            else:
                ego_feat_fused = ego_feat_stage1.to(torch.float32)

            # Motion fusion (if available)
            if motion_token_256 is not None:
                if self.motion_fusion is not None:
                    motion_vlm_proj = self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                    if self.vlm_fusion_type == 'direct':
                        motion_feat_fused = motion_vlm_proj
                    elif self.vlm_fusion_type == 'cross_attention':
                        motion_delta = self.motion_fusion(motion_token_256.to(torch.float32), motion_vlm_proj)
                        motion_feat_fused = motion_token_256.to(torch.float32) + motion_delta
                    else:
                        motion_concat = torch.cat([motion_token_256.to(torch.float32), motion_vlm_proj], dim=-1)
                        motion_delta = self.motion_fusion(motion_concat)
                        motion_feat_fused = motion_token_256.to(torch.float32) + motion_delta
                else:
                    # No fusion module (direct mode without motion_fusion)
                    if self.vlm_fusion_type == 'direct':
                        motion_vlm_proj = self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                        motion_feat_fused = motion_vlm_proj
                    else:
                        motion_feat_fused = motion_token_256.to(torch.float32)
            else:
                motion_feat_fused = None

            # Build vlm_enhanced dict with fused features
            vlm_enhanced = {
                'det_feat': det_feat_fused.to(target_dtype),
                'map_feat': map_feat_fused.to(target_dtype),
                'ego_feat': ego_feat_fused.to(target_dtype),
            }
            if motion_feat_fused is not None:
                vlm_enhanced['motion_feat'] = motion_feat_fused.to(target_dtype)

            stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)

            # Compute unified decoder perception losses
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
            del stage1_outs, stage2_outs  # free large perception state dicts
        else:
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=actions.device, dtype=torch.float32)
            ego_out = None

        if self.use_cma_flow and K_per_cmd > 1:
            num_status_tokens = 2 if hist_traj is not None else 1
            K = K_per_cmd
            T = self.action_horizon

            mode_start = num_status_tokens
            mode_end = num_status_tokens + K * T
            mode_outputs = suffix_out[:, mode_start:mode_end, :]  # [B, K*T, hidden]

            mode_outputs = mode_outputs.reshape(bsz, K, T, -1)

            model_out = self._apply_checkpoint(self.action_out_proj, mode_outputs.to(torch.float32))  # [B, K, T, D]

            # HDP: Target resolution (τ₀-prediction vs velocity-prediction)
            if self.use_tau0_pred:
                # Network predicts clean trajectory directly
                pred_x_gt_modes = model_out  # [B, K, T, D]
            else:
                # Network predicts velocity field v_t
                # Since x_t = t * x_prior + (1-t) * x_gt and u_t = x_prior - x_gt
                # We can infer: pred_x_gt = x_prior - pred_v_t
                pred_x_gt_modes = x_0_modes - model_out  # [B, K, T, D]

            # Select best mode for loss
            pred_x_gt = pred_x_gt_modes[torch.arange(bsz, device=model_out.device), k_star]  # [B, T, D]
            gt_x = actions  # [B, T, D] - GT clean trajectory in normalized space

            # HDP: Loss computation
            if self.use_hdp_hybrid_loss:
                # Use HDP hybrid loss (velocity + waypoint)
                planning_loss = self.compute_hdp_hybrid_loss(pred_x_gt, gt_x)
            elif self.use_tau0_pred:
                # Direct τ₀ prediction with MSE
                planning_loss = F.mse_loss(pred_x_gt, gt_x)
            else:
                # Legacy: use original loss_planning if available
                if self.loss_planning is not None:
                    u_t = u_t_modes[torch.arange(bsz, device=u_t_modes.device), k_star]  # [B, T, D]
                    v_t = model_out[torch.arange(bsz, device=model_out.device), k_star]  # [B, T, D]
                    planning_loss = self.loss_planning(u_t, v_t, time, gt_ego_fut_masks=gt_ego_fut_masks)
                else:
                    planning_loss = torch.tensor(0.0, device=model_out.device)

            cls_out = suffix_out[:, -1, :].contiguous().to(dtype=torch.float32)

            cls_logits = self.traj_cls_head(cls_out)

            traj_cls_loss = F.cross_entropy(cls_logits, k_star)

        else:
            # Single-mode or K=1 case
            suffix_out_float = suffix_out[:, -self.action_horizon :].to(dtype=torch.float32)
            model_out = self._apply_checkpoint(self.action_out_proj, suffix_out_float)  # [B, T, D]

            # HDP: Target resolution
            if self.use_tau0_pred:
                # Network predicts clean trajectory directly
                pred_x_gt = model_out  # [B, T, D]
            else:
                # Network predicts velocity field v_t
                if self.use_cma_flow:
                    # Single mode from CMA-Flow
                    pred_x_gt = x_0_modes[:, 0] - model_out  # [B, T, D]
                else:
                    # Pure single-mode flow matching
                    pred_x_gt = noise - model_out  # [B, T, D]

            gt_x = actions  # [B, T, D]

            # HDP: Loss computation
            if self.use_hdp_hybrid_loss:
                planning_loss = self.compute_hdp_hybrid_loss(pred_x_gt, gt_x)
            elif self.use_tau0_pred:
                planning_loss = F.mse_loss(pred_x_gt, gt_x)
            else:
                # Legacy: use original loss_planning
                if self.loss_planning is not None:
                    if self.use_cma_flow:
                        u_t = u_t_modes[:, 0]  # [B, T, D]
                    else:
                        u_t = u_t_modes.squeeze(1) if u_t_modes.dim() == 4 else u_t_modes
                    v_t = model_out
                    planning_loss = self.loss_planning(u_t, v_t, time, gt_ego_fut_masks=gt_ego_fut_masks)
                else:
                    planning_loss = torch.tensor(0.0, device=model_out.device)

            traj_cls_loss = torch.tensor(0.0, device=model_out.device)

        stats["loss_planning"] = planning_loss

        if gt_occ_dense is not None and occ_out is not None:
            occ_target = gt_occ_dense.permute(0, 3, 1, 2).long()
            occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            loss_occ = F.cross_entropy(occ_logits, occ_target)
        else:
            loss_occ = occ_out.sum() * 0.0
        stats["loss_occ"] = loss_occ

        # Motion loss already computed in perception_losses from unified decoder (already weighted)
        loss_motion = stats.get("loss_motion", occ_out.sum() * 0.0)

        # Prevents training instability from unbalanced loss scales
        weighted_traj_cls_loss = self.loss_traj_cls_weight * traj_cls_loss

        # AR Cotraining: Unified AR loss mechanism
        # Supports two modes: 1) JSONL data (external), 2) text-only (internal)
        ar_loss_total = torch.tensor(0.0, device=planning_loss.device)
        if self.train_vlm:
            if ar_batch is not None:
                # Mode 1: Use external JSONL data for AR cotraining (new approach)
                try:
                    ar_outputs = self.forward_ar_batch(ar_batch)
                    ar_loss_total = ar_outputs['loss_ar']
                    stats['loss_ar'] = ar_loss_total.detach()
                    stats['loss_vlm_raw'] = ar_outputs['loss_vlm_raw']
                    stats['ar_planning_ratio'] = (ar_outputs['loss_vlm_raw'] / (planning_loss.detach() + 1e-6)).item()
                except Exception as e:
                    print(f"[AR Cotraining] Error in forward_ar_batch: {e}")
                    # Fallback: skip AR loss for this iteration
                    ar_loss_total = torch.tensor(0.0, device=planning_loss.device)
            elif self.enable_knowledge_insulation:
                # Mode 2: Use text-only AR (existing approach, backward compatible)
                # ar_loss is already computed in the existing code (Line 1723-1749)
                ar_loss_total = self.ar_loss_weight * ar_loss

        # ========================================================================
        # ✅ FIX: Let MMCV auto-sum atomic losses. Do NOT return aggregated losses!
        # MMCV will automatically compute: loss = sum(all loss_* values)
        #
        # Return only atomic losses:
        # - det_loss_cls_0, det_loss_box_0, ... (from unified_decoder)
        # - map_loss_cls_0, map_loss_line_0, ... (from unified_decoder)
        # - loss_ego_status (from unified_decoder)
        # - loss_motion (from unified_decoder)
        # - loss_planning, loss_occ, loss_depth, loss_traj_cls (from this head)
        #
        # Do NOT return:
        # - loss_perception (aggregation of det+map+ego, causes duplication!)
        # - loss (MMCV will compute this automatically!)
        # ========================================================================

        # Add atomic losses
        stats["loss_planning"] = planning_loss
        stats["loss_occ"] = self.occ_loss_weight * loss_occ
        stats["loss_depth"] = loss_depth_val
        stats["loss_traj_cls"] = weighted_traj_cls_loss

        # Motion loss already in stats from unified_decoder
        stats.setdefault("loss_motion", loss_motion)

        # AR loss (if enabled)
        if ar_loss_total.item() != 0.0:
            stats["loss_ar"] = ar_loss_total

        # Remove aggregated losses that cause duplication
        stats.pop('loss_perception', None)  # Remove if exists
        stats.pop('loss', None)  # Remove if exists (MMCV will compute)

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

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            hist_traj=hist_traj,
        )
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        num_steps = int(self.num_sample_steps if num_steps is None else num_steps)

        if noise is None:
            noise = self.sample_noise((bsz, self.action_horizon, self.action_dim), device)

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features = self.embed_prefix(batch)

        if prefix_embs.dtype != dtype:
            prefix_embs = prefix_embs.to(dtype)

        # Project and reshape VLM features to 4D feature maps
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
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d)

        prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "eager"

        # Single Decoder: Process prefix and cache KV
        _, past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=True,
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

        # Single Decoder: Process perception tokens with cached prefix
        perception_out, past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=perception_full_att_2d_4d,
            position_ids=perception_pos_ids_1d,  # Use 1D position IDs for single decoder
            past_key_values=past_key_values,
            inputs_embeds=perception_embs,
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

        if perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            occ_out = perception_out[:, o0:o1]
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]

            target_dtype = self.det_proj.weight.dtype

            det_feat_stage1 = stage1_outs['det_instance_feature']
            map_feat_stage1 = stage1_outs['map_instance_feature']
            ego_feat_stage1 = stage1_outs['ego_instance_feature']
            motion_token_256 = stage1_outs.get('motion_token', None)

            proj_dtype = self.det_proj.weight.dtype

            det_vlm_proj = self.det_proj(det_out_vlm.to(proj_dtype)).to(torch.float32)
            if self.vlm_fusion_type == 'direct':
                det_feat_fused = det_vlm_proj
            elif self.vlm_fusion_type == 'cross_attention':
                det_delta = self.det_fusion(det_feat_stage1.to(torch.float32), det_vlm_proj)
                det_feat_fused = det_feat_stage1.to(torch.float32) + det_delta
            else:
                det_concat = torch.cat([det_feat_stage1.to(torch.float32), det_vlm_proj], dim=-1)
                det_delta = self.det_fusion(det_concat)
                det_feat_fused = det_feat_stage1.to(torch.float32) + det_delta

            # Map fusion
            map_vlm_proj = self.map_proj(map_out_vlm.to(proj_dtype)).to(torch.float32)
            if self.vlm_fusion_type == 'direct':
                map_feat_fused = map_vlm_proj
            elif self.vlm_fusion_type == 'cross_attention':
                map_delta = self.map_fusion(map_feat_stage1.to(torch.float32), map_vlm_proj)
                map_feat_fused = map_feat_stage1.to(torch.float32) + map_delta
            else:
                map_concat = torch.cat([map_feat_stage1.to(torch.float32), map_vlm_proj], dim=-1)
                map_delta = self.map_fusion(map_concat)
                map_feat_fused = map_feat_stage1.to(torch.float32) + map_delta

            # Ego fusion
            if ego_out is not None:
                ego_vlm_proj = self.ego_proj_down(ego_out.to(proj_dtype)).to(torch.float32)
                if self.vlm_fusion_type == 'direct':
                    ego_feat_fused = ego_vlm_proj
                elif self.vlm_fusion_type == 'cross_attention':
                    ego_delta = self.ego_fusion(ego_feat_stage1.to(torch.float32), ego_vlm_proj)
                    ego_feat_fused = ego_feat_stage1.to(torch.float32) + ego_delta
                else:
                    ego_concat = torch.cat([ego_feat_stage1.to(torch.float32), ego_vlm_proj], dim=-1)
                    ego_delta = self.ego_fusion(ego_concat)
                    ego_feat_fused = ego_feat_stage1.to(torch.float32) + ego_delta
            else:
                ego_feat_fused = ego_feat_stage1.to(torch.float32)

            # Motion fusion (if available)
            if motion_token_256 is not None:
                if self.motion_fusion is not None:
                    motion_vlm_proj = self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                    if self.vlm_fusion_type == 'direct':
                        motion_feat_fused = motion_vlm_proj
                    elif self.vlm_fusion_type == 'cross_attention':
                        motion_delta = self.motion_fusion(motion_token_256.to(torch.float32), motion_vlm_proj)
                        motion_feat_fused = motion_token_256.to(torch.float32) + motion_delta
                    else:
                        motion_concat = torch.cat([motion_token_256.to(torch.float32), motion_vlm_proj], dim=-1)
                        motion_delta = self.motion_fusion(motion_concat)
                        motion_feat_fused = motion_token_256.to(torch.float32) + motion_delta
                else:
                    if self.vlm_fusion_type == 'direct':
                        motion_vlm_proj = self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                        motion_feat_fused = motion_vlm_proj
                    else:
                        motion_feat_fused = motion_token_256.to(torch.float32)
            else:
                motion_feat_fused = None

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

        if self.use_cma_flow:

            cmd = gt_ego_fut_cmd.argmax(dim=-1) if gt_ego_fut_cmd.dim() > 1 else gt_ego_fut_cmd
            x_0_modes = self._sample_from_cma_prior(cmd, device, torch.float32)  # [B, K, T, D] - Already normalized in prior
            K_per_cmd = x_0_modes.shape[1]

            # Align with single-mode: t=1→prior, t=0→GT
            # Training: x_t = t * prior + (1-t) * GT, at t=1: x_1 = prior
            # Inference: start from t=1 (prior), denoise to t=0 (GT)
            x_t_modes = x_0_modes  # Start from t=1 with prior

            time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)
            ego_status_pred = stage2_outs['ego_status_list'][-1].squeeze(1).to(torch.float32) if (
                stage2_outs and 'ego_status_list' in stage2_outs and len(stage2_outs['ego_status_list']) > 0
            ) else x_t_modes.new_zeros((bsz, self.ego_status_dim), dtype=torch.float32)
            max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values

            # Denoise from t=1 (prior) to t=0 (GT), dt < 0
            while time_tensor >= -dt_val / 2:
                expanded_time = time_tensor.expand(bsz)

                v_t_modes = self._denoise_step_multimode(
                    batch,
                    cached_pad_masks,
                    past_key_values,
                    x_t_modes.to(dtype),
                    expanded_time.to(dtype),
                    max_perception_pos,
                    K=K_per_cmd,
                    ego_status_pred=ego_status_pred,
                    hist_traj=hist_traj,
                )  # [B, K, T, D]

                x_t_modes = x_t_modes + dt_val * v_t_modes
                time_tensor += dt_val

            time_zero = torch.zeros(bsz, dtype=torch.float32, device=device)

            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                batch, x_t_modes, time_zero,
                ego_status_pred=ego_status_pred,
                use_gt_status=False,
                hist_traj=hist_traj
            )  # [B, num_status_tokens + K*T + 1, hidden]

            if suffix_embs.dtype != dtype:
                suffix_embs = suffix_embs.to(dtype)

            suffix_len = suffix_embs.shape[1]
            suffix_range = torch.arange(1, suffix_len + 1, device=device).view(1, -1).expand(bsz, -1)
            suffix_pos_ids_1d = max_perception_pos + suffix_range
            suffix_pos_ids_3d = torch.stack([suffix_pos_ids_1d] * 3, dim=0)

            perception_prefix_len = prefix_len + perception_len
            suffix_pad_2d = cached_pad_masks[:, None, :].expand(bsz, suffix_len, perception_prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            suffix_full_att_2d = torch.cat([suffix_pad_2d, suffix_att_2d], dim=2)

            num_status_tokens = 2 if hist_traj is not None else 1
            suffix_block_mask = self._create_cma_flow_suffix_mask(
                batch_size=bsz,
                num_status_tokens=num_status_tokens,
                K=K_per_cmd,
                T=self.action_horizon,
                device=device,
            )  # [B, 1, suffix_len, suffix_len]

            suffix_full_att_2d[:, :, perception_prefix_len:] = suffix_block_mask.squeeze(1)

            suffix_full_att_2d_4d = self._prepare_attention_masks_4d(suffix_full_att_2d)

            # Single Decoder: Process action tokens with cached prefix+perception
            suffix_out, _, _ = self.qwen3_vl_with_expert.forward(
                attention_mask=suffix_full_att_2d_4d,
                position_ids=suffix_pos_ids_1d,  # Use 1D position IDs for single decoder
                past_key_values=past_key_values,
                inputs_embeds=suffix_embs,
                use_cache=False,
            )

            cls_out = suffix_out[:, -1, :].contiguous().to(dtype=torch.float32)

            mode_logits = self.traj_cls_head(cls_out)

            use_oracle_selection = False

            if use_oracle_selection:
                gt_trajs_norm = self.norm_delta(gt_ego_fut_trajs[:, :self.action_horizon, :])
                gt_trajs_norm = gt_trajs_norm[..., :2]
                x_t_positions = x_t_modes[..., :2]
                distances = torch.norm(x_t_positions - gt_trajs_norm.unsqueeze(1), dim=(-2, -1))
                best_mode_idx = torch.argmin(distances, dim=1)
                print(f"[Oracle Mode Selection] Using GT-nearest mode. Normalized distances: {distances[0].cpu().numpy()}")
            else:
                best_mode_idx = torch.argmax(mode_logits, dim=1)

            x_t = x_t_modes[torch.arange(bsz, device=device), best_mode_idx]

        else:
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
                    batch, cached_pad_masks, past_key_values, x_t.to(dtype), expanded_time.to(dtype), max_perception_pos, ego_status_pred=ego_status_pred, hist_traj=hist_traj,
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

    def _denoise_step(self, batch, cached_pad_masks, past_key_values, x_t, timestep, max_cached_position_ids, *, ego_status_pred: Optional[torch.Tensor] = None, hist_traj=None):

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch,
            x_t,
            timestep,
            ego_status_pred=ego_status_pred,
            use_gt_status=False,
            hist_traj=hist_traj,
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
        
        position_ids = torch.stack([
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
        ], dim=0)

        # Single Decoder: Process action tokens
        suffix_out, _, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=suffix_embs,
            use_cache=False,
        )

        suffix_out = suffix_out[:, -self.action_horizon :].to(dtype=torch.float32)

        model_out = self.action_out_proj(suffix_out)  # [B, T, D]

        # HDP: Convert τ₀-prediction to velocity for ODE integration
        if self.use_tau0_pred:
            # model_out is pred_x_gt (clean trajectory)
            # v_t = (x_t - pred_x_gt) / t
            t_safe = timestep.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-4)  # [B, 1, 1]
            v_t = (x_t - model_out) / t_safe
            return v_t
        else:
            # model_out is already v_t (velocity field)
            return model_out

    def _denoise_step_multimode(
        self,
        batch,
        cached_pad_masks,
        past_key_values,
        x_t_modes,  # [B, K, T, D]
        timestep,
        max_cached_position_ids,
        K: int,
        *,
        ego_status_pred: Optional[torch.Tensor] = None,
        hist_traj=None
    ):
        """
        This is the memory-efficient version for inference:
        - Keep batch size as B (not B*K)
        - Concatenate K modes in sequence dimension
        - Use block diagonal attention mask so modes don't interfere
        - Only call VLM once instead of K times

        Memory: O(B * K*T) instead of O(B*K * T)
        Speed: 1 VLM call instead of K calls

        Args:
            x_t_modes: [B, K, T, D] - K trajectories per batch
            K: Number of modes

        Returns:
            v_t_modes: [B, K, T, D] - Velocity for each mode
        """
        B, K_in, T, D = x_t_modes.shape
        assert K_in == K, f"K mismatch: {K_in} vs {K}"
        device = x_t_modes.device


        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch,
            x_t_modes,
            timestep,
            ego_status_pred=ego_status_pred,
            use_gt_status=False,
            hist_traj=hist_traj,
        )

        suffix_len = suffix_embs.shape[1]
        cached_len = cached_pad_masks.shape[1]
        num_status_tokens = 2 if hist_traj is not None else 1

        cached_pad_2d = cached_pad_masks[:, None, :].expand(B, suffix_len, cached_len)

        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        for k in range(K):
            mode_start = num_status_tokens + k * T
            mode_end = num_status_tokens + (k + 1) * T

            mask_template = torch.zeros((B, T, K * T), dtype=torch.bool, device=device)
            mask_template[:, :, k*T:(k+1)*T] = True  # Enable block k
            suffix_att_2d[:, mode_start:mode_end, num_status_tokens:num_status_tokens + K * T] = mask_template

        full_att_2d = torch.cat([cached_pad_2d, suffix_att_2d], dim=2)  # [B, suffix_len, cached_len + suffix_len]
        full_att_2d_4d = self._prepare_attention_masks_4d(full_att_2d)

        suffix_range = torch.arange(1, suffix_len + 1, device=device).view(1, -1).expand(B, -1)
        suffix_pos_ids_1d = max_cached_position_ids + suffix_range

        position_ids = torch.stack([
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
        ], dim=0)

        # Single Decoder: Process multi-mode action tokens (no separate action expert)
        suffix_out, _, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,  # ✅ Use original KV cache, NOT expanded
            inputs_embeds=suffix_embs,
            use_cache=False,
        )

        traj_out = suffix_out[:, num_status_tokens:-1, :].to(dtype=torch.float32)

        traj_out_modes = traj_out.reshape(B, K, T, -1)

        model_out = torch.stack([
            self.action_out_proj(traj_out_modes[:, k, :, :])  # [B, T, D]
            for k in range(K)
        ], dim=1)  # [B, K, T, D]

        # HDP: Convert τ₀-prediction to velocity for ODE integration
        if self.use_tau0_pred:
            # model_out is pred_x_gt_modes (clean trajectories)
            # v_t = (x_t - pred_x_gt) / t
            t_safe = timestep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-4)  # [B, 1, 1, 1]
            v_t_modes = (x_t_modes - model_out) / t_safe
            return v_t_modes
        else:
            # model_out is already v_t_modes (velocity field)
            return model_out

    def forward_ar_batch(self, ar_batch):
        """
        AR (Autoregressive) forward pass for language modeling cotraining

        This method is called during training when ar_batch is provided.
        It performs VLM language modeling on the AR data (JSONL QA samples)
        to provide additional regularization and improve instruction understanding.

        Args:
            ar_batch: dict with keys:
                - ar_input_ids: [B_ar, L] Tokenized input IDs
                - ar_labels: [B_ar, L] Labels for language modeling loss
                - ar_pixel_values: List[List[Tensor]]  # [B][6 views][C,H,W]
                - ar_image_grid_thw: [B_ar, 6, 3] Image grid sizes

        Returns:
            dict with keys:
                - loss_ar: Weighted AR loss (ar_loss_weight * loss_vlm)
                - loss_vlm_raw: Raw VLM language modeling loss (for logging)

        Reference: Plan document Section 2.1
        """
        input_ids = ar_batch['ar_input_ids']  # [B, L]
        labels = ar_batch['ar_labels']  # [B, L]
        pixel_values_list = ar_batch['ar_pixel_values']  # List[List[Tensor]]
        image_grid_thw = ar_batch['ar_image_grid_thw']  # [B, 6, 3]

        B = input_ids.shape[0]

        # Step 1: Flatten pixel_values from [B][6 views] to [B*6]
        # Each sample has 6 views, we need to flatten them for VLM processing
        flat_pixel_values = []
        flat_image_grid_thw = []

        for i in range(len(pixel_values_list)):
            for view_tensor in pixel_values_list[i]:  # 6 views per sample
                flat_pixel_values.append(view_tensor)
            for thw in image_grid_thw[i]:  # 6 thw vectors per sample
                flat_image_grid_thw.append(thw)

        flat_image_grid_thw = torch.stack(flat_image_grid_thw)  # [B*6, 3]

        # Step 2: Build attention_mask and position_ids
        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.qwen3_vl_with_expert.qwen3_vl.processor.tokenizer.pad_token_id).long()

        # CRITICAL: Handle 3D RoPE position IDs
        # Qwen2.5-VL uses 3D RoPE (temporal, height, width dimensions)
        try:
            from .rope2d import get_rope_index_25
            position_ids, _ = get_rope_index_25(
                merge_size=14,  # Qwen2.5-VL patch size (14x14)
                input_ids=input_ids,
                image_grid_thw=flat_image_grid_thw,
            )
        except Exception as e:
            # Fallback: use simple sequential position IDs
            print(f"[AR] Warning: Failed to compute 3D RoPE position IDs: {e}")
            print("[AR] Falling back to simple position IDs")
            position_ids = torch.arange(
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0).expand(B, -1)

        # Step 3: Forward pass through qwen3_vl (language modeling)
        # Only use qwen3_vl, do NOT go through Unified Decoder
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            try:
                outputs = self.qwen3_vl_with_expert.qwen3_vl(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=flat_pixel_values,  # List[Tensor], length B*6
                    image_grid_thw=flat_image_grid_thw,  # [B*6, 3]
                    position_ids=position_ids,  # [B, L] or [B, L, 3]
                    labels=labels,  # CrossEntropyLoss computed internally
                    use_cache=False,
                )
            except Exception as e:
                print(f"[AR] Error in VLM forward pass: {e}")
                print(f"[AR] input_ids shape: {input_ids.shape}")
                print(f"[AR] pixel_values length: {len(flat_pixel_values)}")
                print(f"[AR] image_grid_thw shape: {flat_image_grid_thw.shape}")
                raise

        loss_vlm = outputs.loss  # CrossEntropyLoss on next-token prediction

        # Step 4: Apply loss weight
        scaled_loss = self.ar_loss_weight * loss_vlm

        return dict(
            loss_ar=scaled_loss,
            loss_vlm_raw=loss_vlm.detach(),
        )



