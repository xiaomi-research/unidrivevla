from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.core import reduce_mean
from mmdet.models.builder import build_loss

from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
)
from projects.mmdet3d_plugin.models.blocks import AsymmetricFFN, DeformableFeatureAggregation
from projects.mmdet3d_plugin.models.attention import MultiheadFlashAttention
from projects.mmdet3d_plugin.models.detection3d.decoder import SparseBox3DDecoder
from projects.mmdet3d_plugin.models.detection3d.target import SparseBox3DTarget
from projects.mmdet3d_plugin.models.detection3d.losses import SparseBox3DLoss


@dataclass
class DetHeadOutputs:
    cls_scores: List[Tensor]
    box_preds: List[Tensor]
    quality: List[Optional[Tensor]]
    last_feat: Optional[Tensor] = None




class DetVLAHead(nn.Module):
    """SparseDrive-aligned detection head.

    Design (aligned with SparseDrive Sparse4DHead):
      - Uses VLM's global attention for image feature interaction (instead of deformable)
      - Runs GNN + FFN + refinement on det queries with decouple attention.
      - Supports temporal instance interaction via temp_gnn.
      - Training uses SparseBox3DTarget + SparseBox3DLoss (SparseDrive style).
      - Inference uses SparseBox3DDecoder.decode => boxes_3d/scores_3d/labels_3d.

    Note:
      - `output_dim` can be 11 (x,y,z,w,l,h,sin,cos,vx,vy,vz) for compatibility with UniDriveVLA,
        while regression supervision is controlled by `reg_weights` length (typically 10 in SparseDrive).
    """

    def __init__(
        self,
        num_cls: int = 10,
        embed_dims: int = 256,
        vel_dims: int = 3,
        output_dim: int = 11,
        refine_steps: int = 6,
        with_quality_estimation: bool = False,
        refine_yaw: bool = True,
        loss_cls: Optional[dict] = None,
        loss_reg: Optional[dict] = None,
        sampler: Optional[dict] = None,
        decoder: Optional[dict] = None,
        cls_allow_reverse: Optional[List[int]] = None,
        # Decouple attention config (SparseDrive-aligned)
        decouple_attn: bool = True,
        # GNN config (SparseDrive-aligned)
        gnn_num_heads: int = 8,
        gnn_dropout: float = 0.0,
        # Temp GNN config (SparseDrive-aligned)
        temp_gnn_num_heads: int = 8,
        temp_gnn_dropout: float = 0.0,
        num_single_frame_decoder: int = 1,
        # FFN config (SparseDrive-aligned)
        feedforward_channels: Optional[int] = None,
        # Vision Fusion config (DeepStack features) - SparseDrive DeformableFeatureAggregation
        use_vision_fusion: bool = False,
        deepstack_dims: int = 2048,  # Qwen3-VL output dim per level
        num_ds_levels: int = 3,  # number of deepstack levels
        num_views: int = 6,  # number of camera views
        num_groups: int = 8,  # deformable fusion groups
        num_pts: int = 6,  # SparseDrive det default (learnable pts)
        fix_scale: Optional[List[List[float]]] = None,
        use_camera_embed: bool = True,
        vision_fusion_type: str = "add",  # "add", "concat"
        loss_reg_weights: Optional[List[float]] = None,
        cls_threshold_to_reg: float = -1.0,
        # [ALIGNMENT] Add instance_bank to support intermediate update
        instance_bank: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_cls = int(num_cls)
        self.embed_dims = int(embed_dims)
        self.vel_dims = int(vel_dims)
        self.output_dim = int(output_dim)
        self.refine_steps = int(refine_steps)
        self.with_quality_estimation = bool(with_quality_estimation)
        self.decouple_attn = bool(decouple_attn)
        self.num_single_frame_decoder = int(num_single_frame_decoder)
        self.loss_reg_weights = loss_reg_weights
        self.cls_threshold_to_reg = float(cls_threshold_to_reg)
        self.instance_bank = instance_bank

        # Vision Fusion Module (DeepStack features) - SparseDrive DeformableFeatureAggregation
        self.use_vision_fusion = bool(use_vision_fusion)
        self.num_views = num_views
        self.deepstack_dims = deepstack_dims
        if self.use_vision_fusion:
            # Reshape 1D tokens to 2D feature maps for deformable aggregation
            # Feature maps: List of [B, V, embed_dims, H, W] per level
            if fix_scale is None:
                fix_scale = [
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ]

            self.vision_fusion = DeformableFeatureAggregation(
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_ds_levels,
                num_cams=num_views,
                attn_drop=0.0,
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=num_pts,
                    fix_scale=fix_scale,
                    embed_dims=embed_dims,
                ),
                use_camera_embed=bool(use_camera_embed),
                residual_mode=vision_fusion_type,
            )
            self.ds_level_proj = nn.ModuleList(
                [nn.Conv2d(self.deepstack_dims, self.embed_dims, kernel_size=1) for _ in range(int(num_ds_levels))]
            )
        else:
            self.vision_fusion = None
            self.ds_level_proj = None

        # Decouple attention: feature-content decoupling (SparseDrive-style)
        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        # GNN: MultiheadAttention for instance-instance interaction (SparseDrive-style)
        gnn_embed_dim = self.embed_dims * 2 if self.decouple_attn else self.embed_dims
        self.gnn = MultiheadFlashAttention(
            embed_dims=gnn_embed_dim,
            num_heads=int(gnn_num_heads),
            dropout=float(gnn_dropout),
            batch_first=True,
        )

        # Temp GNN for temporal instance interaction (SparseDrive-aligned)
        self.temp_gnn = MultiheadFlashAttention(
            embed_dims=gnn_embed_dim,
            num_heads=int(temp_gnn_num_heads),
            dropout=float(temp_gnn_dropout),
            batch_first=True,
        )

        # Anchor encoder: SparseDrive style with decouple attention
        if self.decouple_attn:
            self.anchor_encoder = SparseBox3DEncoder(
                vel_dims=self.vel_dims,
                embed_dims=[128, 32, 32, 64],
                mode="cat",
                output_fc=False,
                in_loops=1,
                out_loops=4,
            )
        else:
            self.anchor_encoder = SparseBox3DEncoder(
                embed_dims=self.embed_dims,
                vel_dims=self.vel_dims,
            )

        # Norm + FFN (SparseDrive-style with AsymmetricFFN)
        self.norm = nn.LayerNorm(self.embed_dims)
        feedforward_channels = feedforward_channels or self.embed_dims * 4
        if self.decouple_attn:
            # AsymmetricFFN with decouple attention (SparseDrive-style)
            # Takes 512-dim input (feat + anchor_embed), output is 256-dim via identity_fc
            self.ffn = AsymmetricFFN(
                in_channels=self.embed_dims * 2,
                pre_norm=dict(type='LN'),
                embed_dims=self.embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                ffn_drop=0.0,
                add_identity=True,
            )
        else:
            # Standard FFN
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dims, feedforward_channels),
                nn.ReLU(inplace=True),
                nn.Linear(feedforward_channels, self.embed_dims),
            )

        # Refine layers
        self.refine_layers = nn.ModuleList(
            [
                SparseBox3DRefinementModule(
                    embed_dims=self.embed_dims,
                    output_dim=self.output_dim,
                    num_cls=self.num_cls,
                    refine_yaw=bool(refine_yaw),
                    with_quality_estimation=self.with_quality_estimation,
                )
                for _ in range(max(self.refine_steps, 1))
            ]
        )

        # Target/sampler (SparseDrive style)
        if sampler is None:
            sampler = dict(
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights=None,
                num_dn_groups=5,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
            )
        self.target = SparseBox3DTarget(**sampler)

        # Losses
        if loss_cls is None:
            loss_cls = dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            )
        self.loss_cls = build_loss(loss_cls)

        if loss_reg is None:
            loss_reg = dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=cls_allow_reverse,
            )
        self.loss_reg = build_loss(loss_reg)
        if not isinstance(self.loss_reg, SparseBox3DLoss):
            raise TypeError(f"DetVLAHead expects SparseBox3DLoss; got {type(self.loss_reg)}")

        # Decoder
        if decoder is None:
            decoder = dict(
                num_output=300,
                score_threshold=None,
                sorted=True,
            )
        self.decoder = SparseBox3DDecoder(
            num_output=int(decoder.get("num_output", 300)),
            score_threshold=decoder.get("score_threshold", None),
            sorted=bool(decoder.get("sorted", True)),
        )


    def graph_model(
        self,
        index: int,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """SparseDrive-style graph model with decouple attention.

        Args:
            index: Layer index (not used directly, for compatibility).
            query: Query tensor (B, N, C).
            key: Key tensor (B, N, C).
            value: Value tensor (B, N, C).
            query_pos: Query positional embedding.
            key_pos: Key positional embedding.
            attn_mask: Attention mask.

        Returns:
            Updated instance features.
        """
        if self.decouple_attn:
            # Concatenate content and positional embeddings
            if query_pos is not None:
                query = torch.cat([query, query_pos], dim=-1)
            if key is not None and key_pos is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None

        if value is not None:
            value = self.fc_before(value)

        # Call attention
        attn_out = self.gnn(
            query,
            key if key is not None else query,
            value if value is not None else query,
            attn_mask=attn_mask,
        )

        if self.decouple_attn:
            attn_out = self.fc_after(attn_out)

        return attn_out

    def forward(
        self,
        det_feat: Tensor,
        det_anchor: Tensor,
        # Temporal features from InstanceBank (SparseDrive-aligned)
        temp_instance_feature: Optional[Tensor] = None,
        temp_anchor: Optional[Tensor] = None,
        time_interval: float = 1.0,
        return_first_layer: bool = False,
        # DeepStack features from VLM (for deformable vision fusion)
        deepstack_features: Optional[List[Tensor]] = None,
        all_image_grids: Optional[Tensor] = None,  # [B*6, 3] grid info
        projection_mat: Optional[Tensor] = None,  # [B, num_cams, 3, 4] camera projection
        image_wh: Optional[Tensor] = None,  # [B, num_cams, 2] image width/height
    ) -> DetHeadOutputs:
        """Forward for both train/test.

        Args:
            det_feat: (B, N, C) content features after global attention.
            det_anchor: (B, N, state_dim>=output_dim) anchor states in encoded box format.
            temp_instance_feature: (B, T, C) temporal instance features.
            temp_anchor: (B, T, state_dim) temporal anchors.
            time_interval: Time interval for temporal modeling.
            return_first_layer: If True, also return first layer outputs for InstanceBank update.
            deepstack_features: List of (B*6, C_ds) features at each deepstack level.
            all_image_grids: [B*6, 3] grid info for ViT output reshape.
            projection_mat: [B, num_cams, 3, 4] camera projection matrix for deformable sampling.
            image_wh: [B, num_cams, 2] image width/height for 2D normalization.

        Returns:
            DetHeadOutputs or tuple of (DetHeadOutputs, first_layer_feat, first_layer_anchor, first_layer_cls)
        """
        B, N, C = det_feat.shape

        # Encode anchor first (needed for deformable fusion)
        anchor = det_anchor.to(device=det_feat.device, dtype=det_feat.dtype)
        if anchor.shape[0] == 1 and B != 1:
            anchor = anchor.expand(B, -1, -1)
        anchor_embed = self.anchor_encoder(anchor)

        # Deformable Vision Fusion with DeepStack features
        feature_maps = []
        metas = None
        if self.use_vision_fusion and deepstack_features is not None:
            # deepstack_features are MERGED features from ViT
            # Grid is [T, H, W] = [1, 34, 60], after merge it's [1, 17, 30]
            B = det_feat.shape[0]

            first_grid = all_image_grids[0]  # [T, H, W] for first image
            # DeepStack features are merged, so divide by 2
            merged_H = int(first_grid[1]) // 2  # 34 -> 17
            merged_W = int(first_grid[2]) // 2  # 60 -> 30

            num_tokens = merged_H * merged_W

            for lvl, ds_feat in enumerate(deepstack_features):
                # Support two deepstack formats:
                #   (1) [B*V, num_tokens, C]
                #   (2) [B*V*num_tokens, C]
                if ds_feat.dim() == 3:
                    if ds_feat.shape[0] != B * self.num_views:
                        raise RuntimeError(
                            f"det_vla_head deepstack_features[{lvl}] expected first dim B*num_views={B * self.num_views}, got {ds_feat.shape[0]}"
                        )
                    if ds_feat.shape[1] != num_tokens:
                        raise RuntimeError(
                            f"det_vla_head deepstack_features[{lvl}] token count mismatch: expected {num_tokens}, got {ds_feat.shape[1]}"
                        )
                    ds_feat = ds_feat.permute(0, 2, 1).contiguous()  # [B*V, C, T]
                    ds_feat = ds_feat.view(B, self.num_views, self.deepstack_dims, merged_H, merged_W).contiguous()
                elif ds_feat.dim() == 2:
                    if ds_feat.shape[0] != B * self.num_views * num_tokens:
                        raise RuntimeError(
                            f"det_vla_head deepstack_features[{lvl}] token count mismatch: expected first dim {B * self.num_views * num_tokens}, got {ds_feat.shape[0]}"
                        )
                    ds_feat = ds_feat.view(B, self.num_views, num_tokens, self.deepstack_dims)
                    ds_feat = ds_feat.view(B, self.num_views, merged_H, merged_W, self.deepstack_dims)
                    ds_feat = ds_feat.permute(0, 1, 4, 2, 3).contiguous()  # [B, V, C, H, W]
                else:
                    raise RuntimeError(
                        f"det_vla_head deepstack_features[{lvl}] expects 2D or 3D tensor, got dim={ds_feat.dim()} shape={tuple(ds_feat.shape)}"
                    )

                if self.ds_level_proj is None:
                    raise RuntimeError("det_vla_head ds_level_proj is None while use_vision_fusion=True")
                if lvl >= len(self.ds_level_proj):
                    raise RuntimeError(
                        f"det_vla_head got {len(deepstack_features)} deepstack levels but ds_level_proj has {len(self.ds_level_proj)}"
                    )

                ds_feat_2d = ds_feat.view(B * self.num_views, self.deepstack_dims, merged_H, merged_W)
                # Ensure projection layer matches input dtype (e.g. bfloat16)
                self.ds_level_proj[lvl].to(dtype=ds_feat_2d.dtype)
                ds_feat_2d = self.ds_level_proj[lvl](ds_feat_2d)
                ds_feat = ds_feat_2d.view(B, self.num_views, self.embed_dims, merged_H, merged_W).contiguous()

                feature_maps.append(ds_feat)

            # Build metas dict for DeformableFeatureAggregation
            # grid_sample requires input and grid to have the same dtype.
            # feature_maps are from VLM (BF16), but det_feat might be FP32.
            # We must align projection_mat to feature_maps dtype.
            target_dtype = torch.float32
            metas = {
                "projection_mat": projection_mat.to(dtype=target_dtype),
                "image_wh": image_wh.to(dtype=target_dtype),
            }

        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        cls_scores: List[Tensor] = []
        box_preds: List[Tensor] = []
        quality: List[Optional[Tensor]] = []

        feat = det_feat
        cur_anchor = anchor

        # Store first layer outputs for InstanceBank update (SparseDrive-style)
        first_layer_feat = None
        first_layer_anchor = None
        first_layer_cls = None

        for layer_idx, layer in enumerate(self.refine_layers):
            is_single_frame = layer_idx < self.num_single_frame_decoder

            # Temporal GNN: only for non-single-frame decoders
            if not is_single_frame and temp_instance_feature is not None:
                # Prepare temporal query/key/value
                if self.decouple_attn:
                    temp_query = torch.cat([feat, anchor_embed], dim=-1)
                    temp_key = torch.cat([temp_instance_feature, temp_anchor_embed], dim=-1)
                else:
                    temp_query = feat
                    temp_key = temp_instance_feature

                temp_out = self.temp_gnn(
                    temp_query,
                    temp_key,
                    temp_key,
                )
                if self.decouple_attn:
                    temp_out = self.fc_after(temp_out)
                feat = feat + temp_out

            # GNN: instance-instance interaction (SparseDrive-style)
            if self.decouple_attn:
                gnn_query = torch.cat([feat, anchor_embed], dim=-1)
            else:
                gnn_query = feat

            gnn_out = self.gnn(
                gnn_query,
                gnn_query,
                self.fc_before(feat),
            )
            if self.decouple_attn:
                gnn_out = self.fc_after(gnn_out)
            feat = feat + gnn_out

            # Norm
            feat = self.norm(feat)

            # Vision Fusion (SparseDrive-style: inside loop)
            if self.use_vision_fusion and feature_maps:
                # Cast feature_maps to float32 for grid_sample stability
                feature_maps_f32 = [f.float() for f in feature_maps]
                feat_f32 = feat.float()
                anchor_f32 = cur_anchor.float()
                anchor_embed_f32 = anchor_embed.float()

                fused_feat = self.vision_fusion(
                    instance_feature=feat_f32,
                    anchor=anchor_f32,
                    anchor_embed=anchor_embed_f32,
                    feature_maps=feature_maps_f32,
                    metas=metas,
                )
                # Cast back to original dtype
                feat = fused_feat.to(dtype=feat.dtype)

                # [ALIGNMENT] SparseDrive does NOT have a norm here.
                # Config: "deformable", "ffn", "norm".
                # The norm happens AFTER ffn (or inside ffn as pre-norm).
                # feat = self.norm(feat)

            # FFN with decouple attention: AsymmetricFFN takes 512-dim (feat + anchor_embed)
            if self.decouple_attn:
                # ffn_input is 512-dim: feat (256) + anchor_embed (256)
                ffn_input = torch.cat([feat, anchor_embed], dim=-1)
                # AsymmetricFFN: out (256) = identity_fc(ffn_input, 512->256) + ffn_out(512->256)
                # AsymmetricFFN already includes residual connection via identity_fc
                feat = self.ffn(ffn_input)
            else:
                # Standard FFN (nn.Sequential) does not have residual connection
                ffn_out = self.ffn(feat)
                feat = feat + ffn_out

            # Norm after FFN
            feat = self.norm(feat)

            # Refine
            cur_anchor, cls, qua = layer(
                instance_feature=feat,
                anchor=cur_anchor,
                anchor_embed=anchor_embed,
                time_interval=time_interval,
                return_cls=True,
            )
            cls_scores.append(cls)
            box_preds.append(cur_anchor)
            quality.append(qua)

            # [ALIGNMENT] SparseDrive-style intermediate update
            # Inject propagated queries after the first decoder layer (usually layer 0)
            if len(box_preds) == self.num_single_frame_decoder:
                if self.instance_bank is not None:
                    feat, cur_anchor = self.instance_bank.update(
                        feat, cur_anchor, cls
                    )
                    # Note: SparseDrive also updates DN here, but we skip DN for now as per instruction.

            # Store first layer outputs for InstanceBank update (SparseDrive-style)
            # If num_single_frame_decoder=0 (all layers use temp GNN), we take the first layer (idx 0).
            # If num_single_frame_decoder=1 (first layer is single-frame), we take the first layer (idx 0).
            # Basically, we always want to update InstanceBank with the output of the first decoder layer,
            # regardless of whether it used Temp GNN or not.
            # SparseDrive typically uses the output of the first stage (which might include Temp GNN) for propagation.
            target_layer_idx = max(self.num_single_frame_decoder - 1, 0)
            if layer_idx == target_layer_idx:
                first_layer_feat = feat.clone()
                first_layer_anchor = cur_anchor.clone()
                first_layer_cls = cls.clone()

            # Update anchor_embed after refine
            anchor_embed = self.anchor_encoder(cur_anchor)

            # Update temporal anchor_embed for next decoder
            if temp_anchor_embed is not None:
                # [ALIGNMENT] SparseDrive logic: if we injected propagated queries,
                # temp_anchor_embed should now correspond to the propagated part of the new anchor set.
                # In SparseDrive, num_temp_instances are at the BEGINNING of the anchor set after update.
                if self.instance_bank is not None and self.instance_bank.num_temp_instances > 0:
                     temp_anchor_embed = anchor_embed[:, :self.instance_bank.num_temp_instances]
                else:
                    # Fallback for legacy behavior
                    num_temp = temp_anchor.shape[1] if temp_anchor is not None else 0
                    if num_temp > 0:
                        temp_anchor_embed = anchor_embed[:, :num_temp]

        # Return outputs and optionally first layer data for InstanceBank update
        outputs = DetHeadOutputs(
            cls_scores=cls_scores,
            box_preds=box_preds,
            quality=quality,
            last_feat=feat,
        )
        if return_first_layer:
            return outputs, first_layer_feat, first_layer_anchor, first_layer_cls
        return outputs

    def loss(
        self,
        outs: DetHeadOutputs,
        gt_labels_3d: List[Tensor],
        gt_bboxes_3d: List[Tensor],
        prefix: str = "det_",
    ) -> Dict[str, Tensor]:
        """Compute SparseDrive-aligned losses with per-refine-step suffixes.

        SparseDrive logs per decoder layer, e.g. det_loss_cls_0..5, det_loss_box_0..5,
        det_loss_cns_0..5, det_loss_yns_0..5. This head mirrors that behavior.
        """
        if len(outs.cls_scores) != len(outs.box_preds):
            raise ValueError(
                f"DetHeadOutputs length mismatch: cls={len(outs.cls_scores)} box={len(outs.box_preds)}"
            )

        # Target expects lists per batch element
        device = outs.cls_scores[0].device
        cls_targets = [x.to(device) for x in gt_labels_3d]
        box_targets = [x.to(device) for x in gt_bboxes_3d]

        # Align with SparseDrive: supervise only first len(reg_weights) dims.
        reg_dim = len(getattr(self.target, "reg_weights", []) or [])

        out: Dict[str, Tensor] = {}
        # Use a stable zero tensor on the same device/dtype as predictions.
        zero = outs.box_preds[0].sum() * 0.0

        for decoder_idx, (cls_pred, box_pred, qua) in enumerate(
            zip(outs.cls_scores, outs.box_preds, outs.quality)
        ):
            if reg_dim > 0:
                box_pred_for_loss = box_pred[..., :reg_dim]
            else:
                box_pred_for_loss = box_pred

            cls_tgt, box_tgt, reg_w, _indices = self.target.sample(
                cls_pred,
                box_pred_for_loss,
                cls_targets,
                box_targets,
            )

            if self.loss_reg_weights is not None:
                weight_tensor = box_pred.new_tensor(self.loss_reg_weights)
                if reg_w.shape[-1] <= len(self.loss_reg_weights):
                     weight_tensor = weight_tensor[:reg_w.shape[-1]]
                reg_w = reg_w * weight_tensor

            valid = (cls_tgt >= 0) & (cls_tgt < self.num_cls)

            # [ALIGNMENT] Use reduce_mean for multi-gpu sync of avg_factor
            # Calculate avg_factor BEFORE filtering (SparseDrive behavior)
            num_pos = reduce_mean(valid.sum().float())
            avg_factor = max(num_pos.item(), 1.0)

            # Filter out low confidence boxes for regression loss (SparseDrive-aligned)
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                # cls_pred: (B, N, num_cls)
                scores = cls_pred.max(dim=-1).values.sigmoid()
                mask = scores > threshold
                # Apply mask to valid (which is used to select regression targets)
                # Note: valid is (B, N) boolean mask
                valid = valid & mask

            # mmcv sigmoid_focal_loss CUDA kernel in this repo doesn't support bf16.
            cls_pred_flat = cls_pred.reshape(-1, self.num_cls).float()
            cls_tgt_flat = cls_tgt.reshape(-1).to(dtype=torch.long)
            loss_cls = self.loss_cls(
                cls_pred_flat,
                cls_tgt_flat,
                avg_factor=avg_factor,
            )
            out[f"{prefix}loss_cls_{decoder_idx}"] = loss_cls

            # Always populate reg keys for distributed log_vars stability.
            out.setdefault(f"{prefix}loss_box_{decoder_idx}", zero)
            if self.with_quality_estimation:
                out.setdefault(f"{prefix}loss_cns_{decoder_idx}", zero)
                out.setdefault(f"{prefix}loss_yns_{decoder_idx}", zero)

            if not valid.any():
                continue

            if reg_dim > 0:
                box_tgt = box_tgt[..., :reg_dim]
                reg_w = reg_w[..., :reg_dim]

            reg_out = self.loss_reg(
                box_pred_for_loss[valid],
                box_tgt[valid],
                weight=reg_w[valid],
                avg_factor=avg_factor,
                prefix=prefix,
                quality=None if qua is None else qua[valid],
                cls_target=cls_tgt[valid],
            )

            # SparseBox3DLoss returns keys like f"{prefix}loss_box" (+ cns/yns when quality is not None).
            # Remap them to SparseDrive-style per-layer keys with suffix "_{decoder_idx}".
            key_map = {
                f"{prefix}loss_box": f"{prefix}loss_box_{decoder_idx}",
                f"{prefix}loss_cns": f"{prefix}loss_cns_{decoder_idx}",
                f"{prefix}loss_yns": f"{prefix}loss_yns_{decoder_idx}",
            }
            for k, v in reg_out.items():
                out[key_map.get(k, f"{k}_{decoder_idx}")] = v

        return out

    @torch.no_grad()
    def decode(self, outs: DetHeadOutputs, instance_id: Optional[Tensor] = None) -> List[Dict]:
        cls_scores = [x.float() for x in outs.cls_scores]
        box_preds = [x.float() for x in outs.box_preds]
        quality = outs.quality
        return self.decoder.decode(
            cls_scores=cls_scores,
            box_preds=box_preds,
            quality=quality,
            instance_id=instance_id,
            output_idx=-1,
        )
