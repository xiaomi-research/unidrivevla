from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from projects.mmdet3d_plugin.models.blocks import AsymmetricFFN
from projects.mmdet3d_plugin.models.map.map_blocks import (
    SparsePoint3DEncoder,
    SparsePoint3DRefinementModule,
)
from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget
from projects.mmdet3d_plugin.models.map.loss import SparseLineLoss
from projects.mmdet3d_plugin.models.map.decoder import SparsePoint3DDecoder
from mmdet.core import reduce_mean
from mmdet.models.builder import build_loss
from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation
from projects.mmdet3d_plugin.models.attention import MultiheadFlashAttention


@dataclass
class MapHeadOutputs:
    cls_scores: List[Tensor]
    pts_preds: List[Tensor]


class MapVLAHead(nn.Module):
    """SparseDrive-aligned map head with GNN + decouple attention.

    Design (aligned with SparseDrive SparsePoint3DHead):
      - Anchor init (kmeans_map_100.npy) + SparsePoint3DEncoder => anchor_embed (query_pos)
      - One global-attn with prefix is assumed to have produced `map_feat` (content)
      - Head runs K-step refinement on map queries with GNN + FFN + refine.
      - Uses decouple attention (fc_before/fc_after) for feature-content decoupling.
      - Uses AsymmetricFFN for FFN layers (SparseDrive-style).
      - Training uses SparsePoint3DTarget + SparseLineLoss (SparseDrive style).
      - Inference uses SparsePoint3DDecoder.decode => vectors/scores/labels.
    """

    def __init__(
        self,
        anchor_path: str,
        embed_dims: int = 256,
        num_map_queries: int = 100,
        num_cls: int = 3,
        num_sample: int = 20,
        roi_size: Tuple[int, int] = (30, 60),
        refine_steps: int = 6,
        score_threshold: Optional[float] = None,
        # Decouple attention config (SparseDrive-aligned)
        decouple_attn: bool = True,
        # GNN config (SparseDrive-aligned)
        gnn_num_heads: int = 8,
        gnn_dropout: float = 0.0,
        # FFN config (SparseDrive-aligned)
        feedforward_channels: Optional[int] = None,
        loss_cls: Optional[dict] = None,
        loss_line: Optional[dict] = None,
        assigner: Optional[dict] = None,
        # Vision Fusion config (DeepStack features) - SparseDrive DeformableFeatureAggregation
        use_vision_fusion: bool = False,
        deepstack_dims: int = 2048,
        num_ds_levels: int = 3,
        num_views: int = 6,
        num_groups: int = 8,
        num_pts: int = 3,
        fix_height: Tuple[float, ...] = (0, 0.5, -0.5, 1, -1),
        ground_height: float = -1.84023,
        use_camera_embed: bool = True,
        vision_fusion_type: str = "add",
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_map_queries = num_map_queries
        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size
        self.refine_steps = int(refine_steps)
        self.decouple_attn = bool(decouple_attn)
        self.num_views = num_views
        self.deepstack_dims = deepstack_dims

        # Vision Fusion Module (DeepStack features) - SparseDrive DeformableFeatureAggregation
        self.use_vision_fusion = bool(use_vision_fusion)
        if self.use_vision_fusion:
            self.vision_fusion = DeformableFeatureAggregation(
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_ds_levels,
                num_cams=num_views,
                attn_drop=0.0,
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=num_pts,
                    fix_height=fix_height,
                    ground_height=ground_height,
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

        anchors = np.load(anchor_path)
        # Common SparseDrive format: (N, num_sample, 2)
        if anchors.ndim == 3 and anchors.shape[0] == num_map_queries:
            anchors = anchors[None, ...]  # (1, N, num_sample, 2)
        if anchors.ndim == 4 and anchors.shape[1] != num_map_queries:
            raise ValueError(
                f"kmeans map anchors count mismatch: got {anchors.shape[1]} vs num_map_queries={num_map_queries}"
            )
        if anchors.ndim != 4:
            raise ValueError(f"unexpected kmeans map anchor shape: {anchors.shape}")

        # Flatten polyline anchors to match SparsePoint3DTarget/Loss convention: (B, N, num_sample*2)
        anchors = anchors.reshape(anchors.shape[0], anchors.shape[1], -1)

        # [ALIGNMENT] Remove buffer, use input anchors from InstanceBank
        # self.register_buffer(
        #     "map_anchors", torch.from_numpy(anchors).float(), persistent=False
        # )

        self.anchor_encoder = SparsePoint3DEncoder(
            embed_dims=embed_dims,
            num_sample=num_sample,
            coords_dim=2,
        )

        # Decouple attention (SparseDrive-style)
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

        # Refine layer (shared across all steps, SparseDrive-style)
        self.refine_layer = SparsePoint3DRefinementModule(
            embed_dims=embed_dims,
            num_sample=num_sample,
            coords_dim=2,
            num_cls=num_cls,
            with_cls_branch=True,
        )

        if assigner is None:
            assigner = dict(
                type="HungarianLinesAssigner",
                cost=dict(
                    type="MapQueriesCost",
                    cls_cost=dict(type="FocalLossCost", weight=1.0),
                    reg_cost=dict(
                        type="LinesL1Cost", weight=10.0, beta=0.01, permute=True
                    ),
                ),
            )

        self.target = SparsePoint3DTarget(
            assigner=assigner,
            num_cls=num_cls,
            num_sample=num_sample,
            roi_size=roi_size,
        )

        if loss_cls is None:
            loss_cls = dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            )
        self.loss_cls = build_loss(loss_cls)

        if loss_line is None:
            loss_line = dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01)
        self.loss_reg = SparseLineLoss(
            loss_line=loss_line,
            num_sample=num_sample,
            roi_size=roi_size,
        )

        self.decoder = SparsePoint3DDecoder(coords_dim=2, score_threshold=score_threshold)

    def graph_model(
        self,
        query: Tensor,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """SparseDrive-style graph model with decouple attention.

        Args:
            query: Query tensor (B, N, C).
            value: Value tensor (B, N, C).
            query_pos: Query positional embedding.
            attn_mask: Attention mask.

        Returns:
            Updated instance features.
        """
        if self.decouple_attn:
            # Concatenate content and positional embeddings
            if query_pos is not None:
                query = torch.cat([query, query_pos], dim=-1)
            query_pos = None

        if value is not None:
            value = self.fc_before(value)

        # Call attention
        attn_out = self.gnn(
            query,
            query if value is None else query,
            value if value is not None else query,
            attn_mask=attn_mask,
        )

        if self.decouple_attn:
            attn_out = self.fc_after(attn_out)

        return attn_out

    def forward(
        self,
        map_feat: Tensor,
        map_anchor: Tensor, # [ALIGNMENT] Require map_anchor input
        deepstack_features: Optional[List[Tensor]] = None,
        all_grids: Optional[Tensor] = None,
        projection_mat: Optional[Tensor] = None,
        image_wh: Optional[Tensor] = None,
    ) -> MapHeadOutputs:
        """Forward for both train/test.

        Args:
          map_feat: (B, N, C) content features after global attention.
          map_anchor: (B, N, num_sample*2) map anchors for sampling positions.
          deepstack_features: List of (B*6, C_ds) features at each deepstack level.
          all_grids: [B*6, 3] grid info for each image.
          projection_mat: [B, num_cams, 3, 4] camera projection matrix.
          image_wh: [B, num_cams, 2] image width/height.
        Returns:
          MapHeadOutputs with per-step cls_scores/pts_preds as lists.
        """
        B, N, C = map_feat.shape
        assert N == self.num_map_queries, (N, self.num_map_queries)

        # Prepare anchor for deformable fusion
        # [ALIGNMENT] Use input map_anchor (from InstanceBank)
        anchor = map_anchor.to(device=map_feat.device, dtype=map_feat.dtype)
        if anchor.shape[0] == 1 and B != 1:
            anchor = anchor.expand(B, -1, -1)

        # Encode anchor first (needed for deformable fusion)
        anchor_embed = self.anchor_encoder(anchor)

        # Deformable Vision Fusion with DeepStack features
        feature_maps = []
        metas = None
        if self.use_vision_fusion and deepstack_features is not None:
            # deepstack_features from embed_image_tensor are 2D: [B*6*num_tokens, C]
            # Grid is [T, H, W] = [1, 34, 60], after merge it's [1, 17, 30]
            feature_maps = []

            first_grid = all_grids[0]  # [T, H, W] for first image
            # DeepStack features are merged, so divide by 2
            merged_H = int(first_grid[1]) // 2  # 34 -> 17
            merged_W = int(first_grid[2]) // 2  # 60 -> 30

            num_tokens = merged_H * merged_W

            for lvl, ds_feat in enumerate(deepstack_features):
                # Support two deepstack formats:
                #   (1) [B*V, num_tokens, C]
                #   (2) [B*V*num_tokens, C]
                if ds_feat.dim() == 3:
                    # [B*V, T, C] -> [B, V, C, H, W]
                    if ds_feat.shape[0] != B * self.num_views:
                        raise RuntimeError(
                            f"map_vla_head deepstack_features[{lvl}] expected first dim B*num_views={B * self.num_views}, got {ds_feat.shape[0]}"
                        )
                    if ds_feat.shape[1] != num_tokens:
                        raise RuntimeError(
                            f"map_vla_head deepstack_features[{lvl}] token count mismatch: expected {num_tokens}, got {ds_feat.shape[1]}"
                        )
                    ds_feat = ds_feat.permute(0, 2, 1).contiguous()  # [B*V, C, T]
                    ds_feat = ds_feat.view(B, self.num_views, self.deepstack_dims, merged_H, merged_W).contiguous()
                elif ds_feat.dim() == 2:
                    # [B*V*T, C] -> [B, V, C, H, W]
                    if ds_feat.shape[0] != B * self.num_views * num_tokens:
                        raise RuntimeError(
                            f"map_vla_head deepstack_features[{lvl}] token count mismatch: expected first dim {B * self.num_views * num_tokens}, got {ds_feat.shape[0]}"
                        )
                    ds_feat = ds_feat.view(B, self.num_views, num_tokens, self.deepstack_dims)
                    ds_feat = ds_feat.view(B, self.num_views, merged_H, merged_W, self.deepstack_dims)
                    ds_feat = ds_feat.permute(0, 1, 4, 2, 3).contiguous()  # [B, V, C, H, W]
                else:
                    raise RuntimeError(
                        f"map_vla_head deepstack_features[{lvl}] expects 2D or 3D tensor, got dim={ds_feat.dim()} shape={tuple(ds_feat.shape)}"
                    )

                if self.ds_level_proj is None:
                    raise RuntimeError("map_vla_head ds_level_proj is None while use_vision_fusion=True")
                if lvl >= len(self.ds_level_proj):
                    raise RuntimeError(
                        f"map_vla_head got {len(deepstack_features)} deepstack levels but ds_level_proj has {len(self.ds_level_proj)}"
                    )

                # Project channel dim to embed_dims expected by DeformableFeatureAggregation.
                ds_feat_2d = ds_feat.view(B * self.num_views, self.deepstack_dims, merged_H, merged_W)
                # Ensure projection layer matches input dtype (e.g. bfloat16)
                self.ds_level_proj[lvl].to(dtype=ds_feat_2d.dtype)
                ds_feat_2d = self.ds_level_proj[lvl](ds_feat_2d)
                ds_feat = ds_feat_2d.view(B, self.num_views, self.embed_dims, merged_H, merged_W).contiguous()

                feature_maps.append(ds_feat)

            # Build metas dict for DeformableFeatureAggregation
            target_dtype = torch.float32
            metas = {
                "projection_mat": projection_mat.to(dtype=target_dtype),
                "image_wh": image_wh.to(dtype=target_dtype),
            }

        cls_scores: List[Tensor] = []
        pts_preds: List[Tensor] = []

        feat = map_feat
        cur_anchor = anchor

        for layer_idx in range(max(self.refine_steps, 1)):
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
            cur_anchor, cls, _ = self.refine_layer(
                instance_feature=feat,
                anchor=cur_anchor,
                anchor_embed=anchor_embed,
                time_interval=1.0,
                return_cls=True,
            )

            # [FIX] Numerical Stability: Clamp refined anchors
            if torch.isnan(cur_anchor).any() or torch.isinf(cur_anchor).any():
                 cur_anchor = torch.where(torch.isnan(cur_anchor) | torch.isinf(cur_anchor),
                                          torch.zeros_like(cur_anchor), cur_anchor)
            cur_anchor = torch.clamp(cur_anchor, min=-200.0, max=200.0)

            cls_scores.append(cls)
            pts_preds.append(cur_anchor)

            # Update anchor_embed after refine
            anchor_embed = self.anchor_encoder(cur_anchor)

        return MapHeadOutputs(cls_scores=cls_scores, pts_preds=pts_preds)

    def loss(
        self,
        outs: MapHeadOutputs,
        gt_map_labels: List[Tensor],
        gt_map_pts: List[Tensor],
    ) -> Dict[str, Tensor]:
        """Compute SparseDrive-aligned losses with per-refine-step suffixes.

        SparseDrive logs per decoder layer: map_loss_cls_0..5 and map_loss_line_0..5.
        """
        if len(outs.cls_scores) != len(outs.pts_preds):
            raise ValueError(
                f"MapHeadOutputs length mismatch: cls={len(outs.cls_scores)} pts={len(outs.pts_preds)}"
            )

        device = outs.cls_scores[0].device
        cls_targets = [x.to(device) for x in gt_map_labels]
        pts_targets = [x.to(device) for x in gt_map_pts]

        out: Dict[str, Tensor] = {}
        zero = outs.pts_preds[0].sum() * 0.0

        for decoder_idx, (cls_pred, pts_pred) in enumerate(
            zip(outs.cls_scores, outs.pts_preds)
        ):
            # Convert to float32 for stable matching
            cls_pred_f32 = cls_pred.float()
            pts_pred_f32 = pts_pred.float()

            cls_tgt, pts_tgt, reg_w = self.target.sample(
                cls_pred_f32,
                pts_pred_f32,
                cls_targets,
                pts_targets,
            )

            # SparseDrive keeps cls loss even if no positives; avg_factor uses numel.
            loss_cls = self.loss_cls(
                cls_pred.reshape(-1, self.num_cls).float(),
                cls_tgt.reshape(-1),
                avg_factor=float(cls_tgt.numel()),
            )
            out[f"map_loss_cls_{decoder_idx}"] = loss_cls

            # Always populate map_loss_line_{idx} for distributed stability.
            out.setdefault(f"map_loss_line_{decoder_idx}", zero)

            # [ALIGNMENT] Use reduce_mean for multi-gpu sync of avg_factor
            num_pos = reduce_mean(reg_w.sum().float())
            num_pos = max(num_pos.item(), 1.0)

            loss_reg_dict = self.loss_reg(
                pts_pred,
                pts_tgt,
                weight=reg_w,
                avg_factor=float(num_pos),
                prefix="map_",
            )

            # SparseLineLoss returns {"map_loss_line": ...}. Remap to per-layer key.
            for k, v in loss_reg_dict.items():
                if k == "map_loss_line":
                    out[f"map_loss_line_{decoder_idx}"] = v
                else:
                    out[f"{k}_{decoder_idx}"] = v

        return out

    @torch.no_grad()
    def decode(self, outs: MapHeadOutputs) -> List[Dict]:
        # decoder expects list for cls_scores/pts_preds
        cls_scores = outs.cls_scores
        pts_preds = outs.pts_preds
        # Ensure fp32 for numpy conversion.
        cls_scores = [x.float() for x in cls_scores]
        pts_preds = [x.float() for x in pts_preds]
        return self.decoder.decode(cls_scores=cls_scores, pts_preds=pts_preds)
