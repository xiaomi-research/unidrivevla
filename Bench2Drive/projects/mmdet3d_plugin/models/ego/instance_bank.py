import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from mmcv.cnn import Linear
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.models.blocks import linear_relu_ln


__all__ = ["EgoInstanceBank"]

def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs

@PLUGIN_LAYERS.register_module()
class EgoInstanceBank(nn.Module):
    def __init__(
            self,
            embed_dims,
            anchor_type='nus',
            anchor_handler=None,
            feature_map_scale=None,
            num_temp_instances=0,
            anchor_grad=True,
            max_time_interval=2,
            num_anchor=None,
            with_instance_feat=False,
            plan_anchor=None,
            feat_grad=True
    ):
        super(EgoInstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.max_time_interval = max_time_interval
        self.num_temp_instances = num_temp_instances
        self.with_instance_feat = with_instance_feat
        kernel_size = tuple([int(x / 2) for x in feature_map_scale])

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler

        nus_ego_anchor = [[0, 0.5, -1.84 + 1.56/2, np.log(4.08), np.log(1.73), np.log(1.56), 1, 0, 0, 0, 0]]
        b2d_ego_anchor = [[0, 0.5, -1.84 + 1.49/2, np.log(4.89), np.log(1.84), np.log(1.49), 1, 0, 0, 0, 0]]

        if anchor_type == 'nus':
            self.anchor = nn.Parameter(torch.tensor(nus_ego_anchor, dtype=torch.float32), requires_grad=False)
        elif anchor_type == 'b2d':
            self.anchor = nn.Parameter(torch.tensor(b2d_ego_anchor, dtype=torch.float32), requires_grad=False)

        self.num_anchor = len(self.anchor)

        if self.with_instance_feat:
            self.instance_feature = nn.Parameter(
                torch.zeros([self.anchor.shape[0], self.embed_dims]),
                requires_grad=feat_grad,
            )
        else:
            self.ego_feature_encoder = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.embed_dims),
                nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.embed_dims),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size),
            )

        if plan_anchor is not None:
            self.plan_anchor = nn.Parameter(
                    torch.tensor(np.load(plan_anchor), dtype=torch.float32), requires_grad=False)
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(self.embed_dims, 1, 1), Linear(self.embed_dims, self.embed_dims))

        self.reset()

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None

    def get(self, batch_size, metas, feature_maps, dn_metas=None):
        instance_feature, anchor = self.prepare_ego(batch_size, feature_maps)

        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            if not isinstance(time_interval, torch.Tensor):
                time_interval = anchor.new_tensor(time_interval)
            time_interval = time_interval.to(dtype=anchor.dtype)
            if time_interval.dim() == 0:
                time_interval = time_interval.expand(batch_size)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor, [T_temp2cur], time_intervals=[-time_interval])[0]

            if self.anchor_handler is not None and dn_metas is not None and batch_size == dn_metas["dn_anchor"].shape[0]:
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2), [T_temp2cur], time_intervals=[-time_interval])[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(batch_size, num_dn_group, num_dn, -1)

            temp_ego_feature = self.cached_feature
            temp_ego_anchor = self.cached_anchor
        else:
            temp_ego_feature, temp_ego_anchor = None, None

        return instance_feature, anchor, temp_ego_feature, temp_ego_anchor

    def prepare_ego(self, batch_size, feature_maps,):
        if self.with_instance_feat:
            instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        else:
            feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
            feature_map = feature_maps_inv[0][-1][:, 0]  # 'front-view feature'

            instance_feature = self.ego_feature_encoder(feature_map.to(next(self.ego_feature_encoder.parameters()).dtype))
            instance_feature = instance_feature.unsqueeze(1).squeeze(-1).squeeze(-1)

        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        return instance_feature, anchor


    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)
        instance_feature = torch.where(self.mask[:, None, None], selected_feature, instance_feature)
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
            self,
            instance_feature,
            anchor,
            metas=None,
            feature_maps=None
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()

        self.metas = metas
        self.cached_feature = instance_feature.detach()
        self.cached_anchor = anchor.detach()