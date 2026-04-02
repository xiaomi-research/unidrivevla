import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.datasets.pipelines.vectorize_numpy import VectorizeMapNumpy
from mmcv.cnn import Linear
from projects.mmdet3d_plugin.models.blocks import linear_relu_ln

__all__ = ["PlanningInstanceBank"]

def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs

@PLUGIN_LAYERS.register_module()
class PlanningInstanceBank(nn.Module):
    def __init__(
            self,
            embed_dims,
            anchor_paths,
            anchor_types=None,
            anchor_scales=None,
            num_temp_mode=0,
            num_temp_instances=0,
            confidence_decay=0.6,
            feature_map_scale=None,
            max_time_interval=2,
            feat_grad=True,
            anchor_grad=True,
            ego_fut_ts=6,
            ego_fut_cmd=3,
            ego_fut_mode=6,
            with_instance_feat=False,
            with_all_front_views=False,
            with_custom_status_embed=False,
    ):
        super(PlanningInstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode
        self.num_temp_mode = num_temp_mode
        self.num_temp_instances = num_temp_instances

        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.with_instance_feat = with_instance_feat
        self.with_all_front_views = with_all_front_views
        self.with_custom_status_embed = with_custom_status_embed

        self.anchor_paths = anchor_paths
        self.anchor_types = anchor_types
        self.anchor_group = len(anchor_types)
        self.ego_mode_group = len(anchor_types)

        kernel_size = tuple([int(x / 2) for x in feature_map_scale])

        if anchor_scales is None:
            anchor_scales = [1.0 for _ in range(len(anchor_types))]

        assert anchor_paths is not None
        assert len(anchor_types) == len(anchor_scales)

        anchor_dict = dict()
        if isinstance(anchor_paths, str):
            for anchor_type in anchor_types:
                anchor_dict[anchor_type] = anchor_paths
        elif isinstance(anchor_paths, list):
            for anchor_type, anchor_path in zip(anchor_types, anchor_paths):
                anchor_dict[anchor_type] = anchor_path
        elif isinstance(anchor_paths, dict):
            anchor_dict = anchor_paths
        else:
            raise NotImplementedError

        anchors = []
        for anchor_type, anchor_scale in zip(anchor_types, anchor_scales):
            anchor = np.load(anchor_dict[anchor_type])

            if len(anchor.shape) == 3:  # for planning
                anchor = anchor.reshape(anchor.shape[0], -1)
            elif len(anchor.shape) == 4:
                anchor = anchor.reshape(anchor.shape[0] * anchor.shape[1], -1)

            anchor = anchor * anchor_scale
            anchors.append(anchor)

        anchor = np.concatenate(anchors, axis=0)

        self.anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float32), requires_grad=anchor_grad)  # [N, 12]
        self.num_anchor = len(self.anchor)

        if self.with_instance_feat:
            self.instance_feature = nn.Parameter(
                torch.zeros([self.num_anchor, self.embed_dims]), requires_grad=feat_grad)
        else:
            self.plan_feature_encoder = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size),
            )

        if self.with_custom_status_embed:
            self.custom_status_encoder = nn.Sequential(
                *linear_relu_ln(self.embed_dims, 2, 1, input_dims=6),
                Linear(self.embed_dims, self.embed_dims)
            )

        self.reset()

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.confidence = None
        self.metas = None
        self.mask = None

    def get(self, batch_size, metas, feature_maps, dn_metas=None):
        instance_feature, anchor = self.prepare_planning(batch_size, feature_maps, metas)

        if self.cached_anchor is not None:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            if time_interval.dim() == 0:
                time_interval = time_interval.expand(batch_size)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            bs, _, _ = anchor.shape
            temp_ego_feature = self.cached_feature.reshape(bs, -1, self.embed_dims)
            temp_ego_anchor = self.cached_anchor.reshape(bs, -1, self.ego_fut_ts * 2)
        else:
            temp_ego_feature, temp_ego_anchor = None, None

        return instance_feature, anchor, temp_ego_feature, temp_ego_anchor

    def prepare_planning(self, batch_size, feature_maps, metas=None):
        if self.with_instance_feat:
            instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        else:
            feature_maps_inv = feature_maps_format(feature_maps, inverse=True)

            if self.with_all_front_views:
                feature_map = feature_maps_inv[0][-1][:, :3]
                bs, nc, dim, h, w = feature_map.shape
                feature_map = feature_map.reshape(-1, dim, h, w)
                instance_feature = self.plan_feature_encoder(feature_map)
                instance_feature = instance_feature.reshape(bs, nc, dim, 1, 1)
                instance_feature = instance_feature.sum(1)
            else:
                # only center front-view feature
                feature_map = feature_maps_inv[0][-1][:, 0]
                instance_feature = self.plan_feature_encoder(feature_map)

            if self.with_custom_status_embed:
                custom_status = metas['custom_status'].unsqueeze(1).unsqueeze(1)
                custom_status_embed = self.custom_status_encoder(custom_status)
                instance_feature += custom_status_embed.permute(0, 3, 1, 2)

            instance_feature = instance_feature.unsqueeze(1).squeeze(-1).squeeze(-1)
            instance_feature = torch.tile(instance_feature, (1, self.num_anchor, 1))

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

        bs, nj, _ = anchor.shape
        num_cmd = self.ego_fut_cmd * self.anchor_group
        num_mode = self.ego_fut_mode
        num_temp_mode = self.num_temp_mode

        # [bs * num_cmd, num_mode]
        _instance_feature = instance_feature.reshape(bs * num_cmd, num_mode, self.embed_dims)
        _anchor = anchor.reshape(bs * num_cmd, num_mode, self.ego_fut_ts * 2)
        _confidence = confidence.reshape(bs * num_cmd, num_mode, 1)

        N = num_mode - num_temp_mode
        _confidence = _confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(_confidence, N, _instance_feature, _anchor)

        selected_feature = selected_feature.reshape(bs, num_cmd, N, self.embed_dims)
        selected_anchor = selected_anchor.reshape(bs, num_cmd, N, self.ego_fut_ts * 2)

        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=2).reshape(bs, -1, self.embed_dims)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=2).reshape(bs, -1, self.ego_fut_ts * 2)

        # [bs, num_cmd, num_mode]
        selected_feature = selected_feature.reshape(bs, -1, self.embed_dims)
        selected_anchor = selected_anchor.reshape(bs, -1, self.ego_fut_ts * 2)

        instance_feature = torch.where(self.mask[:, None, None], selected_feature, instance_feature)
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)

        self.confidence = torch.where(
            self.mask[:, None, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)

        return instance_feature, anchor

    def cache(
            self,
            instance_feature,
            anchor,
            confidence,
            metas=None,
            feature_maps=None,
    ):
        if self.num_temp_mode <= 0:
            return

        bs, nj, _ = anchor.shape
        num_cmd = self.ego_fut_cmd * self.anchor_group
        num_mode = self.ego_fut_mode
        num_temp_mode = self.num_temp_mode

        # [bs * num_cmd, num_mode]
        _instance_feature = instance_feature.detach().reshape(bs * num_cmd, -1, self.embed_dims)
        _anchor = anchor.detach().reshape(bs * num_cmd, -1, self.ego_fut_ts * 2)
        _confidence = confidence.detach().reshape(bs * num_cmd, -1, 1)

        self.metas = metas
        _confidence = _confidence.squeeze(-1).sigmoid()
        if self.confidence is not None:
            _confidence[:, : num_temp_mode] = torch.maximum(
                self.confidence.reshape(bs * num_cmd, -1) * self.confidence_decay, _confidence[:, : num_temp_mode])

        (confidence, (cached_feature, cached_anchor)) = topk(
            _confidence, num_temp_mode, _instance_feature, _anchor)

        # [bs, num_cmd, num_mode]
        self.confidence = confidence.view(bs, num_cmd, num_temp_mode)
        self.cached_feature = cached_feature.view(bs, num_cmd, num_temp_mode, self.embed_dims)
        self.cached_anchor = cached_anchor.view(bs, num_cmd, num_temp_mode, self.ego_fut_ts * 2)