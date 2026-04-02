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
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.utils import build_from_cfg
from mmcv.runner import BaseModule
from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import HEADS, LOSSES

from projects.mmdet3d_plugin.models.blocks import linear_relu_ln
from projects.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from projects.mmdet3d_plugin.core.box3d import *

__all__ = ["UnifiedPerceptionDecoder"]


@HEADS.register_module()
class UnifiedPerceptionDecoder(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        task_select=None,
        query_select=None,
        num_stage1_layers=3,
        num_stage2_layers=3,
        num_single_frame_decoder=1,
        cls_threshold_to_reg=0.05,
        decouple_attn=True,
        operation_order=None,
        det_instance_bank=None,
        map_instance_bank=None,
        ego_instance_bank=None,
        motion_instance_bank=None,
        det_anchor_encoder=None,
        map_anchor_encoder=None,
        motion_anchor_encoder=None,
        graph_model=None,
        temp_graph_model=None,
        inter_graph_model=None,
        det_deformable=None,
        map_deformable=None,
        ego_deformable=None,
        motion_deformable=None,
        ffn=None,
        norm_layer=None,
        det_refine_layer=None,
        map_refine_layer=None,
        ego_refine_layer=None,
        motion_refine_layer=None,
        det_sampler=None,
        map_sampler=None,
        motion_sampler=None,
        det_decoder=None,
        map_decoder=None,
        loss_det_cls=None,
        loss_det_reg=None,
        loss_map_cls=None,
        loss_map_reg=None,
        loss_ego_status=None,
        loss_motion_cls=None,
        loss_motion_reg=None,
        det_reg_weights=None,
        map_reg_weights=None,
        motion_anchor=None,
        gt_cls_key="gt_map_labels",
        gt_reg_key="gt_map_pts",
        gt_id_key="map_instance_id",
        with_instance_id=True,
        use_vlm_in_stage2=True,
        with_close_loop=False,
        open_loop_hz=2,
        close_loop_hz=20,
        open_loop_bank_length=None,
        close_loop_bank_length=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)

        if task_select is None:
            task_select = ["det", "map", "ego", "motion"]
        if query_select is None:
            query_select = ["det", "map", "ego"]
        if operation_order is None:
            operation_order = [
                'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm',
                'split', 'deformable', 'concat', 'ffn', 'norm', 'split', 'refine'
            ]

        self.embed_dims = embed_dims
        self.task_select = task_select
        self.query_select = query_select
        self.with_close_loop = with_close_loop
        self.open_loop_hz = open_loop_hz
        self.close_loop_hz = close_loop_hz
        self.open_loop_bank_length = open_loop_bank_length
        self.close_loop_bank_length = close_loop_bank_length
        self.num_stage1_layers = num_stage1_layers
        self.num_stage2_layers = num_stage2_layers
        self.num_single_frame_decoder = num_single_frame_decoder
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.decouple_attn = decouple_attn
        self.with_instance_id = with_instance_id
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.gt_id_key = gt_id_key
        self.det_reg_weights = det_reg_weights
        self.map_reg_weights = map_reg_weights
        self.operation_order = operation_order
        self.use_vlm_in_stage2 = use_vlm_in_stage2

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        num_refine = operation_order.count('refine')
        num_deform = operation_order.count('deformable')

        if 'det' in query_select:
            self.det_instance_bank = build(det_instance_bank, PLUGIN_LAYERS)
            self.det_anchor_encoder = build(det_anchor_encoder, POSITIONAL_ENCODING)
            self.det_deformable = nn.ModuleList([build(det_deformable, ATTENTION) for _ in range(num_deform)])
            self.det_refine = nn.ModuleList([build(det_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.det_sampler = build(det_sampler, BBOX_SAMPLERS)
            self.det_decoder = build(det_decoder, BBOX_CODERS)
            self.loss_det_cls = build(loss_det_cls, LOSSES)
            self.loss_det_reg = build(loss_det_reg, LOSSES)

        if 'map' in query_select:
            self.map_instance_bank = build(map_instance_bank, PLUGIN_LAYERS)
            self.map_anchor_encoder = build(map_anchor_encoder, POSITIONAL_ENCODING)
            self.map_deformable = nn.ModuleList([build(map_deformable, ATTENTION) for _ in range(num_deform)])
            self.map_refine = nn.ModuleList([build(map_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.map_sampler = build(map_sampler, BBOX_SAMPLERS)
            self.map_decoder = build(map_decoder, BBOX_CODERS)
            self.loss_map_cls = build(loss_map_cls, LOSSES)
            self.loss_map_reg = build(loss_map_reg, LOSSES)

        if 'ego' in query_select:
            self.ego_instance_bank = build(ego_instance_bank, PLUGIN_LAYERS)
            self.ego_anchor_encoder = self.det_anchor_encoder if hasattr(self, 'det_anchor_encoder') else build(det_anchor_encoder, POSITIONAL_ENCODING)
            self.ego_deformable = nn.ModuleList([build(ego_deformable, ATTENTION) for _ in range(num_deform)])
            self.ego_refine = nn.ModuleList([build(ego_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.loss_ego_status = build(loss_ego_status, LOSSES)

        if 'motion' in task_select:
            num_motion_refine = 1 + num_stage2_layers
            self.motion_anchor_encoder = build(motion_anchor_encoder, POSITIONAL_ENCODING)
            self.motion_refine = nn.ModuleList([build(motion_refine_layer, PLUGIN_LAYERS) for _ in range(num_motion_refine)])
            self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS) if motion_sampler else None
            self.loss_motion_cls = build(loss_motion_cls, LOSSES)
            self.loss_motion_reg = build(loss_motion_reg, LOSSES)

            if motion_anchor is not None:
                self.motion_anchor = nn.Parameter(
                    torch.tensor(np.load(motion_anchor), dtype=torch.float32),
                    requires_grad=False,
                )

        self.op_config_map = {
            "concat": [None, None],
            "split": [None, None],
            "gnn": [graph_model, ATTENTION],
            "temp_gnn": [temp_graph_model, ATTENTION],
            "inter_gnn": [inter_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [None, None],
            "refine": [None, None],
        }

        self.layers = nn.ModuleList([
            build(*self.op_config_map.get(op, [None, None])) for op in self.operation_order
        ])

        if decouple_attn:
            self.fc_before = nn.Linear(embed_dims, embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self._stage1_cache = {}

        self.run_step = 0
        self.is_init_bank_list = False

    def init_instance_bank_list(self):
        self.is_init_bank_list = True

        if not self.with_close_loop:
            self.bank_length = self.open_loop_bank_length if self.open_loop_bank_length is not None else 1
            if 'det' in self.query_select:
                self.det_instance_bank_list = [self.det_instance_bank]
            if 'map' in self.query_select:
                self.map_instance_bank_list = [self.map_instance_bank]
            if 'ego' in self.query_select:
                self.ego_instance_bank_list = [self.ego_instance_bank]

        else:
            print("close-loop evaluation")
            self.bank_length = self.close_loop_bank_length if self.close_loop_bank_length is not None \
                else self.close_loop_hz // self.open_loop_hz
            if 'det' in self.query_select:
                self.det_instance_bank_list = [copy.deepcopy(self.det_instance_bank) for _ in range(self.bank_length)]
            if 'map' in self.query_select:
                self.map_instance_bank_list = [copy.deepcopy(self.map_instance_bank) for _ in range(self.bank_length)]
            if 'ego' in self.query_select:
                self.ego_instance_bank_list = [copy.deepcopy(self.ego_instance_bank) for _ in range(self.bank_length)]

        # When no perception queries are active, bank_length must still be set
        if not hasattr(self, 'bank_length'):
            self.bank_length = 1

    def _get_num_anchor_cumsum(self, det_n, map_n, ego_n):
        return np.cumsum([0, det_n, map_n, ego_n]).tolist()

    def _concat_queries(self, det_feat, map_feat, ego_feat):
        return torch.cat([det_feat, map_feat, ego_feat], dim=1)

    def _split_queries(self, x, det_n, map_n, ego_n):
        d = x[:, :det_n]
        m = x[:, det_n:det_n + map_n]
        e = x[:, det_n + map_n:det_n + map_n + ego_n]
        return d, m, e

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack([
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ])
        return torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)

    def get_motion_anchor(self, classification, prediction):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor_raw = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor_raw, prediction)

    def _empty_stage1_output(self, B, device, dtype, feature_maps=None, metas=None):
        """Return a no-op stage1 output when query_select is empty."""
        self._stage1_cache = dict(
            bank_idx=0,
            det_feat=None, map_feat=None, ego_feat=None,
            det_anchor=None, map_anchor=None, ego_anchor=None,
            det_anchor_embed=None, map_anchor_embed=None, ego_anchor_embed=None,
            time_interval=None, map_points_embed=None, motion_mode_query=None,
            temp_det_feat=None, temp_map_feat=None, temp_ego_feat=None,
            temp_det_anchor_embed=None, temp_map_anchor_embed=None, temp_ego_anchor_embed=None,
            temp_map_points_embed=None, with_temp=False, num_temp_anchor_cumsum=[0, 0, 0, 0],
            metas=metas, feature_maps=feature_maps,
            det_cls_last=None, map_cls_last=None,
            _is_empty=True,
        )
        return dict(
            det_instance_feature=torch.zeros(B, 0, self.embed_dims, device=device, dtype=dtype),
            det_anchor=None, det_anchor_embed=None,
            det_classifications=[], det_predictions=[], det_qualities=[],
            map_instance_feature=torch.zeros(B, 0, self.embed_dims, device=device, dtype=dtype),
            map_anchor=None, map_anchor_embed=None,
            map_classifications=[], map_predictions=[], map_qualities=[],
            ego_instance_feature=torch.zeros(B, 0, self.embed_dims, device=device, dtype=dtype),
            ego_anchor=None, ego_anchor_embed=None,
            ego_status_list=[],
            motion_token=None, motion_cls_stage1=None, motion_reg_stage1=None,
            time_interval=None,
        )

    def forward_stage1(self, feature_maps, metas):
        if not self.is_init_bank_list:
            self.init_instance_bank_list()

        B = metas['projection_mat'].shape[0]

        # Early exit when no perception tasks are active
        if not self.query_select:
            _device = metas['projection_mat'].device
            _dtype = metas['projection_mat'].dtype
            return self._empty_stage1_output(B, _device, _dtype, feature_maps=feature_maps, metas=metas)

        bank_idx = self.run_step % self.bank_length

        # Fetch features only for active modalities
        det_feat = det_anchor = temp_det_feat = temp_det_anchor = time_interval = None
        map_feat = map_anchor = temp_map_feat = temp_map_anchor = None
        ego_feat = ego_anchor = temp_ego_feat = temp_ego_anchor = None

        if 'det' in self.query_select:
            det_feat, det_anchor, temp_det_feat, temp_det_anchor, time_interval = \
                self.det_instance_bank_list[bank_idx].get(B, metas, dn_metas=self.det_sampler.dn_metas if hasattr(self.det_sampler, 'dn_metas') else None)
        if 'map' in self.query_select:
            map_feat, map_anchor, temp_map_feat, temp_map_anchor, _ = \
                self.map_instance_bank_list[bank_idx].get(B, metas, dn_metas=self.map_sampler.dn_metas if hasattr(self.map_sampler, 'dn_metas') else None)
        if 'ego' in self.query_select:
            ego_feat, ego_anchor, temp_ego_feat, temp_ego_anchor = \
                self.ego_instance_bank_list[bank_idx].get(B, metas, feature_maps)

        # Create placeholder zero tensors for inactive modalities (needed for shape consistency)
        _d = metas['projection_mat'].device
        _t = next(self.parameters()).dtype
        if det_feat is None:
            det_feat = torch.zeros(B, 0, self.embed_dims, device=_d, dtype=_t)
            det_anchor = torch.zeros(B, 0, 11, device=_d, dtype=_t)
        if map_feat is None:
            map_feat = torch.zeros(B, 0, self.embed_dims, device=_d, dtype=_t)
            map_anchor = torch.zeros(B, 0, 20 * 3, device=_d, dtype=_t)
        if ego_feat is None:
            ego_feat = torch.zeros(B, 0, self.embed_dims, device=_d, dtype=_t)
            ego_anchor = torch.zeros(B, 0, 11, device=_d, dtype=_t)

        det_anchor_embed = self.det_anchor_encoder(det_anchor) if 'det' in self.query_select else torch.zeros(B, 0, self.embed_dims, device=_d, dtype=_t)
        if 'map' in self.query_select:
            map_anchor_embed_out = self.map_anchor_encoder(map_anchor)
            if isinstance(map_anchor_embed_out, (tuple, list)):
                map_anchor_embed, map_points_embed = map_anchor_embed_out
            else:
                map_anchor_embed = map_anchor_embed_out
                map_points_embed = None
        else:
            map_anchor_embed = torch.zeros(B, 0, self.embed_dims, device=_d, dtype=_t)
            map_points_embed = None
        ego_anchor_embed = self.ego_anchor_encoder(ego_anchor) if 'ego' in self.query_select else torch.zeros(B, 0, self.embed_dims, device=_d, dtype=_t)

        temp_det_anchor_embed = self.det_anchor_encoder(temp_det_anchor) if ('det' in self.query_select and temp_det_anchor is not None) else None
        if 'map' in self.query_select and temp_map_anchor is not None:
            temp_map_ae = self.map_anchor_encoder(temp_map_anchor)
            temp_map_anchor_embed = temp_map_ae[0] if isinstance(temp_map_ae, (tuple, list)) else temp_map_ae
        else:
            temp_map_anchor_embed = None
        temp_ego_anchor_embed = self.ego_anchor_encoder(temp_ego_anchor) if ('ego' in self.query_select and temp_ego_anchor is not None) else None

        det_cls_list, det_pred_list, det_qt_list = [], [], []
        map_cls_list, map_pred_list, map_qt_list = [], [], []
        ego_status_list = []

        det_n = det_feat.shape[1]
        map_n = map_feat.shape[1]
        ego_n = ego_feat.shape[1]
        num_anchor_cumsum = self._get_num_anchor_cumsum(det_n, map_n, ego_n)

        temp_det_n = temp_det_feat.shape[1] if temp_det_feat is not None else 0
        temp_map_n = temp_map_feat.shape[1] if temp_map_feat is not None else 0
        temp_ego_n = temp_ego_feat.shape[1] if temp_ego_feat is not None else 0
        num_temp_anchor_cumsum = np.cumsum([0, temp_det_n, temp_map_n, temp_ego_n]).tolist()

        with_temp = (temp_det_feat is not None)

        instance_feature = None
        anchor_embed = None
        temp_instance_feature = None
        temp_anchor_embed = None

        deform_i = 0
        refine_i = 0
        current_layer = 0

        det_cls = None
        det_anchor_new = None
        det_qt = None
        map_cls = None
        map_anchor_new = None
        map_qt = None
        ego_status = None

        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None and op not in ['concat', 'split', 'deformable', 'refine']:
                continue

            if op == "concat":
                anchor_embed_list = []
                instance_feature_list = []
                temp_anchor_embed_list = []
                temp_instance_feature_list = []
                instance_num_list = []
                temp_instance_num_list = []

                for modality in self.query_select:
                    if modality == "det":
                        anchor_embed_list.append(det_anchor_embed)
                        instance_feature_list.append(det_feat)
                        instance_num_list.append(det_n)
                        if with_temp and temp_det_feat is not None:
                            temp_anchor_embed_list.append(temp_det_anchor_embed)
                            temp_instance_feature_list.append(temp_det_feat)
                            temp_instance_num_list.append(temp_det_n)
                        else:
                            temp_instance_num_list.append(0)

                    elif modality == "map":
                        anchor_embed_list.append(map_anchor_embed)
                        instance_feature_list.append(map_feat)
                        instance_num_list.append(map_n)
                        if with_temp and temp_map_feat is not None:
                            temp_anchor_embed_list.append(temp_map_anchor_embed)
                            temp_instance_feature_list.append(temp_map_feat)
                            temp_instance_num_list.append(temp_map_n)
                        else:
                            temp_instance_num_list.append(0)

                    elif modality == "ego":
                        anchor_embed_list.append(ego_anchor_embed)
                        instance_feature_list.append(ego_feat)
                        instance_num_list.append(ego_n)
                        if with_temp and temp_ego_feat is not None:
                            temp_anchor_embed_list.append(temp_ego_anchor_embed)
                            temp_instance_feature_list.append(temp_ego_feat)
                            temp_instance_num_list.append(temp_ego_n)
                        else:
                            temp_instance_num_list.append(0)

                anchor_embed = torch.cat(anchor_embed_list, dim=1)
                instance_feature = torch.cat(instance_feature_list, dim=1)

                if with_temp and len(temp_anchor_embed_list) > 0:
                    temp_anchor_embed = torch.cat(temp_anchor_embed_list, dim=1)
                    temp_instance_feature = torch.cat(temp_instance_feature_list, dim=1)
                else:
                    temp_anchor_embed = None
                    temp_instance_feature = None

            elif op == "split":
                instance_num_cumsum = np.cumsum([0] + instance_num_list)
                for modality, query_start, query_end in zip(
                    self.query_select, instance_num_cumsum[:-1], instance_num_cumsum[1:]
                ):
                    if modality == "det":
                        det_anchor_embed = anchor_embed[:, query_start:query_end]
                        det_feat = instance_feature[:, query_start:query_end]
                    elif modality == "map":
                        map_anchor_embed = anchor_embed[:, query_start:query_end]
                        map_feat = instance_feature[:, query_start:query_end]
                    elif modality == "ego":
                        ego_anchor_embed = anchor_embed[:, query_start:query_end]
                        ego_feat = instance_feature[:, query_start:query_end]

            elif op == "temp_gnn":
                if self.layers[i] is not None and with_temp:
                    instance_feature = self.layers[i](
                        query=instance_feature,
                        key=temp_instance_feature,
                        value=temp_instance_feature,
                        query_pos=anchor_embed,
                        key_pos=temp_anchor_embed,
                        num_anchor_cumsum=num_anchor_cumsum,
                        num_temp_anchor_cumsum=num_temp_anchor_cumsum,
                        fc_before=self.fc_before,
                        fc_after=self.fc_after,
                    )

            elif op == "gnn":
                if self.layers[i] is not None:
                    instance_feature = self.layers[i](
                        query=instance_feature,
                        value=instance_feature,
                        query_pos=anchor_embed,
                        num_anchor_cumsum=num_anchor_cumsum,
                        fc_before=self.fc_before,
                        fc_after=self.fc_after,
                    )

            elif op == "inter_gnn":
                if self.layers[i] is not None:
                    instance_feature = self.layers[i](
                        query=instance_feature,
                        value=instance_feature,
                        query_pos=anchor_embed,
                        num_anchor_cumsum=num_anchor_cumsum,
                        map_points_embed=map_points_embed if map_points_embed is not None else None,
                        det_anchor=det_anchor,
                        map_anchor=map_anchor,
                        fc_before=self.fc_before,
                        fc_after=self.fc_after,
                    )

            elif op == "norm" or op == "ffn":
                if self.layers[i] is not None:
                    instance_feature = self.layers[i](instance_feature)

            elif op == "deformable":
                for modality in self.query_select:
                    if modality == "det":
                        det_feat = self.det_deformable[deform_i](
                            instance_feature=det_feat,
                            anchor=det_anchor,
                            anchor_embed=det_anchor_embed,
                            feature_maps=feature_maps,
                            metas=metas,
                            time_interval=time_interval,
                        )
                    elif modality == "map":
                        map_feat = self.map_deformable[deform_i](
                            instance_feature=map_feat,
                            anchor=map_anchor,
                            anchor_embed=map_points_embed if map_points_embed is not None else map_anchor_embed,
                            feature_maps=feature_maps,
                            metas=metas,
                        )
                    elif modality == "ego":
                        ego_feat = self.ego_deformable[deform_i](
                            instance_feature=ego_feat,
                            anchor=ego_anchor,
                            anchor_embed=ego_anchor_embed,
                            feature_maps=feature_maps,
                            metas=metas,
                        )
                deform_i += 1

            elif op == "refine":
                if "det" in self.task_select:
                    det_anchor_new, det_cls, det_qt = self.det_refine[refine_i](
                        det_feat, det_anchor, det_anchor_embed, time_interval=time_interval,
                    )
                    det_anchor = det_anchor_new

                    det_pred_list.append(det_anchor_new)
                    det_cls_list.append(det_cls)
                    det_qt_list.append(det_qt)

                    if len(det_pred_list) == self.num_single_frame_decoder:
                        det_feat, det_anchor = self.det_instance_bank_list[bank_idx].update(
                            det_feat, det_anchor, det_cls
                        )

                    det_anchor_embed = self.det_anchor_encoder(det_anchor)

                    if len(det_pred_list) > self.num_single_frame_decoder and temp_det_anchor_embed is not None:
                        det_temp_n = self.det_instance_bank_list[bank_idx].num_temp_instances
                        temp_det_anchor_embed = det_anchor_embed[:, :det_temp_n]

                if "map" in self.task_select:
                    map_anchor_new, map_cls, map_qt = self.map_refine[refine_i](
                        map_feat, map_anchor, map_anchor_embed,
                    )
                    map_anchor = map_anchor_new

                    map_pred_list.append(map_anchor_new)
                    map_cls_list.append(map_cls)
                    map_qt_list.append(map_qt)

                    if len(map_pred_list) == self.num_single_frame_decoder:
                        map_feat, map_anchor = self.map_instance_bank_list[bank_idx].update(
                            map_feat, map_anchor, map_cls
                        )

                    map_anchor_embed_out = self.map_anchor_encoder(map_anchor)
                    if isinstance(map_anchor_embed_out, (tuple, list)):
                        map_anchor_embed, map_points_embed = map_anchor_embed_out
                    else:
                        map_anchor_embed = map_anchor_embed_out
                        map_points_embed = None

                    if len(map_pred_list) > self.num_single_frame_decoder and temp_map_anchor_embed is not None:
                        map_temp_n = self.map_instance_bank_list[bank_idx].num_temp_instances
                        temp_map_anchor_embed = map_anchor_embed[:, :map_temp_n]
                        temp_map_points_embed = map_points_embed[:, :map_temp_n * 20] if map_points_embed is not None else None

                if "ego" in self.task_select:
                    ego_status = self.ego_refine[refine_i](ego_feat, ego_anchor_embed)
                    ego_status_list.append(ego_status)

                    ego_anchor_embed = self.ego_anchor_encoder(ego_anchor)

                refine_i += 1
                current_layer += 1

                if current_layer >= self.num_stage1_layers:
                    break

        motion_token = None
        motion_cls_s1 = None
        motion_reg_s1 = None
        if 'motion' in self.task_select:
            det_cls_prob = det_cls_list[-1].sigmoid()
            motion_anchor = self.get_motion_anchor(det_cls_prob, det_anchor)
            motion_mode_query = self.motion_anchor_encoder(
                gen_sineembed_for_position(motion_anchor[..., -1, :])
            )
            motion_query = motion_mode_query + (det_feat + det_anchor_embed).unsqueeze(2)
            motion_cls_s1, motion_reg_s1 = self.motion_refine[0](motion_query)

            best_mode = motion_cls_s1.argmax(dim=-1)
            best_mode_idx = best_mode[..., None, None].expand(-1, -1, 1, self.embed_dims)
            motion_token = motion_query.gather(dim=2, index=best_mode_idx).squeeze(2)
        else:
            motion_mode_query = None

        self._stage1_cache = dict(
            bank_idx=bank_idx,
            det_feat=det_feat,
            map_feat=map_feat,
            ego_feat=ego_feat,
            det_anchor=det_anchor,
            map_anchor=map_anchor,
            ego_anchor=ego_anchor,
            det_anchor_embed=det_anchor_embed,
            map_anchor_embed=map_anchor_embed,
            ego_anchor_embed=ego_anchor_embed,
            time_interval=time_interval,
            map_points_embed=map_points_embed,
            motion_mode_query=motion_mode_query,
            temp_det_feat=temp_det_feat,
            temp_map_feat=temp_map_feat,
            temp_ego_feat=temp_ego_feat,
            temp_det_anchor_embed=temp_det_anchor_embed,
            temp_map_anchor_embed=temp_map_anchor_embed,
            temp_ego_anchor_embed=temp_ego_anchor_embed,
            temp_map_points_embed=temp_map_points_embed if 'temp_map_points_embed' in locals() else None,
            with_temp=with_temp,
            num_temp_anchor_cumsum=num_temp_anchor_cumsum,
            metas=metas,
            feature_maps=feature_maps,
            det_cls_last=det_cls_list[-1] if det_cls_list else None,
            map_cls_last=map_cls_list[-1] if map_cls_list else None,
        )

        return dict(
            det_instance_feature=det_feat,
            det_anchor=det_anchor,
            det_anchor_embed=det_anchor_embed,
            det_classifications=det_cls_list,
            det_predictions=det_pred_list,
            det_qualities=det_qt_list,
            map_instance_feature=map_feat,
            map_anchor=map_anchor,
            map_anchor_embed=map_anchor_embed,
            map_classifications=map_cls_list,
            map_predictions=map_pred_list,
            map_qualities=map_qt_list,
            ego_instance_feature=ego_feat,
            ego_anchor=ego_anchor,
            ego_anchor_embed=ego_anchor_embed,
            ego_status_list=ego_status_list,
            motion_token=motion_token,
            motion_cls_stage1=motion_cls_s1,
            motion_reg_stage1=motion_reg_s1,
            time_interval=time_interval,
        )

    def forward_stage2(self, vlm_enhanced, feature_maps, metas):
        c = self._stage1_cache

        # Early exit when no perception tasks are active
        if c.get('_is_empty', False):
            self.run_step += 1
            return dict(
                det_instance_feature=c['det_feat'] if c['det_feat'] is not None else
                    torch.zeros(metas['projection_mat'].shape[0], 0, self.embed_dims,
                                device=metas['projection_mat'].device, dtype=metas['projection_mat'].dtype),
                det_anchor=None, det_classifications=[], det_predictions=[], det_qualities=[],
                map_instance_feature=c['map_feat'] if c['map_feat'] is not None else
                    torch.zeros(metas['projection_mat'].shape[0], 0, self.embed_dims,
                                device=metas['projection_mat'].device, dtype=metas['projection_mat'].dtype),
                map_anchor=None, map_classifications=[], map_predictions=[], map_qualities=[],
                ego_instance_feature=c['ego_feat'] if c['ego_feat'] is not None else
                    torch.zeros(metas['projection_mat'].shape[0], 0, self.embed_dims,
                                device=metas['projection_mat'].device, dtype=metas['projection_mat'].dtype),
                ego_status_list=[],
                motion_cls_list=[], motion_reg_list=[],
                time_interval=None,
            )

        if self.use_vlm_in_stage2:
            det_feat = vlm_enhanced['det_feat']
            map_feat = vlm_enhanced['map_feat']
            ego_feat = vlm_enhanced['ego_feat']
        else:
            det_feat = c['det_feat']
            map_feat = c['map_feat']
            ego_feat = c['ego_feat']
        motion_enhanced = vlm_enhanced.get('motion_feat', None) if self.use_vlm_in_stage2 else None

        det_anchor       = c['det_anchor']
        map_anchor       = c['map_anchor']
        ego_anchor       = c['ego_anchor']
        det_anchor_embed = c['det_anchor_embed']
        map_anchor_embed = c['map_anchor_embed']
        ego_anchor_embed = c['ego_anchor_embed']
        map_points_embed = c.get('map_points_embed', None)
        time_interval    = c['time_interval']
        motion_mode_query = c.get('motion_mode_query', None)

        temp_det_feat         = c.get('temp_det_feat', None)
        temp_map_feat         = c.get('temp_map_feat', None)
        temp_ego_feat         = c.get('temp_ego_feat', None)
        temp_det_anchor_embed = c.get('temp_det_anchor_embed', None)
        temp_map_anchor_embed = c.get('temp_map_anchor_embed', None)
        temp_ego_anchor_embed = c.get('temp_ego_anchor_embed', None)
        temp_map_points_embed = c.get('temp_map_points_embed', None)
        with_temp             = c.get('with_temp', False)
        num_temp_anchor_cumsum = c.get('num_temp_anchor_cumsum', [0, 0, 0, 0])

        det_cls_list, det_pred_list, det_qt_list = [], [], []
        map_cls_list, map_pred_list, map_qt_list = [], [], []
        ego_status_list = []
        motion_cls_list, motion_reg_list = [], []

        query_select_stage2 = [m for m in self.query_select if m != "motion"]

        det_n = det_feat.shape[1]
        map_n = map_feat.shape[1]
        ego_n = ego_feat.shape[1]

        num_anchor_cumsum = self._get_num_anchor_cumsum(det_n, map_n, ego_n)

        temp_det_n = temp_det_feat.shape[1] if temp_det_feat is not None else 0
        temp_map_n = temp_map_feat.shape[1] if temp_map_feat is not None else 0
        temp_ego_n = temp_ego_feat.shape[1] if temp_ego_feat is not None else 0
        num_temp_anchor_cumsum = np.cumsum([0, temp_det_n, temp_map_n, temp_ego_n]).tolist()

        instance_feature      = None
        anchor_embed          = None
        temp_instance_feature = None
        temp_anchor_embed     = None
        instance_num_list     = []

        deform_i      = self.num_stage1_layers
        refine_i      = self.num_stage1_layers
        current_layer = self.num_stage1_layers

        det_cls = det_anchor_new = det_qt = None
        map_cls = map_anchor_new = map_qt = None
        ego_status = None

        stage2_start_idx = 0
        _refine_count = 0
        for idx, op in enumerate(self.operation_order):
            if op == 'refine':
                _refine_count += 1
                if _refine_count == self.num_stage1_layers:
                    stage2_start_idx = idx + 1
                    break

        for i in range(stage2_start_idx, len(self.operation_order)):
            op = self.operation_order[i]

            if self.layers[i] is None and op not in ['concat', 'split', 'deformable', 'refine']:
                continue

            if op == "concat":
                anchor_embed_list           = []
                instance_feature_list       = []
                temp_anchor_embed_list      = []
                temp_instance_feature_list  = []
                instance_num_list           = []
                temp_instance_num_list      = []

                for modality in query_select_stage2:
                    if modality == "det":
                        anchor_embed_list.append(det_anchor_embed)
                        instance_feature_list.append(det_feat)
                        instance_num_list.append(det_n)
                        if with_temp and temp_det_feat is not None:
                            temp_anchor_embed_list.append(temp_det_anchor_embed)
                            temp_instance_feature_list.append(temp_det_feat)
                            temp_instance_num_list.append(temp_det_n)
                        else:
                            temp_instance_num_list.append(0)
                    elif modality == "map":
                        anchor_embed_list.append(map_anchor_embed)
                        instance_feature_list.append(map_feat)
                        instance_num_list.append(map_n)
                        if with_temp and temp_map_feat is not None:
                            temp_anchor_embed_list.append(temp_map_anchor_embed)
                            temp_instance_feature_list.append(temp_map_feat)
                            temp_instance_num_list.append(temp_map_n)
                        else:
                            temp_instance_num_list.append(0)
                    elif modality == "ego":
                        anchor_embed_list.append(ego_anchor_embed)
                        instance_feature_list.append(ego_feat)
                        instance_num_list.append(ego_n)
                        if with_temp and temp_ego_feat is not None:
                            temp_anchor_embed_list.append(temp_ego_anchor_embed)
                            temp_instance_feature_list.append(temp_ego_feat)
                            temp_instance_num_list.append(temp_ego_n)
                        else:
                            temp_instance_num_list.append(0)

                anchor_embed      = torch.cat(anchor_embed_list, dim=1)
                instance_feature  = torch.cat(instance_feature_list, dim=1)

                if with_temp and len(temp_anchor_embed_list) > 0:
                    temp_anchor_embed      = torch.cat(temp_anchor_embed_list, dim=1)
                    temp_instance_feature  = torch.cat(temp_instance_feature_list, dim=1)
                else:
                    temp_anchor_embed     = None
                    temp_instance_feature = None

            elif op == "split":
                instance_num_cumsum = np.cumsum([0] + instance_num_list)
                for modality, qs, qe in zip(
                    query_select_stage2,
                    instance_num_cumsum[:-1],
                    instance_num_cumsum[1:]
                ):
                    if modality == "det":
                        det_anchor_embed = anchor_embed[:, qs:qe]
                        det_feat         = instance_feature[:, qs:qe]
                    elif modality == "map":
                        map_anchor_embed = anchor_embed[:, qs:qe]
                        map_feat         = instance_feature[:, qs:qe]
                    elif modality == "ego":
                        ego_anchor_embed = anchor_embed[:, qs:qe]
                        ego_feat         = instance_feature[:, qs:qe]

            elif op == "temp_gnn":
                if self.layers[i] is not None and with_temp and temp_instance_feature is not None:
                    instance_feature = self.layers[i](
                        query=instance_feature,
                        key=temp_instance_feature,
                        value=temp_instance_feature,
                        query_pos=anchor_embed,
                        key_pos=temp_anchor_embed,
                        num_anchor_cumsum=num_anchor_cumsum,
                        num_temp_anchor_cumsum=num_temp_anchor_cumsum,
                        fc_before=self.fc_before,
                        fc_after=self.fc_after,
                    )

            elif op == "gnn":
                if self.layers[i] is not None:
                    instance_feature = self.layers[i](
                        query=instance_feature,
                        value=instance_feature,
                        query_pos=anchor_embed,
                        num_anchor_cumsum=num_anchor_cumsum,
                        fc_before=self.fc_before,
                        fc_after=self.fc_after,
                    )

            elif op == "inter_gnn":
                if self.layers[i] is not None:
                    instance_feature = self.layers[i](
                        query=instance_feature,
                        value=instance_feature,
                        query_pos=anchor_embed,
                        num_anchor_cumsum=num_anchor_cumsum,
                        map_points_embed=map_points_embed if map_points_embed is not None else None,
                        det_anchor=det_anchor,
                        map_anchor=map_anchor,
                        fc_before=self.fc_before,
                        fc_after=self.fc_after,
                    )

            elif op == "norm" or op == "ffn":
                if self.layers[i] is not None:
                    instance_feature = self.layers[i](instance_feature)

            elif op == "deformable":
                for modality in query_select_stage2:
                    if modality == "det":
                        det_feat = self.det_deformable[deform_i](
                            instance_feature=det_feat,
                            anchor=det_anchor,
                            anchor_embed=det_anchor_embed,
                            feature_maps=feature_maps,
                            metas=metas,
                            time_interval=time_interval,
                        )
                    elif modality == "map":
                        map_feat = self.map_deformable[deform_i](
                            instance_feature=map_feat,
                            anchor=map_anchor,
                            anchor_embed=map_points_embed if map_points_embed is not None else map_anchor_embed,
                            feature_maps=feature_maps,
                            metas=metas,
                        )
                    elif modality == "ego":
                        ego_feat = self.ego_deformable[deform_i](
                            instance_feature=ego_feat,
                            anchor=ego_anchor,
                            anchor_embed=ego_anchor_embed,
                            feature_maps=feature_maps,
                            metas=metas,
                        )
                deform_i += 1

            elif op == "refine":
                if "det" in self.task_select:
                    det_anchor_new, det_cls, det_qt = self.det_refine[refine_i](
                        det_feat, det_anchor, det_anchor_embed, time_interval=time_interval,
                    )
                    det_anchor = det_anchor_new

                    det_pred_list.append(det_anchor_new)
                    det_cls_list.append(det_cls)
                    det_qt_list.append(det_qt)

                    det_anchor_embed = self.det_anchor_encoder(det_anchor)

                    if temp_det_anchor_embed is not None:
                        det_temp_n = self.det_instance_bank.num_temp_instances
                        temp_det_anchor_embed = det_anchor_embed[:, :det_temp_n]

                if "map" in self.task_select:
                    map_anchor_new, map_cls, map_qt = self.map_refine[refine_i](
                        map_feat, map_anchor, map_anchor_embed,
                    )
                    map_anchor = map_anchor_new

                    map_pred_list.append(map_anchor_new)
                    map_cls_list.append(map_cls)
                    map_qt_list.append(map_qt)

                    map_anchor_embed_out = self.map_anchor_encoder(map_anchor)
                    if isinstance(map_anchor_embed_out, (tuple, list)):
                        map_anchor_embed, map_points_embed = map_anchor_embed_out
                    else:
                        map_anchor_embed = map_anchor_embed_out
                        map_points_embed = None

                    if temp_map_anchor_embed is not None:
                        map_temp_n = self.map_instance_bank.num_temp_instances
                        temp_map_anchor_embed = map_anchor_embed[:, :map_temp_n]
                        temp_map_points_embed = map_points_embed[:, :map_temp_n * 20] if map_points_embed is not None else None

                if "ego" in self.task_select:
                    ego_status = self.ego_refine[refine_i](ego_feat, ego_anchor_embed)
                    ego_status_list.append(ego_status)

                    ego_anchor_embed = self.ego_anchor_encoder(ego_anchor)

                if "motion" in self.task_select and motion_mode_query is not None:
                    motion_base = det_feat + det_anchor_embed
                    if motion_enhanced is not None:
                        motion_base = motion_base + motion_enhanced
                    motion_query = motion_mode_query + motion_base.unsqueeze(2)
                    m_cls, m_reg = self.motion_refine[1 + (refine_i - self.num_stage1_layers)](motion_query)
                    motion_cls_list.append(m_cls)
                    motion_reg_list.append(m_reg)

                refine_i += 1
                current_layer += 1

        _metas        = c['metas']
        _feature_maps = c['feature_maps']
        _bank_idx     = c['bank_idx']
        if 'det' in self.query_select:
            self.det_instance_bank_list[_bank_idx].cache(
                det_feat, det_anchor, det_cls,
                metas=_metas, feature_maps=_feature_maps,
            )
        if 'map' in self.query_select:
            self.map_instance_bank_list[_bank_idx].cache(
                map_feat, map_anchor, map_cls,
                metas=_metas, feature_maps=_feature_maps,
            )
        if 'ego' in self.query_select:
            self.ego_instance_bank_list[_bank_idx].cache(
                ego_feat, ego_anchor,
                metas=_metas, feature_maps=_feature_maps,
            )

        self.run_step += 1

        return dict(
            det_instance_feature=det_feat,
            det_anchor=det_anchor,
            det_classifications=det_cls_list,
            det_predictions=det_pred_list,
            det_qualities=det_qt_list,
            map_instance_feature=map_feat,
            map_anchor=map_anchor,
            map_classifications=map_cls_list,
            map_predictions=map_pred_list,
            map_qualities=map_qt_list,
            ego_instance_feature=ego_feat,
            ego_status_list=ego_status_list,
            motion_cls_list=motion_cls_list,
            motion_reg_list=motion_reg_list,
            time_interval=time_interval,
        )

    def loss(self, stage1_outs, stage2_outs, data):
        losses = {}
        zero = stage1_outs['det_instance_feature'].sum() * 0.0

        all_det_cls = stage1_outs['det_classifications'] + stage2_outs['det_classifications']
        all_det_pred = stage1_outs['det_predictions'] + stage2_outs['det_predictions']
        all_det_qt = stage1_outs['det_qualities'] + stage2_outs['det_qualities']

        all_map_cls = stage1_outs['map_classifications'] + stage2_outs['map_classifications']
        all_map_pred = stage1_outs['map_predictions'] + stage2_outs['map_predictions']
        all_map_qt = stage1_outs['map_qualities'] + stage2_outs['map_qualities']

        all_ego_status = stage1_outs['ego_status_list'] + stage2_outs['ego_status_list']

        if "det" in self.task_select and self.loss_det_cls is not None:
            det_output = {
                'classification': all_det_cls,
                'prediction': all_det_pred,
                'quality': all_det_qt,
            }
            gt_boxes = data.get('gt_bboxes_3d')
            gt_labels = data.get('gt_labels_3d')

            if gt_boxes is not None and gt_labels is not None:
                det_losses = self._compute_det_loss(det_output, gt_labels, gt_boxes)
                losses.update(det_losses)
            else:
                for i in range(len(all_det_cls)):
                    losses[f'det_loss_cls_{i}'] = zero
                    losses[f'det_loss_box_{i}'] = zero
                    losses[f'det_loss_cns_{i}'] = zero
                    losses[f'det_loss_yns_{i}'] = zero

        if "map" in self.task_select and self.loss_map_cls is not None:
            map_output = {
                'classification': all_map_cls,
                'prediction': all_map_pred,
                'quality': all_map_qt,
            }
            gt_map_labels = data.get(self.gt_cls_key)
            gt_map_pts = data.get(self.gt_reg_key)

            if gt_map_labels is not None and gt_map_pts is not None:
                map_losses = self._compute_map_loss(map_output, gt_map_labels, gt_map_pts)
                losses.update(map_losses)
            else:
                for i in range(len(all_map_cls)):
                    losses[f'map_loss_cls_{i}'] = zero
                    losses[f'map_loss_line_{i}'] = zero

        if "ego" in self.task_select and self.loss_ego_status is not None:
            ego_gt = data.get('ego_status')
            if ego_gt is not None:
                ego_status_weight = data.get('ego_status_mask', None)
                ego_loss = torch.tensor(0.0, device=zero.device, dtype=zero.dtype)
                for status_pred in all_ego_status:
                    ego_gt_t = ego_gt.to(device=status_pred.device, dtype=status_pred.dtype)
                    if ego_gt_t.dim() == 2:
                        ego_gt_t = ego_gt_t[:, :status_pred.shape[-1]]
                    loss_kwargs = {}
                    if ego_status_weight is not None:
                        w = ego_status_weight.to(device=status_pred.device, dtype=status_pred.dtype)
                        if w.dim() == 2:
                            w = w[:, :status_pred.shape[-1]]
                        loss_kwargs['weight'] = w
                    ego_loss = ego_loss + self.loss_ego_status(
                        status_pred.squeeze(1), ego_gt_t, **loss_kwargs
                    )
                losses['loss_ego_status'] = ego_loss
            else:
                losses['loss_ego_status'] = zero

        has_motion_sampler = hasattr(self, 'motion_sampler') and self.motion_sampler is not None
        if "motion" in self.task_select and has_motion_sampler:
            motion_cls_all = [stage1_outs['motion_cls_stage1']] + stage2_outs.get('motion_cls_list', [])
            motion_reg_all = [stage1_outs['motion_reg_stage1']] + stage2_outs.get('motion_reg_list', [])

            gt_agent_trajs = data.get('gt_agent_fut_trajs')
            gt_agent_masks = data.get('gt_agent_fut_masks')

            if gt_agent_trajs is not None and gt_agent_masks is not None:
                motion_loss = self._compute_motion_loss(
                    motion_cls_all, motion_reg_all,
                    gt_agent_trajs, gt_agent_masks,
                )
                losses['loss_motion'] = motion_loss
            else:
                losses['loss_motion'] = zero
        else:
            losses['loss_motion'] = zero

        return losses

    def _compute_det_loss(self, output, gt_labels, gt_boxes):
        losses = {}
        for i, (cls, pred, qt) in enumerate(
            zip(output['classification'], output['prediction'], output['quality'])
        ):
            if self.det_reg_weights is not None:
                pred = pred[..., : len(self.det_reg_weights)]

            cls_target, box_target, box_weight = \
                self.det_sampler.sample(cls, pred, gt_labels, gt_boxes)

            if self.det_reg_weights is not None:
                box_target = box_target[..., : len(self.det_reg_weights)]

            mask = torch.logical_not(torch.all(box_target == 0, dim=-1))
            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=pred.dtype)), 1.0)

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)

            _cls = cls.flatten(end_dim=1).float()
            _cls_t = cls_target.flatten(end_dim=1)
            losses[f'det_loss_cls_{i}'] = self.loss_det_cls(
                _cls, _cls_t, avg_factor=num_pos
            )

            mask = mask.reshape(-1)
            if self.det_reg_weights is not None:
                box_weight = box_weight * pred.new_tensor(self.det_reg_weights)

            _bw = box_weight.flatten(end_dim=1)[mask]
            _pred = pred.flatten(end_dim=1)[mask]
            _bt = box_target.flatten(end_dim=1)[mask]

            _bt = torch.where(_bt.isnan(), _pred.new_tensor(0.0), _bt)
            _cls_t_masked = _cls_t[mask]
            _qt = qt.flatten(end_dim=1)[mask] if qt is not None else None

            reg_losses = self.loss_det_reg(
                _pred, _bt, weight=_bw, avg_factor=num_pos,
                quality=_qt,
                cls_target=_cls_t_masked,
                prefix=f'det_loss',
                suffix=f'_{i}',
            )

            if isinstance(reg_losses, dict):
                losses.update(reg_losses)
            else:
                losses[f'det_loss_box_{i}'] = reg_losses
        return losses

    def _compute_map_loss(self, output, gt_labels, gt_pts):
        losses = {}
        for i, (cls, pred, qt) in enumerate(
            zip(output['classification'], output['prediction'], output['quality'])
        ):
            if self.map_reg_weights is not None:
                pred = pred[..., : len(self.map_reg_weights)]

            cls_target, box_target, box_weight = \
                self.map_sampler.sample(cls, pred, gt_labels, gt_pts)

            if self.map_reg_weights is not None:
                box_target = box_target[..., : len(self.map_reg_weights)]

            mask = torch.logical_not(torch.all(box_target == 0, dim=-1))
            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=pred.dtype)), 1.0)

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)

            _cls = cls.flatten(end_dim=1).float()
            _cls_t = cls_target.flatten(end_dim=1)
            losses[f'map_loss_cls_{i}'] = self.loss_map_cls(
                _cls, _cls_t, avg_factor=num_pos
            )

            mask = mask.reshape(-1)
            if self.map_reg_weights is not None:
                box_weight = box_weight * pred.new_tensor(self.map_reg_weights)

            _bw = box_weight.flatten(end_dim=1)[mask]
            _pred = pred.flatten(end_dim=1)[mask]
            _bt = box_target.flatten(end_dim=1)[mask]

            _bt = torch.where(_bt.isnan(), _pred.new_tensor(0.0), _bt)
            _cls_t_masked = _cls_t[mask]
            _qt = qt.flatten(end_dim=1)[mask] if qt is not None else None

            reg_losses = self.loss_map_reg(
                _pred, _bt, weight=_bw, avg_factor=num_pos,
                quality=_qt,
                cls_target=_cls_t_masked,
                prefix=f'map_loss',
                suffix=f'_{i}',
            )
            if isinstance(reg_losses, dict):
                losses.update(reg_losses)
            else:
                losses[f'map_loss_line_{i}'] = reg_losses
        return losses

    def _compute_motion_loss(self, cls_list, reg_list, gt_trajs, gt_masks):
        total_loss = torch.tensor(0.0, device=cls_list[0].device, dtype=cls_list[0].dtype)
        motion_cache = {"indices": self.det_sampler.indices}

        for m_cls, m_reg in zip(cls_list, reg_list):
            cls_t, cls_w, best_reg, reg_t, reg_w, num_pos = self.motion_sampler.sample(
                m_reg.float(),
                [t.to(device=m_reg.device, dtype=torch.float32) for t in gt_trajs],
                [t.to(device=m_reg.device, dtype=torch.float32) for t in gt_masks],
                motion_cache,
            )
            num_pos_val = max(float(num_pos.item()), 1.0)

            _cls = m_cls.flatten(end_dim=1).float()
            _cls_t = cls_t.flatten(end_dim=1)
            _cls_w = cls_w.flatten(end_dim=1).float()
            loss_cls = self.loss_motion_cls(_cls, _cls_t, weight=_cls_w, avg_factor=num_pos_val)

            _rw = reg_w.flatten(end_dim=1).unsqueeze(-1).float()
            _rp = best_reg.flatten(end_dim=1).float().cumsum(dim=-2)
            _rt = reg_t.flatten(end_dim=1).float().cumsum(dim=-2)
            loss_reg = self.loss_motion_reg(_rp, _rt, weight=_rw, avg_factor=num_pos_val)

            total_loss = total_loss + loss_cls + loss_reg

        return total_loss

    def post_process(self, stage2_outs):
        det_result = None
        map_result = None

        if self.det_decoder is not None:
            det_cls_list = stage2_outs.get('det_classifications', [])
            det_pred_list = stage2_outs.get('det_predictions', [])
            det_qt_list = stage2_outs.get('det_qualities', [])

            if len(det_cls_list) > 0:
                if len(det_qt_list) == 0 or all(qt is None for qt in det_qt_list):
                    det_qt_list = None

                bank_idx = (self.run_step - 1) % self.bank_length

                instance_id = None
                if self.with_instance_id:
                    try:
                        instance_id = self.det_instance_bank_list[bank_idx].get_instance_id(det_cls_list[-1])
                    except Exception as e:
                        instance_id = None

                det_result = self.det_decoder.decode(
                    det_cls_list,
                    det_pred_list,
                    instance_id=instance_id,
                    quality=det_qt_list,
                    output_idx=-1
                )

        if self.map_decoder is not None:
            map_cls_list = stage2_outs.get('map_classifications', [])
            map_pred_list = stage2_outs.get('map_predictions', [])

            if len(map_cls_list) > 0:
                map_result = self.map_decoder.decode(
                    map_cls_list,
                    map_pred_list,
                    output_idx=-1
                )

        return det_result, map_result
