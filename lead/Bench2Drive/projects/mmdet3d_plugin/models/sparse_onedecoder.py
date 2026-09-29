import math
import copy
import warnings
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
from mmcv.runner import BaseModule, force_fp32
from mmcv.runner.base_module import BaseModule, Sequential

from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES

from .blocks import linear_relu_ln
from .attention import gen_sineembed_for_position

from typing import List, Optional, Tuple, Union
from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.utils import *

__all__ = ["SparseOneDecoder"]


@HEADS.register_module()
class SparseOneDecoder(BaseModule):
    def __init__(
            self,
            ffn: dict = None,
            init_cfg: dict = None,
            custom_op: dict = None,
            embed_dims: int = 256,
            num_decoder: int = 6,
            num_single_frame_decoder: int = -1,

            decouple_attn: bool = True,
            with_instance_id: bool = True,
            cls_threshold_to_reg: float = -1,

            norm_layer: dict = None,
            graph_model: dict = None,
            temp_graph_model: dict = None,
            inter_graph_model: dict = None,
            operation_order: Optional[List[str]] = None,

            det_instance_bank: dict = None,
            map_instance_bank: dict = None,
            ego_instance_bank: dict = None,
            plan_instance_bank: dict = None,
            scenes_instance_bank: dict = None,

            det_anchor_encoder: dict = None,
            map_anchor_encoder: dict = None,
            ego_anchor_encoder: dict = None,
            plan_anchor_encoder: dict = None,

            det_deformable: dict = None,
            map_deformable: dict = None,
            ego_deformable: dict = None,
            plan_deformable: dict = None,
            scenes_attention: dict = None,

            det_refine_layer: dict = None,
            map_refine_layer: dict = None,
            ego_refine_layer: dict = None,
            plan_refine_layer: dict = None,
            motion_refine_layer: dict = None,
            scenes_refine_layer: dict = None,

            loss_det_cls: dict = None,
            loss_det_reg: dict = None,
            loss_map_cls: dict = None,
            loss_map_reg: dict = None,
            loss_ego_cls: dict = None,
            loss_ego_reg: dict = None,
            loss_ego_status: dict = None,
            loss_plan_cls: dict = None,
            loss_plan_reg: dict = None,
            loss_plan_col: dict = None,
            loss_plan_dir: dict = None,
            loss_plan_bound: dict = None,
            loss_plan_status: dict = None,
            loss_motion_cls: dict = None,
            loss_motion_reg: dict = None,
            loss_scenes_reg: dict = None,

            det_reg_weights: List = None,
            map_reg_weights: List = None,

            det_decoder: dict = None,
            map_decoder: dict = None,
            ego_decoder: dict = None,
            plan_decoder: dict = None,
            motion_decoder: dict = None,

            det_sampler: dict = None,
            map_sampler: dict = None,
            ego_sampler: dict = None,
            plan_sampler: dict = None,
            align_sampler: dict = None,
            motion_sampler: dict = None,

            motion_anchor=None,
            plan_speed_refer=None,
            plan_anchor_refer=None,
            combine_layer_loss=True,

            task_select=["det", "map", "motion", "plan"],
            query_select=["det", "map", "plan"],

            num_command=6,
            independent_gnn=True,
            independent_temp_gnn=True,
            independent_inter_gnn=True,

            with_close_loop=False,  # default False, True means use closed-loop memory bank
            open_loop_hz=2,
            close_loop_hz=20,
            open_loop_bank_length=None,
            close_loop_bank_length=None,

            attn_mask_dict=dict(),
            with_attn_mask=False,               # enable mask-attn
            with_distance_attn_mask=False,      # enable and set distance attn_mask
            with_velocity_attn_mask=False,      # enable and set velocity attn_mask
            with_plan_group_attn_mask=False,    # enable and set plan_group attn_mask

            with_command_embed=False,           # embeds command to planning query
            with_target_point_embed=False,      # embeds target points to planning query
            with_target_point_next_embed=False, # embeds next target points to planning query
            with_custom_status_embed=False,     # embeds ego status to planning query
            with_supervise_ego_status=False,    # enable ego_status supervise ego query predicts
            with_ego_instance_feature=False,    # enable plan_instance_feature+= ego_instance_feature

            with_concat_map_points=False,   # map instance to points in [concat, gnn, norm, split] op
            with_deform_map_points=False,   # map instance to points in [deformable] op
            with_concat_plan_points=False,  # plan instance to points in [concat, gnn, norm, split] op
            with_deform_plan_points=False,  # plan instance to points in [deformable] op

            with_topk_mode=False,
            topk_mode_list=None,
            keep_topk_relative_pos=False,

            **kwargs,
    ):
        super(SparseOneDecoder, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder

        self.with_instance_id = with_instance_id
        self.with_command_embed = with_command_embed
        self.with_target_point_embed = with_target_point_embed
        self.with_target_point_next_embed = with_target_point_next_embed
        self.with_custom_status_embed = with_custom_status_embed

        self.attn_mask_dict = attn_mask_dict
        self.with_attn_mask = with_attn_mask
        self.with_ego_instance_feature = with_ego_instance_feature
        self.with_supervise_ego_status = with_supervise_ego_status

        self.with_plan_group_attn_mask = with_plan_group_attn_mask

        self.with_concat_map_points = with_concat_map_points
        self.with_deform_map_points = with_deform_map_points
        self.with_concat_plan_points = with_concat_plan_points
        self.with_deform_plan_points = with_deform_plan_points
        self.with_distance_attn_mask = with_distance_attn_mask
        self.with_velocity_attn_mask = with_velocity_attn_mask

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.cls_threshold_to_reg = cls_threshold_to_reg

        self.independent_gnn = independent_gnn
        self.independent_temp_gnn = independent_temp_gnn
        self.independent_inter_gnn = independent_inter_gnn

        self.embed_dims = embed_dims
        self.task_select = task_select
        self.query_select = query_select

        self.with_close_loop = with_close_loop
        self.open_loop_hz = open_loop_hz
        self.close_loop_hz = close_loop_hz
        self.open_loop_bank_length = open_loop_bank_length
        self.close_loop_bank_length = close_loop_bank_length

        self.with_topk_mode = with_topk_mode
        self.topk_mode_list = topk_mode_list
        self.keep_topk_relative_pos = keep_topk_relative_pos

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        assert len(self.task_select)
        assert len(self.query_select)

        num_refine = operation_order.count('refine')
        num_deform = operation_order.count('deformable')

        if 'det' in self.query_select:
            self.det_instance_bank = build(det_instance_bank, PLUGIN_LAYERS)
            self.det_anchor_encoder = build(det_anchor_encoder, POSITIONAL_ENCODING)
            self.det_deformable = nn.ModuleList([build(det_deformable, ATTENTION) for _ in range(num_deform)])
            self.det_refine = nn.ModuleList([build(det_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.det_sampler = build(det_sampler, BBOX_SAMPLERS)
            self.det_decoder = build(det_decoder, BBOX_CODERS)
            self.loss_det_cls = build(loss_det_cls, LOSSES)
            self.loss_det_reg = build(loss_det_reg, LOSSES)
            self.det_reg_weights = det_reg_weights
            self.num_det_anchor = self.det_instance_bank.num_anchor
            self.num_temp_det_anchor = self.det_instance_bank.num_temp_instances

        if 'map' in self.query_select:
            self.map_instance_bank = build(map_instance_bank, PLUGIN_LAYERS)
            self.map_anchor_encoder = build(map_anchor_encoder, POSITIONAL_ENCODING)
            self.map_deformable = nn.ModuleList([build(map_deformable, ATTENTION) for _ in range(num_deform)])
            self.map_refine = nn.ModuleList([build(map_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.map_sampler = build(map_sampler, BBOX_SAMPLERS)
            self.map_decoder = build(map_decoder, BBOX_CODERS)
            self.map_reg_weights = map_reg_weights
            self.loss_map_cls = build(loss_map_cls, LOSSES)
            self.loss_map_reg = build(loss_map_reg, LOSSES)
            self.num_map_anchor = self.map_instance_bank.num_anchor
            self.num_temp_map_anchor = self.map_instance_bank.num_temp_instances

            if self.with_concat_map_points:
                self.suqueeze_map_instance = nn.Sequential(
                    nn.Linear(embed_dims * 20, embed_dims * 20 // 4),
                    nn.ReLU(),
                    nn.Linear(embed_dims * 20 // 4, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, embed_dims),
                )

        if 'ego' in self.query_select:
            self.ego_instance_bank = build(ego_instance_bank, PLUGIN_LAYERS)
            if ego_anchor_encoder is not None:
                self.ego_anchor_encoder = build(ego_anchor_encoder, POSITIONAL_ENCODING)
            elif hasattr(self, 'det_anchor_encoder'):
                self.ego_anchor_encoder = self.det_anchor_encoder
            else:
                self.ego_anchor_encoder = build(det_anchor_encoder, POSITIONAL_ENCODING)
            self.ego_deformable = nn.ModuleList([build(ego_deformable, ATTENTION) for _ in range(num_deform)])
            self.ego_refine = nn.ModuleList([build(ego_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.ego_decoder = build(ego_decoder, BBOX_CODERS)
            self.ego_sampler = build(ego_sampler, BBOX_SAMPLERS)
            self.loss_ego_cls = build(loss_ego_cls, LOSSES)
            self.loss_ego_reg = build(loss_ego_reg, LOSSES)
            self.loss_ego_status = build(loss_ego_status, LOSSES)

        if 'plan' in self.query_select:
            self.plan_instance_bank = build(plan_instance_bank, PLUGIN_LAYERS)
            self.plan_anchor_encoder = build(plan_anchor_encoder, POSITIONAL_ENCODING)
            self.plan_deformable = nn.ModuleList([build(plan_deformable, ATTENTION) for _ in range(num_deform)])
            self.plan_refine = nn.ModuleList([build(plan_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.plan_decoder = build(plan_decoder, BBOX_CODERS)
            self.plan_sampler = build(plan_sampler, BBOX_SAMPLERS)
            self.align_sampler = build(align_sampler, BBOX_SAMPLERS)
            self.loss_plan_cls = build(loss_plan_cls, LOSSES)
            self.loss_plan_reg = build(loss_plan_reg, LOSSES)
            self.loss_plan_col = build(loss_plan_col, LOSSES)
            self.loss_plan_dir = build(loss_plan_dir, LOSSES)
            self.loss_plan_bound = build(loss_plan_bound, LOSSES)
            self.loss_plan_status = build(loss_plan_status, LOSSES)

            self.ego_fut_ts = self.plan_refine[0].ego_fut_ts
            self.ego_fut_cmd = self.plan_refine[0].ego_fut_cmd
            self.ego_fut_mode = self.plan_refine[0].ego_fut_mode
            self.plan_anchor_group = self.plan_instance_bank.anchor_group
            self.plan_anchor_types = self.plan_instance_bank.anchor_types
            self.plan_speed_refer = plan_speed_refer
            self.plan_anchor_refer = plan_anchor_refer

            if self.with_command_embed:
                self.command_embed_encoder = nn.Sequential(
                    *linear_relu_ln(self.embed_dims, 2, 1, input_dims=num_command),
                    Linear(self.embed_dims, self.embed_dims)
                )

            if self.with_target_point_embed:
                self.target_point_encoder = nn.Sequential(
                    *linear_relu_ln(self.embed_dims, 2, 1),
                    Linear(self.embed_dims, self.embed_dims)
                )

            if self.with_custom_status_embed:
                self.custom_status_encoder = nn.Sequential(
                    *linear_relu_ln(self.embed_dims, 2, 1, input_dims=6),
                    Linear(self.embed_dims, self.embed_dims)
                )

            if self.with_concat_plan_points:
                self.suqueeze_plan_instance = nn.Sequential(
                    nn.Linear(embed_dims * 6, embed_dims * 6 // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dims * 6 // 2, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, embed_dims),
                )

        if 'motion' in self.task_select:
            self.motion_anchor = nn.Parameter(
                torch.tensor(np.load(motion_anchor), dtype=torch.float32), requires_grad=False
            )
            self.motion_anchor_encoder = nn.Sequential(
                *linear_relu_ln(self.embed_dims, 1, 1),
                Linear(self.embed_dims, self.embed_dims)
            )
            self.motion_refine = nn.ModuleList([build(motion_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
            self.motion_decoder = build(motion_decoder, BBOX_CODERS)
            self.loss_motion_cls = build(loss_motion_cls, LOSSES)
            self.loss_motion_reg = build(loss_motion_reg, LOSSES)

            self.fut_ts = self.motion_refine[0].fut_ts
            self.fut_mode = self.motion_refine[0].fut_mode

        if 'scenes' in self.query_select:
            self.scenes_instance_bank = build(scenes_instance_bank, PLUGIN_LAYERS)
            self.scenes_attention = nn.ModuleList([build(scenes_attention, ATTENTION) for _ in range(num_deform)])

        if 'scenes' in self.task_select:
            self.scenes_refine = nn.ModuleList([build(scenes_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.loss_scenes_reg = build(loss_scenes_reg, LOSSES)

        # operation
        self.op_config_map = {
            "concat": [custom_op, PLUGIN_LAYERS],
            "split": [custom_op, PLUGIN_LAYERS],
            "gnn": [graph_model, ATTENTION],
            "temp_gnn": [temp_graph_model, ATTENTION],
            "inter_gnn": [inter_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [custom_op, PLUGIN_LAYERS],
            "refine": [custom_op, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList([build(*self.op_config_map.get(op, [None, None])) for op in self.operation_order])

        self.fc_before = nn.Identity()
        self.fc_after = nn.Identity()

        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)

        if self.with_distance_attn_mask:
            self.distance_tau = nn.Linear(256, 8)

        if self.with_velocity_attn_mask:
            self.velocity_tau = nn.Linear(256, 8)

        self.run_step = 0
        self.attn_mask = None
        self.temp_attn_mask = None
        self.is_init_bank_list = False
        self.combine_layer_loss = combine_layer_loss


    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif 'refine' not in op:
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

        if self.with_distance_attn_mask:
            nn.init.zeros_(self.distance_tau.weight)
            nn.init.uniform_(self.distance_tau.bias, 0.0, 2.0)

        if self.with_velocity_attn_mask:
            nn.init.zeros_(self.velocity_tau.weight)
            nn.init.uniform_(self.velocity_tau.bias, 0.0, 2.0)

        self.init_instance_bank_list()

    def init_instance_bank_list(self):
        self.is_init_bank_list = True

        # open-loop training and evaluation
        if not self.with_close_loop:
            self.bank_length = self.open_loop_bank_length if self.open_loop_bank_length is not None else 1
            if 'det' in self.query_select:
                self.det_instance_bank_list = [self.det_instance_bank]
            if 'map' in self.query_select:
                self.map_instance_bank_list = [self.map_instance_bank]
            if 'ego' in self.query_select:
                self.ego_instance_bank_list = [self.ego_instance_bank]
            if 'plan' in self.query_select:
                self.plan_instance_bank_list = [self.plan_instance_bank]
            if "scenes" in self.query_select:
                self.scenes_instance_bank_list = [self.scenes_instance_bank]

        # close-loop evaluation
        else:
            self.bank_length = self.close_loop_bank_length if self.close_loop_bank_length is not None \
                else self.close_loop_hz // self.open_loop_hz
            if 'det' in self.task_select:
                self.det_instance_bank_list = [copy.deepcopy(self.det_instance_bank) for _ in range(self.bank_length)]
            if 'map' in self.task_select:
                self.map_instance_bank_list = [copy.deepcopy(self.map_instance_bank) for _ in range(self.bank_length)]
            if 'ego' in self.task_select:
                self.ego_instance_bank_list = [copy.deepcopy(self.ego_instance_bank) for _ in range(self.bank_length)]
            if 'plan' in self.task_select:
                self.plan_instance_bank_list = [copy.deepcopy(self.plan_instance_bank) for _ in range(self.bank_length)]
            if "scenes" in self.query_select:
                self.scenes_instance_bank_list = [copy.deepcopy(self.scenes_instance_bank) for _ in range(self.bank_length)]

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

    def get_motion_anchor(self, classification, prediction):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def graph_model(self, index, query, key=None, value=None,
                    query_pos=None, key_pos=None, independent=False, **kwargs):
        if independent:
            # index function
            if not isinstance(index, int):
                return index(query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs)
            return self.layers[index](query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs)

        else:
            if self.decouple_attn:
                query = torch.cat([query, query_pos], dim=-1)
                if key is not None:
                    key = torch.cat([key, key_pos], dim=-1)
                query_pos, key_pos = None, None

            if value is not None:
                value = self.fc_before(value)

            # index function
            if not isinstance(index, int):
                return self.fc_after(index(query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs))

            return self.fc_after(self.layers[index](query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs))

    def forward(self, img, feature_maps, metas):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        if not self.is_init_bank_list:
            self.init_instance_bank_list()

        deform_i = 0
        refine_i = 0

        batch_size = feature_maps[0].shape[0]
        bank_idx = self.run_step % self.bank_length

        # ========= initial query ============
        with_temp = False

        if "ego" in self.query_select:
            ego_instance_feature, ego_anchor, temp_ego_instance_feature, temp_ego_anchor = \
                self.ego_instance_bank_list[bank_idx].get(batch_size, metas, feature_maps)
            ego_anchor_embed = self.ego_anchor_encoder(ego_anchor)
            temp_ego_anchor_embed = self.ego_anchor_encoder(temp_ego_anchor) if temp_ego_anchor is not None else None
            self.num_ego_anchor = ego_anchor.size(1)
            self.num_temp_ego_anchor = temp_ego_anchor.size(1) if temp_ego_anchor is not None else 0
        else:
            ego_instance_feature, ego_anchor, ego_anchor_embed = None, None, None
            temp_ego_instance_feature, temp_ego_anchor, temp_ego_anchor_embed = None, None, None
            self.num_ego_anchor, self.num_temp_ego_anchor = 0, 0

        if "det" in self.query_select:
            det_instance_feature, det_anchor, temp_det_instance_feature, temp_det_anchor, time_interval = \
                self.det_instance_bank_list[bank_idx].get(batch_size, metas, dn_metas=self.det_sampler.dn_metas)
            det_anchor_embed = self.det_anchor_encoder(det_anchor)
            temp_det_anchor_embed = self.det_anchor_encoder(temp_det_anchor) if temp_det_anchor is not None else None
            self.num_det_anchor = det_anchor.size(1)
            self.num_temp_det_anchor = temp_det_anchor.size(1) if temp_det_anchor is not None else 0
        else:
            det_instance_feature, det_anchor, det_anchor_embed = None, None, None
            temp_det_instance_feature, temp_det_anchor, temp_det_anchor_embed = None, None, None
            self.num_det_anchor, self.num_temp_det_anchor = 0, 0

        if "map" in self.query_select:
            map_instance_feature, map_anchor, temp_map_instance_feature, temp_map_anchor, time_interval = \
                self.map_instance_bank_list[bank_idx].get(batch_size, metas, dn_metas=self.map_sampler.dn_metas)
            map_anchor_embed, map_points_embed = self.map_anchor_encoder(map_anchor)
            temp_map_anchor_embed, temp_map_points_embed = self.map_anchor_encoder(
                temp_map_anchor) if temp_map_anchor is not None else (None, None)
            self.num_map_anchor = map_anchor.size(1)
            self.num_map_points = map_points_embed.size(1) if map_points_embed is not None else 0
            self.num_temp_map_anchor = temp_map_anchor.size(1) if temp_map_anchor is not None else 0
            self.num_temp_map_points = temp_map_points_embed.size(1) if temp_map_points_embed is not None else 0
        else:
            map_instance_feature, map_anchor, map_anchor_embed, map_points_embed = None, None, None, None
            temp_map_instance_feature, temp_map_anchor, temp_map_anchor_embed, temp_map_points_embed = None, None, None, None
            self.num_map_anchor, self.num_map_points, self.num_temp_map_anchor, self.num_temp_map_points = 0, 0, 0, 0

        if "plan" in self.query_select:
            plan_instance_feature, plan_anchor, temp_plan_instance_feature, temp_plan_anchor = \
                self.plan_instance_bank_list[bank_idx].get(batch_size, metas, feature_maps)
            plan_anchor_embed, plan_points_embed = self.plan_anchor_encoder(plan_anchor)
            temp_plan_anchor_embed, temp_plan_points_embed = self.plan_anchor_encoder(
                temp_plan_anchor) if temp_plan_anchor is not None else (None, None)
            self.num_plan_anchor = plan_anchor.size(1)
            self.num_plan_points = plan_points_embed.size(1) if plan_points_embed is not None else 0
            self.num_temp_plan_anchor = temp_plan_anchor.size(1) if temp_plan_anchor is not None else 0
            self.num_temp_plan_points = temp_plan_points_embed.size(1) if temp_plan_points_embed is not None else 0
        else:
            plan_instance_feature, plan_anchor, plan_anchor_embed, plan_points_embed = None, None, None, None
            temp_plan_instance_feature, temp_plan_anchor, temp_plan_anchor_embed, temp_plan_points_embed = None, None, None, None
            self.num_plan_anchor, self.num_plan_points, self.num_temp_plan_anchor, self.num_temp_plan_points = 0, 0, 0, 0

        if "scenes" in self.query_select:
            scenes_instance_feature, scenes_anchor_embed, \
                fut_scenes_instance_feature, fut_scenes_anchor_embed, \
                temp_scenes_instance_feature, temp_scenes_anchor_embed = \
                self.scenes_instance_bank_list[bank_idx].get(img, feature_maps, metas)
            self.num_scenes_anchor = scenes_anchor_embed.size(1)
            self.num_temp_scenes_anchor = temp_scenes_anchor_embed.size(
                1) if temp_scenes_anchor_embed is not None else 0
        else:
            scenes_instance_feature, scenes_anchor_embed = None, None
            temp_scenes_instance_feature, temp_scenes_anchor_embed = None, None
            self.num_scenes_anchor, self.num_temp_scenes_anchor = 0, 0

        if (temp_ego_anchor is not None or
            temp_det_anchor is not None or
            temp_map_anchor is not None or
            temp_plan_anchor is not None or
            temp_scenes_anchor_embed is not None):
            with_temp = True

        self.num_anchor_list = []

        for q in self.query_select:
            self.num_anchor_list.append(getattr(self, "num_{}_anchor".format(q)))

        self.num_temp_anchor_list = []
        for q in self.query_select:
            self.num_temp_anchor_list.append(getattr(self, "num_temp_{}_anchor".format(q)))

        self.num_anchor_cumsum = np.cumsum([0] + self.num_anchor_list)
        self.num_temp_anchor_cumsum = np.cumsum([0] + self.num_temp_anchor_list)

        self.total_num_anchor = self.num_anchor_cumsum[-1]
        self.total_num_temp_anchor = self.num_temp_anchor_cumsum[-1]

        self.num_anchor_section = {q: [self.num_anchor_cumsum[i], self.num_anchor_cumsum[i + 1]]
                                   for i, q in enumerate(self.query_select)}
        self.num_temp_anchor_section = {q: [self.num_temp_anchor_cumsum[i], self.num_temp_anchor_cumsum[i + 1]]
                                        for i, q in enumerate(self.query_select)}

        if self.with_attn_mask:
            # init attn_mask
            if self.attn_mask is None:
                # specify interaction
                if self.attn_mask_dict is not None:
                    self.attn_mask = torch.ones((self.total_num_anchor, self.total_num_anchor),
                                                dtype=torch.float, device=img.device).fill_(float("-inf"))
                    for m1 in self.attn_mask_dict:
                        start1, end1 = self.num_anchor_section[m1]
                        for m2 in self.attn_mask_dict[m1]:
                            start2, end2 = self.num_anchor_section[m2]
                            self.attn_mask[start1:end1, start2:end2] = 0.0
                else:
                    self.attn_mask = torch.ones((self.total_num_anchor, self.total_num_anchor),
                                                dtype=torch.float, device=img.device).fill_(0)

            # init temp_attn_mask
            if with_temp and self.temp_attn_mask is None:
                # specify interaction
                if self.attn_mask_dict is not None:
                    self.temp_attn_mask = torch.ones((self.total_num_anchor, self.total_num_temp_anchor),
                                                     dtype=torch.float, device=img.device).fill_(float("-inf"))
                    for m1 in self.attn_mask_dict:
                        start1, end1 = self.num_anchor_section[m1]
                        for m2 in self.attn_mask_dict[m1]:
                            start2, end2 = self.num_temp_anchor_section[m2]
                            self.temp_attn_mask[start1:end1, start2:end2] = 0.0
                else:
                    self.temp_attn_mask = torch.ones((self.total_num_anchor, self.total_num_temp_anchor),
                                                     dtype=torch.float, device=img.device).fill_(0)

        # ========= training ============
        attn_mask = self.attn_mask
        temp_attn_mask = self.temp_attn_mask

        # =================== forward the layers ====================
        det_prediction = []
        det_classification = []
        det_quality = []

        map_prediction = []
        map_classification = []
        map_quality = []

        ego_prediction = []
        ego_classification = []
        ego_status = []

        plan_prediction = []
        plan_classification = []
        plan_status = []

        motion_prediction = []
        motion_classification = []

        scenes_latent_tokens = []
        scenes_latent_embeds = []
        scenes_future_tokens = []
        scenes_future_embeds = []

        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue

            elif op == "concat":
                anchor_embed_list = []
                instance_feature_list = []
                temp_anchor_embed_list = []
                temp_instance_feature_list = []

                instance_num_list = []
                temp_instance_num_list = []

                for modality in self.query_select:
                    if modality == "ego":
                        anchor_embed_list.append(ego_anchor_embed)
                        instance_feature_list.append(ego_instance_feature)
                        instance_num_list.append(ego_instance_feature.size(1))
                        if with_temp and temp_ego_instance_feature is not None:
                            temp_anchor_embed_list.append(temp_ego_anchor_embed)
                            temp_instance_feature_list.append(temp_ego_instance_feature)
                            temp_instance_num_list.append(temp_ego_instance_feature.size(1))
                        else:
                            temp_instance_num_list.append(0)
                    elif modality == "det":
                        anchor_embed_list.append(det_anchor_embed)
                        instance_feature_list.append(det_instance_feature)
                        instance_num_list.append(det_instance_feature.size(1))
                        if with_temp and temp_det_instance_feature is not None:
                            temp_anchor_embed_list.append(temp_det_anchor_embed)
                            temp_instance_feature_list.append(temp_det_instance_feature)
                            temp_instance_num_list.append(temp_det_instance_feature.size(1))
                        else:
                            temp_instance_num_list.append(0)
                    elif modality == "map":
                        if self.with_concat_map_points:
                            if map_instance_feature.size(1) != self.num_map_anchor * 20:
                                map_instance_feature = map_instance_feature.repeat(1, 20, 1)
                            anchor_embed_list.append(map_points_embed)
                            instance_feature_list.append(map_instance_feature)
                            instance_num_list.append(map_instance_feature.size(1))
                            if with_temp and temp_map_instance_feature is not None:
                                if temp_map_instance_feature.size(1) != self.num_temp_map_anchor * 20:
                                    temp_map_instance_feature = temp_map_instance_feature.repeat(1, 20, 1)
                                temp_anchor_embed_list.append(temp_map_points_embed)
                                temp_instance_feature_list.append(temp_map_instance_feature)
                                temp_instance_num_list.append(temp_map_instance_feature.size(1))
                            else:
                                temp_instance_num_list.append(0)
                        else:
                            anchor_embed_list.append(map_anchor_embed)
                            instance_feature_list.append(map_instance_feature)
                            instance_num_list.append(map_instance_feature.size(1))
                            if with_temp and temp_map_instance_feature is not None:
                                temp_anchor_embed_list.append(temp_map_anchor_embed)
                                temp_instance_feature_list.append(temp_map_instance_feature)
                                temp_instance_num_list.append(temp_map_instance_feature.size(1))
                            else:
                                temp_instance_num_list.append(0)
                    elif modality == "plan":
                        if self.with_concat_plan_points:
                            if plan_instance_feature.size(1) != self.num_plan_anchor * 6:
                                plan_instance_feature = plan_instance_feature.repeat(1, 6, 1)
                            anchor_embed_list.append(plan_points_embed)
                            instance_feature_list.append(plan_instance_feature)
                            instance_num_list.append(plan_instance_feature.size(1))
                            if with_temp and temp_plan_instance_feature is not None:
                                if temp_plan_instance_feature.size(1) != self.num_temp_plan_anchor * 6:
                                    temp_plan_instance_feature = temp_plan_instance_feature.repeat(1, 6, 1)
                                temp_anchor_embed_list.append(temp_plan_points_embed)
                                temp_instance_feature_list.append(temp_plan_instance_feature)
                                temp_instance_num_list.append(temp_plan_instance_feature.size(1))
                            else:
                                temp_instance_num_list.append(0)
                        else:
                            anchor_embed_list.append(plan_anchor_embed)
                            instance_feature_list.append(plan_instance_feature)
                            instance_num_list.append(plan_instance_feature.size(1))
                            if with_temp and temp_plan_instance_feature is not None:
                                temp_anchor_embed_list.append(temp_plan_anchor_embed)
                                temp_instance_feature_list.append(temp_plan_instance_feature)
                                temp_instance_num_list.append(temp_plan_instance_feature.size(1))
                            else:
                                temp_instance_num_list.append(0)
                    elif modality == "scenes":
                        anchor_embed_list.append(scenes_anchor_embed)
                        instance_feature_list.append(scenes_instance_feature)
                        instance_num_list.append(scenes_instance_feature.size(1))
                        if with_temp and temp_scenes_instance_feature is not None:
                            temp_anchor_embed_list.append(temp_scenes_anchor_embed)
                            temp_instance_feature_list.append(temp_scenes_instance_feature)
                            temp_instance_num_list.append(temp_scenes_instance_feature.size(1))
                        else:
                            temp_instance_num_list.append(0)

                anchor_embed = torch.cat(anchor_embed_list, dim=1)
                instance_feature = torch.cat(instance_feature_list, dim=1)

                if with_temp:
                    temp_anchor_embed = torch.cat(temp_anchor_embed_list, dim=1)
                    temp_instance_feature = torch.cat(temp_instance_feature_list, dim=1)
                else:
                    temp_anchor_embed = None
                    temp_instance_feature = None

            elif op == "split":
                instance_num_cumsum = np.cumsum([0] + instance_num_list)

                for modality, query_start, query_end in zip(
                        self.query_select, instance_num_cumsum[:-1], instance_num_cumsum[1:]):
                    if modality == "ego":
                        ego_anchor_embed = anchor_embed[:, query_start: query_end]
                        ego_instance_feature = instance_feature[:, query_start: query_end]
                    elif modality == "det":
                        det_anchor_embed = anchor_embed[:, query_start: query_end]
                        det_instance_feature = instance_feature[:, query_start: query_end]
                    elif modality == "map":
                        if self.with_concat_map_points:
                            map_points_embed = anchor_embed[:, query_start: query_end]
                            map_instance_feature = instance_feature[:, query_start: query_end]
                            map_instance_feature = map_instance_feature.view(batch_size, -1, 20, self.embed_dims)
                            map_instance_feature = self.suqueeze_map_instance(map_instance_feature.flatten(2))
                        else:
                            map_anchor_embed = anchor_embed[:, query_start: query_end]
                            map_instance_feature = instance_feature[:, query_start: query_end]
                    elif modality == "plan":
                        if self.with_concat_plan_points:
                            plan_points_embed = anchor_embed[:, query_start: query_end]
                            plan_instance_feature = instance_feature[:, query_start: query_end]
                            plan_instance_feature = plan_instance_feature.view(batch_size, -1, 6, self.embed_dims)
                            plan_instance_feature = self.suqueeze_plan_instance(plan_instance_feature.flatten(2))
                        else:
                            plan_anchor_embed = anchor_embed[:, query_start: query_end]
                            plan_instance_feature = instance_feature[:, query_start: query_end]
                    elif modality == "scenes":
                        scenes_anchor_embed = anchor_embed[:, query_start: query_end]
                        scenes_instance_feature = instance_feature[:, query_start: query_end]

                if with_temp:
                    temp_instance_num_cumsum = np.cumsum([0] + temp_instance_num_list)
                    for modality, query_start, query_end in zip(
                            self.query_select, temp_instance_num_cumsum[:-1], temp_instance_num_cumsum[1:]):
                        if modality == "ego":
                            temp_ego_anchor_embed = temp_anchor_embed[:, query_start: query_end]
                            temp_ego_instance_feature = temp_instance_feature[:, query_start: query_end]
                        elif modality == "det":
                            temp_det_anchor_embed = temp_anchor_embed[:, query_start: query_end]
                            temp_det_instance_feature = temp_instance_feature[:, query_start: query_end]
                        elif modality == "map":
                            if self.with_concat_map_points:
                                temp_map_points_embed = temp_anchor_embed[:, query_start: query_end]
                                temp_map_instance_feature = temp_instance_feature[:, query_start: query_end]
                                temp_map_instance_feature = temp_map_instance_feature.view(batch_size, -1, 20,
                                                                                           self.embed_dims)
                                temp_map_instance_feature = self.suqueeze_map_instance(
                                    temp_map_instance_feature.flatten(2))
                            else:
                                temp_map_anchor_embed = temp_anchor_embed[:, query_start: query_end]
                                temp_map_instance_feature = temp_instance_feature[:, query_start: query_end]
                        elif modality == "plan":
                            if self.with_concat_plan_points:
                                temp_plan_points_embed = temp_anchor_embed[:, query_start: query_end]
                                temp_plan_instance_feature = temp_instance_feature[:, query_start: query_end]
                                temp_plan_instance_feature = temp_plan_instance_feature.view(batch_size, -1, 6,
                                                                                             self.embed_dims)
                                temp_plan_instance_feature = self.suqueeze_plan_instance(
                                    temp_plan_instance_feature.flatten(2))
                            else:
                                temp_plan_anchor_embed = temp_anchor_embed[:, query_start: query_end]
                                temp_plan_instance_feature = temp_instance_feature[:, query_start: query_end]
                        elif modality == "scenes":
                            temp_scenes_anchor_embed = temp_anchor_embed[:, query_start: query_end]
                            temp_scenes_instance_feature = temp_instance_feature[:, query_start: query_end]

            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=temp_attn_mask if with_temp else attn_mask,
                    independent=self.independent_temp_gnn,
                    num_anchor_cumsum=self.num_anchor_cumsum,
                    num_temp_anchor_cumsum=self.num_temp_anchor_cumsum,
                    fc_before=self.fc_before,
                    fc_after=self.fc_after,
                )

            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                    independent=self.independent_gnn,
                    num_anchor_cumsum=self.num_anchor_cumsum,
                    fc_before=self.fc_before,
                    fc_after=self.fc_after,
                )

            elif op == "inter_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                    independent=self.independent_inter_gnn,
                    num_anchor_cumsum=self.num_anchor_cumsum,
                    map_points_embed=map_points_embed if map_points_embed is not None else None,
                    plan_points_embed=plan_points_embed if plan_points_embed is not None else None,
                    det_anchor=det_anchor,
                    map_anchor=map_anchor,
                    plan_anchor=plan_anchor,
                    distance_tau=self.distance_tau if self.with_distance_attn_mask else None,
                    velocity_tau=self.velocity_tau if self.with_velocity_attn_mask else None,
                    fc_before=self.fc_before,
                    fc_after=self.fc_after,
                )

            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)

            elif op == "deformable":
                for modality in self.query_select:
                    if modality == "ego":
                        ego_instance_feature = self.ego_deformable[deform_i](
                            ego_instance_feature, ego_anchor, ego_anchor_embed, feature_maps, metas)
                    elif modality == "det":
                        det_instance_feature = self.det_deformable[deform_i](
                            det_instance_feature, det_anchor, det_anchor_embed, feature_maps, metas)
                    elif modality == "map":
                        map_embeddings = map_points_embed if self.with_deform_map_points else map_anchor_embed
                        map_instance_feature = self.map_deformable[deform_i](
                            map_instance_feature, map_anchor, map_embeddings, feature_maps, metas)
                    elif modality == "plan":
                        plan_embeddings = plan_points_embed if self.with_deform_plan_points else plan_anchor_embed
                        plan_instance_feature = self.plan_deformable[deform_i](
                            plan_instance_feature, plan_anchor, plan_embeddings, feature_maps, metas)
                    elif modality == "scenes":
                        scenes_instance_feature = self.scenes_attention[deform_i](
                            scenes_instance_feature, scenes_instance_feature + scenes_anchor_embed)

                deform_i += 1

            elif op == "refine":
                if "det" in self.task_select:
                    det_anchor, det_cls, det_qt = self.det_refine[refine_i](
                        det_instance_feature, det_anchor, det_anchor_embed, time_interval=time_interval,
                        return_cls=True)
                    det_prediction.append(det_anchor)
                    det_classification.append(det_cls)
                    det_quality.append(det_qt)

                    if len(det_prediction) == self.num_single_frame_decoder:
                        det_instance_feature, det_anchor = self.det_instance_bank_list[bank_idx].update(
                            det_instance_feature, det_anchor, det_cls)

                    det_anchor_embed = self.det_anchor_encoder(det_anchor)

                    if len(det_prediction) > self.num_single_frame_decoder and temp_det_anchor_embed is not None:
                        temp_det_anchor_embed = det_anchor_embed[:,
                                                : self.det_instance_bank_list[bank_idx].num_temp_instances]

                if "map" in self.task_select:
                    map_anchor, map_cls, map_qt = self.map_refine[refine_i](
                        map_instance_feature, map_anchor, map_anchor_embed, time_interval=time_interval,
                        return_cls=True)
                    map_prediction.append(map_anchor)
                    map_classification.append(map_cls)
                    map_quality.append(map_qt)

                    if len(map_prediction) == self.num_single_frame_decoder:
                        map_instance_feature, map_anchor = self.map_instance_bank_list[bank_idx].update(
                            map_instance_feature, map_anchor, map_cls)

                    map_anchor_embed, map_points_embed = self.map_anchor_encoder(map_anchor)

                    if len(map_prediction) > self.num_single_frame_decoder and temp_map_anchor_embed is not None:
                        temp_map_anchor_embed = map_anchor_embed[:, : self.map_instance_bank_list[bank_idx].num_temp_instances]
                        temp_map_points_embed = map_points_embed[:, : self.map_instance_bank_list[bank_idx].num_temp_instances * 20] if map_points_embed is not None else None

                if "motion" in self.task_select:
                    motion_anchor = self.get_motion_anchor(det_cls, det_anchor)
                    motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
                    motion_query = motion_mode_query + (det_instance_feature + det_anchor_embed).unsqueeze(2)
                    motion_cls, motion_reg = self.motion_refine[refine_i](motion_query)

                    motion_classification.append(motion_cls)
                    motion_prediction.append(motion_reg)

                if "ego" in self.task_select:
                    # only predict ego status
                    if self.with_supervise_ego_status:
                        ego_reg, ego_cls, ego_status_ = None, None, self.ego_refine[refine_i](ego_instance_feature,
                                                                                              ego_anchor_embed)
                    else:
                        plan_anchor = torch.tile(self.ego_instance_bank_list[bank_idx].plan_anchor[None],
                                                 (batch_size, 1, 1, 1, 1))
                        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
                        plan_mode_query = self.ego_instance_bank_list[bank_idx].plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)
                        plan_query = plan_mode_query + (ego_instance_feature + ego_anchor_embed).unsqueeze(2)

                        ego_cls, ego_reg, ego_status_ = self.ego_refine[refine_i](plan_query, ego_instance_feature, ego_anchor_embed)

                    ego_classification.append(ego_cls)
                    ego_prediction.append(ego_reg)
                    ego_status.append(ego_status_)

                if "plan" in self.task_select:
                    use_plan_anchor_embed = True
                    if self.with_target_point_embed:
                        target_point = metas['target_point'].unsqueeze(1).unsqueeze(1)
                        target_point_embed = self.target_point_encoder(gen_sineembed_for_position(target_point))
                        plan_anchor_embed += target_point_embed.squeeze(1)

                    if self.with_target_point_next_embed:
                        target_point_next = metas['target_point_next'].unsqueeze(1).unsqueeze(1)
                        target_point_next_embed = self.target_point_encoder(gen_sineembed_for_position(target_point_next))
                        plan_anchor_embed += target_point_next_embed.squeeze(1)

                    if self.with_command_embed:
                        ego_command = metas['gt_ego_fut_cmd'].unsqueeze(1).unsqueeze(1)
                        command_embed = self.command_embed_encoder(ego_command)
                        plan_anchor_embed += command_embed.squeeze(1)

                    if self.with_custom_status_embed:
                        custom_status = metas['custom_status'].unsqueeze(1).unsqueeze(1)
                        custom_status_embed = self.custom_status_encoder(custom_status)
                        plan_anchor_embed += custom_status_embed.squeeze(1)

                    if self.with_ego_instance_feature:
                        plan_instance_feature = plan_instance_feature + ego_instance_feature
                        plan_anchor_embed = plan_anchor_embed + ego_anchor_embed

                    plan_reg, plan_cls = self.plan_refine[refine_i](
                        plan_instance_feature, plan_anchor, plan_anchor_embed, use_plan_anchor_embed)

                    if self.with_topk_mode:
                        ng = self.plan_anchor_group
                        topk = self.topk_mode_list[refine_i]

                        plan_reg = plan_reg.reshape(batch_size, ng, -1, self.ego_fut_ts * 2)
                        plan_cls = plan_cls.reshape(batch_size, ng, -1, 1)
                        plan_instance_feature = plan_instance_feature.reshape(batch_size, ng, -1, self.embed_dims)

                        # all groups of pred_cls share the same ref score
                        plan_cls, topk_indices = plan_cls.topk(topk, dim=2)
                        if self.keep_topk_relative_pos:
                            topk_indices, topk_sort_indices = topk_indices.sort(2)
                            plan_cls = torch.gather(plan_cls, dim=2, index=topk_sort_indices)

                        reg_topk_indices = topk_indices.repeat(1, 1, 1, self.ego_fut_ts * 2)
                        feat_topk_indices = topk_indices.repeat(1, 1, 1, self.embed_dims)

                        plan_reg = torch.gather(plan_reg, dim=2, index=reg_topk_indices)
                        plan_instance_feature = torch.gather(plan_instance_feature, dim=2, index=feat_topk_indices)

                        plan_reg = plan_reg.reshape(batch_size, -1, self.ego_fut_ts * 2)
                        plan_cls = plan_cls.reshape(batch_size, -1, 1)
                        plan_instance_feature = plan_instance_feature.reshape(batch_size, -1, self.embed_dims)

                        self.num_anchor_list[self.query_select.index("plan")] = topk * ng
                        self.num_anchor_cumsum = np.cumsum([0] + self.num_anchor_list)

                    plan_anchor = plan_reg.clone()
                    bs, nj, _ = plan_reg.shape

                    plan_reg = plan_reg.clone().reshape(bs, 1, nj, self.ego_fut_ts, 2)
                    plan_reg[..., 1:, :] = plan_reg[..., 1:, :] - plan_reg[..., :-1, :]

                    plan_prediction.append(plan_reg)
                    plan_classification.append(plan_cls.reshape(bs, 1, -1))
                    plan_status.append(None)

                    plan_anchor_embed, plan_points_embed = self.plan_anchor_encoder(plan_anchor)

                if "scenes" in self.task_select:
                    scenes_instance_feature = self.scenes_refine[refine_i](scenes_instance_feature, scenes_anchor_embed,
                                                                           metas)
                    scenes_latent_tokens.append(scenes_instance_feature)
                    scenes_future_tokens.append(fut_scenes_instance_feature)

                refine_i += 1

            else:
                raise NotImplementedError(f"{op} is not supported.")

        det_output = {
            "classification": det_classification,
            "prediction": det_prediction,
            "quality": det_quality,
            "instance_feature": det_instance_feature,
            "anchor_embed": det_anchor_embed,
        }
        map_output = {
            "classification": map_classification,
            "prediction": map_prediction,
            "quality": map_quality,
            "instance_feature": map_instance_feature,
            "anchor_embed": map_anchor_embed,
        }
        ego_output = {
            "classification": ego_classification,
            "prediction": ego_prediction,
            "status": ego_status
        }
        plan_output = {
            "classification": plan_classification,
            "prediction": plan_prediction,
            "status": plan_status,
        }
        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
        }
        scenes_output = {
            "scenes_latent_tokens": scenes_latent_tokens,
            "scenes_latent_embeds": scenes_latent_embeds,
            "scenes_future_tokens": scenes_future_tokens,
            "scenes_future_embeds": scenes_future_embeds,
        }

        # cache current instances for temporal modeling
        for modality in self.query_select:
            if modality == "ego":
                self.ego_instance_bank_list[bank_idx].cache(
                    ego_instance_feature, ego_anchor, metas, feature_maps)
            if modality == "det":
                self.det_instance_bank_list[bank_idx].cache(
                    det_instance_feature, det_anchor, det_cls, metas, feature_maps)
            if modality == "map":
                self.map_instance_bank_list[bank_idx].cache(
                    map_instance_feature, map_anchor, map_cls, metas, feature_maps)
            if modality == "plan":
                self.plan_instance_bank_list[bank_idx].cache(
                    plan_instance_feature, plan_anchor, plan_cls, metas, feature_maps)
            if modality == "scenes":
                self.scenes_instance_bank_list[bank_idx].cache(
                    scenes_instance_feature, scenes_anchor_embed, metas, feature_maps)

        if self.with_instance_id and "det" in self.task_select:
            det_instance_id = self.det_instance_bank_list[bank_idx].get_instance_id(
                det_cls, det_anchor, self.det_decoder.score_threshold)
            det_output["instance_id"] = det_instance_id

        self.run_step += 1

        return det_output, map_output, ego_output, plan_output, motion_output, scenes_output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, det_output, map_output, ego_output, plan_output, motion_output, scenes_output, data):
        losses = dict()

        if "det" in self.task_select:
            loss_det = self.loss_det(det_output, data)
            losses.update(loss_det)
        if "map" in self.task_select:
            loss_map = self.loss_map(map_output, data)
            losses.update(loss_map)
        if "ego" in self.task_select:
            loss_ego = self.loss_ego(ego_output, data)
            losses.update(loss_ego)
        if "motion" in self.task_select:
            loss_motion = self.loss_motion(motion_output, data)
            losses.update(loss_motion)
        if "scenes" in self.task_select:
            loss_scenes = self.loss_scenes(scenes_output, data)
            losses.update(loss_scenes)
        if "plan" in self.task_select:
            loss_plan = self.loss_plan(det_output, map_output, motion_output, plan_output, data)
            losses.update(loss_plan)

        return losses

    @force_fp32(apply_to=("model_outs"))
    def loss_det(self, model_outs, data):
        quality = model_outs["quality"]
        reg_preds = model_outs["prediction"]
        cls_scores = model_outs["classification"]

        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)):
            reg = reg[..., : len(self.det_reg_weights)]
            cls_target, reg_target, reg_weights = self.det_sampler.sample(
                cls, reg, data["gt_labels_3d"], data["gt_bboxes_3d"])
            reg_target = reg_target[..., : len(self.det_reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_det_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.det_reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(reg_target.isnan(), reg.new_tensor(0.0), reg_target)
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_det_reg(reg, reg_target, weight=reg_weights, avg_factor=num_pos,
                                         prefix="det_", suffix=f"_{decoder_idx}", quality=qt, cls_target=cls_target)

            if self.combine_layer_loss:
                if "det_loss_cls" not in output:
                    output["det_loss_cls"] = 0.0
                if "det_loss_box" not in output:
                    output["det_loss_box"] = 0.0
                    output["det_loss_cns"] = 0.0
                    output["det_loss_yns"] = 0.0
                output["det_loss_cls"] += cls_loss
                output["det_loss_box"] += reg_loss[f"det_loss_box_{decoder_idx}"]
                output["det_loss_cns"] += reg_loss[f"det_loss_cns_{decoder_idx}"]
                output["det_loss_yns"] += reg_loss[f"det_loss_yns_{decoder_idx}"]
            else:
                output[f"det_loss_cls_{decoder_idx}"] = cls_loss
                output.update(reg_loss)

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_map(self, model_outs, data):
        quality = model_outs["quality"]
        reg_preds = model_outs["prediction"]
        cls_scores = model_outs["classification"]

        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)):
            reg = reg[..., : len(self.map_reg_weights)]
            cls_target, reg_target, reg_weights = self.map_sampler.sample(
                cls, reg, data["gt_map_labels"], data["gt_map_pts"])

            reg_target = reg_target[..., : len(self.map_reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_map_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.map_reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(reg_target.isnan(), reg.new_tensor(0.0), reg_target)
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_map_reg(reg, reg_target, weight=reg_weights, avg_factor=num_pos,
                                         prefix="map_", suffix=f"_{decoder_idx}", quality=qt, cls_target=cls_target)

            if self.combine_layer_loss:
                if "map_loss_cls" not in output:
                    output["map_loss_cls"] = 0.0
                if "map_loss_line" not in output:
                    output["map_loss_line"] = 0.0
                output["map_loss_cls"] += cls_loss
                output["map_loss_line"] += reg_loss[f"map_loss_line_{decoder_idx}"]
            else:
                output[f"map_loss_cls_{decoder_idx}"] = cls_loss
                output.update(reg_loss)

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_ego(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(zip(cls_scores, reg_preds, status_preds)):
            # only predict ego status
            if self.with_supervise_ego_status:
                gt_ego_status = data['ego_status']
                ego_status_weight = data['ego_status_mask']
                status_loss = self.loss_ego_status(status.squeeze(1), gt_ego_status, weight=ego_status_weight)
                status_loss = torch.nan_to_num(status_loss)

                if self.combine_layer_loss:
                    if "ego_loss_status" not in output:
                        output["ego_loss_status"] = 0.0
                    output["ego_loss_status"] += status_loss
                else:
                    output[f"ego_loss_status_{decoder_idx}"] = status_loss

            else:
                cls, cls_target, cls_weight, reg_pred, reg_target, reg_weight = self.ego_sampler.sample(
                    cls, reg, data['gt_ego_fut_trajs'], data['gt_ego_fut_masks'], data)

                cls = cls.flatten(end_dim=1)
                cls_target = cls_target.flatten(end_dim=1)
                cls_weight = cls_weight.flatten(end_dim=1)
                cls_loss = self.loss_ego_cls(cls, cls_target, weight=cls_weight)

                reg_weight = reg_weight.flatten(end_dim=1)
                reg_pred = reg_pred.flatten(end_dim=1)
                reg_target = reg_target.flatten(end_dim=1)
                reg_weight = reg_weight.unsqueeze(-1)
                reg_loss = self.loss_ego_reg(reg_pred, reg_target, weight=reg_weight)

                output.update(
                    {
                        f"ego_loss_cls_{decoder_idx}": cls_loss,
                        f"ego_loss_reg_{decoder_idx}": reg_loss,
                    }
                )

                if self.loss_ego_status is not None:
                    status_loss = self.loss_ego_status(status.squeeze(1), data['ego_status'])
                    status_loss = torch.nan_to_num(status_loss)

                    output.update(
                        {
                            f"ego_loss_status_{decoder_idx}": status_loss
                        }
                    )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data):
        reg_preds = model_outs["prediction"]
        cls_scores = model_outs["classification"]

        output = {}
        motion_loss_cache = dict(indices=self.det_sampler.indices)
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            cls_target, cls_weight, reg_pred, reg_target, reg_weight, num_pos = self.motion_sampler.sample(
                reg, data["gt_agent_fut_trajs"], data["gt_agent_fut_masks"], motion_loss_cache)

            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.loss_motion_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.loss_motion_reg(reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos)

            if self.combine_layer_loss:
                if "motion_loss_cls" not in output:
                    output["motion_loss_cls"] = 0.0
                if "motion_loss_reg" not in output:
                    output["motion_loss_reg"] = 0.0
                output["motion_loss_cls"] += cls_loss
                output["motion_loss_reg"] += reg_loss
            else:
                output[f"motion_loss_cls_{decoder_idx}"] = cls_loss
                output[f"motion_loss_reg_{decoder_idx}"] = reg_loss

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_scenes(self, model_outs, data):
        output = dict()
        scenes_mask = data['fut_mask']
        scenes_latent_tokens = model_outs['scenes_latent_tokens']
        scenes_future_tokens = model_outs['scenes_future_tokens']
        for decoder_idx, (pred_scenes_token, target_scenes_token) in enumerate(
                zip(scenes_latent_tokens, scenes_future_tokens)):
            bs, num_tokens, embed_dim = pred_scenes_token.shape
            pred_scenes_token = pred_scenes_token.flatten(0, 1)
            target_scenes_token = target_scenes_token.flatten(0, 1)
            scenes_weight = scenes_mask[:, None, None].repeat(1, num_tokens, embed_dim).flatten(0, 1)
            loss_scenes_reg = self.loss_scenes_reg(pred_scenes_token, target_scenes_token, scenes_weight)

            if self.combine_layer_loss:
                if "scenes_loss_reg" not in output:
                    output["scenes_loss_reg"] = 0.0
                output["scenes_loss_reg"] += loss_scenes_reg

            else:
                output[f"scenes_loss_reg_{decoder_idx}"] = loss_scenes_reg

        return output

    def align_plan_traj_loss(self,
                             ref_pred_cls, ref_pred_reg,
                             ref_gt_trajs, ref_gt_masks,
                             pred_cls, pred_reg,
                             gt_trajs, gt_masks, data):

        _, ref_target, ref_cls_weight, _, _, _ = self.plan_sampler.sample(
            ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, data)

        cls, _, _, reg_pred, reg_target, reg_weight = self.align_sampler.sample(
            pred_cls, pred_reg, gt_trajs, gt_masks, data, ref_target)

        cls = cls.flatten(end_dim=1)
        ref_target = ref_target.flatten(end_dim=1)
        ref_cls_weight = ref_cls_weight.flatten(end_dim=1)
        cls_loss = self.loss_plan_cls(cls, ref_target, weight=ref_cls_weight)

        reg_weight = reg_weight.flatten(end_dim=1)
        reg_pred = reg_pred.cumsum(dim=-2)
        reg_target = reg_target.cumsum(dim=-2)
        reg_pred = reg_pred.flatten(end_dim=1)
        reg_target = reg_target.flatten(end_dim=1)
        reg_weight = reg_weight.unsqueeze(-1)
        reg_loss = self.loss_plan_reg(reg_pred, reg_target, weight=reg_weight)

        return cls_loss, reg_loss

    def align_speed_traj_loss(self,
                              ref_pred_cls, ref_pred_reg,
                              ref_gt_trajs, ref_gt_masks,
                              ref_speed_trajs, ref_speed_masks,
                              pred_cls_list, pred_reg_list,
                              gt_trajs, gt_masks, speed_areas, data):

        _, ref_target, _, _, _, _ = self.plan_sampler.sample(
            ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, data)

        aligned_cls_list = []
        aligned_reg_list = []
        bs_indices = torch.arange(ref_target.shape[0], device=ref_target.device)
        for pred_cls, pred_reg in zip(pred_cls_list, pred_reg_list):
            aligned_cls, _, _, aligned_reg, _, _ = self.align_sampler.sample(
                pred_cls, pred_reg, gt_trajs, gt_masks, data, ref_target)
            aligned_cls = aligned_cls.squeeze(1)[bs_indices, ref_target.squeeze(-1)][:, None, None]
            aligned_reg = aligned_reg[:, :, None]
            aligned_cls_list.append(aligned_cls)
            aligned_reg_list.append(aligned_reg)

        aligned_cls = torch.cat(aligned_cls_list, dim=-1)
        aligned_reg = torch.cat(aligned_reg_list, dim=-3)

        cls, cls_target, cls_weight, reg_pred, reg_target, reg_weight = self.speed_sample(
            ref_speed_trajs, ref_speed_masks, aligned_cls, aligned_reg, gt_trajs, gt_masks, speed_areas, data)

        cls = cls.flatten(end_dim=1)
        cls_target = cls_target.flatten(end_dim=1)
        cls_weight = cls_weight.flatten(end_dim=1)
        cls_loss = self.loss_plan_cls(cls, cls_target, weight=cls_weight)

        reg_weight = reg_weight.flatten(end_dim=1)
        reg_pred = reg_pred.cumsum(dim=-2)
        reg_target = reg_target.cumsum(dim=-2)
        reg_pred = reg_pred.flatten(end_dim=1)
        reg_target = reg_target.flatten(end_dim=1)
        reg_weight = reg_weight.unsqueeze(-1)
        reg_loss = self.loss_plan_reg(reg_pred, reg_target, weight=reg_weight)

        return cls_loss, reg_loss

    def speed_sample(self, ref_speed_trajs, ref_speed_masks, cls_pred, reg_pred, gt_trajs, gt_masks, speed_areas, data):
        ego_fut_mode = cls_pred.shape[2]
        gt_trajs = gt_trajs.unsqueeze(1)
        gt_masks = gt_masks.unsqueeze(1)

        ref_speed_trajs = ref_speed_trajs.unsqueeze(1)
        ref_speed_masks = ref_speed_masks.unsqueeze(1)

        bs, _, num_traj = cls_pred.shape
        bs_indices = torch.arange(bs, device=reg_pred.device)

        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1) if self.ego_fut_cmd > 1 else 0

        cls_pred = cls_pred.reshape(bs, self.ego_fut_cmd, 1, ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, self.ego_fut_cmd, 1, ego_fut_mode, self.ego_fut_ts, 2)

        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]

        target_dist = torch.linalg.norm(ref_speed_trajs, dim=-1)
        target_dist_sum = target_dist.sum(-1)
        gt_masks_sum = ref_speed_masks.sum(-1)
        refer_interval = 1 / float(self.plan_speed_refer[1].split('hz')[0])
        gt_speed = target_dist_sum / (gt_masks_sum * refer_interval + 1e-4)

        mode_idx = torch.ones_like(gt_speed, dtype=torch.long)
        for speed_idx, (start, end) in enumerate(speed_areas):
            speed_mask = torch.logical_and(gt_speed >= start, gt_speed < end)
            mode_idx[speed_mask] = speed_idx

        cls_target = mode_idx
        cls_weight = ref_speed_masks.any(dim=-1)
        tmp_mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, self.ego_fut_ts, 2)
        best_reg = torch.gather(reg_pred, 2, tmp_mode_idx).squeeze(2)

        return cls_pred, cls_target, cls_weight, best_reg, gt_trajs, gt_masks

    def get_pred_trajs(self, cls, reg, type):
        num_group = self.plan_anchor_group
        num_mode = reg.size(2) // num_group

        pred_index = self.plan_anchor_types.index(type)
        pred_start = self.ego_fut_cmd * num_mode * pred_index
        pred_end = self.ego_fut_cmd * num_mode * (pred_index + 1)
        return cls[:, :, pred_start:pred_end], reg[:, :, pred_start:pred_end]

    def get_gt_trajs(self, data, type):
        # temporal and speed waypoints
        if type[0] == "temp" or type[0] == "speed":
            gt_trajs = data['gt_ego_fut_trajs_{}'.format(type[1])]
            gt_masks = data['gt_ego_fut_masks_{}'.format(type[1])]
        # spatial waypoints
        elif type[0] == "spat":
            gt_trajs = data['gt_ego_spat_trajs_{}'.format(type[1])]
            gt_masks = data['gt_ego_spat_masks_{}'.format(type[1])]
        else:
            raise NotImplementedError

        return gt_trajs, gt_masks

    @force_fp32(apply_to=("plan_output"))
    def loss_plan(self,
                  det_output,
                  map_output,
                  motion_output,
                  plan_output,
                  data):
        reg_preds = plan_output["prediction"]
        cls_scores = plan_output["classification"]

        output = {}
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):

            temp_cls_loss = 0.0
            temp_reg_loss = 0.0
            spat_cls_loss = 0.0
            spat_reg_loss = 0.0
            speed_cls_loss = 0.0
            speed_reg_loss = 0.0

            # select reference anchor
            ref_pred_cls, ref_pred_reg = self.get_pred_trajs(cls, reg, self.plan_anchor_refer)
            ref_gt_trajs, ref_gt_masks = self.get_gt_trajs(data, self.plan_anchor_refer)

            speed_dict = dict()
            for anchor_type in self.plan_anchor_types:
                pred_cls, pred_reg = self.get_pred_trajs(cls, reg, anchor_type)
                gt_trajs, gt_masks = self.get_gt_trajs(data, anchor_type)

                if anchor_type[0] in ["temp", "spat"]:
                    cls_loss, reg_loss = self.align_plan_traj_loss(
                        ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, pred_cls, pred_reg, gt_trajs, gt_masks, data)

                    if anchor_type[0] == "temp":
                        temp_cls_loss += cls_loss
                        temp_reg_loss += reg_loss
                    elif anchor_type[0] == "spat":
                        spat_cls_loss += cls_loss
                        spat_reg_loss += reg_loss

                elif anchor_type[0] == "speed":
                    if anchor_type[1] not in speed_dict:
                        speed_dict[anchor_type[1]] = {
                            "pred_cls": [pred_cls],
                            "pred_reg": [pred_reg],
                            "gt_trajs": gt_trajs,
                            "gt_masks": gt_masks,
                            "speed_areas": [anchor_type[2]]
                        }
                    else:
                        assert (speed_dict[anchor_type[1]]["gt_trajs"] == gt_trajs).all()
                        assert (speed_dict[anchor_type[1]]["gt_masks"] == gt_masks).all()
                        speed_dict[anchor_type[1]]["pred_cls"].append(pred_cls)
                        speed_dict[anchor_type[1]]["pred_reg"].append(pred_reg)
                        speed_dict[anchor_type[1]]["speed_areas"].append(anchor_type[2])

            # speed loss
            if len(speed_dict):
                for k, v in speed_dict.items():
                    ref_speed_trajs, ref_speed_masks = self.get_gt_trajs(data, self.plan_speed_refer)
                    cls_loss, reg_loss = self.align_speed_traj_loss(
                        ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, ref_speed_trajs, ref_speed_masks,
                        v["pred_cls"], v["pred_reg"], v["gt_trajs"], v["gt_masks"], v["speed_areas"], data
                    )
                    speed_cls_loss += cls_loss
                    speed_reg_loss += reg_loss

            loss_types = []
            for anchor_type in self.plan_anchor_types:
                if anchor_type[0] not in loss_types:
                    loss_types.append(anchor_type[0])

            if self.combine_layer_loss:
                if "temp" in loss_types:
                    if "plan_loss_temp_cls" not in output:
                        output["plan_loss_temp_cls"] = 0.0
                    if "plan_loss_temp_reg" not in output:
                        output["plan_loss_temp_reg"] = 0.0
                    output["plan_loss_temp_cls"] += temp_cls_loss
                    output["plan_loss_temp_reg"] += temp_reg_loss
                if "spat" in loss_types:
                    if "plan_loss_spat_cls" not in output:
                        output["plan_loss_spat_cls"] = 0.0
                    if "plan_loss_spat_reg" not in output:
                        output["plan_loss_spat_reg"] = 0.0
                    output["plan_loss_spat_cls"] += spat_cls_loss
                    output["plan_loss_spat_reg"] += spat_reg_loss
                if "speed" in loss_types:
                    if "plan_loss_speed_cls" not in output:
                        output["plan_loss_speed_cls"] = 0.0
                    if "plan_loss_speed_reg" not in output:
                        output["plan_loss_speed_reg"] = 0.0
                    output["plan_loss_speed_cls"] += speed_cls_loss
                    output["plan_loss_speed_reg"] += speed_reg_loss

            else:
                if "temp" in loss_types:
                    output[f"plan_loss_temp_cls_{decoder_idx}"] = temp_cls_loss
                    output[f"plan_loss_temp_reg_{decoder_idx}"] = temp_reg_loss
                if "spat" in loss_types:
                    output[f"plan_loss_spat_cls_{decoder_idx}"] = spat_cls_loss
                    output[f"plan_loss_spat_reg_{decoder_idx}"] = spat_reg_loss
                if "speed" in loss_types:
                    output[f"plan_loss_speed_cls_{decoder_idx}"] = speed_cls_loss
                    output[f"plan_loss_speed_reg_{decoder_idx}"] = speed_reg_loss

        return output

    @force_fp32(apply_to=("det_output", "map_output", "ego_output", "plan_output", "motion_output"))
    def post_process(self, det_output, map_output, ego_output, plan_output, motion_output, data, output_idx=-1):
        det_result, map_result, ego_result, plan_result, motion_result = None, None, None, None, None

        if "det" in self.task_select:
            det_result = self.det_decoder.decode(
                det_output["classification"], det_output["prediction"],
                det_output.get("instance_id"), det_output.get("quality"), output_idx=output_idx)
        if "map" in self.task_select:
            map_result = self.map_decoder.decode(
                map_output["classification"], map_output["prediction"],
                map_output.get("instance_id"), map_output.get("quality"), output_idx=output_idx)
        if "motion" in self.task_select:
            motion_result = self.motion_decoder.decode(
                det_output["classification"], det_output["prediction"],
                det_output.get("instance_id"), det_output.get("quality"), motion_output)
        if "ego" in self.task_select:
            if not self.with_supervise_ego_status:
                ego_result = self.ego_decoder.decode(
                    det_output, motion_output, ego_output, data)
        if "plan" in self.task_select:
            plan_result = self.plan_decoder.decode(
                ego_output, det_output, motion_output, plan_output, data)

        return det_result, map_result, ego_result, plan_result, motion_result

