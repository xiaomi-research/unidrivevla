import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from ..blocks import linear_relu_ln
from projects.mmdet3d_plugin.models.utils import nerf_positional_encoding
from functools import partial

@PLUGIN_LAYERS.register_module()
class SparsePlanRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_cmd=3, ego_fut_mode=3, add_anchor=False):
        super(SparsePlanRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode
        self.add_anchor = add_anchor

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )

        self.plan_reg_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, ego_fut_ts * 2),
            Scale([1.0] * ego_fut_ts * 2),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, instance_feature, anchor, anchor_embed, use_plan_anchor_embed=True):
        if use_plan_anchor_embed:
            output = self.plan_reg_branch(instance_feature + anchor_embed)
        else:
            output = self.plan_reg_branch(instance_feature)

        output = output + anchor

        cls = self.plan_cls_branch(instance_feature)

        return output, cls

@PLUGIN_LAYERS.register_module()
class SparsePlanAlignRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_cmd=3, ego_fut_mode=3, anchor_types=None):
        super(SparsePlanAlignRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode

        self.anchor_types = anchor_types
        self.anchor_group = len(anchor_types)

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )

        # check speed planning
        speed_type_dict = dict()
        for anchor_type in anchor_types:
            if anchor_type[0] == "speed":
                if anchor_type[1] not in speed_type_dict:
                    speed_type_dict[anchor_type[1]] = [anchor_type[2]]
                else:
                    speed_type_dict[anchor_type[1]].append(anchor_type[2])

        if len(speed_type_dict):
            first_key = list(speed_type_dict.keys())[0]
            self.speed_areas = speed_type_dict[first_key]
            if len(speed_type_dict) > 1:
                for key, val in speed_type_dict.items():
                    assert self.speed_areas == val

            self.plan_cls_branch_speed = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(embed_dims, 1),
            )

        for anchor_type in anchor_types:
            reg_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, ego_fut_ts * 2),
                Scale([1.0] * ego_fut_ts * 2),
            )
            setattr(self, "plan_reg_branch_{}_{}".format(anchor_type[0], anchor_type[1]), reg_branch)

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

        if hasattr(self, "plan_cls_branch_speed"):
            nn.init.constant_(self.plan_cls_branch_speed[-1].bias, bias_init)

    def forward(self, instance_feature, anchor, anchor_embed, use_plan_anchor_embed=True):
        if use_plan_anchor_embed:
            instance_feature = instance_feature + anchor_embed

        instance_features = torch.stack(instance_feature.chunk(self.anchor_group, dim=1))

        align_query = []
        speed_query_dict = dict()
        for index, anchor_type in enumerate(self.anchor_types):
            if anchor_type[0] in ["temp", "spat"]:
                align_query.append(instance_features[index])
            elif anchor_type[0] == "speed":
                if anchor_type[1] not in speed_query_dict:
                    speed_query_dict[anchor_type[1]] = [None] * len(self.speed_areas)
                speed_index = self.speed_areas.index(anchor_type[2])
                speed_query_dict[anchor_type[1]][speed_index] = instance_features[index]
            else:
                raise NotImplementedError

        align_query = sum(align_query)

        if len(speed_query_dict):
            for speed_index in range(len(self.speed_areas)):
                speed_query = []
                for freq in speed_query_dict.keys():
                    speed_query.append(speed_query_dict[freq][speed_index])
                speed_query = sum(speed_query)
                for freq in speed_query_dict.keys():
                    speed_query_dict[freq][speed_index] = align_query + speed_query

        cls_outputs = []
        reg_outputs = []
        for anchor_type in self.anchor_types:
            reg_branch = getattr(self, "plan_reg_branch_{}_{}".format(anchor_type[0], anchor_type[1]))
            if anchor_type[0] in ["temp", "spat"]:
                reg_output = reg_branch(align_query)
                cls_output = self.plan_cls_branch(align_query)

            elif anchor_type[0] == "speed":
                speed_index = self.speed_areas.index(anchor_type[2])
                speed_query = speed_query_dict[anchor_type[1]][speed_index]
                reg_output = reg_branch(speed_query)
                cls_output = self.plan_cls_branch_speed(speed_query)

            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)

        cls_outputs = torch.cat(cls_outputs, dim=1)
        reg_outputs = torch.cat(reg_outputs, dim=1)

        reg_outputs = reg_outputs + anchor

        return reg_outputs, cls_outputs