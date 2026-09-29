import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class SparseEgoRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_cmd=3, ego_fut_mode=3):
        super(SparseEgoRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, ego_query, ego_feature, ego_anchor_embed):
        bs = ego_query.shape[0]
        plan_cls = self.plan_cls_branch(ego_query).squeeze(-1)
        plan_reg = self.plan_reg_branch(ego_query).reshape(bs, 1, self.ego_fut_cmd * self.ego_fut_mode, self.ego_fut_ts, 2)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)

        return plan_cls, plan_reg, planning_status



@PLUGIN_LAYERS.register_module()
class EgoStatusRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, status_dims=6):
        super(EgoStatusRefinementModule, self).__init__()
        self.embed_dims = embed_dims

        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, status_dims),
        )

    def forward(self, ego_feature, ego_anchor_embed):
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)

        return planning_status
