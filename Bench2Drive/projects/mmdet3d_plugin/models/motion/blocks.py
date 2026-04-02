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
class SparseMotionRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
    ):
        super(SparseMotionRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)


    def forward(self, motion_query):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)

        return motion_cls, motion_reg