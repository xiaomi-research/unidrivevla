"""Torch stand-ins for the mmcv/mmdet symbols the perception stack constructs.

The NAVSIM model imports this module. It does not import mmcv, mmdet, or the
nuScenes / Bench2Drive trees.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Registry:
    def __init__(self, name):
        self._name = name
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register

    def get(self, key):
        if key not in self.module_dict:
            raise KeyError(f"{key} is not in {self._name}: {sorted(self.module_dict)}")
        return self.module_dict[key]


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    cfg = dict(cfg)
    if default_args:
        for key, value in default_args.items():
            cfg.setdefault(key, value)
    cls = cfg.pop("type")
    if isinstance(cls, str):
        cls = registry.get(cls)
    return cls(**cfg)


ATTENTION = Registry("attention")
PLUGIN_LAYERS = Registry("plugin")
POSITIONAL_ENCODING = Registry("pos")
FEEDFORWARD_NETWORK = Registry("ffn")
NORM_LAYERS = Registry("norm")
HEADS = Registry("heads")
LOSSES = Registry("losses")
DETECTORS = Registry("detectors")
BBOX_SAMPLERS = Registry("samplers")
BBOX_CODERS = Registry("coders")
BBOX_ASSIGNERS = Registry("assigners")
MATCH_COST = Registry("match_cost")


def build_head(cfg):
    return build_from_cfg(cfg, HEADS)


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)


def build_assigner(cfg):
    return build_from_cfg(cfg, BBOX_ASSIGNERS)


def build_sampler(cfg):
    return build_from_cfg(cfg, BBOX_SAMPLERS)


def build_match_cost(cfg):
    return build_from_cfg(cfg, MATCH_COST)


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        pass


Sequential = nn.Sequential
Linear = nn.Linear


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class FFN(BaseModule):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, act_cfg=None, add_identity=True, **kwargs):
        super().__init__()
        layers = []
        in_ch = embed_dims
        for i in range(num_fcs - 1):
            layers += [nn.Linear(in_ch, feedforward_channels), nn.ReLU(inplace=True), nn.Dropout(ffn_drop)]
            in_ch = feedforward_channels
        layers.append(nn.Linear(in_ch, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return out
        return out + (x if identity is None else identity)


def bias_init_with_prob(prior_prob):
    return float(-math.log((1 - prior_prob) / prior_prob))


def xavier_init(module, distribution="uniform", bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.xavier_normal_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def build_norm_layer(cfg, num_features, postfix=""):
    cfg = dict(cfg or {})
    layer_type = cfg.pop("type", "LN")
    if layer_type == "LN":
        shape = cfg.pop("normalized_shape", num_features)
        return "ln" + str(postfix), nn.LayerNorm(shape, **cfg)
    if layer_type == "BN":
        return "bn" + str(postfix), nn.BatchNorm2d(num_features, **cfg)
    if layer_type == "GN":
        return "gn" + str(postfix), nn.GroupNorm(cfg.pop("num_groups", 32), num_features)
    raise KeyError(layer_type)


def build_activation_layer(cfg):
    cfg = dict(cfg or {"type": "ReLU"})
    name = cfg.pop("type")
    if name == "ReLU":
        return nn.ReLU(**cfg)
    if name == "GELU":
        return nn.GELU()
    raise KeyError(name)


def build_dropout(cfg, default_args=None):
    if cfg is None:
        return nn.Identity()
    cfg = dict(cfg)
    cfg.pop("type", None)
    p = cfg.get("drop_prob", cfg.get("p", 0.0))
    return nn.Dropout(p)


def auto_fp16(func=None, **kwargs):
    if callable(func):
        return func

    def deco(fn):
        return fn

    return deco


force_fp32 = auto_fp16


def deprecated_api_warning(*args, **kwargs):
    def deco(fn):
        return fn

    return deco


def reduce_mean(tensor):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor / torch.distributed.get_world_size()


def _reduce(loss, weight, reduction, avg_factor):
    if weight is not None:
        loss = loss * weight
    if avg_factor is not None:
        return loss.sum() / avg_factor
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def l1_loss(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
    return _reduce((pred - target).abs(), weight, reduction, avg_factor)


def smooth_l1_loss(pred, target, weight=None, reduction="mean", avg_factor=None, beta=1.0, **kwargs):
    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return _reduce(loss, weight, reduction, avg_factor)


class _LossBase(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction


class L1Loss(_LossBase):
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        return self.loss_weight * l1_loss(pred, target, weight, reduction, avg_factor)


class CrossEntropyLoss(_LossBase):
    def __init__(self, use_sigmoid=False, loss_weight=1.0, reduction="mean", **kwargs):
        super().__init__(loss_weight, reduction)
        self.use_sigmoid = use_sigmoid

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        if self.use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        else:
            loss = F.cross_entropy(pred, target.long(), reduction="none")
        return self.loss_weight * _reduce(loss, weight, reduction, avg_factor)


class FocalLoss(_LossBase):
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0, reduction="mean", **kwargs):
        super().__init__(loss_weight, reduction)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        if target.dtype in (torch.long, torch.int64) and pred.shape != target.shape:
            num_cls = pred.shape[-1]
            target = target.long()
            valid = (target >= 0) & (target < num_cls)
            one = F.one_hot(target.clamp(0, num_cls - 1), num_cls).float()
            target = one * valid.unsqueeze(-1).to(one.dtype)
        prob = pred.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        pt = prob * target + (1 - prob) * (1 - target)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha * (1 - pt).pow(self.gamma) * ce
        return self.loss_weight * _reduce(loss, weight, reduction, avg_factor)


class GaussianFocalLoss(_LossBase):
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        pred = pred.sigmoid().clamp(1e-6, 1 - 1e-6)
        pos = target.eq(1).float()
        neg = target.lt(1).float()
        loss = -(pos * (1 - pred).pow(2) * pred.log() + neg * (1 - target).pow(4) * pred.pow(2) * (1 - pred).log())
        return self.loss_weight * _reduce(loss, weight, reduction, avg_factor)


class FocalLossCost:
    def __init__(self, weight=1.0, alpha=0.25, gamma=2.0, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


class AssignResult:
    def __init__(self, num_gts, assigned_gt_inds, max_overlaps=None, labels=None):
        self.num_gts = num_gts
        self.assigned_gt_inds = assigned_gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels


class BaseAssigner:
    def assign(self, *args, **kwargs):
        raise NotImplementedError


class BaseBBoxCoder:
    def __init__(self, **kwargs):
        pass


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


for _cls in (L1Loss, FocalLoss, CrossEntropyLoss, GaussianFocalLoss):
    LOSSES.register_module(module=_cls)
MATCH_COST.register_module(module=FocalLossCost)
NORM_LAYERS.register_module(name="LN", module=nn.LayerNorm)
NORM_LAYERS.register_module(name="BN", module=nn.BatchNorm2d)
FEEDFORWARD_NETWORK.register_module(module=FFN)
