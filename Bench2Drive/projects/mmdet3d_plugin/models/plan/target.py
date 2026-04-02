import torch
from mmdet.core.bbox.builder import BBOX_SAMPLERS

__all__ = ["PlanningTarget", "SparsePlanTarget"]


def get_cls_target(
        reg_preds,
        reg_target,
        reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx


def get_best_reg(
        reg_preds,
        reg_target,
        reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg



@BBOX_SAMPLERS.register_module()
class PlanningTarget():
    def __init__(
            self,
            ego_fut_ts=6,
            ego_fut_cmd=3,
            ego_fut_mode=3,
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode

    def sample(
            self,
            cls_pred,
            reg_pred,
            gt_reg_target,
            gt_reg_mask,
            data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, self.ego_fut_cmd, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, self.ego_fut_cmd, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


@BBOX_SAMPLERS.register_module()
class SparsePlanTarget():
    def __init__(
            self,
            ego_fut_ts=6,
            ego_fut_cmd=3,
            ego_fut_mode=3,
    ):
        super(SparsePlanTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode

    def sample(
            self,
            cls_pred,
            reg_pred,
            gt_reg_target,
            gt_reg_mask,
            data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs, _, num_traj = cls_pred.shape
        bs_indices = torch.arange(bs, device=reg_pred.device)

        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1) if self.ego_fut_cmd>1 else 0

        cls_pred = cls_pred.reshape(bs, self.ego_fut_cmd, 1, -1)
        reg_pred = reg_pred.reshape(bs, self.ego_fut_cmd, 1, -1, self.ego_fut_ts, 2)

        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]

        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


@BBOX_SAMPLERS.register_module()
class AlignPlanTarget():
    def __init__(
            self,
            ego_fut_ts=6,
            ego_fut_cmd=3,
            ego_fut_mode=3,
    ):
        super(AlignPlanTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode

    def sample(
            self,
            cls_pred,
            reg_pred,
            gt_reg_target,
            gt_reg_mask,
            data,
            ref_target,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs, _, num_traj = cls_pred.shape
        bs_indices = torch.arange(bs, device=reg_pred.device)

        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1) if self.ego_fut_cmd>1 else 0

        cls_pred = cls_pred.reshape(bs, self.ego_fut_cmd, 1, -1)
        reg_pred = reg_pred.reshape(bs, self.ego_fut_cmd, 1, -1, self.ego_fut_ts, 2)

        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]

        cls_target = ref_target.clone()
        cls_weight = gt_reg_mask.any(dim=-1)

        mode_idx = cls_target[..., None, None, None].repeat(1, 1, 1, self.ego_fut_ts, 2)
        best_reg = torch.gather(reg_pred, 2, mode_idx).squeeze(2)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask