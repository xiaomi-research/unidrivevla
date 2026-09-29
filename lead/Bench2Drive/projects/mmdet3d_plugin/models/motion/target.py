import torch
from mmdet.core.bbox.builder import BBOX_SAMPLERS


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
    mode_idx,
):
    _, _, _, ts, d = reg_preds.shape
    gather_idx = mode_idx[..., None, None, None].expand(-1, -1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, gather_idx).squeeze(2)
    return best_reg
def sample_motion_target(
    reg_pred,
    gt_reg_target,
    gt_reg_mask,
    motion_loss_cache,
):
    cached_targets = motion_loss_cache.get("targets")
    if cached_targets is None:
        bs, num_anchor, _, ts, d = reg_pred.shape
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d))
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))
        num_pos = 0
        for i, (pred_idx, target_idx) in enumerate(motion_loss_cache["indices"]):
            if len(gt_reg_target[i]) == 0:
                continue
            reg_target[i, pred_idx] = gt_reg_target[i][target_idx]
            reg_weight[i, pred_idx] = gt_reg_mask[i][target_idx]
            num_pos += len(pred_idx)
        cached_targets = (reg_target, reg_weight, num_pos)
        motion_loss_cache["targets"] = cached_targets
    else:
        reg_target, reg_weight, num_pos = cached_targets

    mode_idx = get_cls_target(reg_pred, reg_target, reg_weight)
    cls_weight = reg_weight.any(dim=-1)
    best_reg = get_best_reg(reg_pred, mode_idx)
    return mode_idx, cls_weight, best_reg, reg_target, reg_weight, num_pos



@BBOX_SAMPLERS.register_module()
class MotionTarget():
    def __init__(
            self,
    ):
        super(MotionTarget, self).__init__()

    def sample(
            self,
            reg_pred,
            gt_reg_target,
            gt_reg_mask,
            motion_loss_cache,
    ):
        return sample_motion_target(
            reg_pred,
            gt_reg_target,
            gt_reg_mask,
            motion_loss_cache,
        )



@BBOX_SAMPLERS.register_module()
class SparseMotionTarget():
    def __init__(
        self,
    ):
        super(SparseMotionTarget, self).__init__()

    def sample(
        self,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        motion_loss_cache,
    ):
        return sample_motion_target(
            reg_pred,
            gt_reg_target,
            gt_reg_mask,
            motion_loss_cache,
        )
