import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    """BEV collision loss between predicted ego trajectory and GT agent boxes.

    Two modes controlled by `soft_margin`:

    soft_margin = 0.0 (default, original behaviour):
        Hard AABB penetration: loss is non-zero only when boxes physically overlap.
        pen_x * pen_y > 0 iff ego and agent boxes intersect in both axes.

    soft_margin > 0.0 (e.g. 1.5):
        Soft distance penalty: loss starts accumulating `soft_margin` metres
        before actual contact, giving the model a gradient signal while the
        ego is still approaching the agent.
        L = relu(thresh_x + margin - dist_x) * relu(thresh_y + margin - dist_y)

    Args:
        ego_w (float): Ego vehicle width in metres. Default: 1.85.
        ego_l (float): Ego vehicle length in metres. Default: 4.084.
        delta (float): Safety margin added to both ego and agent half-extents.
        weight (float): Scalar weight applied to the returned loss.
        soft_margin (float): Extra look-ahead distance in metres. 0 = hard mode.
    """

    def __init__(self, ego_w: float = 1.85, ego_l: float = 4.084,
                 delta: float = 0.5, weight: float = 1.0,
                 soft_margin: float = 1.5):
        super().__init__()
        self.ego_w = ego_w + delta
        self.ego_l = ego_l + delta
        self.weight = weight
        self.soft_margin = float(soft_margin)

    def forward(
        self,
        pred_abs: torch.Tensor,        # [B, T, 2]  absolute ego positions (lidar frame)
        gt_bboxes_3d: list,            # list[B] of tensors [N, 9] – lidar-frame boxes
        gt_ego_fut_masks: torch.Tensor = None,  # [B, T] valid-timestep mask
    ) -> torch.Tensor:
        """
        Args:
            pred_abs: Predicted ego absolute waypoints, shape [B, T, 2] (x, y in lidar frame).
            gt_bboxes_3d: Per-sample list of agent boxes in lidar frame.
                Box format: [cx, cy, cz, dx, dy, dz, yaw, vx, vy]  (9-dim).
            gt_ego_fut_masks: Binary mask [B, T]; timesteps with value < 0.5 are skipped.

        Returns:
            Scalar collision loss (sum of penalty / n_valid_timesteps).
        """
        device = pred_abs.device
        dtype = torch.float32
        pred_abs = pred_abs.to(dtype)

        B, T, _ = pred_abs.shape
        loss = pred_abs.new_zeros(1)

        for b in range(B):
            boxes = gt_bboxes_3d[b]
            if boxes is None:
                continue
            if not torch.is_tensor(boxes):
                boxes = torch.tensor(boxes, device=device, dtype=dtype)
            else:
                boxes = boxes.to(device=device, dtype=dtype)
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
            if boxes.shape[0] == 0:
                continue

            # Agent box parameters (AABB, ignoring yaw for simplicity)
            cx = boxes[:, 0]   # [N]
            cy = boxes[:, 1]   # [N]
            ax = boxes[:, 3]   # agent extent along x  [N]
            ay = boxes[:, 4]   # agent extent along y  [N]

            for t in range(T):
                if gt_ego_fut_masks is not None and gt_ego_fut_masks[b, t] < 0.5:
                    continue

                ex = pred_abs[b, t, 0]   # scalar
                ey = pred_abs[b, t, 1]   # scalar

                dist_x = (ex - cx).abs()                                    # [N]
                dist_y = (ey - cy).abs()                                    # [N]

                thresh_x = self.ego_w / 2 + ax / 2 + self.soft_margin     # [N]
                thresh_y = self.ego_l / 2 + ay / 2 + self.soft_margin     # [N]

                pen_x = F.relu(thresh_x - dist_x)                          # [N]
                pen_y = F.relu(thresh_y - dist_y)                          # [N]

                loss = loss + (pen_x * pen_y).sum()

        if gt_ego_fut_masks is not None:
            n_valid = (gt_ego_fut_masks >= 0.5).sum().clamp(min=1).to(dtype)
        else:
            n_valid = torch.tensor(float(B * T), device=device, dtype=dtype)

        return (loss / n_valid) * self.weight
