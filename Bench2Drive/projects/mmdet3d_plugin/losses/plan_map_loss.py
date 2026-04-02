"""Planning map-based losses using GT map data.

Two losses are provided, adapted from VAD/HiP-AD's PlanMapBoundLoss and
PlanMapDirectionLoss but operating on GT map annotations instead of
predicted map outputs.

GT map format (NuScenes):
  gt_map_pts:    List[Tensor[N_lines, 38, 20, 2]]  – 38 permutations, 20 pts, 2D ego-frame coords
  gt_map_labels: List[Tensor[N_lines]]              – 0=ped_crossing, 1=divider, 2=boundary

Both losses receive the predicted ego trajectory in *absolute* ego-frame coordinates
[B, T, 2] (i.e. after cumsum of deltas).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _segments_intersect(
    line1_start: torch.Tensor,  # [N, 2]
    line1_end:   torch.Tensor,  # [N, 2]
    line2_start: torch.Tensor,  # [N, 2]
    line2_end:   torch.Tensor,  # [N, 2]
) -> torch.Tensor:              # [N] bool
    dx1 = line1_end[:, 0] - line1_start[:, 0]
    dy1 = line1_end[:, 1] - line1_start[:, 1]
    dx2 = line2_end[:, 0] - line2_start[:, 0]
    dy2 = line2_end[:, 1] - line2_start[:, 1]

    det = dx1 * dy2 - dx2 * dy1
    parallel = det == 0

    # Safe division (parallel case handled below)
    safe_det = det.clone()
    safe_det[parallel] = 1.0

    t1 = ((line2_start[:, 0] - line1_start[:, 0]) * dy2
          - (line2_start[:, 1] - line1_start[:, 1]) * dx2) / safe_det
    t2 = ((line2_start[:, 0] - line1_start[:, 0]) * dy1
          - (line2_start[:, 1] - line1_start[:, 1]) * dx1) / safe_det

    intersect = (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
    intersect[parallel] = False
    return intersect


def _prepare_gt_map(gt_map_pts_b: torch.Tensor,
                    gt_map_labels_b: torch.Tensor,
                    cls_idx: int,
                    device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    """Return [M, P, 2] float tensor with lines of the requested class.

    gt_map_pts_b: [N_lines, 38, P, 2]  (38 permutation variants; use index 0)
    gt_map_labels_b: [N_lines]
    """
    if not torch.is_tensor(gt_map_pts_b):
        gt_map_pts_b = torch.tensor(gt_map_pts_b, dtype=dtype, device=device)
    else:
        gt_map_pts_b = gt_map_pts_b.to(device=device, dtype=dtype)

    if not torch.is_tensor(gt_map_labels_b):
        gt_map_labels_b = torch.tensor(gt_map_labels_b, dtype=torch.long, device=device)
    else:
        gt_map_labels_b = gt_map_labels_b.to(device=device)

    mask = gt_map_labels_b == cls_idx
    if mask.sum() == 0:
        return None

    # Use the canonical permutation (index 0): shape [M, P, 2]
    lines = gt_map_pts_b[mask, 0, :, :]
    return lines  # [M, P, 2]


# ─────────────────────────────────────────────────────────────────────────────
# PlanMapBoundLoss  (GT version)
# ─────────────────────────────────────────────────────────────────────────────

@LOSSES.register_module()
class GTMapBoundLoss(nn.Module):
    """Penalise predicted ego trajectory crossing / approaching road boundaries.

    For each ego waypoint the loss is:
      max(0, dis_thresh - min_dist_to_boundary_pts)

    Additionally, if the ego segment (t-1 → t) intersects any boundary segment,
    all subsequent timesteps for that sample get loss=0 (trajectory has already
    left the road; further penalty is meaningless and can destabilise training).

    Args:
        boundary_cls_idx (int): gt_map_labels index for road boundary. Default 2.
        dis_thresh (float): Safety margin in metres. Loss > 0 when closer than this.
        weight (float): Overall loss weight.
    """

    def __init__(self,
                 boundary_cls_idx: int = 2,
                 dis_thresh: float = 1.0,
                 weight: float = 1.0):
        super().__init__()
        self.boundary_cls_idx = boundary_cls_idx
        self.dis_thresh = dis_thresh
        self.weight = weight

    def forward(
        self,
        pred_abs: torch.Tensor,       # [B, T, 2]  absolute ego positions (ego frame)
        gt_map_pts: list,             # list[B] of Tensor[N_lines, 38, P, 2]
        gt_map_labels: list,          # list[B] of Tensor[N_lines]
        gt_ego_fut_masks: torch.Tensor = None,  # [B, T]
    ) -> torch.Tensor:
        B, T, _ = pred_abs.shape
        device = pred_abs.device
        dtype = torch.float32
        pred_abs = pred_abs.to(dtype)

        total_loss = pred_abs.new_zeros(1)
        n_valid = 0

        for b in range(B):
            lines = _prepare_gt_map(gt_map_pts[b], gt_map_labels[b],
                                    self.boundary_cls_idx, device, dtype)
            if lines is None:
                continue  # no boundary lines in this sample

            # lines: [M, P, 2]
            M, P, _ = lines.shape
            lines_flat = lines.reshape(M * P, 2)  # [M*P, 2]

            # Ego trajectory segments: origin→t0, t0→t1, ...
            origin = pred_abs.new_zeros(1, 2)
            traj_ext = torch.cat([origin, pred_abs[b]], dim=0)  # [T+1, 2]

            for t in range(T):
                if gt_ego_fut_masks is not None and gt_ego_fut_masks[b, t] < 0.5:
                    continue

                ego_pt = pred_abs[b, t]  # [2]

                # ── Distance loss ──────────────────────────────────────────
                dist = torch.linalg.norm(
                    ego_pt.unsqueeze(0) - lines_flat, dim=-1)   # [M*P]
                min_dist = dist.min()
                loss_t = F.relu(self.dis_thresh - min_dist)

                # ── Intersection check: stop accumulating after crossing ──
                seg_start = traj_ext[t].unsqueeze(0).expand(M * (P - 1), 2)
                seg_end   = traj_ext[t + 1].unsqueeze(0).expand(M * (P - 1), 2)
                bd_starts = lines[:, :-1, :].reshape(M * (P - 1), 2)
                bd_ends   = lines[:, 1:,  :].reshape(M * (P - 1), 2)
                crossed = _segments_intersect(seg_start, seg_end, bd_starts, bd_ends)
                if crossed.any():
                    # ego has already crossed boundary – zero out remaining ts
                    break

                total_loss = total_loss + loss_t
                n_valid += 1

        if n_valid == 0:
            return pred_abs.new_zeros(1)
        return (total_loss / n_valid) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# PlanMapDirectionLoss  (GT version)
# ─────────────────────────────────────────────────────────────────────────────

@LOSSES.register_module()
class GTMapDirectionLoss(nn.Module):
    """Penalise ego heading angle inconsistent with nearest lane divider direction.

    For each ego waypoint:
      1. Find the nearest lane-divider line instance.
      2. Find the nearest segment on that instance.
      3. Compute the angle difference between ego heading and segment direction.
      4. Loss = |angle_diff| (clamped to [0, π/2] by symmetry handling).

    Loss is zeroed when:
      - Ego is farther than `dis_thresh` from any divider.
      - Ego is nearly static (total displacement < 1 m).

    Args:
        divider_cls_idx (int): gt_map_labels index for lane/road dividers. Default 1.
        dis_thresh (float): Distance beyond which direction loss is ignored (metres).
        weight (float): Overall loss weight.
    """

    def __init__(self,
                 divider_cls_idx: int = 1,
                 dis_thresh: float = 2.0,
                 weight: float = 1.0):
        super().__init__()
        self.divider_cls_idx = divider_cls_idx
        self.dis_thresh = dis_thresh
        self.weight = weight

    def forward(
        self,
        pred_abs: torch.Tensor,       # [B, T, 2]  absolute ego positions (ego frame)
        gt_map_pts: list,             # list[B] of Tensor[N_lines, 38, P, 2]
        gt_map_labels: list,          # list[B] of Tensor[N_lines]
        gt_ego_fut_masks: torch.Tensor = None,  # [B, T]
    ) -> torch.Tensor:
        B, T, _ = pred_abs.shape
        device = pred_abs.device
        dtype = torch.float32
        pred_abs = pred_abs.to(dtype)

        total_loss = pred_abs.new_zeros(1)
        n_valid = 0

        for b in range(B):
            lines = _prepare_gt_map(gt_map_pts[b], gt_map_labels[b],
                                    self.divider_cls_idx, device, dtype)
            if lines is None:
                continue  # no divider lines in this sample

            # Skip nearly-static trajectories
            traj_disp = torch.linalg.norm(pred_abs[b, -1] - pred_abs[b, 0])
            if traj_disp < 1.0:
                continue

            M, P, _ = lines.shape

            # Ego heading at each timestep (finite diff, last ts reuses prev)
            origin = pred_abs.new_zeros(1, 2)
            traj_ext = torch.cat([origin, pred_abs[b]], dim=0)   # [T+1, 2]
            diff = traj_ext[1:] - traj_ext[:-1]                  # [T, 2]
            ego_yaw = torch.atan2(diff[:, 1], diff[:, 0])        # [T]

            for t in range(T):
                if gt_ego_fut_masks is not None and gt_ego_fut_masks[b, t] < 0.5:
                    continue

                ego_pt = pred_abs[b, t]  # [2]

                # ── Find nearest divider instance ──────────────────────────
                # dist from ego_pt to each point of each line: [M, P]
                dist_to_pts = torch.linalg.norm(
                    ego_pt.unsqueeze(0).unsqueeze(0) - lines, dim=-1)  # [M, P]
                min_dist_per_inst = dist_to_pts.min(dim=-1).values     # [M]
                nearest_inst = min_dist_per_inst.argmin()
                nearest_dist = min_dist_per_inst[nearest_inst]

                if nearest_dist > self.dis_thresh:
                    continue  # too far, skip

                # ── Find nearest segment on that instance ──────────────────
                inst_pts = lines[nearest_inst]   # [P, 2]
                dist_to_inst = dist_to_pts[nearest_inst]  # [P]
                pt_idx = dist_to_inst.argmin().clamp(0, P - 2)
                # Segment: pt_idx → pt_idx+1
                seg_dir = inst_pts[pt_idx + 1] - inst_pts[pt_idx]   # [2]
                lane_yaw = torch.atan2(seg_dir[1], seg_dir[0])

                # ── Angle difference (handle symmetry: road works both ways) ─
                yaw_diff = ego_yaw[t] - lane_yaw
                # Normalise to (-π, π]
                yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
                # Roads are bidirectional: fold to [0, π/2]
                yaw_diff = yaw_diff.abs()
                if yaw_diff > math.pi / 2:
                    yaw_diff = math.pi - yaw_diff

                total_loss = total_loss + yaw_diff
                n_valid += 1

        if n_valid == 0:
            return pred_abs.new_zeros(1)
        return (total_loss / n_valid) * self.weight
