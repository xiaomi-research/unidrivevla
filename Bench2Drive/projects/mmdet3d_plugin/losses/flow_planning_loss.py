
from mmdet.models import LOSSES
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _detached_integral(v: torch.Tensor, W: int) -> torch.Tensor:
    """
    HDP Algorithm 1: Detached Integral.

    Integrates per-step deltas into waypoints while limiting gradient
    propagation to a window of W recent timesteps, preventing gradient
    explosion from the cumsum chain.

    Args:
        v: per-step delta [..., T, D]
        W: gradient detach window size

    Returns:
        waypoints [..., T, D]
    """
    cum_detach = torch.cumsum(v.detach(), dim=-2)
    cum_normal = torch.cumsum(v, dim=-2)

    shifted = torch.roll(cum_normal, shifts=W, dims=-2)
    shifted[..., :W, :] = 0
    sum_recent = cum_normal - shifted

    cum_detach_shifted = torch.roll(cum_detach, shifts=W, dims=-2)
    cum_detach_shifted[..., :W, :] = 0

    return cum_detach_shifted + sum_recent


def _masked_mse(loss_unreduced: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        m = mask
        if m.dim() == 3 and m.shape[1] == 1:
            m = m.squeeze(1)
        m = m.to(dtype=loss_unreduced.dtype, device=loss_unreduced.device)
        if m.dim() == 2:
            m = m.unsqueeze(-1)
        loss_unreduced = loss_unreduced * m
        denom = m.sum().clamp_min(1.0) * loss_unreduced.shape[-1]
        return loss_unreduced.sum() / denom
    return loss_unreduced.mean()


@LOSSES.register_module()
class FlowPlanningLoss(nn.Module):
    """
    Flow Matching planning loss with optional Hybrid Loss.

    Main loss: ||v_pred - u_t||²  (flow MSE) with optional Min-SNR weighting.

    Hybrid Loss (optional): waypoint L2 in meter space using detached integral
    for stable gradients (HDP Algorithm 1).
        ||cumsum_detached(denorm(pred_x0)) - cumsum(denorm(x0))||²

    Args:
        use_min_snr_loss: Apply Min-SNR weighting to the main MSE loss.
        min_snr_gamma: Gamma for Min-SNR weighting.
        loss_weight: Global scale for the total loss.
        hybrid_loss_weight: Weight for Hybrid waypoint loss. 0 = disabled.
        detach_window_size: Window W for detached integral.
        delta_mu: Mean for delta denormalization [dx, dy].
        delta_std: Std for delta denormalization [dx, dy].
    """

    def __init__(
        self,
        use_min_snr_loss: bool = False,
        min_snr_gamma: float = 5.0,
        loss_weight: float = 1.0,
        hybrid_loss_weight: float = 0.0,
        detach_window_size: int = 10,
        delta_mu: Optional[list] = None,
        delta_std: Optional[list] = None,
    ):
        super().__init__()
        self.use_min_snr_loss = bool(use_min_snr_loss)
        self.min_snr_gamma = float(min_snr_gamma)
        self.loss_weight = float(loss_weight)
        self.hybrid_loss_weight = float(hybrid_loss_weight)
        self.detach_window_size = int(detach_window_size)

        # B2D Bench2Drive delta normalization statistics
        if delta_mu is None:
            delta_mu = [-0.0222, 2.0249]
        if delta_std is None:
            delta_std = [0.6720, 1.8586]

        self.register_buffer('delta_mu', torch.tensor(delta_mu, dtype=torch.float32))
        self.register_buffer('delta_std', torch.tensor(delta_std, dtype=torch.float32))

    def _denorm_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        mu = self.delta_mu.to(device=delta_norm.device, dtype=delta_norm.dtype)
        std = self.delta_std.to(device=delta_norm.device, dtype=delta_norm.dtype)
        return delta_norm * (std + 1e-6) + mu

    def _compute_min_snr_weight(self, time: torch.Tensor) -> torch.Tensor:
        alpha_t = 1 - time
        sigma_t = time
        snr = (alpha_t ** 2) / (sigma_t ** 2 + 1e-8)
        min_snr = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma))
        return min_snr / (snr + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        time: torch.Tensor,
        gt_ego_fut_masks: Optional[torch.Tensor] = None,
        pred_x0: Optional[torch.Tensor] = None,
        gt_x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            pred:   [B, T, D] — v_pred (velocity field)
            target: [B, T, D] — u_t (flow target)
            time:   [B]
            gt_ego_fut_masks: optional valid mask
            pred_x0: [B, T, D] — predicted x0 in normalized delta space
                     required when hybrid_loss_weight > 0
            gt_x0:  [B, T, D] — GT x0 in normalized delta space (= actions)
                     required when hybrid_loss_weight > 0
        """
        main_loss = F.mse_loss(pred, target, reduction="none")  # [B, T, D]

        if self.use_min_snr_loss:
            if not torch.is_tensor(time):
                time = torch.tensor(time, device=main_loss.device)
            weight = self._compute_min_snr_weight(time).view(-1, 1, 1)
            main_loss = main_loss * weight

        total_loss = _masked_mse(main_loss, gt_ego_fut_masks)

        # Hybrid Loss: waypoint L2 in meter space
        if self.hybrid_loss_weight > 0.0 and pred_x0 is not None and gt_x0 is not None:
            pred_delta_meter = self._denorm_delta(pred_x0)
            gt_delta_meter = self._denorm_delta(gt_x0)

            pred_wpts = _detached_integral(pred_delta_meter, self.detach_window_size)
            gt_wpts = torch.cumsum(gt_delta_meter.detach(), dim=-2)

            hybrid_loss = F.mse_loss(pred_wpts, gt_wpts, reduction="none")
            hybrid_loss = _masked_mse(hybrid_loss, gt_ego_fut_masks)

            total_loss = total_loss + self.hybrid_loss_weight * hybrid_loss

        return total_loss * self.loss_weight
