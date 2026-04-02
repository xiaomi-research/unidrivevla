
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
    Flow Matching planning loss with HDP improvements.

    Two mutually exclusive prediction modes (controlled by use_tau0_pred in
    the planning head, NOT by this loss):

    Mode A — velocity prediction (use_tau0_pred=False, default):
        pred  = v_pred  (velocity field)
        target = u_t = noise - x0
        Main loss: ||v_pred - u_t||²  (flow MSE)

    Mode B — τ₀ prediction (use_tau0_pred=True):
        pred  = pred_x0  (predicted clean trajectory, normalized delta)
        target = x0  (GT clean trajectory, normalized delta)
        Main loss: ||pred_x0 - x0||²  (τ₀ MSE)

    In both modes, Hybrid Loss can be added on top:
        Hybrid loss: ||cumsum_detached(denorm(pred_x0)) - cumsum(denorm(x0))||²
        (waypoint L2 in meter space, with detached integral for stable gradients)

    Args:
        use_min_snr_loss: Apply Min-SNR weighting to the main MSE loss.
        min_snr_gamma: Gamma for Min-SNR weighting.
        loss_weight: Global scale for the total loss.
        hybrid_loss_weight: Weight for Hybrid waypoint loss. 0 = disabled.
        detach_window_size: Window W for detached integral (HDP Alg. 1).
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
        lateral_lambda: float = 0.0,
        jerk_penalty_weight: float = 0.0,
    ):
        super().__init__()
        self.use_min_snr_loss = bool(use_min_snr_loss)
        self.min_snr_gamma = float(min_snr_gamma)
        self.loss_weight = float(loss_weight)
        self.hybrid_loss_weight = float(hybrid_loss_weight)
        self.detach_window_size = int(detach_window_size)
        self.lateral_lambda = float(lateral_lambda)      # >0 enables decoupled XY hybrid loss
        self.jerk_penalty_weight = float(jerk_penalty_weight)

        if delta_mu is None:
            delta_mu = [0.0233, 2.2707]
        if delta_std is None:
            delta_std = [0.3427, 1.8668]

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
            pred:   [B, T, D] — v_pred (velocity mode) or pred_x0 (τ₀ mode)
            target: [B, T, D] — u_t (velocity mode) or gt_x0 (τ₀ mode)
            time:   [B]
            gt_ego_fut_masks: optional valid mask
            pred_x0: [B, T, D] — predicted x0 in normalized delta space
                     (same as pred in τ₀ mode; derived from pred+x_t in velocity mode)
                     required only when hybrid_loss_weight > 0
            gt_x0:  [B, T, D] — GT x0 in normalized delta space (= actions)
                     required only when hybrid_loss_weight > 0
        """
        # ------------------------------------------------------------------ #
        #  Main MSE loss (velocity space OR τ₀ space, depending on mode)      #
        # ------------------------------------------------------------------ #
        main_loss = F.mse_loss(pred, target, reduction="none")  # [B, T, D]

        if self.use_min_snr_loss:
            if not torch.is_tensor(time):
                time = torch.tensor(time, device=main_loss.device)
            weight = self._compute_min_snr_weight(time).view(-1, 1, 1)
            main_loss = main_loss * weight

        total_loss = _masked_mse(main_loss, gt_ego_fut_masks)

        # ------------------------------------------------------------------ #
        #  Hybrid Loss — waypoint L2 in meter space (HDP)                     #
        #  Works in both velocity and τ₀ mode via pred_x0                     #
        # ------------------------------------------------------------------ #
        if self.hybrid_loss_weight > 0.0 and pred_x0 is not None and gt_x0 is not None:
            pred_delta_meter = self._denorm_delta(pred_x0)
            gt_delta_meter = self._denorm_delta(gt_x0)

            pred_wpts = _detached_integral(pred_delta_meter, self.detach_window_size)
            gt_wpts = torch.cumsum(gt_delta_meter.detach(), dim=-2)

            if self.lateral_lambda > 0.0:
                # Decoupled XY hybrid loss: x=[...,0] lateral, y=[...,1] longitudinal
                # L = e_lon^2 + lambda * e_lat^2
                err = (pred_wpts - gt_wpts) ** 2          # [B, T, 2]
                err_decoupled = torch.stack([
                    self.lateral_lambda * err[..., 0],     # lateral (x) with high weight
                    err[..., 1],                           # longitudinal (y) normal weight
                ], dim=-1)
                hybrid_loss = _masked_mse(err_decoupled, gt_ego_fut_masks)
            else:
                hybrid_loss = _masked_mse(
                    F.mse_loss(pred_wpts, gt_wpts, reduction="none"),
                    gt_ego_fut_masks,
                )

            total_loss = total_loss + self.hybrid_loss_weight * hybrid_loss

            # Jerk penalty: penalize abrupt lateral changes (2nd-order diff of pred_wpts x)
            if self.jerk_penalty_weight > 0.0:
                lateral_wpts = pred_wpts[..., 0]           # [B, T]
                jerk = lateral_wpts[..., 2:] - 2 * lateral_wpts[..., 1:-1] + lateral_wpts[..., :-2]
                jerk_loss = _masked_mse(
                    (jerk ** 2).unsqueeze(-1),
                    gt_ego_fut_masks[..., 2:] if gt_ego_fut_masks is not None else None,
                )
                total_loss = total_loss + self.jerk_penalty_weight * jerk_loss

        return total_loss * self.loss_weight
