"""BEV visualizer for PlanT object tokens, route, and predicted waypoints."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
import wandb
from beartype import beartype
from numpy.typing import NDArray

from lead.plant.plant_config import PlantConfig
from lead.tfv6.tfv6 import Prediction

# Object type -> (fill_color, outline_color) in BGR
_OBJECT_COLORS: dict[int, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    1: ((100, 0, 0), (255, 0, 0)),  # vehicle – red
    2: ((100, 100, 0), (255, 255, 0)),  # pedestrian – cyan
    3: ((50, 50, 50), (127, 127, 127)),  # static car – grey
    4: ((0, 0, 0), (255, 0, 255)),  # stop sign – magenta
    5: ((0, 0, 0), (0, 0, 255)),  # traffic light – red box
    6: ((0, 50, 100), (0, 100, 255)),  # emergency – orange
}
_DEFAULT_COLOR = ((127, 127, 127), (255, 0, 255))


def _rotate_point(x: float, y: float, theta_deg: float) -> tuple[float, float]:
    rad = math.radians(theta_deg)
    return (
        x * math.cos(rad) - y * math.sin(rad),
        y * math.cos(rad) + x * math.sin(rad),
    )


def _bbox_corners(
    cx: float,
    cy: float,
    yaw_deg: float,
    half_w: float,
    half_h: float,
) -> NDArray[np.int32]:
    """Return 4 corners of a rotated rectangle as an int32 array."""
    angle = yaw_deg - 90
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        dx = sx * half_w
        dy = sy * half_h
        rx, ry = _rotate_point(dx, dy, angle)
        corners.append((int(cx + rx), int(cy + ry)))
    return np.array(corners, dtype=np.int32)


@beartype
def visualize_plant_bev(
    data: dict,
    predictions: Prediction,
    index: int = 0,
    pix_per_m: float = 5.0,
    range_front: float = 128.0,
    range_sides: float = 64.0,
) -> NDArray[np.uint8]:
    """Render a BEV image showing PlanT object tokens, route, and waypoints.

    Args:
        data: Batch dict with ``idxs``, ``x_objs``, ``route_original``, and
            optionally ``future_waypoints``.
        predictions: Model predictions (uses ``pred_future_waypoints``).
        index: Sample index within the batch.
        pix_per_m: Pixels per meter scale.
        range_front: Forward range in meters.
        range_sides: Side/rear range in meters.

    Returns:
        BGR image of shape ``[H, W, 3]``.
    """
    ppm = pix_per_m
    w = int(range_sides * ppm * 2)
    h = int((range_sides + range_front) * ppm)
    origin_y = h - w // 2
    origin_x = w // 2

    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Detection range ellipse
    cv2.ellipse(
        img,
        (origin_x, origin_y),
        (int(ppm * range_sides), int(ppm * range_front)),
        0,
        180,
        360,
        (50, 50, 50),
        1,
    )
    cv2.ellipse(
        img,
        (origin_x, origin_y),
        (int(ppm * range_sides), int(ppm * range_sides)),
        0,
        0,
        180,
        (50, 50, 50),
        1,
    )

    # Route points (blue)
    if "route_original" in data:
        route = data["route_original"]
        if isinstance(route, torch.Tensor):
            route = route.cpu().numpy()
        for rp in route[index]:
            px = int(rp[1] * ppm + origin_x)
            py = int(-rp[0] * ppm + origin_y)
            cv2.circle(img, (px, py), 3, (255, 0, 0), -1)

    # Object tokens
    if "idxs" in data and "x_objs" in data:
        batch_idxs = data["idxs"]
        x_objs = data["x_objs"]
        if isinstance(batch_idxs, torch.Tensor):
            batch_idxs = batch_idxs.cpu().numpy()
        if isinstance(x_objs, torch.Tensor):
            x_objs = x_objs.cpu().numpy()

        objs = x_objs[batch_idxs[index]]
        for obj in objs:
            cls_id = int(obj[0])
            if cls_id == 0:  # padding
                continue
            yaw = obj[3]
            speed = obj[4] / 3.6  # km/h -> m/s
            half_w = obj[5] * ppm / 2
            half_h = obj[6] * ppm / 2

            px = obj[2] * ppm + origin_x
            py = -obj[1] * ppm + origin_y

            fill, outline = _OBJECT_COLORS.get(cls_id, _DEFAULT_COLOR)

            if half_w < 1:
                half_w = 3
                half_h = 3

            corners = _bbox_corners(px, py, yaw, half_w, half_h)
            overlay = np.zeros_like(img)
            cv2.fillPoly(overlay, [corners], fill)
            cv2.drawContours(overlay, [corners], 0, outline, 1)

            # Speed indicator line
            if cls_id not in (4, 5):  # not stop sign / traffic light
                angle_rad = math.radians(yaw - 90)
                end_px = int(px + speed * ppm * math.cos(angle_rad))
                end_py = int(py + speed * ppm * math.sin(angle_rad))
                cv2.line(overlay, (int(px), int(py)), (end_px, end_py), outline, 1)

            img = cv2.add(img, overlay)

    # Ground-truth waypoints (green)
    if "future_waypoints" in data:
        gt_wps = data["future_waypoints"]
        if isinstance(gt_wps, torch.Tensor):
            gt_wps = gt_wps.cpu().numpy()
        for wp in gt_wps[index]:
            px = int(wp[1] * ppm + origin_x)
            py = int(-wp[0] * ppm + origin_y)
            cv2.circle(img, (px, py), 3, (0, 255, 0), -1)

    # Predicted waypoints (red)
    if predictions.pred_future_waypoints is not None:
        pred_wps = predictions.pred_future_waypoints
        if isinstance(pred_wps, torch.Tensor):
            pred_wps = pred_wps.detach().cpu().numpy()
        for wp in pred_wps[index]:
            px = int(wp[1] * ppm + origin_x)
            py = int(-wp[0] * ppm + origin_y)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

    # Predicted route (orange)
    if predictions.pred_route is not None:
        pred_route = predictions.pred_route
        if isinstance(pred_route, torch.Tensor):
            pred_route = pred_route.detach().cpu().numpy()
        for rp in pred_route[index]:
            px = int(rp[1] * ppm + origin_x)
            py = int(-rp[0] * ppm + origin_y)
            cv2.circle(img, (px, py), 2, (0, 165, 255), -1)

    return img


@beartype
def visualize_plant_sample(
    config: PlantConfig,
    predictions: Prediction,
    data: dict,
    prefix: str = "train",
    log_wandb: bool = False,
    save_image: bool = False,
    save_path: str | None = None,
    postfix: str | None = None,
) -> None:
    """Visualize a PlanT training sample, matching the ``visualize_sample`` API.

    Args:
        config: PlanT training configuration.
        predictions: Model predictions for the current batch.
        data: Batch dict from the PlanT dataloader.
        prefix: Name prefix for logged images.
        log_wandb: If True, log images to Weights & Biases.
        save_image: If True, save images to disk.
        save_path: Directory to save images to.
        postfix: Filename postfix for saved images.
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    if not (log_wandb or save_image):
        return

    with torch.no_grad():
        bev = visualize_plant_bev(data=data, predictions=predictions, index=0)
        bev_rgb = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)

        if log_wandb and config.log_wandb:
            wandb.log(
                {f"viz/{prefix}_plant_bev": wandb.Image(bev_rgb)},
                commit=False,
            )
        if save_image:
            cv2.imwrite(f"{save_path}/plant_bev_{postfix}.png", bev)
