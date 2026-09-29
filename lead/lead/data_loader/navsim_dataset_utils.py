from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype

import lead.common.common_utils as common_utils
from lead.common.constants import (
    TransfuserBoundingBoxIndex,
)
from lead.tfv6 import center_net_decoder as g_t
from lead.training.config_training import TrainingConfig


@beartype
def get_centernet_labels(
    gt_bboxes: jt.Float[npt.NDArray, "N 8"],
    config: TrainingConfig,
    num_bb_classes: int,
) -> dict[str, npt.NDArray]:
    """
    Compute regression and classification targets for bounding boxes.

    Args:
        gt_bboxes: Ground truth bboxes for each image with shape (N, 8). Coordinates in image frame.
        config: TrainingConfig object containing configuration parameters.
        num_bb_classes: Number of bounding box classes.
    Returns:
        A dictionary containing various target tensors for training the CenterNet model.
    """
    feat_h = config.lidar_height_meter
    feat_w = config.lidar_width_meter

    center_heatmap_target = np.zeros([num_bb_classes, feat_h, feat_w], dtype=np.float32)
    wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    pixel_weight = np.zeros(
        [2, feat_h, feat_w],
        dtype=np.float32,
    )  # 2 is the max of the channels above here.

    if not gt_bboxes.shape[0] > 0:
        return {
            "center_net_bounding_boxes": gt_bboxes,
            "center_net_heatmap": center_heatmap_target,
            "center_net_wh": wh_target,
            "center_net_yaw_class": yaw_class_target.squeeze(0),
            "center_net_yaw_res": yaw_res_target,
            "center_net_offset": offset_target,
            "center_net_velocity": velocity_target,
            "center_net_brake": brake_target.squeeze(0),
            "center_net_pixel_weight": pixel_weight,
            "center_net_avg_factor": np.array([1]),
        }

    center_x = (
        gt_bboxes[:, [TransfuserBoundingBoxIndex.X]] / config.bev_down_sample_factor
    )
    center_y = (
        gt_bboxes[:, [TransfuserBoundingBoxIndex.Y]] / config.bev_down_sample_factor
    )
    gt_centers = np.concatenate((center_x, center_y), axis=1)

    for j, ct in enumerate(gt_centers):
        ctx_int, cty_int = ct.astype(int)
        ctx, cty = ct
        if ctx_int < 0 or ctx_int >= feat_w or cty_int < 0 or cty_int >= feat_h:
            print(
                f"Be cautious! Bounding box center {ct} is out of bounds for image size ({feat_h}, {feat_w}).",
                flush=True,
            )
            continue

        extent_x = (
            gt_bboxes[j, TransfuserBoundingBoxIndex.W] / config.bev_down_sample_factor
        )
        extent_y = (
            gt_bboxes[j, TransfuserBoundingBoxIndex.H] / config.bev_down_sample_factor
        )

        radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
        radius = max(2, int(radius))
        ind = 0  # NavSim has only one class for now.

        g_t.gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

        wh_target[0, cty_int, ctx_int] = extent_x
        wh_target[1, cty_int, ctx_int] = extent_y

        yaw_class, yaw_res = common_utils.angle2class(
            gt_bboxes[j, TransfuserBoundingBoxIndex.YAW],
            config.num_dir_bins,
        )

        yaw_class_target[0, cty_int, ctx_int] = yaw_class
        yaw_res_target[0, cty_int, ctx_int] = yaw_res

        velocity_target[0, cty_int, ctx_int] = (
            0  # NavSim does not provide velocity information.
        )
        # Brakes can potentially be continuous but we classify them now.
        # Using mathematical rounding the split is applied at 0.5
        brake_target[0, cty_int, ctx_int] = (
            0  # NavSim does not provide brake information.
        )

        offset_target[0, cty_int, ctx_int] = ctx - ctx_int
        offset_target[1, cty_int, ctx_int] = cty - cty_int
        # All pixels with a bounding box have a weight of 1 all others have a weight of 0.
        # Used to ignore the pixels without bbs in the loss.
        pixel_weight[:, cty_int, ctx_int] = 1.0

    avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
    return {
        "center_net_bounding_boxes": gt_bboxes,
        "center_net_heatmap": center_heatmap_target,
        "center_net_wh": wh_target,
        "center_net_yaw_class": yaw_class_target.squeeze(0),
        "center_net_yaw_res": yaw_res_target,
        "center_net_offset": offset_target,
        "center_net_velocity": velocity_target,
        "center_net_brake": brake_target.squeeze(0),
        "center_net_pixel_weight": pixel_weight,
        "center_net_avg_factor": avg_factor,
    }
