import io
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from beartype import beartype
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

import lead.common.common_utils as common_utils
import lead.common.constants as constants
from lead.common.constants import (
    CARLA_NAVIGATION_COMMAND_STR_MAP,
    RadarLabels,
    TransfuserBoundingBoxIndex,
)
from lead.data_loader import carla_dataset_utils
from lead.inference.closed_loop_inference import ClosedLoopPrediction
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.open_loop_inference import OpenLoopPrediction
from lead.tfv6.center_net_decoder import PredictedBoundingBox
from lead.tfv6.tfv6 import Prediction
from lead.training.config_training import TrainingConfig
from lead.visualization import viz_utils


class Visualizer:
    """Visualizer for LEAD training and inference results. This class provides methods to visualize various aspects
    of the LEAD model's groundtruths and predictions
    """

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        data: dict,
        prediction: Prediction | ClosedLoopPrediction | OpenLoopPrediction | None,
        training: bool = False,
        config_test_time: ClosedLoopConfig | None = None,
        test_time: bool = False,
    ):
        """
        Initialize the Visualizer.

        Args:
            config: Training configuration object.
            data: Dictionary containing input data tensors.
            prediction: Prediction object containing model outputs.
            training: Boolean indicating if in training mode.
            config_test_time: Configuration for test-time settings.
            test_time: Boolean indicating if in test-time mode.
        """
        self.test_time = test_time
        self.training = training
        self.config = config
        self.data = data
        self.predictions = prediction
        self.config_test_time = config_test_time
        if self.config_test_time is None:
            self.config_test_time = ClosedLoopConfig()

        self.scale_factor = 4
        # Initialize variables for visualization
        self.rasterized_lidar = data.get("rasterized_lidar")

        self.size_width = int(
            (self.config.max_y_meter - self.config.min_y_meter)
            * self.config.pixels_per_meter,
        )
        self.size_height = int(
            (self.config.max_x_meter - self.config.min_x_meter)
            * self.config.pixels_per_meter,
        )
        self.origin = (
            (self.size_height * self.scale_factor)
            // (
                (self.config.max_x_meter - self.config.min_x_meter)
                / max((-self.config.min_x_meter), 1)
            ),
            (self.size_width * self.scale_factor) // 2,
        )
        self.loc_pixels_per_meter = self.config.pixels_per_meter * self.scale_factor

        start_color = np.array([255, 255, 255], dtype=np.float32)
        end_color = np.array(constants.LIDAR_COLOR, dtype=np.float32)

        bev = self.rasterized_lidar.detach().cpu().numpy()[0][0]
        bev = bev / (bev.max() + 1e-6)
        bev = bev.astype(np.float32)

        bev_img = np.zeros((*bev.shape, 3), dtype=np.float32)
        for c in range(3):
            bev_img[..., c] = start_color[c] + (end_color[c] - start_color[c]) * bev
        bev_img = bev_img.astype(np.uint8)

        self.bev_image = cv2.resize(
            bev_img,
            dsize=(
                bev_img.shape[1] * self.scale_factor,
                bev_img.shape[0] * self.scale_factor,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        self.meta_panel = 255 * np.ones((639, 1492, 3), dtype=np.uint8)
        if self.test_time:
            self.meta_panel = 255 * np.ones((115, 1492, 3), dtype=np.uint8)

    def visualize_training_labels(self):
        """Visualize groundtruth labels during training."""
        # Perspectives
        self._process_all_perspectives()

        # BEV
        self._bev_semantic(ground_truth=True)
        self._route()
        self._future_waypoints()
        self._ego_bounding_box()
        self._bounding_boxes()
        self._target_point()
        self._radars(plot_detection_label=True, plot_detection_prediction=False)
        self.bev_image = np.rot90(self.bev_image, k=1)
        self.bev_image = np.ascontiguousarray(self.bev_image)

        # Text
        self._meta()

        # Concatenate
        return self._concatenate_all_perspectives_and_bev()

    def visualize_training_prediction(self):
        """Visualize model predictions during training."""
        # Perspective
        self._process_all_perspectives(training=True)

        # BEV
        self._bev_semantic(ground_truth=False)
        self._target_point()
        if self.config.use_planning_decoder:
            self._pred_future_waypoints()
            self._pred_route()
        self._ego_bounding_box()
        self._pred_bounding_box()
        self._target_point()
        self._radars(plot_detection_label=False, plot_detection_prediction=True)
        self.bev_image = np.rot90(self.bev_image, k=1)
        self.bev_image = np.ascontiguousarray(self.bev_image)

        # Text
        self._meta()

        # Concatenate
        return self._concatenate_all_perspectives_and_bev()

    def visualize_inference_prediction(self):
        """Visualize model predictions during inference."""
        # Perspective
        self._process_all_perspectives(training=True)

        # BEV
        self._bev_semantic(ground_truth=False)
        self._pred_route()
        # self._pred_future_waypoints()
        self._target_point()
        self._radars(plot_detection_label=False, plot_detection_prediction=True)
        self._ego_bounding_box()
        self._pred_bounding_box()
        self.bev_image = np.rot90(self.bev_image, k=1)
        self.bev_image = np.ascontiguousarray(self.bev_image)

        # Text
        self._meta()

        return self._concatenate_all_perspectives_and_bev()

    def _radars(self, plot_detection_label: bool, plot_detection_prediction: bool):
        if not self.config.use_radars:
            return

        ppm = self.loc_pixels_per_meter
        origin = self.origin
        bev = self.bev_image

        vmax = 20.0  # m/s for scaling radius
        min_r, max_r = 1.0, 20  # px

        # Visualize raw radars
        for i in range(1, self.config.num_radar_sensors + 1):
            radar_i = self.data.get(f"radar{i}")
            if radar_i is None:
                continue

            arr = radar_i[0]

            mask = (arr[:, :3] != 0).any(axis=1)  # drop zero-padded
            pts = arr[mask]
            if pts.shape[0] == 0:
                continue

            x = pts[:, 0]
            y = pts[:, 1]
            v = np.abs(np.nan_to_num(pts[:, 3], nan=0.0))

            for xm, ym, vk in zip(x, y, v, strict=False):
                px = int(origin[0] + xm * ppm)
                py = int(origin[1] + ym * ppm)

                rpx = int(
                    np.clip(min_r + (abs(vk) / vmax) * (max_r - min_r), min_r, max_r),
                )
                filled = vk > 0  # filled = approaching, outline = receding

                viz_utils.draw_gaussian_blob(
                    bev,
                    px,
                    py,
                    rpx,
                    constants.RADAR_COLOR,
                    filled,
                )

        # Visualize radar detections labels
        if plot_detection_label:
            radar_detections = self.data.get("radar_detections")
            radar_detection_waypoints = self.data.get("radar_detection_waypoints")
            radar_detection_num_waypoints = self.data.get(
                "radar_detection_num_waypoints",
            )

            if radar_detections is not None:
                radar_detections = radar_detections[0].cpu().numpy()
                valid_mask = radar_detections[:, RadarLabels.VALID].astype(bool)
                valid_detections = radar_detections[valid_mask]

                if valid_detections.shape[0] > 0:
                    x = valid_detections[:, 0]  # x coordinates
                    y = valid_detections[:, 1]  # y coordinates
                    v = np.abs(
                        np.nan_to_num(valid_detections[:, 2], nan=0.0),
                    )  # velocity magnitude

                    for xm, ym, vk in zip(x, y, v, strict=False):
                        px = int(origin[0] + xm * ppm)
                        py = int(origin[1] + ym * ppm)

                        rpx = 1 + int(
                            np.clip(
                                min_r + (abs(vk) / vmax) * (max_r - min_r),
                                min_r,
                                max_r,
                            ),
                        )
                        filled = vk > 0  # filled = approaching, outline = receding
                        viz_utils.draw_gaussian_blob(
                            bev,
                            px,
                            py,
                            rpx,
                            constants.RADAR_DETECTION_COLOR,
                            filled,
                        )

                # Draw waypoints for valid detections
                if (
                    radar_detection_waypoints is not None
                    and radar_detection_num_waypoints is not None
                ):
                    radar_detection_waypoints = (
                        radar_detection_waypoints[0].cpu().numpy()
                    )
                    radar_detection_num_waypoints = (
                        radar_detection_num_waypoints[0].cpu().numpy()
                    )

                    # Get waypoints only for valid detections
                    valid_waypoints = radar_detection_waypoints[valid_mask]
                    valid_num_waypoints = radar_detection_num_waypoints[valid_mask]

                    for waypoints, num_wps in zip(
                        valid_waypoints,
                        valid_num_waypoints,
                        strict=False,
                    ):
                        if num_wps > 0:
                            # Draw waypoints as small circles
                            for i in range(int(num_wps)):
                                wp_x = waypoints[i, 0]
                                wp_y = waypoints[i, 1]

                                wp_px = int(origin[0] + wp_x * ppm)
                                wp_py = int(origin[1] + wp_y * ppm)

                                # Draw small circles for waypoints
                                cv2.circle(
                                    bev,
                                    (wp_px, wp_py),
                                    radius=3,
                                    color=constants.RADAR_DETECTION_COLOR,
                                    thickness=-1,
                                )

                            # Draw lines connecting waypoints
                            for i in range(int(num_wps) - 1):
                                wp1_x = waypoints[i, 0]
                                wp1_y = waypoints[i, 1]
                                wp2_x = waypoints[i + 1, 0]
                                wp2_y = waypoints[i + 1, 1]

                                wp1_px = int(origin[0] + wp1_x * ppm)
                                wp1_py = int(origin[1] + wp1_y * ppm)
                                wp2_px = int(origin[0] + wp2_x * ppm)
                                wp2_py = int(origin[1] + wp2_y * ppm)

                                cv2.line(
                                    bev,
                                    (wp1_px, wp1_py),
                                    (wp2_px, wp2_py),
                                    color=constants.RADAR_DETECTION_COLOR,
                                    thickness=1,
                                    lineType=cv2.LINE_AA,
                                )

        # Visualize radar detection predictions
        if (
            plot_detection_prediction
            and self.predictions is not None
            and self.predictions.pred_radar_predictions is not None
        ):
            radar_predictions = self.predictions.pred_radar_predictions[0].cpu()
            valid_mask = torch.sigmoid(radar_predictions[:, RadarLabels.VALID]) > 0.5
            valid_predictions = radar_predictions[valid_mask].detach().float().numpy()

            if valid_predictions.shape[0] > 0:
                x = valid_predictions[:, 0]  # x coordinates
                y = valid_predictions[:, 1]  # y coordinates
                v = np.abs(
                    np.nan_to_num(valid_predictions[:, 2], nan=0.0),
                )  # velocity magnitude

                for xm, ym, vk in zip(x, y, v, strict=False):
                    px = int(origin[0] + xm * ppm)
                    py = int(origin[1] + ym * ppm)

                    rpx = 1 + int(
                        np.clip(
                            min_r + (abs(vk) / vmax) * (max_r - min_r),
                            min_r,
                            max_r,
                        ),
                    )
                    filled = vk > 0  # filled = approaching, outline = receding
                    viz_utils.draw_gaussian_blob(
                        bev,
                        px,
                        py,
                        rpx,
                        constants.RADAR_DETECTION_COLOR,
                        filled,
                    )

    def _process_all_perspectives(self, training=False):
        self.perspectives = {}
        for perspective_modality in ["rgb", "semantic"]:  # , "depth"]:
            if not training:
                perspective = self.data.get(perspective_modality)
            else:
                perspective = {
                    "depth": self.predictions.pred_depth,
                    "semantic": self.predictions.pred_semantic,
                    "rgb": self.data.get("rgb"),
                }[perspective_modality]
            if perspective is None:
                continue
            perspective = perspective[0]
            if perspective_modality == "depth":
                perspective = perspective.unsqueeze(0)
                perspective = torch.clamp(perspective.repeat(3, 1, 1), min=0, max=255)
            elif perspective_modality == "semantic":
                if training:
                    perspective = perspective.argmax(dim=0, keepdim=True)
                else:
                    perspective = perspective.unsqueeze(0)

            perspective_image = (
                perspective.permute(1, 2, 0)
                .detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.uint8)
            )
            perspective_image = np.ascontiguousarray(perspective_image)

            if perspective_modality == "semantic":
                converter = np.uint8(
                    list(constants.TRANSFUSER_SEMANTIC_COLORS.values()),
                )
                perspective_image = converter[perspective_image[..., 0]]
            if perspective_modality == "depth":
                image = perspective_image[..., 0].astype(np.float32)
                image = np.clip(image, 1e-3, None)
                log_image = np.log(image)
                log_image -= log_image.min()
                log_image /= log_image.max() + 1e-6
                log_image = (log_image * 255).astype(np.uint8)
                perspective_image = cv2.applyColorMap(log_image, cv2.COLORMAP_WINTER)

            # Draw target speed prediction on RGB image
            if (
                perspective_modality == "rgb"
                and training
                and self.predictions is not None
            ):
                if (
                    hasattr(self.predictions, "pred_target_speed_scalar")
                    and self.predictions.pred_target_speed_scalar is not None
                ):
                    pred_speed = float(
                        self.predictions.pred_target_speed_scalar.detach()
                        .cpu()
                        .float()[0],
                    )

                    # Position: lower middle of the image
                    img_height, img_width = perspective_image.shape[:2]
                    text = f"Target Speed: {pred_speed:.1f} m/s"

                    # Use PIL for better text rendering
                    img_pil = Image.fromarray(perspective_image)
                    draw = ImageDraw.Draw(img_pil)

                    try:
                        font = ImageFont.truetype("3rd_party/Roboto-Bold.ttf", 20)
                    except:
                        font = ImageFont.load_default()

                    # Get text size
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Position at lower middle
                    x = (img_width - text_width) // 2
                    y = img_height - text_height - 20

                    # Draw background rectangle for better visibility
                    padding_horizontal = 6
                    padding_top = 3
                    padding_bottom = 6
                    draw.rectangle(
                        [
                            (x - padding_horizontal, y - padding_top),
                            (
                                x + text_width + padding_horizontal,
                                y + text_height + padding_bottom,
                            ),
                        ],
                        fill=(0, 0, 0, 180),
                    )

                    # Draw text in white
                    draw.text((x, y), text, font=font, fill=(255, 255, 255))

                    perspective_image = np.array(img_pil)

            self.perspectives[perspective_modality] = perspective_image

    def _concatenate_all_perspectives_and_bev(
        self,
        border_size: int = 10,
        border_color: tuple = (255, 255, 255),
    ) -> np.ndarray:
        """
        Concatenate all perspectives vertically, then add BEV on the left side with white borders.

        Args:
            border_size: Size of the white border between components in pixels
            border_color: Color of the border as (B, G, R) for OpenCV

        Returns:
            Concatenated image as a numpy array.
        """
        lidar_image = np.ascontiguousarray(self.bev_image, dtype=np.uint8)

        if not self.perspectives:
            raise ValueError("No perspectives available")

        # Get all available perspectives and resize them to same width
        perspective_images = []
        target_width = lidar_image.shape[1]  # Use BEV width as reference

        for modality in ["rgb", "semantic"]:  # , "depth"]:  # Define order
            if modality in self.perspectives:
                img = np.ascontiguousarray(self.perspectives[modality], dtype=np.uint8)
                # Resize to match target width
                target_height = int(img.shape[0] * (target_width / img.shape[1]))
                img_resized = cv2.resize(img, (target_width, target_height))
                perspective_images.append(img_resized)

        if not perspective_images:
            raise ValueError("No valid perspective images found")

        # Add borders between perspective images and concatenate vertically
        bordered_perspectives = []
        for i, img in enumerate(perspective_images):
            # Add top border (except for first image)
            if i > 0:
                top_border = np.full(
                    (
                        border_size,
                        img.shape[1],
                        img.shape[2] if len(img.shape) == 3 else 1,
                    ),
                    border_color,
                    dtype=np.uint8,
                )
                if len(img.shape) == 2:  # Grayscale
                    top_border = top_border.squeeze(-1)
                bordered_perspectives.append(top_border)

            bordered_perspectives.append(img)

        # Concatenate all perspectives vertically
        stacked_perspectives = np.concatenate(bordered_perspectives, axis=0)

        self.stacked_perspectives = stacked_perspectives

        # Add only right border to perspectives (left border will be shared with BEV)
        if len(stacked_perspectives.shape) == 3:
            right_border = np.full(
                (
                    stacked_perspectives.shape[0],
                    border_size,
                    stacked_perspectives.shape[2],
                ),
                border_color,
                dtype=np.uint8,
            )
        else:  # Grayscale
            right_border = np.full(
                (stacked_perspectives.shape[0], border_size),
                border_color[0],
                dtype=np.uint8,
            )

        stacked_perspectives = np.concatenate(
            (stacked_perspectives, right_border),
            axis=1,
        )

        # Resize BEV to match the total height of stacked perspectives
        target_height = stacked_perspectives.shape[0]
        bev_width = int(lidar_image.shape[1] * (target_height / lidar_image.shape[0]))
        lidar_resized = cv2.resize(lidar_image, (bev_width, target_height))

        # Add only left border to BEV (right border will be shared with perspectives)
        if len(lidar_resized.shape) == 3:
            bev_left_border = np.full(
                (lidar_resized.shape[0], border_size, lidar_resized.shape[2]),
                border_color,
                dtype=np.uint8,
            )
        else:  # Grayscale
            bev_left_border = np.full(
                (lidar_resized.shape[0], border_size),
                border_color[0],
                dtype=np.uint8,
            )

        lidar_bordered = np.concatenate((bev_left_border, lidar_resized), axis=1)

        # Add shared border between BEV and perspectives
        if len(lidar_bordered.shape) == 3:
            shared_border = np.full(
                (lidar_bordered.shape[0], border_size, lidar_bordered.shape[2]),
                border_color,
                dtype=np.uint8,
            )
        else:  # Grayscale
            shared_border = np.full(
                (lidar_bordered.shape[0], border_size),
                border_color[0],
                dtype=np.uint8,
            )

        # Concatenate BEV on the left, shared border, then perspectives on the right
        ret = np.concatenate(
            (lidar_bordered, shared_border, stacked_perspectives),
            axis=1,
        )

        # Add third person view if available
        if self.config.load_bev_3rd_person_images:
            third_person_image = self.data.get("bev_3rd_person")
            if third_person_image is not None:
                third_person_image = third_person_image[0].detach().cpu().numpy()
                third_person_image = np.ascontiguousarray(
                    third_person_image,
                    dtype=np.uint8,
                )

                target_height = ret.shape[0]
                target_width = int(
                    third_person_image.shape[1]
                    * (target_height / third_person_image.shape[0]),
                )
                third_person_image = cv2.resize(
                    third_person_image,
                    (target_width, target_height),
                )

                # Add borders to third person view
                if len(third_person_image.shape) == 3:
                    tp_left_border = np.full(
                        (
                            third_person_image.shape[0],
                            border_size,
                            third_person_image.shape[2],
                        ),
                        border_color,
                        dtype=np.uint8,
                    )
                    tp_right_border = np.full(
                        (
                            third_person_image.shape[0],
                            border_size,
                            third_person_image.shape[2],
                        ),
                        border_color,
                        dtype=np.uint8,
                    )
                else:  # Grayscale
                    tp_left_border = np.full(
                        (third_person_image.shape[0], border_size),
                        border_color[0],
                        dtype=np.uint8,
                    )
                    tp_right_border = np.full(
                        (third_person_image.shape[0], border_size),
                        border_color[0],
                        dtype=np.uint8,
                    )

                third_person_bordered = np.concatenate(
                    (tp_left_border, third_person_image, tp_right_border),
                    axis=1,
                )
                ret = np.concatenate((ret, third_person_bordered), axis=1)

        # Add top and bottom borders to the entire composition
        if len(ret.shape) == 3:
            top_border = np.full(
                (border_size, ret.shape[1], ret.shape[2]),
                border_color,
                dtype=np.uint8,
            )
            bottom_border = np.full(
                (border_size, ret.shape[1], ret.shape[2]),
                border_color,
                dtype=np.uint8,
            )
        else:  # Grayscale
            top_border = np.full(
                (border_size, ret.shape[1]),
                border_color[0],
                dtype=np.uint8,
            )
            bottom_border = np.full(
                (border_size, ret.shape[1]),
                border_color[0],
                dtype=np.uint8,
            )

        ret = np.concatenate((top_border, ret, bottom_border), axis=0)

        # Concatenate meta panel at the bottom (commented out in original)
        meta_panel = np.ascontiguousarray(self.meta_panel, dtype=np.uint8)
        target_width = ret.shape[1]
        target_height = int(meta_panel.shape[0] * (target_width / meta_panel.shape[1]))
        meta_panel_resized = cv2.resize(meta_panel, (target_width, target_height))

        # Add border above meta panel
        if len(meta_panel_resized.shape) == 3:
            meta_top_border = np.full(
                (border_size, meta_panel_resized.shape[1], meta_panel_resized.shape[2]),
                border_color,
                dtype=np.uint8,
            )
        else:
            meta_top_border = np.full(
                (border_size, meta_panel_resized.shape[1]),
                border_color[0],
                dtype=np.uint8,
            )

        ret = np.concatenate((ret, meta_top_border, meta_panel_resized), axis=0)

        return ret

    def _bev_semantic(self, ground_truth=False):
        # Get data
        if ground_truth:
            bev_semantic = self.data.get("bev_semantic")
            if bev_semantic is not None:
                bev_semantic = bev_semantic[0].detach().cpu().float().numpy()
                bev_semantic = bev_semantic.astype(np.int32)
        else:
            bev_semantic = self.predictions.pred_bev_semantic
            if bev_semantic is not None:
                bev_semantic = bev_semantic.argmax(dim=1)
                bev_semantic = bev_semantic[0].detach().cpu().numpy()

        # Visualization
        if bev_semantic is not None:
            converter = np.array(
                list(constants.CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER.values()),
            )
            converter[1][0:3] = 40
            bev_semantic_image = converter[bev_semantic, ...].astype("uint8")
            alpha = np.ones_like(bev_semantic) * 0.33
            alpha = alpha.astype(np.float32)
            alpha[bev_semantic == 0] = 0.0
            alpha[bev_semantic == 1] = 0.15

            alpha = cv2.resize(
                alpha,
                dsize=(
                    alpha.shape[1] * self.scale_factor,
                    alpha.shape[0] * self.scale_factor,
                ),
                interpolation=cv2.INTER_NEAREST,
            )
            alpha = np.expand_dims(alpha, 2)
            bev_semantic_image = cv2.resize(
                bev_semantic_image,
                dsize=(self.bev_image.shape[1], self.bev_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            self.bev_image = bev_semantic_image * alpha + (1 - alpha) * self.bev_image

    def _target_point(self):
        target_point = self.data.get("target_point")
        target_point_next = self.data.get("target_point_next")
        target_point_previous = self.data.get("target_point_previous")

        if target_point_previous is not None:
            x_tp = (
                target_point_previous[0][0] * self.loc_pixels_per_meter + self.origin[0]
            )
            y_tp = (
                target_point_previous[0][1] * self.loc_pixels_per_meter + self.origin[1]
            )
            viz_utils.draw_circle_with_number(
                self.bev_image,
                int(x_tp),
                int(y_tp),
                constants.TP_DEFAULT_COLOR,
                radius=14,
                number=0,
            )

        if target_point_next is not None:
            x_tp = target_point_next[0][0] * self.loc_pixels_per_meter + self.origin[0]
            y_tp = target_point_next[0][1] * self.loc_pixels_per_meter + self.origin[1]
            viz_utils.draw_circle_with_number(
                self.bev_image,
                int(x_tp),
                int(y_tp),
                constants.TP_DEFAULT_COLOR,
                radius=14,
                number=2,
            )

        if target_point is not None:
            x_tp = target_point[0][0] * self.loc_pixels_per_meter + self.origin[0]
            y_tp = target_point[0][1] * self.loc_pixels_per_meter + self.origin[1]
            viz_utils.draw_circle_with_number(
                self.bev_image,
                int(x_tp),
                int(y_tp),
                constants.TP_DEFAULT_COLOR,
                radius=18,
                number=1,
            )

    @beartype
    def _draw_points_in_perspective(
        self,
        points: NDArray,
        size: int,
        connected: bool,
        color: tuple[int, int, int],
    ):
        if not (len(points) > 0 and points.shape[0] > 0):
            return
        # Image
        rgb = self.perspectives["rgb"]

        # Get camera configuration dynamically
        camera_calibration = self.config.camera_calibration
        camera_rots = []
        camera_poses = []
        camera_fovs = []
        camera_widths = []

        for i in range(1, min(4, self.config.num_used_cameras + 1)):
            camera_config = camera_calibration[i]
            camera_rots.append(camera_config["rot"])
            camera_poses.append(camera_config["pos"])
            camera_fovs.append(camera_config["fov"])
            camera_widths.append(camera_config["width"])

        # Reverse rotation and position lists to match original behavior
        camera_rots = camera_rots[:3][::-1]
        camera_poses = camera_poses[:3][::-1]
        camera_height = camera_calibration[1]["height"]

        # Project and draw points for each camera
        x_offset = 0
        for i in range(3):
            # Project points to current camera
            projected_points, inside_image = common_utils.project_points_to_image(
                camera_rots[i],
                camera_poses[i],
                camera_fovs[i],
                camera_widths[i],
                camera_height,
                points,
            )

            if connected:
                # Draw lines between all consecutive points (cv2 handles clipping)
                for j in range(len(projected_points) - 1):
                    if not inside_image[j + 1]:
                        continue
                    x1, y1 = projected_points[j]
                    x2, y2 = projected_points[j + 1]

                    # Adjust x coordinates for horizontal concatenation
                    img_x1 = int(x1 + x_offset)
                    img_y1 = int(y1)
                    img_x2 = int(x2 + x_offset)
                    img_y2 = int(y2)

                    cv2.line(rgb, (img_x1, img_y1), (img_x2, img_y2), color, size)
            else:
                # Draw points (circles) only for visible points
                for (x, y), inside in zip(projected_points, inside_image, strict=False):
                    if inside:
                        # Adjust x coordinate for horizontal concatenation
                        img_x = int(x + x_offset)
                        img_y = int(y)

                        # Bounds check
                        if 0 <= img_x < rgb.shape[1] and 0 <= img_y < rgb.shape[0]:
                            cv2.circle(rgb, (img_x, img_y), size, color, -1)

            # Move to next camera's horizontal position
            x_offset += camera_widths[i]

    def _route(self):
        route = self.data.get("route")
        if route is not None and (
            self.config.use_planning_decoder or self.config.visualize_dataset
        ):
            wps = route.detach().cpu().numpy()[0]

            for i in range(len(wps) - 1):
                x1 = int(wps[i][0] * self.loc_pixels_per_meter + self.origin[0])
                y1 = int(wps[i][1] * self.loc_pixels_per_meter + self.origin[1])
                x2 = int(wps[i + 1][0] * self.loc_pixels_per_meter + self.origin[0])
                y2 = int(wps[i + 1][1] * self.loc_pixels_per_meter + self.origin[1])

                color = viz_utils.lighter_shade(
                    constants.PREDICTION_ROUTE_COLOR,
                    i,
                    len(wps),
                )
                cv2.line(
                    self.bev_image,
                    (x1, y1),
                    (x2, y2),
                    color=color,
                    thickness=constants.PREDICTION_ROUTE_RADIUS,
                    lineType=cv2.LINE_AA,
                )
            # self._draw_points_in_perspective(points=wps,
            #                                  size=3,
            #                                  connected=True,
            #                                  color=constants.PREDICTION_ROUTE_COLOR)

    def _future_waypoints(self):
        if self.config.use_planning_decoder or self.config.visualize_dataset:
            future_waypoints = self.data.get("future_waypoints")
            # future_yaws = self.data.get("future_yaws")
            if future_waypoints is not None and future_waypoints[0] is not None:
                wps = future_waypoints.detach().cpu().numpy()[0]
                if len(wps) > 0:
                    for i, wp in enumerate(wps):
                        wp_x = wp[0] * self.loc_pixels_per_meter + self.origin[0]
                        wp_y = wp[1] * self.loc_pixels_per_meter + self.origin[1]
                        color = viz_utils.lighter_shade(
                            constants.GROUNDTRUTH_FUTURE_WAYPOINT_COLOR,
                            i,
                            len(wps),
                        )
                        cv2.circle(
                            self.bev_image,
                            (int(wp_x), int(wp_y)),
                            radius=constants.PREDICTION_WAYPOINT_RADIUS,
                            color=color,
                            thickness=-1,
                        )
                # self._draw_points_in_perspective(points=wps,
                #                                  size=5,
                #                                  connected=False,
                #                                  color=constants.PREDICTION_WAYPOINT_COLOR)

            past_waypoints = self.data.get("past_waypoints")
            # past_yaws = self.data.get("past_yaws")
            if past_waypoints is not None and past_waypoints[0] is not None:
                wps = past_waypoints.detach().cpu().numpy()[0]
                for i, wp in enumerate(wps):
                    wp_x = wp[0] * self.loc_pixels_per_meter + self.origin[0]
                    wp_y = wp[1] * self.loc_pixels_per_meter + self.origin[1]
                    color = viz_utils.lighter_shade(
                        constants.GROUND_TRUTH_PAST_WAYPOINT_COLOR,
                        i,
                        len(wps),
                    )
                    cv2.circle(
                        self.bev_image,
                        (int(wp_x), int(wp_y)),
                        radius=constants.PREDICTION_WAYPOINT_RADIUS,
                        color=color,
                        thickness=-1,
                    )

    def _pred_route(self):
        pred_route = self.predictions.pred_route
        if pred_route is not None:
            wps = pred_route.detach().cpu().float().numpy()[0]
            for i in range(len(wps) - 1):
                x1 = int(wps[i][0] * self.loc_pixels_per_meter + self.origin[0])
                y1 = int(wps[i][1] * self.loc_pixels_per_meter + self.origin[1])
                color = viz_utils.lighter_shade(
                    constants.PREDICTION_ROUTE_COLOR,
                    i,
                    len(wps),
                )
                cv2.circle(
                    self.bev_image,
                    (x1, y1),
                    radius=constants.PREDICTION_ROUTE_RADIUS,
                    color=color,
                    thickness=-1,
                )
            # self._draw_points_in_perspective(points=wps, size=3, connected=True, color=constants.PREDICTION_ROUTE_COLOR)

    def _pred_future_waypoints(self):
        pred_waypoints = self.predictions.pred_future_waypoints
        if pred_waypoints is not None:
            wps = pred_waypoints.detach().cpu().float().numpy()[0]
            for i, wp in enumerate(wps):
                wp_x = wp[0] * self.loc_pixels_per_meter + self.origin[0]
                wp_y = wp[1] * self.loc_pixels_per_meter + self.origin[1]
                color = viz_utils.lighter_shade(
                    constants.PREDICTION_WAYPOINT_COLOR,
                    i,
                    len(wps),
                )
                cv2.circle(
                    self.bev_image,
                    (int(wp_x), int(wp_y)),
                    radius=constants.PREDICTION_WAYPOINT_RADIUS,
                    color=color,
                    thickness=-1,
                )
            # self._draw_points_in_perspective(points=wps, size=5, connected=False, color=constants.PREDICTION_WAYPOINT_COLOR)

    def _ego_bounding_box(self):
        """Visualize ego bounding box on BEV image."""
        ego_box = np.array(
            [
                int(
                    self.bev_image.shape[1]
                    * (
                        -self.config.min_x_meter
                        / (self.config.max_x_meter - self.config.min_x_meter)
                    ),
                ),
                int(self.bev_image.shape[0] / 2),
                self.config.ego_extent_x * self.loc_pixels_per_meter,
                self.config.ego_extent_y * self.loc_pixels_per_meter,
                np.deg2rad(0.0),
                self.data.get("speed")[0].item(),
            ],
        )
        self.bev_image = viz_utils.draw_box(
            self.bev_image,
            ego_box,
            color=constants.EGO_BB_COLOR,
            thickness=4,
        )

    def _pred_bounding_box(self):
        """Visualize predicted bounding boxes on BEV image."""
        if self.config.detect_boxes:
            if isinstance(self.predictions, OpenLoopPrediction):
                pred_bounding_boxes: list[PredictedBoundingBox] = (
                    self.predictions.pred_bounding_box_image_system
                )
                for box in pred_bounding_boxes:
                    if box.score < self.config.bb_confidence_threshold:
                        continue
                    box = deepcopy(box)
                    inv_brake = 1.0 - box.brake
                    color_box = deepcopy(
                        list(constants.TRANSFUSER_BOUNDING_BOX_COLORS.values())[
                            box.clazz
                        ],
                    )
                    color_box = list(color_box)
                    color_box[1] = color_box[1] * inv_brake
                    box = box.scale(self.scale_factor)
                    self.bev_image = viz_utils.draw_box(
                        self.bev_image,
                        box,
                        color=color_box,
                    )

            elif (
                isinstance(self.predictions, Prediction)
                and self.predictions.pred_bounding_box is not None
            ):
                bb = self.predictions.pred_bounding_box.pred_bounding_box_image_system[
                    0
                ]
                if bb is not None:
                    for box in bb:
                        if (
                            box[TransfuserBoundingBoxIndex.SCORE]
                            < self.config.bb_confidence_threshold
                        ):
                            continue
                        box = deepcopy(box)
                        inv_brake = 1.0 - box[6]
                        color_box = deepcopy(
                            list(constants.TRANSFUSER_BOUNDING_BOX_COLORS.values())[
                                int(box[TransfuserBoundingBoxIndex.CLASS])
                            ],
                        )
                        color_box = list(color_box)
                        color_box[1] = color_box[1] * inv_brake
                        box[:4] = box[:4] * self.scale_factor
                        self.bev_image = viz_utils.draw_box(
                            self.bev_image,
                            box,
                            color=color_box,
                        )

    def _bounding_boxes(self):
        """Visualize ground truth bounding boxes on BEV image."""
        bounding_boxes = self.data.get("center_net_bounding_boxes")
        if bounding_boxes is not None:
            bounding_boxes = bounding_boxes.detach().cpu().numpy()[0]
            real_boxes = bounding_boxes.sum(axis=-1) != 0.0
            bounding_boxes = bounding_boxes[real_boxes]
            for box in bounding_boxes:
                box = deepcopy(box)
                box[:4] = box[:4] * self.scale_factor
                color_box = deepcopy(
                    list(constants.TRANSFUSER_BOUNDING_BOX_COLORS.values())[
                        int(box[TransfuserBoundingBoxIndex.CLASS])
                    ],
                )
                self.bev_image = viz_utils.draw_box(
                    self.bev_image,
                    box,
                    color=color_box,
                )
        vehicles_future_waypoints = self.data.get("vehicles_future_waypoints")
        vehicle_future_yaws = self.data.get("vehicles_future_yaws")

        if vehicles_future_waypoints is not None:
            for future_box_waypoints in vehicles_future_waypoints:
                wps = future_box_waypoints[0]
                for i, wp in enumerate(wps):
                    wp_x = wp[0] * self.loc_pixels_per_meter + self.origin[0]
                    wp_y = wp[1] * self.loc_pixels_per_meter + self.origin[1]
                    color = viz_utils.lighter_shade(
                        constants.GROUNDTRUTH_BB_WP_COLOR,
                        i,
                        len(wps),
                    )
                    cv2.circle(
                        self.bev_image,
                        (int(wp_x), int(wp_y)),
                        radius=constants.PREDICTION_WAYPOINT_RADIUS // 4,
                        color=color,
                        thickness=-1,
                    )

        if vehicles_future_waypoints is not None and vehicle_future_yaws is not None:
            for future_box_waypoints, future_box_yaws in zip(
                vehicles_future_waypoints,
                vehicle_future_yaws,
                strict=False,
            ):
                wps = future_box_waypoints[0]
                yaws = future_box_yaws[0]
                for i, wp in enumerate(wps):
                    color = viz_utils.lighter_shade(
                        constants.GROUNDTRUTH_BB_WP_COLOR,
                        i,
                        len(wps),
                    )
                    self._draw_bounding_box_from_waypoint(
                        wp[0],
                        wp[1],
                        yaws[i],
                        self.config.ego_extent_x,
                        self.config.ego_extent_y,
                        color=color,
                    )

    def _meta(self):
        """Render meta information panel."""
        # Configuration variables
        column_width = 400

        # Starting position
        start_x = 10
        start_y = 10
        line_height = 20
        separator = "-" * 45

        # Define units for attributes
        units_map = {
            # Velocities
            "speed": "m/s",
            "target_speed": "m/s",
            "second_highest_speed": "m/s",
            "speed_limit": "m/s",
            "target_speed_limit": "m/s",
            "second_highest_speed_limit": "m/s",
            # Accelerations
            "accel_x": "m/s²",
            "accel_y": "m/s²",
            "accel_z": "m/s²",
            "privileged_acceleration": "m/s²",
            # Distances
            "distance_to_next_junction": "m",
            "distance_to_junction": "m",
            "signed_dist_to_lane_change": "m",
            "distance_to_stop_sign": "m",
            "meters_travelled": "m",
            "ego_lane_width": "m",
            "perturbation_translation": "m",
            # Angles
            "theta": "rad",
            "privileged_yaw": "rad",
            "perturbation_rotation": "rad",
            # Angular velocities
            "angular_velocity_x": "rad/s",
            "angular_velocity_y": "rad/s",
            "angular_velocity_z": "rad/s",
        }

        # Collect all text lines
        text_lines = []

        # Collect all attributes - float values with 2 decimal places
        for attr_name in [
            "steer",
            "throttle",
            "brake",
            "signed_dist_to_lane_change",
            "distance_to_next_junction",
            "speed_limit",
            "target_speed_limit",
            "route_curvature",
            "route_labels_curvature",
            "distance_to_junction",
            "ego_lane_width",
            "perturbation_translation",
            "perturbation_rotation",
            "speed",
            "accel_x",
            "accel_y",
            "accel_z",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
            "target_speed",
            "theta",
            "privileged_yaw",
            "second_highest_speed",
            "second_highest_speed_limit",
            "distance_to_stop_sign",
            "privileged_acceleration",
            "meters_travelled",
        ]:
            attr_data = self.data.get(attr_name)
            if attr_data is not None:
                attr_data = attr_data[0]
                if isinstance(attr_data, torch.Tensor):
                    attr_data = attr_data.item()
                unit = units_map.get(attr_name, "")
                unit_str = f" {unit}" if unit else ""
                text_lines.append(f"{attr_name} {attr_data:.2f}{unit_str}")

        # Add navigation command in natural language
        for nav in ["command", "next_command"]:
            nav_command_data = self.data.get(nav)
            if nav_command_data is not None:
                nav_command_value = nav_command_data[0].argmax()
                if isinstance(nav_command_value, torch.Tensor):
                    nav_command_value = int(nav_command_value.item())
                else:
                    nav_command_value = int(nav_command_value)
                nav_command_text = CARLA_NAVIGATION_COMMAND_STR_MAP.get(
                    nav_command_value,
                )
                text_lines.append(f"{nav} {nav_command_text}")

        # Collect other values
        for attr_name in [
            "vehicle_hazard",
            "light_hazard",
            "walker_hazard",
            "stop_sign_hazard",
            "stop_sign_close",
            "num_lanes_opposite_direction",
            "changed_route",
            "does_emergency_brake_for_pedestrians",
            "weather_setting",
            "emergency_brake_for_special_vehicle",
            "visual_visibility",
            "num_parking_vehicles_in_proximity",
            "slower_bad_visibility",
            "slower_clutterness",
            "slower_occluded_junction",
            "used_traffic_bounding_boxes",
            "europe_traffic_light",
            "over_head_traffic_light",
            "jpeg_storage_quality",
            "augment_sample",
            "rear_danger_8",
            "rear_danger_16",
            "town",
            "frame_number",
            "num_dangerous_adversarial",
            "num_safe_adversarial",
            "num_ignored_adversarial",
            "rear_adversarial_id",
            "stuck_detector",
            "force_move",
            "bucket_identity",
            "perturbate_sensor",
        ]:
            attr_data = self.data.get(attr_name)
            if attr_data is not None:
                attr_data = attr_data[0]
                if isinstance(attr_data, torch.Tensor):
                    attr_data = attr_data.item()
                text_lines.append(f"{attr_name} {attr_data}")

        # Collect long text attributes
        for attr_name in [
            "route_number",
            "current_active_scenario_type",
            "previous_active_scenario_type",
            "scenario_type",
        ]:
            attr_data = self.data.get(attr_name)
            if attr_data is not None:
                attr_data = attr_data[0]
                if isinstance(attr_data, torch.Tensor):
                    attr_data = attr_data.item()
                text_lines.append(f"{attr_name} {attr_data}")

        # Collect prediction values
        if self.predictions is not None:
            if self.predictions.pred_target_speed_scalar is not None:
                text_lines.append(
                    f"pred_target_speed {float(self.predictions.pred_target_speed_scalar.detach().cpu().float()[0]):.2f} m/s",
                )

        # Collect target points
        for attr_name in ["target_point_previous", "target_point", "target_point_next"]:
            attr_data = self.data.get(attr_name)
            if attr_data is not None:
                attr_data = attr_data[0]
                if isinstance(attr_data, torch.Tensor):
                    attr_data = attr_data.detach().cpu().numpy()
                text_lines.append(
                    f"{attr_name} (x={attr_data[0]:.1f}m, y={attr_data[1]:.1f}m)",
                )

        # Sort all lines alphabetically
        text_lines = sorted(text_lines)

        # Load fonts once
        font_regular = ImageFont.truetype("3rd_party/Roboto-Regular.ttf", 17)
        font_bold = ImageFont.truetype("3rd_party/Roboto-Bold.ttf", 17)

        img_pil = Image.fromarray(self.meta_panel)
        draw = ImageDraw.Draw(img_pil)

        # Fixed layout parameters
        column_width = 350
        num_columns = 4
        start_x = 10
        start_y = 10
        line_height = 20

        # Process text lines and prepare items
        items = []
        for text in text_lines:
            if text == "new_column":
                continue
            if text == separator:
                items.append({"type": "separator", "text": text})
                continue

            # Split text into attribute name and value
            if " " in text or ":" in text:
                split_idx = text.find(" ") if " " in text else text.find(":")
                if text.find(":") != -1 and (
                    text.find(" ") == -1 or text.find(":") < text.find(" ")
                ):
                    split_idx = text.find(":")
                    attr_name_part = text[: split_idx + 1]
                    value_part = text[split_idx + 1 :].strip()
                else:
                    attr_name_part = text[:split_idx]
                    value_part = text[split_idx:].strip()

                # Calculate widths for right-alignment
                value_bbox = draw.textbbox((0, 0), value_part, font=font_regular)
                value_width = value_bbox[2] - value_bbox[0]

                items.append(
                    {
                        "type": "item",
                        "name": attr_name_part,
                        "value": value_part,
                        "value_width": value_width,
                    },
                )
            else:
                items.append({"type": "other", "text": text})

        # Draw all text lines (left to right, top to bottom)
        item_index = 0
        for item in items:
            if item["type"] == "separator":
                # Draw separator across all columns on current row
                row = item_index // num_columns
                y = start_y + row * line_height
                draw.text((start_x, y), item["text"], font=font_regular, fill=(0, 0, 0))
                item_index += 1
                continue

            if item["type"] == "other":
                row = item_index // num_columns
                col = item_index % num_columns
                x = start_x + col * column_width
                y = start_y + row * line_height
                draw.text((x, y), item["text"], font=font_regular, fill=(0, 0, 0))
                item_index += 1
                continue

            if item["type"] == "item":
                row = item_index // num_columns
                col = item_index % num_columns
                x = start_x + col * column_width
                y = start_y + row * line_height

                # Draw attribute name in bold, left aligned
                draw.text((x, y), item["name"], font=font_bold, fill=(0, 0, 0))

                # Draw value right-aligned within this column's space
                value_x = x + column_width - item["value_width"] - 20
                draw.text(
                    (value_x, y),
                    item["value"],
                    font=font_regular,
                    fill=(0, 0, 0),
                )
                item_index += 1

        # Convert back to cv2
        self.meta_panel = np.array(img_pil)

    def _draw_bounding_box_from_waypoint(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        extent_x: float,
        extent_y: float,
        color,
    ):
        bb = np.array(
            [ego_x, ego_y, self.config.ego_extent_x, self.config.ego_extent_y, ego_yaw],
        )
        bb = carla_dataset_utils.bb_vehicle_to_image_system(
            bb[None],
            pixels_per_meter=self.config.pixels_per_meter,
            min_x=self.config.min_x_meter,
            min_y=self.config.min_y_meter,
        )[0]
        bb[:4] = bb[:4] * self.scale_factor
        self.bev_image = viz_utils.draw_box(
            self.bev_image,
            bb,
            color=color,
            thickness=1,
        )


def visualize_sample(
    config: TrainingConfig,
    predictions: Prediction,
    data: dict,
    factor: float = 1.0,
    prefix="train",
    log_wandb=False,
    save_image=False,
    save_path: str = None,
    postfix: str = None,
):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    if not (log_wandb or save_image):
        return
    with torch.no_grad():
        # Visualize feature map for debugging
        visualize_feature_maps(
            config=config,
            predictions=predictions,
            data=data,
            factor=factor,
            prefix=prefix,
            log_wandb=log_wandb,
            save_image=save_image,
            save_path=save_path,
            postfix=postfix,
        )

        # Visualize training prediction
        visualizer = Visualizer(
            config=config,
            data=data,
            prediction=predictions,
            training=True,
        )

        prediction = visualizer.visualize_training_prediction()
        prediction_pil = Image.fromarray(prediction)  # Convert to PIL image
        new_size = (
            int(prediction_pil.width * factor),
            int(prediction_pil.height * factor),
        )
        prediction_resized = prediction_pil.resize(new_size)
        if log_wandb and config.log_wandb:
            wandb.log(
                {f"viz/{prefix}_pred": wandb.Image(prediction_resized)},
                commit=False,
            )
        if save_image:
            prediction_resized.save(f"{save_path}/prediction_{postfix}.png")

        # Visualize ground truth labels
        visualizer = Visualizer(
            config,
            data=data,
            prediction=predictions,
            training=True,
        )
        ground_truth = visualizer.visualize_training_labels()
        ground_truth_pil = Image.fromarray(ground_truth)  # Convert to PIL image
        new_size = (
            int(ground_truth_pil.width * factor),
            int(ground_truth_pil.height * factor),
        )
        ground_truth_resized = ground_truth_pil.resize(new_size)
        if log_wandb and config.log_wandb:
            wandb.log(
                {f"viz/_{prefix}_gt": wandb.Image(ground_truth_resized)},
                commit=False,
            )
        if save_image:
            ground_truth_resized.save(f"{save_path}/gt_{postfix}.png")


def visualize_feature_maps(
    config: TrainingConfig,
    predictions: Prediction,
    data: dict,
    factor: float = 1.0,
    prefix="train",
    log_wandb=False,
    save_image=False,
    save_path: str = None,
    postfix: str = None,
):
    n_rows = 4
    n_cols = 4
    _, axs = plt.subplots(n_rows, n_cols, figsize=(24, 12))
    images = [
        (data["rasterized_lidar"][0, 0].detach().cpu().numpy(), "BEV LiDAR", "hot"),
        (
            data["center_net_heatmap"][0, 0].detach().cpu().numpy(),
            "Heatmap Label",
            "hot",
        ),
        (data["center_net_wh"][0, 0].detach().cpu().numpy(), "WH Label", "hot"),
        (
            data["center_net_yaw_class"][0].detach().cpu().numpy(),
            "Yaw Class Label",
            "hot",
        ),
        (
            data["center_net_yaw_res"][0, 0].detach().cpu().numpy(),
            "Yaw Res Label",
            "hot",
        ),
        (data["center_net_offset"][0, 0].detach().cpu().numpy(), "Offset Label", "hot"),
        (
            data["center_net_velocity"][0, 0].detach().cpu().numpy(),
            "Velocity Label",
            "hot",
        ),
        (data["bev_semantic"][0].detach().cpu().numpy(), "BEV Semantic Label", "hot"),
        (
            predictions.pred_bounding_box.center_heatmap_pred[0]
            .detach()
            .argmax(0)
            .cpu()
            .numpy(),
            "Heatmap Prediction",
            "hot",
        ),
        (
            predictions.pred_bev_semantic[0].detach().argmax(0).cpu().numpy(),
            "BEV Semantic Prediction",
            "hot",
        ),
        (
            predictions.pred_future_waypoints[0].detach().cpu().numpy()
            if predictions.pred_future_waypoints is not None
            else None,
            "Waypoints",
            "hot",
        ),
        (
            predictions.pred_route[0].detach().cpu().numpy()
            if predictions.pred_route is not None
            else None,
            "Route",
            "hot",
        ),
        (data.get("radar")[0].cpu().numpy(), "Radar Input", "hot"),
        (data.get("radar_detections"), "Radar Detection Label", "hot"),
        (predictions.pred_radar_predictions, "Radar Detection Prediction", "hot"),
    ]

    for i, (img, title, cmap) in enumerate(images):
        ax = axs[i // n_cols, i % n_cols]
        ax.set_title(title)

        if img is None:
            continue

        if title == "Waypoints":
            ax.scatter(img[:, 0], img[:, 1], c="lime", s=20)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, 48)
            ax.set_ylim(-32, 32)
            ax.invert_yaxis()
        elif title == "Route":
            ax.plot(img[:, 0], img[:, 1], c="cyan", marker="o", markersize=3)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, 48)
            ax.set_ylim(-32, 32)
            ax.invert_yaxis()
        elif title == "Radar Input":
            x, y, vel = (img[:, 0], img[:, 1], img[:, 3])
            for xm, ym, vk in zip(x, y, vel, strict=False):
                color = "red" if vk > 0 else "blue"
                ax.scatter(xm, ym, c=color, s=abs(vk) * 3 + 2)
            ax.set_xlim(config.min_x_meter, config.max_x_meter)
            ax.set_ylim(config.min_y_meter, config.max_y_meter)
            ax.invert_yaxis()
        elif title == "Radar Detection Label":
            radar_labels = img[0]
            x, y, v, valid = (
                radar_labels[:, RadarLabels.X],
                radar_labels[:, RadarLabels.Y],
                radar_labels[:, RadarLabels.V],
                radar_labels[:, RadarLabels.VALID],
            )
            for xm, ym, vk, validk in zip(x, y, v, valid, strict=False):
                if validk > 0.5:
                    ax.scatter(xm, ym, c="blue", s=abs(vk) * 3 + 2)
            ax.set_xlim(config.min_x_meter, config.max_x_meter)
            ax.set_ylim(config.min_y_meter, config.max_y_meter)
            ax.invert_yaxis()
        elif title == "Radar Detection Prediction":
            radar_detection = img[0].detach().cpu().float().numpy()
            x, y, v, valid = (
                radar_detection[:, RadarLabels.X],
                radar_detection[:, RadarLabels.Y],
                radar_detection[:, RadarLabels.V],
                radar_detection[:, RadarLabels.VALID],
            )
            for xm, ym, vk, validk in zip(x, y, v, valid, strict=False):
                if validk > 0.0:  # Implicit sigmoid threshold 0.5
                    ax.scatter(xm, ym, c="blue", s=abs(vk) * 3 + 2)
            ax.set_xlim(config.min_x_meter, config.max_x_meter)
            ax.set_ylim(config.min_y_meter, config.max_y_meter)
            ax.invert_yaxis()
        else:
            ax.imshow(img, cmap=cmap)

    plt.tight_layout()

    # Save the figure to disk if needed
    if save_image:
        out_path = f"{save_path}/{prefix}_bev_{postfix}.png"
        plt.savefig(out_path, dpi=300)

    # Log the whole figure to wandb
    if log_wandb and config.log_wandb:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        wandb.log(
            {"train_viz/feature_maps": wandb.Image(Image.open(buf))},
            commit=False,
        )
        buf.close()

    plt.close()
