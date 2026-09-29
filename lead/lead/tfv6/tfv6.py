from __future__ import annotations

import typing
from dataclasses import dataclass

import jaxtyping as jt
import torch
from beartype import beartype
from torch import nn

from lead.common.constants import SourceDataset
from lead.tfv6.bev_decoder import BEVDecoder
from lead.tfv6.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
    CenterNetDecoder,
)
from lead.tfv6.perspective_decoder import PerspectiveDecoder
from lead.tfv6.planning_decoder import PlanningDecoder
from lead.tfv6.radar_detector import RadarDetector
from lead.tfv6.transfuser_backbone import TransfuserBackbone
from lead.training.config_training import TrainingConfig


class TFv6(nn.Module):
    @beartype
    def __init__(
        self,
        device: torch.device,
        config: TrainingConfig,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.log = {}

        self.backbone = TransfuserBackbone(self.device, self.config)

        if self.config.use_semantic and self.config.use_carla_data:
            self.semantic_decoder = PerspectiveDecoder(
                config=self.config,
                in_channels=self.backbone.num_image_features,
                out_channels=self.config.num_semantic_classes,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="semantic",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_depth and self.config.use_carla_data:
            self.depth_decoder = PerspectiveDecoder(
                config=self.config,
                in_channels=self.backbone.num_image_features,
                out_channels=1,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="depth",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                self.bev_semantic_decoder = BEVDecoder(
                    self.config,
                    self.config.num_bev_semantic_classes,
                    self.device,
                    source_data=SourceDataset.CARLA,
                )

            if self.config.use_navsim_data:
                self.bev_semantic_decoder_navsim = BEVDecoder(
                    self.config,
                    self.config.navsim_num_bev_semantic_classes,
                    self.device,
                    source_data=SourceDataset.NAVSIM,
                )

        if self.config.detect_boxes:
            if self.config.use_carla_data:
                self.center_net_decoder = CenterNetDecoder(
                    self.config.num_bb_classes,
                    self.config,
                    self.device,
                    source_data=SourceDataset.CARLA,
                )
            if self.config.use_navsim_data:
                self.center_net_decoder_navsim = CenterNetDecoder(
                    self.config.navsim_num_bb_classes,
                    self.config,
                    self.device,
                    source_data=SourceDataset.NAVSIM,
                )

        if self.config.radar_detection and self.config.use_carla_data:
            self.radar_detector = RadarDetector(
                bev_input_dim=self.backbone.num_lidar_features,
                config=self.config,
                device=self.device,
            )

        if self.config.use_planning_decoder:
            self.planning_decoder = PlanningDecoder(
                input_bev_channels=self.backbone.num_lidar_features,
                config=self.config,
                device=self.device,
            ).to(self.device)

    @beartype
    def forward(self, data: dict[str, typing.Any]) -> Prediction:
        self.log = {}
        pred_route = pred_future_waypoints = pred_target_speed_distribution = (
            pred_target_speed_scalar
        ) = pred_headings = None
        pred_semantic = pred_depth = pred_bounding_box = pred_bev_semantic = None
        pred_bounding_box_navsim = pred_bev_semantic_navsim = None

        # Backbone
        bev_features, image_features = self.backbone(data)

        # Radar detection
        radar_features = radar_predictions = None
        if self.config.use_carla_data and self.config.radar_detection:
            radar_features, radar_predictions = self.radar_detector(bev_features, data)

        # Planning heads
        if self.config.use_planning_decoder:
            planner_radar_features = radar_features
            planner_radar_predictions = radar_predictions
            if not self.config.use_radar_detection or not self.config.use_carla_data:
                planner_radar_features = planner_radar_predictions = None
            (
                pred_route,
                pred_future_waypoints,
                pred_target_speed_distribution,
                pred_target_speed_scalar,
                pred_headings,
            ) = self.planning_decoder(
                bev_features,
                planner_radar_features,
                planner_radar_predictions,
                data,
                log=self.log,
            )

        # Semantic segmentation forward pass
        if self.config.use_carla_data and self.config.use_semantic:
            pred_semantic = self.semantic_decoder(data, image_features, self.log)

        # Depth estimation forward pass
        if self.config.use_carla_data and self.config.use_depth:
            pred_depth = self.depth_decoder(data, image_features, self.log)

        # Bounding box detection forward pass
        bev_feature_grid = self.backbone.top_down(bev_features)
        if self.config.detect_boxes:
            if self.config.use_carla_data:
                pred_bounding_box = self.center_net_decoder(
                    data,
                    bev_feature_grid,
                    self.log,
                )
            if self.config.use_navsim_data:
                pred_bounding_box_navsim = self.center_net_decoder_navsim(
                    data,
                    bev_feature_grid,
                    self.log,
                )

        # BEV semantic segmentation forward pass
        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                pred_bev_semantic = self.bev_semantic_decoder(
                    bev_feature_grid,
                    self.log,
                )
            if self.config.use_navsim_data:
                pred_bev_semantic_navsim = self.bev_semantic_decoder_navsim(
                    bev_feature_grid,
                    self.log,
                )

        # Collect predictions
        return Prediction(
            # Planning prediction
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_route=pred_route,
            # CARLA perception prediction
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bounding_box=pred_bounding_box,
            pred_bev_semantic=pred_bev_semantic,
            pred_radar_features=radar_features,
            pred_radar_predictions=radar_predictions,
            # NavSim perception prediction
            pred_bounding_box_navsim=pred_bounding_box_navsim,
            pred_bev_semantic_navsim=pred_bev_semantic_navsim,
            pred_headings=pred_headings,
        )

    @beartype
    def compute_loss(
        self,
        predictions: Prediction,
        data: dict[str, typing.Any],
    ) -> tuple[dict, dict]:
        loss = {}
        # Semantic segmentation loss
        if self.config.use_semantic and self.config.use_carla_data:
            self.semantic_decoder.compute_loss(
                predictions.pred_semantic,
                data,
                loss,
                log=self.log,
            )

        # Depth estimation loss
        if self.config.use_depth and self.config.use_carla_data:
            self.depth_decoder.compute_loss(
                predictions.pred_depth,
                data,
                loss,
                log=self.log,
            )

        # BEV semantic segmentation loss
        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                self.bev_semantic_decoder.compute_loss(
                    predictions.pred_bev_semantic,
                    data,
                    loss,
                    log=self.log,
                )
            if self.config.use_navsim_data:
                self.bev_semantic_decoder_navsim.compute_loss(
                    predictions.pred_bev_semantic_navsim,
                    data,
                    loss,
                    log=self.log,
                )

        # Bounding box detection loss
        if self.config.detect_boxes:
            if self.config.use_carla_data:
                self.center_net_decoder.compute_loss(
                    data=data,
                    bounding_box_features=predictions.pred_bounding_box,
                    losses=loss,
                    log=self.log,
                )
            if self.config.use_navsim_data:
                self.center_net_decoder_navsim.compute_loss(
                    data=data,
                    bounding_box_features=predictions.pred_bounding_box_navsim,
                    losses=loss,
                    log=self.log,
                )

        # Radar detection loss
        if self.config.radar_detection and self.config.use_carla_data:
            self.radar_detector.compute_loss(
                pred=predictions.pred_radar_predictions,
                data=data,
                loss=loss,
                log=self.log,
            )

        # Planning loss
        if self.config.use_planning_decoder:
            self.planning_decoder.compute_loss(
                data=data,
                predictions=predictions,
                loss=loss,
                log=self.log,
            )

        return loss, self.log


@jt.jaxtyped(typechecker=beartype)
@dataclass
class Prediction:
    """Raw output predictions from the model."""

    # Planning prediction
    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_target_speed_distribution: (
        jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    )
    pred_target_speed_scalar: jt.Float[torch.Tensor, " bs"] | None
    pred_route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None

    # CARLA perception prediction
    pred_semantic: (
        jt.Float[torch.Tensor, "bs num_semantic_classes img_height img_width"] | None
    )
    pred_bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    pred_depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    pred_bounding_box: CenterNetBoundingBoxPrediction | None
    pred_radar_features: jt.Float[torch.Tensor, "B Q C"] | None
    pred_radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None

    # NavSim perception prediction
    pred_bounding_box_navsim: CenterNetBoundingBoxPrediction | None
    pred_bev_semantic_navsim: (
        jt.Float[torch.Tensor, "bs num_bev_classes_navsim bev_height bev_width"] | None
    )
    pred_headings: jt.Float[torch.Tensor, "bs n_waypoints"] | None
