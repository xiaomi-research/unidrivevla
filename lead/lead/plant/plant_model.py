from __future__ import annotations

import typing

import torch
from beartype import beartype
from torch import nn

from lead.plant.plant_backbone import PlantBackbone
from lead.plant.plant_config import PlantConfig
from lead.plant.plant_planning_decoder import PlantPlanningDecoder
from lead.tfv6.tfv6 import Prediction


class PlantModel(nn.Module):
    """PlanT model: object token backbone + planning decoder.

    Drop-in replacement for :class:`~lead.tfv6.tfv6.TFv6`.  Composes
    :class:`PlantBackbone` (BERT encoder) with :class:`PlantPlanningDecoder`
    (reuses LEAD's planning heads and losses).
    """

    @beartype
    def __init__(self, device: torch.device, config: PlantConfig):
        super().__init__()
        self.device = device
        self.config = config
        self.log: dict = {}

        self.backbone = PlantBackbone(device, config)
        self.planning_decoder = PlantPlanningDecoder(config=config, device=device)

    @beartype
    def forward(self, data: dict[str, typing.Any]) -> Prediction:
        self.log = {}

        context_tokens = self.backbone(data)

        route, waypoints, speed_dist, speed_scalar = self.planning_decoder(
            context_tokens,
            data,
            self.log,
        )

        return Prediction(
            # Planning
            pred_future_waypoints=waypoints,
            pred_route=route,
            pred_target_speed_distribution=speed_dist,
            pred_target_speed_scalar=speed_scalar,
            pred_headings=None,
            # Perception (not used by PlanT)
            pred_semantic=None,
            pred_bev_semantic=None,
            pred_depth=None,
            pred_bounding_box=None,
            pred_radar_features=None,
            pred_radar_predictions=None,
            pred_bounding_box_navsim=None,
            pred_bev_semantic_navsim=None,
        )

    @beartype
    def compute_loss(
        self,
        predictions: Prediction,
        data: dict[str, typing.Any],
    ) -> tuple[dict, dict]:
        loss: dict = {}
        self.planning_decoder.compute_loss(predictions, data, loss, log=self.log)
        return loss, self.log
