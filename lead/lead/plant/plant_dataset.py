"""PlanT dataset: extends CARLAData with object tokenization and custom collation."""

from __future__ import annotations

import logging
import time

import torch
from beartype import beartype

from lead.common import common_utils
from lead.data_loader.carla_dataset import CARLAData
from lead.plant.plant_config import PlantConfig
from lead.plant.plant_tokenizer import tokenize_bboxes
from lead.plant.plant_variables import PlantVariables
from lead.training import mixed_training_utils

LOG = logging.getLogger(__name__)


class PlantCARLAData(CARLAData):
    """CARLAData extended with PlanT object tokenization.

    Adds ``plant_objects``, ``route_original``, and ``speed_limit`` keys
    to each sample for the :class:`PlantBackbone`.
    """

    @beartype
    def __init__(self, root: str | list[str], config: PlantConfig, **kwargs):
        super().__init__(root=root, config=config, **kwargs)
        self.speed_cats = PlantVariables.speed_cats

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # Bboxes + tokenization stand in for "sensor" loading on the PlanT path.
        sensor_start = time.time()

        # Load raw bounding boxes for PlanT tokenization
        boxes_path = str(self.bboxes[index], encoding="utf-8")
        raw_boxes = common_utils.read_pickle(boxes_path)

        # Load measurement for route / speed_limit / ego info
        measurement_file = str(self.metas[index], encoding="utf-8")
        meta = common_utils.read_pickle(measurement_file)

        # Tokenize bounding boxes into PlanT object format
        plant_objects = tokenize_bboxes(
            boxes=raw_boxes,
            plant_range=self.config.plant_range,
            plant_range_factor_front=self.config.plant_range_factor_front,
            plant_input_static_cars=self.config.plant_input_static_cars,
        )
        data["plant_objects"] = plant_objects  # list[list[float]], variable length

        # Route waypoints (num_route_points, 2D)
        n_route = self.config.plant_num_route_points
        route_original = meta.get("route_original", meta.get("route", []))[:n_route]

        # Pad to n_route if shorter
        while len(route_original) < n_route:
            route_original.append(route_original[-1] if route_original else [0.0, 0.0])
        data["route_original"] = torch.tensor(route_original, dtype=torch.float32)

        # Speed limit category
        speed_limit_kmh = round(meta.get("speed_limit", 50) * 3.6)
        data["speed_limit"] = torch.tensor(
            self.speed_cats.get(speed_limit_kmh, 0),
            dtype=torch.int,
        )

        # Ego speed for optional input
        data["input_ego_speed"] = torch.tensor(
            meta.get("speed", 0.0),
            dtype=torch.float32,
        )

        data["loading_sensor_time"] = time.time() - sensor_start
        data["loading_time"] = data["loading_meta_time"] + data["loading_sensor_time"]
        return data


def plant_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length PlanT object tokens.

    Flattens all ``plant_objects`` across the batch into a single tensor
    with an index mapping (``idxs``), matching PlanT's ``generate_batch``
    semantics.  All other keys are delegated to the standard mixed-data
    collation.
    """
    max_objects = max(len(sample["plant_objects"]) for sample in batch)
    bs = len(batch)

    # Flat object tensor: padding token at index 0
    x_all = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    batch_idxs = torch.zeros((bs, max_objects), dtype=torch.int32)

    n = 1  # 0 is reserved for padding
    for i, sample in enumerate(batch):
        objects = sample["plant_objects"]
        n_obj = len(objects)
        batch_idxs[i, :n_obj] = torch.arange(n, n + n_obj)
        n += n_obj
        x_all.extend(objects)

    # Remove plant_objects from samples before standard collation
    for sample in batch:
        del sample["plant_objects"]

    collated = mixed_training_utils.mixed_data_collate_fn(batch)

    # Add PlanT-specific tensors
    collated["x_objs"] = torch.tensor(x_all, dtype=torch.float32)
    collated["idxs"] = batch_idxs

    return collated
