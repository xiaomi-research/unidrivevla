"""Convert CARLA bounding box dicts into PlanT object token lists."""

from __future__ import annotations

import logging

import numpy as np
from beartype import beartype

from lead.plant.plant_variables import PlantVariables
from lead.plant.static_extents import CAR_EXTENTS, STATIC_EXTENTS

logger = logging.getLogger(__name__)

EMERGENCY_TYPE_IDS = {
    "vehicle.dodge.charger_police",
    "vehicle.dodge.charger_police_2020",
    "vehicle.carlamotors.firetruck",
    "vehicle.ford.ambulance",
}


def _normalize_angle_degree(x: float) -> float:
    x = x % 360.0
    if x > 180.0:
        x -= 360.0
    return x


def _rad2deg(theta: float) -> float:
    return _normalize_angle_degree(np.rad2deg(theta).item())


@beartype
def tokenize_bboxes(
    boxes: list[dict],
    plant_range: int | float,
    plant_range_factor_front: int | float,
    plant_input_static_cars: bool,
) -> list[list[float]]:
    """Convert raw CARLA bounding box dicts to PlanT object token lists.

    Each token is ``[class_id, x, y, yaw_deg, speed_kmh, extent_y*2, extent_x*2]``.

    Args:
        boxes: List of bbox dicts from CARLA data (same format as ``boxes/*.json.gz``).
        plant_range: Detection range in meters.
        plant_range_factor_front: Front range multiplier.
        plant_input_static_cars: Whether to include static cars.

    Returns:
        List of 7-element lists, one per valid object token.
    """
    class_nums = dict(PlantVariables.class_nums)
    if not plant_input_static_cars:
        class_nums.pop("static_car", None)

    car_types = PlantVariables.car_types
    max_dist = plant_range
    front_factor_sq = plant_range_factor_front**2

    labels_data = boxes[1:]  # remove ego car (first entry)

    # Filter and fix extents
    for x in labels_data:
        if "position" not in x:
            x["class"] = "irrelevant"
            continue

        pos_x, pos_y, pos_z = x["position"]

        # Range filtering
        if x["class"] in ["traffic_light", "stop_sign"]:
            if pos_x**2 + pos_y**2 > 30**2 or abs(pos_z) > 30:
                x["class"] = "too far"
        else:
            x_div = front_factor_sq if pos_x > 0 else 1
            if pos_x**2 / x_div + pos_y**2 > max_dist**2 or abs(pos_z) > 30:
                x["class"] = "too far"

        # Emergency vehicles
        if x["class"] == "car" and x.get("type_id", "") in EMERGENCY_TYPE_IDS:
            x["class"] = "emergency"

        # Filter irrelevant statics and fix extents
        elif x["class"] == "static":
            type_id = x.get("type_id", "")
            if type_id not in [
                "static.prop.constructioncone",
                "static.prop.trafficwarning",
            ]:
                x["class"] = "irrelevant_static"
            elif type_id in STATIC_EXTENTS:
                x["extent"] = STATIC_EXTENTS[type_id]

        elif x["class"] == "static_car":
            mesh_path = x.get("mesh_path", "")
            if mesh_path in CAR_EXTENTS:
                x["extent"] = CAR_EXTENTS[mesh_path]
                scale = x.get("scale")
                if scale is not None:
                    s = float(scale)
                    x["extent"] = [a * s for a in x["extent"]]

    # Build tokens for dynamic objects (car, walker, emergency)
    input_objects = [
        [
            class_nums[x["class"].lower()],
            x["position"][0],
            x["position"][1],
            _rad2deg(x["yaw"]),
            x["speed"] * 3.6,
            x["extent"][1] * 2
            + (0 if "scenario" not in x or "Door" not in x["scenario"] else 1),
            x["extent"][0] * 2,
        ]
        for x in labels_data
        if x["class"].lower() in car_types
    ]

    # Build tokens for static objects, traffic lights, stop signs
    input_objects += [
        [
            class_nums[x["class"].lower()],
            x["position"][0],
            x["position"][1],
            _rad2deg(x["yaw"]),
            0.0,
            x["extent"][1] * 2,
            x["extent"][0] * 2,
        ]
        for x in labels_data
        if x["class"].lower() not in car_types
        and x["class"].lower() in class_nums
        and (
            x["class"].lower() != "traffic_light"
            or (x.get("state") in ["Red", "Yellow"] and x.get("affects_ego"))
        )
        and (x["class"] != "stop_sign" or x.get("affects_ego"))
    ]

    return input_objects
