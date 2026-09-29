from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional
from multiprocessing import Pool

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **_: x


DET_CLASS_MAP = {
    "car": "car",
    "van": "van",
    "truck": "truck",
    "bus": "truck",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "bicycle",
    "walker": "pedestrian",
    "pedestrian": "pedestrian",
    "traffic_light": "traffic_light",
    "stop_sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
    "static_prop_car": "car",
    "cone": "traffic_cone",
    "traffic_cone": "traffic_cone",
    "static.prop.constructioncone": "traffic_cone",
    "static.prop.warningconstruction": "traffic_cone",
    "static.prop.warningaccident": "traffic_cone",
    "static.prop.trafficwarning": "traffic_cone",
    "static.prop.dirtdebris01": "others",
    "static.prop.dirtdebris02": "others",
    "static.prop.mesh": "others",
    "static.prop.vendingmachine": "others",
}
DET_TYPE_ID_MAP = {
    "vehicle.bh.crossbike": "bicycle",
    "vehicle.diamondback.century": "bicycle",
    "vehicle.gazelle.omafiets": "bicycle",
    "vehicle.ford.ambulance": "van",
    "vehicle.carlamotors.firetruck": "truck",
}


LEFT2RIGHT = np.eye(4, dtype=np.float32)
LEFT2RIGHT[1, 1] = -1.0
LEFTHAND_EGO_TO_LIDAR = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
LEAD_STAGE1_INFOS_VERSION = 15
LEAD_SENSOR_PERTURBATION_VERSION = 1
LEAD_DETECTION_FILTER_VERSION = 4
LEAD_DETECTION_CLASS_MAPPING_VERSION = 2
LEAD_BBOX_CENTER_OFFSET_VERSION = 1
LEAD_DETECTION_Z_RANGE = (-4.0, 4.0)
LEAD_DETECTION_TRAFFIC_CONTROL_FILTER = "tfv6_affects_ego_non_green_v1"

# CARLA actors are not consistently located at their 3D bounding-box center.
# These are bounding_box.location values from the CARLA 0.9.15 build used by
# LEAD evaluation. Raw LEAD positions store actor origins.
CARLA_VEHICLE_BBOX_LOCAL_CENTERS = {
    "vehicle.audi.a2": (0.0, 0.000339584, 0.781796873),
    "vehicle.audi.etron": (-0.003597412, 0.0, 0.820468724),
    "vehicle.audi.tt": (-0.000372467, -0.000000076, 0.692656219),
    "vehicle.bh.crossbike": (0.0, 0.000000076, 0.808221281),
    "vehicle.bmw.grandtourer": (0.0, -0.000533142, 0.765078127),
    "vehicle.carlamotors.carlacola": (0.0, 0.000001678, 1.233749986),
    "vehicle.carlamotors.european_hgv": (0.012723388, 0.005304871, 1.730624914),
    "vehicle.carlamotors.firetruck": (-0.253442079, 0.005304871, 1.913359284),
    "vehicle.chevrolet.impala": (0.000059204, 0.001790009, 0.701484382),
    "vehicle.citroen.c3": (0.0, 0.002367172, 0.761953115),
    "vehicle.diamondback.century": (0.0, 0.000000076, 0.815744758),
    "vehicle.dodge.charger_2020": (-0.005347290, 0.000000381, 0.765546858),
    "vehicle.dodge.charger_police": (0.023636932, 0.000600586, 0.792656243),
    "vehicle.dodge.charger_police_2020": (0.109497376, 0.000000076, 0.817421854),
    "vehicle.ford.ambulance": (-0.288559556, 0.001940689, 1.215624928),
    "vehicle.ford.crown": (0.185408324, 0.0, 0.768437505),
    "vehicle.ford.mustang": (0.032142028, -0.000000076, 0.650468707),
    "vehicle.gazelle.omafiets": (-0.010000000, 0.000000076, 0.894649029),
    "vehicle.harley-davidson.low_rider": (0.0, 0.000000076, 0.824649036),
    "vehicle.jeep.wrangler_rubicon": (0.000216827, 0.001030045, 0.932265580),
    "vehicle.kawasaki.ninja": (0.0, 0.000000076, 0.768860757),
    "vehicle.lincoln.mkz_2017": (0.004044190, 0.0, 0.718828082),
    "vehicle.lincoln.mkz_2020": (-0.006292725, 0.0, 0.748828113),
    "vehicle.mercedes.coupe": (-0.000000153, -0.000774307, 0.666796863),
    "vehicle.mercedes.coupe_2020": (-0.002593536, 0.0, 0.720390618),
    "vehicle.mercedes.sprinter": (-0.010541687, 0.004136200, 1.288203120),
    "vehicle.micro.microlino": (0.0, -0.000059891, 0.677187502),
    "vehicle.mini.cooper_s": (0.0, -0.000060043, 0.693046868),
    "vehicle.mini.cooper_s_2021": (-0.031253204, -0.000000839, 0.891015589),
    "vehicle.mitsubishi.fusorosa": (-0.445223391, -0.120886989, 2.127343655),
    "vehicle.nissan.micra": (0.0, 0.000968933, 0.781250000),
    "vehicle.nissan.patrol": (-0.057376403, 0.000158920, 0.935156226),
    "vehicle.nissan.patrol_2021": (0.028383788, -0.000000076, 1.016718745),
    "vehicle.seat.leon": (-0.000149994, -0.001558380, 0.739609361),
    "vehicle.tesla.cybertruck": (0.000001221, 0.0, 1.049062490),
    "vehicle.tesla.model3": (0.029219665, 0.0, 0.735859334),
    "vehicle.toyota.prius": (0.001988220, -0.000383835, 0.724140584),
    "vehicle.vespa.zx125": (0.010000000, 0.000000076, 0.794649005),
    "vehicle.volkswagen.t2": (0.001331787, -0.000537415, 1.014453053),
    "vehicle.volkswagen.t2_2021": (0.116671748, -0.000352173, 0.991249979),
    "vehicle.yamaha.yzf": (0.0, 0.000000076, 0.774649024),
}

CARLA_STATIC_MESH_BBOX_LOCAL_CENTERS = {
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": (-0.006292725, 0.0, 0.764335930),
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": (-0.005347290, 0.0, 0.765585899),
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": (0.116671748, -0.000001221, 0.991249979),
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": (0.185408324, 0.0, 0.768437505),
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": (0.028383788, 0.0, 1.016718745),
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": (-0.002593536, 0.0, 0.725351572),
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": (0.029219665, 0.0, 0.735859334),
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": (-0.031253204, 0.000000610, 0.891015589),
}

TFV6_STATIC_TYPE_WHITELIST = {
    "static.prop.constructioncone",
    "static.prop.trafficwarning",
}

class MissingBBoxCenterOffsetError(ValueError):
    pass


class InvalidCriticalSupervisionError(ValueError):
    pass



def _read_lzma_pickle_from_zip(zf: zipfile.ZipFile, member: str) -> Any:
    return pickle.loads(lzma.decompress(zf.read(member)))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        value = float(value)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def _require_finite_float(
    meta: dict[str, Any],
    key: str,
    context: str = "",
) -> float:
    raw_value = meta.get(key)
    location = f" ({context})" if context else ""
    if raw_value is None:
        raise InvalidCriticalSupervisionError(f"missing {key!r}{location}")
    if hasattr(raw_value, "item"):
        raw_value = raw_value.item()
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise InvalidCriticalSupervisionError(
            f"invalid {key!r}={raw_value!r}{location}"
        ) from exc
    if not math.isfinite(value):
        raise InvalidCriticalSupervisionError(
            f"non-finite {key!r}={raw_value!r}{location}"
        )
    return value


def _sensor_perturbation(meta: dict[str, Any]) -> tuple[float, float]:
    return (
        _as_float(meta.get("perturbation_translation")),
        _as_float(meta.get("perturbation_rotation")),
    )


def _perturb_lead_points(
    points: Any,
    translation: float,
    rotation_degrees: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] < 2:
        raise ValueError(f"Expected points with at least two coordinates, got {points.shape}")
    yaw = math.radians(float(rotation_degrees))
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    xy = points[..., :2] - np.array([0.0, float(translation)], dtype=np.float64)
    transformed = xy @ rotation
    return transformed.astype(np.float32)


def _perturb_lead_vectors(vectors: Any, rotation_degrees: float) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1] < 2:
        raise ValueError(f"Expected vectors with at least two coordinates, got {vectors.shape}")
    yaw = math.radians(float(rotation_degrees))
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return (vectors[..., :2] @ rotation).astype(np.float32)


def _normal_sparse_from_perturbed_sparse(
    translation: float,
    rotation_degrees: float,
) -> np.ndarray:
    perturbed_from_normal_lead = np.eye(4, dtype=np.float64)
    yaw = math.radians(float(rotation_degrees))
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    perturbed_from_normal_lead[:2, :2] = rotation.T
    perturbed_from_normal_lead[:2, 3] = -rotation.T @ np.array(
        [0.0, float(translation)], dtype=np.float64
    )
    normal_lead_from_perturbed_lead = np.linalg.inv(perturbed_from_normal_lead)
    swap = LEFTHAND_EGO_TO_LIDAR.astype(np.float64)
    return (swap @ normal_lead_from_perturbed_lead @ swap).astype(np.float32)


def _extract_zip_member(
    zf: zipfile.ZipFile,
    member: str,
    destination: Path,
) -> None:
    source_size = int(zf.getinfo(member).file_size)
    if destination.is_file() and destination.stat().st_size == source_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(zf.read(member))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_pickle_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class _ShardedInfosIterable:
    """Iterate generated shards without retaining the full dataset in RAM."""

    def __init__(self, root: Path, shards: list[dict[str, Any]], num_samples: int):
        self.root = root
        self.shards = shards
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        for shard in self.shards:
            with open(self.root / shard["file"], "rb") as f:
                yield from pickle.load(f)


def _command_to_one_hot(command: Any, max_dim: int = 6) -> np.ndarray:
    if hasattr(command, "value"):
        command = command.value
    if isinstance(command, str):
        normalized = command.strip().upper().replace(" ", "").replace("_", "")
        command = {
            "LEFT": 1,
            "TURNLEFT": 1,
            "RIGHT": 2,
            "TURNRIGHT": 2,
            "STRAIGHT": 3,
            "GOSTRAIGHT": 3,
            "LANEFOLLOW": 4,
            "FOLLOWLANE": 4,
            "CHANGELANELEFT": 5,
            "CHANGELANERIGHT": 6,
        }.get(normalized, 4)
    try:
        raw_command = int(command)
    except Exception:
        raw_command = 4
    if raw_command < 0:
        raw_command = 4
    command_idx = raw_command - 1
    if command_idx < 0 or command_idx >= max_dim:
        command_idx = 3
    one_hot = np.zeros(max_dim, dtype=np.float32)
    one_hot[command_idx] = 1.0
    return one_hot


def _dedupe_target_points_and_commands(
    target_points: Any,
    commands: Any,
) -> tuple[list[list[float]], list[Any]]:
    if target_points is None or commands is None:
        return [], []

    target_points = list(target_points)
    filtered_targets = []
    filtered_commands = []
    for point, command in zip(target_points, commands):
        point_xy = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
        if point_xy.shape[0] != 2 or not np.all(np.isfinite(point_xy)):
            continue
        if (
            len(target_points) == 2
            or not filtered_targets
            or not np.allclose(point_xy, filtered_targets[-1])
        ):
            filtered_targets.append(point_xy.tolist())
            filtered_commands.append(command)
    return filtered_targets, filtered_commands


def _inverse_conversion_2d(point: Any, translation: Any, yaw: Any) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)[:2]
    yaw = _as_float(yaw)
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return (rotation.T @ (point - translation)).astype(np.float32)


def _lead_navigation_targets(
    meta: dict[str, Any],
    translation: float = 0.0,
    rotation_degrees: float = 0.0,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Any]]:
    target_points, commands = _dedupe_target_points_and_commands(
        meta.get("next_target_points_3.25"),
        meta.get("next_commands_3.25"),
    )
    if len(target_points) < 2:
        return None

    previous_global = target_points[0]
    current_global = target_points[1]
    next_global = target_points[2] if len(target_points) > 2 else current_global
    ego_position = meta.get("pos_global", [0.0, 0.0])
    ego_yaw = meta.get("theta", 0.0)

    ego_points = np.stack(
        [
            _inverse_conversion_2d(previous_global, ego_position, ego_yaw),
            _inverse_conversion_2d(current_global, ego_position, ego_yaw),
            _inverse_conversion_2d(next_global, ego_position, ego_yaw),
        ]
    )
    previous, current, next_point = _perturb_lead_points(
        ego_points, translation, rotation_degrees
    )
    command = commands[0] if commands else 4
    return previous, current, next_point, command


def _ego_status(
    meta: dict[str, Any],
    rotation_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    status = np.zeros(6, dtype=np.float32)
    status[0] = _require_finite_float(meta, "speed")
    status[1] = _as_float(meta.get("accel_x"))
    status[2] = _as_float(meta.get("accel_y"))
    status[3] = _as_float(meta.get("angular_velocity_x"))
    status[4] = _as_float(meta.get("angular_velocity_y"))
    status[5] = _as_float(meta.get("steer"))
    status[1:3] = _perturb_lead_vectors(status[1:3], rotation_degrees)
    status[3:5] = _perturb_lead_vectors(status[3:5], rotation_degrees)

    mask = np.ones(6, dtype=np.float32)
    if abs(status[0]) > 40.0:
        mask[0] = 0.0
    if np.linalg.norm(status[1:3]) > 80.0:
        mask[1:3] = 0.0
    return status, mask


def _ego_future(
    meta: dict[str, Any],
    horizon: int,
    step: int,
    translation: float = 0.0,
    rotation_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    future_positions = np.asarray(meta.get("future_positions", []), dtype=np.float32)
    if future_positions.ndim != 2 or future_positions.shape[1] < 2:
        return np.zeros((horizon, 2), dtype=np.float32), np.zeros(horizon, dtype=np.float32)

    absolute = np.zeros((horizon + 1, 2), dtype=np.float32)
    masks = np.zeros(horizon, dtype=np.float32)
    for i in range(horizon):
        src_idx = (i + 1) * step
        if src_idx < len(future_positions):
            absolute[i + 1] = _perturb_lead_points(
                future_positions[src_idx, :2], translation, rotation_degrees
            )
            masks[i] = 1.0
        else:
            absolute[i + 1] = absolute[i]
    offsets = absolute[1:] - absolute[:-1]
    offsets[masks == 0] = 0.0
    return offsets.astype(np.float32), masks.astype(np.float32)


def _compute_trajectory_stats(
    infos: list[dict[str, Any]],
    *,
    frame_stride: int,
    frame_offset: int,
    route_quality_filter: str,
    route_skip_first: int,
    route_skip_last: int,
    ego_plan_horizon: int,
    ego_plan_step: int,
    agent_motion_horizon: int,
    agent_motion_step: int,
    sensor_perturbation_probability: float,
) -> dict[str, Any]:
    delta_sum = np.zeros(2, dtype=np.float64)
    delta_square_sum = np.zeros(2, dtype=np.float64)
    delta_min = np.full(2, np.inf, dtype=np.float64)
    delta_max = np.full(2, -np.inf, dtype=np.float64)
    absolute_min = np.full(2, np.inf, dtype=np.float64)
    absolute_max = np.full(2, -np.inf, dtype=np.float64)
    valid_waypoints = 0.0
    route_abs_sum = 0.0
    route_abs_count = 0.0
    route_final_abs_sum = 0.0
    route_final_abs_count = 0.0
    source_sim_hz_values = set()
    data_save_freq_values = set()
    jpeg_storage_quality_values = set()
    perturbation_probability = float(sensor_perturbation_probability)
    if not 0.0 <= perturbation_probability <= 1.0:
        raise ValueError(f"sensor perturbation probability must be in [0, 1], got {perturbation_probability}")
    perturbed_samples = 0

    for info in infos:
        source_sim_hz_values.add(int(info["source_sim_hz"]))
        data_save_freq_values.add(int(info["data_save_freq"]))
        jpeg_storage_quality_values.add(int(info["jpeg_storage_quality"]))
        masks = np.asarray(info["gt_ego_fut_masks_4hz"], dtype=np.float64).reshape(-1) > 0
        variants = [("gt_ego_fut_trajs_4hz", "gt_route", 1.0 - perturbation_probability)]
        if perturbation_probability > 0.0:
            required = {
                "gt_ego_fut_trajs_4hz_perturbed",
                "gt_route_perturbed",
                "perturbed_image_path",
            }
            if not required.issubset(info):
                raise RuntimeError(f"Missing sensor perturbation fields in {info.get('folder')} frame {info.get('frame_idx')}")
            variants.append(
                ("gt_ego_fut_trajs_4hz_perturbed", "gt_route_perturbed", perturbation_probability)
            )
            perturbed_samples += 1

        for trajectory_key, route_key, weight in variants:
            if weight <= 0.0:
                continue
            deltas = np.asarray(info[trajectory_key], dtype=np.float64).reshape(-1, 2)
            valid_deltas = deltas[masks]
            if len(valid_deltas) > 0:
                delta_sum += weight * valid_deltas.sum(axis=0)
                delta_square_sum += weight * np.square(valid_deltas).sum(axis=0)
                delta_min = np.minimum(delta_min, valid_deltas.min(axis=0))
                delta_max = np.maximum(delta_max, valid_deltas.max(axis=0))
                valid_waypoints += weight * len(valid_deltas)

                absolute = np.cumsum(deltas, axis=0)[masks]
                absolute_min = np.minimum(absolute_min, absolute.min(axis=0))
                absolute_max = np.maximum(absolute_max, absolute.max(axis=0))

            route = np.asarray(info[route_key], dtype=np.float64).reshape(-1, 2)
            route_abs_sum += weight * np.abs(route).sum()
            route_abs_count += weight * route.size
            route_final_abs_sum += weight * np.abs(route[-1]).sum()
            route_final_abs_count += weight * route.shape[-1]

    if valid_waypoints == 0:
        raise RuntimeError("Cannot compute LEAD trajectory statistics without valid waypoints")
    if (
        len(source_sim_hz_values) != 1
        or len(data_save_freq_values) != 1
        or len(jpeg_storage_quality_values) != 1
    ):
        raise RuntimeError(
            "LEAD trajectory statistics require uniform source/save/image settings, got "
            f"source_sim_hz={sorted(source_sim_hz_values)} "
            f"data_save_freq={sorted(data_save_freq_values)} "
            f"jpeg_storage_quality={sorted(jpeg_storage_quality_values)}"
        )
    source_sim_hz = source_sim_hz_values.pop()
    data_save_freq = data_save_freq_values.pop()
    jpeg_storage_quality = jpeg_storage_quality_values.pop()

    delta_mu = delta_sum / valid_waypoints
    delta_variance = delta_square_sum / valid_waypoints - np.square(delta_mu)
    delta_std = np.sqrt(np.maximum(delta_variance, 0.0))
    route_zero_ade = route_abs_sum / max(route_abs_count, 1)
    route_zero_fde = route_final_abs_sum / max(route_final_abs_count, 1)

    return {
        "stats_version": 7,
        "lead_stage1_infos_version": LEAD_STAGE1_INFOS_VERSION,
        "sensor_perturbation_version": LEAD_SENSOR_PERTURBATION_VERSION,
        "sensor_perturbation_probability": perturbation_probability,
        "detection_filter_version": LEAD_DETECTION_FILTER_VERSION,
        "detection_class_mapping_version": LEAD_DETECTION_CLASS_MAPPING_VERSION,
        "detection_z_range": list(LEAD_DETECTION_Z_RANGE),
        "detection_visibility_filter": "lead_b2d_visibility_v1",
        "detection_traffic_control_filter": LEAD_DETECTION_TRAFFIC_CONTROL_FILTER,
        "detection_box_dimension_order": "length_width_height",
        "detection_box_center": "carla_bbox_center",
        "bbox_center_offset_version": LEAD_BBOX_CENTER_OFFSET_VERSION,
        "num_sensor_perturbation_samples": int(perturbed_samples),
        "perturbed_image_layout": "rgb_perturbated_strip",
        "frame_sample_stride": int(frame_stride),
        "frame_sample_offset": int(frame_offset),
        "route_quality_filter": str(route_quality_filter),
        "route_skip_first": int(route_skip_first),
        "route_skip_last": int(route_skip_last),
        "source_sim_hz": int(source_sim_hz),
        "data_save_freq": int(data_save_freq),
        "jpeg_storage_quality": int(jpeg_storage_quality),
        "perception_sample_hz": float(source_sim_hz) / float(data_save_freq * frame_stride),
        "planning_step_sim_frames": int(ego_plan_step),
        "planning_freq_hz": 20.0 / float(ego_plan_step),
        "planning_horizon_steps": int(ego_plan_horizon),
        "planning_horizon_seconds": float(ego_plan_horizon * ego_plan_step) / 20.0,
        "agent_motion_step_sim_frames": int(agent_motion_step),
        "agent_motion_freq_hz": 20.0 / float(agent_motion_step),
        "agent_motion_horizon_steps": int(agent_motion_horizon),
        "agent_motion_horizon_seconds": float(agent_motion_horizon * agent_motion_step) / 20.0,
        "num_train_samples": len(infos),
        "valid_waypoints": int(round(valid_waypoints)),
        "delta_mu": delta_mu.tolist(),
        "delta_std": delta_std.tolist(),
        "delta_min": delta_min.tolist(),
        "delta_max": delta_max.tolist(),
        "absolute_min": absolute_min.tolist(),
        "absolute_max": absolute_max.tolist(),
        "route_zero_ade": float(route_zero_ade),
        "route_zero_fde": float(route_zero_fde),
        "route_zero_total": float(route_zero_ade + route_zero_fde),
    }


def _circle_line_intersections(
    circle_center: np.ndarray,
    circle_radius: float,
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> list[np.ndarray]:
    p1 = np.asarray(point_1, dtype=np.float64) - circle_center
    p2 = np.asarray(point_2, dtype=np.float64) - circle_center
    dx, dy = p2 - p1
    dr2 = float(dx * dx + dy * dy)
    if dr2 <= 1e-12:
        return []
    determinant = float(p1[0] * p2[1] - p2[0] * p1[1])
    discriminant = circle_radius * circle_radius * dr2 - determinant * determinant
    if discriminant < -1e-9:
        return []
    discriminant = max(discriminant, 0.0)
    root = math.sqrt(discriminant)
    signs = (1.0, -1.0) if dy < 0 else (-1.0, 1.0)
    intersections = []
    for sign in signs:
        x = (
            determinant * dy
            + sign * (-1.0 if dy < 0 else 1.0) * dx * root
        ) / dr2
        y = (-determinant * dx + sign * abs(dy) * root) / dr2
        intersections.append(circle_center + np.array([x, y], dtype=np.float64))
    if discriminant <= 1e-9:
        return intersections[:1]
    return intersections


def _tfv6_iterative_route(
    route: np.ndarray,
    num_points: int,
    first_distance: float,
) -> np.ndarray:
    interpolated = []
    last_interpolated = np.zeros(2, dtype=np.float64)
    current_route_index = 0
    current_point = route[current_route_index].astype(np.float64)
    last_point = np.zeros(2, dtype=np.float64)
    radius = float(first_distance)
    first_iteration = True

    while len(interpolated) < num_points:
        if not first_iteration:
            current_route_index += 1
            last_point = current_point

        if current_route_index < len(route):
            current_point = route[current_route_index].astype(np.float64)
            intersections = _circle_line_intersections(
                last_interpolated,
                radius,
                last_interpolated,
                current_point,
            )
        elif len(route) >= 2:
            current_point = route[-1].astype(np.float64)
            last_point = route[-2].astype(np.float64)
            intersections = _circle_line_intersections(
                last_interpolated,
                radius,
                last_point,
                current_point,
            )
        else:
            return np.zeros((num_points, 2), dtype=np.float32)

        if not intersections:
            return np.zeros((num_points, 2), dtype=np.float32)
        if len(intersections) == 1:
            intersection = intersections[0]
        else:
            direction = current_point - last_point
            intersection = max(
                intersections,
                key=lambda point: float(np.dot(point, direction)),
            )

        last_interpolated = intersection
        interpolated.append(intersection)
        radius = 1.0
        first_iteration = False

    return np.asarray(interpolated, dtype=np.float32)


def _lead_route_target(
    meta: dict[str, Any],
    smoothing_points: int = 20,
    prediction_points: int = 10,
    first_distance: float = 2.5,
    spacing: float = 1.0,
    translation: float = 0.0,
    rotation_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    route = np.asarray(meta.get("route", []), dtype=np.float32)
    if route.ndim != 2 or route.shape[1] < 2 or len(route) == 0:
        return np.zeros((prediction_points, 2), dtype=np.float32), np.zeros(prediction_points, dtype=np.float32)

    route = route[:smoothing_points, :2]
    route = _perturb_lead_points(route, translation, rotation_degrees)
    _, unique_indices = np.unique(route, return_index=True, axis=0)
    route = route[np.sort(unique_indices).astype(np.int64)]
    if len(route) < 2:
        return np.zeros((prediction_points, 2), dtype=np.float32), np.zeros(prediction_points, dtype=np.float32)
    if abs(spacing - 1.0) > 1e-6:
        raise ValueError("TFV6 route interpolation requires 1m spacing after the first point")

    smoothed = _tfv6_iterative_route(route, smoothing_points, first_distance)
    masks = np.ones(prediction_points, dtype=np.float32)
    if not np.any(smoothed):
        masks[:] = 0.0
    return smoothed[:prediction_points], masks


def _is_parked_static_mesh(box: dict[str, Any]) -> bool:
    return (
        str(box.get("class", "")).lower() == "static"
        and "ParkedVehicles" in str(box.get("mesh_path", ""))
    )


def _box_detection_name(box: dict[str, Any]) -> Optional[str]:
    raw_name = str(box.get("class", "")).lower()
    if raw_name == "ego_car":
        return None
    type_name = str(box.get("type_id", "")).lower()
    if raw_name == "static":
        if _is_parked_static_mesh(box):
            return "car"
        if type_name in TFV6_STATIC_TYPE_WHITELIST:
            return "traffic_cone"
        return None
    return (
        DET_TYPE_ID_MAP.get(type_name)
        or DET_CLASS_MAP.get(raw_name)
        or DET_CLASS_MAP.get(type_name)
    )


def _box_local_bbox_center_offset(box: dict[str, Any]) -> np.ndarray:
    raw_name = str(box.get("class", "")).lower()
    if raw_name == "car":
        type_name = str(box.get("type_id", "")).lower()
        offset = CARLA_VEHICLE_BBOX_LOCAL_CENTERS.get(type_name)
        if offset is None:
            raise MissingBBoxCenterOffsetError(
                f"Missing CARLA 0.9.15 bbox center offset for vehicle type {type_name!r}"
            )
        return np.asarray(offset, dtype=np.float64)
    if raw_name == "static":
        mesh_path = str(box.get("mesh_path", ""))
        if _is_parked_static_mesh(box):
            offset = CARLA_STATIC_MESH_BBOX_LOCAL_CENTERS.get(mesh_path)
            if offset is None:
                raise MissingBBoxCenterOffsetError(
                    f"Missing CARLA 0.9.15 bbox center offset for static mesh {mesh_path!r}"
                )
            return np.asarray(offset, dtype=np.float64)
        type_name = str(box.get("type_id", "")).lower()
        if type_name == "static.prop.constructioncone":
            return np.asarray([0.0, 0.0, 0.292890608], dtype=np.float64)
        if type_name == "static.prop.trafficwarning":
            return np.asarray([0.0, 0.0, 1.784765601], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _box_center_in_lead_ego(
    box: dict[str, Any],
    ego_matrix: Any = None,
) -> np.ndarray:
    actor_origin = np.asarray(
        box.get("position", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(-1)
    if actor_origin.size < 3:
        raise ValueError(f"Expected 3D actor position, got {actor_origin.shape}")
    actor_origin = actor_origin[:3]
    local_offset = _box_local_bbox_center_offset(box)
    if not np.any(local_offset):
        return actor_origin.astype(np.float32)

    actor_matrix = np.asarray(box.get("matrix", []), dtype=np.float64)
    ego_matrix = np.asarray(ego_matrix if ego_matrix is not None else [], dtype=np.float64)
    if (
        actor_matrix.shape == (4, 4)
        and ego_matrix.shape == (4, 4)
        and np.all(np.isfinite(actor_matrix))
        and np.all(np.isfinite(ego_matrix))
    ):
        relative_rotation = ego_matrix[:3, :3].T @ actor_matrix[:3, :3]
        return (actor_origin + relative_rotation @ local_offset).astype(np.float32)

    yaw = _as_float(box.get("yaw"))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    yaw_rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (actor_origin + yaw_rotation @ local_offset).astype(np.float32)


def _agent_future_offsets(
    box: dict[str, Any],
    horizon: int = 6,
    step: int = 10,
    max_step_distance: float = 10.0,
    translation: float = 0.0,
    rotation_degrees: float = 0.0,
    validity_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert LEAD actor future positions to UniDriveVLA motion offsets.

    LEAD stores 20Hz actor futures for two seconds. The default spacing of 10
    preserves the legacy six-slot tensor, whose last two slots are invalid;
    horizon=8 and step=5 uses all native 4Hz samples.
    """
    future_positions = np.asarray(box.get("future_positions", []), dtype=np.float32)
    if future_positions.ndim != 2 or future_positions.shape[1] < 2:
        return np.zeros((horizon, 2), dtype=np.float32), np.zeros(horizon, dtype=np.float32)

    if validity_mask is not None:
        validity_mask = np.asarray(validity_mask, dtype=np.float32).reshape(-1)
        if len(validity_mask) != horizon:
            raise ValueError(
                f"Expected actor validity mask with {horizon} entries, got {len(validity_mask)}"
            )

    local_offset = _box_local_bbox_center_offset(box)[:2]
    if np.any(local_offset):
        future_yaws = np.asarray(box.get("future_yaws", []), dtype=np.float64).reshape(-1)
        if len(future_yaws) < len(future_positions):
            raise ValueError(
                "Vehicle future_positions require matching future_yaws to convert "
                f"actor origins to bbox centers, got {len(future_positions)} positions "
                f"and {len(future_yaws)} yaws"
            )
        center_positions = future_positions.copy()
        cos_yaw = np.cos(future_yaws[: len(future_positions)])
        sin_yaw = np.sin(future_yaws[: len(future_positions)])
        center_positions[:, 0] += cos_yaw * local_offset[0] - sin_yaw * local_offset[1]
        center_positions[:, 1] += sin_yaw * local_offset[0] + cos_yaw * local_offset[1]
        future_positions = center_positions

    absolute = np.zeros((horizon + 1, 2), dtype=np.float32)
    masks = np.zeros(horizon, dtype=np.float32)
    absolute[0] = _perturb_lead_points(
        future_positions[0, :2], translation, rotation_degrees
    )
    last_valid = absolute[0].copy()
    for i in range(horizon):
        src_idx = (i + 1) * step
        if src_idx < len(future_positions):
            candidate = _perturb_lead_points(
                future_positions[src_idx, :2], translation, rotation_degrees
            )
            is_valid = (
                bool(validity_mask[i] > 0)
                if validity_mask is not None
                else np.linalg.norm(candidate - last_valid) <= max_step_distance
            )
            if is_valid:
                absolute[i + 1] = candidate
                last_valid = candidate
                masks[i] = 1.0
            else:
                absolute[i + 1] = last_valid
        else:
            absolute[i + 1] = last_valid

    offsets = absolute[1:] - absolute[:-1]
    offsets[masks == 0] = 0.0
    return offsets.astype(np.float32), masks.astype(np.float32)


def _right_hand_lidar2global(
    meta: dict[str, Any],
    translation: float = 0.0,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    ego2world_left = np.asarray(meta.get("ego_matrix", np.eye(4)), dtype=np.float32)
    # Match Bench2Drive's map convention: global y is flipped to right-handed
    # coordinates and the ego frame is converted to the lidar-like frame used by
    # Sparse4D map projection.
    lidar2global = LEFT2RIGHT @ ego2world_left @ np.linalg.inv(LEFTHAND_EGO_TO_LIDAR)
    normal_from_perturbed = _normal_sparse_from_perturbed_sparse(
        translation, rotation_degrees
    )
    return lidar2global @ normal_from_perturbed


def _saved_frame_timestamp(meta: dict[str, Any], frame_idx: int, source_hz: int = 20) -> float:
    dataset_info = meta.get("dataset_information") or {}
    data_save_freq = int(_as_float(dataset_info.get("data_save_freq"), 1.0))
    data_save_freq = max(1, data_save_freq)
    return float(frame_idx) * float(data_save_freq) / float(source_hz)


def _lead_ego_boxes_to_sparse4d(boxes: np.ndarray) -> np.ndarray:
    """Convert box centers and headings into the Sparse4D local frame.

    Box dimensions remain intrinsic (length, width, height); changing the
    coordinate basis must not swap the object's length and width.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 9).copy()
    if len(boxes) == 0:
        return boxes
    boxes[:, [0, 1]] = boxes[:, [1, 0]]
    boxes[:, 6] = (np.pi / 2.0) - boxes[:, 6]
    boxes[:, [7, 8]] = boxes[:, [8, 7]]
    return boxes


def _lead_ego_trajs_to_sparse4d(trajs: np.ndarray) -> np.ndarray:
    trajs = np.asarray(trajs, dtype=np.float32).copy()
    if trajs.size == 0:
        return trajs
    trajs[..., [0, 1]] = trajs[..., [1, 0]]
    return trajs


def _box_to_unidrive(
    box: dict[str, Any],
    translation: float = 0.0,
    rotation_degrees: float = 0.0,
    ego_matrix: Any = None,
) -> tuple[str, np.ndarray, int] | None:
    name = _box_detection_name(box)
    if name is None:
        return None

    pos = _box_center_in_lead_ego(box, ego_matrix=ego_matrix)
    extent = np.asarray(box.get("extent", [0.5, 0.5, 0.5]), dtype=np.float32)
    if pos.shape[0] < 3 or extent.shape[0] < 3:
        return None

    pos = pos.copy()
    pos[:2] = _perturb_lead_points(pos[:2], translation, rotation_degrees)
    yaw = _as_float(box.get("yaw")) - math.radians(float(rotation_degrees))
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    speed = _as_float(box.get("speed"))
    vx = speed * math.cos(yaw)
    vy = speed * math.sin(yaw)

    # UniDriveVLA/B2D boxes are center xyz + full lwh + yaw + planar velocity.
    gt_box = np.array(
        [
            pos[0],
            pos[1],
            pos[2],
            max(float(extent[0]) * 2.0, 0.1),
            max(float(extent[1]) * 2.0, 0.1),
            max(float(extent[2]) * 2.0, 0.1),
            yaw,
            vx,
            vy,
        ],
        dtype=np.float32,
    )
    return name, gt_box, int(box.get("id", -1))


def _passes_detection_filter(
    box: dict[str, Any],
    ego_matrix: Any = None,
) -> bool:
    """Apply the TFv6 traffic-control, height, and visibility guards."""
    raw_name = str(box.get("class", "")).lower()
    if raw_name == "ego_car":
        return True

    affects_ego = box.get("affects_ego", False)
    if isinstance(affects_ego, str):
        affects_ego = affects_ego.strip().lower() in {"1", "true", "yes"}
    else:
        affects_ego = bool(affects_ego)
    if raw_name == "traffic_light":
        state = str(box.get("state", "")).strip().lower()
        if not affects_ego or state == "green":
            return False
    elif raw_name == "stop_sign" and not affects_ego:
        return False

    position = (
        _box_center_in_lead_ego(box, ego_matrix)
        if ego_matrix is not None
        else np.asarray(box.get("position", []), dtype=np.float32).reshape(-1)
    )
    if position.size < 3 or not np.isfinite(position[2]):
        return False
    if not (LEAD_DETECTION_Z_RANGE[0] < float(position[2]) < LEAD_DETECTION_Z_RANGE[1]):
        return False

    num_points = _as_float(box.get("num_points"), -1.0)
    visible_pixels = _as_float(box.get("visible_pixels"), -1.0)
    semantic_id = int(_as_float(box.get("transfuser_semantics_id"), -1.0))
    if raw_name in {"walker", "pedestrian"}:
        min_points, min_visible = 5.0, 15.0
    elif raw_name == "static" and semantic_id == 1:
        # TFV6 applies the stricter threshold only to parked/static vehicles.
        # Static obstacles such as warning signs use a different semantic class.
        min_points, min_visible = 3.0, 5.0
    elif raw_name in {
        "car",
        "van",
        "truck",
        "bus",
        "bicycle",
        "bike",
        "motorcycle",
        "static_prop_car",
    }:
        min_points, min_visible = 1.0, 1.0
    else:
        return True

    invisible_to_both = (
        0.0 <= num_points < min_points
        and 0.0 <= visible_pixels < min_visible
    )
    return not invisible_to_both



def _route_member_prefix(names: list[str]) -> str:
    for name in names:
        if name.endswith("/metas/") or "/metas/" in name:
            return name.split("/metas/")[0] + "/"
    raise RuntimeError("No metas directory found in zip")


def _passes_tf_v6_posttrain_quality(
    zf: zipfile.ZipFile,
    names: set[str],
    route_prefix: str,
) -> bool:
    results_member = f"{route_prefix}results.json"
    if results_member not in names or f"{route_prefix}metas/0000.pkl" not in names:
        return False
    try:
        result = json.loads(zf.read(results_member).decode("utf-8"))
    except Exception:
        return False

    status = str(result.get("status", ""))
    score = float((result.get("scores") or {}).get("score_composed", -1.0))
    num_infractions = int(result.get("num_infractions", 0))
    infractions = result.get("infractions") or {}
    min_speed_count = len(infractions.get("min_speed_infractions") or [])
    outside_lane_count = len(infractions.get("outside_route_lanes") or [])

    acceptable = score == 100.0 or status == "Perfect"
    acceptable = acceptable or (score < 100.0 and num_infractions == min_speed_count)
    acceptable = acceptable or (
        status == "Completed"
        and num_infractions == min_speed_count + outside_lane_count
    )
    failed_statuses = {
        "Failed - Agent couldn't be set up",
        "Failed",
        "Failed - Simulation crashed",
        "Failed - Agent crashed",
    }
    return acceptable and status not in failed_statuses


def _iter_zip_infos(
    zip_path: Path,
    image_root: Path,
    require_images: bool,
    frame_stride: int,
    frame_offset: int,
    ego_plan_horizon: int,
    ego_plan_step: int,
    agent_motion_horizon: int,
    agent_motion_step: int,
    route_quality_filter: str,
    route_skip_first: int,
    route_skip_last: int,
    sensor_perturbation_probability: float,
    extract_perturbed_images: bool,
):
    scenario = zip_path.parent.name
    route_id = zip_path.stem
    perturbation_enabled = float(sensor_perturbation_probability) > 0.0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        names_set = set(names)
        route_prefix = _route_member_prefix(names)
        if route_quality_filter == "tfv6_posttrain" and not _passes_tf_v6_posttrain_quality(
            zf, names_set, route_prefix
        ):
            return

        meta_members = sorted(n for n in names if n.startswith(route_prefix + "metas/") and n.endswith(".pkl"))
        bbox_members = {n for n in names if n.startswith(route_prefix + "bboxes/") and n.endswith(".pkl")}
        num_seq = len(bbox_members)
        if num_seq <= route_skip_first + route_skip_last:
            return

        for meta_member in meta_members:
            frame_idx = int(Path(meta_member).stem)
            if frame_idx < route_skip_first or frame_idx >= num_seq - route_skip_last:
                continue
            if frame_idx % frame_stride != frame_offset:
                continue
            bbox_member = f"{route_prefix}bboxes/{frame_idx:04d}.pkl"
            if bbox_member not in bbox_members:
                continue

            image_rel = Path("rgb") / scenario / route_id / "rgb" / f"{frame_idx:04d}.jpg"
            if require_images and not (image_root / image_rel).exists():
                continue

            perturbed_image_rel = (
                Path("rgb_perturbated")
                / scenario
                / route_id
                / "rgb_perturbated"
                / f"{frame_idx:04d}.jpg"
            )
            if perturbation_enabled:
                perturbed_member = f"{route_prefix}rgb_perturbated/{frame_idx:04d}.jpg"
                if perturbed_member not in names_set:
                    continue
                perturbed_destination = image_root / perturbed_image_rel
                if extract_perturbed_images:
                    try:
                        _extract_zip_member(zf, perturbed_member, perturbed_destination)
                    except Exception as exc:
                        print(
                            f"[warn] skip LEAD perturbation image zip={zip_path} "
                            f"frame={frame_idx:04d} error={type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                elif require_images and not perturbed_destination.is_file():
                    continue

            try:
                meta = _read_lzma_pickle_from_zip(zf, meta_member)
                boxes = _read_lzma_pickle_from_zip(zf, bbox_member)
            except Exception as exc:
                print(
                    f"[warn] skip corrupt LEAD sample zip={zip_path} "
                    f"frame={frame_idx:04d} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            perturbation_translation, perturbation_rotation = _sensor_perturbation(meta)
            gt_names, gt_boxes, gt_ids = [], [], []
            gt_agent_fut_trajs, gt_agent_fut_masks = [], []
            gt_boxes_perturbed, gt_agent_fut_trajs_perturbed = [], []
            ego_extent = None
            for box in boxes:
                if str(box.get("class", "")).lower() == "ego_car":
                    ego_extent = np.asarray(box.get("extent", []), dtype=np.float32)
                if not _passes_detection_filter(box, meta.get("ego_matrix")):
                    continue
                converted = _box_to_unidrive(
                    box, ego_matrix=meta.get("ego_matrix")
                )
                if converted is None:
                    continue
                name, gt_box, gt_id = converted
                gt_names.append(name)
                gt_boxes.append(gt_box)
                gt_ids.append(gt_id)
                fut_traj, fut_mask = _agent_future_offsets(
                    box, horizon=agent_motion_horizon, step=agent_motion_step
                )
                gt_agent_fut_trajs.append(fut_traj)
                gt_agent_fut_masks.append(fut_mask)

                if perturbation_enabled:
                    converted_perturbed = _box_to_unidrive(
                        box,
                        perturbation_translation,
                        perturbation_rotation,
                        ego_matrix=meta.get("ego_matrix"),
                    )
                    if converted_perturbed is None:
                        raise RuntimeError("Perturbed box conversion changed the instance set")
                    perturbed_name, perturbed_box, perturbed_id = converted_perturbed
                    if perturbed_name != name or perturbed_id != gt_id:
                        raise RuntimeError("Perturbed box conversion changed instance identity")
                    gt_boxes_perturbed.append(perturbed_box)
                    perturbed_fut_traj, perturbed_fut_mask = _agent_future_offsets(
                        box,
                        horizon=agent_motion_horizon,
                        step=agent_motion_step,
                        translation=perturbation_translation,
                        rotation_degrees=perturbation_rotation,
                        validity_mask=fut_mask,
                    )
                    if not np.array_equal(perturbed_fut_mask, fut_mask):
                        raise RuntimeError("Perturbed actor trajectory changed its validity mask")
                    gt_agent_fut_trajs_perturbed.append(perturbed_fut_traj)

            status, status_mask = _ego_status(meta)
            traj_4hz, mask_4hz = _ego_future(meta, horizon=ego_plan_horizon, step=ego_plan_step)
            traj_5hz, mask_5hz = _ego_future(meta, horizon=6, step=4)
            traj_2hz, mask_2hz = _ego_future(meta, horizon=6, step=10)
            route_target, route_mask = _lead_route_target(meta)
            if not np.all(route_mask > 0):
                continue
            navigation_targets = _lead_navigation_targets(meta)
            if navigation_targets is None:
                continue
            target_point_previous, target_point, target_point_next, command = navigation_targets

            gt_boxes_arr = _lead_ego_boxes_to_sparse4d(np.asarray(gt_boxes, dtype=np.float32))
            gt_agent_fut_trajs_arr = _lead_ego_trajs_to_sparse4d(
                np.asarray(gt_agent_fut_trajs, dtype=np.float32).reshape(
                    -1, agent_motion_horizon, 2
                )
            )
            perturbed_fields = {}
            if perturbation_enabled:
                status_perturbed, status_mask_perturbed = _ego_status(
                    meta, rotation_degrees=perturbation_rotation
                )
                traj_4hz_perturbed, mask_4hz_perturbed = _ego_future(
                    meta,
                    horizon=ego_plan_horizon,
                    step=ego_plan_step,
                    translation=perturbation_translation,
                    rotation_degrees=perturbation_rotation,
                )
                traj_5hz_perturbed, mask_5hz_perturbed = _ego_future(
                    meta,
                    horizon=6,
                    step=4,
                    translation=perturbation_translation,
                    rotation_degrees=perturbation_rotation,
                )
                traj_2hz_perturbed, mask_2hz_perturbed = _ego_future(
                    meta,
                    horizon=6,
                    step=10,
                    translation=perturbation_translation,
                    rotation_degrees=perturbation_rotation,
                )
                route_target_perturbed, route_mask_perturbed = _lead_route_target(
                    meta,
                    translation=perturbation_translation,
                    rotation_degrees=perturbation_rotation,
                )
                navigation_targets_perturbed = _lead_navigation_targets(
                    meta,
                    translation=perturbation_translation,
                    rotation_degrees=perturbation_rotation,
                )
                if navigation_targets_perturbed is None or not np.all(route_mask_perturbed > 0):
                    continue
                (
                    target_point_previous_perturbed,
                    target_point_perturbed,
                    target_point_next_perturbed,
                    command_perturbed,
                ) = navigation_targets_perturbed
                if command_perturbed != command:
                    raise RuntimeError("Sensor perturbation changed the navigation command")
                if not (
                    np.array_equal(status_mask_perturbed, status_mask)
                    and np.array_equal(mask_4hz_perturbed, mask_4hz)
                    and np.array_equal(mask_5hz_perturbed, mask_5hz)
                    and np.array_equal(mask_2hz_perturbed, mask_2hz)
                ):
                    raise RuntimeError("Sensor perturbation changed a supervision validity mask")

                perturbed_fields = dict(
                    sensor_perturbation_version=LEAD_SENSOR_PERTURBATION_VERSION,
                    perturbation_translation=float(perturbation_translation),
                    perturbation_rotation=float(perturbation_rotation),
                    perturbed_image_path=perturbed_image_rel.as_posix(),
                    lidar2global_perturbed=_right_hand_lidar2global(
                        meta,
                        perturbation_translation,
                        perturbation_rotation,
                    ).astype(np.float32),
                    gt_boxes_perturbed=_lead_ego_boxes_to_sparse4d(
                        np.asarray(gt_boxes_perturbed, dtype=np.float32)
                    ),
                    gt_agent_fut_trajs_perturbed=_lead_ego_trajs_to_sparse4d(
                        np.asarray(gt_agent_fut_trajs_perturbed, dtype=np.float32).reshape(
                            -1, agent_motion_horizon, 2
                        )
                    ),
                    ego_status_perturbed=status_perturbed,
                    ego_status_mask_perturbed=status_mask_perturbed,
                    target_point_previous_perturbed=target_point_previous_perturbed,
                    target_point_perturbed=target_point_perturbed,
                    target_point_next_perturbed=target_point_next_perturbed,
                    gt_ego_fut_trajs_4hz_perturbed=traj_4hz_perturbed,
                    gt_ego_fut_masks_4hz_perturbed=mask_4hz_perturbed,
                    gt_ego_fut_trajs_5hz_perturbed=traj_5hz_perturbed,
                    gt_ego_fut_masks_5hz_perturbed=mask_5hz_perturbed,
                    gt_route_perturbed=route_target_perturbed,
                    gt_route_masks_perturbed=route_mask_perturbed,
                    gt_ego_fut_trajs_2hz_perturbed=traj_2hz_perturbed,
                    gt_ego_fut_masks_2hz_perturbed=mask_2hz_perturbed,
                )

            target_speed = _require_finite_float(meta, "target_speed", meta_member)
            brake_value = _require_finite_float(meta, "brake", meta_member)
            if brake_value not in {0.0, 1.0}:
                raise InvalidCriticalSupervisionError(
                    f"invalid 'brake'={brake_value!r} ({meta_member})"
                )
            brake = 1.0 if bool(brake_value) else 0.0
            data_save_freq = int(_as_float((meta.get("dataset_information") or {}).get("data_save_freq"), 1.0))
            data_save_freq = max(1, data_save_freq)
            jpeg_storage_quality = int(_as_float(meta.get("jpeg_storage_quality"), 30.0))

            yield dict(
                folder=f"{scenario}/{route_id}",
                scenario=scenario,
                route_id=route_id,
                town_name=meta.get("town", route_id.split("_")[0]),
                frame_idx=frame_idx,
                timestamp=_saved_frame_timestamp(meta, frame_idx),
                source_sim_hz=20,
                data_save_freq=data_save_freq,
                jpeg_storage_quality=jpeg_storage_quality,
                image_path=image_rel.as_posix(),
                sensor_information=meta.get("sensor_information", {}),
                ego_extent=ego_extent,
                ego_matrix=np.asarray(meta.get("ego_matrix", np.eye(4)), dtype=np.float32),
                lidar2global=_right_hand_lidar2global(meta).astype(np.float32),
                lead_stage1_infos_version=LEAD_STAGE1_INFOS_VERSION,
                frame_sample_stride=frame_stride,
                frame_sample_offset=frame_offset,
                route_quality_filter=route_quality_filter,
                route_skip_first=route_skip_first,
                detection_filter_version=LEAD_DETECTION_FILTER_VERSION,
                detection_z_range=list(LEAD_DETECTION_Z_RANGE),
                detection_class_mapping_version=LEAD_DETECTION_CLASS_MAPPING_VERSION,
                detection_visibility_filter="lead_b2d_visibility_v1",
                detection_traffic_control_filter=LEAD_DETECTION_TRAFFIC_CONTROL_FILTER,
                detection_box_dimension_order="length_width_height",
                detection_box_center="carla_bbox_center",
                bbox_center_offset_version=LEAD_BBOX_CENTER_OFFSET_VERSION,
                route_skip_last=route_skip_last,
                planning_freq_hz=20.0 / float(ego_plan_step),
                planning_horizon_steps=ego_plan_horizon,
                planning_horizon_seconds=float(ego_plan_horizon * ego_plan_step) / 20.0,
                agent_motion_step_sim_frames=agent_motion_step,
                agent_motion_freq_hz=20.0 / float(agent_motion_step),
                agent_motion_horizon_steps=agent_motion_horizon,
                agent_motion_horizon_seconds=float(
                    agent_motion_horizon * agent_motion_step
                ) / 20.0,
                route_points_prediction=10,
                target_speed_classes=[0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0],
                perception_coord_system="sparse4d_lateral_forward",
                planning_coord_system="lead_forward_lateral",
                gt_names=gt_names,
                gt_boxes=gt_boxes_arr,
                gt_ids=np.asarray(gt_ids, dtype=np.int32),
                gt_agent_fut_trajs=gt_agent_fut_trajs_arr,
                gt_agent_fut_masks=np.asarray(
                    gt_agent_fut_masks, dtype=np.float32
                ).reshape(-1, agent_motion_horizon),
                ego_status=status,
                ego_status_mask=status_mask,
                gt_ego_fut_cmd=_command_to_one_hot(command),
                target_point_previous=target_point_previous,
                target_point=target_point,
                target_point_next=target_point_next,
                gt_ego_fut_trajs_4hz=traj_4hz,
                gt_ego_fut_masks_4hz=mask_4hz,
                gt_ego_fut_trajs_5hz=traj_5hz,
                gt_ego_fut_masks_5hz=mask_5hz,
                gt_route=route_target,
                gt_route_masks=route_mask,
                gt_target_speed=np.asarray([target_speed], dtype=np.float32),
                gt_brake=np.asarray([brake], dtype=np.float32),
                gt_ego_fut_trajs_2hz=traj_2hz,
                gt_ego_fut_masks_2hz=mask_2hz,
                **perturbed_fields,
            )


def _process_zip(args_tuple):
    (
        zip_path,
        image_root,
        require_images,
        frame_stride,
        frame_offset,
        ego_plan_horizon,
        ego_plan_step,
        agent_motion_horizon,
        agent_motion_step,
        route_quality_filter,
        route_skip_first,
        route_skip_last,
        sensor_perturbation_probability,
        extract_perturbed_images,
    ) = args_tuple
    try:
        return list(
            _iter_zip_infos(
                Path(zip_path),
                Path(image_root),
                require_images,
                frame_stride,
                frame_offset,
                ego_plan_horizon,
                ego_plan_step,
                agent_motion_horizon,
                agent_motion_step,
                route_quality_filter,
                route_skip_first,
                route_skip_last,
                sensor_perturbation_probability,
                extract_perturbed_images,
            )
        )
    except (
        MissingBBoxCenterOffsetError,
        InvalidCriticalSupervisionError,
    ):
        raise
    except Exception as exc:
        print(
            f"[warn] skip corrupt LEAD zip={zip_path} error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return []


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UniDriveVLA Stage1 infos from LEAD official zips.")
    parser.add_argument(
        "--lead-zip-root",
        default=None,
        help="Directory containing LEAD scenario/route zip files.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root containing extracted LEAD rgb/ images used by the training loader.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for lead_infos_train.pkl and lead_infos_val.pkl.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument(
        "--split-strategy",
        choices=["hash", "sorted"],
        default="hash",
        help="Route-level split. Hash avoids the old lexicographic Town/scenario bias.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help=(
            "Write a compact manifest plus pickle shards of this many samples. "
            "Use this for full 4Hz data so every distributed rank does not load "
            "one million annotations into host memory. Zero keeps legacy output."
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Keep every fourth native 4Hz LEAD frame, giving about 1Hz training samples.",
    )
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument(
        "--route-quality-filter",
        choices=["tfv6_posttrain", "none"],
        default="tfv6_posttrain",
        help="Filter routes with the same completion/quality rules as TFV6 posttraining.",
    )
    parser.add_argument("--route-skip-first", type=int, default=8)
    parser.add_argument("--route-skip-last", type=int, default=8)
    parser.add_argument(
        "--ego-plan-horizon",
        type=int,
        default=8,
        help="Number of ego planning waypoints. Default 8 matches TFV6/LEAD.",
    )
    parser.add_argument(
        "--ego-plan-step",
        type=int,
        default=5,
        help="Source 20Hz frame spacing between ego planning waypoints. Default 5 gives 4Hz/2s for 8 steps.",
    )
    parser.add_argument(
        "--agent-motion-horizon",
        type=int,
        default=6,
        help="Number of agent motion slots. Use 8 for native LEAD 4Hz/2s motion.",
    )
    parser.add_argument(
        "--agent-motion-step",
        type=int,
        default=10,
        help="Source 20Hz frame spacing for agent motion. Use 5 for native LEAD 4Hz.",
    )
    parser.add_argument(
        "--sensor-perturbation-probability",
        type=float,
        default=0.5,
        help="Probability of selecting the synchronized LEAD perturbated sensor frame during training.",
    )
    parser.add_argument(
        "--no-extract-perturbed-images",
        action="store_true",
        help="Require already-extracted rgb_perturbated files instead of extracting selected frames from zips.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Number of zip-processing worker processes.",
    )
    parser.add_argument("--no-require-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    lead_zip_root = Path(args.lead_zip_root)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_stride = max(1, int(args.frame_stride))
    frame_offset = int(args.frame_offset) % frame_stride
    ego_plan_horizon = max(1, int(args.ego_plan_horizon))
    ego_plan_step = max(1, int(args.ego_plan_step))
    agent_motion_horizon = max(1, int(args.agent_motion_horizon))
    agent_motion_step = max(1, int(args.agent_motion_step))
    route_quality_filter = str(args.route_quality_filter)
    route_skip_first = max(0, int(args.route_skip_first))
    route_skip_last = max(0, int(args.route_skip_last))
    sensor_perturbation_probability = float(args.sensor_perturbation_probability)
    if not 0.0 <= sensor_perturbation_probability <= 1.0:
        raise ValueError(
            "--sensor-perturbation-probability must be in [0, 1], "
            f"got {sensor_perturbation_probability}"
        )
    val_ratio = float(args.val_ratio)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1), got {val_ratio}")
    shard_size = max(0, int(args.shard_size))
    if shard_size and val_ratio != 0.0:
        raise ValueError(
            "Sharded LEAD infos currently require --val-ratio 0. "
            "Training-only full-4Hz generation must not mix route splitting "
            "with streaming shard emission."
        )

    zip_paths = sorted(lead_zip_root.rglob("*.zip"))
    infos = []
    worker_args = [
        (
            str(zip_path),
            str(image_root),
            not args.no_require_images,
            frame_stride,
            frame_offset,
            ego_plan_horizon,
            ego_plan_step,
            agent_motion_horizon,
            agent_motion_step,
            route_quality_filter,
            route_skip_first,
            route_skip_last,
            sensor_perturbation_probability,
            not args.no_extract_perturbed_images,
        )
        for zip_path in zip_paths
    ]
    if shard_size:
        shard_buffer = []
        shards = []
        folder_to_id = {}
        folder_names = []
        folder_ids = []
        frame_idxs = []
        first_info = None
        all_perturbation_fields = True
        num_samples = 0
        perturbation_required = {
            "sensor_perturbation_version",
            "perturbation_translation",
            "perturbation_rotation",
            "perturbed_image_path",
            "lidar2global_perturbed",
            "gt_boxes_perturbed",
            "gt_agent_fut_trajs_perturbed",
            "ego_status_perturbed",
            "ego_status_mask_perturbed",
            "target_point_previous_perturbed",
            "target_point_perturbed",
            "target_point_next_perturbed",
            "gt_ego_fut_trajs_4hz_perturbed",
            "gt_ego_fut_masks_4hz_perturbed",
            "gt_ego_fut_trajs_5hz_perturbed",
            "gt_ego_fut_masks_5hz_perturbed",
            "gt_route_perturbed",
            "gt_route_masks_perturbed",
            "gt_ego_fut_trajs_2hz_perturbed",
            "gt_ego_fut_masks_2hz_perturbed",
        }

        def flush_shard() -> None:
            nonlocal shard_buffer
            if not shard_buffer:
                return
            shard_index = len(shards)
            shard_file = f"lead_infos_train-{shard_index:05d}.pkl"
            _atomic_pickle_dump(shard_buffer, out_dir / shard_file)
            shards.append({"file": shard_file, "length": len(shard_buffer)})
            shard_buffer = []

        def consume_chunk(chunk: list[dict[str, Any]]) -> bool:
            nonlocal first_info, all_perturbation_fields, num_samples
            if args.max_samples is not None:
                remaining = int(args.max_samples) - num_samples
                if remaining <= 0:
                    return False
                chunk = chunk[:remaining]
            for info in chunk:
                if first_info is None:
                    first_info = info
                folder = str(info["folder"])
                folder_id = folder_to_id.get(folder)
                if folder_id is None:
                    folder_id = len(folder_names)
                    folder_to_id[folder] = folder_id
                    folder_names.append(folder)
                folder_ids.append(folder_id)
                frame_idxs.append(int(info["frame_idx"]))
                if sensor_perturbation_probability > 0.0:
                    all_perturbation_fields = (
                        all_perturbation_fields
                        and perturbation_required.issubset(info)
                    )
                shard_buffer.append(info)
                num_samples += 1
                if len(shard_buffer) >= shard_size:
                    flush_shard()
            return args.max_samples is None or num_samples < int(args.max_samples)

        if args.num_workers <= 1:
            iterator = map(_process_zip, worker_args)
            for chunk in tqdm(iterator, total=len(worker_args), desc="LEAD zips"):
                if not consume_chunk(chunk):
                    break
        else:
            with Pool(processes=args.num_workers) as pool:
                # Ordered emission preserves route/frame order for temporal sampling.
                iterator = pool.imap(_process_zip, worker_args, chunksize=8)
                for chunk in tqdm(iterator, total=len(worker_args), desc="LEAD zips"):
                    if not consume_chunk(chunk):
                        pool.terminate()
                        break
        flush_shard()
        if not num_samples or first_info is None:
            raise RuntimeError("No LEAD Stage1 samples were generated. Check zip/image paths.")

        manifest = {
            "lead_sharded_infos_version": 1,
            "num_samples": num_samples,
            "shards": shards,
            "folder_names": folder_names,
            "folder_ids": np.asarray(folder_ids, dtype=np.int32),
            "frame_idxs": np.asarray(frame_idxs, dtype=np.int32),
            "first_info": first_info,
            "all_sensor_perturbation_fields": bool(all_perturbation_fields),
        }
        _atomic_pickle_dump(manifest, out_dir / "lead_infos_train.pkl")
        _atomic_pickle_dump([], out_dir / "lead_infos_val.pkl")

        train_infos = _ShardedInfosIterable(out_dir, shards, num_samples)
        trajectory_stats = _compute_trajectory_stats(
            train_infos,
            frame_stride=frame_stride,
            frame_offset=frame_offset,
            route_quality_filter=route_quality_filter,
            route_skip_first=route_skip_first,
            route_skip_last=route_skip_last,
            ego_plan_horizon=ego_plan_horizon,
            ego_plan_step=ego_plan_step,
            agent_motion_horizon=agent_motion_horizon,
            agent_motion_step=agent_motion_step,
            sensor_perturbation_probability=sensor_perturbation_probability,
        )
        trajectory_stats.update(
            val_ratio=0.0,
            split_strategy=str(args.split_strategy),
            num_val_samples=0,
            num_total_samples=num_samples,
            infos_storage="sharded_pickle_v1",
            infos_shard_size=shard_size,
            infos_num_shards=len(shards),
        )
        _atomic_json_dump(trajectory_stats, out_dir / "lead_trajectory_stats.json")
        print(
            f"wrote train={num_samples} val=0 shards={len(shards)} "
            f"stats=lead_trajectory_stats.json to {out_dir}"
        )
        return

    if args.num_workers <= 1:
        iterator = map(_process_zip, worker_args)
        for chunk in tqdm(iterator, total=len(worker_args), desc="LEAD zips"):
            infos.extend(chunk)
            if args.max_samples is not None and len(infos) >= args.max_samples:
                infos = infos[: args.max_samples]
                break
    else:
        with Pool(processes=args.num_workers) as pool:
            iterator = pool.imap_unordered(_process_zip, worker_args, chunksize=8)
            for chunk in tqdm(iterator, total=len(worker_args), desc="LEAD zips"):
                infos.extend(chunk)
                if args.max_samples is not None and len(infos) >= args.max_samples:
                    infos = infos[: args.max_samples]
                    pool.terminate()
                    break

    infos.sort(key=lambda x: (x["folder"], x["frame_idx"]))

    if not infos:
        raise RuntimeError("No LEAD Stage1 samples were generated. Check zip/image paths.")

    route_ids = sorted({info["folder"] for info in infos})
    if val_ratio == 0.0:
        val_routes = set()
    else:
        val_count = max(1, int(len(route_ids) * val_ratio)) if len(route_ids) > 1 else 1
        if args.split_strategy == "hash":
            route_order = sorted(
                route_ids,
                key=lambda route: hashlib.sha1(route.encode("utf-8")).hexdigest(),
            )
            val_routes = set(route_order[:val_count])
        else:
            val_routes = set(route_ids[-val_count:])
    train_infos = [info for info in infos if info["folder"] not in val_routes]
    val_infos = [info for info in infos if info["folder"] in val_routes]
    if val_ratio > 0.0 and not train_infos:
        split = max(1, int(len(infos) * (1.0 - val_ratio)))
        if split >= len(infos):
            split = max(1, len(infos) - 1)
        train_infos = infos[:split]
        val_infos = infos[split:] or infos[-1:]

    _atomic_pickle_dump(train_infos, out_dir / "lead_infos_train.pkl")
    _atomic_pickle_dump(val_infos, out_dir / "lead_infos_val.pkl")

    trajectory_stats = _compute_trajectory_stats(
        train_infos,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        route_quality_filter=route_quality_filter,
        route_skip_first=route_skip_first,
        route_skip_last=route_skip_last,
        ego_plan_horizon=ego_plan_horizon,
        ego_plan_step=ego_plan_step,
        agent_motion_horizon=agent_motion_horizon,
        agent_motion_step=agent_motion_step,
        sensor_perturbation_probability=sensor_perturbation_probability,
    )
    trajectory_stats.update(
        val_ratio=val_ratio,
        split_strategy=str(args.split_strategy),
        num_val_samples=len(val_infos),
        num_total_samples=len(infos),
    )
    _atomic_json_dump(trajectory_stats, out_dir / "lead_trajectory_stats.json")

    print(
        f"wrote train={len(train_infos)} val={len(val_infos)} "
        f"stats=lead_trajectory_stats.json to {out_dir}"
    )


if __name__ == "__main__":
    main()
