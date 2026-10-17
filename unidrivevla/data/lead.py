"""LEAD-main planning samples from the official 123D/SceneLoader API.

The LEAD SDK is used only by the offline converter. This module is Python-3.9-compatible and reads canonical caches.
"""

import hashlib
import math

import numpy as np

from unidrivevla.data.external import ExternalCacheDataset, write_jpeg
from unidrivevla.data.sample import NAN3, camera

LEAD_CAMERA_VIEWS = ("front_left", "front", "front_right", "back_right", "back", "back_left")
TICK_US = 50_000
INTERVAL_TICKS = 10
HORIZON = 6


def _pinhole_intrinsic(camera_metadata, image_shape):
    """Recover K via the documented py123d project_to_image API; reject non-pinhole/distorted models."""
    h, w = map(int, image_shape[:2])
    points = np.array([[0.0, 0.0, 1.0], [0.25, 0.0, 1.0], [-0.25, 0.0, 1.0],
                       [0.0, 0.25, 1.0], [0.0, -0.25, 1.0]], dtype=np.float64)
    projected = camera_metadata.project_to_image(points)
    if not isinstance(projected, tuple) or len(projected) < 2:
        raise TypeError("unsupported py123d camera projection result")
    pixels = np.asarray(projected[0], dtype=np.float64).reshape(-1, 2)
    visible = np.asarray(projected[1], dtype=bool).reshape(-1)
    if len(pixels) != len(points) or not visible.all() or not np.isfinite(pixels).all():
        raise ValueError("LEAD camera does not provide the expected calibrated pinhole rays")
    center = pixels[0]
    fx = (pixels[1, 0] - pixels[2, 0]) / 0.5
    fy = (pixels[3, 1] - pixels[4, 1]) / 0.5
    expected = center + np.array([[0.25 * fx, 0], [-0.25 * fx, 0], [0, 0.25 * fy], [0, -0.25 * fy]])
    if fx <= 0 or fy <= 0 or not np.allclose(pixels[1:], expected, rtol=1e-3, atol=1e-2):
        raise ValueError("only a calibrated pinhole LEAD camera can be converted without an explicit rectifier")
    return np.array([[fx, 0, center[0]], [0, fy, center[1]], [0, 0, 1]], dtype=np.float64)


def _pose_parts(pose):
    """py123d PoseSE3 public geometry: its array is XYZ followed by quaternion; rotation_matrix is world-from-body."""
    array = np.asarray(pose.array, dtype=np.float64).reshape(-1)
    rotation = np.asarray(pose.rotation_matrix, dtype=np.float64)
    if array.size < 3 or rotation.shape != (3, 3) or not np.isfinite(array).all() or not np.isfinite(rotation).all():
        raise ValueError("invalid py123d PoseSE3")
    return array[:3], rotation


def _wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def exact_label_origin(scene_data):
    """Read the official current body-IMU pose; never substitute localized input or future truth."""
    try:
        pose = scene_data.ego_state.imu_se3
    except AttributeError as exc:
        raise RuntimeError("unsupported LEAD SceneData API: current exact scene_data.ego_state.imu_se3 is required") from exc
    if pose is None:
        raise RuntimeError("unsupported LEAD SceneData API: current exact ego_state.imu_se3 is missing")
    return pose


def lead_sample(scene_data, future_states, log_name, label_origin_pose, phase=0, tick_us=TICK_US,
                interval_ticks=INTERVAL_TICKS):
    """Map one official LEAD SceneData anchor to canonical planning-only supervision."""
    if len(scene_data.cameras) != len(LEAD_CAMERA_VIEWS):
        raise ValueError(f"LEAD main expects six recorded RGB cameras, got {len(scene_data.cameras)}")
    past_positions = np.asarray(scene_data.past_ego_positions, dtype=np.float64).reshape(-1, 2)
    past_yaws = np.asarray(scene_data.past_ego_yaws, dtype=np.float64).reshape(-1)
    history_ages = np.arange(3, -1, -1, dtype=np.int64) * interval_ticks
    if len(past_positions) <= history_ages[0] or len(past_yaws) <= history_ages[0]:
        raise ValueError("LEAD anchor lacks 1.5 seconds of localized driving_meta history")
    if not np.isfinite(past_positions[history_ages]).all() or not np.isfinite(past_yaws[history_ages]).all():
        raise ValueError("LEAD localized input history contains non-finite poses")
    history = np.column_stack((past_positions[history_ages], past_yaws[history_ages])).astype(np.float32)

    driving_meta = scene_data.driving_meta
    if not driving_meta or "localized_ego_state_se3" not in driving_meta:
        raise ValueError("LEAD anchor lacks localized_ego_state_se3")
    origin_pose = np.asarray(driving_meta["localized_ego_state_se3"], dtype=np.float64)
    if origin_pose.shape != (4, 4) or not np.isfinite(origin_pose).all():
        raise ValueError("LEAD localized_ego_state_se3 must be a finite 4x4 matrix")
    label_xyz, label_rotation = _pose_parts(label_origin_pose)
    origin_xy = label_xyz[:2]
    origin_yaw = math.atan2(label_rotation[1, 0], label_rotation[0, 0])
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    to_local = np.array([[c, s], [-s, c]], dtype=np.float64)

    trajectory = np.zeros((HORIZON, 3), dtype=np.float32)
    trajectory_mask = np.zeros(HORIZON, dtype=np.float32)
    for step in range(1, HORIZON + 1):
        future = future_states.get(step * interval_ticks)
        if future is None:
            continue
        # Official py123d EgoStateSE3 stores its exact global pose as imu_se3.
        xyz, rotation = _pose_parts(future.imu_se3)
        local_xy = to_local @ (xyz[:2] - origin_xy)
        future_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        local_yaw = _wrap_angle(future_yaw - origin_yaw)
        if not np.isfinite(local_xy).all() or not math.isfinite(local_yaw):
            raise ValueError(f"LEAD supervised future pose at {step * interval_ticks} ticks is non-finite")
        trajectory[step - 1, :2] = local_xy
        trajectory[step - 1, 2] = local_yaw
        if not np.isfinite(trajectory[step - 1]).all():
            raise ValueError(f"LEAD supervised future pose at {step * interval_ticks} ticks overflows canonical dtype")
        trajectory_mask[step - 1] = 1.0
    if not trajectory_mask.any():
        raise ValueError("LEAD anchor has no exact future ego poses")

    cameras, assets = [], {}
    camera_timestamps = {}
    digest = hashlib.sha256(f"lead\0{log_name}\0{scene_data.cameras[0].timestamp.time_us}".encode("utf-8")).hexdigest()
    for view, item in zip(LEAD_CAMERA_VIEWS, scene_data.cameras):
        image = np.asarray(item.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"LEAD {view} image must be uint8 RGB HxWx3, got {image.shape} {image.dtype}")
        metadata = item.metadata
        intrinsic = _pinhole_intrinsic(metadata, image.shape)
        sensor_pose = metadata.camera_to_imu_se3
        sensor_xyz, sensor_rotation = _pose_parts(sensor_pose)
        sensor2ego = np.eye(4, dtype=np.float64)
        sensor2ego[:3, :3] = sensor_rotation
        sensor2ego[:3, 3] = sensor_xyz
        rel = f"assets/images/{digest}/{view}.jpg"
        h, w = image.shape[:2]
        cameras.append(camera(view, rel, intrinsic, sensor2ego, (w, h)))
        assets[rel[7:]] = write_jpeg(image, quality=95)
        camera_timestamps[view] = int(item.timestamp.time_us)

    if max(camera_timestamps.values()) - min(camera_timestamps.values()) > tick_us:
        raise ValueError("LEAD camera timestamps differ by more than one simulator tick")
    velocity = (past_positions[0] - past_positions[1]) / (tick_us / 1e6)
    acceleration = (past_positions[0] - 2 * past_positions[1] + past_positions[2]) / ((tick_us / 1e6) ** 2)
    angular_velocity = _wrap_angle(past_yaws[0] - past_yaws[1]) / (tick_us / 1e6)
    current_us = int(scene_data.cameras[0].timestamp.time_us)
    features = {
        "cameras": cameras,
        "history_trajectory": history,
        "ego_velocity": np.array([velocity[0], velocity[1], np.nan], dtype=np.float32),
        "ego_acceleration": np.array([acceleration[0], acceleration[1], np.nan], dtype=np.float32),
        "ego_angular_velocity": np.array([np.nan, np.nan, angular_velocity], dtype=np.float32),
        "ego_steer": np.float32(np.nan),
        "command": "unknown",
        "timestamp": current_us / 1e6,
        "ego2global": origin_pose,
        "stream_id": f"lead:{log_name}:phase={int(phase) % 2}",
        "camera_timestamps_us": camera_timestamps,
    }
    targets = {
        "trajectory": trajectory,
        "trajectory_mask": trajectory_mask,
        "task_valid": {"planning": True, "det": False, "map": False, "motion": False,
                       "occupancy": False, "geometry": False},
    }
    return features, targets, assets


def lead_loader(cfg, agent, split):
    from omegaconf import OmegaConf

    source_split = OmegaConf.select(cfg, f"lead.{split}_split") or split
    return ExternalCacheDataset(cfg.lead.cache_root, "lead", source_split)
