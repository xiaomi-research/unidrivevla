"""LEAD-main planning samples from the official 123D/SceneLoader API.

The LEAD SDK is used only by the offline converter. This module is Python-3.9-compatible and reads canonical caches.
"""

import hashlib
import math

import numpy as np

from unidrivevla.data.external import ExternalCacheDataset, write_jpeg
from unidrivevla.data.sample import NAN3, camera

LEAD_CAMERA_VIEWS = ("front_left", "front", "front_right", "back_right", "back", "back_left")
# Ordered like LEAD's CAMERA_ID_BY_LEAD_INDEX (1 through 6), as returned by SceneLoader.
TICK_US = 50_000
HISTORY_INTERVAL_TICKS = 10
TARGET_INTERVAL_TICKS = 5
TARGET_INTERVAL_US = TICK_US * TARGET_INTERVAL_TICKS
HORIZON = 8


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


def _rigid_pose(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3)
            or np.linalg.det(matrix[:3, :3]) < 0.99):
        raise ValueError(f"LEAD {name} must be a finite rigid 4x4 pose")
    return matrix


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


def lead_inference_features(scene_data, *, ego2global, timestamp_seconds):
    """Build online canonical features from LEAD's localized input and current RGB cameras only.

    `SceneLoader.past_ego_positions/yaws` are already relative to the observed localized anchor. If the
    caller supplies a filtered current localization, re-express that same past in its frame. This helper
    deliberately never reads `ego_state`, future states, or the exact label origin.
    """
    if len(scene_data.cameras) != len(LEAD_CAMERA_VIEWS):
        raise ValueError(f"LEAD main expects six recorded RGB cameras, got {len(scene_data.cameras)}")
    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp):
        raise ValueError("LEAD online timestamp must be finite seconds")
    current_pose = _rigid_pose(ego2global, "online ego2global")
    driving_meta = scene_data.driving_meta
    if not driving_meta or "localized_ego_state_se3" not in driving_meta:
        raise ValueError("LEAD online frame lacks localized_ego_state_se3")
    source_pose = _rigid_pose(driving_meta["localized_ego_state_se3"], "localized_ego_state_se3")
    source_yaw = math.atan2(source_pose[1, 0], source_pose[0, 0])
    current_yaw = math.atan2(current_pose[1, 0], current_pose[0, 0])
    to_current = current_pose[:2, :2].T @ source_pose[:2, :2]
    translation = current_pose[:2, :2].T @ (source_pose[:2, 3] - current_pose[:2, 3])

    past_positions = np.asarray(scene_data.past_ego_positions, dtype=np.float64).reshape(-1, 2)
    past_yaws = np.asarray(scene_data.past_ego_yaws, dtype=np.float64).reshape(-1)
    ages = np.arange(3, -1, -1, dtype=np.int64) * HISTORY_INTERVAL_TICKS
    if len(past_positions) <= ages[0] or len(past_yaws) <= ages[0]:
        raise ValueError("LEAD online frame lacks 1.5 seconds of localized history")
    if not np.isfinite(past_positions[ages]).all() or not np.isfinite(past_yaws[ages]).all():
        raise ValueError("LEAD online localized history contains non-finite poses")
    history_xy = past_positions[ages] @ to_current.T + translation
    history_yaw = _wrap_angle(past_yaws[ages] + source_yaw - current_yaw)
    history = np.column_stack((history_xy, history_yaw)).astype(np.float32)
    if not np.isfinite(history).all():
        raise ValueError("LEAD online history overflows canonical pose dtype")

    # The model status uses only the observed 20 Hz localized history, never future or exact label poses.
    dt = TICK_US / 1e6
    velocity = to_current @ ((past_positions[0] - past_positions[1]) / dt)
    acceleration = to_current @ ((past_positions[0] - 2 * past_positions[1] + past_positions[2]) / (dt ** 2))
    angular_velocity = _wrap_angle(past_yaws[0] - past_yaws[1]) / dt

    cameras, camera_timestamps = {}, {}
    for view, item in zip(LEAD_CAMERA_VIEWS, scene_data.cameras):
        image = np.asarray(item.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"LEAD {view} image must be uint8 RGB HxWx3, got {image.shape} {image.dtype}")
        intrinsic = _pinhole_intrinsic(item.metadata, image.shape)
        sensor_xyz, sensor_rotation = _pose_parts(item.metadata.camera_to_imu_se3)
        sensor2ego = np.eye(4, dtype=np.float64)
        sensor2ego[:3, :3] = sensor_rotation
        sensor2ego[:3, 3] = sensor_xyz
        _rigid_pose(sensor2ego, f"{view} camera_to_imu_se3")
        height, width = image.shape[:2]
        cameras[view] = {"image": image, "intrinsic": intrinsic, "sensor2ego": sensor2ego,
                         "size": (width, height)}
        camera_timestamps[view] = int(item.timestamp.time_us)
    if max(camera_timestamps.values()) - min(camera_timestamps.values()) > TICK_US:
        raise ValueError("LEAD camera timestamps differ by more than one simulator tick")

    log_name = getattr(getattr(scene_data, "log_metadata", None), "log_name", "online")
    return {
        "cameras": cameras,
        "history_trajectory": history,
        "ego_velocity": np.array([velocity[0], velocity[1], np.nan], dtype=np.float32),
        "ego_acceleration": np.array([acceleration[0], acceleration[1], np.nan], dtype=np.float32),
        "ego_angular_velocity": np.array([np.nan, np.nan, angular_velocity], dtype=np.float32),
        "ego_steer": np.float32(np.nan),
        "command": "unknown",
        "timestamp": timestamp,
        "ego2global": current_pose,
        "stream_id": f"lead:{log_name}",
        "camera_timestamps_us": camera_timestamps,
    }


def lead_sample(scene_data, future_states, log_name, label_origin_pose):
    """Map one official LEAD SceneData anchor to canonical planning-only supervision."""
    if not scene_data.cameras:
        raise ValueError("LEAD sample has no recorded RGB cameras")
    driving_meta = scene_data.driving_meta
    if not driving_meta or "localized_ego_state_se3" not in driving_meta:
        raise ValueError("LEAD anchor lacks localized_ego_state_se3")
    origin_pose = np.asarray(driving_meta["localized_ego_state_se3"], dtype=np.float64)
    label_xyz, label_rotation = _pose_parts(label_origin_pose)
    origin_xy = label_xyz[:2]
    origin_yaw = math.atan2(label_rotation[1, 0], label_rotation[0, 0])
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    to_local = np.array([[c, s], [-s, c]], dtype=np.float64)

    trajectory = np.zeros((HORIZON, 3), dtype=np.float32)
    trajectory_mask = np.zeros(HORIZON, dtype=np.float32)
    for step in range(1, HORIZON + 1):
        future_tick = step * TARGET_INTERVAL_TICKS
        future = future_states.get(future_tick)
        if future is None:
            continue
        # Official py123d EgoStateSE3 stores its exact global pose as imu_se3.
        xyz, rotation = _pose_parts(future.imu_se3)
        local_xy = to_local @ (xyz[:2] - origin_xy)
        future_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        local_yaw = _wrap_angle(future_yaw - origin_yaw)
        if not np.isfinite(local_xy).all() or not math.isfinite(local_yaw):
            raise ValueError(f"LEAD supervised future pose at {future_tick} ticks is non-finite")
        trajectory[step - 1, :2] = local_xy
        trajectory[step - 1, 2] = local_yaw
        if not np.isfinite(trajectory[step - 1]).all():
            raise ValueError(f"LEAD supervised future pose at {future_tick} ticks overflows canonical dtype")
        trajectory_mask[step - 1] = 1.0
    if not trajectory_mask.any():
        raise ValueError("LEAD anchor has no exact future ego poses")

    current_us = int(scene_data.cameras[0].timestamp.time_us)
    features = lead_inference_features(scene_data, ego2global=origin_pose,
                                       timestamp_seconds=current_us / 1e6)
    cameras, assets = [], {}
    digest = hashlib.sha256(f"lead\0{log_name}\0{scene_data.cameras[0].timestamp.time_us}".encode("utf-8")).hexdigest()
    for view, item in features["cameras"].items():
        rel = f"assets/images/{digest}/{view}.jpg"
        cameras.append(camera(view, rel, item["intrinsic"], item["sensor2ego"], item["size"]))
        assets[rel[7:]] = write_jpeg(item["image"], quality=95)
    features["cameras"] = cameras
    features["stream_id"] = f"lead:{log_name}"
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
    dataset = ExternalCacheDataset(cfg.lead.cache_root, "lead", source_split)
    profile = getattr(agent, "profile", None)
    profiles = getattr(agent, "profiles", None)
    if profile is None and isinstance(profiles, dict):
        profile = profiles.get("lead")
    if profile is None:
        profile = OmegaConf.select(cfg, "profile")
    interval = float(getattr(profile, "pose_interval", TARGET_INTERVAL_US / 1e6))
    horizon = int(getattr(profile, "horizon", HORIZON))
    dataset.validate_sampling(round(interval * 1e6), horizon)
    return dataset
