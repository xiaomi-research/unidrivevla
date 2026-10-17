"""PhysicalAI-AV planning samples and a cache-only training loader.

The licensed SDK is deliberately absent from module imports: run the offline converter in its supported Python
environment, then train on the plain NPZ/PNG cache with the normal UniDriveVLA environment.
"""

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from unidrivevla.data.external import ExternalCacheDataset, write_jpeg, write_png
from unidrivevla.data.sample import NAN3, camera

# Verified PhysicalAI-AV SDK feature identifiers and camera-to-profile mapping.
CAMERA_FEATURES = (
    ("camera_front_wide_120fov", "CAMERA_FRONT_WIDE_120FOV", "front", 90.0),
    ("camera_front_tele_30fov", "CAMERA_FRONT_TELE_30FOV", "front_tele", 18.0),
    ("camera_cross_left_120fov", "CAMERA_CROSS_LEFT_120FOV", "front_left", 90.0),
    ("camera_cross_right_120fov", "CAMERA_CROSS_RIGHT_120FOV", "front_right", 90.0),
    ("camera_rear_left_70fov", "CAMERA_REAR_LEFT_70FOV", "back_left", 55.0),
    ("camera_rear_right_70fov", "CAMERA_REAR_RIGHT_70FOV", "back_right", 55.0),
    ("camera_rear_tele_30fov", "CAMERA_REAR_TELE_30FOV", "back", 18.0),
)
HORIZON = 6
INTERVAL_US = 500_000


def pinhole_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Explicit square-pixel pinhole K from a horizontal field of view."""
    if width <= 0 or height <= 0 or not 0.0 < fov_deg < 180.0:
        raise ValueError("rectified size must be positive and horizontal FOV must be in (0, 180) degrees")
    focal = (width - 1) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([[focal, 0.0, (width - 1) / 2.0],
                     [0.0, focal, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def ftheta_remap(camera_model, size, fov_deg: float, min_valid_fraction: float = 0.98):
    """Map the chosen pinhole image through the SDK's calibrated FTheta ray2pixel model."""
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise ValueError("rectified size must be positive")
    model_width, model_height = int(camera_model.width), int(camera_model.height)
    principal_point = np.asarray(camera_model.principal_point, dtype=np.float64)
    if model_width <= 0 or model_height <= 0 or principal_point.shape != (2,) or not np.isfinite(principal_point).all():
        raise ValueError("PhysicalAI FTheta calibration has invalid resolution or principal point")
    # These are the numeric calibration fields exposed by the official ncore FTheta model. Validate
    # parameters before ray projection so NaNs cannot be hidden by the pixel-validity mask.
    for name in ("angle_to_pixeldist_poly", "pixeldist_to_angle_poly", "linear_cde", "max_angle"):
        value = getattr(camera_model, name, None)
        if value is not None and not np.isfinite(np.asarray(value, dtype=np.float64)).all():
            raise ValueError(f"PhysicalAI FTheta calibration {name} contains non-finite values")
    intrinsic = pinhole_intrinsic(width, height, fov_deg)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = np.stack(((u - intrinsic[0, 2]) / intrinsic[0, 0],
                     (v - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(u)), axis=-1)
    pixels = np.asarray(camera_model.ray2pixel(rays.reshape(-1, 3)), dtype=np.float64).reshape(height, width, 2)
    # The official FTheta formula divides by the ray's xy norm, which is singular at optical center.
    center_ray = np.linalg.norm(rays[..., :2], axis=-1) <= 1e-12
    pixels[center_ray] = principal_point
    valid = (np.isfinite(pixels).all(axis=-1) & (pixels[..., 0] >= 0) & (pixels[..., 0] <= camera_model.width - 1)
             & (pixels[..., 1] >= 0) & (pixels[..., 1] <= camera_model.height - 1))
    map_x = np.where(valid, pixels[..., 0], -1).astype(np.float32)
    map_y = np.where(valid, pixels[..., 1], -1).astype(np.float32)
    if not valid.any():
        raise ValueError("the requested pinhole FOV has no pixels inside the calibrated FTheta image")
    if float(valid.mean()) < min_valid_fraction:
        raise ValueError(f"per-camera FOV {fov_deg:g} covers only {valid.mean():.1%} of the rectified image; "
                         f"at least {min_valid_fraction:.0%} is required")
    return intrinsic, map_x, map_y, valid


def rectify_ftheta(image: np.ndarray, camera_model, fov_deg: float, size, min_valid_fraction: float = 0.98):
    """Return RGB pinhole pixels, explicit K, and a pixel-validity mask."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"PhysicalAI camera frame must be uint8 RGB HxWx3, got {image.shape} {image.dtype}")
    width, height = map(int, size)
    if image.shape[:2] != (camera_model.height, camera_model.width):
        raise ValueError(f"decoded camera size {image.shape[1]}x{image.shape[0]} does not match calibrated size "
                         f"{camera_model.width}x{camera_model.height}")
    intrinsic, map_x, map_y, valid = ftheta_remap(camera_model, size, fov_deg, min_valid_fraction)
    return _apply_rectification(image, intrinsic, map_x, map_y, valid)


def _apply_rectification(image, intrinsic, map_x, map_y, valid):
    rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rectified[~valid] = 0
    return rectified, intrinsic, valid


def _pose_rows(egomotion, timestamps_us):
    state = egomotion(np.asarray(timestamps_us, dtype=np.int64))
    translation = np.asarray(state.pose.translation, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(state.pose.rotation.as_matrix(), dtype=np.float64).reshape(-1, 3, 3)
    if len(translation) != len(timestamps_us) or len(rotation) != len(timestamps_us):
        raise ValueError("PhysicalAI egomotion returned the wrong number of poses")
    if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        raise ValueError("PhysicalAI egomotion contains non-finite pose values")
    return translation, rotation


def physicalai_sample(clip_id, timestamp_us, egomotion, camera_rows):
    """Pure mapping for verified SDK-shaped inputs; targets only the six future ego poses."""
    present_views = {row[0] for row in camera_rows}
    required_views = {row[2] for row in CAMERA_FEATURES}
    if present_views != required_views or len(camera_rows) != len(required_views):
        raise ValueError(f"PhysicalAI requires its full seven-camera rig; missing={sorted(required_views - present_views)}, "
                         f"unexpected={sorted(present_views - required_views)}")
    history_times = np.asarray([timestamp_us - 3 * INTERVAL_US, timestamp_us - 2 * INTERVAL_US,
                                timestamp_us - INTERVAL_US, timestamp_us], dtype=np.int64)
    future_times = timestamp_us + np.arange(1, HORIZON + 1, dtype=np.int64) * INTERVAL_US
    times = np.concatenate((history_times, future_times))
    positions, rotations = _pose_rows(egomotion, times)
    origin = positions[3]
    origin_rotation = rotations[3]
    local_positions = (positions - origin) @ origin_rotation
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    yaw_delta = (yaw - yaw[3] + np.pi) % (2 * np.pi) - np.pi
    history = np.column_stack((local_positions[:4, :2], yaw_delta[:4])).astype(np.float32)
    trajectory = np.column_stack((local_positions[4:, :2], yaw_delta[4:])).astype(np.float32)

    cameras, assets = [], {}
    digest = hashlib.sha256(f"physicalai\0{clip_id}\0{timestamp_us}".encode("utf-8")).hexdigest()
    for view, image, intrinsic, sensor2ego, valid, _camera_timestamp_us in camera_rows:
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        sensor2ego = np.asarray(sensor2ego, dtype=np.float64)
        if intrinsic.shape != (3, 3) or sensor2ego.shape != (4, 4):
            raise ValueError(f"PhysicalAI {view} calibration has invalid matrix shape")
        if not np.isfinite(intrinsic).all() or not np.isfinite(sensor2ego).all():
            raise ValueError(f"PhysicalAI {view} calibration contains non-finite values")
        image_rel = f"assets/images/{digest}/{view}.jpg"
        mask_rel = f"assets/images/{digest}/{view}_valid.png"
        h, w = image.shape[:2]
        cameras.append(camera(view, image_rel, intrinsic, sensor2ego, (w, h)))
        cameras[-1]["rectified_fov_deg"] = float(2.0 * math.degrees(math.atan((w - 1) / (2.0 * intrinsic[0, 0]))))
        cameras[-1]["valid_mask_path"] = mask_rel
        assets[image_rel[7:]] = write_jpeg(image, quality=95)
        assets[mask_rel[7:]] = write_png(valid.astype(np.uint8) * 255)

    # Ego velocity and acceleration use only past positions; missing steering/status fields stay NaN.
    velocity = ((history[3, :2] - history[2, :2]) / (INTERVAL_US / 1e6)).astype(np.float32)
    acceleration = np.full(3, np.nan, dtype=np.float32)
    acceleration[:2] = ((history[3, :2] - 2 * history[2, :2] + history[1, :2]) /
                        ((INTERVAL_US / 1e6) ** 2)).astype(np.float32)
    angular_velocity = NAN3.copy()
    angular_velocity[2] = np.float32((history[3, 2] - history[2, 2]) / (INTERVAL_US / 1e6))
    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, :3], ego2global[:3, 3] = origin_rotation, origin
    features = {
        "cameras": cameras,
        "history_trajectory": history,
        "ego_velocity": np.array([velocity[0], velocity[1], np.nan], dtype=np.float32),
        "ego_acceleration": acceleration,
        "ego_angular_velocity": angular_velocity,
        "ego_steer": np.float32(np.nan),
        "command": "unknown",
        "timestamp": float(timestamp_us) / 1e6,
        "ego2global": ego2global,
        "stream_id": f"physicalai:{clip_id}",
        "camera_timestamps_us": {row[0]: np.int64(row[5]) for row in camera_rows},
    }
    targets = {
        "trajectory": trajectory,
        "trajectory_mask": np.ones(HORIZON, dtype=np.float32),
        "task_valid": {"planning": True, "det": False, "map": False, "motion": False,
                       "occupancy": False, "geometry": False},
    }
    return features, targets, assets


def convert_dataset_interface(dataset, clip_ids, cache_root, split, image_size=(960, 544),
                              sample_interval_us=INTERVAL_US, sync_tolerance_us=50_000, fov_overrides=None):
    """Convert already cached, license-authorized official SDK clips; never downloads data."""
    image_width, image_height = map(int, image_size)
    if sample_interval_us != INTERVAL_US:
        raise ValueError("canonical UniDriveVLA trajectories require 0.5s sample spacing")
    features_api = dataset.features
    camera_api = features_api.CAMERA
    fov_overrides = dict(fov_overrides or {})
    unknown_cameras = set(fov_overrides) - {row[0] for row in CAMERA_FEATURES}
    if unknown_cameras:
        raise ValueError(f"FOV override names are not configured PhysicalAI cameras: {sorted(unknown_cameras)}")
    camera_handles = []
    for name, enum_name, view, default_fov in CAMERA_FEATURES:
        fov = float(fov_overrides.get(name, default_fov))
        if not 0.0 < fov < 180.0:
            raise ValueError(f"FOV for {name} must be in (0, 180) degrees")
        camera_handles.append((name, getattr(camera_api, enum_name), view, fov))
    written = skipped = 0
    with ExternalCacheWriter(cache_root, "physicalai", split) as writer:
        for clip_id in clip_ids:
            # The official interface returns calibrated model objects, sensor poses, SeekVideoReaders and the
            # egomotion interpolator for these feature names.
            intrinsics = dataset.get_clip_feature(clip_id, "camera_intrinsics")
            extrinsics = dataset.get_clip_feature(clip_id, "sensor_extrinsics")
            egomotion = dataset.get_clip_feature(clip_id, features_api.LABELS.EGOMOTION)
            video_rows = []
            for name, feature, view, fov in camera_handles:
                model = intrinsics.camera_models.get(name)
                sensor_pose = extrinsics.sensor_poses.get(name)
                if model is None or sensor_pose is None:
                    raise ValueError(f"clip {clip_id} lacks required calibrated camera {name}")
                sensor2ego = np.asarray(sensor_pose.as_matrix(), dtype=np.float64)
                if sensor2ego.shape != (4, 4) or not np.isfinite(sensor2ego).all():
                    raise ValueError(f"clip {clip_id} camera {name} extrinsic contains non-finite or invalid values")
                reader = dataset.get_clip_feature(clip_id, feature)
                remap = ftheta_remap(model, (image_width, image_height), fov)
                video_rows.append((name, view, fov, model, sensor2ego, reader,
                                   np.asarray(reader.timestamps, dtype=np.int64), remap))
            master = next(row for row in video_rows if row[1] == "front")
            next_sample = int(master[6][0])
            for timestamp_us in master[6]:
                timestamp_us = int(timestamp_us)
                if timestamp_us < next_sample:
                    continue
                next_sample = timestamp_us + sample_interval_us
                decoded = []
                try:
                    for name, view, fov, model, sensor2ego, reader, _, remap in video_rows:
                        frames, actual_times = reader.decode_images_from_timestamps(np.asarray([timestamp_us],
                                                                                                  dtype=np.int64))
                        actual_times = np.asarray(actual_times, dtype=np.int64).reshape(-1)
                        if len(frames) != 1 or len(actual_times) != 1 or abs(int(actual_times[0]) - timestamp_us) > sync_tolerance_us:
                            raise ValueError(f"camera {name} has no frame within {sync_tolerance_us} us")
                        image, intrinsic, valid = _apply_rectification(frames[0], *remap)
                        decoded.append((view, image, intrinsic, sensor2ego, valid, int(actual_times[0])))
                    # feature list carries actual per-camera capture times through the canonical cache.
                    camera_rows = [(view, image, intrinsic, sensor2ego, valid, camera_time)
                                   for view, image, intrinsic, sensor2ego, valid, camera_time in decoded]
                    features, targets, assets = physicalai_sample(clip_id, timestamp_us, egomotion, camera_rows)
                    writer.write_sample(f"{clip_id}:{timestamp_us}", features, targets, assets)
                    written += 1
                except (IndexError, ValueError, RuntimeError) as exc:
                    skipped += 1
                    print(f"skip {clip_id} at {timestamp_us}: {exc}")
    return {"written": written, "skipped": skipped}


def physicalai_loader(cfg, agent, split):
    from omegaconf import OmegaConf

    source_split = OmegaConf.select(cfg, f"physicalai.{split}_split") or split
    dataset = ExternalCacheDataset(cfg.physicalai.cache_root, "physicalai", source_split)
    profile = getattr(agent, "profile", None)
    profiles = getattr(agent, "profiles", None)
    if profile is None and isinstance(profiles, dict):
        profile = profiles.get("physicalai")
    required_views = set(getattr(profile, "cameras", ())) or {view for _, _, view, _ in CAMERA_FEATURES}
    for row in dataset.rows:
        present = {camera.get("view") for camera in row["features"].get("cameras", [])}
        missing = required_views - present
        if missing:
            raise ValueError(f"PhysicalAI cache token {row.get('token')!r} is missing camera views {sorted(missing)}; "
                             "reconvert it with the current seven-camera profile")
    return dataset
