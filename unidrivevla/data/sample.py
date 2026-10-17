"""The canonical sample: the only contract between data sources and models.

A loader returns a torch Dataset whose items are (features, targets, token). Geometry uses the explicitly declared
coordinate_frame: ego (x forward, y left, z up) or native LiDAR for the main nuScenes/B2D heads. Missing frame metadata
means the legacy ego contract; native samples must declare their rotation-only native_to_ego transform. Public policy
pred_traj always uses ego axes. Poses are (x, y, heading), time steps are 0.5 s. The origin is the source's reference point
(NAVSIM: the rear axle; nuScenes, Bench2Drive: the LiDAR, where their planning metrics are defined); the profile's
ego_box and ground_height are given in it. Classes, map element types and driving
commands are names: a model maps them to its own vocabulary, so a source never knows the model. Arrays are numpy;
the model's worker_transform turns them into tensors.

features
    coordinate_frame      optional "ego" or "lidar"; native samples must supply it
    native_to_ego         rotation-only (4, 4) native-head-to-ego axes; identity in ego mode, same reference origin
    cameras               [{"view", "path" (or "image": RGB uint8 HWC array), "intrinsic" (3, 3),
                          "sensor2ego" (4, 4), "size" (width, height)}],
                          every camera the source has; views are VIEWS
    history_trajectory    (4, 3) ego poses at t = -1.5, -1.0, -0.5, 0 s
    ego_velocity          (3,) m/s, NaN where the source has none
    ego_acceleration      (3,) m/s^2
    ego_angular_velocity  (3,) rad/s
    ego_steer             ()  steering, NaN if absent
    command               str, one of COMMANDS
    timestamp, ego2global optional: seconds (distinct sequences never within seconds of each other) and the 4x4 pose
                          of this frame in a world frame; together they let temporal models carry state across frames
    augment               set by the training loader, not by sources: {"flip", "rotate"} of the frame's sequence
targets (empty at test time)
    trajectory            (T, 3) future ego poses, every future step the source has
    trajectory_mask       (T,)
    boxes                 (N, 7) x, y, z, length, width, height, yaw
    box_velocity          (N, 2)
    box_names             [str], the source's class names
    box_ids               [str], track ids
    agent_future          (N, K, 2) future positions of the boxes in the current ego frame
    agent_future_mask     (N, K)
    map                   [(type name, (M, 2) polyline)]
    occ_path, occ_format  optional occupancy label file and its format
    ego_status_mask       optional (S,) weight of each value of the profile's status_layout in the ego status loss
    task_valid            optional legacy-compatible mapping of explicit bools: planning, det, map, motion, occupancy,
                          geometry. False means the label is unknown/unavailable, not an empty negative scene. An
                          empty list with a true flag is a genuinely labelled empty scene. New sources should provide
                          all six flags; legacy sources may omit the mapping (other tasks fully supervised,
                          occupancy optional and enabled only when occ_path exists).

A source with frame order has `sequences()`: lists of dataset indices, each in time order (0.5 s apart), which the
training loader feeds consecutively.
"""

import math
from typing import Dict, List

import numpy as np

VIEWS = ("front", "front_tele", "front_left", "front_right", "left", "right", "back_left", "back_right", "back")
COMMANDS = ("left", "straight", "right", "unknown", "lane_follow", "change_lane_left", "change_lane_right")
TASKS = ("planning", "det", "map", "motion", "occupancy", "geometry")
NAN3 = np.full(3, np.nan, dtype=np.float32)


def yaw_of(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def to_local(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Global (x, y, yaw) poses in the frame of the global pose `origin`."""
    c, s = math.cos(origin[2]), math.sin(origin[2])
    dx, dy = poses[:, 0] - origin[0], poses[:, 1] - origin[1]
    heading = (poses[:, 2] - origin[2] + math.pi) % (2 * math.pi) - math.pi
    return np.stack([c * dx + s * dy, -s * dx + c * dy, heading], axis=1).astype(np.float32)


def transform_xy(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """(N, 2) points on the ground plane through the 4x4 `transform`."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (points @ transform[:2, :2].T + transform[:2, 3]).astype(np.float32)


def transform_boxes(boxes: np.ndarray, velocity: np.ndarray, transform: np.ndarray):
    """(N, 7) boxes and (N, 2) velocities from the frame of `transform` (4x4, source -> ego) into the ego frame."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7).copy()
    centers = np.concatenate([boxes[:, :3], np.ones((len(boxes), 1))], axis=1) @ transform.T
    boxes[:, :3] = centers[:, :3]
    boxes[:, 6] = (boxes[:, 6] + yaw_of(transform) + math.pi) % (2 * math.pi) - math.pi
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 2) @ transform[:2, :2].T
    return boxes.astype(np.float32), velocity.astype(np.float32)


def camera(view: str, path: str, intrinsic, sensor2ego, size) -> dict:
    if view not in VIEWS:
        raise ValueError(f"camera view {view!r} is not one of {VIEWS}")
    return {"view": view, "path": str(path), "intrinsic": np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
            "sensor2ego": np.asarray(sensor2ego, dtype=np.float64).reshape(4, 4), "size": tuple(int(v) for v in size)}


def check_sample(features: Dict, targets: Dict) -> List[str]:
    """Contract violations of one sample; empty when it follows the contract."""
    problems = []
    for key in ("cameras", "history_trajectory", "ego_velocity", "ego_acceleration", "ego_angular_velocity",
                "ego_steer", "command"):
        if key not in features:
            problems.append(f"features lack {key}")
    if features.get("command") not in COMMANDS:
        problems.append(f"command {features.get('command')!r} is not one of {COMMANDS}")
    if np.shape(features.get("history_trajectory")) != (4, 3):
        problems.append(f"history_trajectory has shape {np.shape(features.get('history_trajectory'))}")
    for cam in features.get("cameras", []):
        if cam.get("view") not in VIEWS:
            problems.append(f"camera view {cam.get('view')!r}")
    if not targets:
        return problems
    task_valid = targets.get("task_valid")
    if task_valid is not None:
        if not isinstance(task_valid, dict):
            problems.append("targets.task_valid must be a mapping")
        else:
            missing, extra = set(TASKS) - set(task_valid), set(task_valid) - set(TASKS)
            if missing:
                problems.append(f"targets.task_valid lacks {sorted(missing)}")
            if extra:
                problems.append(f"targets.task_valid has unknown tasks {sorted(extra)}")
            for task, valid in task_valid.items():
                if task in TASKS and not isinstance(valid, (bool, np.bool_)):
                    problems.append(f"targets.task_valid[{task!r}] must be bool")
            if bool(task_valid.get("motion")) and not bool(task_valid.get("det")):
                problems.append("motion supervision requires compatible detection targets")
    validity = task_valid if isinstance(task_valid, dict) and set(task_valid) == set(TASKS) else {
        task: bool(targets.get("occ_path")) if task == "occupancy" else True for task in TASKS
    }
    n = len(targets.get("boxes", []))
    required_shapes = []
    if validity["planning"]:
        required_shapes.extend((("trajectory", (None, 3)), ("trajectory_mask", (None,))))
    if validity["det"]:
        required_shapes.extend((("boxes", (n, 7)), ("box_velocity", (n, 2))))
        if "box_names" not in targets:
            problems.append("targets lack box_names")
    if validity["map"] and "map" not in targets:
        problems.append("targets lack map")
    if validity["motion"]:
        required_shapes.extend((("agent_future", (n, None, 2)), ("agent_future_mask", (n, None))))
    for key, shape in required_shapes:
        got = np.shape(targets.get(key))
        if len(got) != len(shape) or any(s is not None and s != g for s, g in zip(shape, got)):
            problems.append(f"targets[{key}] has shape {got}, expected {shape}")
    if validity["det"] and len(targets.get("box_names", [])) != n:
        problems.append("box_names and boxes differ in length")
    if validity["map"]:
        for entry in targets.get("map", []):
            if len(entry) != 2 or np.ndim(entry[1]) != 2 or np.shape(entry[1])[1] != 2:
                problems.append(f"map entry {entry[0] if entry else entry!r} is not (name, (M, 2))")
                break
    if validity["occupancy"] and not targets.get("occ_path"):
        problems.append("occupancy supervision is valid but occ_path is missing")
    return problems
