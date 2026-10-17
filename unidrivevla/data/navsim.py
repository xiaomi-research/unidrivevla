"""NAVSIM scenes (the 1.1 or 2.0 devkit on PYTHONPATH) as canonical samples.

The builders run inside the official NAVSIM machinery: run_dataset_caching.py with agent=unidrivevla_navsim (cache_mode)
writes one feature and one target file per scene, and training reads them with the official CacheOnlyDataset. Cameras
are image paths (the devkit's SensorConfig.load_image_path), so a cached scene takes a few KB. NAVSIM's lidar2ego is the
identity: boxes, trajectories and sensor2lidar are all in the rear-axle ego frame. navsim is imported lazily, so the
package imports without a NAVSIM tree.
"""

import inspect
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from unidrivevla.data.registry import register
from unidrivevla.data.sample import NAN3, TASKS, camera, to_local

NAVSIM_VIEWS = {"front": "cam_f0", "front_left": "cam_l0", "left": "cam_l1", "back_left": "cam_l2",
                "front_right": "cam_r0", "right": "cam_r1", "back_right": "cam_r2", "back": "cam_b0"}
OPENSCENE_COMMANDS = ("left", "straight", "right", "unknown")  # order of the driving_command one-hot
MAP_RADIUS_M = 60.0
EDGE_M = 0.25  # road polygons are merged over gaps this wide; dividers within twice it of the road edge are the edge
NAVSIM_TASKS = TASKS


def _target_shape(targets, key, ndim, last_dim=None):
    try:
        value = np.asarray(targets[key])
    except (TypeError, ValueError):
        raise ValueError(f"NAVSIM target {key!r} is not an array") from None
    if value.ndim != ndim or (last_dim is not None and value.shape[-1] != last_dim):
        raise ValueError(f"NAVSIM target {key!r} has invalid shape {value.shape}")
    return value


def infer_navsim_task_validity(targets, map_available=None):
    """Infer only supervision represented by this built-in NAVSIM sample/cache.

    The standard NAVSIM target builder never writes occupancy labels. For legacy caches, field presence is the only
    schema evidence available: a present empty boxes/map array is a labelled-empty sample, while an absent field is
    unknown. Newly computed targets can additionally pass whether Scene.map_api was actually available.
    """
    if not isinstance(targets, Mapping):
        raise TypeError("NAVSIM targets must be a mapping")

    planning = False
    if "trajectory" in targets and "trajectory_mask" in targets:
        trajectory = _target_shape(targets, "trajectory", 2, 3)
        trajectory_mask = _target_shape(targets, "trajectory_mask", 1)
        if len(trajectory) != len(trajectory_mask):
            raise ValueError("NAVSIM trajectory and trajectory_mask lengths differ")
        planning = bool(len(trajectory) and np.any(trajectory_mask > 0))

    det = False
    box_keys = ("boxes", "box_velocity", "box_names")
    if all(key in targets for key in box_keys):
        boxes = _target_shape(targets, "boxes", 2, 7)
        velocity = _target_shape(targets, "box_velocity", 2)
        names = targets["box_names"]
        if velocity.shape[1] not in (2, 3) or len(boxes) != len(velocity) or len(boxes) != len(names):
            raise ValueError("NAVSIM boxes, box_velocity, and box_names must have matching rows")
        det = True

    map_valid = False
    if "map" in targets:
        map_rows = targets["map"]
        if not isinstance(map_rows, (list, tuple)):
            raise TypeError("NAVSIM target 'map' must be a list of labelled polylines")
        for row in map_rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2 or not isinstance(row[0], str):
                raise ValueError("NAVSIM map rows must be (name, (M, 2)) pairs")
            points = _target_shape({"points": row[1]}, "points", 2, 2)
            if len(points) < 2:
                raise ValueError("NAVSIM map polylines must contain at least two points")
        # On old caches, key presence is the only evidence retained by NAVSIM's pickle schema. New scenes also
        # expose map_api directly, so a missing API cannot be mistaken for a labelled empty map.
        map_valid = True if map_available is None else bool(map_available)

    motion = False
    motion_keys = ("agent_future", "agent_future_mask")
    if det and all(key in targets for key in motion_keys):
        future = _target_shape(targets, "agent_future", 3, 2)
        future_mask = _target_shape(targets, "agent_future_mask", 2)
        if future.shape[:2] != future_mask.shape or future.shape[0] != len(targets["boxes"]):
            raise ValueError("NAVSIM agent_future and agent_future_mask must match detection rows and horizons")
        motion = True

    occupancy = bool(targets.get("occ_path"))
    return {
        "planning": planning,
        "det": det,
        "map": map_valid,
        "motion": motion,
        "occupancy": occupancy,
        "geometry": bool(det or map_valid),
    }


def normalize_navsim_targets(targets):
    """Add explicit per-task validity to new samples and pre-validity NAVSIM cache rows."""
    if not isinstance(targets, Mapping):
        raise TypeError("NAVSIM targets must be a mapping")
    normalized = dict(targets)
    explicit = normalized.get("task_valid")
    if explicit is None:
        normalized["task_valid"] = infer_navsim_task_validity(normalized)
        return normalized
    if not isinstance(explicit, Mapping) or set(explicit) != set(NAVSIM_TASKS):
        raise ValueError(f"NAVSIM task_valid must contain exactly {NAVSIM_TASKS}")
    if any(not isinstance(explicit[task], (bool, np.bool_)) for task in NAVSIM_TASKS):
        raise TypeError("NAVSIM task_valid values must be booleans")
    flags = {task: bool(explicit[task]) for task in NAVSIM_TASKS}
    if flags["motion"] and not flags["det"]:
        raise ValueError("NAVSIM motion supervision requires detection annotations")
    if flags["geometry"] and not (flags["det"] or flags["map"]):
        raise ValueError("NAVSIM geometry supervision requires detection or map annotations")
    if flags["occupancy"] and not normalized.get("occ_path"):
        raise ValueError("NAVSIM occupancy supervision requires occ_path")
    available = infer_navsim_task_validity(normalized)
    unsupported = [task for task, valid in flags.items() if valid and not available[task]]
    if unsupported:
        raise ValueError(f"NAVSIM task_valid marks unavailable annotations valid: {unsupported}")
    normalized["task_valid"] = flags
    return normalized


def sensor_config():
    from navsim.common.dataclasses import SensorConfig

    current = [3]  # the current frame of the 4 history frames, every camera, no lidar
    return SensorConfig(cam_f0=current, cam_l0=current, cam_l1=current, cam_l2=current, cam_r0=current,
                        cam_r1=current, cam_r2=current, cam_b0=current, lidar_pc=False, load_image_path=True)


def global_from_local(xy: np.ndarray, pose) -> np.ndarray:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack([c * xy[:, 0] - s * xy[:, 1] + pose[0], s * xy[:, 0] + c * xy[:, 1] + pose[1]], axis=1)


def local_xy(xy: np.ndarray, origin) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return to_local(np.concatenate([xy, np.zeros((len(xy), 1))], axis=1), np.asarray(origin, np.float64))[:, :2]


class SampleFeatureBuilder:
    """AgentInput -> canonical features. Duck-typed NAVSIM feature builder."""

    def get_unique_name(self) -> str:
        return "unidrivevla_sample_feature"

    def compute_features(self, agent_input) -> dict:
        cams = agent_input.cameras[-1]
        cameras = []
        for view, attr in NAVSIM_VIEWS.items():
            cam = getattr(cams, attr)
            if not isinstance(cam.image, (str, Path)):
                raise TypeError(f"{attr}.image must be a file path: the devkit needs the SensorConfig.load_image_path patch")
            with Image.open(cam.image) as image:
                size = image.size
            sensor2ego = np.eye(4)
            sensor2ego[:3, :3] = cam.sensor2lidar_rotation
            sensor2ego[:3, 3] = np.asarray(cam.sensor2lidar_translation).reshape(3)
            cameras.append(camera(view, cam.image, cam.intrinsics, sensor2ego, size))
        status = agent_input.ego_statuses[-1]
        command = np.asarray(status.driving_command).reshape(-1)
        velocity = np.asarray(status.ego_velocity, np.float32).reshape(-1)[:2]
        acceleration = np.asarray(status.ego_acceleration, np.float32).reshape(-1)[:2]
        return {
            "cameras": cameras,
            "history_trajectory": np.stack([np.asarray(s.ego_pose, np.float32) for s in agent_input.ego_statuses])[-4:],
            "ego_velocity": np.array([velocity[0], velocity[1], np.nan], np.float32),
            "ego_acceleration": np.array([acceleration[0], acceleration[1], np.nan], np.float32),
            "ego_angular_velocity": NAN3.copy(),
            "ego_steer": np.float32(np.nan),
            "command": OPENSCENE_COMMANDS[int(command.argmax())],
        }


def agent_futures(scene, current: int, steps: int):
    """Future positions of the current frame's boxes (by track token) in the current ego frame."""
    frame = scene.frames[current]
    tokens = list(frame.annotations.track_tokens)
    future = np.zeros((len(tokens), steps, 2), np.float32)
    mask = np.zeros((len(tokens), steps), np.float32)
    origin = np.asarray(frame.ego_status.ego_pose, np.float64)
    index = {token: i for i, token in enumerate(tokens)}
    for k in range(steps):
        if current + 1 + k >= len(scene.frames):
            break
        later = scene.frames[current + 1 + k]
        boxes = np.asarray(later.annotations.boxes, np.float64).reshape(-1, 7)
        rows = [(index[t], j) for j, t in enumerate(later.annotations.track_tokens) if t in index]
        if not rows:
            continue
        mine, theirs = map(list, zip(*rows))
        pose = np.asarray(later.ego_status.ego_pose, np.float64)
        future[mine, k] = local_xy(global_from_local(boxes[theirs, :2], pose), origin)
        mask[mine, k] = 1
    return future, mask


def lines_of(geometry):
    """The LineStrings of a (multi) line or polygon geometry; polygons give their rings."""
    if geometry.is_empty or geometry.geom_type in ("Point", "MultiPoint"):
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [geometry]
    if geometry.geom_type == "Polygon":
        return [geometry.exterior, *geometry.interiors]
    return [line for part in geometry.geoms for line in lines_of(part)]


def map_polylines(scene, frame):
    """Map elements as nuScenes builds them: crosswalk outlines (ped_crossing); the boundaries of lanes, joined where
    they meet, except those on the road edge (divider; lane connectors are virtual lanes through junctions, without
    painted lines); the outline of the union of roadblocks, roadblock connectors and intersections (boundary).
    Neighbouring lanes share their boundary objects, so each is taken once."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from shapely.ops import linemerge, unary_union

    if scene.map_api is None:
        return []
    areas = [L.ROADBLOCK, L.ROADBLOCK_CONNECTOR, L.INTERSECTION]
    pose = np.asarray(frame.ego_status.ego_pose, np.float64)
    found = scene.map_api.get_proximal_map_objects(Point2D(pose[0], pose[1]), MAP_RADIUS_M, [L.CROSSWALK, L.LANE, *areas])
    polygons = [record.polygon for layer in areas for record in found.get(layer, [])]
    area = unary_union([p.buffer(EDGE_M) for p in polygons]).buffer(-EDGE_M) if polygons else None  # closes slivers
    sides = {side.id: side.linestring for lane in found.get(L.LANE, [])
             for side in (lane.left_boundary, lane.right_boundary) if getattr(side, "linestring", None) is not None}
    dividers = lines_of(linemerge(list(sides.values()))) if sides else []
    if area is not None and not area.is_empty:
        edge = area.boundary.buffer(2 * EDGE_M)
        dividers = [line for line in dividers if not line.within(edge)]
    elements = [("ped_crossing", record.polygon.exterior) for record in found.get(L.CROSSWALK, [])]
    elements += [("divider", line) for line in dividers]
    elements += [("boundary", line) for line in (lines_of(area) if area is not None else [])]
    return [(name, local_xy(np.asarray(line.coords)[:, :2], pose)) for name, line in elements if len(line.coords) >= 2]


class SampleTargetBuilder:
    """Scene -> canonical targets. Duck-typed NAVSIM target builder."""

    def get_unique_name(self) -> str:
        return "unidrivevla_sample_target"

    def compute_targets(self, scene) -> dict:
        meta = scene.scene_metadata
        current, steps = meta.num_history_frames - 1, meta.num_future_frames
        trajectory = scene.get_future_trajectory(num_trajectory_frames=steps).poses.astype(np.float32)
        frame = scene.frames[current]
        annotations = frame.annotations
        future, future_mask = agent_futures(scene, current, steps)
        targets = {
            "trajectory": trajectory,
            "trajectory_mask": np.ones(len(trajectory), np.float32),
            "boxes": np.asarray(annotations.boxes, np.float32).reshape(-1, 7),
            "box_velocity": np.asarray(annotations.velocity_3d, np.float32).reshape(-1, 3)[:, :2],
            "box_names": [str(n) for n in annotations.names],
            "box_ids": [str(t) for t in annotations.track_tokens],
            "agent_future": future,
            "agent_future_mask": future_mask,
            "map": map_polylines(scene, frame),
        }
        targets["task_valid"] = infer_navsim_task_validity(targets, map_available=scene.map_api is not None)
        return targets


class WithTokens(Dataset):
    """(features, targets) of a NAVSIM dataset plus the scene token."""

    def __init__(self, dataset, tokens):
        self.dataset, self.tokens = dataset, list(tokens)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, targets = self.dataset[index]
        return features, normalize_navsim_targets(targets), self.tokens[index]


class AgentInputs(Dataset):
    """Features of each scene's AgentInput, with its token and no targets: closed-loop evaluation input."""

    def __init__(self, loader):
        self.loader = loader
        self.tokens = list(loader.tokens)
        self.builder = SampleFeatureBuilder()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        return self.builder.compute_features(self.loader.get_agent_input_from_token(token)), {}, token


def scene_loader(cfg, scene_filter):
    from navsim.common.dataloader import SceneLoader

    kwargs = {}
    if "synthetic_scenes_path" in inspect.signature(SceneLoader).parameters:  # NAVSIM 2.0; read for two-stage splits
        path = OmegaConf.select(cfg, "synthetic_scenes_path")
        kwargs["synthetic_scenes_path"] = Path(path) if path else None
    return SceneLoader(data_path=Path(cfg.navsim_log_path), sensor_blobs_path=Path(cfg.sensor_blobs_path),
                       scene_filter=scene_filter, sensor_config=sensor_config(), **kwargs)


@register("navsim")
def navsim_loader(cfg, agent, split: str):
    """train / val: cached scenes of train_logs / val_logs; test: every scene of train_test_split without targets."""
    from hydra.utils import instantiate
    from navsim.planning.training.dataset import CacheOnlyDataset
    from navsim.planning.training.dataset import Dataset as SceneDataset

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    if split == "test":
        return AgentInputs(scene_loader(cfg, scene_filter))
    features, targets = [SampleFeatureBuilder()], [SampleTargetBuilder()]
    logs = list(cfg.train_logs if split == "train" else cfg.val_logs)
    if OmegaConf.select(cfg, "use_cache_without_dataset"):
        if cfg.force_cache_computation:
            raise AssertionError("force_cache_computation must be False when training from the cache")
        dataset = CacheOnlyDataset(cache_path=cfg.cache_path, feature_builders=features, target_builders=targets,
                                   log_names=logs)
        return WithTokens(dataset, dataset.tokens)
    scene_filter.log_names = [n for n in scene_filter.log_names if n in logs] if scene_filter.log_names else logs
    loader = scene_loader(cfg, scene_filter)
    dataset = SceneDataset(scene_loader=loader, feature_builders=features, target_builders=targets,
                           cache_path=cfg.cache_path, force_cache_computation=cfg.force_cache_computation)
    return WithTokens(dataset, loader.tokens)
