"""UniDriveVLA as a policy adapter.

A profile (config/profile/*.yaml) holds everything that depends on the training data: class, map-type and command
vocabularies, camera order and view tokens, horizons, ranges, anchors, and trajectory normalization. The model code
holds none of it, so the same code trains on any registered source. `prepare_sample` runs in the dataloader workers
and turns a source sample (unidrivevla.data.sample) in its declared profile frame into planning-head tensors,
fixed-size and padded so batches stack.
"""

import functools
import logging
import math
import os
from dataclasses import dataclass, field, fields
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn

logger = logging.getLogger(__name__)

POSE_INTERVAL_S = 0.5
MAP_PAD = 1e5  # permutation rows of open polylines, as the nuScenes VectorizeMap pads them


@dataclass
class Profile:
    name: str
    cameras: List[str]
    view_tokens: List[str]
    horizon: int
    commands: List[str]
    command_prompts: List[str]
    status_layout: List[str]
    det_classes: List[str]
    box_range: List[float]  # x_min, y_min, x_max, y_max in profile.frame
    map_classes: List[str]
    map_roi: List[float]  # x/y extent, centred at the source origin in profile.frame
    ego_box: List[float]  # x, y, z, length, width, height, yaw of the ego anchor
    anchors_dir: str
    delta_mean: List[float]
    delta_std: List[float]
    frame: str = "ego"
    pose_interval: float = 0.5  # seconds between planned poses; LEAD release is 4 Hz (0.25)
    image_hw: List[int] = field(default_factory=lambda: [544, 960])
    motion_steps: int = 12
    train_precision: str = "native"  # parameter construction dtype recipe; metadata only in the profile adapter
    ddp_cast_frozen_parameters: bool = False  # DDP recipe metadata; other distributed modes keep their own policy
    ddp_reduce_dtype: str = "auto"  # gradient communication mode; "auto" preserves the legacy/runtime default
    command_aliases: Dict[str, str] = field(default_factory=dict)
    det_class_map: Dict[str, str] = field(default_factory=dict)
    box_radius: Optional[float] = None
    max_boxes: int = 128  # minimum padded rows; larger samples retain every target
    map_class_map: Dict[str, str] = field(default_factory=dict)
    map_points: int = 20
    max_map: int = 64  # minimum padded rows; larger samples retain every target
    map_clip: bool = True  # clip lines to map_roi on the ego axes; false: the source's lines are its range already
    ground_height: float = 0.0
    reverse_classes: List[str] = field(default_factory=list)  # yaw and yaw + pi are both right (cls_allow_reverse)
    yawless_classes: List[str] = field(default_factory=list)  # no yaw regression (cls_wise_reg_weights)
    occ_classes: Optional[int] = None
    occ_format: Optional[str] = None
    prefix_max_len: Optional[int] = None
    ego_keypoints: Optional[Dict] = None  # the ego query's SparseBox3DKeyPointsGenerator; null: keypoints_box()
    train_resize: Optional[float] = None  # the official resize + crop at this training scale (null: stretch to image_hw)
    zero_velocity_classes: List[str] = field(default_factory=list)  # cached detections of these keep no velocity
    decoder_norm: bool = True  # LayerNorm at the unified decoder's "norm" operations
    motion_velocity_weight: float = 0.0  # the detection box loss's vx, vy weight once motion trains (stage 2)
    prompt_style: str = "nuscenes"  # nuscenes / b2d: the official prompt text; ego: canonical-frame wording
    use_hist_traj: bool = True  # the history token in the action suffix (Bench2Drive has none)
    use_occ_tokens: bool = True  # the 625 occupancy queries (Bench2Drive's perception stream has none)
    keypoint_from_anchor: bool = False  # Bench2Drive's deformable block feeds the anchor embed to learnable points
    box_size_clamp: Optional[List[float]] = None  # log-size clamp; null: nuScenes [-0.5, 3.5], Bench2Drive [-5, 5]
    inference_attn_impl: str = "sdpa"
    cast_inference_mask: bool = True  # false: Bench2Drive leaves the inference mask in fp32
    detach_plan_geometry: bool = False  # Bench2Drive detaches the trajectory before the collision / map losses
    summed_planning_loss: bool = False  # true: Bench2Drive's sum returned as `loss` too, those terms counted twice
    official_command_text: bool = False  # true: the official heads' command.view(-1) read of the one-hot in the prompt
    official_attention_mask: bool = False  # main's padding-key visibility in Flex attention; opt-in for native profiles
    processor_use_fast: Optional[bool] = None  # Bench2Drive loads the processor with use_fast=False
    cast_stream_dtype: bool = False  # Bench2Drive casts a single-stream embed to the expert dtype
    nan_to_num_cost: bool = False  # Bench2Drive's map matcher
    enable_knowledge_insulation: bool = False  # optional main branch; B2D includes its prompt-token CE
    ar_loss_weight: float = 0.1  # main defaults: nuScenes 0.1, Bench2Drive native 1.0
    lr_microstep: bool = False  # main's native schedules update the LR every data micro-batch

    @classmethod
    def from_config(cls, cfg) -> "Profile":
        values = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        profile = cls(**values)
        if not isinstance(profile.train_precision, str) or profile.train_precision not in ("native", "bf16", "fp32"):
            raise ValueError("profile.train_precision must be one of 'native', 'bf16', or 'fp32'")
        if type(profile.ddp_cast_frozen_parameters) is not bool:
            raise ValueError("profile.ddp_cast_frozen_parameters must be a boolean")
        if not isinstance(profile.ddp_reduce_dtype, str) or profile.ddp_reduce_dtype not in ("auto", "native", "fp32", "bf16"):
            raise ValueError("profile.ddp_reduce_dtype must be one of 'auto', 'native', 'fp32', or 'bf16'")
        if len(profile.cameras) != len(profile.view_tokens):
            raise ValueError("profile.cameras and profile.view_tokens differ in length")
        if len(profile.commands) != len(profile.command_prompts):
            raise ValueError("profile.commands and profile.command_prompts differ in length")
        if profile.frame not in ("ego", "lidar"):
            raise ValueError(f"profile.frame must be 'ego' or 'lidar', got {profile.frame!r}")
        return profile

    def command_id(self, name: str) -> int:
        name = self.command_aliases.get(name, name)
        if name not in self.commands:
            raise KeyError(f"command {name!r} is neither in profile.commands {self.commands} nor aliased")
        return self.commands.index(name)

    def det_label(self, name: str) -> Optional[int]:
        name = self.det_class_map.get(name, name)
        return self.det_classes.index(name) if name in self.det_classes else None

    def map_label(self, name: str) -> Optional[int]:
        name = self.map_class_map.get(name, name)
        return self.map_classes.index(name) if name in self.map_classes else None


# ----------------------------------------------------------------------------- canonical sample -> head tensors

def status_value(features: Dict, name: str) -> float:
    if name == "speed":
        return float(np.hypot(*np.asarray(features["ego_velocity"], dtype=np.float64)[:2]))
    if name == "steer":
        return float(features["ego_steer"])
    source = {"acc": "ego_acceleration", "vel": "ego_velocity", "rot": "ego_angular_velocity"}[name[:3]]
    return float(np.asarray(features[source], dtype=np.float64)["xyz".index(name[-1])])


def resize_crop(size, final_hw, train_resize: Optional[float], train: bool):
    """The official get_augmentation's resize and crop (bot_pct_lim 0): training scales by train_resize and crops the
    bottom-aligned final size (a negative top pads), evaluation scales to cover it and crops the centre.
    (scale, resized (w, h), crop box), or None: without train_resize the image is stretched to the final size."""
    if train_resize is None:
        return None
    (width, height), (final_h, final_w) = size, final_hw
    scale = train_resize if train else max(final_h / height, final_w / width)
    new_w, new_h = int(width * scale), int(height * scale)
    if train and new_w > final_w:
        raise NotImplementedError("the official random horizontal crop: train_resize makes images wider than image_hw")
    left = 0 if train else int(max(0, new_w - final_w) / 2)
    top = new_h - final_h
    return scale, (new_w, new_h), (left, top, left + final_w, top + final_h)


def load_bgr(path, height: int, width: int, image_augment=None, geometry=None, crop=None) -> np.ndarray:
    """uint8: the head normalizes on the GPU, so the workers and the host-to-device copy move a quarter of float32."""
    if isinstance(path, np.ndarray):
        if path.dtype != np.uint8 or path.ndim != 3 or path.shape[2] != 3:
            raise ValueError("in-memory camera images must be RGB uint8 HxWx3 arrays")
        source = Image.fromarray(path)
    else:
        source = Image.open(path)
    with source as image:
        if crop is None:
            image = image.convert("RGB").resize((width, height), Image.BILINEAR)
        else:  # bicubic: Pillow's default, which the official resize leaves
            image = image.convert("RGB").resize(crop[1], Image.BICUBIC).crop(crop[2])
    if geometry is not None:
        if geometry["flip"]:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.rotate(geometry["rotate"])
    if image_augment is not None:
        image = image_augment(image)
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


def geometry_matrix(width: int, height: int, geometry) -> np.ndarray:
    """4x4 on projected pixels for load_bgr's flip, then rotation about the image centre (the official
    ResizeCropFlipImage after its resize)."""
    out = np.eye(4)
    if geometry is None:
        return out
    flip = np.array([[-1.0, 0, width], [0, 1, 0], [0, 0, 1]]) if geometry["flip"] else np.eye(3)
    angle = math.radians(geometry["rotate"])
    rotate = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    center = np.array([width, height]) / 2
    rotate[:2, 2] = center - rotate[:2, :2] @ center
    out[:3, :3] = rotate @ flip
    return out


def camera_inputs(features: Dict, profile: Profile, image_augment=None, train: bool = False) -> Dict[str, torch.Tensor]:
    height, width = profile.image_hw
    by_view = {cam["view"]: cam for cam in features["cameras"]}
    missing = [view for view in profile.cameras if view not in by_view]
    if missing:
        raise KeyError(f"profile {profile.name} needs cameras {missing}, the source has {sorted(by_view)}")
    geometry = features.get("augment")
    augment_mat = geometry_matrix(width, height, geometry)
    images, mats, focals = [], [], []
    for view in profile.cameras:
        cam = by_view[view]
        intrinsic = np.array(cam["intrinsic"], dtype=np.float64)
        crop = resize_crop(cam["size"], (height, width), profile.train_resize, train)
        if crop is None:
            intrinsic[0] *= width / cam["size"][0]
            intrinsic[1] *= height / cam["size"][1]
        else:
            intrinsic[:2] *= crop[0]
            intrinsic[:2, 2] -= crop[2][:2]
        view_mat = np.eye(4)
        view_mat[:3, :3] = intrinsic
        mats.append(augment_mat @ view_mat @ np.linalg.inv(np.asarray(cam["sensor2ego"], dtype=np.float64)))
        focals.append(intrinsic[0, 0])
        pixels = cam["image"] if "image" in cam else cam["path"]
        images.append(load_bgr(pixels, height, width, image_augment, geometry, crop).transpose(2, 0, 1))
    views = len(profile.cameras)
    return {
        "img": torch.from_numpy(np.stack(images)),
        "projection_mat": torch.from_numpy(np.stack(mats).astype(np.float32)),
        "image_wh": torch.tensor([[width, height]] * views, dtype=torch.float32),
        "focal": torch.tensor(focals, dtype=torch.float32),
    }


def pad_rows(array: np.ndarray, count: int):
    count = max(count, len(array))
    out = np.zeros((count,) + array.shape[1:], dtype=array.dtype)
    mask = np.zeros((count,), np.float32)
    kept = len(array)
    out[:kept] = array[:kept]
    mask[:kept] = 1
    return out, mask


def permute_line(line: np.ndarray) -> np.ndarray:
    """(P, 2) polyline -> (2 (P - 1), P, 2): every start point and direction of a closed line; the line and its reverse,
    padded, for an open one."""
    points = len(line)
    count = 2 * (points - 1)
    if np.allclose(line[0], line[-1], atol=1e-3):
        ring = line[:-1]
        rows = [np.roll(ring, shift, axis=0) for shift in range(points - 1)]
        rows += [np.roll(np.flip(ring, 0), shift, axis=0) for shift in range(points - 1)]
        out = np.zeros((count, points, 2), np.float32)
        out[:, :-1] = np.stack(rows)
        out[:, -1] = out[:, 0]
        return out
    head = np.stack([line, np.flip(line, 0)]).astype(np.float32)
    return np.concatenate([head, np.full((count - 2, points, 2), MAP_PAD, np.float32)], axis=0)


def map_targets(polylines, profile: Profile):
    from shapely.geometry import LineString, MultiLineString
    from shapely.geometry import box as roi_box

    half_x, half_y = profile.map_roi[0] / 2, profile.map_roi[1] / 2
    roi = roi_box(-half_x, -half_y, half_x, half_y)
    pts, labels = [], []
    for name, coords in polylines:
        label = profile.map_label(name)
        coords = np.asarray(coords, dtype=np.float64)
        if label is None or len(coords) < 2:
            continue
        if not profile.map_clip:  # the source's lines, sampled as they come
            if len(coords) != profile.map_points:
                line = LineString(coords)
                coords = np.array([line.interpolate(d).coords[0][:2]
                                   for d in np.linspace(0.0, line.length, profile.map_points)])
            pts.append(permute_line(coords.astype(np.float32)))
            labels.append(label)
            continue
        clipped = LineString(coords).intersection(roi)
        parts = clipped.geoms if isinstance(clipped, MultiLineString) else [clipped]
        for part in parts:
            if not isinstance(part, LineString) or part.length < 1e-3:
                continue
            distances = np.linspace(0.0, part.length, profile.map_points)
            sampled = np.array([part.interpolate(d).coords[0][:2] for d in distances], dtype=np.float32)
            pts.append(permute_line(sampled))
            labels.append(label)
    shape = (0, 2 * (profile.map_points - 1), profile.map_points, 2)
    pts = np.stack(pts) if pts else np.zeros(shape, np.float32)
    return pad_rows(pts, profile.max_map), pad_rows(np.asarray(labels, np.int64), profile.max_map)[0]


def box_targets(targets: Dict, profile: Profile):
    boxes = np.asarray(targets.get("boxes", []), dtype=np.float32).reshape(-1, 7)
    velocity = np.asarray(targets.get("box_velocity", np.zeros((len(boxes), 2))), dtype=np.float32).reshape(-1, 2)
    labels = np.array([-1 if (l := profile.det_label(n)) is None else l for n in targets.get("box_names", [])], np.int64)
    if len(velocity) != len(boxes) or len(labels) != len(boxes):
        raise ValueError("boxes, box_velocity, and box_names must have the same number of rows")
    x_min, y_min, x_max, y_max = profile.box_range
    keep = (labels >= 0) & (boxes[:, 0] >= x_min) & (boxes[:, 0] <= x_max) & (boxes[:, 1] >= y_min) & (boxes[:, 1] <= y_max)
    if profile.box_radius is not None:
        keep &= np.hypot(boxes[:, 0], boxes[:, 1]) <= profile.box_radius
    # motion: per-step displacements, each from the box's last observed position
    steps = profile.motion_steps
    future = np.asarray(targets.get("agent_future", np.zeros((len(boxes), 0, 2))), dtype=np.float32)
    future_mask = np.asarray(targets.get("agent_future_mask", np.zeros((len(boxes), 0))), dtype=np.float32)
    if len(boxes) == 0 or future.ndim != 3:
        future, future_mask = np.zeros((len(boxes), 0, 2), np.float32), np.zeros((len(boxes), 0), np.float32)
    future, future_mask = future[:, :steps], future_mask[:, :steps]
    deltas = np.zeros((len(boxes), steps, 2), np.float32)
    masks = np.zeros((len(boxes), steps), np.float32)
    last = boxes[:, :2].copy()
    for k in range(min(steps, future.shape[1])):
        seen = future_mask[:, k] > 0
        deltas[seen, k] = future[seen, k] - last[seen]
        masks[seen, k] = 1
        last[seen] = future[seen, k]
    gt = np.concatenate([boxes, velocity], axis=1)[keep]
    box, box_mask = pad_rows(gt, profile.max_boxes)
    return {
        "gt_bboxes_3d": torch.from_numpy(box),
        "gt_labels_3d": torch.from_numpy(pad_rows(labels[keep], profile.max_boxes)[0]),
        "box_mask": torch.from_numpy(box_mask),
        "gt_agent_fut_trajs": torch.from_numpy(pad_rows(deltas[keep], profile.max_boxes)[0]),
        "gt_agent_fut_masks": torch.from_numpy(pad_rows(masks[keep], profile.max_boxes)[0]),
    }


def load_occ(path: str, fmt: str) -> np.ndarray:
    if fmt == "occ3d":  # Occ3D / OccWorld labels.npz, (200, 200, 16) semantics
        return np.load(path)["semantics"].astype(np.int64)
    raise ValueError(f"unknown occupancy format {fmt!r}")


def prepare_sample(features: Dict, targets: Dict, profile: Profile, image_augment=None, train: bool = False):
    """Source-frame sample -> planning-head tensors; runs in the dataloader workers."""
    native_to_ego = _source_frame_rotation(features, profile)
    out = camera_inputs(features, profile, image_augment, train)
    if "coordinate_frame" in features:
        out["coordinate_frame"] = str(features["coordinate_frame"])
        out["native_to_ego"] = torch.from_numpy(native_to_ego.copy())
    status = np.nan_to_num(np.array([status_value(features, n) for n in profile.status_layout], np.float32))
    pose = features.get("ego2global")
    if pose is not None or "stream_id" in features:
        pose = np.eye(4) if pose is None else np.asarray(pose, np.float64)
        out["img_metas"] = {"T_global": pose, "T_global_inv": np.linalg.inv(pose)}
        if "stream_id" in features:
            out["img_metas"]["stream_id"] = features["stream_id"]
    out.update(
        # float64: absolute seconds (nuScenes ~1.5e9) keep their 0.5 s steps
        timestamp=torch.tensor(float(features.get("timestamp", 0.0)), dtype=torch.float64),
        ego_status=torch.from_numpy(status),
        gt_ego_fut_cmd=torch.tensor(profile.command_id(features["command"]), dtype=torch.long),
        hist_traj=torch.from_numpy(np.asarray(features["history_trajectory"], np.float32)[:, :2].copy()),
    )
    if not targets:
        return out, {}
    task_valid = targets.get("task_valid")
    if task_valid is None:  # legacy sources were fully supervised; occupancy was optional in the old path
        task_valid = {task: True for task in ("planning", "det", "map", "motion", "geometry")}
        task_valid["occupancy"] = bool(profile.occ_classes and targets.get("occ_path"))
    else:
        from unidrivevla.data.sample import TASKS

        if (not isinstance(task_valid, dict) or set(task_valid) != set(TASKS)
                or any(not isinstance(v, (bool, np.bool_)) for v in task_valid.values())):
            raise ValueError(f"task_valid must provide bools for exactly {TASKS}")
        task_valid = {task: bool(value) for task, value in task_valid.items()}
        if task_valid["motion"] and not task_valid["det"]:
            raise ValueError("motion supervision requires compatible detection targets")
    required = {
        "planning": ("trajectory", "trajectory_mask"),
        "det": ("boxes", "box_velocity", "box_names"),
        "map": ("map",),
        "motion": ("boxes", "box_velocity", "box_names", "agent_future", "agent_future_mask"),
    }
    for task, keys in required.items():
        missing = [key for key in keys if key not in targets]
        if task_valid[task] and missing:
            raise ValueError(f"task_valid[{task!r}] is true but targets lack {missing}")
    if task_valid["occupancy"] and (not profile.occ_classes or not targets.get("occ_path")):
        raise ValueError("occupancy supervision is valid but the profile or occupancy label path is missing")
    horizon = profile.horizon
    trajectory = np.zeros((horizon, 3), np.float32)
    mask = np.zeros((horizon,), np.float32)
    given = np.asarray(targets.get("trajectory", []), np.float32).reshape(-1, 3)[:horizon]
    trajectory[: len(given)] = given
    mask[: len(given)] = np.asarray(targets.get("trajectory_mask", []), np.float32)[:horizon]
    box_source = targets if task_valid["det"] else {}
    labels = {
        "gt_ego_fut_trajs": torch.from_numpy(np.diff(trajectory[:, :2], axis=0, prepend=np.zeros((1, 2), np.float32))),
        "gt_ego_fut_masks": torch.from_numpy(mask),
        "gt_task_valid": task_valid,
        **box_targets(box_source, profile),
    }
    (map_pts, map_mask), map_labels = map_targets(targets.get("map", []), profile)
    labels.update(gt_map_pts=torch.from_numpy(map_pts), gt_map_labels=torch.from_numpy(map_labels),
                  map_mask=torch.from_numpy(map_mask))
    if "ego_status_mask" in targets:  # per-value weight of the ego status loss, from sources that have one
        labels["ego_status_mask"] = torch.from_numpy(np.asarray(targets["ego_status_mask"], np.float32))
    if profile.occ_classes:
        labels["gt_occ_dense"] = (torch.from_numpy(load_occ(
            targets["occ_path"], targets.get("occ_format") or profile.occ_format))
            if task_valid["occupancy"] else None)
    return out, labels


# ----------------------------------------------------------------------------- planning head configuration

def keypoints_box(size_clamp=None):
    cfg = dict(type="SparseBox3DKeyPointsGenerator", num_learnable_pts=6,
               fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]])
    if size_clamp is not None:
        cfg["size_clamp"] = [float(v) for v in size_clamp]
    return cfg


def ego_keypoints(profile: Profile) -> dict:
    if profile.ego_keypoints:
        cfg = dict(type="SparseBox3DKeyPointsGenerator", **profile.ego_keypoints)
    else:
        return keypoints_box(profile.box_size_clamp)
    if profile.box_size_clamp is not None:
        cfg["size_clamp"] = [float(v) for v in profile.box_size_clamp]
    return cfg


def deformable(num_cams: int, keypoints: dict, keypoint_from_anchor: bool = False) -> dict:
    from unidrivevla.models.models.blocks import DAF
    cfg = dict(type="DeformableFeatureAggregation", embed_dims=256, num_groups=8, num_levels=4, num_cams=num_cams,
               attn_drop=0.15, use_deformable_func=DAF is not None, use_camera_embed=True, residual_mode="cat",
               kps_generator=keypoints)
    if keypoint_from_anchor:
        cfg["keypoint_from_anchor"] = True
    return cfg


def head_kwargs(profile: Profile, vlm_path: str, vae_path: Optional[str], train_vlm: bool, with_motion: bool,
                attn_implementation: str = "flex", gradient_checkpointing: bool = True,
                cache_frozen_prefix: bool = False, *, vlm_variant: str = "2b",
                visual_lora=None, visual_wasp=None, llm_wasp=None, visual_projector_wasp=None,
                canbus_sft_prompt: bool = False, coc_language_prompt: bool = False,
                coc_jsonl_path: Optional[str] = None, coc_language_loss_weight: Optional[float] = None,
                coc_planning_loss_weight: float = 0.0) -> dict:
    if coc_planning_loss_weight:
        raise ValueError("coc_planning_loss_weight must be 0; the release recipes add no CoC planning loss")
    embed_dims, groups, drop, num_decoder, num_single = 256, 8, 0.1, 6, 1
    single = ["concat", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]
    temporal = ["concat", "temp_gnn", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]
    anchors = profile.anchors_dir
    num_cls, num_map = len(profile.det_classes), len(profile.map_classes)
    reverse = [profile.det_classes.index(c) for c in profile.reverse_classes if c in profile.det_classes]
    yawless = [profile.det_classes.index(c) for c in profile.yawless_classes if c in profile.det_classes]
    height, width = profile.image_hw
    attn = lambda dims: dict(type="MultiheadFlashAttention", embed_dims=dims, num_heads=groups, batch_first=True, dropout=drop)  # noqa: E731
    size_clamp = profile.box_size_clamp
    from_anchor = profile.keypoint_from_anchor
    queries = ["det", "map", "ego"] + (["motion"] if with_motion else [])  # the official stage 2 adds the motion task
    zero_velocity = dict(class_names=list(profile.det_classes), zero_velocity_classes=list(profile.zero_velocity_classes)) \
        if profile.zero_velocity_classes else {}
    decoder = dict(
        type="UnifiedPerceptionDecoder", embed_dims=embed_dims, task_select=queries, query_select=queries,
        num_stage1_layers=3, num_stage2_layers=3,
        num_single_frame_decoder=num_single, cls_threshold_to_reg=0.05, decouple_attn=True, use_vlm_in_stage2=True,
        operation_order=single * num_single + temporal * (num_decoder - num_single),
        det_instance_bank=dict(type="InstanceBank", num_anchor=900, embed_dims=embed_dims,
                               anchor=os.path.join(anchors, "det_900.npy"),
                               anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"), num_temp_instances=600,
                               confidence_decay=0.6, feat_grad=False, **zero_velocity),
        map_instance_bank=dict(type="InstanceBank", num_anchor=100, embed_dims=embed_dims,
                               anchor=os.path.join(anchors, "map_100.npy"),
                               anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"), num_temp_instances=0,
                               confidence_decay=0.6, feat_grad=True),
        ego_instance_bank=dict(type="EgoInstanceBank", embed_dims=embed_dims, ego_box=list(profile.ego_box),
                               num_temp_instances=1, feature_map_scale=(height // 16, width // 16)),
        det_anchor_encoder=dict(type="SparseBox3DEncoder", vel_dims=3, embed_dims=[128, 32, 32, 64], mode="cat",
                                output_fc=False, in_loops=1, out_loops=4),
        map_anchor_encoder=dict(type="SparsePoint3DEncoder", embed_dims=embed_dims, num_sample=profile.map_points),
        graph_model=dict(type="SeparateAttention", query_select=queries, separate_list=[["det"], ["map"]],
                         decouple_list=[True, False], attn=[attn(embed_dims * 2), attn(embed_dims)]),
        temp_graph_model=dict(type="TemporalSeparateAttention", query_select=queries,
                              query_list=[["det"], ["map"], ["ego"]], key_list=[["det"], ["map"], ["det", "map"]],
                              decouple_list=[True, False, False],
                              attn=[attn(embed_dims * 2), attn(embed_dims), attn(embed_dims)]),
        inter_graph_model=dict(type="InteractiveAttention", query_select=queries, query_list=[["ego"]],
                               key_list=[["det", "map"]], decouple_list=[False], attn=[attn(embed_dims)]),
        det_deformable=deformable(len(profile.cameras), keypoints_box(size_clamp), from_anchor),
        map_deformable=deformable(len(profile.cameras), dict(
            type="SparsePoint3DKeyPointsGenerator", embed_dims=embed_dims, num_sample=profile.map_points,
            num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1), ground_height=profile.ground_height), from_anchor),
        ego_deformable=deformable(len(profile.cameras), ego_keypoints(profile), from_anchor),
        ffn=dict(type="AsymmetricFFN", in_channels=embed_dims * 2, pre_norm=dict(type="LN"), embed_dims=embed_dims,
                 feedforward_channels=embed_dims * 4, num_fcs=2, ffn_drop=drop, act_cfg=dict(type="ReLU", inplace=True)),
        norm_layer=dict(type="LN", normalized_shape=embed_dims) if profile.decoder_norm else None,
        det_refine_layer=dict(type="SparseBox3DRefinementModule", embed_dims=embed_dims, num_cls=num_cls, refine_yaw=True,
                              with_quality_estimation=True),
        map_refine_layer=dict(type="SparsePoint3DRefinementModule", embed_dims=embed_dims, num_sample=profile.map_points,
                              num_cls=num_map),
        ego_refine_layer=dict(type="EgoStatusRefinementModule", embed_dims=embed_dims,
                              status_dims=len(profile.status_layout)),
        motion_refine_layer=dict(type="SparseMotionRefinementModule", embed_dims=embed_dims,
                                 fut_ts=profile.motion_steps, fut_mode=6),
        det_sampler=dict(type="SparseBox3DTarget", num_dn_groups=0, num_temp_dn_groups=0,
                         dn_noise_scale=[2.0] * 3 + [0.5] * 7, max_dn_gt=32, add_neg_dn=True, cls_weight=2.0,
                         box_weight=0.25,
                         reg_weights=[2.0] * 3 + [0.5] * 3 + [profile.motion_velocity_weight if with_motion else 0.0] * 2
                         + [0.0] * 2,
                         cls_wise_reg_weights={i: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0] for i in yawless}),
        map_sampler=dict(type="SparsePoint3DTarget",
                         assigner=dict(type="HungarianLinesAssigner", cost=dict(
                             type="MapQueriesCost", cls_cost=dict(type="FocalLossCost", weight=1.0),
                             reg_cost=dict(type="LinesL1Cost", weight=10.0, beta=0.01, permute=True),
                             **({"nan_to_num": True} if profile.nan_to_num_cost else {}))),
                         num_cls=num_map, num_sample=profile.map_points, roi_size=tuple(profile.map_roi)),
        motion_sampler=dict(type="SparseMotionTarget"),
        det_decoder=dict(type="SparseBox3DDecoder"),
        map_decoder=dict(type="SparsePoint3DDecoder"),
        loss_det_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_det_reg=dict(type="SparseBox3DLoss", loss_box=dict(type="L1Loss", loss_weight=0.25),
                          loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                          loss_yawness=dict(type="GaussianFocalLoss"), cls_allow_reverse=reverse),
        loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_map_reg=dict(type="SparseLineLoss", loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                          num_sample=profile.map_points, roi_size=tuple(profile.map_roi)),
        loss_ego_status=dict(type="L1Loss", loss_weight=1.0),
        loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2 if with_motion else 0.0),
        loss_motion_reg=dict(type="L1Loss", loss_weight=0.2 if with_motion else 0.0),
        det_reg_weights=[2.0] * 3 + [1.0] * 7,
        map_reg_weights=[1.0] * (2 * profile.map_points),
        motion_anchor=os.path.join(anchors, "motion_6.npy"),
    )
    if with_motion:
        decoder.update(motion_anchor_encoder=dict(type="SparseBox3DEncoder", vel_dims=3, embed_dims=[128, 32, 32, 64],
                                                  mode="cat", output_fc=False, in_loops=1, out_loops=4))
        if profile.name == "nuscenes" and profile.frame == "lidar":
            # Main's nuScenes stage-2 inference adds per-detection agent motion. This has no learned state; the
            # dataset/frame gate keeps B2D and legacy ego-frame checkpoints on their existing output contract.
            decoder["motion_decoder"] = dict(type="SparseMotionDecoder")
    vae = dict(
        type="VAERes3D",
        encoder_cfg=dict(type="Encoder2D", ch=64, out_ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                         attn_resolutions=(50,), dropout=0.0, resamp_with_conv=True, in_channels=128, resolution=200,
                         z_channels=128, double_z=False),
        decoder_cfg=dict(type="Decoder3D", ch=64, out_ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                         attn_resolutions=(50,), dropout=0.0, resamp_with_conv=True, in_channels=128, resolution=200,
                         z_channels=64, give_pre_end=False),
        num_classes=profile.occ_classes or 9, expansion=8, vqvae_cfg=None,
    )
    return dict(
        pretrained_path=vlm_path, action_dim=2, action_horizon=profile.horizon, dtype="bfloat16", train_vlm=train_vlm,
        attn_implementation=attn_implementation, inference_attn_impl=profile.inference_attn_impl,
        loss_planning=dict(type="FlowPlanningLoss", use_min_snr_loss=True, min_snr_gamma=5.0, loss_weight=1.0,
                           hybrid_loss_weight=0.5, detach_window_size=3,
                           delta_mu=list(profile.delta_mean), delta_std=list(profile.delta_std)),
        with_depth_supervision=False, occworld_vae_config=vae, occworld_vae_path=vae_path or "",
        # raw: the vision tower's pre-merger hidden states after its deepstack blocks and last block, one DFA level each
        feat_grad=True, feature_source="raw", unified_decoder_cfg=decoder, driving_deepstack=True,
        vlm_fusion_cfg=dict(type="direct"), feature_fusion_cfg=dict(type="none"),
        view_tokens=list(profile.view_tokens), command_prompts=list(profile.command_prompts),
        prefix_max_len=profile.prefix_max_len, gradient_checkpointing=gradient_checkpointing,
        cache_frozen_prefix=cache_frozen_prefix,
        delta_mean=tuple(profile.delta_mean), delta_std=tuple(profile.delta_std),
        coordinate_frame=profile.frame, prompt_style=profile.prompt_style,
        enable_knowledge_insulation=profile.enable_knowledge_insulation,
        ar_loss_weight=profile.ar_loss_weight if coc_language_loss_weight is None else coc_language_loss_weight,
        vlm_variant=vlm_variant,
        visual_lora_cfg=visual_lora, visual_wasp_cfg=visual_wasp, peft_cfg=llm_wasp,
        visual_projector_wasp_cfg=visual_projector_wasp,
        canbus_sft_prompt=canbus_sft_prompt, coc_language_prompt=coc_language_prompt,
        coc_jsonl_path=coc_jsonl_path or "",
        use_hist_traj=profile.use_hist_traj, use_occ_tokens=profile.use_occ_tokens,
        detach_plan_geometry=profile.detach_plan_geometry, summed_planning_loss=profile.summed_planning_loss,
        official_command_text=profile.official_command_text,
        official_attention_mask=profile.official_attention_mask,
        cast_inference_mask=profile.cast_inference_mask, processor_use_fast=profile.processor_use_fast,
        cast_stream_dtype=profile.cast_stream_dtype,
    )


def register_modules():
    """Imports every module the head's registries build from."""
    import unidrivevla.models.losses.collision_loss  # noqa: F401
    import unidrivevla.models.losses.flow_planning_loss  # noqa: F401
    import unidrivevla.models.losses.plan_map_loss  # noqa: F401
    import unidrivevla.models.models.attention  # noqa: F401
    import unidrivevla.models.models.blocks  # noqa: F401
    import unidrivevla.models.models.detection3d.decoder  # noqa: F401
    import unidrivevla.models.models.detection3d.detection3d_blocks  # noqa: F401
    import unidrivevla.models.models.detection3d.losses  # noqa: F401
    import unidrivevla.models.models.detection3d.target  # noqa: F401
    import unidrivevla.models.models.ego.blocks  # noqa: F401
    import unidrivevla.models.models.ego.instance_bank  # noqa: F401
    import unidrivevla.models.models.instance_bank  # noqa: F401
    import unidrivevla.models.models.map.decoder  # noqa: F401
    import unidrivevla.models.models.map.loss  # noqa: F401
    import unidrivevla.models.models.map.map_blocks  # noqa: F401
    import unidrivevla.models.models.map.match_cost  # noqa: F401
    import unidrivevla.models.models.map.target  # noqa: F401
    import unidrivevla.models.models.motion.blocks  # noqa: F401
    import unidrivevla.models.models.motion.decoder  # noqa: F401
    import unidrivevla.models.models.motion.target  # noqa: F401
    import unidrivevla.models.models.separate_attn  # noqa: F401
    import unidrivevla.models.unidrivevla.vae  # noqa: F401
    from unidrivevla.models.unidrivevla.dense_heads.qwenvl3_vla_planning_head import QwenVL3APlanningHead

    return QwenVL3APlanningHead


# ----------------------------------------------------------------------------- policy

UNPAD = (("gt_bboxes_3d", "box_mask"), ("gt_labels_3d", "box_mask"), ("gt_agent_fut_trajs", "box_mask"),
         ("gt_agent_fut_masks", "box_mask"), ("gt_map_pts", "map_mask"), ("gt_map_labels", "map_mask"))


def _motion_only_key(name: str) -> bool:
    """Whether a state key belongs only to the stage-2 motion branch."""
    parts = str(name).split(".")
    motion_parts = {"motion_proj_up", "motion_proj_down", "motion_instance_bank", "motion_anchor_encoder",
                    "motion_deformable", "motion_refine", "motion_sampler", "motion_decoder", "motion_anchor",
                    "loss_motion_cls", "loss_motion_reg"}
    # `motion.*` is also used by the tiny policy fixture that aliases this initializer.
    return any(part in motion_parts for part in parts) or parts[-2:-1] == ["motion"]


def _checkpoint_motion_value(checkpoint, state, motion_keys) -> bool:
    """Resolve stage from safe checkpoint metadata or the saved motion tensors themselves."""
    values = []
    if isinstance(checkpoint, dict):
        config = checkpoint.get("agent_config")
        if config is not None:
            try:
                value = config.get("with_motion")
            except AttributeError:
                value = None
            if value is not None:
                if not isinstance(value, bool):
                    raise TypeError("checkpoint agent_config.with_motion must be boolean")
                values.append(("agent_config.with_motion", value))

    for key in ("_unidrivevla_with_motion", "_multidomain_with_motion"):
        if key not in state:
            continue
        value = state[key]
        if not torch.is_tensor(value) or value.numel() != 1:
            raise TypeError(f"checkpoint {key} must be a scalar tensor")
        values.append((key, bool(value.item())))

    has_motion_tensors = bool(motion_keys)
    if values and any(value != values[0][1] for _, value in values[1:]):
        raise ValueError("checkpoint motion metadata fields disagree")
    source_motion = values[0][1] if values else has_motion_tensors
    if values and source_motion != has_motion_tensors:
        raise ValueError("checkpoint motion metadata disagrees with its motion tensors")
    return source_motion


def poses_from_xy(xy: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) waypoints -> (B, T, 3) poses, heading from the step direction."""
    delta = torch.diff(xy, dim=1, prepend=torch.zeros_like(xy[:, :1]))
    return torch.cat([xy, torch.atan2(delta[..., 1], delta[..., 0]).unsqueeze(-1)], dim=-1)


def _rotation_only(value, name="native_to_ego") -> np.ndarray:
    """Validate a proper 4x4 axis rotation; frame conversions keep the LiDAR origin fixed."""
    try:
        value = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 rotation-only matrix") from exc
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(value[:3, 3], 0, atol=1e-5)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3)
            or np.linalg.det(value[:3, :3]) < 0.99):
        raise ValueError(f"{name} must be a proper 4x4 rotation with zero translation")
    return value


def _source_frame_rotation(features, profile):
    """Validate a source's declared head frame and return its LiDAR-to-EGO axes rotation."""
    frame = getattr(profile, "frame", "ego")
    if frame not in ("ego", "lidar"):
        raise ValueError(f"profile.frame must be 'ego' or 'lidar', got {frame!r}")
    declared = features.get("coordinate_frame")
    if declared is None:
        if "native_to_ego" in features:
            raise ValueError("features.native_to_ego requires an explicit features.coordinate_frame")
        if frame == "lidar":
            raise ValueError("LiDAR-frame profiles require features.coordinate_frame and features.native_to_ego")
        return np.eye(4, dtype=np.float64)
    if not isinstance(declared, str):
        raise ValueError("per-sample features.coordinate_frame must be a string")
    if declared != frame:
        raise ValueError(f"source coordinate_frame {declared!r} does not match profile.frame {frame!r}")
    value = features.get("native_to_ego")
    if value is None:
        raise ValueError("declared source coordinate_frame requires features.native_to_ego")
    transform = _rotation_only(value)
    if frame == "ego" and not np.allclose(transform, np.eye(4), atol=1e-5):
        raise ValueError("ego-frame samples must use identity features.native_to_ego")
    return transform


def _batch_frame_rotation(features, profile, batch_size, device, dtype):
    """Read validated frame metadata from a collated policy batch."""
    frame = getattr(profile, "frame", "ego")
    declared = features.get("coordinate_frame")
    if declared is None:
        if "native_to_ego" in features:
            raise ValueError("native_to_ego policy input requires coordinate_frame")
        if frame == "lidar":
            raise ValueError("LiDAR-frame policy input lacks coordinate_frame/native_to_ego metadata")
        return torch.eye(4, device=device, dtype=dtype).expand(batch_size, -1, -1)
    values = declared if isinstance(declared, (list, tuple)) else [declared] * batch_size
    if len(values) != batch_size or any(value != frame for value in values):
        raise ValueError(f"batch coordinate_frame values must all match profile.frame={frame!r}")
    value = features.get("native_to_ego")
    if value is None:
        raise ValueError("declared coordinate_frame input lacks native_to_ego")
    matrix = value if torch.is_tensor(value) else torch.as_tensor(value)
    if matrix.shape == (4, 4):
        matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)
    if matrix.shape != (batch_size, 4, 4):
        raise ValueError(f"native_to_ego must have shape ({batch_size}, 4, 4), got {tuple(matrix.shape)}")
    matrix = matrix.to(device=device, dtype=dtype)
    # The source transform validated every matrix in its CPU worker. Keep cheap shape/frame checks here and
    # avoid a GPU synchronization on every policy call.
    if frame == "ego" and not torch.equal(matrix, torch.eye(4, device=device, dtype=dtype).expand_as(matrix)):
        raise ValueError("ego-frame inputs must use identity native_to_ego")
    return matrix


def _checkpoint_profile_config(checkpoint, profile_name=None):
    """Return the saved profile mapping for a single-profile or multi-profile agent checkpoint."""
    config = checkpoint.get("agent_config") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict):
        return None
    profiles = config.get("profiles")
    if isinstance(profiles, dict) and profile_name is not None:
        value = profiles.get(profile_name)
        return value if isinstance(value, dict) else None
    value = config.get("profile")
    if isinstance(value, dict):
        return value
    return None


def _check_checkpoint_frames(checkpoint, profiles, require_all=False, checkpoint_frame=None):
    """Reject checkpoint/profile coordinate mismatches, requiring provenance for legacy native checkpoints."""
    if isinstance(checkpoint_frame, dict):
        if any(value not in ("ego", "lidar") for value in checkpoint_frame.values()):
            raise ValueError("checkpoint_frame mappings may contain only 'ego' or 'lidar' values")
    elif checkpoint_frame not in (None, "ego", "lidar"):
        raise ValueError("checkpoint_frame must be null, 'ego', 'lidar', or a profile-to-frame mapping")
    if not profiles or not all(isinstance(profile, Profile) for profile in profiles.values()):
        return  # Keep small initializer fixtures and non-UniDrive modules independent of profile metadata.
    if isinstance(checkpoint_frame, dict):
        unknown = set(checkpoint_frame) - set(profiles)
        if unknown:
            raise ValueError(f"checkpoint_frame mapping contains unknown profile names: {sorted(unknown)}")
    for name, profile in profiles.items():
        explicit_frame = checkpoint_frame.get(name) if isinstance(checkpoint_frame, dict) else checkpoint_frame
        saved = _checkpoint_profile_config(checkpoint, name)
        if saved is None and len(profiles) == 1:
            saved = _checkpoint_profile_config(checkpoint)
        saved_frame = None
        if saved is not None:
            saved_name = saved.get("name")
            if saved_name is not None and str(saved_name) != str(profile.name):
                raise ValueError(f"checkpoint profile {saved_name!r} does not match target profile {profile.name!r}")
            saved_frame = saved.get("frame")
        if saved_frame is not None:
            if explicit_frame is not None and explicit_frame != saved_frame:
                raise ValueError("explicit checkpoint_frame disagrees with checkpoint profile metadata")
            if saved_frame != profile.frame:
                raise ValueError(f"checkpoint profile {name!r} frame {saved_frame!r} does not match target "
                                 f"frame {profile.frame!r}")
            continue
        source_frame = explicit_frame
        if source_frame is None:
            if profile.frame == "lidar":
                raise ValueError(f"checkpoint has no coordinate-frame provenance for native profile {name!r}; "
                                 "set agent.checkpoint_frame explicitly for a verified legacy checkpoint")
            if require_all:
                raise ValueError(f"checkpoint has no profile metadata for {name!r}")
            continue
        if source_frame != profile.frame:
            raise ValueError(f"explicit checkpoint_frame {source_frame!r} does not match target frame "
                             f"{profile.frame!r} for profile {name!r}")


class UniDriveVLAPolicy(nn.Module):
    def __init__(self, profile, vlm_path: str, vae_path: Optional[str] = None, lr: float = 2e-4,
                 weight_decay: float = 1e-7, warmup_steps: int = 125, warmup_start: float = 1 / 3,
                 min_lr_ratio: float = 1e-3, vlm_lr_mult: float = 0.5, train_vlm: bool = False,
                 lora: Optional[Dict] = None, with_motion: bool = False, checkpoint_path: Optional[str] = None,
                 attn_implementation: str = "flex", gradient_checkpointing: bool = True,
                 cache_frozen_prefix: bool = False, shared_runtime=None,
                 checkpoint_frame: Optional[str] = None, vlm_variant: str = "2b",
                 visual_lora: Optional[Dict] = None, visual_wasp: Optional[Dict] = None,
                 llm_wasp: Optional[Dict] = None, visual_projector_wasp: Optional[Dict] = None,
                 canbus_sft_prompt: bool = False, coc_language_prompt: bool = False,
                 coc_jsonl_path: Optional[str] = None,
                 coc_language_loss_weight: Optional[float] = None, coc_planning_loss_weight: float = 0.0):
        super().__init__()
        self.profile = profile if isinstance(profile, Profile) else Profile.from_config(profile)
        interval = float(self.profile.pose_interval)
        self.trajectory_sampling = SimpleNamespace(num_poses=self.profile.horizon, interval_length=interval,
                                                   time_horizon=self.profile.horizon * interval)
        self.lr, self.weight_decay, self.vlm_lr_mult = lr, weight_decay, vlm_lr_mult
        self.warmup_steps, self.warmup_start, self.min_lr_ratio = warmup_steps, warmup_start, min_lr_ratio
        self.checkpoint_path = checkpoint_path
        if checkpoint_frame not in (None, "ego", "lidar"):
            raise ValueError("checkpoint_frame must be null, 'ego', or 'lidar'")
        self.checkpoint_frame = checkpoint_frame
        self.with_motion = bool(with_motion)
        head_cls = register_modules()
        self.head = head_cls(**head_kwargs(
            self.profile, vlm_path, vae_path, train_vlm, with_motion, attn_implementation,
            gradient_checkpointing, cache_frozen_prefix, vlm_variant=vlm_variant,
            visual_lora=visual_lora, visual_wasp=visual_wasp, llm_wasp=llm_wasp,
            visual_projector_wasp=visual_projector_wasp, canbus_sft_prompt=canbus_sft_prompt,
            coc_language_prompt=coc_language_prompt, coc_jsonl_path=coc_jsonl_path,
            coc_language_loss_weight=coc_language_loss_weight,
            coc_planning_loss_weight=coc_planning_loss_weight), lora_cfg=None if llm_wasp else lora,
                             shared_runtime=shared_runtime)

    def worker_transform(self, image_augment=None, train: bool = False):
        return functools.partial(prepare_sample, profile=self.profile, image_augment=image_augment, train=train)

    def initialize(self, strict: bool = False) -> None:
        if not self.checkpoint_path:
            if strict:
                raise ValueError("inference requires a complete checkpoint_path")
            return
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError(f"{self.checkpoint_path} does not contain a state dict")
        profile = getattr(self, "profile", None)
        if isinstance(profile, Profile):
            _check_checkpoint_frames(checkpoint, {profile.name: profile}, checkpoint_frame=self.checkpoint_frame)
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        if strict:
            self.load_state_dict(state, strict=True)
            return
        own = self.state_dict()
        source_motion_keys = {key for key in state if _motion_only_key(key)}
        source_motion = _checkpoint_motion_value(checkpoint, state, source_motion_keys)
        own_motion_keys = {key for key in own if _motion_only_key(key)}
        target_motion = bool(getattr(self, "with_motion", own_motion_keys))
        if source_motion and not target_motion:
            raise ValueError("cannot initialize a motion-enabled checkpoint into a motion-disabled policy")
        transition = not source_motion and target_motion
        if any(key in state and not torch.is_tensor(state[key]) for key in own):
            raise TypeError("checkpoint model state contains a non-tensor model value")
        loaded = {}
        for key, value in state.items():
            if key in ("_unidrivevla_with_motion", "_multidomain_with_motion"):
                continue
            if key not in own:
                continue
            if not torch.is_tensor(value):
                raise TypeError(f"checkpoint tensor {key!r} is not a tensor")
            if value.shape != own[key].shape:
                raise RuntimeError(f"checkpoint tensor {key} has shape {tuple(value.shape)}, expected "
                                   f"{tuple(own[key].shape)}")
            loaded[key] = value
        if not loaded:
            raise KeyError(f"{self.checkpoint_path} holds none of the model's tensors")
        missing = set(own) - set(loaded)
        allowed_missing = own_motion_keys if transition else set()
        if missing != allowed_missing:
            bad_missing = sorted(missing - allowed_missing)
            unexpected_motion = sorted(source_motion_keys - set(own))
            raise KeyError(f"checkpoint is incomplete for this policy; missing required model tensors "
                           f"{bad_missing[:5]}, unexpected motion tensors {unexpected_motion[:5]}")
        if transition and not own_motion_keys:
            raise ValueError("stage-1 to stage-2 initialization requested but the policy has no motion tensors")
        if transition:
            self.load_state_dict(loaded, strict=False)
            logger.info("initialized stage 2 from %s; left %d new motion tensors fresh", self.checkpoint_path,
                        len(own_motion_keys))
        else:
            self.load_state_dict(loaded, strict=True)
            logger.info("loaded complete model state (%d tensors) from %s", len(loaded), self.checkpoint_path)

    def training_runtime_state_dict(self):
        modules, covered = {}, []
        for name, module in self.head.named_modules():
            if any(name.startswith(parent + ".") for parent in covered):
                continue
            save = getattr(module, "training_runtime_state_dict", None)
            if callable(save):
                modules[name] = save()
                if name:
                    covered.append(name)
        samplers = {}
        decoder = getattr(self.head, "unified_decoder", None)
        for name in ("det_sampler", "map_sampler"):
            sampler = getattr(decoder, name, None)
            save = getattr(sampler, "training_runtime_state_dict", None)
            if callable(save):
                samplers[name] = save()
        return {"modules": modules, "samplers": samplers}

    def load_training_runtime_state_dict(self, state, device=None):
        modules = dict(self.head.named_modules())
        for name, module_state in state.get("modules", {}).items():
            module = modules.get(name)
            restore = getattr(module, "load_training_runtime_state_dict", None)
            if not callable(restore):
                raise RuntimeError(f"model runtime state references unsupported module {name!r}")
            restore(module_state, device=device)
        decoder = getattr(self.head, "unified_decoder", None)
        for name, sampler_state in state.get("samplers", {}).items():
            sampler = getattr(decoder, name, None)
            restore = getattr(sampler, "load_training_runtime_state_dict", None)
            if not callable(restore):
                raise RuntimeError(f"model runtime state references unsupported sampler {name!r}")
            restore(sampler_state, device=device)

    def forward(self, features, targets=None, tokens=None):
        task = features.get("task", "planning")
        if not self.training:
            if task != "planning":
                raise ValueError(f"policy task {task!r} is only supported during training")
            pred = self.head.forward_test(**{k: v for k, v in features.items()
                                             if k not in ("task", "coordinate_frame", "native_to_ego")})
            xy_head = pred["traj"].float()
            matrix = _batch_frame_rotation(features, self.profile, xy_head.shape[0], xy_head.device, xy_head.dtype)
            xyz_head = torch.cat((xy_head, torch.zeros_like(xy_head[..., :1])), dim=-1)
            xy_ego = torch.einsum("bij,btj->bti", matrix[:, :3, :3], xyz_head)[..., :2]
            result = {"pred_traj": poses_from_xy(xy_ego)}
            if self.profile.frame == "lidar":
                result["pred_traj_head"] = poses_from_xy(xy_head)
            det = pred.get("det")
            if det is not None and any("trajs_3d" in sample and "trajs_score" in sample for sample in det):
                result["det"] = det
                result["det_coordinate_frame"] = self.profile.frame
            return result
        if task == "vqa":
            if not getattr(self.head, "train_vlm", False):
                raise ValueError("VQA training requires agent.head.train_vlm=true")
            vlm = getattr(getattr(self.head, "qwen3_vl_with_expert", None), "qwen3_vl", None)
            trainable_vlm = (any(param.requires_grad for param in vlm.parameters()) if vlm is not None else
                             any(param.requires_grad for name, param in self.head.named_parameters()
                                 if "qwen3_vl" in name))
            if not trainable_vlm:
                raise ValueError("VQA training requires at least one trainable Qwen VLM parameter")
            ar_batch = features.get("ar_batch")
            if ar_batch is None:
                raise ValueError("task='vqa' requires features['ar_batch']")
            output = self.head.forward_ar_batch(ar_batch)
            loss = output["loss_ar"]
            return SimpleNamespace(loss=loss, loss_ar=loss.detach(),
                                   loss_vlm_raw=output["loss_vlm_raw"].detach())
        if task != "planning":
            raise ValueError(f"unknown policy task {task!r}")
        batch = {**features, **(targets or {})}
        batch.pop("task", None)
        batch.pop("coordinate_frame", None)
        batch.pop("native_to_ego", None)
        for key, mask_key in UNPAD:
            data, mask = batch[key], batch[mask_key].bool()
            batch[key] = [data[i][mask[i]] for i in range(data.shape[0])]
        batch.pop("box_mask")
        batch.pop("map_mask")
        if tokens is not None:
            batch["sample_tokens"] = list(tokens)
        losses = {k: v for k, v in self.head.forward_train(**batch)["losses"].items() if torch.is_tensor(v)}
        total = sum(torch.nan_to_num(v) for v in losses.values())
        return SimpleNamespace(loss=total, **{k: v.detach() for k, v in losses.items() if v.numel() == 1})

    def compute_loss(self, features, targets, predictions) -> torch.Tensor:
        """Validation: mean L2 of the planned waypoints over the valid steps."""
        gt = torch.cumsum(targets["gt_ego_fut_trajs"].float(), dim=1)
        mask = targets["gt_ego_fut_masks"].float()
        prediction = predictions.get("pred_traj_head", predictions["pred_traj"])
        l2 = torch.linalg.norm(prediction[..., :2] - gt, dim=-1)
        return (l2 * mask).sum() / mask.sum().clamp(min=1.0)

    def get_optimizers(self, total_steps: int = 0, steps_per_epoch: int = 0, micro_steps: int = 0,
                       grad_accum: int = 1):
        from unidrivevla.train.optim import MasterAdamW

        vlm, rest = [], []  # the official paramwise_cfg: the VLM's parameters at lr * vlm_lr_mult
        for name, param in self.named_parameters():
            if param.requires_grad:
                (vlm if name.startswith("head.qwen3_vl_with_expert.qwen3_vl.") else rest).append(param)
        groups = [{"params": rest}] + ([{"params": vlm, "lr": self.lr * self.vlm_lr_mult}] if vlm else [])
        optimizer = MasterAdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        if total_steps <= 0 and micro_steps <= 0:
            return optimizer
        micro_schedule = bool(getattr(getattr(self, "profile", None), "lr_microstep", False))
        if micro_schedule and micro_steps <= 0:
            raise ValueError("profile.lr_microstep requires a positive micro_steps value from Trainer")
        if grad_accum < 1:
            raise ValueError("grad_accum must be positive")
        schedule_steps = micro_steps if micro_schedule else total_steps
        if schedule_steps <= 0:
            return optimizer
        warmup_steps = self.warmup_steps * grad_accum if micro_schedule else self.warmup_steps

        def factor(step: int) -> float:
            """mmcv's CosineAnnealing with linear warmup, as the official configs: the cosine runs over every step,
            the first warmup_steps scale it up from warmup_start."""
            progress = min(1.0, step / schedule_steps)
            value = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if step < warmup_steps:
                value *= 1.0 - (1.0 - step / warmup_steps) * (1.0 - self.warmup_start)
            return value

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "micro" if micro_schedule else "step"}}
