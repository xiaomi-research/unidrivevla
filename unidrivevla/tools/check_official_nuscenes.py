"""python unidrivevla/tools/check_official_nuscenes.py [--frame lidar|ego] [--info-dir DIR] [--path-root DIR]
        [--data-root DIR] [--official DIR] [--version v1.0-mini] [--split train|val|both] [--limit N]

The official train pipeline (PhotoMetricDistortion off, no flip / 2D or 3D rotation; resize_lim is the constant 0.6)
and the test pipeline, against NuScenesSource + prepare_sample on the same infos. Official tensors stay in the LiDAR
frame and the pkl camera order. By default this compares the raw main LiDAR tensors; --frame ego retains the legacy
canonical-axis comparison. Cameras are reordered with the planning head's [0, 2, 1, 4, 5, 3]. It also checks the
GroupInBatchSampler parts (sequences_split_num=2) and
the OccWorld label path (a synthetic file: the servers have no Occ3D labels). Prints each head input's max abs
diff. Exits 1 on a mismatch.

gt_depth is not compared: with_depth_supervision is false, and the train pipeline's lidar projection is unused.
"""

import argparse
import importlib.util
import os
import pickle
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from unidrivevla.adapters.unidrivevla import Profile, load_occ, prepare_sample  # noqa: E402
from unidrivevla.data.nuscenes import NuScenesSource  # noqa: E402
from unidrivevla.data.registry import SequenceBatchSampler  # noqa: E402

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
CAM_PKL = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# qwenvl3_vla_planning_head permutes the pkl order into the profile / view-token order
CAM_PERM = [0, 2, 1, 4, 5, 3]
PROFILE_CAMS = ["front", "front_left", "front_right", "back_left", "back_right", "back"]
FINAL_HW = (544, 960)  # official input_shape (960, 544) stored as (W, H); final_dim is (H, W)

failures = []


def check(ok, message):
    print(("ok   " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)


def install_stubs():
    """The official pipeline modules only need mmdet's registries and to_tensor. mmcv is used if it imports."""
    reg = types.ModuleType("mmdet")
    datasets = types.ModuleType("mmdet.datasets")
    builder = types.ModuleType("mmdet.datasets.builder")
    pipes = types.ModuleType("mmdet.datasets.pipelines")

    class Registry:
        def __init__(self):
            self.mods = {}

        def register_module(self, cls=None, force=False, name=None):
            def deco(c):
                self.mods[name or c.__name__] = c
                return c
            return deco(cls) if cls is not None else deco

    builder.PIPELINES = Registry()
    datasets.DATASETS = Registry()
    datasets.builder = builder

    def to_tensor(data):
        import torch
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return torch.as_tensor(data)

    pipes.to_tensor = to_tensor
    pipes.Compose = None
    datasets.pipelines = pipes
    for name, mod in (("mmdet", reg), ("mmdet.datasets", datasets), ("mmdet.datasets.builder", builder),
                      ("mmdet.datasets.pipelines", pipes)):
        sys.modules[name] = mod
    if not hasattr(np, "int"):  # numpy 2 dropped the alias the official get_ann_info passes to np.array
        np.int = np.int64  # type: ignore[attr-defined]

    try:
        import prettytable  # noqa: F401
    except ImportError:
        pretty = types.ModuleType("prettytable")

        class PrettyTable:
            def __init__(self, *args, **kwargs):
                pass

            def add_row(self, *args, **kwargs):
                pass

        pretty.PrettyTable = PrettyTable
        sys.modules["prettytable"] = pretty

    try:
        import mmcv  # noqa: F401
        from mmcv.parallel import DataContainer  # noqa: F401
        from mmcv.utils import print_log  # noqa: F401
    except ImportError:
        import cv2
        mmcv = types.ModuleType("mmcv")
        mmcv.imread = lambda path, flag="color", **kwargs: cv2.imread(str(path), cv2.IMREAD_COLOR)

        class DataContainer:
            def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=None):
                self.data = data

        parallel = types.ModuleType("mmcv.parallel")
        parallel.DataContainer = DataContainer
        utils = types.ModuleType("mmcv.utils")
        utils.print_log = lambda *args, **kwargs: None
        mmcv.parallel, mmcv.utils = parallel, utils
        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.parallel"] = parallel
        sys.modules["mmcv.utils"] = utils

    try:
        import nuscenes  # noqa: F401
    except ImportError:
        nus = types.ModuleType("nuscenes")

        class NuScenes:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("nuscenes devkit is not installed; scene names come from scene.json")

        nus.NuScenes = NuScenes
        utils = types.ModuleType("nuscenes.utils")
        data_classes = types.ModuleType("nuscenes.utils.data_classes")
        data_classes.Box = type("Box", (), {})
        det = types.ModuleType("nuscenes.eval.detection.config")
        det.config_factory = lambda *args, **kwargs: None
        common = types.ModuleType("nuscenes.eval.common.config")
        common.config_factory = lambda *args, **kwargs: None
        for name, mod in (("nuscenes", nus), ("nuscenes.utils", utils),
                          ("nuscenes.utils.data_classes", data_classes),
                          ("nuscenes.eval.detection.config", det),
                          ("nuscenes.eval.common.config", common)):
            sys.modules[name] = mod


def load_py(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def load_official(root: Path):
    install_stubs()
    plug = root / "projects" / "mmdet3d_plugin"
    namespace("projects", root / "projects")
    namespace("projects.mmdet3d_plugin", plug)
    namespace("projects.mmdet3d_plugin.core", plug / "core")
    namespace("projects.mmdet3d_plugin.datasets", plug / "datasets")
    load_py("projects.mmdet3d_plugin.core.box3d", plug / "core" / "box3d.py")
    load_py("projects.mmdet3d_plugin.datasets.utils", plug / "datasets" / "utils.py")
    pipes = plug / "datasets" / "pipelines"
    modules = [load_py(f"official_pipe_{name}", pipes / f"{name}.py")
               for name in ("loading", "augment", "transform", "vectorize")]
    dataset = load_py("projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset",
                      plug / "datasets" / "nuscenes_3d_dataset.py")
    registry = sys.modules["mmdet.datasets.builder"].PIPELINES.mods
    return dataset, registry, modules


def abs_path(path, root):
    path = str(path)
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))


def augmentation(train):
    """get_augmentation with resize_lim (0.6, 0.6), bot_pct_lim (0, 0), flip and both rotations forced off."""
    height, width, final_h, final_w = 900, 1600, *FINAL_HW
    resize = 0.6 if train else max(final_h / height, final_w / width)
    new_w, new_h = int(width * resize), int(height * resize)
    crop_h = new_h - final_h
    crop_w = 0 if train else int(max(0, new_w - final_w) / 2)
    return {"resize": resize, "resize_dims": (new_w, new_h),
            "crop": (crop_w, crop_h, crop_w + final_w, crop_h + final_h),
            "flip": False, "rotate": 0, "rotate_3d": 0}


def make_pipeline(registry, train):
    load, resize, hist = (registry[n] for n in
                          ("LoadMultiViewImageFromFiles", "ResizeCropFlipImage", "LoadAnnotations3D_E2E"))
    steps = [load(to_float32=True), resize(), hist(with_hist_traj=True)]
    if train:
        steps += [registry["BBoxRotation"](),
                  registry["CircleObjectRangeFilter"](class_dist_thred=[55] * len(CLASSES)),
                  registry["InstanceNameFilter"](classes=CLASSES),
                  registry["VectorizeMap"](roi_size=(30, 60), simplify=False, normalize=False, sample_num=20, permute=True)]
    steps.append(registry["NuScenesSparse4DAdaptor"]())
    return steps


def run_pipeline(steps, data):
    for step in steps:
        data = step(data)
    return data


def as_np(value):
    import torch
    if value is None:
        return None
    if type(value).__name__ == "DataContainer":
        value = value.data
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def rotation_of(info):
    from pyquaternion import Quaternion
    return Quaternion(info["lidar2ego_rotation"]).rotation_matrix


def map_xy(points, rotation):
    points = np.asarray(points, np.float64)
    return points @ rotation[:2, :2].T


def map_boxes(boxes, rotation):
    """(N, 7+) LiDAR boxes onto the ego axes: full rotation of the centre, heading vector and xy velocity."""
    boxes = np.asarray(boxes, np.float64)
    if len(boxes) == 0:
        return boxes.copy()
    out = boxes.copy()
    rotation_4 = np.eye(4)
    rotation_4[:3, :3] = rotation
    centers = np.concatenate([boxes[:, :3], np.ones((len(boxes), 1))], axis=1) @ rotation_4.T
    out[:, :3] = centers[:, :3]
    heading = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))], axis=1) @ rotation.T
    out[:, 6] = np.arctan2(heading[:, 1], heading[:, 0])
    if boxes.shape[1] > 7:
        out[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
    return out


def angle_error(a, b, yaw_axis):
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if delta.size and delta.shape[-1] > yaw_axis:
        yaw = delta[..., yaw_axis]
        delta = delta.copy()
        delta[..., yaw_axis] = (yaw + np.pi) % (2 * np.pi) - np.pi
    return np.abs(delta).max() if delta.size else 0.0


class MapGap:
    """prepare_sample's map tensors versus VectorizeMap. Aggregated because the gap is one adapter behaviour."""

    def __init__(self):
        self.samples = 0
        self.count_mismatch = 0
        self.max_extra = 0
        self.pad_mismatch = 0
        self.point = 0.0
        self.point_at = ""

    def add(self, token, ours_pts, ours_labels, official_pts, official_labels):
        self.samples += 1
        if ours_pts.shape != official_pts.shape or ours_labels.shape != official_labels.shape:
            self.count_mismatch += 1
            self.max_extra = max(self.max_extra, int(len(ours_labels) - len(official_labels)))
            return
        if ours_pts.size == 0:
            return
        ours_pts, official_pts = np.asarray(ours_pts, np.float64), np.asarray(official_pts, np.float64)
        ours_pad = np.abs(ours_pts).max(axis=-1) >= 1e4
        official_pad = np.abs(official_pts).max(axis=-1) >= 1e4
        if not np.array_equal(ours_pad, official_pad):
            self.pad_mismatch += 1
        real = ~ours_pad & ~official_pad
        if real.any():
            err = float(np.abs(ours_pts[real] - official_pts[real]).max())
            if err > self.point:
                self.point, self.point_at = err, token


class Report:
    def __init__(self):
        self.max = {}
        self.where = {}
        self.bad = set()

    def observe(self, name, got, ref, atol, token, yaw_axis=None):
        got, ref = np.asarray(got), np.asarray(ref)
        if got.shape != ref.shape:
            check(False, f"{name} shape {tuple(got.shape)} vs official {tuple(ref.shape)} at {token}")
            self.bad.add(name)
            return
        if got.size == 0:
            err = 0.0
        elif yaw_axis is not None:
            err = float(angle_error(got, ref, yaw_axis))
        else:
            err = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max())
        if name not in self.max or err > self.max[name]:
            self.max[name], self.where[name] = err, token
        if err > atol:
            self.bad.add(name)
            if len([f for f in failures if f.startswith(name)]) < 3:
                check(False, f"{name} max abs {err:.3e} at {token} (tol {atol:.0e})")

    def finish(self, atol):
        for name in sorted(self.max):
            if name in self.bad:
                continue
            base = name.split(" ", 1)[-1]
            check(True, f"{name} max abs {self.max[name]:.3e} at {self.where[name]} (tol {atol[base]:.0e})")


def valid_rows(array, mask):
    array = np.asarray(array)
    if mask is None:
        return array
    keep = np.asarray(mask).astype(bool).reshape(-1)
    return array[:len(keep)][keep]


def map_points(official, rotation):
    """Rotate real samples; VectorizeMap's 1e5 padding is not a point."""
    pts = np.asarray(official, np.float64)
    real = np.isfinite(pts).all(axis=-1) & (np.abs(pts) < 1e4).all(axis=-1)
    out = np.full_like(pts, 1e5)
    if real.any():
        out[real] = pts[real] @ rotation[:2, :2].T
    return out


def indices_of(n, scenes, limit):
    if limit <= 0 or limit >= n:
        return list(range(n))
    picks = []
    for scene in scenes:
        picks += [scene[0], scene[min(1, len(scene) - 1)], scene[len(scene) // 2], scene[-1]]
    chosen = sorted(set(picks))
    return chosen[:limit] if len(chosen) > limit else chosen


def scene_names(data_root, version):
    import json
    path = os.path.join(data_root, version, "scene.json")
    with open(path) as handle:
        return {row["token"]: row["name"] for row in json.load(handle)}


def official_dataset(cls, infos, vad, names):
    dataset = cls.__new__(cls)
    dataset.data_infos = infos
    dataset.use_valid_flag = False
    dataset.with_velocity = True
    dataset.CLASSES = CLASSES
    dataset.modality = {"use_camera": True}
    dataset.vad_infos_map = vad
    dataset.test_mode = False
    dataset.nusc = types.SimpleNamespace(get=lambda table, token: {"name": names[token]})
    return dataset


def load_vad(info_dir, split):
    for folder in (Path(info_dir), Path(info_dir).parent):
        path = folder / f"vad_nuscenes_infos_temporal_{split}.pkl"
        if path.is_file():
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            infos = data["infos"] if isinstance(data, dict) else data
            return {info["token"]: info for info in infos}, str(path)
    return None, None


def compare_split(split, train, source, dataset, pipeline, profile, report, atol, limit, maps, frame):
    chosen = indices_of(len(source), source.scenes, limit)
    print(f"{split}: {len(chosen)} of {len(source)} samples, {'train' if train else 'test'} pipeline")
    pitch = roll = yaw_gap = 0.0
    for done, index in enumerate(chosen, start=1):
        if done == 1 or done % 40 == 0 or done == len(chosen):
            print(f"{split} {done}/{len(chosen)}", flush=True)
        token = source.tokens[index]
        try:
            data = dataset.get_data_info(index)
            if data.get("gt_ego_his_trajs") is None:
                scene, k = source.position[index]
                data["gt_ego_his_trajs"] = vad_his_offsets(source.infos, scene, k)
            if list(source.infos[index]["cams"]) != CAM_PKL:
                check(False, f"pkl camera order {list(source.infos[index]['cams'])} at {token}")
                return pitch, roll, yaw_gap
            data["img_filename"] = [abs_path(path, source.path_root) for path in data["img_filename"]]
            data["aug_config"] = augmentation(train)
            data = run_pipeline(pipeline, data)
            features, targets = source[index][:2]
            ours_f, ours_t = prepare_sample(features, targets if train else targets, profile, train=train)
            actual_rotation = rotation_of(source.infos[index])
            rotation = actual_rotation if frame == "ego" else np.eye(3)
            pitch = max(pitch, abs(np.arctan2(-actual_rotation[2, 0], np.hypot(
                actual_rotation[0, 0], actual_rotation[1, 0]))))
            roll = max(roll, abs(np.arctan2(actual_rotation[2, 1], actual_rotation[2, 2])))
            if "gt_bboxes_3d" in data:
                official_boxes = map_boxes(as_np(data["gt_bboxes_3d"]), rotation)
                raw = as_np(data["gt_bboxes_3d"])
                if len(raw):
                    added = (raw[:, 6] + np.arctan2(rotation[1, 0], rotation[0, 0]) + np.pi) % (2 * np.pi) - np.pi
                    delta = (added - official_boxes[:, 6] + np.pi) % (2 * np.pi) - np.pi
                    yaw_gap = max(yaw_gap, float(np.abs(delta).max()))
            compare_fields(split, token, data, ours_f, ours_t, rotation, report, atol, train, maps)
        except Exception as exc:
            check(False, f"{split} {token}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return pitch, roll, yaw_gap


def compare_fields(split, token, data, ours_f, ours_t, rotation, report, atol, train, maps):
    prefix = f"{split} "
    perm = CAM_PERM
    report.observe(prefix + "img", ours_f["img"].numpy(), as_np(data["img"])[perm], atol["img"], token)
    official_proj = as_np(data["projection_mat"])[perm] @ np.linalg.inv(np.block(
        [[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    report.observe(prefix + "projection_mat", ours_f["projection_mat"].numpy(), official_proj, atol["projection_mat"], token)
    report.observe(prefix + "image_wh", ours_f["image_wh"].numpy(), as_np(data["image_wh"])[perm], atol["image_wh"], token)
    report.observe(prefix + "focal", ours_f["focal"].numpy(), as_np(data["focal"])[perm], atol["focal"], token)
    report.observe(prefix + "timestamp", ours_f["timestamp"].numpy(), np.asarray(data["timestamp"], np.float64),
                   atol["timestamp"], token)
    inv_axes = np.eye(4)
    inv_axes[:3, :3] = rotation.T
    report.observe(prefix + "T_global", ours_f["img_metas"]["T_global"], as_np(data["T_global"]) @ inv_axes,
                   atol["T_global"], token)
    report.observe(prefix + "T_global_inv", ours_f["img_metas"]["T_global_inv"],
                   np.linalg.inv(as_np(data["T_global"]) @ inv_axes), atol["T_global_inv"], token)
    status = as_np(data["ego_status"]).reshape(-1)
    report.observe(prefix + "ego_status", ours_f["ego_status"].numpy(), status[:6], atol["ego_status"], token)
    if status.shape != (10,):
        check(False, f"ego_status length {status.shape} at {token}, the converter writes 10")
    hist = as_np(data.get("hist_traj"))
    if hist is None:
        check(False, f"hist_traj missing at {token}")
    else:
        report.observe(prefix + "hist_traj", ours_f["hist_traj"].numpy(), map_xy(hist, rotation), atol["hist_traj"], token)
    command = as_np(data["gt_ego_fut_cmd"]).reshape(-1)
    report.observe(prefix + "gt_ego_fut_cmd", np.asarray([int(ours_f["gt_ego_fut_cmd"])]),
                   np.asarray([int(np.argmax(command))]), atol["gt_ego_fut_cmd"], token)
    report.observe(prefix + "gt_ego_fut_trajs", ours_t["gt_ego_fut_trajs"].numpy(),
                   map_xy(as_np(data["gt_ego_fut_trajs"]), rotation), atol["gt_ego_fut_trajs"], token)
    report.observe(prefix + "gt_ego_fut_masks", ours_t["gt_ego_fut_masks"].numpy(),
                   as_np(data["gt_ego_fut_masks"]).reshape(-1), atol["gt_ego_fut_masks"], token)
    if not train:
        return
    boxes = map_boxes(as_np(data["gt_bboxes_3d"]), rotation)
    labels = as_np(data["gt_labels_3d"]).reshape(-1)
    ours_boxes = valid_rows(ours_t["gt_bboxes_3d"].numpy(), ours_t["box_mask"].numpy())
    ours_labels = valid_rows(ours_t["gt_labels_3d"].numpy(), ours_t["box_mask"].numpy())
    report.observe(prefix + "gt_bboxes_3d", ours_boxes, boxes, atol["gt_bboxes_3d"], token, yaw_axis=6)
    report.observe(prefix + "gt_labels_3d", ours_labels, labels, atol["gt_labels_3d"], token)
    agents = as_np(data["gt_agent_fut_trajs"])
    report.observe(prefix + "gt_agent_fut_trajs",
                   valid_rows(ours_t["gt_agent_fut_trajs"].numpy(), ours_t["box_mask"].numpy()),
                   map_xy(agents, rotation), atol["gt_agent_fut_trajs"], token)
    report.observe(prefix + "gt_agent_fut_masks",
                   valid_rows(ours_t["gt_agent_fut_masks"].numpy(), ours_t["box_mask"].numpy()),
                   as_np(data["gt_agent_fut_masks"]), atol["gt_agent_fut_masks"], token)
    official_map = map_points(as_np(data["gt_map_pts"]), rotation).astype(np.float32)
    ours_map = valid_rows(ours_t["gt_map_pts"].numpy(), ours_t["map_mask"].numpy())
    ours_labels = valid_rows(ours_t["gt_map_labels"].numpy(), ours_t["map_mask"].numpy())
    maps.add(token, ours_map, ours_labels, official_map, as_np(data["gt_map_labels"]).reshape(-1))


def lidar_global(info):
    from pyquaternion import Quaternion

    def rigid(rotation, translation):
        out = np.eye(4)
        out[:3, :3] = Quaternion(rotation).rotation_matrix
        out[:3, 3] = np.asarray(translation, np.float64).reshape(3)
        return out

    return rigid(info["ego2global_rotation"], info["ego2global_translation"]) @ rigid(
        info["lidar2ego_rotation"], info["lidar2ego_translation"])


def vad_his_offsets(infos, scene, k):
    """vad_nuscenes_converter his_ts=4: four xy offsets of the lidar origin in the current lidar frame.

    The mini infos pkl has no gt_ego_his_trajs and the temporal vad pkl is not on disk; this is that converter,
    and LoadAnnotations3D_E2E integrates it.
    """
    his_ts = 4
    pos, diff, cursor = np.zeros((5, 3)), np.zeros((5, 3)), k
    for i in range(his_ts, -1, -1):
        if cursor is not None:
            pos[i] = lidar_global(infos[scene[cursor]])[:3, 3]
            if cursor + 1 < len(scene):
                diff[i] = lidar_global(infos[scene[cursor + 1]])[:3, 3] - pos[i]
            cursor = cursor - 1 if cursor - 1 >= 0 else None
        else:
            pos[i] = pos[i + 1] - diff[i + 1]
            diff[i] = diff[i + 1]
    current = lidar_global(infos[scene[k]])
    local = (pos - current[:3, 3]) @ current[:3, :3]
    return (local[1:] - local[:-1])[:, :2].astype(np.float32)


def compare_sequences(dataset, source):
    """Official flags start a sequence on an empty sweeps list. This mini pkl stores no sweeps at all, which would
    make every frame its own sequence and fail the sequences_split_num=2 length assert. Empty sweeps are the
    converter's mark for the first frame of a scene, so the flags below use scene_token changes."""
    n_empty = sum(len(info.get("sweeps") or []) == 0 for info in source.infos)
    saved = dataset.data_infos
    if n_empty == len(source.infos):
        marked = []
        previous = None
        for info in source.infos:
            copied = dict(info)
            copied["sweeps"] = [] if info["scene_token"] != previous else [0]
            previous = info["scene_token"]
            marked.append(copied)
        dataset.data_infos = marked
        print(f"note sweeps are empty on every frame; sequence flags follow scene_token, which empty sweeps mark")
    dataset.sequences_split_num = 2
    dataset._set_sequence_group_flag()
    dataset.data_infos = saved
    official = [np.where(dataset.flag == group)[0].tolist() for group in range(int(dataset.flag.max()) + 1)]
    ours = [part for seq in source.sequences() for part in SequenceBatchSampler.cut(list(seq), 2)]
    check(official == ours, f"sequence parts: {len(ours)} parts, official {len(official)}, equal {official == ours}")
    if official != ours:
        for index, (left, right) in enumerate(zip(official, ours)):
            if left != right:
                check(False, f"first part mismatch at group {index}: official {left[:8]} ({len(left)}) ours {right[:8]} ({len(right)})")
                break
        if len(official) != len(ours):
            check(False, f"part count official {len(official)} ours {len(ours)}")


def compare_future_boxes(dataset, source, report, atol, limit):
    chosen = indices_of(len(source), source.scenes, limit)
    for index in chosen:
        token = source.tokens[index]
        ann = dataset.get_ann_info(index)
        if "fut_boxes" not in ann:
            check(False, "official get_ann_info did not build fut_boxes (vad infos missing)")
            return
        rotation = rotation_of(source.infos[index])
        ours = source.eval_info(token)["boxes"]
        official = ann["fut_boxes"]
        for step in range(6):
            raw = official[step] if step < len(official) else np.zeros((0, 7))
            mapped = map_boxes(raw, rotation)
            got = ours[step] if step < len(ours) else np.zeros((0, 5), np.float32)
            ref = mapped[:, [0, 1, 3, 4, 6]] if len(mapped) else np.zeros((0, 5))
            report.observe("fut_boxes", got, ref, atol["fut_boxes"], f"{token} t+{step + 1}", yaw_axis=4)


def compare_occupancy(registry, data_root, version, info_path, path_root):
    """LoadOccWorldLabels reads gts/{scene}/{token}/labels.npz 'semantics'. No real Occ3D files, so a synthetic one."""
    names = scene_names(data_root, version)
    with tempfile.TemporaryDirectory() as tmp:
        scene = next(iter(names.values()))
        token = next(iter(names))
        folder = Path(tmp) / "gts" / scene / token
        folder.mkdir(parents=True)
        semantics = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        np.savez(folder / "labels.npz", semantics=semantics)
        loaded = registry["LoadOccWorldLabels"](data_root=tmp, input_dataset="gts")(
            {"sample_idx": token, "scene_name": scene})
        official = loaded["gt_occ_dense"]
        # our path is occ_root/scene_name/token/labels.npz; the official root joins data_root/gts
        our_path = os.path.join(tmp, "gts", scene, token, "labels.npz")
        ours = load_occ(our_path, "occ3d")
        same = official is not None and np.array_equal(ours, official.astype(np.int64))
        check(same, f"occupancy semantics equal on a synthetic labels.npz ({None if official is None else official.shape})")
        source = NuScenesSource(info_path, path_root=path_root, data_root=data_root, version=version,
                                occ_root=os.path.join(tmp, "gts"))
        sample_token = source.tokens[0]
        expected = os.path.join(tmp, "gts", names[source.infos[0]["scene_token"]], sample_token, "labels.npz")
        check("occ_path" not in source.targets(0), "a missing labels.npz is omitted, as LoadOccWorldLabels sets None")
        os.makedirs(os.path.dirname(expected), exist_ok=True)
        np.savez(expected, semantics=semantics)
        present = source.targets(0)
        check(present["occ_path"] == expected and present["occ_format"] == "occ3d",
              "occ_path is occ_root/scene.json name/token/labels.npz")
        missing = registry["LoadOccWorldLabels"](data_root=tmp, input_dataset="gts")(
            {"sample_idx": "none", "scene_name": scene})
        check(missing["gt_occ_dense"] is None, "official LoadOccWorldLabels sets gt_occ_dense None when the file is absent")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--info-dir", default=os.environ.get("NUSCENES_INFOS", "/data/yongkangli/UniDriveVLA/nuScenes/data/infos/mini"))
    parser.add_argument("--path-root", default=os.environ.get("NUSCENES_PATH_ROOT", "/data/yongkangli/UniDriveVLA/nuScenes"))
    parser.add_argument("--data-root", default=os.environ.get("NUSCENES_ROOT", "/data/yongkangli/UniDriveVLA/nuScenes/data/nuscenes"))
    parser.add_argument("--version", default=os.environ.get("NUSCENES_VERSION", "v1.0-mini"))
    parser.add_argument("--official", default=os.environ.get("NUSCENES_OFFICIAL", "/data/yongkangli/UniDriveVLA/nuScenes"))
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--frame", choices=("lidar", "ego"), default="lidar",
                        help="compare native main LiDAR tensors by default; ego preserves the legacy rebase check")
    parser.add_argument("--limit", type=int, default=0, help="samples per split; 0 means every sample")
    args = parser.parse_args()

    profile_name = "nuscenes_ego" if args.frame == "ego" else "nuscenes"
    profile = Profile.from_config(__import__("omegaconf").OmegaConf.load(
        PACKAGE / "config" / "profile" / f"{profile_name}.yaml"))
    check(profile.frame == args.frame, f"profile frame {profile.frame} matches requested oracle frame {args.frame}")
    check(list(profile.cameras) == PROFILE_CAMS, f"profile camera order is the head order {PROFILE_CAMS}")
    print(f"oracle frame: {args.frame}")
    dataset_mod, registry, _modules = load_official(Path(args.official))
    atol = {"img": 0, "projection_mat": 1e-3, "image_wh": 0, "focal": 1e-3, "timestamp": 1e-4,
            "T_global": 1e-4, "T_global_inv": 1e-4, "ego_status": 1e-5, "hist_traj": 1e-3,
            "gt_ego_fut_cmd": 0, "gt_ego_fut_trajs": 1e-3, "gt_ego_fut_masks": 0,
            "gt_bboxes_3d": 1e-3, "gt_labels_3d": 0, "gt_agent_fut_trajs": 1e-3, "gt_agent_fut_masks": 0,
            "gt_map_pts": 1e-3, "gt_map_labels": 0, "fut_boxes": 1e-3}
    report, maps = Report(), MapGap()
    splits = ("train", "val") if args.split == "both" else (args.split,)
    noted = False
    for split in splits:
        info_path = os.path.join(args.info_dir, f"nuscenes_infos_{split}.pkl")
        source = NuScenesSource(info_path, path_root=args.path_root, data_root=args.data_root, version=args.version,
                                frame=args.frame)
        vad, vad_file = load_vad(args.info_dir, split)
        if vad is None:
            print(f"note {split}: no vad_nuscenes_infos_temporal pkl; hist offsets follow vad_nuscenes_converter")
        else:
            print(f"note {split}: vad infos {vad_file}")
        names = scene_names(args.data_root, args.version)
        dataset = official_dataset(dataset_mod.NuScenes3DDataset, source.infos, vad if vad is not None else {}, names)
        train = split == "train"
        pipeline = make_pipeline(registry, train)
        pitch, roll, yaw_gap = compare_split(split, train, source, dataset, pipeline, profile, report, atol,
                                             args.limit, maps, args.frame)
        if not noted:
            print(f"note lidar2ego pitch max {pitch:.3e} rad, roll max {roll:.3e} rad, "
                  f"additive yaw vs heading max {yaw_gap:.3e} rad")
            noted = True
        if train:
            try:
                compare_sequences(dataset, source)
            except AssertionError as exc:
                check(False, f"official sequence split assert: {exc}")
        compare_future_boxes(dataset, source, report, atol, args.limit)
        if split == "train":
            compare_occupancy(registry, args.data_root, args.version, info_path, args.path_root)
    report.finish(atol)
    if maps.samples:
        same = maps.point <= atol["gt_map_pts"] and maps.count_mismatch == 0 and maps.pad_mismatch == 0
        check(same, f"gt_map_pts over {maps.samples} train samples: line-count mismatches {maps.count_mismatch} "
                    f"(max extra lines {maps.max_extra}), permutation-pad mismatches {maps.pad_mismatch}, "
                    f"real-point max abs {maps.point:.3e} at {maps.point_at}")
        if not same:
            if args.frame == "ego":
                print("note map: VectorizeMap samples the extractor's polylines whole, in the LiDAR frame. "
                      "prepare_sample intersects them with the ego-axis AABB (map_roi) and resamples. lidar2ego's "
                      "roll swings the lidar ROI across that box, so rings split and cut-line samples move.")
            else:
                print("note map: this run compares raw LiDAR map tensors; inspect source extraction or sampling "
                      "against the official pipeline for any mismatch.")
    print("photometric distortion, flip and rotation were off; depth maps were not compared")
    print("OK" if not failures else f"{len(failures)} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
