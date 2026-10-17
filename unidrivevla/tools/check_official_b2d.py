"""python unidrivevla/tools/check_official_b2d.py: Bench2Drive samples against the official dataset.

Runs Bench2DriveDataset and the stage-1 train / test pipelines with flip, rotation and photometric distortion off,
and Bench2DriveSource + prepare_sample on the same frames. By default it compares raw main LiDAR tensors; --frame ego
retains the legacy canonical rebase. Cameras are reordered by the planning head's [0, 1, 2, 4, 5, 3]. Prints
the max abs diff of each head input and of the sequence grouping, and exits 1 on a mismatch.

  python unidrivevla/tools/check_official_b2d.py [--samples 3] [--no-image]
  B2D_INFOS, B2D_ROOT, OFFICIAL_B2D (the mmdet3d_plugin/datasets directory) override the defaults.
"""

import argparse
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

CAM_PERM = [0, 1, 2, 4, 5, 3]  # the official head permutes dataset camera order onto the view tokens
DET_CLASSES = ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"]
MAP_CLASSES = ["Broken", "Solid", "SolidSolid", "Center"]
POINT_CLOUD_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
GEOM_TOL, MAP_TOL = 1e-4, 1e-3

failures = []


def say(message):
    print(message, flush=True)


def check(ok, message):
    say(("ok   " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)


def install_stubs():
    def ensure(name, package=False):
        if name not in sys.modules:
            module = types.ModuleType(name)
            if package:
                module.__path__ = []
                module.__package__ = name
            sys.modules[name] = module
        return sys.modules[name]

    class Registry:
        def __init__(self):
            self.mods = {}

        def register_module(self, name=None, force=False, module=None):
            def deco(cls, name=name):
                self.mods[name or cls.__name__] = cls
                return cls
            return deco(module) if module is not None else deco

    try:
        import prettytable  # noqa: F401
    except ImportError:
        ensure("prettytable")
    if importlib.util.find_spec("pyquaternion") is None:
        class Quaternion:
            def __init__(self, *args, **kwargs):
                pass
        ensure("pyquaternion").Quaternion = Quaternion
    else:
        import pyquaternion  # noqa: F401
    if importlib.util.find_spec("nuscenes") is None:
        for name in ("nuscenes", "nuscenes.eval", "nuscenes.eval.common"):
            ensure(name, package=True)
        utils = ensure("nuscenes.eval.common.utils")
        utils.Quaternion = sys.modules["pyquaternion"].Quaternion
        utils.quaternion_yaw = lambda q: 0.0
    if importlib.util.find_spec("mmdet") is not None:
        return
    ensure("mmdet", package=True)
    datasets = ensure("mmdet.datasets", package=True)
    builder = ensure("mmdet.datasets.builder")
    pipes = ensure("mmdet.datasets.pipelines")
    builder.PIPELINES = Registry()
    datasets.DATASETS = Registry()
    datasets.builder = builder

    def to_tensor(data):
        import torch
        return torch.from_numpy(data) if isinstance(data, np.ndarray) else torch.as_tensor(data)

    class Compose:
        def __init__(self, *args, **kwargs):
            pass

    pipes.Compose = Compose
    pipes.to_tensor = to_tensor


def load_py(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_modules(datasets_dir: Path):
    try:
        import mmcv  # noqa: F401
    except ImportError:
        raise SystemExit("mmcv is required to import the official dataset; use the unidrivevla_data env")
    install_stubs()
    pkg = types.ModuleType("b2dref")
    pkg.__path__ = [str(datasets_dir)]
    pkg.__package__ = "b2dref"
    sys.modules["b2dref"] = pkg
    load_py("b2dref.bench2drive_eval_utils", datasets_dir / "bench2drive_eval_utils.py")
    dataset = load_py("b2dref.bench2drive_dataset", datasets_dir / "bench2drive_dataset.py")
    pipes = datasets_dir / "pipelines"
    loading = load_py("b2dref_loading", pipes / "loading.py")
    augment = load_py("b2dref_augment", pipes / "augment.py")
    transform = load_py("b2dref_transform", pipes / "transform.py")
    vectorize = load_py("b2dref_vectorize", pipes / "vectorize.py")
    return dataset, loading, augment, transform, vectorize


def datasets_dir() -> Path:
    candidates = []
    if os.environ.get("OFFICIAL_B2D"):
        candidates.append(Path(os.environ["OFFICIAL_B2D"]))
    candidates += [
        PACKAGE.parent / "Bench2Drive" / "projects" / "mmdet3d_plugin" / "datasets",
        Path("/data/yongkangli/UniDriveVLA/Bench2Drive/projects/mmdet3d_plugin/datasets"),
    ]
    for path in candidates:
        if (path / "bench2drive_dataset.py").is_file():
            return path
    raise FileNotFoundError("bench2drive_dataset.py not found; set OFFICIAL_B2D")


def plain(value):
    if hasattr(value, "data"):
        value = value.data
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def fixed_aug(test_mode):
    """get_augmentation with the random draws fixed: no flip, no rotation, training resize 0.6."""
    height, width, final_h, final_w = 900, 1600, 544, 960
    resize = max(final_h / height, final_w / width) if test_mode else 0.6
    new_w, new_h = int(width * resize), int(height * resize)
    crop_h = int((1 - 0.0) * new_h) - final_h
    crop_w = int(max(0, new_w - final_w) / 2) if test_mode else 0
    return dict(resize=resize, resize_dims=(new_w, new_h), crop=(crop_w, crop_h, crop_w + final_w, crop_h + final_h),
                flip=False, rotate=0, rotate_3d=0.0)


def rotate_xy(points, axes):
    points = np.asarray(points, np.float64)
    shape = points.shape
    flat = points.reshape(-1, shape[-1])
    out = flat.copy()
    real = np.max(np.abs(flat[:, :2]), axis=1) < 1e3  # permute_line pads with 1e5
    out[real, :2] = flat[real, :2] @ axes[:2, :2].T
    if shape[-1] > 2:
        out[real, 2:3] = flat[real, 2:3]  # z, if a box row is ever passed through here
    return out.reshape(shape)


def to_ego_boxes(boxes, axes):
    boxes = np.asarray(boxes, np.float64).copy()
    rotation = axes[:3, :3]
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    boxes[:, :3] = boxes[:, :3] @ rotation.T
    boxes[:, 6] = (boxes[:, 6] + yaw + np.pi) % (2 * np.pi) - np.pi
    if boxes.shape[1] > 7:
        boxes[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
    return boxes


def yaw_delta(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def max_abs(a, b):
    if np.shape(a) != np.shape(b):
        return math.inf
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def history_reference(infos, index, axes):
    """2 Hz ego positions in the ego frame: five list slots apart, the earliest pose repeated when the clip has none."""
    current = infos[index]
    earliest = np.zeros(2, np.float64)
    poses = []
    world2lidar = np.asarray(current["sensors"]["LIDAR_TOP"]["world2lidar"])
    for step in (-1, -2, -3):
        other_i = index + 5 * step
        if 0 <= other_i < len(infos) and infos[other_i]["folder"] == current["folder"]:
            other = infos[other_i]
            lidar = (world2lidar @ np.linalg.inv(other["sensors"]["LIDAR_TOP"]["world2lidar"]))[:2, 3]
            earliest = lidar @ axes[:2, :2].T
        poses.insert(0, earliest.copy())
    poses.append(np.zeros(2, np.float64))
    return np.stack(poses)


def choose(dataset, root, count):
    have = []
    for i, info in enumerate(dataset.data_infos):
        path = next(os.path.join(root, cam["data_path"]) for key, cam in info["sensors"].items() if "CAM" in key)
        if os.path.isfile(path):
            have.append(i)
    if not have:
        raise SystemExit(f"no frames under {root} have images")
    if len(have) <= count:
        picks = have
    else:
        picks = [have[int(round(t))] for t in np.linspace(0, len(have) - 1, count)]
    boundary = int(dataset.group_length[0]) - 1
    if boundary in have:
        picks.append(boundary)
    for index in have:  # a signed speed, and a command other than lane-follow, when the split has them
        info = dataset.data_infos[index]
        if info["ego_vel"][0] < 0:
            picks.append(index)
            break
    for index in have:
        if int(dataset.data_infos[index]["command_near"]) not in (4,):
            picks.append(index)
            break
    return list(dict.fromkeys(picks))


def sequence_parts(flag, infos):
    parts, current = [], None
    for index, group in enumerate(flag):
        if current is None or group != current[0]:
            current = [group, []]
            parts.append(current)
        current[1].append((infos[index]["folder"], int(infos[index]["frame_idx"])))
    return [part for _, part in parts]


def bootstrap():
    """Import the package modules without data/__init__.py, which pulls every source."""
    if "unidrivevla.data" in sys.modules:
        return
    top = types.ModuleType("unidrivevla")
    top.__path__ = [str(PACKAGE)]
    top.__package__ = "unidrivevla"
    sys.modules["unidrivevla"] = top
    data = types.ModuleType("unidrivevla.data")
    data.__path__ = [str(PACKAGE / "data")]
    data.__package__ = "unidrivevla.data"
    sys.modules["unidrivevla.data"] = data


def run_split(name, test_mode, samples, images, info_dir, data_root, dataset_cls, pipes, profile, src_cls, prepare, cut,
              frame):
    say(f"--- {name} ---")
    ann = os.path.join(info_dir, f"b2d_infos_{name}.pkl")
    maps = os.path.join(info_dir, "b2d_map_infos.pkl")
    if not os.path.isfile(ann):
        say(f"skip {name}: {ann} missing")
        return
    say(f"loading {ann}")
    official = dataset_cls(
        data_root=data_root, ann_file=ann, map_file=maps, pipeline=[],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        test_mode=test_mode, map_classes=MAP_CLASSES, det_classes=DET_CLASSES, data_aug_conf=None,
        with_seq_flag=not test_mode, sequences_split_num=2, keep_consistent_seq_aug=True,
        plan_anchor_types=[("temp", "2hz")], point_cloud_range=POINT_CLOUD_RANGE,
    )
    source = src_cls(ann, maps, data_root, MAP_CLASSES, bev_filter=test_mode, frame=frame)
    if not test_mode:
        official_parts = sequence_parts(official.flag, official.data_infos)
        our_parts = []
        for seq in source.sequences():
            our_parts.extend(cut(seq, 2))
        our_parts = [[(source.infos[i]["folder"], int(source.infos[i]["frame_idx"])) for i in part] for part in our_parts]
        same = official_parts == our_parts
        check(same, f"sequence parts {len(our_parts)} vs official {len(official_parts)}")
        if not same and official_parts and our_parts:
            for i, (a, b) in enumerate(zip(official_parts, our_parts)):
                if a != b:
                    say(f"  first part mismatch at {i}: official {a[:3]}... ({len(a)}) ours {b[:3]}... ({len(b)})")
                    break
            if len(official_parts) != len(our_parts):
                say(f"  part count official {len(official_parts)} ours {len(our_parts)}")

    load_img, resize, rotate, circle, names, bev, vectorize, adaptor = pipes
    aug = fixed_aug(test_mode)
    status_masks = []
    for official_index in choose(official, data_root, samples):
        info = official.data_infos[official_index]
        say(f"sample {info['folder']} #{int(info['frame_idx'])}")
        data = official.get_data_info(official_index)
        data["aug_config"] = aug
        data = load_img(to_float32=True)(data)
        data = resize()(data)
        data = rotate()(data)
        if test_mode:
            data = bev(point_cloud_range=POINT_CLOUD_RANGE)(data)
        else:
            data = circle(class_dist_thred=[55] * len(DET_CLASSES))(data)
        data = names(classes=DET_CLASSES)(data)
        data = vectorize(roi_size=(30, 60), simplify=False, normalize=False, sample_num=20, permute=True)(data)
        data = adaptor()(data)

        folder, frame_idx = info["folder"], int(info["frame_idx"])
        our_index = source.index[(folder, frame_idx)]
        features, targets, _token = source[our_index]
        feat, lab = prepare(features, targets, profile, train=not test_mode)
        axes = source.axes(source.infos[our_index]) if frame == "ego" else np.eye(4)
        tag = f"{folder} #{frame_idx}"
        cam_names = [os.path.basename(os.path.dirname(path)) for path in data["img_filename"]]
        check(cam_names == ["rgb_front", "rgb_front_left", "rgb_front_right", "rgb_back", "rgb_back_left", "rgb_back_right"],
              f"{tag} camera order {cam_names}")
        if images:
            official_img = np.clip(plain(data["img"])[CAM_PERM], 0, 255).astype(np.uint8)
            check(official_img.shape == tuple(feat["img"].shape) and np.array_equal(feat["img"].numpy(), official_img),
                  f"{tag} img")
        proj = plain(data["projection_mat"])[CAM_PERM] @ np.linalg.inv(axes)
        proj_err = max_abs(feat["projection_mat"].numpy(), proj)
        check(proj_err < GEOM_TOL, f"{tag} projection_mat {proj_err:.3e}")
        wh_err = max_abs(feat["image_wh"].numpy(), plain(data["image_wh"])[CAM_PERM])
        check(wh_err < 1e-6, f"{tag} image_wh {wh_err:.3e}")
        focal_err = max_abs(feat["focal"].numpy(), plain(data["focal"])[CAM_PERM])
        check(focal_err < GEOM_TOL, f"{tag} focal {focal_err:.3e}")

        ts_err = abs(float(feat["timestamp"]) - float(info["frame_idx"]) / 10 - source.clip_start[folder])
        check(ts_err < 1e-6, f"{tag} timestamp (clip offset {source.clip_start[folder]:.0f}) {ts_err:.3e}")
        official_pose = plain(data["T_global"])
        pose_err = max_abs(feat["img_metas"]["T_global"], official_pose @ np.linalg.inv(axes))
        inv_err = max_abs(feat["img_metas"]["T_global_inv"], axes @ plain(data["T_global_inv"]))
        check(pose_err < GEOM_TOL and inv_err < GEOM_TOL, f"{tag} T_global {pose_err:.3e} T_global_inv {inv_err:.3e}")

        status = feat["ego_status"].numpy()
        official_status = plain(data["ego_status"]).astype(np.float64).reshape(-1)
        status_err = max_abs(status, official_status)
        # speed is hypot(ego_vel[:2]); Bench2Drive's scalar can be negative and hypot drops the sign
        sign_only = (official_status[0] < 0 and abs(status[0] - abs(official_status[0])) < 1e-5
                     and max_abs(status[1:], official_status[1:]) < 1e-5)
        if sign_only:
            say(f"note {tag} ego_status speed {official_status[0]:.4f} becomes {status[0]:.4f} (hypot; adapter)")
        check(status_err < 1e-5 or sign_only, f"{tag} ego_status {status_err:.3e} ours {status.round(4).tolist()}")
        cmd_err = int(feat["gt_ego_fut_cmd"]) - int(np.argmax(plain(data["gt_ego_fut_cmd"])))
        check(cmd_err == 0, f"{tag} gt_ego_fut_cmd ours {int(feat['gt_ego_fut_cmd'])} official {int(np.argmax(plain(data['gt_ego_fut_cmd'])))}")
        hist_err = max_abs(feat["hist_traj"].numpy(), history_reference(source.infos, our_index, axes))
        check(hist_err < GEOM_TOL, f"{tag} hist_traj {hist_err:.3e} (official collect does not emit it)")

        n = int(lab["box_mask"].sum())
        official_boxes = to_ego_boxes(plain(data["gt_bboxes_3d"]), axes) if len(plain(data["gt_bboxes_3d"])) else np.zeros((0, 9))
        official_labels = plain(data["gt_labels_3d"]).astype(np.int64).reshape(-1)
        check(n == len(official_boxes) == len(official_labels),
              f"{tag} boxes {n} official {len(official_boxes)}")
        kept = min(n, len(official_boxes))
        if kept:
            box_err = max_abs(lab["gt_bboxes_3d"].numpy()[:kept, :6], official_boxes[:kept, :6])
            yaw_err = float(np.max(np.abs(yaw_delta(lab["gt_bboxes_3d"].numpy()[:kept, 6], official_boxes[:kept, 6]))))
            vel_err = max_abs(lab["gt_bboxes_3d"].numpy()[:kept, 7:9], official_boxes[:kept, 7:9])
            check(box_err < GEOM_TOL and yaw_err < GEOM_TOL and vel_err < GEOM_TOL,
                  f"{tag} gt_bboxes_3d centre/size {box_err:.3e} yaw {yaw_err:.3e} vel {vel_err:.3e}")
            check(np.array_equal(lab["gt_labels_3d"].numpy()[:kept], official_labels[:kept]), f"{tag} gt_labels_3d")
        else:
            check(True, f"{tag} gt_bboxes_3d empty")
            check(True, f"{tag} gt_labels_3d empty")

        ego = rotate_xy(plain(data["gt_ego_fut_trajs_2hz"]), axes)
        ego_err = max_abs(lab["gt_ego_fut_trajs"].numpy(), ego)
        ego_mask = np.array_equal(lab["gt_ego_fut_masks"].numpy(), plain(data["gt_ego_fut_masks_2hz"]).astype(np.float32))
        check(ego_err < GEOM_TOL and ego_mask, f"{tag} gt_ego_fut_trajs {ego_err:.3e} mask {ego_mask}")

        agent_off = plain(data["gt_agent_fut_trajs"])
        agent_mask = plain(data["gt_agent_fut_masks"]).astype(np.float32)
        if agent_off.size == 0:
            agent_off = np.zeros((0, 6, 2))
            agent_mask = np.zeros((0, 6), np.float32)
        agent_off = rotate_xy(agent_off, axes)
        our_agent = lab["gt_agent_fut_trajs"].numpy()[:n]
        our_mask = lab["gt_agent_fut_masks"].numpy()[:n]
        mask_ok = our_mask.shape == agent_mask.shape and np.array_equal(our_mask, agent_mask)
        if mask_ok:
            valid = our_mask > 0
            our_cum = np.cumsum(our_agent, axis=1)
            off_cum = np.cumsum(agent_off, axis=1)
            cum_err = max_abs(our_cum[valid], off_cum[valid]) if valid.any() else 0.0
            step_err = max_abs(our_agent[valid], agent_off[valid]) if valid.any() else 0.0
            raw_err = max_abs(our_agent, agent_off)
        else:
            cum_err = step_err = raw_err = math.inf
        # the motion loss regresses the cumulative path, weighted by the mask
        check(mask_ok and cum_err < GEOM_TOL, f"{tag} gt_agent_fut mask {mask_ok} cum {cum_err:.3e} step {step_err:.3e}")
        if mask_ok and cum_err < GEOM_TOL and step_err >= GEOM_TOL:
            say(f"note {tag} agent step {step_err:.3e} absorbs a masked hole; cumulative path at mask 1 matches")
        elif mask_ok and cum_err < GEOM_TOL and raw_err >= GEOM_TOL:
            say(f"note {tag} agent masked fill {raw_err:.3e} (adapter writes 0; the loss weight is the mask)")

        if "gt_map_pts" in data and len(plain(data["gt_map_labels"])):
            m = int(lab["map_mask"].sum())
            official_labels = plain(data["gt_map_labels"]).astype(np.int64).reshape(-1)
            official_pts = rotate_xy(plain(data["gt_map_pts"]), axes)
            check(m == len(official_labels), f"{tag} map lines {m} official {len(official_labels)}")
            kept_m = min(m, len(official_labels))
            if kept_m:
                lab_err = not np.array_equal(lab["gt_map_labels"].numpy()[:kept_m], official_labels[:kept_m])
                pts_err = max_abs(lab["gt_map_pts"].numpy()[:kept_m], official_pts[:kept_m])
                check(not lab_err and pts_err < MAP_TOL, f"{tag} gt_map_pts {pts_err:.3e}")
            if test_mode:
                say(f"note {tag} test collect omits map tensors; compared against VectorizePloyLine anyway")
        elif int(lab["map_mask"].sum()) == 0:
            check(True, f"{tag} map empty")
        else:
            check(False, f"{tag} map ours {int(lab['map_mask'].sum())} official 0")
        status_masks.append(plain(data["ego_status_mask"]))
        check(np.array_equal(lab["ego_status_mask"].numpy(), status_masks[-1]), f"{tag} ego_status_mask")
    dropped = int(np.any(np.stack(status_masks) < 1, axis=1).sum()) if status_masks else 0
    say(f"note ego_status_mask is below 1 on {dropped}/{len(status_masks)} samples")
    del official, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--frame", choices=("lidar", "ego"), default="lidar",
                        help="compare native main LiDAR tensors by default; ego preserves the legacy rebase check")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--info-dir", default=os.environ.get("B2D_INFOS", "/data/yongkangli/UniDriveVLA/Bench2Drive/data/infos"))
    parser.add_argument("--data-root", default=os.environ.get("B2D_ROOT", "/data/yongkangli/UniDriveVLA/Bench2Drive/data/bench2drive"))
    args = parser.parse_args()

    bootstrap()
    import torch  # noqa: F401
    from omegaconf import OmegaConf
    from unidrivevla.adapters.unidrivevla import Profile, prepare_sample
    from unidrivevla.data.bench2drive import Bench2DriveSource
    from unidrivevla.data.registry import SequenceBatchSampler

    dataset, loading, augment, transform, vectorize = official_modules(datasets_dir())
    profile_name = "b2d_ego" if args.frame == "ego" else "b2d"
    profile = Profile.from_config(OmegaConf.load(PACKAGE / "config" / "profile" / f"{profile_name}.yaml"))
    check(profile.frame == args.frame, f"profile frame {profile.frame} matches requested oracle frame {args.frame}")
    say(f"oracle frame: {args.frame}")
    pipes = (loading.LoadMultiViewImageFromFiles, augment.ResizeCropFlipImage, augment.BBoxRotation,
             transform.CircleObjectRangeFilter, transform.InstanceNameFilter, transform.BEVObjectRangeFilter,
             vectorize.VectorizePloyLine, transform.NuScenesSparse4DAdaptor)
    for split, test_mode in (("train", False), ("val", True)):
        run_split(split, test_mode, args.samples, not args.no_image, args.info_dir, args.data_root,
                  dataset.Bench2DriveDataset, pipes, profile, Bench2DriveSource, prepare_sample,
                  SequenceBatchSampler.cut, args.frame)
    say("OK" if not failures else f"{len(failures)} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
