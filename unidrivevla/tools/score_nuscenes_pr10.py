"""Score native nuScenes predictions with PR10, without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
from io import BytesIO
import json
import math
from numbers import Real
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types

import numpy as np


PR10 = "5fce121f4bf04eaf7e109880d4f26eed62c0eed0"
PR10_BUNDLE_SHA256 = "ceeea55deeb4b4f0dff7c9f3e0b12c3b088bc241bd6568c9f01e1238475ab3aa"
PR10_DET_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
PR10_MAP_CLASSES = ("ped_crossing", "divider", "boundary")
FIELDS = {"boxes_3d", "scores_3d", "labels_3d", "instance_ids", "vectors", "scores", "labels",
          "final_planning", "ego_status_pred"}
MOTION_FIELDS = {"boxes_3d", "scores_3d", "labels_3d", "trajs_3d", "trajs_score"}
TRACKING_METRICS = (
    "amota", "amotp", "recall", "motar", "gt", "mota", "motp", "mt", "ml", "faf",
    "tp", "fp", "fn", "ids", "frag", "tid", "lgd",
)
STATUS_NAMES = (
    "accel_x", "accel_y", "accel_z", "rot_rate_x", "rot_rate_y", "rot_rate_z",
    "vel_x", "vel_y", "vel_z", "steer",
)
# These class-specific TP errors are explicitly undefined by nuScenes for these classes.
PR10_NA_METRICS = frozenset({
    "img_bbox_NuScenes/traffic_cone_attr_err",
    "img_bbox_NuScenes/traffic_cone_vel_err",
    "img_bbox_NuScenes/traffic_cone_orient_err",
    "img_bbox_NuScenes/barrier_attr_err",
    "img_bbox_NuScenes/barrier_vel_err",
})
PR10_SOURCE_BUNDLE_NAME = "pr10_nuscenes_scorer.tar.gz"
NUSCENES_SPLITS = {"v1.0-trainval": "val", "v1.0-mini": "mini_val"}


def eval_split_for_version(version: str, *, motion_only: bool) -> str:
    """Resolve only PR10's implemented validation splits; Stage2 remains the trainval protocol."""
    if version not in NUSCENES_SPLITS:
        raise ValueError(f"unsupported PR10 nuScenes version {version!r}")
    if not motion_only and version != "v1.0-trainval":
        raise ValueError("PR10 Stage2 full scoring requires v1.0-trainval")
    return NUSCENES_SPLITS[version]


def validate_split_token_coverage(info_tokens, devkit_tokens, *, version: str, split: str) -> dict:
    """Require the supplied val infos to be exactly the official devkit split, with no duplicate tokens."""
    info_tokens = tuple(info_tokens)
    devkit_tokens = tuple(devkit_tokens)
    if (not info_tokens or any(not isinstance(token, str) or not token for token in info_tokens)
            or len(set(info_tokens)) != len(info_tokens)):
        raise ValueError("PR10 val infos must contain a nonempty, unique token list")
    if (not devkit_tokens or any(not isinstance(token, str) or not token for token in devkit_tokens)
            or len(set(devkit_tokens)) != len(devkit_tokens)):
        raise ValueError("official nuScenes split must contain a nonempty, unique token list")
    if set(info_tokens) != set(devkit_tokens):
        missing = len(set(devkit_tokens) - set(info_tokens))
        extra = len(set(info_tokens) - set(devkit_tokens))
        raise ValueError(f"PR10 val infos do not match official {split} split for {version}: "
                         f"{missing} missing, {extra} extra tokens")
    return {"version": version, "split": split, "val_tokens": len(info_tokens),
            "devkit_split_tokens": len(devkit_tokens)}


def packaged_pr10_bundle() -> Path:
    return Path(__file__).resolve().parents[1] / "references" / PR10_SOURCE_BUNDLE_NAME


def extract_pr10_bundle(bundle: Path, destination: Path) -> tuple[Path, str]:
    """Verify and safely unpack the package-carried scorer sources; no .git checkout is needed."""
    source_bytes = Path(bundle).read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != PR10_BUNDLE_SHA256:
        raise ValueError(f"PR10 source bundle hash mismatch: {digest}")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(source_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            rel = PurePosixPath(name)
            if (not name or "\\" in name or ":" in name or rel.is_absolute()
                    or any(part in (".", "..") for part in rel.parts)):
                raise ValueError(f"unsafe PR10 bundle member path: {name!r}")
            target = (destination / Path(*rel.parts)).resolve()
            try:
                target.relative_to(destination)
            except ValueError:
                raise ValueError(f"PR10 bundle member escapes extraction root: {name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported PR10 bundle member type: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read PR10 bundle member: {name!r}")
            with source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink)
    nusc_root = destination / "nuScenes"
    required = (
        destination / "LICENSE.txt",
        nusc_root / "projects" / "configs" / "UniDriveVLA" / "unidrivevla_nusc_2b_stage2.py",
        nusc_root / "projects" / "configs" / "UniDriveVLA" / "unidrivevla_nusc_2b_stage3.py",
        nusc_root / "projects" / "mmdet3d_plugin" / "datasets" / "nuscenes_3d_dataset.py",
        nusc_root / "projects" / "mmdet3d_plugin" / "datasets" / "evaluation" / "motion" / "motion_eval_uniad.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"PR10 source bundle is incomplete: {missing}")
    return nusc_root, digest


def _is_allowed_na(key, value):
    if key not in PR10_NA_METRICS:
        return False
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def install_pr10_scoring_imports(nusc_root: Path):
    """Import only the pinned scorer/data modules, skipping the plugin's model/DeepSpeed __init__ chain.

    `datasets.utils` and `planning_eval` use absolute imports below `projects.mmdet3d_plugin`, so a
    namespace-like parent module must exist. Importing the PR10 plugin root executes its `__init__.py`,
    which eagerly imports models and their DeepSpeed/CUDA stack even though scoring needs only datasets.
    """
    project_root = Path(nusc_root) / "projects"
    plugin_root = project_root / "mmdet3d_plugin"
    plugin_init = plugin_root / "__init__.py"
    if not plugin_init.is_file():
        raise FileNotFoundError(f"missing pinned PR10 plugin: {plugin_init}")

    projects = importlib.import_module("projects")
    project_paths = [Path(path).resolve() for path in getattr(projects, "__path__", ())]
    if project_root.resolve() not in project_paths:
        raise ImportError(f"projects package did not resolve to the pinned PR10 root {project_root}")

    name = "projects.mmdet3d_plugin"
    if name in sys.modules:
        raise RuntimeError("PR10 plugin was imported before the CPU scoring shim; use a fresh scorer process")
    package = types.ModuleType(name)
    package.__file__ = str(plugin_init)
    package.__package__ = name
    package.__path__ = [str(plugin_root)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = package.__path__
    package.__spec__ = spec
    sys.modules[name] = package

    datasets = importlib.import_module(name + ".datasets")
    if not hasattr(datasets, "NuScenes3DDataset"):
        raise ImportError("pinned PR10 dataset registry did not expose NuScenes3DDataset")
    planning = importlib.import_module(name + ".datasets.evaluation.planning.planning_eval")
    return planning


def load_results(directory, tokens, det_classes=PR10_DET_CLASSES, map_classes=PR10_MAP_CLASSES, motion=False):
    import torch

    paths = sorted(directory.glob("rank-*.pt"))
    if not paths or not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("PR10 scoring requires prediction shards and a nonempty, unique val token list")
    rows = {}
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict):
            raise TypeError(f"invalid prediction shard {path}")
        for token, result in shard.items():
            if token in rows:
                raise ValueError(f"duplicate PR10 prediction token: {token}")
            row = result["img_bbox"]
            required_fields = MOTION_FIELDS if motion else FIELDS
            allowed_fields = required_fields | ({"instance_ids", "cls_scores"} if motion else {"cls_scores"})
            if not required_fields.issubset(row) or set(row) - allowed_fields:
                raise ValueError(f"invalid PR10 prediction fields for {token}: {sorted(row)}")
            if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in row.values()):
                raise ValueError(f"invalid PR10 prediction tensor for {token}")
            boxes = row["boxes_3d"]
            aligned_detection_fields = ("scores_3d", "labels_3d") + (() if motion else ("instance_ids",))
            if (boxes.ndim != 2 or boxes.shape[1] < 9 or
                    any(row[k].shape != (len(boxes),) for k in aligned_detection_fields) or
                    ("instance_ids" in row and row["instance_ids"].shape != (len(boxes),)) or
                    ("cls_scores" in row and row["cls_scores"].shape != (len(boxes),))):
                raise ValueError(f"invalid PR10 detection or tracking shape for {token}")
            if (row["labels_3d"].dtype.is_floating_point or row["labels_3d"].dtype.is_complex
                    or row["labels_3d"].dtype == torch.bool
                    or ("instance_ids" in row and (row["instance_ids"].dtype.is_floating_point
                         or row["instance_ids"].dtype.is_complex or row["instance_ids"].dtype == torch.bool))
                    or (row["labels_3d"].numel() and
                        (int(row["labels_3d"].min()) < 0 or int(row["labels_3d"].max()) >= len(det_classes)))):
                raise ValueError(f"invalid PR10 detection class or tracking ID dtype/value for {token}")
            if motion:
                if (row["trajs_3d"].shape != (len(boxes), 6, 12, 2)
                        or row["trajs_score"].shape != (len(boxes), 6)):
                    raise ValueError(f"PR10 Stage3 motion must be N×6×12×2 with six mode scores for {token}")
            else:
                vectors = row["vectors"]
                # Keep the original empty-map form `(0,)`; PR10's map writer accepts an empty list.
                if (vectors.numel() and (vectors.ndim != 3 or vectors.shape[-1] != 2)) or any(
                        row[k].shape != (len(vectors),) for k in ("scores", "labels")):
                    raise ValueError(f"invalid PR10 map shape for {token}")
                if (row["labels"].dtype.is_floating_point or row["labels"].dtype.is_complex
                        or row["labels"].dtype == torch.bool or (row["labels"].numel() and
                        (int(row["labels"].min()) < 0 or int(row["labels"].max()) >= len(map_classes)))):
                    raise ValueError(f"invalid PR10 map class ID for {token}")
                if row["final_planning"].shape != (6, 2) or row["ego_status_pred"].shape != (10,):
                    raise ValueError(f"invalid PR10 planning or ego-status shape for {token}")
            row = {key: (value.float() if value.dtype.is_floating_point else value)
                   for key, value in row.items()}
            if not motion:
                # The original map decoder returns NumPy values, which the PR10 JSON writer accepts.
                row.update(vectors=[v.float().numpy() for v in vectors], scores=row["scores"].float().numpy(),
                           labels=row["labels"].numpy())
            rows[token] = {"img_bbox": row}
    expected = set(tokens)
    if set(rows) != expected:
        raise ValueError(f"PR10 val coverage mismatch: {len(expected - rows.keys())} missing, "
                         f"{len(rows.keys() - expected)} extra prediction tokens")
    return [rows[token] for token in tokens]


def validate_metric_coverage(metrics, det_classes=PR10_DET_CLASSES, map_classes=PR10_MAP_CLASSES):
    """Require every scorer family and the required aggregate/class summaries, not one sentinel per family."""
    required = {
        "img_bbox_NuScenes/NDS", "img_bbox_NuScenes/mAP", "img_bbox_NuScenes/mATE",
        "img_bbox_NuScenes/mASE", "img_bbox_NuScenes/mAOE", "img_bbox_NuScenes/mAVE",
        "img_bbox_NuScenes/mAAE", "mAP_normal", "L2", "obj_col", "obj_box_col",
        "ego_status/count", "ego_status/dim", "ego_status/mae", "ego_status/rmse",
        "ego_status/max_abs",
    }
    required.update(f"img_bbox_NuScenes/{key}" for key in TRACKING_METRICS)
    required.update(map_classes)
    required.update(PR10_NA_METRICS)
    required.update(f"ego_status/{name}_{suffix}" for name in STATUS_NAMES for suffix in ("mae", "rmse"))
    missing = sorted(required - set(metrics))
    missing_classes = [name for name in det_classes if not any(
        key.startswith(f"img_bbox_NuScenes/{name}_AP_dist_") for key in metrics)]
    if missing or missing_classes:
        details = []
        if missing:
            details.append(f"missing metrics {missing}")
        if missing_classes:
            details.append(f"missing detection AP classes {missing_classes}")
        raise ValueError("incomplete PR10 metric coverage: " + "; ".join(details))
    for key, value in metrics.items():
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite and not _is_allowed_na(key, value):
            raise ValueError(f"nonfinite or nonnumeric PR10 metric {key}: {value!r}")


MOTION_CLASSES = ("car", "pedestrian")
MOTION_TP_METRICS = ("min_ade_err", "min_fde_err", "miss_rate_err")
MOTION_METRIC_KEYS = tuple(f"{name}_{metric}" for name in MOTION_CLASSES for metric in ("EPA", *MOTION_TP_METRICS))


def validate_motion_metric_coverage(metrics, classes=None):
    def finite(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    if set(metrics) != set(MOTION_METRIC_KEYS):
        raise ValueError("incomplete PR10 Stage3 motion metrics")
    if classes is None:  # Older numeric-only result files have no per-class coverage.
        if not all(finite(value) for value in metrics.values()):
            raise ValueError("undefined motion metrics require per-class coverage")
        return
    if set(classes) != set(MOTION_CLASSES):
        raise ValueError("incomplete PR10 Stage3 motion class coverage")
    for name in MOTION_CLASSES:
        row = classes[name]
        gt, pred, status = row["ground_truth"], row["predictions"], row["status"]
        if any(type(count) is not int or count < 0 for count in (gt, pred)):
            raise ValueError(f"invalid motion counts for {name}: {row}")
        allowed = ({"no_ground_truth"} if gt == 0 else {"no_predictions"} if pred == 0 else
                   {"scored", "below_min_recall", "no_matches"})
        if status not in allowed:
            raise ValueError(f"motion status contradicts counts for {name}: {row}")
        for metric in ("EPA", *MOTION_TP_METRICS):
            value = metrics[f"{name}_{metric}"]
            if status == "no_ground_truth" or (status in ("no_predictions", "no_matches") and metric != "EPA"):
                if value is not None:
                    raise ValueError(f"undefined {name}_{metric} must be null")
            elif not finite(value):
                raise ValueError(f"invalid {name}_{metric}: {value!r}")
        if status in ("no_predictions", "no_matches") and metrics[f"{name}_EPA"] != -0.5 * pred / gt:
            raise ValueError(f"incorrect unmatched EPA for {name}")


def _score_motion_classes(gt_boxes, pred_boxes, config):
    from nuscenes.eval.detection.algo import calc_tp
    from projects.mmdet3d_plugin.datasets.evaluation.motion.motion_utils import MotionMetricData, accumulate

    metrics, classes = {}, {}
    if config.dist_th_tp != 2.0:
        raise ValueError("PR10 motion requires 2 m center matching")
    for name in MOTION_CLASSES:
        gt = sum(box.detection_name == name for box in gt_boxes.all)
        pred = sum(box.detection_name == name for box in pred_boxes.all)
        row = classes[name] = {"ground_truth": gt, "predictions": pred, "max_recall_bin": None}
        errors = dict.fromkeys(MOTION_TP_METRICS)
        if not gt:
            row["status"], epa = "no_ground_truth", None
        elif not pred:
            row["status"], epa = "no_predictions", 0.0
        else:
            result = accumulate(gt_boxes, pred_boxes, name, config.dist_fcn_callable, config.dist_th_tp)
            if isinstance(result, MotionMetricData):
                # PR10's empty-match branch returns only metric data. All eligible boxes are false positives.
                row["status"], epa = "no_matches", -0.5 * pred / gt
            else:
                data, _, epa = result
                row["max_recall_bin"] = int(data.max_recall_ind)
                first_bin = round(100 * config.min_recall) + 1
                row["status"] = "below_min_recall" if data.max_recall_ind < first_bin else "scored"
                errors = {metric: calc_tp(data, config.min_recall, metric) for metric in MOTION_TP_METRICS}
        metrics[f"{name}_EPA"] = epa
        metrics.update({f"{name}_{metric}": value for metric, value in errors.items()})
    return metrics, classes


def score_motion(dataset, rows, output, split, threshold):
    import mmcv
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from projects.mmdet3d_plugin.datasets.evaluation.motion.motion_utils import MotionBox, load_gt, load_prediction

    mmcv.dump(rows, str(output / ("results_mini.pkl" if split == "mini_val" else "results.pkl")))
    submission = dataset.format_motion_results(rows, jsonfile_prefix=str(output), thresh=threshold)
    nusc, config = dataset.nusc, dataset.det3d_eval_configs
    pred, _ = load_prediction(submission, config.max_boxes_per_sample, MotionBox)
    gt = load_gt(nusc, split, MotionBox, verbose=False, seconds=6)
    if set(pred.sample_tokens) != set(gt.sample_tokens):
        raise ValueError("motion prediction tokens differ from ground truth")

    def filtered(boxes):
        # The devkit infers box type from the first box, which does not exist in an entirely empty split.
        if not boxes.all:
            return boxes
        return filter_eval_boxes(nusc, add_center_dist(nusc, boxes), config.class_range, verbose=False)

    return _score_motion_classes(filtered(gt), filtered(pred), config)


def json_safe_metrics(metrics):
    """Preserve only the pinned devkit's known N/A values as JSON null; reject other NaN/Inf above."""
    return {key: (None if _is_allowed_na(key, value) else value)
            for key, value in metrics.items()}


def loose_segmentation_coverage(occupancy, tokens, get_occupancy):
    """Count eligible PR10 loose-planning tokens without changing its skip-missing-token policy."""
    if occupancy is None:
        raise ValueError("PR10 loose planning could not load the requested segmentation PKL")
    valid, missing, malformed = 0, [], []
    for token in tokens:
        value = get_occupancy(occupancy, token)
        if value is None:
            missing.append(token)
        elif tuple(value.shape) != (1, 6, 200, 200):
            malformed.append(token)
        else:
            valid += 1
    if not valid:
        raise ValueError("PR10 loose planning has no val token with a usable occupancy raster")
    return {"val_tokens": len(tokens), "loose_tokens_scored": valid,
            "loose_tokens_missing_or_invalid": len(missing) + len(malformed),
            "loose_missing_examples": missing[:10], "loose_invalid_examples": malformed[:10]}


def _check_external_pr10_root(root: Path):
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != PR10:
        raise ValueError(f"expected PR10 checkout {PR10}, got {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "nuScenes/projects/"], check=True)
    untracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "--", "nuScenes/projects/"],
        text=True,
    ).strip()
    if untracked:
        raise ValueError(f"pinned PR10 scorer tree contains untracked project files: {untracked.splitlines()[:5]}")


def _score_with_nusc_root(args, nusc: Path, reference_source: str):
    infos = args.infos.resolve()
    data = args.data_root.resolve()
    output = args.output_dir.resolve()
    predictions = args.predictions.resolve()
    segmentation = args.segmentation.resolve() if args.segmentation else None
    motion_only = bool(args.motion_only)
    version = str(args.version)
    split = eval_split_for_version(version, motion_only=motion_only)
    ann = infos / "nuscenes_infos_val.pkl"
    vad = infos / "vad_nuscenes_infos_temporal_val.pkl"
    tables = ("attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
              "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility")
    required_data = [data / version / f"{table}.json" for table in tables]
    required_inputs = [ann, *required_data]
    if not motion_only:
        required_inputs.append(vad)
        if segmentation is None:
            raise ValueError("full PR10 Stage2 scoring requires --segmentation for loose planning")
        required_inputs.append(segmentation)
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"missing PR10 scoring input: {path}")
    if not output.is_dir() or any(p.name not in {"predictions", "eval.log"} for p in output.iterdir()):
        raise ValueError("PR10 scoring requires a fresh output directory (predictions and eval.log are allowed)")

    os.chdir(nusc)
    sys.path.insert(0, str(nusc))
    os.environ.update(CUDA_VISIBLE_DEVICES="", NUSCENES_ROOT=str(nusc), NUSCENES_DATA_ROOT=str(data),
                      DATA_INFOS_ROOT=str(infos), WORK_DIR=str(output))
    from mmcv import Config
    from mmdet.datasets import build_dataset

    # Explicitly register PR10 datasets/pipelines without executing the plugin's model/DeepSpeed imports.
    planning_eval = install_pr10_scoring_imports(nusc)
    config_name = "unidrivevla_nusc_2b_stage3.py" if motion_only else "unidrivevla_nusc_2b_stage2.py"
    config = Config.fromfile(str(nusc / "projects/configs/UniDriveVLA" / config_name))
    val = config.data.val
    vad_path = str(vad) if vad.is_file() else ""
    for dataset_cfg in (val, val.eval_config):
        dataset_cfg.update(ann_file=str(ann), vad_ann_file=vad_path, data_root=str(data), version=version)
    val.update(pipeline=None, work_dir=str(output), test_mode=True)
    source_mode = dict(config.evaluation.eval_mode)
    if motion_only:
        threshold = float(source_mode.get("motion_threshhold", -1))
        if not source_mode.get("with_motion") or threshold != 0.2:
            raise ValueError(f"pinned Stage3 must enable motion at threshold 0.20, got {source_mode}")
    else:
        mode = source_mode
        if (not all(mode[key] for key in
                    ("with_det", "with_tracking", "with_map", "with_planning", "with_ego_status"))
                or mode["with_motion"]):
            raise ValueError("the pinned Stage2 config does not enable the five expected PR10 tasks")
    dataset = build_dataset(val)
    tokens = [info["token"] for info in dataset.data_infos]
    info_version = str(dataset.metadata.get("version", ""))
    dataset_version = str(getattr(dataset, "version", ""))
    devkit_version = str(getattr(dataset.nusc, "version", ""))
    if info_version != version or dataset_version != version or devkit_version != version:
        raise ValueError(f"requested {version}, but val infos/dataset/devkit versions are "
                         f"{info_version!r}/{dataset_version!r}/{devkit_version!r}")
    from nuscenes.utils.splits import create_splits_scenes
    split_scenes = set(create_splits_scenes().get(split, ()))
    if not split_scenes:
        raise ValueError(f"nuScenes devkit has no scenes for official split {split!r}")
    scene_names = {record["token"]: record["name"] for record in dataset.nusc.scene}
    if len(scene_names) != len(dataset.nusc.scene):
        raise ValueError("nuScenes devkit has duplicate scene tokens")
    missing_scene_names = split_scenes - set(scene_names.values())
    if missing_scene_names:
        raise ValueError(f"nuScenes {version} tables lack official {split} scenes: "
                         f"{sorted(missing_scene_names)[:5]}")
    official_tokens = []
    for sample in dataset.nusc.sample:
        scene_name = scene_names.get(sample["scene_token"])
        if scene_name is None:
            raise ValueError(f"nuScenes sample references an unknown scene: {sample['token']}")
        if scene_name in split_scenes:
            official_tokens.append(sample["token"])
    split_coverage = validate_split_token_coverage(tokens, official_tokens, version=version, split=split)
    if tuple(dataset.CLASSES) != PR10_DET_CLASSES or tuple(dataset.MAP_CLASSES) != PR10_MAP_CLASSES:
        raise ValueError(f"PR10 class-ID order mismatch: det={dataset.CLASSES}, map={dataset.MAP_CLASSES}")
    rows = load_results(predictions, tokens, dataset.CLASSES, dataset.MAP_CLASSES, motion=motion_only)
    if motion_only:
        metrics, classes = score_motion(dataset, rows, output, split, threshold)
        validate_motion_metric_coverage(metrics, classes)
        coverage = {
            "reference_source": reference_source,
            "classes": classes,
            "min_recall": dataset.det3d_eval_configs.min_recall,
            **split_coverage,
            "motion_tokens": len(rows),
            "detections_with_motion": sum(len(row["img_bbox"]["boxes_3d"]) for row in rows),
            "seconds": 6,
            "steps": 12,
            "detection_score_threshold": threshold,
        }
        metrics_path = output / "nuscenes_pr10_motion_metrics.json"
        coverage_path = output / "nuscenes_pr10_motion_coverage.json"
    else:
        strict_valid = sum(bool(np.asarray(info["gt_ego_fut_masks"]).reshape(-1).all())
                           for info in dataset.data_infos)
        if not strict_valid:
            raise ValueError("PR10 strict planning has no val sample with a complete future trajectory")
        coverage = dict(split_coverage)

        # Dataset.evaluate does not expose the original planning evaluator's segmentation path argument.
        default = planning_eval._DEFAULT_SEG_PKL
        original_loader = planning_eval._load_occ_map

        def checked_occupancy_loader(path, logger, label):
            occupancy = original_loader(path, logger, label)
            coverage.update(loose_segmentation_coverage(occupancy, tokens, planning_eval._get_occupancy))
            return occupancy

        planning_eval._DEFAULT_SEG_PKL = str(segmentation)
        planning_eval._load_occ_map = checked_occupancy_loader
        try:
            metrics = dataset.evaluate(rows, eval_mode=mode)
        finally:
            planning_eval._DEFAULT_SEG_PKL = default
            planning_eval._load_occ_map = original_loader
        if not coverage:
            raise ValueError("PR10 scorer did not run loose planning with the requested segmentation PKL")
        validate_metric_coverage(metrics, dataset.CLASSES, dataset.MAP_CLASSES)
        if metrics.get("ego_status/count") != len(tokens) or metrics.get("ego_status/dim") != 10:
            raise ValueError("PR10 ego-status scoring did not cover all val tokens and all ten dimensions")
        coverage.update(reference_source=reference_source, strict_valid_samples=strict_valid)
        metrics_path = output / "nuscenes_pr10_metrics.json"
        coverage_path = output / "nuscenes_pr10_coverage.json"
        metrics = json_safe_metrics(metrics)

    with metrics_path.open("x", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=lambda value: value.tolist(), allow_nan=False)
    with coverage_path.open("x", encoding="utf-8") as file:
        json.dump(coverage, file, indent=2)
    if motion_only:
        for name, row in classes.items():
            print(name, row, {key: metrics[f"{name}_{key}"] for key in ("EPA", *MOTION_TP_METRICS)})
        print(f"PR10 Stage3 motion coverage: val={coverage['val_tokens']}, "
              f"detections={coverage['detections_with_motion']}, threshold={coverage['detection_score_threshold']}, "
              f"horizon={coverage['seconds']}s/{coverage['steps']} steps")
    else:
        print(f"PR10 sample coverage: val={coverage['val_tokens']}, strict={coverage['strict_valid_samples']}, "
              f"loose={coverage['loose_tokens_scored']}, loose_missing_or_invalid="
              f"{coverage['loose_tokens_missing_or_invalid']}")


def score(args):
    if args.pr10_root is not None:
        root = args.pr10_root.resolve()
        _check_external_pr10_root(root)
        _score_with_nusc_root(args, root / "nuScenes", reference_source=f"git:{PR10}")
        return

    bundle = args.reference_bundle.resolve() if args.reference_bundle else packaged_pr10_bundle()
    with tempfile.TemporaryDirectory(prefix="unidrivevla-pr10-reference-") as temporary:
        nusc, digest = extract_pr10_bundle(bundle, Path(temporary))
        _score_with_nusc_root(args, nusc, reference_source=f"bundle:{PR10}:{digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr10-root", type=Path, help="exact PR10 git checkout (optional when the source bundle is used)")
    parser.add_argument("--reference-bundle", type=Path, help="fixed, hash-verified PR10 scorer source archive")
    parser.add_argument("--version", choices=tuple(NUSCENES_SPLITS), default="v1.0-trainval",
                        help="dataset version; Stage3 motion supports trainval or mini, Stage2 requires trainval")
    for name in ("infos", "data-root", "predictions", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--segmentation", type=Path, help="PR10 planning segmentation PKL (required for Stage2)")
    parser.add_argument("--motion-only", action="store_true", help="score Stage3 car/pedestrian motion at 0.20")
    score(parser.parse_args())


if __name__ == "__main__":
    main()
