"""Score native nuScenes predictions with PR10, without loading a model."""

import argparse
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np


PR10 = "5fce121f4bf04eaf7e109880d4f26eed62c0eed0"
PR10_DET_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
PR10_MAP_CLASSES = ("ped_crossing", "divider", "boundary")
FIELDS = {"boxes_3d", "scores_3d", "labels_3d", "instance_ids", "vectors", "scores", "labels",
          "final_planning", "ego_status_pred"}
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


def load_results(directory, tokens, det_classes=PR10_DET_CLASSES, map_classes=PR10_MAP_CLASSES):
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
            if not FIELDS.issubset(row) or set(row) - FIELDS - {"cls_scores"}:
                raise ValueError(f"invalid PR10 prediction fields for {token}: {sorted(row)}")
            if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in row.values()):
                raise ValueError(f"invalid PR10 prediction tensor for {token}")
            boxes = row["boxes_3d"]
            if (boxes.ndim != 2 or boxes.shape[1] < 9 or
                    any(row[k].shape != (len(boxes),) for k in ("scores_3d", "labels_3d", "instance_ids")) or
                    ("cls_scores" in row and row["cls_scores"].shape != (len(boxes),))):
                raise ValueError(f"invalid PR10 detection or tracking shape for {token}")
            if (row["labels_3d"].dtype.is_floating_point or row["labels_3d"].dtype.is_complex
                    or row["labels_3d"].dtype == torch.bool or row["instance_ids"].dtype.is_floating_point
                    or row["instance_ids"].dtype.is_complex or row["instance_ids"].dtype == torch.bool
                    or (row["labels_3d"].numel() and
                        (int(row["labels_3d"].min()) < 0 or int(row["labels_3d"].max()) >= len(det_classes)))):
                raise ValueError(f"invalid PR10 detection class or tracking ID dtype/value for {token}")
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
        if not finite and not (key in PR10_NA_METRICS and math.isnan(float(value))):
            raise ValueError(f"nonfinite or nonnumeric PR10 metric {key}: {value!r}")


def json_safe_metrics(metrics):
    """Preserve only the pinned devkit's known N/A values as JSON null; reject other NaN/Inf above."""
    return {key: (None if key in PR10_NA_METRICS and math.isnan(float(value)) else value)
            for key, value in metrics.items()}


def loose_segmentation_coverage(occupancy, tokens, get_occupancy):
    """Count eligible PR10 loose-planning tokens without changing its skip-missing-token policy."""
    valid, missing, malformed = 0, [], []
    for token in tokens:
        value = get_occupancy(occupancy, token)
        if value is None:
            missing.append(token)
        elif tuple(value.shape) != (1, 6, 200, 200):
            malformed.append(token)
        else:
            valid += 1
    return {"val_tokens": len(tokens), "loose_tokens_scored": valid,
            "loose_tokens_missing_or_invalid": len(missing) + len(malformed),
            "loose_missing_examples": missing[:10], "loose_invalid_examples": malformed[:10]}


def score(args):
    root = args.pr10_root.resolve()
    infos = args.infos.resolve()
    data = args.data_root.resolve()
    output = args.output_dir.resolve()
    predictions = args.predictions.resolve()
    segmentation = args.segmentation.resolve()
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
    ann = infos / "nuscenes_infos_val.pkl"
    vad = infos / "vad_nuscenes_infos_temporal_val.pkl"
    tables = ("attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
              "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility")
    required_data = [data / "v1.0-trainval" / f"{table}.json" for table in tables]
    for path in (ann, vad, segmentation, *required_data):
        if not path.is_file():
            raise FileNotFoundError(f"missing PR10 scoring input: {path}")
    if not output.is_dir() or any(p.name not in {"predictions", "eval.log"} for p in output.iterdir()):
        raise ValueError("PR10 scoring requires a fresh output directory (predictions and eval.log are allowed)")

    nusc = root / "nuScenes"
    os.chdir(nusc)
    sys.path.insert(0, str(nusc))
    os.environ.update(CUDA_VISIBLE_DEVICES="", NUSCENES_ROOT=str(nusc), NUSCENES_DATA_ROOT=str(data),
                      DATA_INFOS_ROOT=str(infos), WORK_DIR=str(output))
    from mmcv import Config
    from mmdet.datasets import build_dataset

    # Explicitly register the scorer dataset/pipelines without executing PR10's eager model/DeepSpeed imports.
    planning_eval = install_pr10_scoring_imports(nusc)

    config = Config.fromfile(str(nusc / "projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage2.py"))
    val = config.data.val
    for dataset in (val, val.eval_config):
        dataset.update(ann_file=str(ann), vad_ann_file=str(vad), data_root=str(data))
    val.update(pipeline=None, work_dir=str(output), test_mode=True)
    mode = dict(config.evaluation.eval_mode)
    if not all(mode[key] for key in ("with_det", "with_tracking", "with_map", "with_planning", "with_ego_status")) or mode["with_motion"]:
        raise ValueError("the pinned evaluation config does not enable the five expected PR10 tasks")
    dataset = build_dataset(val)
    tokens = [info["token"] for info in dataset.data_infos]
    if tuple(dataset.CLASSES) != PR10_DET_CLASSES or tuple(dataset.MAP_CLASSES) != PR10_MAP_CLASSES:
        raise ValueError(f"PR10 class-ID order mismatch: det={dataset.CLASSES}, map={dataset.MAP_CLASSES}")
    rows = load_results(predictions, tokens, dataset.CLASSES, dataset.MAP_CLASSES)
    strict_valid = sum(bool(np.asarray(info["gt_ego_fut_masks"]).reshape(-1).all())
                       for info in dataset.data_infos)
    coverage = {}

    # Dataset.evaluate does not expose the original planning evaluator's segmentation path argument.
    default = planning_eval._DEFAULT_SEG_PKL
    original_loader = planning_eval._load_occ_map
    def checked_occupancy_loader(path, logger, label):
        occupancy = original_loader(path, logger, label)
        if occupancy is None:
            coverage.update({"val_tokens": len(tokens), "loose_tokens_scored": 0,
                             "loose_tokens_missing_or_invalid": len(tokens)})
            return occupancy
        coverage.update(loose_segmentation_coverage(occupancy, tokens, planning_eval._get_occupancy))
        return occupancy

    planning_eval._DEFAULT_SEG_PKL = str(segmentation)
    planning_eval._load_occ_map = checked_occupancy_loader
    try:
        metrics = dataset.evaluate(rows, eval_mode=mode)
    finally:
        planning_eval._DEFAULT_SEG_PKL = default
        planning_eval._load_occ_map = original_loader
    validate_metric_coverage(metrics, dataset.CLASSES, dataset.MAP_CLASSES)
    if metrics.get("ego_status/count") != len(tokens) or metrics.get("ego_status/dim") != 10:
        raise ValueError("PR10 ego-status scoring did not cover all val tokens and all ten dimensions")
    with (output / "nuscenes_pr10_metrics.json").open("x", encoding="utf-8") as file:
        json.dump(json_safe_metrics(metrics), file, indent=2, default=lambda value: value.tolist())
    coverage["strict_valid_samples"] = strict_valid
    with (output / "nuscenes_pr10_coverage.json").open("x", encoding="utf-8") as file:
        json.dump(coverage, file, indent=2)
    print(f"PR10 sample coverage: val={coverage['val_tokens']}, strict={strict_valid}, "
          f"loose={coverage['loose_tokens_scored']}, loose_missing_or_invalid="
          f"{coverage['loose_tokens_missing_or_invalid']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pr10-root", "infos", "data-root", "segmentation", "predictions", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
