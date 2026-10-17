"""CPU oracle comparing new planning scores to the active main helper classes."""

import importlib.util
import ast
import contextlib
import copy
import hashlib
import io
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def package(name, path=None):
    mod = types.ModuleType(name)
    mod.__path__ = [] if path is None else [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main_b2d_aggregate(path):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Bench2DriveDataset")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    scope = {"copy": copy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["evaluate"]


def aggregate_b2d(evaluate, values):
    aggregate = {}

    class Captured(dict):
        def __deepcopy__(self, memo):
            return Captured(self)

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            aggregate[key] = value

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        results = [{"metric_results": Captured(value) if value.get("fut_valid_flag") else value}
                   for value in values]
        evaluate(object(), results,
                 eval_mode={"with_det": False, "with_map": False, "with_motion": False})
    return aggregate


def main_helpers():
    mmcv = package("mmcv")
    mmcv_utils = package("mmcv.utils")
    mmcv_utils.print_log = lambda *args, **kwargs: None
    mmcv.utils = mmcv_utils
    mmdet = package("mmdet")
    datasets = package("mmdet.datasets")
    datasets.build_dataset = datasets.build_dataloader = lambda *args, **kwargs: None
    mmdet.datasets = datasets
    if importlib.util.find_spec("matplotlib") is None:
        package("matplotlib")
        package("matplotlib.pyplot")

    package("projects", ROOT / "nuScenes" / "projects")
    package("projects.mmdet3d_plugin", ROOT / "nuScenes" / "projects" / "mmdet3d_plugin")
    package("projects.mmdet3d_plugin.core", ROOT / "nuScenes" / "projects" / "mmdet3d_plugin" / "core")
    box3d = package("projects.mmdet3d_plugin.core.box3d")
    for name, value in {"X": 0, "Y": 1, "Z": 2, "W": 3, "L": 4, "H": 5, "YAW": 6}.items():
        setattr(box3d, name, value)
    package("projects.mmdet3d_plugin.datasets", ROOT / "nuScenes" / "projects" / "mmdet3d_plugin" / "datasets")
    nusc_utils_path = Path(os.environ.get(
        "UNI_MAIN_NUSC_UTILS", ROOT / "nuScenes" / "projects" / "mmdet3d_plugin" / "datasets" / "utils.py"))
    nusc_path = Path(os.environ.get(
        "UNI_MAIN_NUSC_PLANNING", ROOT / "nuScenes" / "projects" / "mmdet3d_plugin" /
        "datasets" / "evaluation" / "planning" / "planning_eval.py"))
    load("projects.mmdet3d_plugin.datasets.utils", nusc_utils_path)
    nusc = load("main_nusc_planning", nusc_path)

    package("nuscenes")
    package("nuscenes.utils")
    classes = package("nuscenes.utils.data_classes")
    classes.Box = type("Box", (), {})
    b2d_path = Path(os.environ.get(
        "UNI_MAIN_B2D_METRIC", ROOT / "Bench2Drive" / "projects" / "mmdet3d_plugin" /
        "datasets" / "evaluation" / "planning" / "metric_stp3.py"))
    b2d_dataset_path = Path(os.environ.get(
        "UNI_MAIN_B2D_DATASET", ROOT / "Bench2Drive" / "projects" / "mmdet3d_plugin" / "datasets" /
        "bench2drive_dataset.py"))
    b2d = load("main_b2d_planning", b2d_path)
    b2d_aggregate = main_b2d_aggregate(b2d_dataset_path)
    for name, path in (("nuScenes", nusc_path), ("Bench2Drive metric", b2d_path),
                       ("Bench2Drive aggregator", b2d_dataset_path)):
        print(f"main oracle source {name}: {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    return nusc, b2d, b2d_aggregate


def close(actual, expected, label, atol=1e-6):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if not np.allclose(a, b, atol=atol, rtol=1e-6, equal_nan=True):
        if a.shape == b.shape and a.size:
            index = np.unravel_index(np.nanargmax(np.abs(a - b)), a.shape)
            raise AssertionError(f"{label}: maxdiff={abs(a[index] - b[index]):.6g} at {index}; "
                                 f"new={a[index]!r}, main={b[index]!r}, shape={a.shape}")
        raise AssertionError(f"{label}: shapes differ, new={a.shape}, main={b.shape}")


def strict_case(pred, gt, mask, boxes, token):
    return token, pred.astype(np.float32), {
        "trajectory": gt.astype(np.float32), "mask": mask.astype(np.float32),
        "axes": np.array([[0.84, -0.13], [0.11, 0.93]], np.float32),
        "strict_boxes": boxes,
    }


def nusc_checks(main):
    from unidrivevla.eval.planning_metrics import _load_occupancy, planning_metrics

    gt = np.array([[0, 0], [.5, .1], [1, .4], [1.5, .9], [2, 1.6], [2.5, 2.5]], np.float32)
    pred = gt.copy()
    pred[2:, 1] += 3
    raw_gt = np.linalg.solve(np.array([[.84, -.13], [.11, .93]]), gt.T).T
    raw_pred = np.linalg.solve(np.array([[.84, -.13], [.11, .93]]), pred.T).T
    boxes = [np.zeros((0, 7), np.float32) for _ in range(6)]
    boxes[0] = np.array([[raw_gt[0, 0], raw_gt[0, 1] + .5, 0, .6, .6, 1.5, .0]], np.float32)
    boxes[2] = np.array([[raw_pred[2, 0], raw_pred[2, 1], 0, .6, .6, 1.5, .0]], np.float32)
    mask_full = np.ones(6, np.float32)
    complete = strict_case(pred, gt, mask_full, boxes, "complete")
    partial = strict_case(pred + 1, gt, np.array([1, 1, 1, 1, 0, 0], np.float32), boxes, "partial")
    records = [complete, partial]
    ours = planning_metrics(records, "nuscenes")
    reference = main.PlanningMetric()
    reference.update(torch.tensor(raw_pred[None]), torch.tensor(raw_gt[None]),
                     torch.ones((1, 6, 2)), [torch.tensor(x[None]) for x in boxes])
    expected = reference.compute()
    close(ours["strict_samples"], 1, "strict complete-future denominator")
    close(float(expected["obj_col"][0]), 1, "GT collision present for masking test")
    close(float(expected["obj_box_col"][0]), 0, "predicted GT collision masked")
    close(float(expected["obj_box_col"][2]), 1, "predicted-only collision present")
    for i in range(6):
        key = f"nuscenes/strict/{(i + 1) / 2:g}s"
        close(ours[f"{key}/L2"], expected["L2"][i], f"strict L2 step {i}")
        close(ours[f"{key}/obj_col"], expected["obj_col"][i], f"strict GT collision step {i}")
        close(ours[f"{key}/obj_box_col"], expected["obj_box_col"][i], f"strict prediction collision step {i}")

    raw_occ = np.zeros((1, 7, 200, 200), np.uint8)
    processed = np.zeros((6, 200, 200), np.uint8)
    processed[1, 100, 100] = 1
    raw_occ[0, 2:] = np.flip(processed[1:], axis=-1)
    occupancy = _load_occupancy(raw_occ)
    records = [("complete", pred, complete[2]), ("partial", pred + 1, partial[2])]
    loose = planning_metrics(records, "nuscenes", {"complete": occupancy, "partial": occupancy})
    loose_main = main.PlanningMetricLoose()
    for token, prediction, info in records:
        current_gt = np.linalg.solve(np.array([[.84, -.13], [.11, .93]]), info["trajectory"].T).T
        current_pred = np.linalg.solve(np.array([[.84, -.13], [.11, .93]]), prediction.T).T
        valid_mask = np.repeat(info["mask"][:, None], 2, axis=1)
        loose_main.update(torch.tensor(current_pred[None]), torch.tensor(current_gt[None]),
                          torch.tensor(valid_mask[None]), torch.tensor(occupancy[None]))
    expected_loose = loose_main.compute()
    close(loose["loose_samples"], 2, "loose occupancy denominator includes incomplete future")
    for i in range(6):
        key = f"nuscenes/loose/{(i + 1) / 2:g}s"
        close(loose[f"{key}/L2"], expected_loose["L2"][i], f"loose L2 step {i}")
        close(loose[f"{key}/obj_col"], expected_loose["obj_col"][i], f"loose GT collision step {i}")
        close(loose[f"{key}/obj_box_col"], expected_loose["obj_box_col"][i], f"loose prediction collision step {i}")


def b2d_checks(main, main_aggregate):
    from unidrivevla.eval.planning_metrics import _b2d_occupancy, planning_metrics

    gt = np.array([[0, 0], [.5, .1], [1, .4], [1.5, .9], [2, 1.6], [2.5, 2.5]], np.float32)
    pred = gt.copy()
    pred[2:, 1] += 3
    raw_boxes = np.array([[0, 0, 0, 1.8, 4.2, 1.5, 0.2, 0, 0],
                          [7, 3, 0, .6, .8, 1.6, -.5, 0, 0]], np.float32)
    main_boxes = raw_boxes.copy()
    main_boxes[:, [3, 4]] = main_boxes[:, [4, 3]]
    main_boxes[:, 6] = -(main_boxes[:, 6] + np.pi / 2)
    attrs = np.zeros((2, 34), np.float32)
    attrs[:, 12:18] = 1
    attrs[:, :12] = np.tile([0, .2], 6)
    attrs[:, 27] = [2, 0]  # active main's class-index mismatch is intentional: class 2 rasterizes as human.
    attrs[:, 28:34] = .03
    info = {"trajectory": gt, "mask": np.ones(6, np.float32),
            "axes": np.array([[.89, -.16], [.09, .96]], np.float32),
            "planning_boxes": main_boxes, "planning_attr": attrs}

    metric = main.PlanningMetric()
    gt_agent_boxes = torch.tensor(main_boxes.copy())
    gt_agent_boxes[:, [3, 4]] = gt_agent_boxes[:, [4, 3]]
    gt_agent_boxes[:, 6] = -gt_agent_boxes[:, 6] - np.pi / 2
    seg, ped = metric.get_label(gt_agent_boxes, torch.tensor(attrs[None]))
    expected_occ = torch.logical_or(seg, ped)
    ours_occ = _b2d_occupancy(main_boxes, attrs)
    if not np.array_equal(ours_occ, expected_occ.numpy()[0]):
        raise AssertionError("Bench2Drive occupancy differs from main metric_stp3.get_label")
    if not expected_occ.any():
        raise AssertionError("class id 2 did not exercise main's literal class mask")

    partial = dict(info, mask=np.array([1, 1, 1, 1, 1, 0], np.float32))
    records = [("valid", pred, info), ("partial", pred + 1, partial)]
    ours = planning_metrics(records, "b2d")
    pred_lidar = np.linalg.solve(info["axes"], pred.T).T
    gt_lidar = np.linalg.solve(info["axes"], gt.T).T
    expected = {key: 0.0 for key in (f"plan_{metric_name}_{sec}s"
                                     for metric_name in ("L2", "obj_col", "obj_box_col") for sec in (1, 2, 3))}
    for sec in (1, 2, 3):
        steps = sec * 2
        expected[f"plan_L2_{sec}s"] += metric.compute_L2(torch.tensor(pred_lidar[:steps]), torch.tensor(gt_lidar[:steps]))
        obj, box = metric.evaluate_coll(torch.tensor(pred_lidar[None, :steps]), torch.tensor(gt_lidar[None, :steps]),
                                        expected_occ[:, :steps])
        expected[f"plan_obj_col_{sec}s"] += float(obj.mean())
        expected[f"plan_obj_box_col_{sec}s"] += float(box.mean())
    valid_result = {"fut_valid_flag": True, **expected}
    invalid_result = {"fut_valid_flag": False, **{key: 0.0 for key in expected}}
    main_means = aggregate_b2d(main_aggregate, [valid_result, invalid_result])
    for key in expected:
        close(ours[f"b2d/{key}"], main_means[key], f"B2D main aggregation {key}", atol=5.1e-5)
    close(ours["valid_samples"], 1, "B2D complete-future counter")
    no_valid = planning_metrics([("partial", pred, partial)], "b2d")
    close(no_valid["valid_samples"], 0, "B2D zero-valid count")
    if any(key.startswith("b2d/") for key in no_valid):
        raise AssertionError("B2D zero-valid input must not emit a placeholder metric")
    try:
        aggregate_b2d(main_aggregate, [invalid_result])
    except TypeError:
        pass  # active main's inline aggregator has no zero-valid guard (metric_dict stays None)
    else:
        raise AssertionError("main B2D zero-valid aggregation behavior changed")


def b2d_real_sample_check():
    info_dir = Path(os.environ.get("B2D_INFOS", ROOT / "Bench2Drive" / "data" / "infos"))
    info_path = info_dir / "b2d_infos_val.pkl"
    if not info_path.is_file():
        print(f"real B2D sample oracle skipped: {info_path} missing")
        return

    package("unidrivevla.data", ROOT / "unidrivevla" / "data")
    registry = package("unidrivevla.data.registry")
    registry.register = lambda _name: (lambda fn: fn)
    load("unidrivevla.data.sample", ROOT / "unidrivevla" / "data" / "sample.py")
    source_module = load("unidrivevla.data.bench2drive", ROOT / "unidrivevla" / "data" / "bench2drive.py")
    with open(info_path, "rb") as stream:
        infos = pickle.load(stream)
    source = source_module.Bench2DriveSource.__new__(source_module.Bench2DriveSource)
    source.infos = infos
    source.det_classes = list(source_module.B2D_DET_CLASSES)
    source.index = {(row["folder"], int(row["frame_idx"])): i for i, row in enumerate(infos)}
    source.tokens = [f"{row['folder']}/{int(row['frame_idx']):05d}" for row in infos]
    source.by_token = {token: i for i, token in enumerate(source.tokens)}
    groups = [list(range(i, len(infos), 5)) for i in range(5)]
    source.ordered = [i for group in groups for i in group]
    lengths = [len(group) for group in groups]
    source.group_cumsum = np.cumsum(np.array([0, *lengths[:-1]], dtype=np.int64))
    source.reordered_of = np.empty(len(infos), np.int64)
    for reordered, original in enumerate(source.ordered):
        source.reordered_of[original] = reordered

    dataset_path = Path(os.environ.get(
        "UNI_MAIN_B2D_DATASET", ROOT / "Bench2Drive" / "projects" / "mmdet3d_plugin" /
        "datasets" / "bench2drive_dataset.py"))
    tree = ast.parse(dataset_path.read_text())
    dataset_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Bench2DriveDataset")
    method = next(node for node in dataset_cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "get_box_attr_labels")
    mapping_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "NameMapping" for target in node.targets))
    main_name_mapping = eval(compile(ast.Expression(mapping_node.value), str(dataset_path), "eval"),
                             {"__builtins__": {}, "dict": dict, "range": range})
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(dataset_path), "exec"), scope)
    main_get_attrs = scope["get_box_attr_labels"]
    reference_infos = []
    for i in source.ordered:
        row = dict(infos[i])
        row["gt_names"] = np.asarray([main_name_mapping.get(str(name), str(name)) for name in row["gt_names"]])
        reference_infos.append(row)
    reference = types.SimpleNamespace(data_infos=reference_infos, sample_rate=1,
                                      future_frames=6, det_classes=source.det_classes)

    for index, info in enumerate(infos):
        actual = source.eval_info(source.tokens[index])
        if not len(actual["planning_boxes"]):
            continue
        boxes = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 9)
        names = [main_name_mapping.get(str(name), str(name)) for name in np.asarray(info["gt_names"])]
        labels = np.array([source.det_classes.index(name) if name in source.det_classes else -1 for name in names])
        x0, y0, x1, y1 = source_module.LIDAR_RANGE
        keep = ((np.asarray(info["num_points"]) != 0) & (labels >= 0)
                & (boxes[:, 0] > x0) & (boxes[:, 1] > y0)
                & (boxes[:, 0] < x1) & (boxes[:, 1] < y1))
        main_boxes = boxes.copy()
        main_boxes[:, [3, 4]] = main_boxes[:, [4, 3]]
        main_boxes[:, 6] = -(main_boxes[:, 6] + np.pi / 2)
        expected_attrs = main_get_attrs(reference, int(source.reordered_of[index]), 6)[keep]
        got_boxes, got_attrs = actual["planning_boxes"], actual["planning_attr"]
        if got_boxes.shape != main_boxes[keep].shape or got_attrs.shape != expected_attrs.shape:
            raise AssertionError(f"real B2D {source.tokens[index]} planning target shape differs from main")
        close(got_boxes[:, :6], main_boxes[keep, :6], "real B2D filtered actor boxes")
        yaw_error = np.abs((got_boxes[:, 6] - main_boxes[keep, 6] + np.pi) % (2 * np.pi) - np.pi)
        close(yaw_error, np.zeros_like(yaw_error), "real B2D filtered actor yaw")
        close(got_attrs, expected_attrs, "real B2D get_box_attr_labels")
        print(f"real B2D sample oracle: {source.tokens[index]} actors={len(got_boxes)} attr=maxdiff={np.max(np.abs(got_attrs - expected_attrs)):.2e}")
        return
    print("real B2D sample oracle skipped: no validation sample had an actor in the official test range")


def main_check():
    nusc, b2d, b2d_aggregate = main_helpers()
    nusc_checks(nusc)
    b2d_checks(b2d, b2d_aggregate)
    b2d_real_sample_check()
    print("planning metric oracle: nuScenes strict/loose and B2D match active main helpers")


if __name__ == "__main__":
    main_check()
