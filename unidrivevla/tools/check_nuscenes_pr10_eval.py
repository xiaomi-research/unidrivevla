"""CPU contract check for the optional full PR10 nuScenes evaluator."""

from contextlib import nullcontext
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import tarfile
import types
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unidrivevla.eval import EVALUATORS, build_evaluators
from unidrivevla.eval.nuscenes_pr10 import _PredictionOnlyNuScenesSource, _append_prediction_rows
from unidrivevla.adapters.unidrivevla import Profile, UniDriveVLAPolicy
from unidrivevla.tools.score_nuscenes_pr10 import (
    PR10,
    PR10_BUNDLE_SHA256,
    PR10_DET_CLASSES,
    PR10_MAP_CLASSES,
    PR10_NA_METRICS,
    MOTION_METRIC_KEYS,
    STATUS_NAMES,
    TRACKING_METRICS,
    eval_split_for_version,
    extract_pr10_bundle,
    install_pr10_scoring_imports,
    json_safe_metrics,
    load_results,
    loose_segmentation_coverage,
    packaged_pr10_bundle,
    validate_metric_coverage,
    validate_split_token_coverage,
)
from unidrivevla.train.trainer import Trainer


METRIC_GROUP_KEYS = (
    "img_bbox_NuScenes/NDS",  # detection
    "img_bbox_NuScenes/amota",  # tracking
    "mAP_normal",  # map
    "L2",  # planning
    "ego_status/mae",  # ego status
)


class FakeContext:
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cpu")
        self.is_main = rank == 0


class FakeSource(torch.utils.data.Dataset):
    tokens = ["scene-a-0", "scene-a-1", "scene-b-0", "scene-c-0", "scene-c-1", "scene-c-2"]

    def __init__(self):
        self.target_calls = 0

    def __len__(self):
        return len(self.tokens)

    def sequences(self):
        return [[0, 1], [2], [3, 4, 5]]

    def __getitem__(self, index):
        return self.features(index), self.targets(index), self.tokens[index]

    def features(self, index):
        return {"img_metas": {"sample_index": index}}

    def targets(self, index):
        self.target_calls += 1
        return {"unused_target_payload": index}


class FakeModel:
    def __init__(self):
        self.stream_ids = []
        self.sample_indices = []
        self.calls = 0

    def __call__(self, features):
        if features.get("return_nuscenes_pr10") is not True:
            raise AssertionError("PR10 evaluator did not request its full prediction fields")
        metas = features.get("img_metas")
        if not isinstance(metas, list) or not metas or any("stream_id" not in item for item in metas):
            raise AssertionError("scene stream IDs were not passed to the policy")
        self.stream_ids.extend(item["stream_id"] for item in metas)
        self.sample_indices.extend(item["sample_index"] for item in metas)
        self.calls += 1
        row = {
            "boxes_3d": torch.zeros((1, 9), dtype=torch.float32),
            "scores_3d": torch.tensor([0.9], dtype=torch.float32),
            "labels_3d": torch.tensor([0], dtype=torch.long),
            "instance_ids": torch.tensor([1], dtype=torch.long),
            "vectors": torch.zeros((3, 20, 2), dtype=torch.float32),
            "scores": torch.zeros((3,), dtype=torch.float32),
            "labels": torch.zeros((3,), dtype=torch.long),
            "final_planning": torch.zeros((6, 2), dtype=torch.float32),
            "ego_status_pred": torch.zeros((10,), dtype=torch.float32),
        }
        return {"nuscenes_pr10": [row for _ in metas]}


class FakeMotionModel(FakeModel):
    def __call__(self, features):
        if features.get("return_nuscenes_pr10_motion") is not True:
            raise AssertionError("Stage3 evaluator did not request motion outputs")
        metas = features["img_metas"]
        self.stream_ids.extend(item["stream_id"] for item in metas)
        self.sample_indices.extend(item["sample_index"] for item in metas)
        self.calls += 1
        row = {
            "boxes_3d": torch.zeros((1, 9), dtype=torch.float32),
            "scores_3d": torch.tensor([0.9], dtype=torch.float32),
            "labels_3d": torch.tensor([0], dtype=torch.long),
            "trajs_3d": torch.zeros((1, 6, 12, 2), dtype=torch.float32),
            "trajs_score": torch.full((1, 6), 1 / 6, dtype=torch.float32),
        }
        return {"nuscenes_pr10_motion": [row for _ in metas]}


class ReportSink:
    def __init__(self):
        self.ctx = FakeContext()
        self.aborted = False

    def abort_evaluations(self):
        self.aborted = True


def _build_evaluator(root: Path, name="nuscenes_pr10", *, motion=False, include_segmentation=True,
                     external_root=True, version="v1.0-trainval"):
    pr10_root = root / "pr10"
    info_dir = root / "infos"
    data_root = root / "data"
    for path in (pr10_root, info_dir, data_root):
        path.mkdir(parents=True, exist_ok=True)
    segmentation = root / "planing_gt_segmentation_val"
    if include_segmentation:
        segmentation.write_bytes(b"fixture; the scorer subprocess is mocked")
    env = {"NUSCENES_PR10_PYTHON": sys.executable,
           "NUSCENES_PR10_ROOT": str(pr10_root) if external_root else "",
           "NUSCENES_PR10_BUNDLE": ""}
    cfg = OmegaConf.create({
        "data_loader": "nuscenes",
        "agent": {"with_motion": motion},
        "nuscenes": {"info_dir": str(info_dir), "data_root": str(data_root), "version": version},
        "evaluator": {
            "name": name,
            "every_n_epochs": 3,
            "on_end": True,
            **({"seg_pkl_path": str(segmentation)} if include_segmentation else {}),
        },
        "dataloader": {"eval_batch_size": 1, "params": {"batch_size": 4, "num_workers": 0,
                                                           "pin_memory": False}},
    })
    with patch.dict(os.environ, env, clear=False):
        evaluator = build_evaluators(cfg, root / "run")[0]
    return evaluator


def _check_registered_final_only_mode(root: Path):
    if "nuscenes_pr10" not in EVALUATORS:
        raise AssertionError("nuscenes_pr10 evaluator is not registered through unidrivevla.eval")
    evaluator = _build_evaluator(root)
    if type(evaluator) is not EVALUATORS["nuscenes_pr10"]:
        raise AssertionError("evaluator registry did not construct the PR10 evaluator")
    if evaluator.every_n_epochs != 0 or not evaluator.on_end:
        raise AssertionError("PR10 full metrics must run once at end, not on every training epoch")

    regular = _build_evaluator(root / "regular", name="nuscenes")
    if type(regular) is EVALUATORS["nuscenes_pr10"]:
        raise AssertionError("the default nuScenes evaluator unexpectedly selected PR10 full metrics")
    motion = _build_evaluator(root / "motion", name="nuscenes_pr10_motion", motion=True,
                              include_segmentation=False, external_root=False, version="v1.0-mini")
    if type(motion) is not EVALUATORS["nuscenes_pr10_motion"] or not motion.motion_only:
        raise AssertionError("Stage3 motion evaluator was not registered separately")
    if (motion.every_n_epochs != 0 or not motion.on_end or motion.segmentation_path is not None
            or motion.pr10_root is not None or not motion.pr10_bundle.is_file()
            or motion.version != "v1.0-mini"):
        raise AssertionError("Stage3 motion should run at final inference without segmaps or a .git checkout")
    try:
        _build_evaluator(root / "stage2-mini", motion=False, version="v1.0-mini")
    except ValueError as exc:
        if "Stage2" not in str(exc):
            raise AssertionError(f"Stage2 mini version rejection was unclear: {exc}") from exc
    else:
        raise AssertionError("Stage2 full scoring unexpectedly accepted v1.0-mini")
    return evaluator


def _check_nuscenes_version_split_contract():
    if eval_split_for_version("v1.0-trainval", motion_only=True) != "val":
        raise AssertionError("PR10 trainval motion must use the official val split")
    if eval_split_for_version("v1.0-mini", motion_only=True) != "mini_val":
        raise AssertionError("PR10 mini motion must use the official mini_val split")
    try:
        eval_split_for_version("v1.0-mini", motion_only=False)
    except ValueError:
        pass
    else:
        raise AssertionError("PR10 Stage2 full metrics unexpectedly accepted v1.0-mini")
    coverage = validate_split_token_coverage(
        ["mini-a", "mini-b"], ["mini-b", "mini-a"], version="v1.0-mini", split="mini_val")
    if coverage != {"version": "v1.0-mini", "split": "mini_val", "val_tokens": 2,
                    "devkit_split_tokens": 2}:
        raise AssertionError(f"PR10 split coverage omitted the mini version/split: {coverage}")
    try:
        validate_split_token_coverage(["mini-a"], ["mini-a", "mini-b"],
                                      version="v1.0-mini", split="mini_val")
    except ValueError:
        pass
    else:
        raise AssertionError("PR10 val-info coverage accepted an incomplete official split")


def _check_pr10_sequence_loader_shards(root: Path):
    from unidrivevla.data import LOADERS

    evaluator = _build_evaluator(root / "sequence-shards")
    source = FakeSource()

    class FakePolicy:
        def worker_transform(self, image_augment=None, train=False):
            if image_augment is not None or train:
                raise AssertionError("PR10 eval worker transform must disable augmentation")
            return lambda features, targets: (dict(features), targets)

    policy = FakePolicy()
    world = 4
    all_tokens = {}
    schedules = {}
    padding_counts = {}
    with patch.dict(LOADERS, {"nuscenes": lambda *_args: source}):
        for strategy, pad_eval in (("ddp", False), ("fsdp", True)):
            rank_tokens = []
            rank_counts = []
            rank_padding = []
            for rank in range(world):
                evaluator.loader = None
                trainer = SimpleNamespace(
                    ctx=FakeContext(rank, world), agent=policy, cfg=evaluator.cfg,
                    args=SimpleNamespace(strategy=strategy),
                )
                loader = evaluator.eval_loader(trainer)
                rank_counts.append(len(loader))
                padding = 0
                for features, _, tokens in loader:
                    if len(tokens) != 1 or not features["img_metas"][0].get("stream_id"):
                        raise AssertionError("PR10 sequence loader did not keep one token and stream per forward")
                    if tokens[0] is None:
                        padding += 1
                    else:
                        rank_tokens.append(tokens[0])
                rank_padding.append(padding)
            if sorted(rank_tokens) != sorted(source.tokens) or len(rank_tokens) != len(set(rank_tokens)):
                raise AssertionError(f"{strategy} PR10 sharding lost or duplicated real tokens: {rank_tokens}")
            schedules[strategy] = rank_counts
            padding_counts[strategy] = rank_padding
            all_tokens[strategy] = rank_tokens

    if schedules["ddp"] != [2, 1, 3, 0] or any(padding_counts["ddp"]):
        raise AssertionError(f"DDP PR10 evaluation must preserve unpadded uneven/empty shards: {schedules['ddp']}")
    if schedules["fsdp"] != [3, 3, 3, 3] or sum(padding_counts["fsdp"]) != 6:
        raise AssertionError(f"FSDP PR10 evaluation must pad all ranks to the same forwards: "
                             f"{schedules['fsdp']} pads={padding_counts['fsdp']}")

    row = {"scores_3d": torch.tensor([0.5])}
    rows = {}
    _append_prediction_rows(rows, [source.tokens[0], None], [row, row])
    if set(rows) != {source.tokens[0]}:
        raise AssertionError("FSDP padding rows leaked into PR10 scoring coverage")
    try:
        _append_prediction_rows(rows, [source.tokens[0]], [row])
    except ValueError as exc:
        if "duplicate PR10 prediction" not in str(exc):
            raise AssertionError(f"duplicate PR10 prediction rejection was unclear: {exc}") from exc
    else:
        raise AssertionError("PR10 result aggregation accepted a duplicate real token")


def _check_pr10_scoring_import_shim(root: Path):
    """The pinned plugin root must remain unexecuted while its dataset namespace stays importable."""
    nusc = root / "pr10-import" / "nuScenes"
    plugin = nusc / "projects" / "mmdet3d_plugin"
    (plugin / "datasets" / "evaluation" / "planning").mkdir(parents=True)
    (nusc / "projects").mkdir(exist_ok=True)
    (nusc / "projects" / "__init__.py").write_text("", encoding="utf-8")
    (plugin / "__init__.py").write_text("raise RuntimeError('eager plugin root executed')\n", encoding="utf-8")
    (plugin / "datasets" / "__init__.py").write_text("NuScenes3DDataset = object\n", encoding="utf-8")
    (plugin / "datasets" / "evaluation" / "planning" / "planning_eval.py").write_text(
        "SCORING_MODULE_IMPORTED = True\n", encoding="utf-8")

    old_path = list(sys.path)
    old_modules = {key: value for key, value in sys.modules.items()
                   if key == "projects" or key.startswith("projects.")}
    for key in old_modules:
        sys.modules.pop(key, None)
    try:
        sys.path.insert(0, str(nusc))
        planning = install_pr10_scoring_imports(nusc)
        if not getattr(planning, "SCORING_MODULE_IMPORTED", False):
            raise AssertionError("the pinned planning scorer module was not imported")
        if "projects.mmdet3d_plugin" not in sys.modules:
            raise AssertionError("the absolute-import package shim was not installed")
    finally:
        for key in list(sys.modules):
            if key == "projects" or key.startswith("projects."):
                sys.modules.pop(key, None)
        sys.modules.update(old_modules)
        sys.path[:] = old_path


def _check_packaged_pr10_reference(root: Path):
    bundle = packaged_pr10_bundle()
    if not bundle.is_file():
        raise AssertionError(f"ZIP/wheel scorer bundle is missing: {bundle}")
    manifest_path = bundle.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("revision") != PR10 or manifest.get("bundle") != bundle.name
            or manifest.get("sha256") != PR10_BUNDLE_SHA256
            or manifest.get("license") != "Apache-2.0 (upstream repository LICENSE.txt included)"
            or manifest.get("build") != "git archive with core.autocrlf=false and tar.umask=0000; gzip mtime=0"
            or "LICENSE.txt" not in manifest.get("source_paths", [])):
        raise AssertionError("portable PR10 source manifest does not match its pinned source, hash and license")
    nusc, digest = extract_pr10_bundle(bundle, root / "pr10-reference")
    if digest != manifest["sha256"]:
        raise AssertionError("portable PR10 bundle hash differs from its source manifest")
    required = (
        nusc / "projects" / "configs" / "UniDriveVLA" / "unidrivevla_nusc_2b_stage2.py",
        nusc / "projects" / "configs" / "UniDriveVLA" / "unidrivevla_nusc_2b_stage3.py",
        nusc / "projects" / "mmdet3d_plugin" / "datasets" / "evaluation" / "motion" / "motion_eval_uniad.py",
    )
    if any(not path.is_file() for path in required):
        raise AssertionError("portable PR10 bundle lacks a pinned Stage2/Stage3 scorer source")
    license_path = nusc.parent / "LICENSE.txt"
    if not license_path.is_file() or "Apache License" not in license_path.read_text(encoding="utf-8"):
        raise AssertionError("portable PR10 source bundle must retain the upstream Apache-2.0 license")
    with tarfile.open(bundle, "r:gz") as archive:
        if any(name == ".git" or "/.git/" in name for name in archive.getnames()):
            raise AssertionError("portable PR10 source bundle must not contain Git metadata")
    if (nusc / ".git").exists() or (nusc.parent / ".git").exists():
        raise AssertionError("portable PR10 scorer bundle must not require a .git directory")


def _check_full_metric_and_loose_coverage(root: Path):
    metrics = {
        "img_bbox_NuScenes/NDS": 0.1,
        "img_bbox_NuScenes/mAP": 0.1,
        "img_bbox_NuScenes/mATE": 0.1,
        "img_bbox_NuScenes/mASE": 0.1,
        "img_bbox_NuScenes/mAOE": 0.1,
        "img_bbox_NuScenes/mAVE": 0.1,
        "img_bbox_NuScenes/mAAE": 0.1,
        "mAP_normal": 0.1,
        "L2": 0.1,
        "obj_col": 0.1,
        "obj_box_col": 0.1,
        "ego_status/count": 2,
        "ego_status/dim": 10,
        "ego_status/mae": 0.1,
        "ego_status/rmse": 0.1,
        "ego_status/max_abs": 0.1,
    }
    metrics.update({f"img_bbox_NuScenes/{key}": 0.1 for key in TRACKING_METRICS})
    metrics.update({name: 0.1 for name in PR10_MAP_CLASSES})
    metrics.update({f"ego_status/{name}_{suffix}": 0.1 for name in STATUS_NAMES
                    for suffix in ("mae", "rmse")})
    metrics.update({f"img_bbox_NuScenes/{name}_AP_dist_0.5": 0.1 for name in PR10_DET_CLASSES})
    metrics.update({key: 0.1 for key in PR10_NA_METRICS})
    validate_metric_coverage(metrics)

    # The pinned nuScenes devkit intentionally reports these five class-specific errors as NaN.
    for key in PR10_NA_METRICS:
        metrics[key] = float("nan")
    validate_metric_coverage(metrics)
    sanitized = json_safe_metrics(metrics)
    assert all(sanitized[key] is None for key in PR10_NA_METRICS)
    for key in ("img_bbox_NuScenes/car_attr_err", "obj_box_col"):
        broken = dict(metrics)
        broken[key] = float("nan")
        try:
            validate_metric_coverage(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"PR10 metric coverage accepted unapproved NaN {key}")

    for missing in ("obj_col", "obj_box_col", "mAP_normal", "img_bbox_NuScenes/amota"):
        broken = dict(metrics)
        broken.pop(missing)
        try:
            validate_metric_coverage(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"PR10 metric coverage accepted missing {missing}")
    no_boundary_ap = dict(metrics)
    no_boundary_ap.pop("boundary")
    try:
        validate_metric_coverage(no_boundary_ap)
    except ValueError:
        pass
    else:
        raise AssertionError("PR10 metric coverage accepted missing boundary AP")

    occupancy = {"a": torch.zeros((7, 200, 200)), "b": torch.zeros((6, 200, 200)),
                 "bad": torch.zeros((6, 100, 100))}

    def get_occupancy(maps, token):
        value = maps.get(token)
        if value is None:
            return None
        if value.shape[0] % 2:
            value = value[1:]
        return value[:6].unsqueeze(0) if value.shape[0] >= 6 else None

    full = loose_segmentation_coverage(occupancy, ["a", "b"], get_occupancy)
    if full["loose_tokens_scored"] != 2 or full["loose_tokens_missing_or_invalid"] != 0:
        raise AssertionError(f"PR10 full loose coverage was miscounted: {full}")
    partial = loose_segmentation_coverage(occupancy, ["a", "missing"], get_occupancy)
    if partial["loose_tokens_scored"] != 1 or partial["loose_tokens_missing_or_invalid"] != 1:
        raise AssertionError(f"PR10 skip-missing coverage was not reported: {partial}")
    malformed = loose_segmentation_coverage(occupancy, ["a", "bad"], get_occupancy)
    if malformed["loose_tokens_scored"] != 1 or malformed["loose_tokens_missing_or_invalid"] != 1:
        raise AssertionError(f"PR10 malformed occupancy coverage was not reported: {malformed}")
    for unusable, tokens in ((None, ["a"]), ({}, ["a"]), (occupancy, ["bad"])):
        try:
            loose_segmentation_coverage(unusable, tokens, get_occupancy)
        except ValueError:
            pass
        else:
            raise AssertionError("PR10 scoring accepted a loose-planning split without any usable raster")


def _check_scene_shards(evaluator):
    from unidrivevla.data import registry as data_registry

    source = FakeSource()
    evaluator.dataset = source
    dataset = data_registry.TransformedDataset(
        _PredictionOnlyNuScenesSource(source), lambda features, targets: (dict(features), targets))
    params = {"batch_size": 1, "num_workers": 0, "pin_memory": False}

    def read_rank(rank):
        loader = evaluator.make_eval_loader(dataset, params, FakeContext(rank=rank, world_size=2))
        frames = []
        for features, _, tokens in loader:
            frames.append((features["img_metas"][0]["sample_index"], tokens[0],
                           features["img_metas"][0]["stream_id"]))
        return frames

    rank0, rank1 = read_rank(0), read_rank(1)
    rank0_indices = [index for index, _, _ in rank0]
    rank1_indices = [index for index, _, _ in rank1]
    rank0_streams = {token: stream for _, token, stream in rank0}
    rank1_streams = {token: stream for _, token, stream in rank1}
    if rank0_indices != [0, 1, 3, 4, 5] or rank1_indices != [2]:
        raise AssertionError(f"rank shards must contain complete scenes, got {rank0_indices} / {rank1_indices}")
    if set(rank0_streams) != {source.tokens[i] for i in rank0_indices}:
        raise AssertionError("rank 0 stream map does not cover exactly its assigned scenes")
    if set(rank1_streams) != {source.tokens[i] for i in rank1_indices}:
        raise AssertionError("rank 1 stream map does not cover exactly its assigned scenes")
    if rank0_streams[source.tokens[0]] != rank0_streams[source.tokens[1]]:
        raise AssertionError("frames from one scene received different stream IDs")
    if rank0_streams[source.tokens[3]] != rank0_streams[source.tokens[5]]:
        raise AssertionError("frames from one scene received different stream IDs")
    if rank0_streams[source.tokens[0]] == rank0_streams[source.tokens[3]]:
        raise AssertionError("different scenes received the same stream ID")

    try:
        evaluator.make_eval_loader(dataset, {**params, "batch_size": 2}, FakeContext())
    except ValueError as exc:
        if "batch_size=1" not in str(exc):
            raise AssertionError(f"non-unit eval batch rejected without a useful error: {exc}") from exc
    else:
        raise AssertionError("PR10 tracking accepted eval batch_size != 1")


def _check_stage3_adapter_export():
    config = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "profile" / "nuscenes.yaml")
    profile = replace(Profile.from_config(config), frame="lidar")

    class MotionCaptureHead(nn.Module):
        with_motion = True
        canbus_sft_prompt = False
        qwen3_vl_with_expert = SimpleNamespace(qwen3_vl=SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))))

        def forward_test(self, **kwargs):
            if kwargs.get("return_nuscenes_pr10") is not True or kwargs.get("use_gt_ego_status") is not False:
                raise AssertionError("Stage3 exporter did not request model-predicted ego status")
            return {
                "traj": torch.zeros((1, 6, 2), dtype=torch.float32),
                "ego_status_pred": torch.zeros((1, 10), dtype=torch.float32),
                "det": [{
                    "boxes_3d": torch.zeros((1, 9), dtype=torch.float32),
                    "scores_3d": torch.tensor([0.9], dtype=torch.float32),
                    "labels_3d": torch.tensor([0], dtype=torch.long),
                    "trajs_3d": torch.zeros((1, 6, 12, 2), dtype=torch.float32),
                    "trajs_score": torch.full((1, 6), 1 / 6, dtype=torch.float32),
                }],
            }

    policy = UniDriveVLAPolicy.__new__(UniDriveVLAPolicy)
    nn.Module.__init__(policy)
    policy.profile = profile
    policy.with_motion = True
    policy.head = MotionCaptureHead()
    policy.eval()
    features = {
        "img": torch.zeros((1, 6, 3, 8, 8)),
        "coordinate_frame": ["lidar"],
        "native_to_ego": torch.eye(4).unsqueeze(0),
    }
    out = policy({**features, "return_nuscenes_pr10_motion": True})
    row = out["nuscenes_pr10_motion"][0]
    if tuple(row["trajs_3d"].shape) != (1, 6, 12, 2) or tuple(row["trajs_score"].shape) != (1, 6):
        raise AssertionError("Stage3 adapter export lost the six-mode, 12-step PR10 motion fields")
    if "nuscenes_pr10" in out:
        raise AssertionError("Stage3 motion export unexpectedly selected the Stage2 full-metrics schema")

    policy.with_motion = False
    policy.head.with_motion = False
    try:
        policy({**features, "return_nuscenes_pr10_motion": True})
    except ValueError as exc:
        if "motion on" not in str(exc):
            raise AssertionError(f"Stage3 motion mismatch rejected without a clear message: {exc}") from exc
    else:
        raise AssertionError("Stage3 motion export accepted a motion-disabled policy")


def _check_stage3_motion_evaluator(root: Path):
    from unidrivevla.data import LOADERS

    evaluator = _build_evaluator(root / "stage3", name="nuscenes_pr10_motion", motion=True,
                                 include_segmentation=False, external_root=False, version="v1.0-mini")
    source = FakeSource()
    model = FakeMotionModel()

    class FakePolicy:
        def worker_transform(self, image_augment=None, train=False):
            assert image_augment is None and not train
            return lambda features, targets: (dict(features), targets)

    trainer = SimpleNamespace(ctx=FakeContext(), model=model, agent=FakePolicy(), cfg=evaluator.cfg)

    def pass_through_loader(dataset, device):
        return iter(dataset)

    with patch.dict(LOADERS, {"nuscenes": lambda *_args: source}), \
         patch("unidrivevla.adapters.isolated_evaluation_state", side_effect=lambda *_: nullcontext()), \
         patch.object(importlib.import_module("unidrivevla.distributed.parallelize"), "check_eval_loader"), \
         patch("unidrivevla.data.DevicePrefetcher", side_effect=pass_through_loader):
        evaluator.eval_loader(trainer)
        if source.target_calls:
            raise AssertionError("Stage3 motion inference built unused GT targets")
        evaluator.during_training(trainer, None, "stage3")

    if model.calls != len(source) or source.target_calls:
        raise AssertionError("Stage3 motion export did not produce exactly one target-free row per val sample")
    _, output = evaluator.pending[0]
    shard_paths = sorted((output / "predictions").glob("rank-*.pt"))
    if len(shard_paths) != 1:
        raise AssertionError(f"Stage3 motion expected one prediction shard, got {shard_paths}")
    saved = torch.load(shard_paths[0], map_location="cpu", weights_only=True)
    expected = {"boxes_3d", "scores_3d", "labels_3d", "trajs_3d", "trajs_score"}
    for token, result in saved.items():
        row = result["img_bbox"]
        if not expected.issubset(row) or tuple(row["trajs_3d"].shape) != (1, 6, 12, 2):
            raise AssertionError(f"incomplete Stage3 motion output for {token}")
    rows = load_results(output / "predictions", source.tokens, motion=True)
    assert len(rows) == len(source.tokens)

    metrics = {key: 0.1 for key in MOTION_METRIC_KEYS}
    classes = {"car": {"ground_truth": 8, "predictions": 2, "status": "scored"},
               "pedestrian": {"ground_truth": 4, "predictions": 2, "status": "no_matches"}}
    for key in metrics:
        if key.startswith("pedestrian_"):
            metrics[key] = -0.25 if key == "pedestrian_EPA" else None
    scorer_calls = []

    def fake_scorer(command, *, check, stdin, stdout, stderr, env):
        scorer_calls.append((command, env))
        if not check or env.get("CUDA_VISIBLE_DEVICES") != "":
            raise AssertionError("Stage3 reference scorer must run as a checked CPU child")
        if "--motion-only" not in command or "--segmentation" in command or "--pr10-root" in command:
            raise AssertionError(f"ZIP Stage3 scoring did not use bundled motion-only reference: {command}")
        if "--reference-bundle" not in command:
            raise AssertionError("ZIP Stage3 scoring did not pass the bundled fixed scorer")
        if command[command.index("--version") + 1] != "v1.0-mini":
            raise AssertionError("Stage3 mini version did not reach the independent PR10 scorer")
        output_dir = Path(command[command.index("--output-dir") + 1])
        (output_dir / "nuscenes_pr10_motion_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (output_dir / "nuscenes_pr10_motion_coverage.json").write_text(json.dumps({"classes": classes}), encoding="utf-8")
        stdout.write(b"mock PR10 motion scorer\n")
        return SimpleNamespace(returncode=0)

    with patch("unidrivevla.eval.nuscenes_pr10.subprocess.run", side_effect=fake_scorer):
        result = evaluator.finish(None)
    if len(scorer_calls) != 1 or result != [("stage3", metrics)]:
        raise AssertionError(f"Stage3 motion scorer result was not returned: {result}")

    # A partial Stage3 motion result must be a failed evaluation, never a successful empty report.
    # Exercise the metric coverage validator directly for a missing pedestrian FDE output.
    try:
        from unidrivevla.tools.score_nuscenes_pr10 import validate_motion_metric_coverage
        incomplete = dict(metrics)
        incomplete.pop("pedestrian_min_fde_err")
        validate_motion_metric_coverage(incomplete, classes)
    except ValueError:
        pass
    else:
        raise AssertionError("Stage3 motion metric coverage accepted a missing pedestrian FDE")
    validate_motion_metric_coverage(dict.fromkeys(MOTION_METRIC_KEYS, 0.1))
    for key, value in (("car_EPA", None), ("car_min_ade_err", float("nan")),
                       ("pedestrian_min_fde_err", 0.0), ("pedestrian_EPA", 0.0)):
        try:
            validate_motion_metric_coverage({**metrics, key: value}, classes)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid motion result {key}={value}")


def _check_prediction_roundtrip_and_score(evaluator):
    from unidrivevla.data import LOADERS

    source = FakeSource()
    model = FakeModel()

    class FakePolicy:
        def worker_transform(self, image_augment=None, train=False):
            assert image_augment is None and not train
            return lambda features, targets: (dict(features), targets)

    trainer = SimpleNamespace(
        ctx=FakeContext(),
        model=model,
        agent=FakePolicy(),
        cfg=evaluator.cfg,
    )

    def pass_through_loader(dataset, device):
        return iter(dataset)

    with patch.dict(LOADERS, {"nuscenes": lambda *_args: source}), \
         patch("unidrivevla.adapters.isolated_evaluation_state", side_effect=lambda *_: nullcontext()), \
         patch.object(importlib.import_module("unidrivevla.distributed.parallelize"), "check_eval_loader"), \
         patch("unidrivevla.data.DevicePrefetcher", side_effect=pass_through_loader):
        loader = evaluator.eval_loader(trainer)
        if source.target_calls:
            raise AssertionError(f"PR10 prediction loader built {source.target_calls} unused GT target sets")
        evaluator.during_training(trainer, None, "fixture")

    if model.calls != len(source):
        raise AssertionError(f"expected one prediction per sample, got {model.calls} for {len(source)}")
    if source.target_calls:
        raise AssertionError(f"PR10 prediction export built {source.target_calls} unused GT target sets")
    if model.sample_indices != list(range(len(source))) or not (
            model.stream_ids[0] == model.stream_ids[1] != model.stream_ids[2]
            and model.stream_ids[3] == model.stream_ids[4] == model.stream_ids[5]
            and model.stream_ids[2] != model.stream_ids[3]):
        raise AssertionError(f"model did not receive one stable stream ID per scene: {model.stream_ids}")
    if not evaluator.pending:
        raise AssertionError("successful prediction export was not queued for the scorer")
    _, output = evaluator.pending[0]
    prediction_files = sorted((output / "predictions").glob("rank-*.pt"))
    if len(prediction_files) != 1:
        raise AssertionError(f"expected one rank prediction file, got {prediction_files}")
    saved = torch.load(prediction_files[0], map_location="cpu", weights_only=True)
    if set(saved) != set(source.tokens):
        raise AssertionError("weights-only PR10 prediction export lost sample tokens")
    expected_fields = {
        "boxes_3d", "scores_3d", "labels_3d", "instance_ids", "vectors", "scores",
        "labels", "final_planning", "ego_status_pred",
    }
    for token, result in saved.items():
        row = result["img_bbox"]
        if set(row) != expected_fields or any(not torch.is_tensor(value) for value in row.values()):
            raise AssertionError(f"PR10 prediction schema/dtypes are invalid for {token}: {row.keys()}")
    for index, token in enumerate(source.tokens):
        saved[token]["img_bbox"]["instance_ids"] = torch.tensor([index])
    torch.save(saved, prediction_files[0])
    ordered = load_results(output / "predictions", list(reversed(source.tokens)))
    assert [r["img_bbox"]["instance_ids"].item() for r in ordered] == list(reversed(range(len(source))))
    assert all(isinstance(r["img_bbox"]["vectors"], list) and
               not torch.is_tensor(r["img_bbox"]["scores"]) for r in ordered)
    try:
        load_results(output / "predictions", source.tokens + ["missing-val-token"])
    except ValueError as exc:
        assert "1 missing" in str(exc)
    else:
        raise AssertionError("PR10 scorer accepted incomplete val coverage")

    # The PR10 `ego_status_pred` scorer uses `.numpy()`, so all floating fields must leave bf16 first.
    bf16_dir = output / "bf16"
    bf16_dir.mkdir()
    bf16 = {source.tokens[0]: dict(saved[source.tokens[0]])}
    bf16_row = bf16[source.tokens[0]]["img_bbox"]
    for key, value in list(bf16_row.items()):
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            bf16_row[key] = value.to(torch.bfloat16)
    torch.save(bf16, bf16_dir / "rank-0000.pt")
    bf16_result = load_results(bf16_dir, [source.tokens[0]])[0]["img_bbox"]
    assert bf16_result["ego_status_pred"].dtype == torch.float32
    assert bf16_result["final_planning"].dtype == torch.float32
    assert bf16_result["boxes_3d"].dtype == torch.float32
    assert bf16_result["vectors"][0].dtype.name == "float32"

    # Preserve the pinned map scorer's valid empty-list row as well as nonempty vectors.
    empty_dir = output / "empty-map"
    empty_dir.mkdir()
    empty_rows = {source.tokens[0]: {"img_bbox": dict(bf16[source.tokens[0]]["img_bbox"])}}
    empty_rows[source.tokens[0]]["img_bbox"].update(
        vectors=torch.empty((0,), dtype=torch.float32),
        scores=torch.empty((0,), dtype=torch.float32),
        labels=torch.empty((0,), dtype=torch.long),
    )
    torch.save(empty_rows, empty_dir / "rank-0000.pt")
    empty_result = load_results(empty_dir, [source.tokens[0]])[0]["img_bbox"]
    assert empty_result["vectors"] == [] and empty_result["scores"].shape == (0,)

    complete = dict(zip(METRIC_GROUP_KEYS, (0.42, 0.31, 0.7, 0.5, 0.1)))
    scorer_calls = []
    metric_payload = complete

    def fake_scorer(command, *, check, stdin, stdout, stderr, env):
        scorer_calls.append((command, env))
        if not check:
            raise AssertionError("scorer subprocess must use check=True")
        if command[command.index("--version") + 1] != "v1.0-trainval":
            raise AssertionError("Stage2 PR10 scorer no longer defaults to v1.0-trainval")
        if env.get("CUDA_VISIBLE_DEVICES") != "":
            raise AssertionError("CPU scorer subprocess must hide CUDA")
        if {"PYTHONPATH", "PYTHONHOME"} & set(env):
            raise AssertionError("PR10 scorer inherited the current training Python environment")
        output_dir = Path(command[command.index("--output-dir") + 1])
        (output_dir / "nuscenes_pr10_metrics.json").write_text(
            json.dumps(metric_payload), encoding="utf-8")
        stdout.write(b"mock PR10 scorer\n")
        return SimpleNamespace(returncode=0)

    with patch("unidrivevla.eval.nuscenes_pr10.subprocess.run", side_effect=fake_scorer):
        good = evaluator.finish(None)
    if len(scorer_calls) != 1 or good != [("fixture", complete)]:
        raise AssertionError(f"complete five-group metrics were not accepted: {good}")
    command, env = scorer_calls[0]
    if command[0] != sys.executable or env.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("PR10 scoring did not use the configured isolated CPU interpreter")
    for flag in ("--pr10-root", "--infos", "--data-root", "--segmentation", "--predictions", "--output-dir"):
        if flag not in command:
            raise AssertionError(f"PR10 scorer command is missing required asset argument {flag}")

    evaluator.eval_loader = lambda _trainer: evaluator.make_eval_loader(
        source, params, FakeContext())
    with patch("unidrivevla.adapters.isolated_evaluation_state", side_effect=lambda *_: nullcontext()), \
         patch.object(importlib.import_module("unidrivevla.distributed.parallelize"), "check_eval_loader"), \
         patch("unidrivevla.data.DevicePrefetcher", side_effect=pass_through_loader):
        evaluator.during_training(trainer, None, "missing-metric")
    metric_payload = {key: value for key, value in complete.items() if key != "mAP_normal"}
    with patch("unidrivevla.eval.nuscenes_pr10.subprocess.run", side_effect=fake_scorer):
        bad = evaluator.finish(None)
    if len(bad) != 1 or "error" not in bad[0][1]:
        raise AssertionError(f"missing PR10 metric group was accepted: {bad}")
    sink = ReportSink()
    try:
        Trainer.report(sink, evaluator, bad, synchronize=False)
    except RuntimeError as exc:
        if "nuscenes_pr10" not in str(exc):
            raise AssertionError(f"metric failure was propagated without evaluator identity: {exc}") from exc
    else:
        raise AssertionError("missing full-metric group did not fail the parent run")
    if not sink.aborted:
        raise AssertionError("metric failure did not abort owned evaluation work")


def main():
    with tempfile.TemporaryDirectory(prefix="unidrivevla-pr10-eval-") as temp:
        root = Path(temp)
        _check_packaged_pr10_reference(root)
        _check_nuscenes_version_split_contract()
        _check_pr10_scoring_import_shim(root)
        _check_full_metric_and_loose_coverage(root)
        _check_stage3_adapter_export()
        evaluator = _check_registered_final_only_mode(root)
        _check_scene_shards(evaluator)
        _check_pr10_sequence_loader_shards(root)
        _check_prediction_roundtrip_and_score(evaluator)
        _check_stage3_motion_evaluator(root)
    print("OK: PR10 CPU import shim, full metric keys, loose coverage, evaluator registration, "
          "Stage2/Stage3 export, scene sharding and prediction roundtrips")


if __name__ == "__main__":
    main()
