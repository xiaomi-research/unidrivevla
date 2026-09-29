#!/usr/bin/env python3
"""NeuroNCAP model server for UniDriveVLA.

The server implements the small HTTP API expected by NeuroNCAP:

  POST /reset
  POST /infer

Run this from the UniDriveVLA nuScenes environment, not from the
NeuroNCAP-only environment.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import random
import sys
import tempfile
import traceback
from collections import OrderedDict
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def optional_env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None

    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def configure_hugsim_coc(planning_head: Any) -> tuple[bool, bool, str, str]:
    """Apply explicit HUGSIM CoC overrides before the planning head is built."""
    enable_coc = optional_env_flag("HUGSIM_ENABLE_COC")
    if enable_coc is None:
        enable_coc = optional_env_flag("ENABLE_COC")
    enable_generate = optional_env_flag("HUGSIM_ENABLE_COC_GENERATE")
    eval_source = os.environ.get("HUGSIM_COC_EVAL_SOURCE")
    eval_injection = os.environ.get("HUGSIM_COC_EVAL_INJECTION")

    if enable_coc is not None:
        planning_head["enable_coc"] = enable_coc
        if not enable_coc:
            planning_head["enable_coc_context"] = False
            planning_head["enable_coc_language_output"] = False
            planning_head["enable_coc_generate"] = False
            planning_head["coc_jsonl_path"] = None
            planning_head["coc_eval_source"] = "none"
            planning_head["coc_eval_injection"] = "none"
    if enable_generate is not None:
        planning_head["enable_coc_generate"] = enable_generate
    if eval_source is not None and eval_source.strip():
        eval_source = eval_source.strip().lower()
        if eval_source not in ("pred", "gt", "none"):
            raise ValueError(
                "HUGSIM_COC_EVAL_SOURCE must be one of pred/gt/none, "
                f"got {eval_source!r}"
            )
        planning_head["coc_eval_source"] = eval_source
    if eval_injection is not None and eval_injection.strip():
        eval_injection = eval_injection.strip().lower()
        if eval_injection not in ("answer", "context", "none"):
            raise ValueError(
                "HUGSIM_COC_EVAL_INJECTION must be one of answer/context/none, "
                f"got {eval_injection!r}"
            )
        planning_head["coc_eval_injection"] = eval_injection

    enable_coc = bool(planning_head.get("enable_coc", False))
    enable_generate = bool(planning_head.get("enable_coc_generate", False))
    eval_source = str(planning_head.get("coc_eval_source", "none")).lower()
    eval_injection = str(planning_head.get("coc_eval_injection", "none")).lower()

    if enable_generate and not enable_coc:
        raise ValueError(
            "HUGSIM_ENABLE_COC_GENERATE is enabled, but planning_head.enable_coc "
            "is false or missing in the model config"
        )
    if enable_generate and eval_source != "pred":
        raise ValueError(
            "CoC generation requires HUGSIM_COC_EVAL_SOURCE=pred; "
            f"got {eval_source!r}"
        )
    if enable_generate and eval_injection == "none":
        raise ValueError(
            "CoC generation is enabled but HUGSIM_COC_EVAL_INJECTION=none; "
            "use answer or context so planning is conditioned on the prediction"
        )
    return enable_coc, enable_generate, eval_source, eval_injection


def seed_inference(seed: int | None = None, announce: bool = True) -> int:
    if seed is None:
        seed = int(os.environ.get("HUGSIM_INFERENCE_SEED", "0"))
    deterministic = env_flag("HUGSIM_DETERMINISTIC", False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if announce:
        print(
            f"Inference seed: {seed} (cudnn_deterministic={deterministic})",
            flush=True,
        )
    return seed


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("module", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def prepare_checkpoint_state_dict(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    min_parameter_coverage: float,
) -> OrderedDict[str, Any]:
    if not 0.0 <= min_parameter_coverage <= 1.0:
        raise ValueError(
            "HUGSIM_CHECKPOINT_MIN_PARAMETER_COVERAGE must be between 0 and 1, "
            f"got {min_parameter_coverage}"
        )

    source = checkpoint_state_dict(checkpoint)
    raw_state = OrderedDict()
    ema_keys = []
    for source_key, value in source.items():
        if not isinstance(source_key, str):
            continue
        key = source_key[7:] if source_key.startswith("module.") else source_key
        if key.startswith("ema_"):
            ema_keys.append(key)
            continue
        raw_state[key] = value

    model_state = model.state_dict()
    shape_mismatches = []
    for key in raw_state.keys() & model_state.keys():
        checkpoint_value = raw_state[key]
        model_value = model_state[key]
        if not hasattr(checkpoint_value, "shape"):
            shape_mismatches.append((key, "non-tensor", tuple(model_value.shape)))
        elif tuple(checkpoint_value.shape) != tuple(model_value.shape):
            shape_mismatches.append(
                (key, tuple(checkpoint_value.shape), tuple(model_value.shape))
            )
    if shape_mismatches:
        examples = "; ".join(
            f"{key}: checkpoint={checkpoint_shape}, model={model_shape}"
            for key, checkpoint_shape, model_shape in shape_mismatches[:10]
        )
        raise RuntimeError(
            f"Checkpoint contains {len(shape_mismatches)} shape mismatch(es): {examples}"
        )

    parameters = dict(model.named_parameters())
    total_parameter_numel = sum(parameter.numel() for parameter in parameters.values())
    matched_parameters = [key for key in parameters if key in raw_state]
    matched_parameter_numel = sum(parameters[key].numel() for key in matched_parameters)
    parameter_coverage = (
        matched_parameter_numel / total_parameter_numel if total_parameter_numel else 1.0
    )

    missing_parameters = sorted(set(parameters) - set(raw_state))
    missing_state = sorted(set(model_state) - set(raw_state))
    unexpected_state = sorted(set(raw_state) - set(model_state))
    print(
        "Checkpoint coverage: "
        f"parameters={parameter_coverage:.6%} "
        f"({matched_parameter_numel}/{total_parameter_numel} numel), "
        f"missing_parameters={len(missing_parameters)}, "
        f"missing_state={len(missing_state)}, "
        f"unexpected_state={len(unexpected_state)}, "
        f"ignored_ema={len(ema_keys)}",
        flush=True,
    )
    if missing_parameters:
        print(f"  missing parameter examples: {missing_parameters[:10]}", flush=True)
    if unexpected_state:
        print(f"  unexpected state examples: {unexpected_state[:10]}", flush=True)

    if parameter_coverage < min_parameter_coverage:
        raise RuntimeError(
            f"Checkpoint parameter coverage {parameter_coverage:.6%} is below required "
            f"minimum {min_parameter_coverage:.6%}"
        )
    return raw_state


def register_ncap_pipeline_transforms() -> None:
    from mmdet.datasets.builder import PIPELINES

    if "NCAPEnsurePerViewImageShape" in PIPELINES.module_dict:
        return

    @PIPELINES.register_module()
    class NCAPEnsurePerViewImageShape:
        """Normalize img_shape metadata for NCAP temporary multi-view images."""

        def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
            if "img" in results:
                results["img_shape"] = [img.shape for img in results["img"]]
                results["ori_shape"] = [img.shape for img in results["img"]]
                results["pad_shape"] = [img.shape for img in results["img"]]
            return results


class UniDriveVLARunner:
    def __init__(self, config: str, checkpoint: str, device: str) -> None:
        self.adapter_root = Path(__file__).resolve().parents[1]
        self.repo_root = Path(
            os.environ.get("UNIDRIVEVLA_MODEL_ROOT", str(self.adapter_root))
        ).expanduser().resolve()
        self.nuscenes_root = self.repo_root / "nuScenes"
        if not self.nuscenes_root.is_dir():
            raise FileNotFoundError(f"UniDriveVLA nuScenes root not found: {self.nuscenes_root}")
        os.chdir(self.nuscenes_root)
        sys.path.insert(0, str(self.nuscenes_root))

        print(f"HUGSIM adapter root: {self.adapter_root}", flush=True)
        print(f"UniDriveVLA model root: {self.repo_root}", flush=True)

        import mmcv
        from mmcv import Config
        from mmcv.parallel import collate, scatter
        from mmcv.runner import _load_checkpoint, load_state_dict
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model

        register_ncap_pipeline_transforms()

        self.mmcv = mmcv
        self.collate = collate
        self.scatter = scatter
        self.device = torch.device(device)
        self.server_info = {
            "config": str(Path(config).expanduser().resolve()),
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "device": str(self.device),
            "adapter_root": str(self.adapter_root),
            "model_root": str(self.repo_root),
        }
        self.inference_seed = seed_inference()
        self.tmpdir = tempfile.TemporaryDirectory(prefix="unidrivevla_ncap_")
        self.pose_history: list[tuple[int, np.ndarray]] = []
        self._printed_input_shape = False
        self._printed_result_keys = False
        self._printed_aug_config = False
        self._last_debug_info: dict[str, Any] = {}

        cfg = Config.fromfile(config)
        if cfg.get("custom_imports", None):
            from mmcv.utils import import_modules_from_strings

            import_modules_from_strings(**cfg["custom_imports"])

        if getattr(cfg, "plugin", False):
            import importlib

            plugin_dir = getattr(cfg, "plugin_dir", "projects/mmdet3d_plugin/")
            module_path = ".".join(Path(plugin_dir).parts)
            importlib.import_module(module_path)

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        align_vocab_to_tokenizer = env_flag(
            "HUGSIM_ALIGN_VOCAB_TO_TOKENIZER", False
        )
        if "planning_head" not in cfg.model:
            raise KeyError("Model config has no planning_head for vocab alignment")
        cfg.model.planning_head.align_vocab_to_tokenizer = align_vocab_to_tokenizer
        coc_settings = configure_hugsim_coc(cfg.model.planning_head)
        self.server_info["coc"] = {
            "enabled": coc_settings[0],
            "generate": coc_settings[1],
            "eval_source": coc_settings[2],
            "eval_injection": coc_settings[3],
        }
        use_gt_ego_status = os.environ.get("HUGSIM_USE_GT_EGO_STATUS", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.use_gt_ego_status = use_gt_ego_status
        cfg.model.test_use_gt_ego_status = use_gt_ego_status
        cfg.data.test.test_mode = True
        cfg.data.test.pop("samples_per_gpu", None)
        cfg.data.test.pipeline.insert(
            1,
            dict(type="NCAPEnsurePerViewImageShape"),
        )
        print(
            "NCAP test pipeline: "
            + " -> ".join(step.get("type", str(step)) for step in cfg.data.test.pipeline),
            flush=True,
        )
        print(f"NCAP use gt ego status: {use_gt_ego_status}", flush=True)
        print(
            f"NCAP align vocab to tokenizer: {align_vocab_to_tokenizer}",
            flush=True,
        )
        print(
            "HUGSIM CoC inference: "
            f"enable_coc={coc_settings[0]}, "
            f"generate={coc_settings[1]}, "
            f"source={coc_settings[2]}, "
            f"injection={coc_settings[3]}",
            flush=True,
        )

        self.cfg = cfg
        self.dataset = build_dataset(cfg.data.test)
        self.model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        # Older UniDriveVLA detectors own this switch and forward it to the
        # planning head. Newer planning-only detectors accept arbitrary config
        # kwargs but do not retain or forward this one, so inject it with the
        # inference batch below when the detector has no native switch.
        self.inject_use_gt_ego_status = not hasattr(
            self.model, "test_use_gt_ego_status"
        )
        print(
            "NCAP ego status forwarding: "
            + (
                "batch argument (planning-only detector)"
                if self.inject_use_gt_ego_status
                else "detector attribute"
            ),
            flush=True,
        )
        print(f"Loading checkpoint: {checkpoint}", flush=True)
        checkpoint_data = _load_checkpoint(checkpoint, map_location="cpu")
        if not isinstance(checkpoint_data, Mapping):
            raise RuntimeError(f"Checkpoint is not a mapping: {type(checkpoint_data)}")
        min_coverage = float(
            os.environ.get("HUGSIM_CHECKPOINT_MIN_PARAMETER_COVERAGE", "0.99")
        )
        state_dict = prepare_checkpoint_state_dict(
            self.model, checkpoint_data, min_coverage
        )
        checkpoint_logger = logging.getLogger("unidrivevla_checkpoint")
        checkpoint_logger.setLevel(logging.CRITICAL)
        load_state_dict(
            self.model, state_dict, strict=False, logger=checkpoint_logger
        )
        print("Checkpoint loaded.", flush=True)
        self.model.CLASSES = checkpoint_data.get("meta", {}).get("CLASSES", self.dataset.CLASSES)
        if "PALETTE" in checkpoint_data.get("meta", {}):
            self.model.PALETTE = checkpoint_data["meta"]["PALETTE"]
        elif hasattr(self.dataset, "PALETTE"):
            self.model.PALETTE = self.dataset.PALETTE

        print(f"Moving model to device: {self.device}", flush=True)
        self.model.to(self.device)
        print("Model moved to device.", flush=True)
        self.model.eval()
        print("Model set to eval mode.", flush=True)

        self.timestamps = np.array([int(info["timestamp"]) for info in self.dataset.data_infos], dtype=np.int64)
        print("UniDriveVLA runner initialized.", flush=True)

    def reset(self) -> None:
        seed_inference(self.inference_seed, announce=False)
        self.pose_history.clear()
        for module in self.model.modules():
            reset = getattr(module, "reset", None)
            if callable(reset):
                try:
                    reset()
                except TypeError:
                    pass

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample = self._build_sample(payload)
        data = self.collate([sample], samples_per_gpu=1)
        if self.device.type == "cuda":
            data = self.scatter(data, [self.device.index or 0])[0]
        if self.inject_use_gt_ego_status:
            data["use_gt_ego_status"] = self.use_gt_ego_status
        if not self._printed_input_shape:
            img = data.get("img")
            img_shape = tuple(img.shape) if torch.is_tensor(img) else type(img)
            print(f"NCAP model input img shape: {img_shape}", flush=True)
            self._printed_input_shape = True

        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)

        trajectory = self._extract_trajectory(result)
        aux_outputs = self._extract_aux_outputs(result)
        return {
            "trajectory": trajectory,
            "aux_outputs": aux_outputs,
            "debug_info": self._last_debug_info,
        }

    def _nearest_index(self, timestamp: int) -> int:
        return int(np.argmin(np.abs(self.timestamps - int(timestamp))))

    def _build_sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        timestamp_us = int(payload["timestamp"])
        idx = self._nearest_index(timestamp_us)
        input_dict = self.dataset.get_data_info(idx)

        image_paths, image_shape = self._write_images(payload["images"], timestamp_us)
        aug_config = self._augmentation_for_image_shape(image_shape)
        input_dict["aug_config"] = aug_config
        ego2world = np.asarray(payload["ego2world"], dtype=np.float64)
        canbus = np.asarray(payload["canbus"], dtype=np.float32)
        calibration = payload["calibration"]
        lidar2ego = np.asarray(calibration["lidar2ego"], dtype=np.float64)
        lidar2global = ego2world @ lidar2ego
        ego_status = self._ego_status_from_canbus(canbus)

        input_dict["img_filename"] = image_paths
        input_dict["timestamp"] = timestamp_us / 1e6
        input_dict["lidar2global"] = lidar2global
        input_dict["ego_status"] = ego_status
        command_idx = self._command_index(payload.get("command", 2))
        input_dict["gt_ego_fut_cmd"] = self._command_onehot(command_idx)
        input_dict["gt_ego_fut_trajs"] = np.zeros((6, 2), dtype=np.float32)
        input_dict["gt_ego_fut_masks"] = np.zeros((6,), dtype=np.float32)

        hist_rel = self._history_in_current_lidar(lidar2global, timestamp_us)
        hist_deltas = self._relative_points_to_deltas(hist_rel)
        input_dict["gt_ego_his_trajs"] = hist_deltas
        input_dict["hist_traj"] = hist_rel.astype(np.float32)

        lidar2imgs = []
        lidar2cams = []
        cam_intrinsics = []
        for cam_name in CAMERA_ORDER:
            cam2ego = np.asarray(calibration["camera2ego"][cam_name], dtype=np.float64)
            cam2image = np.asarray(calibration["camera2image"][cam_name], dtype=np.float64)
            viewpad = np.eye(4, dtype=np.float64)
            viewpad[: cam2image.shape[0], : cam2image.shape[1]] = cam2image
            lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
            lidar2img = viewpad @ lidar2cam
            lidar2cams.append(lidar2cam)
            lidar2imgs.append(lidar2img)
            cam_intrinsics.append(cam2image)

        input_dict["lidar2img"] = lidar2imgs
        input_dict["lidar2cam"] = lidar2cams
        input_dict["cam_intrinsic"] = cam_intrinsics

        data_info = self.dataset.data_infos[idx]
        self._last_debug_info = {
            "timestamp_us": timestamp_us,
            "dataset_index": idx,
            "dataset_timestamp_us": int(self.timestamps[idx]),
            "dataset_sample_token": data_info.get("token"),
            "camera_order": CAMERA_ORDER,
            "image_shape": list(image_shape),
            "aug_config": self._jsonable(aug_config),
            "command": command_idx,
            "canbus": self._jsonable(canbus),
            "ego_status": self._jsonable(ego_status),
            "hist_traj": self._jsonable(hist_rel),
            "hist_deltas": self._jsonable(hist_deltas),
            "lidar2ego": self._jsonable(lidar2ego),
            "lidar2global": self._jsonable(lidar2global),
            "lidar2cam": self._jsonable(lidar2cams),
            "lidar2img": self._jsonable(lidar2imgs),
            "camera_intrinsics": self._jsonable(cam_intrinsics),
            "server": self.server_info,
        }

        return self.dataset.pipeline(input_dict)

    def _augmentation_for_image_shape(self, image_shape: tuple[int, int]) -> dict[str, Any] | None:
        aug_config = self.dataset.get_augmentation()
        data_aug_conf = getattr(self.dataset, "data_aug_conf", None)
        if aug_config is None or data_aug_conf is None:
            return aug_config

        image_h, image_w = image_shape
        final_h, final_w = data_aug_conf["final_dim"]
        resize = max(final_h / image_h, final_w / image_w)
        resized_w, resized_h = int(image_w * resize), int(image_h * resize)
        bot_pct = float(np.mean(data_aug_conf["bot_pct_lim"]))
        crop_h = int((1 - bot_pct) * resized_h) - final_h
        crop_w = int(max(0, resized_w - final_w) / 2)
        adjusted = dict(aug_config)
        adjusted.update(
            {
                "resize": resize,
                "resize_dims": (resized_w, resized_h),
                "crop": (crop_w, crop_h, crop_w + final_w, crop_h + final_h),
                "flip": False,
                "rotate": 0,
                "rotate_3d": 0,
            }
        )
        if not self._printed_aug_config:
            print(
                "NCAP image augmentation: "
                f"input_hw={image_shape}, final_hw=({final_h}, {final_w}), "
                f"resize={resize:.6f}, crop={adjusted['crop']}",
                flush=True,
            )
            self._printed_aug_config = True
        return adjusted

    def _write_images(self, encoded_images: dict[str, str], timestamp_us: int) -> tuple[list[str], tuple[int, int]]:
        paths = []
        image_shape = None
        for cam_name in CAMERA_ORDER:
            if cam_name not in encoded_images:
                raise KeyError(f"Missing camera image: {cam_name}")
            image = torch.load(io.BytesIO(base64.b64decode(encoded_images[cam_name])), map_location="cpu")
            image_np = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
            image_np = np.asarray(image_np, dtype=np.uint8)
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                raise ValueError(
                    f"Expected {cam_name} image shape HxWx3, got {image_np.shape}"
                )
            if image_shape is None:
                image_shape = tuple(int(x) for x in image_np.shape[:2])
            elif tuple(image_np.shape[:2]) != image_shape:
                raise ValueError(
                    f"Camera image shape mismatch: {cam_name} has {image_np.shape[:2]}, "
                    f"expected {image_shape}"
                )
            path = Path(self.tmpdir.name) / f"current_{cam_name}.png"
            Image.fromarray(np.ascontiguousarray(image_np)).save(
                path, format="PNG", compress_level=1
            )
            paths.append(str(path))
        if image_shape is None:
            raise ValueError("No images were provided")
        return paths, image_shape

    def _ego_status_from_canbus(self, canbus: np.ndarray) -> np.ndarray:
        status = np.zeros((10,), dtype=np.float32)
        if canbus.shape[0] >= 16:
            status[0:3] = canbus[7:10]
            status[3:6] = canbus[10:13]
            status[6:9] = canbus[13:16]
        if canbus.shape[0] >= 17:
            status[9] = canbus[16]
        return status

    def _history_in_current_lidar(self, lidar2global: np.ndarray, timestamp_us: int) -> np.ndarray:
        if self.pose_history and timestamp_us <= self.pose_history[-1][0]:
            self.pose_history.clear()
        self.pose_history.append((timestamp_us, lidar2global.copy()))

        cutoff_us = timestamp_us - 2_000_000
        self.pose_history = [entry for entry in self.pose_history if entry[0] >= cutoff_us]
        sample_times = [timestamp_us - offset for offset in (1_500_000, 1_000_000, 500_000, 0)]
        sampled_poses = [
            min(self.pose_history, key=lambda entry: abs(entry[0] - sample_time))[1]
            for sample_time in sample_times
        ]

        global2lidar = np.linalg.inv(lidar2global)
        points = []
        for pose in sampled_poses:
            point_world = np.array([pose[0, 3], pose[1, 3], pose[2, 3], 1.0], dtype=np.float64)
            point_lidar = global2lidar @ point_world
            points.append(point_lidar[:2])
        return np.asarray(points, dtype=np.float32)

    def _relative_points_to_deltas(self, points: np.ndarray) -> np.ndarray:
        deltas = np.zeros_like(points, dtype=np.float32)
        deltas[0] = points[0]
        deltas[1:] = points[1:] - points[:-1]
        return deltas

    def _command_index(self, command: Any) -> int:
        try:
            command_idx = int(command)
        except (TypeError, ValueError):
            command_idx = 2
        if command_idx not in (0, 1, 2):
            command_idx = 2
        return command_idx

    def _command_onehot(self, command: Any) -> np.ndarray:
        onehot = np.zeros((3,), dtype=np.float32)
        onehot[self._command_index(command)] = 1.0
        return onehot

    def _jsonable(self, value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.float()
            return tensor.cpu().tolist()
        if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
            return self._jsonable(value.tensor)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, tuple):
            return [self._jsonable(x) for x in value]
        if isinstance(value, list):
            return [self._jsonable(x) for x in value]
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        return value

    def _extract_trajectory(self, result: Any) -> list[list[float]]:
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            result = result.get("img_bbox", result)
        traj = result.get("final_planning") if isinstance(result, dict) else None
        if traj is None:
            raise RuntimeError(f"Could not find final_planning in model output: {type(result)}")
        if torch.is_tensor(traj):
            traj = traj.detach().cpu().float().numpy()
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim == 3:
            traj = traj[0]
        traj = traj[:6, :2]
        if traj.shape != (6, 2):
            raise RuntimeError(f"Expected trajectory shape (6, 2), got {traj.shape}")
        return traj.tolist()

    def _extract_aux_outputs(self, result: Any) -> dict[str, Any]:
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            result = result.get("img_bbox", result)
        if not isinstance(result, dict):
            return self._empty_aux_outputs()

        if not self._printed_result_keys:
            print(f"NCAP model output keys: {sorted(result.keys())}", flush=True)
            self._printed_result_keys = True

        boxes = result.get("boxes_3d")
        scores = result.get("scores_3d")
        labels = result.get("labels_3d")
        instance_ids = result.get("instance_ids")
        future_trajs = result.get("trajs_3d")
        raw_output = self._extract_raw_output(result)

        if boxes is None:
            return self._empty_aux_outputs(raw_output)

        boxes_np = self._as_numpy(boxes).astype(np.float32)
        if boxes_np.ndim != 2 or boxes_np.shape[1] < 7 or boxes_np.shape[0] == 0:
            return self._empty_aux_outputs(raw_output)

        # UniDriveVLA boxes are [x, y, z, width, length, height, yaw, ...].
        objects_in_bev = boxes_np[:, [0, 1, 4, 3, 6]].astype(np.float32)
        n_objects = objects_in_bev.shape[0]

        scores_np = self._as_numpy(scores).reshape(-1).astype(np.float32) if scores is not None else np.ones((n_objects,), dtype=np.float32)
        labels_np = self._as_numpy(labels).reshape(-1).astype(np.int64) if labels is not None else np.full((n_objects,), -1, dtype=np.int64)
        if scores_np.shape[0] != n_objects:
            scores_np = np.resize(scores_np, (n_objects,)).astype(np.float32)
        if labels_np.shape[0] != n_objects:
            labels_np = np.resize(labels_np, (n_objects,)).astype(np.int64)

        if instance_ids is not None:
            ids_np = self._as_numpy(instance_ids).reshape(-1).astype(np.int64)
            if ids_np.shape[0] != n_objects:
                ids_np = np.arange(n_objects, dtype=np.int64)
        else:
            ids_np = np.arange(n_objects, dtype=np.int64)

        classes = [
            self.dataset.CLASSES[int(label)] if 0 <= int(label) < len(self.dataset.CLASSES) else "unknown"
            for label in labels_np
        ]

        if future_trajs is not None:
            future_np = self._as_numpy(future_trajs).astype(np.float32)
            if future_np.ndim != 4 or future_np.shape[0] != n_objects or future_np.shape[-1] < 2:
                future_np = np.empty((n_objects, 0, 0, 2), dtype=np.float32)
            else:
                future_np = future_np[..., :2]
        else:
            future_np = np.empty((n_objects, 0, 0, 2), dtype=np.float32)

        print(f"NCAP aux outputs: {n_objects} objects", flush=True)
        return {
            "objects_in_bev": objects_in_bev.tolist(),
            "object_classes": classes,
            "object_scores": scores_np.tolist(),
            "object_ids": ids_np.tolist(),
            "future_trajs": future_np.tolist(),
            "raw_output": raw_output,
        }

    def _extract_raw_output(self, result: dict[str, Any]) -> dict[str, Any]:
        def value(key: str, default: Any = None) -> Any:
            raw = result.get(key, default)
            return self._jsonable(raw) if raw is not None else None

        map_classes = getattr(self.dataset, "MAP_CLASSES", ())
        return {
            "schema_version": 1,
            "box_coordinate_frame": "lidar",
            "box_fields": ["x", "y", "z", "width", "length", "height", "yaw", "vx", "vy"],
            "boxes_3d": value("boxes_3d", []),
            "scores_3d": value("scores_3d", []),
            "class_scores_3d": value("cls_scores", []),
            "labels_3d": value("labels_3d", []),
            "instance_ids": value("instance_ids", []),
            "agent_future_trajectories": value("trajs_3d", []),
            "map_coordinate_frame": "lidar",
            "map_vectors": value("vectors", []),
            "map_scores": value("scores", []),
            "map_labels": value("labels", []),
            "map_classes": list(map_classes),
            "ego_status_pred": value("ego_status_pred"),
            "sample_token": value("sample_token"),
            "coc_pred_text": value("coc_pred_text"),
            "coc_gt_text": value("coc_gt_text"),
        }

    def _empty_aux_outputs(self, raw_output: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "objects_in_bev": np.empty((0, 5), dtype=np.float32).tolist(),
            "object_classes": [],
            "object_scores": np.empty((0,), dtype=np.float32).tolist(),
            "object_ids": np.empty((0,), dtype=np.int64).tolist(),
            "future_trajs": np.empty((0, 0, 0, 2), dtype=np.float32).tolist(),
            "raw_output": raw_output or {},
        }

    def _as_numpy(self, value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.float()
            return tensor.cpu().numpy()
        if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
            tensor = value.tensor.detach()
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.float()
            return tensor.cpu().numpy()
        return np.asarray(value)


class Handler(BaseHTTPRequestHandler):
    runner: UniDriveVLARunner | None = None

    def do_GET(self) -> None:
        if self.path == "/alive":
            self._send_json(
                {
                    "ok": True,
                    "ready": self.runner is not None,
                    "server_info": self.runner.server_info if self.runner is not None else None,
                }
            )
            return
        self._send_json({"error": f"Unknown path: {self.path}"}, status=404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8"))

            if self.path == "/reset":
                self.runner.reset()
                self._send_json({"ok": True})
            elif self.path == "/infer":
                self._send_json(self.runner.infer(payload))
            else:
                self._send_json({"error": f"Unknown path: {self.path}"}, status=404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": repr(exc), "traceback": traceback.format_exc()}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[UniDriveVLA server] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniDriveVLA NeuroNCAP model server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Handler.runner = UniDriveVLARunner(args.config, args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"UniDriveVLA NeuroNCAP server listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
