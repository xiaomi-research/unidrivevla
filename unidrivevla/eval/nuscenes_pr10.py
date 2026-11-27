"""Export native predictions and score them with the pinned PR10 nuScenes evaluator.

Use ``evaluator.name=nuscenes_pr10`` for the Stage2 full score or
``evaluator.name=nuscenes_pr10_motion`` for the Stage3 motion-only score. The
same evaluator is available from both train.py's final evaluation and infer.py.
"""

import json
import math
import os
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from unidrivevla.eval.openloop import OpenLoopEvaluator
from unidrivevla.eval.registry import fresh_eval_dir, register


class _PredictionOnlyNuScenesSource:
    """Reuse source features for PR10 inference without constructing unused training targets."""

    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def sequences(self):
        return self.source.sequences()

    def __getitem__(self, index):
        return self.source.features(index), {}, self.source.tokens[index]


def _append_prediction_rows(rows, tokens, batch):
    if len(tokens) != len(batch):
        raise ValueError("PR10 prediction count does not match batch tokens")
    for token, row in zip(tokens, batch):
        if token is None:  # FSDP equal-forward padding only; never score its repeated frame.
            continue
        if token in rows:
            raise ValueError(f"duplicate PR10 prediction for token {token}")
        rows[token] = {"img_bbox": {key: value.detach().cpu() for key, value in row.items()}}


def required_path(var, *, directory=False):
    value = os.environ.get(var)
    # Keep a venv's Python symlink under its own bin/ so it can find pyvenv.cfg.
    path = Path(value).expanduser().absolute() if value else None
    if path is None or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"nuscenes_pr10 requires {var} to point to an existing {'directory' if directory else 'file'}")
    return path


def optional_pr10_root():
    """Return an external checkout when supplied; ZIP/wheel users can use the pinned bundled scorer."""
    value = os.environ.get("NUSCENES_PR10_ROOT")
    if not value:
        return None
    path = Path(value).expanduser().absolute()
    if not path.is_dir():
        raise ValueError(f"NUSCENES_PR10_ROOT must be an existing directory: {path}")
    return path


def default_pr10_bundle():
    value = os.environ.get("NUSCENES_PR10_BUNDLE")
    path = (Path(value).expanduser().absolute() if value else
            Path(__file__).resolve().parents[1] / "references" / "pr10_nuscenes_scorer.tar.gz")
    if not path.is_file():
        raise ValueError(f"pinned PR10 scorer bundle is missing: {path}")
    return path


@register("nuscenes_pr10")
class NuScenesPR10Evaluator(OpenLoopEvaluator):
    source = "nuscenes"
    motion_only = False

    def __init__(self, cfg, section, output_dir: Path):
        if OmegaConf.select(cfg, "data_loader") != "nuscenes":
            raise ValueError("nuscenes_pr10 requires data=nuscenes")
        self.version = str(OmegaConf.select(cfg, "nuscenes.version", default="v1.0-trainval"))
        if self.version not in ("v1.0-trainval", "v1.0-mini"):
            raise ValueError("PR10 nuScenes evaluators support v1.0-trainval or v1.0-mini")
        if not self.motion_only and self.version != "v1.0-trainval":
            raise ValueError("nuscenes_pr10 Stage2 full scoring requires v1.0-trainval")
        if OmegaConf.select(section, "data_overrides"):
            raise ValueError("configure nuscenes paths directly so PR10 inference and scoring use the same val data")
        if self.motion_only:
            # PR10 Stage3 scores motion only and must not require a planning segmentation PKL.
            source = self.source
            self.source = ""
            try:
                super().__init__(cfg, section, output_dir)
            finally:
                self.source = source
        else:
            super().__init__(cfg, section, output_dir)
        self.every_n_epochs = 0  # PR10 scores the last model, outside the training loop.
        configured_motion = bool(OmegaConf.select(cfg, "agent.with_motion", default=False))
        if configured_motion != self.motion_only:
            requested = "Stage3 motion" if self.motion_only else "Stage2 no-motion"
            raise ValueError(f"{self.benchmark if hasattr(self, 'benchmark') else self.source} PR10 evaluator requires {requested} policy")
        self.pr10_root = optional_pr10_root()
        self.pr10_bundle = None if self.pr10_root else default_pr10_bundle()
        self.scorer_python = required_path("NUSCENES_PR10_PYTHON")
        if not self.motion_only and self.segmentation_path is None:
            raise ValueError("nuscenes_pr10 requires the official NUSCENES_PLANNING_SEG_PKL for the loose score")
        self.pending = []

    def eval_loader(self, trainer):
        """Build the same model inputs as normal inference while skipping unused target conversion."""
        if self.loader is None:
            from unidrivevla.data import LOADERS
            from unidrivevla.data.registry import evaluation_params, with_worker_transform

            select_policy = getattr(trainer.agent, "policy_for_source", None)
            policy = select_policy(self.source) if callable(select_policy) else trainer.agent
            self.dataset = LOADERS[self.source](self.cfg, policy, "val")
            prediction_source = _PredictionOnlyNuScenesSource(self.dataset)
            self.loader = self.make_eval_loader(
                with_worker_transform(prediction_source, policy),
                evaluation_params(trainer.cfg), trainer.ctx,
                pad_eval=getattr(getattr(trainer, "args", None), "strategy", None) == "fsdp",
            )
        return self.loader

    def make_eval_loader(self, dataset, params, ctx, pad_eval=False):
        from unidrivevla.data.registry import make_loader

        if int(params["batch_size"]) != 1:
            raise ValueError("nuscenes_pr10 requires dataloader.eval_batch_size=1 for tracking")
        return make_loader(dataset, params, False, ctx.rank, ctx.world_size,
                           sequence_eval=True, pad_eval=pad_eval)

    @torch.no_grad()
    def during_training(self, trainer, ckpt, tag):
        from unidrivevla.adapters import isolated_evaluation_state
        from unidrivevla.data import DevicePrefetcher
        from unidrivevla.distributed.parallelize import check_eval_loader

        ctx = trainer.ctx
        group = ctx.gloo_group() if ctx.world_size > 1 else None
        prepared = [None, None]
        if ctx.is_main:
            try:
                output = fresh_eval_dir(self.root, tag).resolve()
                (output / "predictions").mkdir()
                prepared[0] = str(output)
            except OSError as exc:
                prepared[1] = f"{type(exc).__name__}: {exc}"
        if ctx.world_size > 1:
            dist.broadcast_object_list(prepared, src=0, group=group)
        if prepared[1]:
            if ctx.is_main:
                self.results.append((tag, {"error": prepared[1]}))
            return

        output = Path(prepared[0])
        error = None
        try:
            loader = self.eval_loader(trainer)
            check_eval_loader(trainer.model, loader, ctx)
            rows = {}
            with isolated_evaluation_state(trainer.model, trainer.agent):
                for features, _, tokens in DevicePrefetcher(loader, ctx.device):
                    metas = features["img_metas"]
                    if len(metas) != len(tokens):
                        raise ValueError("PR10 image metadata does not match batch tokens")
                    if any(not isinstance(meta, dict) or not meta.get("stream_id") for meta in metas):
                        raise ValueError("PR10 sequence loader did not attach a temporal stream ID")
                    export_key = "nuscenes_pr10_motion" if self.motion_only else "nuscenes_pr10"
                    export_flag = "return_nuscenes_pr10_motion" if self.motion_only else "return_nuscenes_pr10"
                    batch = trainer.model({**features, "img_metas": metas, export_flag: True})[export_key]
                    _append_prediction_rows(rows, tokens, batch)
            with (output / "predictions" / f"rank-{ctx.rank:04d}.pt").open("xb") as file:
                torch.save(rows, file)
        except Exception as exc:
            error = f"rank {ctx.rank}: {type(exc).__name__}: {exc}"

        errors = [error]
        if ctx.world_size > 1:
            errors = [None] * ctx.world_size
            dist.all_gather_object(errors, error, group=group)
        if ctx.is_main:
            failures = [item for item in errors if item]
            if failures:
                self.results.append((tag, {"error": "; ".join(failures)}))
            else:
                self.pending.append((tag, output))

    def finish(self, final_ckpt):
        # Run the CPU scorer after infer.py/Trainer releases the model on every rank.
        scorer = Path(__file__).resolve().parents[1] / "tools" / "score_nuscenes_pr10.py"
        for tag, output in self.pending:
            log = output / "eval.log"
            command = [str(self.scorer_python), str(scorer)]
            if self.pr10_root is not None:
                command.extend(("--pr10-root", str(self.pr10_root)))
            else:
                command.extend(("--reference-bundle", str(self.pr10_bundle)))
            command.extend(("--infos", str(Path(self.cfg.nuscenes.info_dir).expanduser().resolve()),
                       "--data-root", str(Path(self.cfg.nuscenes.data_root).expanduser().resolve()),
                       "--predictions", str(output / "predictions"), "--output-dir", str(output),
                       "--version", self.version))
            if self.motion_only:
                command.append("--motion-only")
            else:
                command.extend(("--segmentation", str(self.segmentation_path.resolve())))
            try:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
                # Do not inherit the model environment's Python modules or distributed rank settings.
                for name in ("PYTHONPATH", "PYTHONHOME", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
                    env.pop(name, None)
                with log.open("xb") as file:
                    subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=file, stderr=subprocess.STDOUT,
                                   env=env)
                result_name = "nuscenes_pr10_motion_metrics.json" if self.motion_only else "nuscenes_pr10_metrics.json"
                metrics = json.loads((output / result_name).read_text(encoding="utf-8"))
                keys = (tuple(f"{name}_{metric}" for name in ("car", "pedestrian")
                              for metric in ("EPA", "min_ade_err", "min_fde_err", "miss_rate_err"))
                        if self.motion_only else (
                        "img_bbox_NuScenes/NDS", "img_bbox_NuScenes/amota", "mAP_normal", "L2", "ego_status/mae")
                        )
                for key in keys:
                    if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]):
                        raise ValueError(f"PR10 scorer did not return a finite {key}")
                self.results.append((tag, metrics))
            except Exception as exc:
                self.results.append((tag, {"error": f"{type(exc).__name__}: {exc}; see {log}"}))
        self.pending.clear()
        return self.poll()


@register("nuscenes_pr10_motion")
class NuScenesPR10MotionEvaluator(NuScenesPR10Evaluator):
    """Stage3's optional PR10 6 s / 12-step car and pedestrian motion scorer."""

    motion_only = True
