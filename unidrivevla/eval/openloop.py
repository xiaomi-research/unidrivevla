"""Open-loop planning evaluation with the active main nuScenes/B2D metrics."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from omegaconf import OmegaConf

from unidrivevla.eval.planning_metrics import load_nuscenes_segmentation, planning_metrics
from unidrivevla.eval.registry import fresh_eval_dir, register

logger = logging.getLogger(__name__)


class OpenLoopEvaluator:
    """Run model prediction once, then score source-specific targets on CPU."""

    in_process = True
    source = ""

    def __init__(self, cfg, section, output_dir: Path):
        self.cfg = cfg
        self.section = section
        self.root = output_dir / "eval"
        self.every_n_epochs = int(OmegaConf.select(section, "every_n_epochs") or 0)
        self.on_end = bool(OmegaConf.select(section, "on_end", default=True))
        self.dataset = self.loader = None
        self.results = []
        self.segmentation_path = None
        self.segmentation = None
        if self.source == "nuscenes":
            configured = (OmegaConf.select(section, "seg_pkl_path")
                          or os.environ.get("NUSCENES_PLANNING_SEG_PKL"))
            infos = OmegaConf.select(cfg, "nuscenes.info_dir")
            default = Path(infos) / "planing_gt_segmentation_val" if infos else (
                Path(__file__).resolve().parents[2] / "nuScenes" / "data" / "infos" / "planing_gt_segmentation_val")
            path = Path(configured).expanduser() if configured else default
            if path.is_file():
                self.segmentation_path = path
            elif configured:
                raise FileNotFoundError(f"configured nuScenes planning segmentation PKL not found: {path}")

    def eval_loader(self, trainer):
        if self.loader is None:
            from unidrivevla.data import LOADERS, make_loader
            from unidrivevla.data.registry import evaluation_params, with_worker_transform

            source_cfg = OmegaConf.merge(OmegaConf.create(), self.cfg,
                                         OmegaConf.select(self.section, "data_overrides") or {})
            select_policy = getattr(trainer.agent, "policy_for_source", None)
            policy = select_policy(self.source) if callable(select_policy) else trainer.agent
            self.dataset = LOADERS[self.source](source_cfg, policy, "val")
            self.loader = make_loader(with_worker_transform(self.dataset, policy), evaluation_params(trainer.cfg),
                                      False, trainer.ctx.rank, trainer.ctx.world_size)
        return self.loader

    def during_training(self, trainer, ckpt: Path, tag: str) -> None:
        """Called on every rank with the model holding the weights to score."""
        from unidrivevla.adapters import predict

        predictions = predict(trainer.model, self.eval_loader(trainer), trainer.ctx)
        if not trainer.ctx.is_main:
            return
        out_dir = None
        try:
            out_dir = fresh_eval_dir(self.root, tag)
            if self.segmentation_path is not None and self.segmentation is None:
                self.segmentation = load_nuscenes_segmentation(str(self.segmentation_path))
            select_policy = getattr(trainer.agent, "policy_for_source", None)
            policy = select_policy(self.source) if callable(select_policy) else trainer.agent
            sampling = policy.trajectory_sampling
            horizon_s = float(sampling.num_poses * sampling.interval_length)
            if horizon_s < 3.0:
                raise ValueError(f"planning metrics need a 3 s horizon, the model predicts {horizon_s} s")
            samples = [(token, poses, self.dataset.eval_info(token)) for token, poses in predictions.items()]
            metrics = self.score(samples)
            if self.source == "nuscenes" and self.segmentation is None:
                metrics["nuscenes/loose_status"] = "no planning segmentation PKL found; main loose metric unavailable"
            with open(out_dir / f"{self.source}_metrics.json", "x", encoding="utf-8") as stream:
                json.dump(metrics, stream, indent=2)
            self.results.append((tag, metrics))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if out_dir is not None:
                try:
                    with open(out_dir / "eval_error.txt", "x", encoding="utf-8") as stream:
                        stream.write(message + "\n")
                except OSError:
                    pass
            logger.exception("%s evaluation %s failed", self.source, tag)
            location = f", see {out_dir}" if out_dir is not None else ""
            self.results.append((tag, {"error": message + location}))

    def score(self, samples):
        return planning_metrics(samples, self.source, self.segmentation)

    def poll(self):
        results, self.results = self.results, []
        return results

    def release(self) -> None:
        self.dataset = self.loader = None

    def finish(self, final_ckpt: Optional[Path]):
        return self.poll()


@register("nuscenes")
class NuScenesEvaluator(OpenLoopEvaluator):
    source = "nuscenes"


@register("b2d")
class Bench2DriveEvaluator(OpenLoopEvaluator):
    source = "b2d"
