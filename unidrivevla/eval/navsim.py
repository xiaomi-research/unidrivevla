"""NAVSIM PDMS (1.1) / EPDMS (2.0) through each tree's own scoring entry: navtest-style splits with the one-stage
entry, the 2.0 two-stage splits (navhard_two_stage, navsafe_two_stage, ...) with the official two-stage one.

During training the trainer's model generates the trajectories on the training GPUs (like validation), second-stage
synthetic scenes included, and the official scoring runs on CPU in the background with a replay agent, so evaluation
never loads a second model onto the GPUs training uses. The final evaluation runs the trained agent once training
has ended. The agent's eval forward returns `pred_traj`, (batch, poses, 3) in the ego frame at 0.5 s spacing.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import OmegaConf

from unidrivevla.eval.registry import SubprocessEvaluator, average_row, fresh_eval_dir, register

logger = logging.getLogger(__name__)

DEVKITS = ("navsim1.1", "navsim2.0")
# One agent process: a worker pool would build one 3B model per worker.
FINAL_OVERRIDES = ["worker=sequential"]
TWO_STAGE_ENTRY = "run_pdm_score.py"  # NAVSIM 2.0 two-stage EPDMS
POSE_INTERVAL_S = 0.5


def devkit_root() -> Path:
    import navsim

    return Path(navsim.__file__).resolve().parents[1]


def scoring_entry(root: Path, two_stage: bool) -> str:
    """The 2.x releases that score two-stage splits keep one-stage scoring in run_pdm_score_one_stage.py; NAVSIM 1.1
    and 2.0 itself (v2.0) score with run_pdm_score.py."""
    if two_stage:
        return TWO_STAGE_ENTRY
    one_stage = root / "navsim" / "planning" / "script" / "run_pdm_score_one_stage.py"
    return one_stage.name if one_stage.is_file() else "run_pdm_score.py"


@register("navsim")
class NavsimEvaluator(SubprocessEvaluator):
    def __init__(self, cfg, section, output_dir: Path):
        super().__init__(cfg, section, output_dir)
        self.loader = None

    def split(self) -> str:
        return str(OmegaConf.select(self.section, "split") or "navtest")

    def split_overrides(self) -> List[str]:
        return [f"train_test_split={self.split()}"] + [str(o) for o in (OmegaConf.select(self.section, "split_overrides") or [])]

    def two_stage(self, root: Path) -> bool:
        if root.name == "navsim2.0" and self.split().endswith("two_stage"):
            return True
        if self.split().endswith("two_stage"):
            raise ValueError(f"split {self.split()} needs the NAVSIM 2.0 devkit, found {root}")
        return False

    def overrides(self, config_dir: Path, out_dir: Path, tag: str, agent: str) -> List[str]:
        out = [
            f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{config_dir.resolve().as_posix()}]",
            f"agent={agent}",
            *self.split_overrides(),
            f"experiment_name={OmegaConf.select(self.cfg, 'experiment_name') or 'unidrivevla'}_{tag}",
            f"output_dir={out_dir.resolve().as_posix()}",
        ]
        metric_cache = OmegaConf.select(self.section, "metric_cache_path")
        if metric_cache:
            out.append(f"metric_cache_path={metric_cache}")
        return out + [str(o) for o in (OmegaConf.select(self.section, "overrides") or [])]

    def agent_config(self, ckpt: Path, config_dir: Path) -> None:
        """The NAVSIM wrapper around the trained policy: canonical NAVSIM features, the policy's transform and forward."""
        policy = OmegaConf.to_container(self.cfg.agent, resolve=True)
        policy["checkpoint_path"] = str(ckpt)
        agent = {"_target_": "unidrivevla.adapters.navsim_agent.UniDriveVLANavsimAgent", "_convert_": "all", "policy": policy}
        (config_dir / "agent").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(agent), config_dir / "agent" / "trained_agent.yaml")

    def command(self, ckpt: Path, out_dir: Path, tag: str) -> List[str]:
        root = devkit_root()
        if root.name not in DEVKITS:
            raise ValueError(f"cannot pick a NAVSIM scoring entry for devkit {root}")
        config_dir = out_dir / "config"
        self.agent_config(ckpt, config_dir)
        overrides = self.overrides(config_dir, out_dir, tag, "trained_agent") + FINAL_OVERRIDES
        entry = str(root / "navsim" / "planning" / "script" / scoring_entry(root, self.two_stage(root)))
        return [sys.executable, entry, *overrides]

    def replay_command(self, trajectories: Path, out_dir: Path, tag: str) -> List[str]:
        root = devkit_root()
        config_dir = out_dir / "config"
        (config_dir / "agent").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create({"_target_": "unidrivevla.eval.navsim_replay.ReplayAgent",
                              "trajectories_path": str(trajectories.resolve())}),
            config_dir / "agent" / "replay_agent.yaml",
        )
        entry = str(root / "navsim" / "planning" / "script" / scoring_entry(root, self.two_stage(root)))
        return [sys.executable, entry, *self.overrides(config_dir, out_dir, tag, "replay_agent")]

    def release(self) -> None:
        self.loader = None  # its worker transform holds the policy

    def during_training(self, trainer, ckpt: Path, tag: str) -> None:
        trajectories = self.generate(trainer, tag)
        if trainer.ctx.is_main and trajectories is not None:
            out_dir = trajectories.parent
            try:
                env = {**self.child_env(), "CUDA_VISIBLE_DEVICES": ""}
                self.launch(self.replay_command(trajectories, out_dir, tag), out_dir, tag, env)
            except Exception as exc:
                self.fail(tag, f"{type(exc).__name__}: {exc}", out_dir)
                logger.exception("NAVSIM replay evaluation %s could not start", tag)

    def eval_loader(self, trainer):
        if self.loader is None:
            from hydra import compose
            from hydra.core.hydra_config import HydraConfig

            from unidrivevla.data import make_loader
            from unidrivevla.data.navsim import navsim_loader
            from unidrivevla.data.registry import evaluation_params, with_worker_transform

            split_cfg = compose(config_name=HydraConfig.get().job.config_name, overrides=self.split_overrides())
            select_policy = getattr(trainer.agent, "policy_for_source", None)
            policy = select_policy("navsim") if callable(select_policy) else trainer.agent
            dataset = with_worker_transform(navsim_loader(split_cfg, policy, "test"), policy)
            self.loader = make_loader(dataset, evaluation_params(trainer.cfg), False, trainer.ctx.rank,
                                      trainer.ctx.world_size)
        return self.loader

    def generate(self, trainer, tag: str) -> Optional[Path]:
        """Trajectories of every split scene from the trainer's model, gathered into one pickle on rank 0."""
        from unidrivevla.adapters import predict

        merged = predict(trainer.model, self.eval_loader(trainer), trainer.ctx)
        if not trainer.ctx.is_main:
            return None
        out_dir = None
        try:
            out_dir = fresh_eval_dir(self.root, tag)
            from navsim.common.dataclasses import Trajectory
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

            trajectories = {
                token: Trajectory(poses, TrajectorySampling(num_poses=len(poses), interval_length=POSE_INTERVAL_S))
                for token, poses in merged.items()
            }
            path = out_dir / "trajectories.pkl"
            with open(path, "xb") as stream:
                pickle.dump(trajectories, stream)
            logger.info("eval %s: %d trajectories from the training model -> %s", out_dir.name,
                        len(trajectories), path)
            return path
        except Exception as exc:
            self.fail(tag, f"{type(exc).__name__}: {exc}", out_dir)
            logger.exception("NAVSIM trajectory export %s failed", tag)
            return None

    def read_result(self, out_dir: Path) -> Dict[str, float]:
        csvs = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            raise ValueError(f"no score csv in {out_dir}")
        return {**average_row(csvs[-1]), "csv": str(csvs[-1])}
