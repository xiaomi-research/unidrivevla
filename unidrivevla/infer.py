"""torchrun unidrivevla/infer.py data=navsim|nuscenes|b2d agent.checkpoint_path=CKPT [overrides]: scores a trained
checkpoint into a fresh output directory. In-process metrics run here; subprocess metrics run after GPU release."""

import gc
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # the package's parent, as in train.py

# Keep evaluation output separate and reject reused paths before Hydra writes config files.
if __name__ == "__main__":
    from unidrivevla.run_paths import prepare_run_paths

    prepare_run_paths("infer", sys.argv[1:])

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from unidrivevla.adapters import initialize_for_inference, load_checkpoint_agent_config  # noqa: E402
from unidrivevla.distributed import parallelize  # noqa: E402
from unidrivevla.train import Trainer  # noqa: E402


def restore_checkpoint_agent_config(cfg: DictConfig, checkpoint: Path, vlm_path_override=None) -> bool:
    metadata = load_checkpoint_agent_config(checkpoint)
    if metadata is None:
        return False
    agent = OmegaConf.create(metadata)
    target = metadata["_target_"]
    data_loader = str(OmegaConf.select(cfg, "data_loader") or "")
    if target == "unidrivevla.adapters.multidomain.MultiProfilePolicy":
        profiles = OmegaConf.to_container(agent.profiles, resolve=True)
        sources = (OmegaConf.select(cfg, "multidomain.sources") or []) if data_loader == "multidomain" else [
            {"name": data_loader}
        ]
        missing = sorted({str(source["name"]) for source in sources} - set(profiles))
        if missing:
            raise ValueError(f"checkpoint profiles do not include inference source(s): {missing}")
        agent.exact_resume = False
    else:
        saved_profile = str(OmegaConf.select(agent, "profile.name"))
        configured_profile = str(OmegaConf.select(cfg, "profile.name") or "")
        if data_loader == "multidomain" or (configured_profile and configured_profile != saved_profile):
            raise ValueError(f"checkpoint profile {saved_profile!r} does not match data={data_loader!r}")
    agent.checkpoint_path = str(checkpoint)
    agent.train_vlm = False
    agent.gradient_checkpointing = False
    if vlm_path_override:
        agent.vlm_path = str(vlm_path_override)
    # Trainer instantiates cfg.agent after device setup; Hydra's node-level flag keeps checkpoint data inert.
    agent._recursive_ = False
    OmegaConf.update(cfg, "agent", agent, merge=False)
    return True


def _requested_vlm_path(cfg, task_overrides, environment):
    for override in task_overrides:
        key = str(override).split("=", 1)[0].lstrip("+")
        if key == "agent.vlm_path":
            return str(OmegaConf.select(cfg, "agent.vlm_path"))
    return environment or None


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg.agent.checkpoint_path:
        raise ValueError("infer.py needs agent.checkpoint_path")
    ckpt = Path(cfg.agent.checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"inference checkpoint does not exist: {ckpt}")
    from hydra.core.hydra_config import HydraConfig

    vlm_path = _requested_vlm_path(cfg, HydraConfig.get().overrides.task,
                                   os.environ.get("VLM_PRETRAINED_PATH"))
    restore_checkpoint_agent_config(cfg, ckpt, vlm_path)
    trainer = Trainer(cfg)
    live = [e for e in trainer.evaluators if getattr(e, "in_process", False)]
    if live:
        initialize_for_inference(trainer.agent)
        trainer.model = parallelize(trainer.agent, trainer.args, trainer.ctx)
        for evaluator in live:
            evaluator.during_training(trainer, ckpt, "infer")
    trainer.model = trainer.agent = None
    for evaluator in trainer.evaluators:
        evaluator.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer.ctx.close()
    if trainer.ctx.is_main:
        for evaluator in trainer.evaluators:
            trainer.report(evaluator, evaluator.finish(ckpt))
    trainer.monitor.close()


if __name__ == "__main__":
    main()
