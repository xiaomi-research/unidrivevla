"""Model export for evaluators and sharded training state for resume."""

import logging
import re
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributed.checkpoint.api import CheckpointException

logger = logging.getLogger(__name__)

STEP_DIR = re.compile(r"^step_(\d+)(?:_e(\d+)_b(\d+))?$")
DONE_FILE = "trainer_state.pt"


def canonical(name: str) -> str:
    """Parameter name without DDP and activation-checkpoint wrapper prefixes, as in DCP state dicts."""
    name = name.replace("_checkpoint_wrapped_module.", "")
    return name[len("module."):] if name.startswith("module.") else name


def export_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Full, unsharded weights on rank 0 (CPU); empty dict elsewhere. Collective."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=False),
    )


def training_state(model: nn.Module, optimizer, ema=None, ignore_frozen_params: bool = False):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_sd, optim_sd = get_state_dict(
        model, optimizer, options=StateDictOptions(ignore_frozen_params=ignore_frozen_params))
    state = {"model": model_sd, "optim": optim_sd}
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


def _prune_optimizer_slots(optimizer_state: dict, checkpoint_keys) -> None:
    """Drop get_state_dict's synthetic zero slots that the saved optimizer never initialized."""
    states = optimizer_state.get("state")
    if not isinstance(states, dict):
        raise TypeError("optimizer state must contain a mapping at 'state'")
    checkpoint_keys = set(checkpoint_keys)
    sorted_keys = sorted(checkpoint_keys)

    def exists(prefix):
        if prefix in checkpoint_keys:
            return True
        child_prefix = prefix + "."
        index = bisect_left(sorted_keys, child_prefix)
        return index < len(sorted_keys) and sorted_keys[index].startswith(child_prefix)

    def prune(value, prefix):
        if isinstance(value, dict):
            for name in list(value):
                child_prefix = f"{prefix}.{name}"
                if not exists(child_prefix):
                    del value[name]
                else:
                    prune(value[name], child_prefix)
            return bool(value)
        return prefix in checkpoint_keys

    for name in list(states):
        # Keep the empty per-parameter mapping: PyTorch's set_state_dict requires
        # one for every optimizer parameter, while an empty mapping stays cold.
        prune(states[name], f"optim.state.{name}")


class ResumeCheckpointer:
    """Sharded model + optimizer state via torch.distributed.checkpoint, written in the background."""

    def __init__(self, root: Path, ctx, async_save: bool = True):
        self.root = Path(root)
        self.ctx = ctx
        self.async_save = async_save
        self.pending = None

    def complete(self):
        if not self.root.is_dir():
            return []
        return sorted(
            ((int(m.group(1)), int(m.group(2)) if m.group(2) is not None else -1,
              int(m.group(3)) if m.group(3) is not None else -1), p)
            for p in self.root.iterdir()
            if (m := STEP_DIR.match(p.name)) and (p / DONE_FILE).is_file()
        )

    def latest(self) -> Optional[Path]:
        done = self.complete()
        return done[-1][1] if done else None

    def save(self, step: int, model: nn.Module, optimizer, extra: dict, ema=None, rank_state=None,
             position=None) -> None:
        import shutil

        import torch.distributed.checkpoint as dcp

        self.wait()
        suffix = "" if position is None else f"_e{int(position[0]):04d}_b{int(position[1]):08d}"
        path = self.root / f"step_{step:08d}{suffix}"
        if self.ctx.is_main:
            # Keep the newest finished state until the one being written now is finished too.
            for _, old in self.complete()[:-1]:
                shutil.rmtree(old, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            (path / DONE_FILE).unlink(missing_ok=True)
        if rank_state is not None:
            self.ctx.barrier()
            rank_path = path / f"rank_{self.ctx.rank:05d}.pt"
            temporary = rank_path.with_suffix(".pt.tmp")
            torch.save(rank_state, temporary)
            temporary.replace(rank_path)
            self.ctx.barrier()
        extra = {**extra, "ema": ema is not None, "model_state_complete": True,
                 "rank_state_complete": rank_state is not None, "world_size": self.ctx.world_size}
        state = training_state(model, optimizer, ema)

        # DCP only exchanges plan metadata between ranks; gloo does that on the CPU (object gathers over NCCL
        # have crashed with SIGSEGV here).
        if self.async_save:
            try:
                future = dcp.async_save(state, checkpoint_id=str(path), process_group=self.ctx.gloo_group())
            except (Exception, CheckpointException) as exc:
                from concurrent.futures import Future

                future = Future()
                future.set_exception(exc)
            self.pending = (future, path, extra)
        else:
            error = None
            try:
                dcp.save(state, checkpoint_id=str(path), process_group=self.ctx.gloo_group())
            except (Exception, CheckpointException) as exc:
                error = exc
            self._finish(path, extra, data_error=error)

    def _gather_errors(self, error):
        if self.ctx.world_size == 1:
            return [error]
        import torch.distributed as dist

        errors = [None] * self.ctx.world_size
        dist.all_gather_object(errors, error, group=self.ctx.gloo_group())
        return errors

    def _check_errors(self, stage: str, error):
        message = None if error is None else f"{type(error).__name__}: {error}"
        errors = self._gather_errors(message)
        failed = [f"rank {rank}: {value}" for rank, value in enumerate(errors) if value is not None]
        if failed:
            raise RuntimeError(f"checkpoint {stage} failed ({'; '.join(failed)})") from error

    def _write_done(self, path: Path, extra: dict) -> None:
        temporary = path / (DONE_FILE + ".tmp")
        torch.save(extra, temporary)
        temporary.replace(path / DONE_FILE)
        logger.info("Resume state written to %s", path)

    def _finish(self, path: Path, extra: dict, future=None, data_error=None) -> None:
        error = data_error
        if future is not None:
            try:
                future.result()
            except (Exception, CheckpointException) as exc:
                error = exc
        self._check_errors("data write", error)

        error = None
        if self.ctx.is_main:
            try:
                self._write_done(path, extra)
            except Exception as exc:
                error = exc
        self._check_errors("completion marker write", error)

    def load(self, path: Path, model: nn.Module, optimizer, ema=None, require_rank_state: bool = False) -> dict:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.filesystem import FileSystemReader
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_state_dict

        extra = torch.load(path / DONE_FILE, map_location="cpu", weights_only=False)
        if require_rank_state:
            if not extra.get("rank_state_complete"):
                raise RuntimeError(f"checkpoint {path} has no per-rank state; exact resume needs a new checkpoint")
            if int(extra.get("world_size", -1)) != self.ctx.world_size:
                raise RuntimeError(f"exact resume requires the saved world size {extra.get('world_size')} to match "
                                   f"the current world size {self.ctx.world_size}")
            rank_path = path / f"rank_{self.ctx.rank:05d}.pt"
            if not rank_path.is_file():
                raise RuntimeError(f"checkpoint {path} is missing rank state for rank {self.ctx.rank}")
            extra["_rank_state"] = torch.load(rank_path, map_location="cpu", weights_only=False)
        saved_ema = ema if extra.get("ema") else None
        ignore_frozen = not extra.get("model_state_complete", False)
        state = training_state(model, optimizer, saved_ema, ignore_frozen_params=ignore_frozen)
        metadata = FileSystemReader(str(path)).read_metadata()
        if "optim" in state:
            # PyTorch 2.6's get_state_dict can materialize default optimizer slots for every
            # parameter. Keep only slots present in the source metadata so untouched parameters
            # remain genuinely uninitialized after set_state_dict.
            _prune_optimizer_slots(state["optim"], metadata.state_dict_metadata)
        dcp.load(state, checkpoint_id=str(path), process_group=self.ctx.gloo_group())
        if ignore_frozen:
            # Older resume files omitted frozen tensors; their initialized stage-1 values remain authoritative.
            for key, value in get_model_state_dict(model).items():
                state["model"].setdefault(key, value)
        # anything else missing is a real mismatch
        result = set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optim"],
            options=StateDictOptions(ignore_frozen_params=ignore_frozen, strict=False),
        )
        for parameter, value in list(optimizer.state.items()):
            if isinstance(value, dict) and not value:
                optimizer.state.pop(parameter, None)
        frozen = {canonical(name) for name, p in model.named_parameters() if not p.requires_grad}
        missing = [key for key in result.missing_keys if canonical(key) not in frozen]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"resume state {path} does not match the model: missing {missing[:8]}, "
                               f"unexpected {list(result.unexpected_keys)[:8]}")
        if ema is not None and saved_ema is None:
            logger.info("resume state %s has no EMA; the average restarts from the loaded weights", path)
            ema.reset()
        return extra

    def wait(self) -> None:
        if self.pending is not None:
            future, path, extra = self.pending
            try:
                self._finish(path, extra, future)
            finally:
                self.pending = None
