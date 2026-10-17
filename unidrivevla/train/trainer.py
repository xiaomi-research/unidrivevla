"""Training loop for policy adapters (unidrivevla.adapters). Parallelism, data sources and evaluators are plugged in;
the loop knows neither the model nor the dataset."""

import contextlib
import gc
import hashlib
import inspect
import json
import logging
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from unidrivevla.adapters import check_policy, isolated_evaluation_state
from unidrivevla.data import DevicePrefetcher, build_split, import_plugins, loader_name, make_loader
from unidrivevla.data.augment import geometry
from unidrivevla.data.registry import SequenceBatchSampler, evaluation_params
from unidrivevla.distributed import DistributedContext, ResumeCheckpointer, export_model_state, gradient_sync, parallelize
from unidrivevla.eval import build_evaluators
from unidrivevla.train.args import TrainArgs
from unidrivevla.train.ema import EMA
from unidrivevla.train.monitor import Monitor

logger = logging.getLogger("unidrivevla")

METRIC_PREFIX = "loss_"  # scalar tensor attributes of the training output logged next to the loss
EMA_SUFFIX = "-EMA"
_COMPUTE_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}


def _replicated_parameter_names(module, patterns):
    """Canonical names of parameters excluded from FSDP2 sharding by the existing policy."""
    from unidrivevla.distributed.checkpoint import canonical
    from unidrivevla.distributed.parallelize import replicated_params

    replicated = replicated_params(module, patterns)
    return {canonical(name) for name, parameter in module.named_parameters()
            if parameter in replicated and parameter.is_floating_point()}


def _compute_parameter_dtypes(module, args, fsdp_replicated_names=None):
    """Describe forward parameter dtypes after parallelize(), including FSDP2 mixed precision."""
    from unidrivevla.distributed.checkpoint import canonical

    strategy = args.strategy
    if strategy == "fsdp":
        if args.precision not in ("bf16", "fp32"):
            raise ValueError(f"cannot export FSDP compute parameter dtypes for precision={args.precision!r}")
        compute_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
        if fsdp_replicated_names is None:
            fsdp_replicated_names = _replicated_parameter_names(module, args.replicate_frozen)
        else:
            fsdp_replicated_names = {canonical(name) for name in fsdp_replicated_names}
    elif strategy == "ddp":
        fsdp_replicated_names = set()
        compute_dtype = None
    else:
        raise ValueError(f"cannot export compute parameter dtypes for strategy={strategy!r}")

    result = {}
    identities = {}
    for name, parameter in module.named_parameters():
        name = canonical(name)
        if not parameter.is_floating_point():
            continue
        dtype = parameter.dtype
        if strategy == "fsdp" and name not in fsdp_replicated_names:
            dtype = compute_dtype
        dtype_name = _COMPUTE_DTYPE_NAMES.get(dtype)
        if dtype_name is None:
            raise ValueError(f"unsupported compute parameter dtype {dtype} for {name!r}")
        if name in result:
            if identities[name] is not parameter:
                raise ValueError(f"canonical parameter name collision for {name!r}")
            continue
        identities[name] = parameter
        result[name] = dtype_name
    if strategy == "fsdp":
        missing = fsdp_replicated_names - set(result)
        if missing:
            raise ValueError(f"FSDP replicated parameters are absent from the model: {sorted(missing)[:5]}")
    return result


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _device_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_tree(item, device) for item in value)
    return value


class Trainer:
    def __init__(self, cfg: DictConfig):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.cfg = cfg
        self.args = TrainArgs.from_cfg(cfg)
        if self.args.exact_resume:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.ctx = DistributedContext(self.args.nccl_timeout)
        self.args.validate(self.ctx.world_size)
        if self.args.exact_resume:
            torch.use_deterministic_algorithms(True)
            if self.ctx.device.type == "cuda":
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        if self.args.seed is not None:
            seed = int(self.args.seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        import_plugins(cfg)
        self.agent = instantiate(cfg.agent)
        check_policy(self.agent)
        self.output_dir = Path(cfg.output_dir)
        if self.ctx.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = Monitor(cfg, self.output_dir, self.ctx.is_main)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_interval = "epoch"
        self.train_loader = None
        self.val_loader = None
        self.clip_params = []
        self.ema = None
        self.global_batch = 0
        self.checkpointer = ResumeCheckpointer(self.output_dir / "checkpoints", self.ctx, self.args.async_save)
        self.evaluators = build_evaluators(cfg, self.output_dir)
        self.start_epoch = 0
        self.resume_batch = 0
        self._resume_rank_state = None
        self._resume_epoch_complete = False
        self.epoch = 0
        self.global_step = 0
        self.best_val = math.inf
        self.micro = 0
        self.batch_in_epoch = 0

    def fit(self) -> None:
        self.build_data()
        if hasattr(self.agent, "initialize"):
            self.agent.initialize()
        if self.args.exact_resume and not all(callable(getattr(self.agent, name, None)) for name in
                                              ("training_runtime_state_dict", "load_training_runtime_state_dict")):
            raise ValueError("train.exact_resume requires the agent to implement training runtime-state save/load hooks")
        if self.args.strategy == "fsdp":
            # FSDP2 composes onto the module during parallelize(). Snapshot the ignored parameters using
            # the same helper as that path, before managed parameters become DTensors.
            self._fsdp_replicated_parameter_names = _replicated_parameter_names(
                self.agent, self.args.replicate_frozen)
        self.model = parallelize(self.agent, self.args, self.ctx)
        self.build_optimizer()
        if self.args.ema_decay > 0:
            # EMA scaling rule: the same averaging horizon in samples at any global batch size.
            decay = self.args.ema_decay ** (self.global_batch / self.args.ema_reference_batch)
            self.ema = EMA(self.agent, decay)
            if self.ctx.is_main:
                logger.info("ema decay=%.6f (%s at global batch %d, running %d) over %d trainable tensors", decay,
                            self.args.ema_decay, self.args.ema_reference_batch, self.global_batch, len(self.ema.params))
        if self.args.resume:
            self.resume()
        try:
            try:
                self.train_loop()
            finally:
                self.checkpointer.wait()
            live = [evaluator for evaluator in self.evaluators
                    if getattr(evaluator, "in_process", False) and evaluator.on_end]
            if live:
                with self.averaged():
                    for evaluator in live:
                        evaluator.during_training(self, None, "final")
            self.finish()
        except BaseException:
            self.abort_evaluations()
            raise

    def build_data(self) -> None:
        params = self.cfg.dataloader.params
        rank, world = self.ctx.rank, self.ctx.world_size
        train_ds = build_split(self.cfg, self.agent, "train")
        self.train_loader = make_loader(
            train_ds, params, True, rank, world, sequence_split=int(OmegaConf.select(self.cfg, "dataloader.sequence_split") or 0),
            seed=int(self.args.seed or 0), draw_augment=geometry(OmegaConf.select(self.cfg, "augment")),
            exact_resume=self.args.exact_resume)
        if not self.args.skip_validation:
            self.val_loader = make_loader(build_split(self.cfg, self.agent, "val"), evaluation_params(self.cfg),
                                          False, rank, world)
        if OmegaConf.select(self.cfg, "vqa.enabled", default=False):
            from unidrivevla.data.tasks import TaskLoader
            from unidrivevla.data.vqa import build_vqa_loader

            vqa_loader = build_vqa_loader(self.cfg, self.agent, rank, world)
            mode = str(OmegaConf.select(self.cfg, "vqa.mode") or "combined")
            self.train_loader = TaskLoader(self.train_loader, vqa_loader, mode, self.args.exact_resume,
                                           int(self.args.seed or 0), rank, world)
            logger.info("VQA mode=%s batch_size=%d batches=%d; planning batches define the epoch", mode,
                        vqa_loader.batch_size, len(vqa_loader))
        if self.args.global_batch:
            per_step = int(params.batch_size) * world
            if self.args.global_batch % per_step:
                raise ValueError(f"train.global_batch {self.args.global_batch} is not a multiple of batch_size x GPUs "
                                 f"= {per_step}; set train.global_batch=0 to use train.grad_accum")
            self.args.grad_accum = self.args.global_batch // per_step
        self.global_batch = int(params.batch_size) * world * self.args.grad_accum
        if self.args.exact_resume and self.args.strategy == "fsdp" and len(self.train_loader) % self.args.grad_accum:
            raise ValueError("train.exact_resume with train.strategy=fsdp requires the number of training batches per "
                             "epoch to be divisible by train.grad_accum; use DDP to checkpoint partial gradients")
        if self.ctx.is_main:
            logger.info(
                "data=%s train=%d val=%s global_batch=%d steps/epoch=%d order=%s",
                loader_name(self.cfg),
                len(train_ds),
                len(self.val_loader.dataset) if self.val_loader is not None else "skipped",
                self.global_batch,
                len(self.train_loader) // self.args.grad_accum,
                "sequences" if isinstance(self.train_loader.batch_sampler, SequenceBatchSampler) else "shuffled",
            )

    def build_optimizer(self) -> None:
        """The model's optimizer and schedule; a get_optimizers(total_steps, steps_per_epoch) gets the training
        length in optimizer steps. Optional micro_steps/grad_accum are passed to agents that schedule per batch.
        Scheduler intervals may be optimizer step, data micro-step, or epoch."""
        steps_per_epoch = max(1, len(self.train_loader) // self.args.grad_accum)
        # Accumulation carries across epoch boundaries; floor only once over the entire training run.
        total_steps = (len(self.train_loader) * self.args.max_epochs) // self.args.grad_accum
        micro_steps = len(self.train_loader) * self.args.max_epochs
        if total_steps < 1:
            raise ValueError("training has no complete gradient accumulation window; lower global_batch/grad_accum")
        if self.args.max_steps:
            total_steps = min(total_steps, self.args.max_steps)
            micro_steps = min(micro_steps, self.args.max_steps * self.args.grad_accum)
        get_optimizers = self.agent.get_optimizers
        parameters = inspect.signature(get_optimizers).parameters
        if "total_steps" in parameters:
            kwargs = {"total_steps": total_steps}
            if "steps_per_epoch" in parameters:
                kwargs["steps_per_epoch"] = steps_per_epoch
            if "micro_steps" in parameters:
                kwargs["micro_steps"] = micro_steps
            if "grad_accum" in parameters:
                kwargs["grad_accum"] = self.args.grad_accum
            optim_cfg = get_optimizers(**kwargs)
        else:
            optim_cfg = get_optimizers()
        scheduler = None
        if isinstance(optim_cfg, dict):
            self.optimizer = optim_cfg["optimizer"]
            scheduler = optim_cfg.get("lr_scheduler")
        else:
            self.optimizer = optim_cfg
        self.lr_interval = "epoch"
        if isinstance(scheduler, dict):
            self.lr_interval = scheduler.get("interval", "epoch")
            scheduler = scheduler["scheduler"]
            if self.lr_interval not in ("step", "micro", "epoch"):
                raise ValueError(f"lr scheduler interval must be step, micro, or epoch, got {self.lr_interval!r}")
        self.lr_scheduler = scheduler
        self.clip_params = [p for p in self.agent.parameters() if p.requires_grad]
        low = [p for p in self.clip_params if p.dtype != torch.float32]
        if low and not getattr(self.optimizer, "fp32_master", False):
            raise ValueError(
                f"{len(low)} trainable parameters are {low[0].dtype} and {type(self.optimizer).__name__} would update "
                "them in that precision; use unidrivevla.train.optim.MasterAdamW, train.strategy=fsdp or "
                "train.precision=fp32")

    def resume(self) -> None:
        path = self.checkpointer.latest()
        if path is None:
            if self.ctx.is_main:
                logger.info("resume: no checkpoint under %s, starting fresh", self.checkpointer.root)
            return
        extra = self.checkpointer.load(path, self.model, self.optimizer, self.ema,
                                       require_rank_state=self.args.exact_resume)
        if self.args.exact_resume:
            if not extra.get("exact_resume"):
                raise RuntimeError(f"checkpoint {path} was not saved with train.exact_resume=true")
            if extra.get("training_signature") != self.training_signature():
                raise RuntimeError("exact resume requires the same resolved training config, except resume and "
                                   "checkpoint/logging cadence settings")
            rank_state = extra.pop("_rank_state")
            if rank_state.get("rank") != self.ctx.rank or rank_state.get("world_size") != self.ctx.world_size:
                raise RuntimeError("exact resume requires the same rank layout and world size")
            if rank_state.get("device_type") != self.ctx.device.type:
                raise RuntimeError("exact resume requires the same device type")
            self.resume_batch = int(extra.get("batch_in_epoch", 0))
            self.batch_in_epoch = self.resume_batch
            self._resume_epoch_complete = bool(extra.get("epoch_complete", False))
            self.micro = int(rank_state["micro"])
            if self.micro != int(extra.get("micro", self.micro)):
                raise RuntimeError("checkpoint micro-step state is inconsistent")
            self.restore_gradients(rank_state["gradients"])
            runtime_state = _device_tree(rank_state["model_runtime"], self.ctx.device)
            self.agent.load_training_runtime_state_dict(runtime_state, device=self.ctx.device)
            self._resume_rank_state = rank_state
        self.start_epoch = int(extra["epoch"])
        self.global_step = int(extra["global_step"])
        self.best_val = float(extra["best_val"])
        if self.lr_scheduler is not None and extra.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        if self.ctx.is_main:
            logger.info("resumed from %s at epoch %d batch %d step %d%s", path, self.start_epoch,
                        self.resume_batch, self.global_step,
                        " (exact)" if self.args.exact_resume else " (best effort; epoch position only)")

    def training_signature(self) -> str:
        values = OmegaConf.to_container(self.cfg, resolve=True)
        values = dict(values) if isinstance(values, dict) else values
        train = values.get("train", {}) if isinstance(values, dict) else {}
        if isinstance(train, dict):
            for key in ("resume", "save_every_n_steps", "log_every", "async_save"):
                train.pop(key, None)
            train["exact_resume"] = True
        if isinstance(values, dict):
            values.pop("resume", None)
            # Output/provenance identities may change when continuing the same training state.
            values.pop("experiment_name", None)
            values.pop("output_dir", None)
        payload = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def rng_state(self) -> dict:
        state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                 "cuda": None, "device_type": self.ctx.device.type}
        if self.ctx.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.ctx.device)
        return state

    def restore_rng(self, state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None:
            torch.cuda.set_rng_state(state["cuda"], self.ctx.device)

    def restore_gradients(self, gradients: dict) -> None:
        params = dict(self.agent.named_parameters())
        unknown = set(gradients) - params.keys()
        if unknown:
            raise RuntimeError(f"exact resume has gradients for unknown parameters: {sorted(unknown)[:8]}")
        for parameter in params.values():
            parameter.grad = None
        for name, gradient in gradients.items():
            parameter = params[name]
            parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)

    def step_micro_scheduler(self) -> None:
        # Main updates LR on every micro-batch; PyTorch's warning assumes optimizer.step per batch.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`",
                                    category=UserWarning)
            self.lr_scheduler.step()

    def loader_state(self) -> dict:
        loader = self.train_loader
        sampler = getattr(loader, "batch_sampler", None)
        if not callable(getattr(sampler, "set_epoch", None)):
            sampler = getattr(loader, "sampler", None)
        state = {"loader": type(loader).__qualname__, "length": len(loader),
                 "sampler": type(sampler).__qualname__ if sampler is not None else None,
                 "sampler_epoch": getattr(sampler, "epoch", None)}
        task_state = getattr(loader, "state_dict", None)
        if callable(task_state):
            state["tasks"] = task_state()
        return state

    def train_loop(self) -> None:
        args, model, device = self.args, self.model, self.ctx.device
        batches = DevicePrefetcher(self.train_loader, device)
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if not callable(getattr(sampler, "set_epoch", None)):
            sampler = getattr(self.train_loader, "sampler", None)
        for epoch in range(self.start_epoch, args.max_epochs):
            self.epoch = epoch
            model.train()
            if callable(getattr(sampler, "set_epoch", None)):
                sampler.set_epoch(epoch)
            restoring = self._resume_rank_state if epoch == self.start_epoch else None
            if restoring is not None:
                task_state = restoring.get("loader_state", {}).get("tasks")
                if task_state is not None:
                    restore = getattr(self.train_loader, "restore_state", None)
                    if not callable(restore):
                        raise RuntimeError("exact resume requires the task loader to restore its saved cursor")
                    restore(task_state, epoch_complete=self._resume_epoch_complete)
            iterator = iter(batches)
            resume_at = self.resume_batch if self._resume_rank_state is not None and epoch == self.start_epoch else 0
            for _ in range(resume_at):
                try:
                    next(iterator)
                except StopIteration as exc:
                    raise RuntimeError(f"checkpoint batch position {resume_at} exceeds the current epoch length") from exc
            if self._resume_rank_state is not None and epoch == self.start_epoch:
                expected = self._resume_rank_state.get("loader_state")
                if resume_at and expected is not None and self.loader_state() != expected:
                    raise RuntimeError("exact resume could not reproduce the saved sampler/task position")
                self.restore_rng(self._resume_rank_state["rng"])
                self._resume_rank_state = None
            loss_sum = torch.zeros((), device=device)
            loss_count = 0
            window = {"time": time.perf_counter(), "wait": batches.wait_s, "step": self.global_step}
            stop = False
            while True:
                try:
                    features, targets, tokens = next(iterator)
                except StopIteration:
                    break
                self.batch_in_epoch = resume_at + loss_count + 1
                self.micro += 1
                sync = self.micro % args.grad_accum == 0
                with gradient_sync(model, sync):
                    out = model(features, targets, tokens)
                    loss = out.loss
                    (loss / args.grad_accum).backward()
                loss_sum += loss.detach().float()
                loss_count += 1
                if not sync:
                    if self.lr_scheduler is not None and self.lr_interval == "micro":
                        self.step_micro_scheduler()
                    continue
                grad_norm = None
                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.clip_params, args.grad_clip, foreach=True)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None and self.lr_interval == "micro":
                    self.step_micro_scheduler()
                elif self.lr_scheduler is not None and self.lr_interval == "step":
                    self.lr_scheduler.step()
                if self.ema is not None:
                    self.ema.update()
                self.global_step += 1
                if self.global_step % args.log_every == 0:
                    self.log_train(epoch, out, loss, grad_norm, window, batches)
                if args.save_every_n_steps and self.global_step % args.save_every_n_steps == 0:
                    self.save_resume(epoch, self.batch_in_epoch)
                if args.max_steps and self.global_step >= args.max_steps:
                    stop = True
                    break
            train_loss = self.ctx.all_reduce_mean(loss_sum / max(loss_count, 1)).item()
            if stop and self.batch_in_epoch < len(self.train_loader) and args.exact_resume:
                self.export(epoch, train_loss, None, False)
                if not args.save_every_n_steps or self.global_step % args.save_every_n_steps:
                    self.save_resume(epoch, self.batch_in_epoch)
                break
            if self.lr_scheduler is not None and self.lr_interval == "epoch":
                self.lr_scheduler.step()
            self.end_epoch(epoch, train_loss)
            self.batch_in_epoch = 0
            self.resume_batch = 0
            if stop:
                break

    def log_train(self, epoch: int, out, loss, grad_norm, window: dict, batches: DevicePrefetcher) -> None:
        names, values = ["loss"], [loss.detach().float()]
        # sorted, so every rank stacks the same keys in the same order for the all-reduce
        for key in sorted(vars(out)):
            value = getattr(out, key)
            if key.startswith(METRIC_PREFIX) and torch.is_tensor(value) and value.numel() == 1:
                names.append(key)
                values.append(value.detach().float().reshape(()))
        if grad_norm is not None:
            names.append("grad_norm")
            values.append((grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm).float())
        stats = self.ctx.all_reduce_mean(torch.stack(values)).tolist()
        now = time.perf_counter()
        steps = max(self.global_step - window["step"], 1)
        elapsed = now - window["time"]
        waited = batches.wait_s - window["wait"]
        window.update(time=now, wait=batches.wait_s, step=self.global_step)
        if not self.ctx.is_main:
            return
        metrics = {f"train/{name}": value for name, value in zip(names, stats)}
        metrics.update({
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "perf/step_ms": 1000.0 * elapsed / steps,
            "perf/data_wait_ms": 1000.0 * waited / steps,
            "perf/samples_per_s": self.global_batch * steps / max(elapsed, 1e-9),
        })
        if torch.cuda.is_available():
            metrics["perf/max_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        self.monitor.log(metrics, self.global_step, epoch + 1)
        logger.info("[epoch %d step %d] %s", epoch + 1, self.global_step,
                    " ".join(f"{key.split('/', 1)[1]}={value:.4g}" for key, value in metrics.items()))

    def averaged(self):
        """Context in which the model holds its EMA weights (a no-op without EMA)."""
        return self.ema.swapped() if self.ema is not None else contextlib.nullcontext()

    def ckpt_suffixes(self):
        return ["", EMA_SUFFIX] if self.ema is not None else [""]

    @torch.no_grad()
    def validate(self) -> float:
        from unidrivevla.distributed.parallelize import check_eval_loader

        check_eval_loader(self.model, self.val_loader, self.ctx)
        device = self.ctx.device
        with isolated_evaluation_state(self.model, self.agent):
            total = torch.zeros((), device=device)
            count = 0
            with self.averaged():
                for features, targets, tokens in DevicePrefetcher(self.val_loader, device):
                    preds = self.model(features, targets, tokens)
                    total += self.agent.compute_loss(features, targets, preds).detach().float()
                    count += 1
            sums = self.ctx.all_reduce_sum(torch.stack([total, torch.tensor(float(count), device=device)]))
            return (sums[0] / sums[1].clamp(min=1.0)).item()

    def end_epoch(self, epoch: int, train_loss: float) -> None:
        val_loss = None
        if self.val_loader is not None and (epoch + 1) % self.args.val_every_n_epochs == 0:
            val_loss = self.validate()
        improved = val_loss is not None and val_loss < self.best_val
        if improved:
            self.best_val = val_loss
        epoch_metrics = {"epoch/train_loss": train_loss}
        if val_loss is not None:
            epoch_metrics["val/loss"] = val_loss
        self.monitor.log(epoch_metrics, self.global_step, epoch + 1)
        self.monitor.plot()
        tagged = self.export(epoch, train_loss, val_loss, improved)
        self.evaluate(epoch, tagged)
        self.save_resume(epoch, epoch_complete=True)

    def evaluate(self, epoch: int, ckpt: Path) -> None:
        """Periodic evaluation of the epoch's checkpoint; every rank takes part, with the model holding its weights."""
        due = [e for e in self.evaluators if e.every_n_epochs and (epoch + 1) % e.every_n_epochs == 0]
        if due:
            with self.averaged():
                for evaluator in due:
                    evaluator.during_training(self, ckpt, f"epoch_{epoch + 1:04d}")
        for evaluator in self.evaluators:
            self.report(evaluator, evaluator.poll() if self.ctx.is_main else [])

    def report(self, evaluator, results, synchronize: bool = True) -> None:
        failure = None
        if self.ctx.is_main:
            errors = [f"{evaluator.benchmark}/{tag}: {result['error']}" for tag, result in results
                      if isinstance(result, dict) and result.get("error")]
            if errors:
                failure = "; ".join(errors)
        if synchronize and self.ctx.world_size > 1 and dist.is_initialized():
            status = [failure]
            dist.broadcast_object_list(status, src=0, group=self.ctx.gloo_group())
            failure = status[0]
        if failure:
            self.abort_evaluations()
            raise RuntimeError(f"evaluation failed: {failure}")
        if self.ctx.is_main:
            for tag, result in results:
                logger.info("eval %s %s: %s", evaluator.benchmark, tag, result)
                self.monitor.log({f"eval/{evaluator.benchmark}/{key}": value for key, value in result.items()},
                                 self.global_step, self.epoch + 1)

    def abort_evaluations(self) -> None:
        """Stop evaluator-owned child jobs after a fatal evaluation or training error."""
        for evaluator in getattr(self, "evaluators", ()):
            cancel = getattr(evaluator, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    logger.exception("could not stop owned %s evaluation jobs", getattr(evaluator, "benchmark", ""))

    def export(self, epoch: int, train_loss: float, val_loss: Optional[float], improved: bool) -> Path:
        """Writes last.ckpt (and last-EMA.ckpt), then the top-k and best copies. Returns, on every rank, the
        checkpoint evaluators score for this epoch: the tagged top-k copy, or last without validation (EMA twin
        with EMA)."""
        suffixes = self.ckpt_suffixes()
        for suffix in suffixes:
            with self.averaged() if suffix else contextlib.nullcontext():
                state = export_model_state(self.model)
            if self.ctx.is_main:
                self.write_ckpt(state, self.output_dir / f"last{suffix}.ckpt")
        tag = None if val_loss is None else f"epoch_{epoch + 1:04d}_val_{val_loss:.6f}"
        tagged = self.output_dir / f"{tag or 'last'}{suffixes[-1]}.ckpt"
        if not self.ctx.is_main:
            return tagged
        logger.info(
            "epoch %d/%d train/loss=%.4f val/loss=%s -> %s",
            epoch + 1, self.args.max_epochs, train_loss,
            "skipped" if val_loss is None else f"{val_loss:.4f}", self.output_dir / "last.ckpt",
        )
        if tag is None:
            return tagged
        for suffix in suffixes:
            last = self.output_dir / f"last{suffix}.ckpt"
            self.copy_ckpt(last, self.output_dir / f"{tag}{suffix}.ckpt")
            if improved:
                self.copy_ckpt(last, self.output_dir / f"best{suffix}.ckpt")
        self.prune_topk(val_loss, f"{tag}.ckpt")
        if improved:
            logger.info("new best val/loss=%.6f", val_loss)
        return tagged

    def write_ckpt(self, state: dict, path: Path) -> None:
        """Complete model state, including stage-1 weights frozen in stage 2."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {"state_dict": state}
        if getattr(self, "cfg", None) is not None and OmegaConf.select(self.cfg, "agent") is not None:
            payload["agent_config"] = OmegaConf.to_container(self.cfg.agent, resolve=True)
        agent = getattr(self, "agent", None)
        args = getattr(self, "args", None)
        if isinstance(agent, torch.nn.Module) and args is not None:
            payload["compute_parameter_dtypes"] = _compute_parameter_dtypes(
                agent, args, getattr(self, "_fsdp_replicated_parameter_names", None))
        torch.save(payload, temporary)
        temporary.replace(path)

    def copy_ckpt(self, src: Path, dst: Path) -> None:
        shutil.copyfile(src, dst)

    def prune_topk(self, val_loss: float, name: str) -> None:
        meta_path = self.output_dir / "best_checkpoints.json"
        best = []
        if meta_path.is_file():
            try:
                best = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                best = []
        best.append({"path": name, "val_loss": float(val_loss)})
        best = sorted(best, key=lambda item: item["val_loss"])[: self.args.top_k]
        keep = {item["path"] for item in best}
        keep |= {Path(name).stem + EMA_SUFFIX + ".ckpt" for name in keep}
        for path in self.output_dir.glob("epoch_*.ckpt"):
            if path.name not in keep:
                path.unlink(missing_ok=True)
        meta_path.write_text(json.dumps(best, indent=2))

    def save_resume(self, epoch: int, batch_in_epoch: Optional[int] = None, epoch_complete: bool = False) -> None:
        batch_in_epoch = self.batch_in_epoch if batch_in_epoch is None else int(batch_in_epoch)
        next_epoch = epoch + 1 if epoch_complete else epoch
        extra = {
            "epoch": next_epoch,
            "batch_in_epoch": 0 if epoch_complete else batch_in_epoch,
            "epoch_complete": bool(epoch_complete),
            "global_step": self.global_step,
            "micro": self.micro,
            "best_val": self.best_val,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "exact_resume": self.args.exact_resume,
            "training_signature": self.training_signature() if self.args.exact_resume else None,
        }
        rank_state = None
        if self.args.exact_resume:
            save_runtime = getattr(self.agent, "training_runtime_state_dict", None)
            if not callable(save_runtime):
                raise RuntimeError("exact resume requires the agent to implement training_runtime_state_dict")
            gradients = {}
            for name, parameter in self.agent.named_parameters():
                if parameter.grad is None:
                    continue
                if hasattr(parameter.grad, "to_local"):
                    raise RuntimeError("exact resume cannot save a partial sharded gradient; align grad_accum with "
                                       "epoch batches or use DDP")
                gradients[name] = _cpu_tree(parameter.grad)
            rank_state = {
                "rank": self.ctx.rank,
                "world_size": self.ctx.world_size,
                "device_type": self.ctx.device.type,
                "micro": self.micro,
                "rng": self.rng_state(),
                "gradients": gradients,
                "model_runtime": _cpu_tree(save_runtime()),
                "loader_state": _cpu_tree(self.loader_state()),
            }
        self.checkpointer.save(self.global_step, self.model, self.optimizer, extra, self.ema,
                               rank_state=rank_state, position=(next_epoch, extra["batch_in_epoch"]))

    def finish(self) -> None:
        suffix = self.ckpt_suffixes()[-1]
        best = self.output_dir / f"best{suffix}.ckpt"
        final = best if best.is_file() else self.output_dir / f"last{suffix}.ckpt"
        # Release the GPUs before the final evaluation runs in its own processes. torch internals behind the
        # compiled blocks keep references to the weights, so their storage is dropped explicitly; FSDP2 shards
        # (DTensor) go with their modules.
        for tensor in [*self.agent.parameters(), *self.agent.buffers()]:
            if not hasattr(tensor, "to_local"):
                tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        self.model = self.optimizer = self.lr_scheduler = self.ema = None
        self.agent = None
        self.clip_params = []
        self.train_loader = self.val_loader = None
        for evaluator in self.evaluators:
            evaluator.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.ctx.is_main:
                logger.info("released GPU memory: %.2f GiB still allocated, %.2f GiB reserved",
                            torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30)
        self.ctx.close()
        try:
            if self.ctx.is_main:
                for evaluator in self.evaluators:
                    self.report(evaluator, evaluator.finish(final), synchronize=False)
        except BaseException:
            self.abort_evaluations()
            raise
        finally:
            self.monitor.close()
