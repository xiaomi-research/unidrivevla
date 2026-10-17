"""Adapters plug a model into the framework through one contract, so data sources and benchmarks never touch it.

A policy adapter is a torch.nn.Module with

    trajectory_sampling    num_poses and interval_length (0.5 s) of the predicted trajectory
    forward(features, targets=None, tokens=None)
                           training: an object with `.loss` (extra tensor attributes are logged);
                           eval: {"pred_traj": (batch, num_poses, 3)} ego-frame poses (x, y, heading)
    compute_loss(features, targets, predictions)   validation loss from the eval forward
    get_optimizers()       an optimizer, or {"optimizer": ..., "lr_scheduler": ...}; the scheduler advances once per epoch,
                           or every optimizer step when given as {"scheduler": ..., "interval": "step"}. A
                           get_optimizers(total_steps, steps_per_epoch) receives the training length in optimizer steps

and optionally

    worker_transform(image_augment=None, train=False)
                                           a function (features, targets) -> (features, targets) run in the dataloader
                                           workers: it turns the canonical sample of unidrivevla.data.sample into the
                                           model's tensors (train: the training split's preprocessing)
    initialize()                           loads weights before parallelization

Batches reach forward in their own dtype; under FSDP mixed precision the model casts its inputs itself. Data sources
emit the canonical sample and benchmarks read only `pred_traj`, so adding either leaves the model alone.
"""

import contextlib
import copy
import inspect
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

REQUIRED = ("trajectory_sampling", "forward", "compute_loss", "get_optimizers")
AGENT_TARGETS = {"unidrivevla.adapters.unidrivevla.UniDriveVLAPolicy",
                 "unidrivevla.adapters.multidomain.MultiProfilePolicy"}


def load_checkpoint_agent_config(checkpoint):
    """Return allowlisted, weights-only agent metadata; legacy checkpoints return None."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    metadata = saved.get("agent_config") if isinstance(saved, dict) else None
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or metadata.get("_target_") not in AGENT_TARGETS:
        raise ValueError("checkpoint agent_config must name a supported UniDriveVLA policy")
    return metadata


def initialize_for_inference(policy):
    initialize = getattr(policy, "initialize", None)
    if not callable(initialize):
        return
    try:
        supports_strict = "strict" in inspect.signature(initialize).parameters
    except (TypeError, ValueError):
        supports_strict = False
    initialize(strict=True) if supports_strict else initialize()
    module = _unwrap_policy(policy)
    checkpoint_path = getattr(module, "checkpoint_path", None)
    if isinstance(module, nn.Module) and checkpoint_path:
        _restore_checkpoint_compute_parameter_dtypes(module, checkpoint_path)


def _restore_checkpoint_compute_parameter_dtypes(module, checkpoint_path):
    """Restore saved forward dtypes after strict weight loading, without changing buffers or Parameter identity."""
    from unidrivevla.distributed.checkpoint import canonical

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or "compute_parameter_dtypes" not in checkpoint:
        return  # Legacy checkpoints keep the historical load_state_dict dtype behavior.
    saved = checkpoint["compute_parameter_dtypes"]
    if not isinstance(saved, dict) or any(not isinstance(name, str) for name in saved):
        raise TypeError("checkpoint compute_parameter_dtypes must be a string-keyed mapping")
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    invalid = {name: dtype for name, dtype in saved.items()
               if not isinstance(dtype, str) or dtype not in dtype_by_name}
    if invalid:
        raise ValueError(f"checkpoint has unsupported compute parameter dtypes: {invalid}")

    parameters = {}
    identities = {}
    for name, parameter in module.named_parameters():
        if not parameter.is_floating_point():
            continue
        name = canonical(name)
        if name in parameters:
            if identities[name] is not parameter:
                raise ValueError(f"canonical inference parameter name collision for {name!r}")
            continue
        parameters[name] = parameter
        identities[name] = parameter
    missing = set(parameters) - set(saved)
    unexpected = set(saved) - set(parameters)
    if missing or unexpected:
        raise ValueError("checkpoint compute_parameter_dtypes does not match the policy's floating parameters; "
                         f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}")

    casts = [(parameters[name], dtype_by_name[dtype]) for name, dtype in saved.items()]
    with torch.no_grad():
        for parameter, dtype in casts:
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)


def check_policy(model) -> None:
    missing = [name for name in REQUIRED if not hasattr(model, name)]
    if missing:
        raise TypeError(f"{type(model).__name__} is not a policy adapter, it lacks {missing} (see unidrivevla.adapters)")


def _unwrap_policy(module):
    """Unwrap only known parallel/compile wrappers; FSDP2 composes onto the original module."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        FullyShardedDataParallel = ()
    while True:
        if not isinstance(module, nn.Module):
            owner = getattr(module, "owner", None)
            if isinstance(owner, nn.Module):
                module = owner
                continue
            break
        if isinstance(module, DistributedDataParallel) or (
                FullyShardedDataParallel and isinstance(module, FullyShardedDataParallel)):
            module = module.module
            continue
        original = getattr(module, "_orig_mod", None)
        if isinstance(original, nn.Module):
            module = original
            continue
        break
    return module


def _clone_runtime_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_runtime_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_state(item) for item in value]
    if isinstance(value, tuple):
        items = [_clone_runtime_state(item) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    return copy.deepcopy(value)


def _runtime_state_hooks(module):
    save = getattr(module, "training_runtime_state_dict", None)
    load = getattr(module, "load_training_runtime_state_dict", None)
    return (save, load) if callable(save) and callable(load) else (None, None)


def _runtime_snapshots(policy):
    save, load = _runtime_state_hooks(policy)
    if save is not None:
        return [(load, _clone_runtime_state(save()))]
    candidates = list(policy.named_modules()) if isinstance(policy, nn.Module) else [("", policy)]
    snapshots = []
    covered = []
    for name, module in candidates:
        if any(name.startswith(parent + ".") for parent in covered):
            continue
        save, load = _runtime_state_hooks(module)
        if save is not None:
            snapshots.append((load, _clone_runtime_state(save())))
            if name:
                covered.append(name)
    return snapshots


def _evaluation_heads(policy):
    heads = getattr(policy, "heads", None)
    if isinstance(heads, (nn.ModuleDict, dict)):
        modules = [head for head in heads.values() if isinstance(head, nn.Module)]
        if modules:
            return modules
    head = getattr(policy, "head", None)
    if isinstance(head, nn.Module):
        return [head]
    return [policy]


def _reset_temporal_state(policy):
    seen = set()
    for head in _evaluation_heads(policy):
        reset_temporal = getattr(head, "reset_temporal", None)
        if callable(reset_temporal):
            reset_temporal()
        modules = head.modules() if isinstance(head, nn.Module) else (head,)
        for module in modules:
            if id(module) in seen:
                continue
            seen.add(id(module))
            reset = getattr(module, "reset", None)
            save, _ = _runtime_state_hooks(module)
            # A head-level reset owns cached-anchor modules; other runtime-backed modules
            # (for example a temporal target queue) expose their own reset hook.
            if callable(reset) and save is not None and not (
                    callable(reset_temporal) and module is not head and hasattr(module, "cached_anchor")):
                reset()


@contextlib.contextmanager
def isolated_evaluation_state(model, runtime_policy=None):
    """Run an extra eval pass from empty temporal caches, then restore live state and every module mode.

    Runtime tensor snapshots stay on their original device and temporarily duplicate cache memory.
    """
    policy = _unwrap_policy(runtime_policy if runtime_policy is not None else model)
    modes = []
    mode_modules = set()
    targets = [model, policy]
    for target in tuple(targets):
        for name in ("owner", "core"):
            wrapped = getattr(target, name, None)
            if isinstance(wrapped, nn.Module):
                targets.append(wrapped)
    for target in targets:
        if isinstance(target, nn.Module):
            for module in target.modules():
                if id(module) not in mode_modules:
                    mode_modules.add(id(module))
                    modes.append((module, module.training))
    snapshots = _runtime_snapshots(policy)
    try:
        if snapshots:
            _reset_temporal_state(policy)
        model.eval()
        yield
    finally:
        try:
            for load, state in snapshots:
                load(state)
        finally:
            for module, training in modes:
                module.training = training


@torch.no_grad()
def predict(model, loader, ctx) -> Optional[Dict[str, np.ndarray]]:
    """`pred_traj` of every sample of `loader` (each rank runs its shard), gathered into {token: poses} on rank 0;
    None on the other ranks."""
    from unidrivevla.data import DevicePrefetcher
    from unidrivevla.distributed.parallelize import check_eval_loader

    check_eval_loader(model, loader, ctx)
    mine: Dict[str, np.ndarray] = {}
    with isolated_evaluation_state(model):
        for features, _, tokens in DevicePrefetcher(loader, ctx.device):
            mine.update(zip(tokens, model(features)["pred_traj"].float().cpu().numpy()))
    gathered = [None] * ctx.world_size if ctx.is_main else None
    dist.gather_object(mine, gathered, dst=0, group=ctx.gloo_group())
    return {token: poses for part in gathered for token, poses in part.items()} if ctx.is_main else None
