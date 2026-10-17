"""One Qwen backbone with profile-local planning heads for mixed-domain training."""

import contextlib
import functools
import logging
import math
from dataclasses import fields
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from unidrivevla.adapters.unidrivevla import (POSE_INTERVAL_S, Profile, _checkpoint_motion_value,
                                               _check_checkpoint_frames, _motion_only_key, head_kwargs,
                                               register_modules)

logger = logging.getLogger(__name__)

MULTIDOMAIN_FORMAT_KEY = "_multidomain_format_version"
MULTIDOMAIN_MOTION_KEY = "_multidomain_with_motion"
MULTIDOMAIN_FORMAT_VERSION = 2


def _as_profiles(values) -> Dict[str, Profile]:
    if isinstance(values, DictConfig):
        values = OmegaConf.to_container(values, resolve=True)
    if not isinstance(values, dict) or len(values) < 2:
        raise ValueError("multidomain policy needs at least two named profiles")
    profiles = {str(name): value if isinstance(value, Profile) else Profile.from_config(value)
                for name, value in values.items()}
    for name, profile in profiles.items():
        if profile.name != name:
            raise ValueError(f"profile key {name!r} does not match profile.name={profile.name!r}")
    return dict(sorted(profiles.items()))


class ProfileRuntimeView:
    """A non-module view: processor and execution flags are local, weights stay on the owning policy."""

    def __init__(self, core, processor, profile: Profile, cast_stream_dtype: bool, attn_implementation: str):
        self.core, self.processor = core, processor
        self.profile = profile.name
        self.cast_stream_dtype = bool(cast_stream_dtype)
        self._vla_attn_impl = str(attn_implementation)
        self._inference_attn_impl = profile.inference_attn_impl
        self.shared_view_tokens = True

    def __getattr__(self, name):
        return getattr(self.core, name)

    def pixel_patches(self, images_tensor):
        return self.core.pixel_patches(images_tensor, processor=self.processor)

    def embed_image(self, image_paths, chunk_size=6):
        return self.core.embed_image(image_paths, chunk_size=chunk_size, processor=self.processor)

    def embed_image_tensor(self, images_tensor, chunk_size=6):
        return self.core.embed_image_tensor(images_tensor, chunk_size=chunk_size, processor=self.processor)

    def forward(self, *args, **kwargs):
        kwargs.setdefault("cast_stream_dtype", self.cast_stream_dtype)
        kwargs.setdefault("attn_implementation", self._vla_attn_impl)
        return self.core.forward(*args, **kwargs)

    @contextlib.contextmanager
    def inference_config(self):
        """Apply a profile's attention config only for its inference call, then restore the shared config."""
        configs = [self.core.qwen3_vl.config, self.core.qwen3_vl.language_model.config,
                   self.core.qwen3_vl.visual.config, self.core.qwen3_perception_expert.config,
                   self.core.qwen3_action_expert.config]
        saved = [(cfg, getattr(cfg, "_attn_implementation", None)) for cfg in configs]
        try:
            for cfg in configs:
                cfg._attn_implementation = self._inference_attn_impl
            yield
        finally:
            for cfg, value in saved:
                cfg._attn_implementation = value


def _build_shared_core(vlm_path, view_tokens, dtype, train_vlm, lora, processor_use_fast, vlm_variant):
    from unidrivevla.models.unidrivevla.dense_heads.qwenvl3_vla_planning_head import get_qwen_config
    from unidrivevla.models.unidrivevla.dense_heads.unidrivevla_vlm_qwenvl3 import Qwen3VLWithExpertModel

    if vlm_variant == "8b":
        qwen_vlm = get_qwen_config("qwen3_vl_8b_36l")
        expert = get_qwen_config("qwen3_8b_expert_36l")
    else:
        qwen_vlm = get_qwen_config("qwen3_vl_28l")
        expert = get_qwen_config("qwen3_28l")
    return Qwen3VLWithExpertModel(qwen_vlm, expert, expert, vlm_path, precision=dtype, train_vlm=train_vlm,
                                  lora_cfg=lora, view_tokens=view_tokens,
                                  processor_use_fast=processor_use_fast)


def _build_processor(vlm_path, use_fast):
    from transformers import AutoProcessor

    kwargs = {} if use_fast is None else {"use_fast": use_fast}
    return AutoProcessor.from_pretrained(vlm_path, **kwargs)


def _token_ids(processor, tokens):
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    missing = [token for token in tokens if token not in vocab]
    if missing:
        tokenizer.add_tokens(missing)
    return {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}


class ProfilePolicyView:
    """Source-specific evaluation surface over a MultiProfilePolicy."""

    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.profile = owner.profiles[name]
        self.trajectory_sampling = SimpleNamespace(num_poses=self.profile.horizon, interval_length=POSE_INTERVAL_S,
                                                   time_horizon=self.profile.horizon * POSE_INTERVAL_S)
        self.head = owner.heads[name]

    def forward(self, features, targets=None, tokens=None):
        features = dict(features)
        features.setdefault("profile", [self.name])
        return self.owner.forward(features, targets, tokens)

    __call__ = forward

    def worker_transform(self, image_augment=None, train=False):
        return self.owner.worker_transform(image_augment=image_augment, train=train, only=self.name)

    def compute_loss(self, features, targets, predictions):
        return self.owner.compute_loss(features, targets, predictions)

    def initialize(self, strict: bool = False):
        return self.owner.initialize(strict=strict)

    def to(self, *args, **kwargs):
        self.owner.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        self.owner.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        return self.owner.parameters(recurse=recurse)

    def training_runtime_state_dict(self):
        from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy

        return UniDriveVLAPolicy.training_runtime_state_dict(SimpleNamespace(head=self.head))

    def load_training_runtime_state_dict(self, state, device=None):
        from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy

        return UniDriveVLAPolicy.load_training_runtime_state_dict(SimpleNamespace(head=self.head), state,
                                                                 device=device)


def _embedding_migration(source, target, base_vocab_size, source_token_ids, target_token_ids, name):
    """Copy common vocabulary and matching view-token rows, retaining initialized rows for new tokens."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise RuntimeError(f"cannot migrate incompatible embedding tensor {name}: {tuple(source.shape)} -> {tuple(target.shape)}")
    if source.shape[0] < base_vocab_size or target.shape[0] < base_vocab_size:
        raise RuntimeError(f"embedding tensor {name} is shorter than base vocabulary {base_vocab_size}")
    migrated = target.clone()
    migrated[:base_vocab_size].copy_(source[:base_vocab_size])
    for token, source_id in source_token_ids.items():
        target_id = target_token_ids.get(token)
        if target_id is None:
            continue
        if source_id >= source.shape[0] or target_id >= target.shape[0]:
            raise RuntimeError(f"view token {token!r} is outside embedding tensor {name}")
        migrated[target_id].copy_(source[source_id])
    return migrated


class MultiProfilePolicy(nn.Module):
    """Shared Qwen VLM/experts plus one official-semantics head and processor per data profile."""

    def __init__(self, profiles, vlm_path: str, vae_path: Optional[str] = None, lr: float = 2e-4,
                 weight_decay: float = 1e-7, warmup_steps: int = 125, warmup_start: float = 1 / 3,
                 min_lr_ratio: float = 1e-3, vlm_lr_mult: float = 0.5, train_vlm: bool = False,
                 lora: Optional[Dict] = None, with_motion: bool = False, checkpoint_path: Optional[str] = None,
                 initial_profile: Optional[str] = None, checkpoint_with_motion: Optional[bool] = None,
                 exact_resume: bool = False,
                 attn_implementation: str = "sdpa", gradient_checkpointing: bool = True,
                 cache_frozen_prefix: bool = False, dtype: str = "bfloat16", vlm_variant: str = "2b",
                 checkpoint_frame=None,
                 _shared_core_factory=None, _processor_factory=None, _head_factory=None):
        super().__init__()
        self.profiles = _as_profiles(profiles)
        if initial_profile is not None and initial_profile not in self.profiles:
            raise ValueError(f"initial_profile {initial_profile!r} must be one of {sorted(self.profiles)}")
        if checkpoint_with_motion is not None and not isinstance(checkpoint_with_motion, bool):
            raise TypeError("checkpoint_with_motion must be true, false, or null")
        self.initial_profile = initial_profile
        self.checkpoint_path = checkpoint_path
        self.checkpoint_with_motion = checkpoint_with_motion
        if (checkpoint_frame is not None and checkpoint_frame not in ("ego", "lidar")
                and (not isinstance(checkpoint_frame, dict)
                     or any(value not in ("ego", "lidar") for value in checkpoint_frame.values()))):
            raise ValueError("checkpoint_frame must be null, 'ego', 'lidar', or a profile-to-frame mapping")
        self.checkpoint_frame = checkpoint_frame
        self.with_motion = bool(with_motion)
        self.lr, self.weight_decay, self.vlm_lr_mult = lr, weight_decay, vlm_lr_mult
        self.warmup_steps, self.warmup_start, self.min_lr_ratio = warmup_steps, warmup_start, min_lr_ratio
        self.train_vlm = bool(train_vlm)
        self.trajectory_sampling = SimpleNamespace(num_poses=self.profiles[next(iter(self.profiles))].horizon,
                                                   interval_length=POSE_INTERVAL_S)
        self.heads = nn.ModuleDict()
        self._profile_runtime = {}
        self._target_view_token_ids = {}

        # Every rank builds this same sorted union before optimizer/DDP sees the embedding parameter.
        view_tokens = sorted({token for profile in self.profiles.values() for token in profile.view_tokens})
        core_factory = _shared_core_factory or _build_shared_core
        shared_core = core_factory(vlm_path=vlm_path, view_tokens=view_tokens, dtype=dtype, train_vlm=train_vlm,
                                   lora=lora, processor_use_fast=None, vlm_variant=vlm_variant)
        self.shared_core = shared_core
        processor_factory = _processor_factory or _build_processor
        head_factory = _head_factory
        if head_factory is None:
            head_cls = register_modules()
            head_factory = lambda **kwargs: head_cls(**kwargs)

        core_ids = _token_ids(shared_core.processor, view_tokens)
        self._base_vocab_size = int(getattr(shared_core, "view_token_base_vocab_size", len(shared_core.processor.tokenizer)))
        for profile in self.profiles.values():
            processor = processor_factory(vlm_path, use_fast=profile.processor_use_fast)
            token_ids = _token_ids(processor, view_tokens)
            if token_ids != core_ids:
                raise ValueError(f"profile {profile.name!r} tokenizer assigns different IDs to shared view tokens")
            runtime = ProfileRuntimeView(shared_core, processor, profile, profile.cast_stream_dtype,
                                         attn_implementation)
            if _head_factory is None:
                kwargs = head_kwargs(profile, vlm_path, vae_path, train_vlm, with_motion, attn_implementation,
                                     gradient_checkpointing, cache_frozen_prefix)
                kwargs.update(lora_cfg=lora, shared_runtime=runtime, vlm_variant=vlm_variant)
                head = head_factory(**kwargs)
            else:
                head = head_factory(profile, runtime)
            self.heads[profile.name] = head
            self._profile_runtime[profile.name] = runtime
            self._target_view_token_ids[profile.name] = token_ids

        if not self.heads:
            raise ValueError("multidomain policy has no profile heads")
        self.register_buffer(MULTIDOMAIN_FORMAT_KEY,
                             torch.tensor(MULTIDOMAIN_FORMAT_VERSION, dtype=torch.int32), persistent=True)
        self.register_buffer(MULTIDOMAIN_MOTION_KEY, torch.tensor(self.with_motion, dtype=torch.bool), persistent=True)

    @property
    def head(self):
        """Default head for VQA input preparation; domain planning always uses an explicit profile tag."""
        return self.heads[next(iter(self.profiles))]

    def profile_for_source(self, name: str) -> Profile:
        try:
            return self.profiles[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown source profile {name!r}; available: {sorted(self.profiles)}") from exc

    def policy_for_source(self, name: str) -> ProfilePolicyView:
        self.profile_for_source(name)
        return ProfilePolicyView(self, str(name))

    def training_runtime_state_dict(self):
        return {"profiles": {name: self.policy_for_source(name).training_runtime_state_dict()
                             for name in self.profiles}}

    def load_training_runtime_state_dict(self, state, device=None):
        if not isinstance(state, dict) or set(state) != {"profiles"}:
            raise ValueError("multidomain runtime state must contain exactly the 'profiles' mapping")
        profiles = state["profiles"]
        if not isinstance(profiles, dict) or set(profiles) != set(self.profiles):
            got = sorted(profiles) if isinstance(profiles, dict) else type(profiles).__name__
            raise ValueError(f"multidomain runtime state profiles {got} do not match {sorted(self.profiles)}")
        if device is not None:
            try:
                device = torch.device(device)
            except (TypeError, RuntimeError) as exc:
                raise ValueError(f"invalid runtime-state restore device {device!r}") from exc
        for name, profile_state in profiles.items():
            if not isinstance(profile_state, dict):
                raise TypeError(f"runtime state for profile {name!r} must be a mapping")
            self.policy_for_source(name).load_training_runtime_state_dict(profile_state, device=device)

    def trajectory_sampling_for_source(self, name: str):
        profile = self.profile_for_source(name)
        return SimpleNamespace(num_poses=profile.horizon, interval_length=POSE_INTERVAL_S,
                               time_horizon=profile.horizon * POSE_INTERVAL_S)

    def worker_transform(self, image_augment=None, train=False, only=None):
        transforms = {name: self._make_transform(name, image_augment, train)
                      for name in ([only] if only else self.profiles)}

        def run(features, targets):
            name = features.get("profile")
            if name is None:
                if only is None:
                    raise ValueError("multidomain worker transform requires canonical features['profile']")
                name = only
            try:
                transform = transforms[str(name)]
            except KeyError as exc:
                raise ValueError(f"unknown source profile {name!r}") from exc
            canonical = dict(features)
            canonical.pop("profile", None)
            transformed, targets = transform(canonical, targets)
            transformed = dict(transformed)
            transformed["profile"] = str(name)
            return transformed, targets

        return run

    def _make_transform(self, name, image_augment, train):
        from unidrivevla.adapters.unidrivevla import prepare_sample

        return functools.partial(prepare_sample, profile=self.profiles[name], image_augment=image_augment, train=train)

    @staticmethod
    def _source_name(features, required=True):
        value = features.get("profile")
        if value is None:
            if required:
                raise ValueError("multidomain policy requires features['profile'] from its source loader")
            return None
        if isinstance(value, (list, tuple)):
            names = {str(item) for item in value}
            if len(names) != 1:
                raise ValueError(f"a batch must contain one homogeneous profile, got {sorted(names)}")
            value = next(iter(names))
        name = str(value)
        if not name:
            raise ValueError("features['profile'] cannot be empty")
        return name

    def forward(self, features, targets=None, tokens=None):
        features = dict(features)
        task = features.get("task", "planning")
        name = self._source_name(features, required=(task != "vqa"))
        name = name or next(iter(self.profiles))
        if name not in self.heads:
            raise KeyError(f"unknown source profile {name!r}; available: {sorted(self.heads)}")
        head, runtime = self.heads[name], self._profile_runtime[name]
        features.pop("profile", None)
        if not self.training and task == "planning":
            with runtime.inference_config():
                return self._forward_single_profile(head, self.profiles[name], features, targets, tokens)
        return self._forward_single_profile(head, self.profiles[name], features, targets, tokens)

    def _forward_single_profile(self, head, profile, features, targets, tokens):
        # Reuse the single-profile adapter's task and loss path, including the head's combined AR loss.
        from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy

        policy = SimpleNamespace(head=head, profile=profile, training=self.training)
        return UniDriveVLAPolicy.forward(policy, features, targets, tokens)

    def compute_loss(self, features, targets, predictions):
        if "pred_traj" not in predictions:
            raise ValueError("validation predictions must contain pred_traj")
        gt = torch.cumsum(targets["gt_ego_fut_trajs"].float(), dim=1)
        mask = targets["gt_ego_fut_masks"].float()
        prediction = predictions.get("pred_traj_head", predictions["pred_traj"])
        l2 = torch.linalg.norm(prediction[..., :2] - gt, dim=-1)
        return (l2 * mask).sum() / mask.sum().clamp(min=1.0)

    def get_optimizers(self, total_steps: int = 0, steps_per_epoch: int = 0):
        from unidrivevla.train.optim import MasterAdamW

        vlm, rest, seen = [], [], set()
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                (vlm if name.startswith("shared_core.qwen3_vl.") else rest).append(param)
        groups = [{"params": rest}] + ([{"params": vlm, "lr": self.lr * self.vlm_lr_mult}] if vlm else [])
        optimizer = MasterAdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        if total_steps <= 0:
            return optimizer

        def factor(step: int) -> float:
            progress = min(1.0, step / total_steps)
            value = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if step < self.warmup_steps:
                value *= 1.0 - (1.0 - step / self.warmup_steps) * (1.0 - self.warmup_start)
            return value

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def initialize(self, strict: bool = False):
        if not self.checkpoint_path:
            if strict:
                raise ValueError("strict inference loading requires agent.checkpoint_path")
            return
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise TypeError(f"{self.checkpoint_path} does not contain a state dict")
        if any(str(key).startswith("module.") for key in state):
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
        if any(key.startswith("shared_core.") for key in state):
            _check_checkpoint_frames(checkpoint, self.profiles, require_all=True,
                                     checkpoint_frame=self.checkpoint_frame)
            self._load_multidomain_state(state, checkpoint, strict=strict)
            return
        if strict:
            raise ValueError("strict inference loading requires a complete multidomain checkpoint; single-profile "
                             "migration is training-only")
        if self.initial_profile is None:
            raise ValueError("loading a single-profile checkpoint requires explicit agent.initial_profile")
        _check_checkpoint_frames(checkpoint, {self.initial_profile: self.profiles[self.initial_profile]},
                                 checkpoint_frame=self.checkpoint_frame)

        own = self.state_dict()
        source_motion_keys = {name for name in state if _motion_only_key(name)}
        source_motion = _checkpoint_motion_value(checkpoint, state, source_motion_keys)
        if source_motion and not self.with_motion:
            raise ValueError("cannot initialize a motion-enabled checkpoint into a motion-disabled multidomain policy")
        transition = not source_motion and self.with_motion
        source_profile = self.profiles[self.initial_profile]
        source_ids = self._source_view_token_ids(source_profile)
        target_ids = self._target_view_token_ids[self.initial_profile]
        loaded = {}
        for name, value in state.items():
            if not torch.is_tensor(value) or name.startswith("_multidomain_format_version"):
                continue
            if name.startswith("head.qwen3_vl_with_expert."):
                target_name = "shared_core." + name.removeprefix("head.qwen3_vl_with_expert.")
            elif name.startswith("head."):
                target_name = f"heads.{self.initial_profile}." + name.removeprefix("head.")
            else:
                continue
            if target_name not in own:
                continue
            target = own[target_name]
            if value.shape != target.shape:
                if "embed_tokens.weight" in target_name or "lm_head.weight" in target_name:
                    value = _embedding_migration(value, target, self._base_vocab_size, source_ids, target_ids,
                                                 target_name)
                else:
                    raise RuntimeError(f"checkpoint tensor {name} has shape {tuple(value.shape)}, expected "
                                       f"{tuple(target.shape)} for {target_name}")
            loaded[target_name] = value
        core_keys = {name for name in own if name.startswith("shared_core.")}
        missing_core = sorted(core_keys - loaded.keys())
        if missing_core:
            raise KeyError(f"single-profile checkpoint omits shared-core tensors, e.g. {missing_core[:5]}")
        selected_head_prefix = f"heads.{self.initial_profile}."
        selected_head_keys = {name for name in own if name.startswith(selected_head_prefix)}
        selected_motion_keys = {name for name in selected_head_keys
                                if self._motion_only_key(name, (self.initial_profile,))}
        required_head = selected_head_keys - (selected_motion_keys if transition else set())
        missing_head = sorted(required_head - loaded.keys())
        if missing_head:
            raise KeyError(f"single-profile checkpoint omits required tensors for selected head "
                           f"{self.initial_profile!r}, e.g. {missing_head[:5]}")
        if transition and not selected_motion_keys:
            raise ValueError("stage-1 to stage-2 migration requested but the selected profile has no motion tensors")
        self.load_state_dict(loaded, strict=False)
        logger.info("initialized shared core and selected private head from %s (%d tensors); profile=%s%s",
                    self.checkpoint_path, len(loaded), self.initial_profile,
                    "; new motion tensors left fresh" if transition else "")

    @staticmethod
    def _motion_only_key(name: str, profiles) -> bool:
        return _motion_only_key(name)

    def _checkpoint_motion_value(self, checkpoint, state) -> Optional[bool]:
        motion_keys = {name for name in state if self._motion_only_key(name, self.profiles)}
        value = _checkpoint_motion_value(checkpoint, state, motion_keys)
        if self.checkpoint_with_motion is not None and self.checkpoint_with_motion != value:
            raise ValueError("checkpoint_with_motion disagrees with checkpoint metadata or motion tensors")
        return value

    def _load_multidomain_state(self, state, checkpoint, strict: bool = False) -> None:
        source_motion = self._checkpoint_motion_value(checkpoint, state)
        if source_motion is None:
            raise ValueError("multidomain checkpoint has no motion metadata; set agent.checkpoint_with_motion explicitly")
        if source_motion and not self.with_motion:
            raise ValueError("cannot initialize a motion-enabled checkpoint into a motion-disabled policy")
        transition = not source_motion and self.with_motion
        if transition and strict:
            raise ValueError("strict inference loading cannot add fresh motion modules; use a same-stage checkpoint")

        source_version = state.get(MULTIDOMAIN_FORMAT_KEY)
        if source_version is not None:
            if not torch.is_tensor(source_version) or source_version.numel() != 1:
                raise TypeError(f"checkpoint {MULTIDOMAIN_FORMAT_KEY} must be a scalar tensor")
            source_version = int(source_version.item())
            if not 1 <= source_version <= MULTIDOMAIN_FORMAT_VERSION:
                raise ValueError(f"unsupported multidomain checkpoint format {source_version}")

        own = self.state_dict()
        metadata = {MULTIDOMAIN_FORMAT_KEY, MULTIDOMAIN_MOTION_KEY}
        loaded = {}
        unexpected = []
        for name, value in state.items():
            if name in metadata:
                continue
            if name not in own:
                unexpected.append(name)
                continue
            if not torch.is_tensor(value):
                raise TypeError(f"multidomain checkpoint tensor {name!r} is not a tensor")
            if value.shape != own[name].shape:
                raise RuntimeError(f"checkpoint tensor {name} has shape {tuple(value.shape)}, expected "
                                   f"{tuple(own[name].shape)}")
            loaded[name] = value
        if unexpected:
            raise KeyError(f"multidomain checkpoint has unexpected tensors, e.g. {sorted(unexpected)[:5]}")

        own_motion = {name for name in own if self._motion_only_key(name, self.profiles)}
        source_motion_keys = {name for name in state if self._motion_only_key(name, self.profiles)}
        if not self.with_motion and own_motion:
            raise RuntimeError("motion-disabled policy unexpectedly constructed motion-only tensors")
        if not source_motion and source_motion_keys:
            raise ValueError("checkpoint says with_motion=false but contains motion-only tensors")

        # These describe the receiving architecture, so keep this model's values when restoring weights.
        loaded[MULTIDOMAIN_FORMAT_KEY] = own[MULTIDOMAIN_FORMAT_KEY]
        loaded[MULTIDOMAIN_MOTION_KEY] = own[MULTIDOMAIN_MOTION_KEY]
        missing = set(own) - set(loaded)
        if transition:
            if not own_motion or missing != own_motion:
                unrelated = sorted(missing - own_motion)
                raise KeyError("stage-1 multidomain checkpoint may leave only new motion tensors fresh; "
                               f"unrelated or partial missing tensors: {unrelated[:5]}")
            result = self.load_state_dict(loaded, strict=False)
            if set(result.missing_keys) != own_motion or result.unexpected_keys:
                raise RuntimeError("stage-1 to stage-2 multidomain load did not match the validated motion-only gap")
            logger.info("initialized multidomain stage 2 from stage 1; left %d motion tensors fresh", len(own_motion))
            return
        if missing:
            raise KeyError(f"multidomain checkpoint omits tensors, e.g. {sorted(missing)[:5]}")
        self.load_state_dict(loaded, strict=True)

    def _source_view_token_ids(self, profile):
        tokenizer = self.shared_core.processor.tokenizer
        vocab = tokenizer.get_vocab()
        next_id = self._base_vocab_size
        out = {}
        for token in profile.view_tokens:
            token_id = int(vocab[token])
            if token_id < self._base_vocab_size:
                out[token] = token_id
            else:
                out[token] = next_id
                next_id += 1
        return out
