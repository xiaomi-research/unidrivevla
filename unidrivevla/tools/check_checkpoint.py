"""python unidrivevla/tools/check_checkpoint.py: CPU round trips for stage-2 frozen weights and resume state."""

import socket
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.adapters import initialize_for_inference  # noqa: E402
from unidrivevla.adapters.unidrivevla import (  # noqa: E402
    Profile, UniDriveVLAPolicy, _validate_stage3_checkpoint_config,
)
from unidrivevla.distributed.checkpoint import DONE_FILE, ResumeCheckpointer, export_model_state, training_state  # noqa: E402
from unidrivevla.train.trainer import Trainer, _compute_parameter_dtypes  # noqa: E402
from unidrivevla.train.optim import MasterAdamW  # noqa: E402


class TinyPolicy(nn.Module):
    initialize = UniDriveVLAPolicy.initialize

    def __init__(self, checkpoint_path=None, stage2=False):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.vlm = nn.Linear(2, 2, bias=False)
        self.lora = nn.Linear(2, 2, bias=False)
        self.plan = nn.Linear(2, 1, bias=False)
        if stage2:
            self.motion = nn.Linear(2, 1)
            self.vlm.requires_grad_(False)
            self.lora.requires_grad_(False)


class MainNamespacePolicy(UniDriveVLAPolicy):
    """Small real-initialize fixture with the same planning_head -> head migration as main."""

    def __init__(self, profile, checkpoint_path=None, with_motion=False, train_vlm=True, checkpoint_frame=None,
                 checkpoint_transition=None, lr=2.0e-4):
        nn.Module.__init__(self)
        self.profile = profile
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frame = checkpoint_frame
        self.checkpoint_transition = checkpoint_transition
        self.lr = lr
        self.weight_decay, self.warmup_steps, self.warmup_start, self.min_lr_ratio = 1.0e-7, 125, 1.0 / 3, 1.0e-3
        self.vlm_lr_mult, self.vlm_variant = 0.5, "2b"
        self.train_vlm = bool(train_vlm)
        self.with_motion = with_motion
        self.coc_language_prompt = bool(train_vlm)
        self.coc_language_loss_weight = 0.1 if train_vlm else 0.0
        self.coc_jsonl_path = "/data/stage2-coc.jsonl" if train_vlm else ""
        self.coc_planning_loss_weight = 0.0
        self._checkpoint_stage3_options = {
            "weight_decay": self.weight_decay, "warmup_steps": self.warmup_steps,
            "warmup_start": self.warmup_start, "min_lr_ratio": self.min_lr_ratio,
            "vlm_lr_mult": self.vlm_lr_mult, "vlm_variant": self.vlm_variant,
            "gradient_checkpointing": True, "cache_frozen_prefix": True,
            "attn_implementation": "sdpa", "canbus_sft_prompt": False,
            "vlm_path": "/weights/qwen", "vae_path": "/weights/vae",
            "lora": {"r": 64, "lora_alpha": 128, "target_modules": ["q_proj", "v_proj"]},
            "visual_lora": {"r": 64, "lora_alpha": 128, "target_modules": "projector"},
            "visual_wasp": {"rank": 64, "alpha": 128, "top_k": 16, "beta": 0.01,
                            "target_modules": ["qkv", "proj"]},
            "llm_wasp": {"rank": 64, "alpha": 128, "top_k": 16, "beta": 0.003,
                         "target_modules": ["q_proj", "v_proj"]},
            "visual_projector_wasp": None,
        }
        self.head = nn.Module()
        self.head.vlm = nn.Linear(2, 2, bias=False)
        self.head.visual_wasp = nn.Linear(2, 2, bias=False)
        self.head.projector_lora = nn.Linear(2, 2, bias=False)
        self.head.llm_wasp = nn.Linear(2, 2, bias=False)
        self.head.plan = nn.Linear(2, 1, bias=False)
        if with_motion:
            self.head.motion = nn.Linear(2, 1, bias=False)
        if not train_vlm:
            self.head.vlm.requires_grad_(False)
            self.head.visual_wasp.requires_grad_(False)
            self.head.projector_lora.requires_grad_(False)
            self.head.llm_wasp.requires_grad_(False)


class ComputeDtypePolicy(nn.Module):
    initialize = UniDriveVLAPolicy.initialize

    def __init__(self, checkpoint_path=None, dtype=torch.float32, frozen_dtype=None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.primary = nn.Parameter(torch.tensor([1.25, -0.75], dtype=dtype))
        self.tied_alias = self.primary
        self.frozen = nn.Parameter(torch.tensor([0.5, 2.0], dtype=frozen_dtype or dtype), requires_grad=False)
        self.frozen_integer = nn.Parameter(torch.tensor(3, dtype=torch.int64), requires_grad=False)
        self.register_buffer("float_buffer", torch.tensor([3.0], dtype=torch.float32))
        self.register_buffer("integer_buffer", torch.tensor([7], dtype=torch.int64))

    def forward(self):
        values = torch.tensor([0.25, -1.5], dtype=self.primary.dtype)
        return (self.primary * values).sum() + (self.frozen * values).sum() + self.float_buffer[0]


class BranchPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2)
        self.heads = nn.ModuleDict({"nuscenes": nn.Linear(2, 2), "navsim": nn.Linear(2, 2)})

    def forward(self, profile, value):
        return (self.shared(value) + self.heads[profile](value)).square().sum()


def optimizer_states_by_name(model, optimizer):
    return {name: {key: value.detach().clone() if torch.is_tensor(value) else value
                   for key, value in optimizer.state[parameter].items()}
            for name, parameter in model.named_parameters() if optimizer.state.get(parameter)}


def tree_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def branch_step(model, optimizer, profile):
    optimizer.zero_grad(set_to_none=True)
    value = torch.tensor([[0.25, -0.75], [1.5, 0.5]], dtype=torch.float32)
    model(profile, value).backward()
    optimizer.step()


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"ok   {message}")


def check_main_checkpoint_migration(root):
    profile_path = Path(__file__).resolve().parents[1] / "config" / "profile" / "nuscenes.yaml"
    profile = Profile.from_config(OmegaConf.load(profile_path))
    torch.manual_seed(51)
    stage2 = MainNamespacePolicy(profile, with_motion=False, train_vlm=True)
    with torch.no_grad():
        for index, parameter in enumerate(stage2.parameters(), 1):
            parameter.fill_(float(index) / 10)
    main_state = {
        "module.planning_head." + name[len("head."):]: value.detach().clone()
        for name, value in stage2.state_dict().items()
    }
    main_stage2_path = root / "main-stage2-deepspeed.pt"
    torch.save({"module": main_state, "meta": {"config": "main checkpoint is raw nuScenes LiDAR"}},
               main_stage2_path)

    stage3 = MainNamespacePolicy(profile, str(main_stage2_path), with_motion=True, train_vlm=False,
                                 checkpoint_frame="lidar")
    with torch.no_grad():
        stage3.head.motion.weight.fill_(9.0)
    fresh_motion = stage3.head.motion.weight.detach().clone()
    stage3.initialize()
    for name, value in stage2.state_dict().items():
        target_name = "head." + name[len("head."):]
        check(torch.equal(stage3.state_dict()[target_name], value),
              f"DeepSpeed Stage2 module/planning_head mapping loads {target_name}")
    check(torch.equal(stage3.head.motion.weight, fresh_motion),
          "Stage2-to-Stage3 migration leaves only the new motion branch initialized")
    check(all(not parameter.requires_grad for name, parameter in stage3.named_parameters()
              if name.startswith("head.vlm.") or name.startswith("head.visual_wasp.")
              or name.startswith("head.projector_lora.") or name.startswith("head.llm_wasp.")),
          "Stage3 freezes the inherited VLM, RESPECT and projector LoRA parameters")

    strict_transition = MainNamespacePolicy(profile, str(main_stage2_path), with_motion=True, train_vlm=False,
                                            checkpoint_frame="lidar")
    try:
        strict_transition.initialize(strict=True)
    except RuntimeError as exc:
        check("strict loading" in str(exc), "strict evaluation rejects a Stage2-to-Stage3 partial migration")
    else:
        raise AssertionError("strict loading accepted an incomplete Stage2 checkpoint for Stage3 evaluation")

    full_stage3 = MainNamespacePolicy(profile, with_motion=True, train_vlm=False)
    with torch.no_grad():
        for index, parameter in enumerate(full_stage3.parameters(), 1):
            parameter.fill_(float(index) / 7)
    full_main_state = {
        "module.planning_head." + name[len("head."):]: value.detach().clone()
        for name, value in full_stage3.state_dict().items()
    }
    main_stage3_path = root / "main-stage3-deepspeed.pt"
    torch.save({"module": full_main_state, "meta": {"iter": 1}}, main_stage3_path)
    evaluated = MainNamespacePolicy(profile, str(main_stage3_path), with_motion=True, train_vlm=False,
                                    checkpoint_frame="lidar")
    evaluated.initialize(strict=True)
    check(all(torch.equal(value, evaluated.state_dict()["head." + name[len("head."):]])
              for name, value in full_stage3.state_dict().items()),
          "strict evaluation imports a complete DeepSpeed Stage3 module/planning_head checkpoint")

    no_frame = MainNamespacePolicy(profile, str(main_stage2_path), with_motion=True, train_vlm=False)
    try:
        no_frame.initialize()
    except ValueError as exc:
        check("coordinate-frame provenance" in str(exc), "legacy main checkpoint requires explicit source frame")
    else:
        raise AssertionError("legacy LiDAR checkpoint was accepted without frame provenance")
    wrong_frame = MainNamespacePolicy(profile, str(main_stage2_path), with_motion=True, train_vlm=False,
                                      checkpoint_frame="ego")
    try:
        wrong_frame.initialize()
    except ValueError as exc:
        check("does not match target frame" in str(exc), "main checkpoint source frame mismatch is rejected")
    else:
        raise AssertionError("wrong-source-frame checkpoint was accepted")

    incomplete_state = dict(main_state)
    incomplete_state.pop("module.planning_head.projector_lora.weight")
    incomplete_path = root / "main-stage2-missing-projector.pt"
    torch.save({"module": incomplete_state}, incomplete_path)
    incomplete = MainNamespacePolicy(profile, str(incomplete_path), with_motion=True, train_vlm=False,
                                     checkpoint_frame="lidar")
    try:
        incomplete.initialize()
    except KeyError as exc:
        check("head.projector_lora.weight" in str(exc),
              "DeepSpeed migration rejects a missing frozen projector adapter tensor")
    else:
        raise AssertionError("DeepSpeed migration silently accepted a missing frozen projector adapter")


def check_main_stage3_checkpoint_loading(root):
    profile_path = Path(__file__).resolve().parents[1] / "config" / "profile" / "nuscenes.yaml"
    profile = Profile.from_config(OmegaConf.load(profile_path))
    profile.prefix_max_len = 3840
    profile.official_attention_mask = False
    profile.official_command_text = False
    profile.lr_microstep = True
    profile.motion_anchor_encoder = "MotionAnchorSineEncoder"
    torch.manual_seed(79)
    source = MainNamespacePolicy(profile, with_motion=True, train_vlm=False,
                                 checkpoint_transition="nusc_2b_stage2_to_stage3", lr=1.0e-4)
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), 1):
            parameter.fill_(float(index) / 13)
    main_state = {
        "planning_head." + name[len("head."):]: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    native_stage3_path = root / "native-stage3.ckpt"
    stage3_agent = {
        "_target_": "unidrivevla.adapters.unidrivevla.UniDriveVLAPolicy",
        "profile": asdict(profile), "train_vlm": False, "with_motion": True, "lr": 1.0e-4,
        "coc_language_prompt": False, "coc_language_loss_weight": 0.0, "coc_jsonl_path": "",
        "coc_planning_loss_weight": 0.0, **source._checkpoint_stage3_options,
    }
    torch.save({"state_dict": source.state_dict(),
                "agent_config": stage3_agent}, native_stage3_path)
    native_restored = MainNamespacePolicy(profile, str(native_stage3_path), with_motion=True, train_vlm=False,
                                          checkpoint_frame="lidar", checkpoint_transition="nusc_2b_stage2_to_stage3",
                                          lr=1.0e-4)
    native_restored.initialize(strict=True)
    check(all(torch.equal(value, native_restored.state_dict()[name])
              for name, value in source.state_dict().items()),
          "strict evaluation loads this repository's native Stage3 checkpoint")

    old_main_stage3_path = root / "old-main-stage3.pt"
    torch.save({"module": main_state}, old_main_stage3_path)
    main_restored = MainNamespacePolicy(profile, str(old_main_stage3_path), with_motion=True, train_vlm=False,
                                        checkpoint_frame="lidar", checkpoint_transition="nusc_2b_stage2_to_stage3",
                                        lr=1.0e-4)
    main_restored.initialize(strict=True)
    check(all(torch.equal(value, main_restored.state_dict()["head." + name[len("head."):]])
              for name, value in source.state_dict().items()),
          "strict evaluation maps a complete planning_head Stage3 checkpoint")

    stage3_train_start = MainNamespacePolicy(profile, str(old_main_stage3_path), with_motion=True, train_vlm=False,
                                             checkpoint_frame="lidar",
                                             checkpoint_transition="nusc_2b_stage2_to_stage3", lr=1.0e-4)
    try:
        stage3_train_start.initialize(strict=False)
    except ValueError as exc:
        check("same-stage training resume" in str(exc),
              "dedicated Stage3 training rejects a motion-enabled metadata-free main checkpoint")
    else:
        raise AssertionError("Stage3 training accepted a motion-enabled DeepSpeed checkpoint as a Stage2 start")

    extra_state = dict(main_state)
    extra_state["planning_head.unexpected.weight"] = torch.ones(1)
    extra_path = root / "old-main-stage3-extra.pt"
    torch.save({"module": extra_state}, extra_path)
    extra_target = MainNamespacePolicy(profile, str(extra_path), with_motion=True, train_vlm=False,
                                      checkpoint_frame="lidar", checkpoint_transition="nusc_2b_stage2_to_stage3",
                                      lr=1.0e-4)
    try:
        extra_target.initialize(strict=True)
    except KeyError as exc:
        check("unexpected model tensors" in str(exc), "strict main checkpoint rejects unrelated model tensors")
    else:
        raise AssertionError("strict loading silently ignored an unexpected main model tensor")


def check_stage3_checkpoint_config_contract(root):
    profile_path = Path(__file__).resolve().parents[1] / "config" / "profile" / "nuscenes.yaml"
    target_profile = Profile.from_config(OmegaConf.load(profile_path))
    target_profile.prefix_max_len = 3840
    target_profile.official_attention_mask = False
    target_profile.official_command_text = False
    target_profile.lr_microstep = True
    target_profile.motion_anchor_encoder = "MotionAnchorSineEncoder"
    source_profile = asdict(target_profile)
    source_profile["inference_attn_impl"] = "eager"
    source_profile["motion_anchor_encoder"] = None  # Stage 2 has no motion branch.
    options = {
        "weight_decay": 1.0e-7, "warmup_steps": 125, "warmup_start": 1.0 / 3,
        "min_lr_ratio": 1.0e-3, "vlm_lr_mult": 0.5, "vlm_variant": "2b",
        "gradient_checkpointing": True, "cache_frozen_prefix": True, "attn_implementation": "sdpa",
        "canbus_sft_prompt": False, "vlm_path": "/weights/qwen", "vae_path": "/weights/vae",
        "lora": {"r": 64, "lora_alpha": 128, "target_modules": ["q_proj", "v_proj"]},
        "visual_lora": {"r": 64, "lora_alpha": 128, "target_modules": "projector"},
        "visual_wasp": {"rank": 64, "alpha": 128, "top_k": 16, "beta": 0.01,
                         "target_modules": ["qkv", "proj"]},
        "llm_wasp": {"rank": 64, "alpha": 128, "top_k": 16, "beta": 0.003,
                      "target_modules": ["q_proj", "v_proj"]},
        "visual_projector_wasp": None,
    }
    source_agent = {
        "_target_": "unidrivevla.adapters.unidrivevla.UniDriveVLAPolicy",
        "profile": source_profile,
        "train_vlm": True, "with_motion": False, "lr": 2.0e-4,
        "coc_language_prompt": True, "coc_language_loss_weight": 0.1,
        "coc_jsonl_path": "/data/stage2-coc.jsonl", "coc_planning_loss_weight": 0.0,
        **deepcopy(options),
    }
    target = SimpleNamespace(
        checkpoint_transition="nusc_2b_stage2_to_stage3", profile=target_profile,
        train_vlm=False, with_motion=True, lr=1.0e-4, coc_language_prompt=False,
        coc_language_loss_weight=0.0, coc_jsonl_path="", coc_planning_loss_weight=0.0,
        _checkpoint_stage3_options=options,
    )
    _validate_stage3_checkpoint_config(target, {"agent_config": source_agent}, source_motion=False)
    check(True, "Stage3 continuation permits only the recorded motion/encoder, CoC-off, freeze, and LR transition")

    # Exercise the same contract through the actual initialize() entry point, not only the validator helper.
    source_native_profile = deepcopy(target_profile)
    source_native_profile.motion_anchor_encoder = None
    source_native = MainNamespacePolicy(source_native_profile, with_motion=False, train_vlm=True,
                                        checkpoint_frame="lidar", lr=2.0e-4)
    compatible_path = root / "native-stage2-compatible.ckpt"
    torch.save({"state_dict": source_native.state_dict(), "agent_config": source_agent}, compatible_path)
    migrated = MainNamespacePolicy(target_profile, str(compatible_path), with_motion=True, train_vlm=False,
                                   checkpoint_frame="lidar", checkpoint_transition="nusc_2b_stage2_to_stage3",
                                   lr=1.0e-4)
    migrated._checkpoint_stage3_options = {**options, "cache_frozen_prefix": False}
    migrated.initialize()
    check(all(torch.equal(source_native.state_dict()[name], migrated.state_dict()[name])
              for name in source_native.state_dict()),
          "native Stage2 agent_config passes the dedicated Stage3 continuation through initialize()")

    incompatible_agent = deepcopy(source_agent)
    incompatible_agent["visual_wasp"]["alpha"] = 256  # same tensor shape; adapter semantics differ
    incompatible_path = root / "native-stage2-incompatible-adapter.ckpt"
    torch.save({"state_dict": source_native.state_dict(), "agent_config": incompatible_agent}, incompatible_path)
    incompatible = MainNamespacePolicy(target_profile, str(incompatible_path), with_motion=True, train_vlm=False,
                                       checkpoint_frame="lidar", checkpoint_transition="nusc_2b_stage2_to_stage3",
                                       lr=1.0e-4)
    incompatible._checkpoint_stage3_options = options
    try:
        incompatible.initialize()
    except ValueError as exc:
        check("agent.visual_wasp" in str(exc),
              "initialize() rejects same-shape Stage2 checkpoints with changed WASP alpha")
    else:
        raise AssertionError("Stage3 continuation accepted same-shape mismatched adapter metadata")

    generic = MainNamespacePolicy(target_profile, str(incompatible_path), with_motion=True, train_vlm=False,
                                 checkpoint_frame="lidar", checkpoint_transition=None, lr=1.0e-4)
    generic.initialize()
    check(True, "generic Stage2-to-motion fine-tuning remains outside the dedicated compatibility contract")

    stage3_source_agent = deepcopy(source_agent)
    stage3_source_agent.update({
        "profile": asdict(target_profile), "train_vlm": False, "with_motion": True,
        "lr": 1.0e-4, "coc_language_prompt": False, "coc_language_loss_weight": 0.0,
        "coc_jsonl_path": "", "coc_planning_loss_weight": 0.0,
        "checkpoint_frame": "lidar",
        "checkpoint_transition": "nusc_2b_stage2_to_stage3",
    })
    stage3_model = MainNamespacePolicy(target_profile, with_motion=True, train_vlm=False,
                                       checkpoint_frame="lidar", lr=1.0e-4)
    stage3_path = root / "native-stage3-agent-config.ckpt"
    torch.save({"state_dict": stage3_model.state_dict(), "agent_config": stage3_source_agent}, stage3_path)
    from unidrivevla.infer import restore_checkpoint_agent_config
    infer_cfg = OmegaConf.create({"data_loader": "nuscenes", "profile": {"name": "nuscenes"}, "agent": {}})
    check(restore_checkpoint_agent_config(infer_cfg, stage3_path),
          "infer.py restores native Stage3 metadata before strict checkpoint loading")
    infer_cfg.profile.inference_attn_impl = "eager"
    restore_checkpoint_agent_config(infer_cfg, stage3_path)
    check(infer_cfg.agent.profile.inference_attn_impl == "sdpa", "default inference keeps the saved backend")
    restore_checkpoint_agent_config(infer_cfg, stage3_path, task_overrides=["++profile.inference_attn_impl=eager"])
    check(infer_cfg.agent.profile.inference_attn_impl == "eager", "explicit inference backend survives restoration")
    check(infer_cfg.agent.gradient_checkpointing is False,
          "strict inference applies the existing gradient-checkpointing override")
    restored_profile = Profile.from_config(OmegaConf.to_container(infer_cfg.agent.profile, resolve=True))
    infer_target = MainNamespacePolicy(
        restored_profile, str(stage3_path), with_motion=bool(infer_cfg.agent.with_motion),
        train_vlm=bool(infer_cfg.agent.train_vlm), checkpoint_frame=infer_cfg.agent.checkpoint_frame,
        checkpoint_transition=infer_cfg.agent.checkpoint_transition, lr=float(infer_cfg.agent.lr))
    infer_target._checkpoint_stage3_options["gradient_checkpointing"] = bool(infer_cfg.agent.gradient_checkpointing)
    infer_target.initialize(strict=True)
    from unidrivevla.adapters import load_checkpoint_agent_config
    old = deepcopy(source_agent)
    old["profile"].pop("inference_attn_impl")
    old["profile"].pop("compile_feature_fusion")
    old["profile"]["status_layout"] = list(range(10))
    old["canbus_sft_prompt"] = False
    old_path = root / "older-profile.ckpt"
    for changes, expected in (({}, "eager"), ({"canbus_sft_prompt": True}, None),
                              ({"coc_language_prompt": False}, None), ({"with_motion": True}, None)):
        torch.save({"agent_config": {**old, **changes}}, old_path)
        migrated_profile = load_checkpoint_agent_config(old_path)["profile"]
        assert migrated_profile.get("inference_attn_impl") == expected
        assert migrated_profile.get("compile_feature_fusion") == (True if expected else None)
    old_stage3 = {**old, "with_motion": True, "coc_language_prompt": False,
                  "checkpoint_transition": "nusc_2b_stage2_to_stage3"}
    torch.save({"agent_config": old_stage3}, old_path)
    assert load_checkpoint_agent_config(old_path)["profile"]["compile_feature_fusion"] is True
    old["profile"]["compile_feature_fusion"] = False
    old["profile"]["inference_attn_impl"] = "sdpa"
    torch.save({"agent_config": old}, old_path)
    assert load_checkpoint_agent_config(old_path)["profile"]["inference_attn_impl"] == "sdpa"
    assert load_checkpoint_agent_config(old_path)["profile"]["compile_feature_fusion"] is False
    multi = {"_target_": "unidrivevla.adapters.multidomain.MultiProfilePolicy",
             "profiles": {name: {"name": name, "inference_attn_impl": "sdpa"} for name in ("nuscenes", "b2d")}}
    torch.save({"agent_config": multi}, old_path)
    restore_checkpoint_agent_config(infer_cfg, old_path, task_overrides=["++profile.inference_attn_impl=eager"])
    assert [infer_cfg.agent.profiles[name].inference_attn_impl for name in ("nuscenes", "b2d")] == ["eager", "sdpa"]
    check(True, "infer.py gradient-checkpointing override passes strict Stage3 metadata compatibility")

    stage3_training_target = MainNamespacePolicy(
        target_profile, str(stage3_path), with_motion=True, train_vlm=False, checkpoint_frame="lidar",
        checkpoint_transition="nusc_2b_stage2_to_stage3", lr=1.0e-4)
    try:
        stage3_training_target.initialize()
    except ValueError as exc:
        check("same-stage training resume" in str(exc),
              "dedicated Stage2-to-Stage3 training rejects a motion-enabled checkpoint (use DCP resume)")
    else:
        raise AssertionError("dedicated Stage3 training accepted a motion-enabled same-stage checkpoint")

    negative_cases = []
    changed = deepcopy(source_agent)
    changed["profile"]["inference_attn_impl"] = "sddpa"
    negative_cases.append((changed, "inference_attn_impl"))
    changed = deepcopy(source_agent)
    changed["profile"]["official_command_text"] = True
    negative_cases.append((changed, "profile mismatch"))
    changed = deepcopy(source_agent)
    changed["profile"]["delta_mean"][0] += 0.01
    negative_cases.append((changed, "profile mismatch"))
    changed = deepcopy(source_agent)
    changed["profile"]["pose_interval"] = 0.25
    negative_cases.append((changed, "profile mismatch"))
    changed = deepcopy(source_agent)
    changed["visual_wasp"]["alpha"] = 256
    negative_cases.append((changed, "agent.visual_wasp"))
    changed = deepcopy(source_agent)
    changed["visual_wasp"]["top_k"] = 8
    negative_cases.append((changed, "agent.visual_wasp"))
    changed = deepcopy(source_agent)
    changed["llm_wasp"]["beta"] = 0.004
    negative_cases.append((changed, "agent.llm_wasp"))
    changed = deepcopy(source_agent)
    changed["visual_lora"]["lora_alpha"] = 64
    negative_cases.append((changed, "agent.visual_lora"))
    for bad_agent, expected in negative_cases:
        try:
            _validate_stage3_checkpoint_config(target, {"agent_config": bad_agent}, source_motion=False)
        except ValueError as exc:
            check(expected in str(exc), f"Stage3 continuation rejects changed same-shape semantics ({expected})")
        else:
            raise AssertionError(f"Stage3 continuation accepted a changed {expected} checkpoint config")

    generic = SimpleNamespace(checkpoint_transition=None)
    _validate_stage3_checkpoint_config(generic, {"agent_config": negative_cases[0][0]}, source_motion=False)
    check(True, "generic fine-tuning is not gated by the dedicated Stage3 checkpoint contract")
    stage3_source = deepcopy(source_agent)
    stage3_source["train_vlm"], stage3_source["with_motion"] = False, True
    stage3_source["lr"] = 1.0e-4
    stage3_source["coc_language_prompt"] = False
    stage3_source["coc_language_loss_weight"] = 0.0
    stage3_source["coc_jsonl_path"] = ""
    stage3_source["profile"]["motion_anchor_encoder"] = "MotionAnchorSineEncoder"
    stage3_source["profile"]["official_command_text"] = True  # explicit live inference override
    _validate_stage3_checkpoint_config(target, {"agent_config": stage3_source}, source_motion=True, strict=True)
    online_profile = deepcopy(target_profile)
    online_profile.official_command_text = True
    online_source = MainNamespacePolicy(online_profile, with_motion=True, train_vlm=False,
                                       checkpoint_frame="lidar", lr=1.0e-4)
    online_agent = deepcopy(stage3_source_agent)
    online_agent["profile"] = asdict(online_profile)
    online_path = root / "native-stage3-online-nav-override.ckpt"
    torch.save({"state_dict": online_source.state_dict(), "agent_config": online_agent}, online_path)
    online_target = MainNamespacePolicy(
        target_profile, str(online_path), with_motion=True, train_vlm=False, checkpoint_frame="lidar",
        checkpoint_transition="nusc_2b_stage2_to_stage3", lr=1.0e-4)
    online_target.initialize(strict=True)
    check(True, "strict Stage3 inference permits only the supported one-hot navigation-text override")

    bad_stage3_agent = deepcopy(stage3_source_agent)
    bad_stage3_agent["llm_wasp"]["beta"] = 0.004
    bad_stage3_path = root / "native-stage3-bad-peft.ckpt"
    torch.save({"state_dict": stage3_model.state_dict(), "agent_config": bad_stage3_agent}, bad_stage3_path)
    bad_stage3_target = MainNamespacePolicy(
        target_profile, str(bad_stage3_path), with_motion=True, train_vlm=False, checkpoint_frame="lidar",
        checkpoint_transition="nusc_2b_stage2_to_stage3", lr=1.0e-4)
    try:
        bad_stage3_target.initialize(strict=True)
    except ValueError as exc:
        check("agent.llm_wasp" in str(exc), "strict Stage3 inference rejects same-shape PEFT beta drift")
    else:
        raise AssertionError("strict Stage3 inference accepted same-shape mismatched PEFT metadata")
    _validate_stage3_checkpoint_config(target, {"meta": {"legacy_main_ds": True}}, source_motion=False)
    check(True, "Stage3 online overrides and metadata-free DeepSpeed checkpoints keep their existing paths")


def start_cpu_process_group():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


class Context:
    is_main = True
    rank = 0
    world_size = 1

    def barrier(self):
        pass

    def gloo_group(self):
        if not hasattr(self, "group"):
            self.group = dist.new_group(backend="gloo")
        return self.group


def check_sparse_optimizer_resume(root, optimizer_name):
    def make_optimizer(model):
        if optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
        return MasterAdamW(model.parameters(), lr=0.002, weight_decay=0.01)

    torch.manual_seed(31)
    initial = BranchPolicy().state_dict()
    uninterrupted, interrupted = BranchPolicy(), BranchPolicy()
    uninterrupted.load_state_dict(initial)
    interrupted.load_state_dict(initial)
    continuous_optimizer, saved_optimizer = make_optimizer(uninterrupted), make_optimizer(interrupted)
    branch_step(uninterrupted, continuous_optimizer, "nuscenes")
    branch_step(interrupted, saved_optimizer, "nuscenes")

    # MasterAdamW can create zero-valued state for no-gradient parameters during step();
    # remove those untouched-profile placeholders to model a genuinely cold branch.
    for model, optimizer in ((uninterrupted, continuous_optimizer), (interrupted, saved_optimizer)):
        for name, parameter in model.named_parameters():
            if name.startswith("heads.navsim."):
                optimizer.state.pop(parameter, None)
    saved_states = optimizer_states_by_name(interrupted, saved_optimizer)
    check(saved_states and all(name.startswith(("shared.", "heads.nuscenes.")) for name in saved_states),
          f"{optimizer_name} state remains sparse before the navsim branch is used")

    checkpointer = ResumeCheckpointer(root / f"sparse-{optimizer_name}", Context(), async_save=False)
    checkpointer.save(1, interrupted, saved_optimizer, {"epoch": 0, "global_step": 1, "best_val": 0.0})
    checkpoint = checkpointer.latest()
    metadata = FileSystemReader(str(checkpoint)).read_metadata()
    saved_optimizer_keys = {key for key in metadata.state_dict_metadata if key.startswith("optim.state.")}
    check(not any(key.startswith("optim.state.heads.navsim.") for key in saved_optimizer_keys),
          f"{optimizer_name} DCP metadata omits cold navsim slots")

    resumed = BranchPolicy()
    resumed.load_state_dict(initial)
    resumed_optimizer = make_optimizer(resumed)
    checkpointer.load(checkpoint, resumed, resumed_optimizer)
    restored_states = optimizer_states_by_name(resumed, resumed_optimizer)
    check(tree_equal(saved_states, restored_states), f"{optimizer_name} restores only source optimizer slots")
    check(not any(name.startswith("heads.navsim.") for name in restored_states),
          f"{optimizer_name} leaves the unused profile optimizer state uninitialized")
    check(not any(name.startswith("heads.navsim.") and parameter in resumed_optimizer.state
                  for name, parameter in resumed.named_parameters()),
          f"{optimizer_name} does not persist empty cold-profile placeholders")

    branch_step(uninterrupted, continuous_optimizer, "navsim")
    branch_step(resumed, resumed_optimizer, "navsim")
    check(tree_equal(uninterrupted.state_dict(), resumed.state_dict()),
          f"{optimizer_name} first-use branch weights match uninterrupted training bitwise")
    check(tree_equal(optimizer_states_by_name(uninterrupted, continuous_optimizer),
                     optimizer_states_by_name(resumed, resumed_optimizer)),
          f"{optimizer_name} first-use optimizer state matches uninterrupted training bitwise")
    if optimizer_name == "master_adamw":
        states = optimizer_states_by_name(resumed, resumed_optimizer)
        check(states["shared.weight"]["step"].item() == 2
              and states["heads.nuscenes.weight"]["step"].item() == 1
              and states["heads.navsim.weight"]["step"].item() == 1,
              "MasterAdamW preserves actual shared/profile step counts across restore")


def check_inference_compute_dtypes(root):
    from types import SimpleNamespace

    source = ComputeDtypePolicy(dtype=torch.bfloat16)
    with torch.no_grad():
        source.primary.add_(torch.tensor([0.125, -0.25], dtype=torch.bfloat16))
        source.frozen.copy_(torch.tensor([-1.25, 0.375], dtype=torch.bfloat16))
    checkpoint = root / "bf16-compute.ckpt"
    trainer = Trainer.__new__(Trainer)
    trainer.agent = source
    trainer.args = SimpleNamespace(strategy="ddp", precision="native")
    trainer.write_ckpt(source.state_dict(), checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check(saved["compute_parameter_dtypes"] == {"primary": "bfloat16", "frozen": "bfloat16"},
          "training checkpoint records actual BF16 compute dtypes once per tied parameter")

    restored = ComputeDtypePolicy(checkpoint_path=str(checkpoint))
    parameter_identity, tied_identity = id(restored.primary), id(restored.tied_alias)
    initialize_for_inference(restored)
    check(restored.primary.dtype == torch.bfloat16 and restored.frozen.dtype == torch.bfloat16,
          "strict inference initialization restores trainable and frozen BF16 parameter dtypes")
    check(torch.equal(source(), restored()), "BF16 training and inference outputs match bitwise after round trip")
    captures = []
    def capture(tr, ckpt, tag):
        assert tr.model is tr.agent and tr.model.primary.dtype == torch.float32
        assert torch.equal(tr.model.primary, saved["state_dict"]["primary"].float())
        assert torch.get_float32_matmul_precision() == "highest"
        captures.append((Path(ckpt).name, tag))

    formal = SimpleNamespace(use_model_init_dtypes=True, on_end=True, benchmark="test",
                             during_training=capture, release=lambda: None, finish=lambda _: [])
    trainer.ctx = SimpleNamespace(device=torch.device("cpu"), is_main=True, world_size=1,
                                  barrier=lambda: None, close=lambda: None)
    trainer.agent = ComputeDtypePolicy(checkpoint_path=str(checkpoint))
    old_precision = torch.get_float32_matmul_precision()
    trainer.evaluate_checkpoint_in_model_dtypes([formal], checkpoint, "infer")
    assert torch.get_float32_matmul_precision() == old_precision
    trainer.output_dir = root / "final-pr10"
    trainer.output_dir.mkdir()
    torch.save(saved, trainer.output_dir / "last.ckpt")
    trainer.evaluators, trainer.ema = [formal], None
    trainer.cfg = OmegaConf.create({"agent": {}})
    trainer.monitor = SimpleNamespace(close=lambda: None)
    with patch("unidrivevla.train.trainer.instantiate",
               side_effect=lambda cfg: ComputeDtypePolicy(checkpoint_path=cfg["checkpoint_path"])):
        trainer.finish()
    check(captures == [(checkpoint.name, "infer"), ("last.ckpt", "final")],
          "standalone and final PR10 scoring load the saved weights in model construction dtypes")
    check(id(restored.primary) == parameter_identity == id(restored.tied_alias) == tied_identity,
          "dtype restoration preserves Parameter identity and tied aliases")
    check(restored.float_buffer.dtype == torch.float32 and torch.equal(restored.float_buffer, source.float_buffer)
          and restored.integer_buffer.dtype == torch.int64,
          "inference dtype restoration leaves floating and integer buffers unchanged")

    legacy = root / "legacy-no-compute-dtypes.ckpt"
    torch.save({"state_dict": source.state_dict()}, legacy)
    legacy_target = ComputeDtypePolicy(checkpoint_path=str(legacy))
    initialize_for_inference(legacy_target)
    check(legacy_target.primary.dtype == torch.float32 and legacy_target.frozen.dtype == torch.float32,
          "legacy checkpoints without dtype metadata retain historical inference behavior")

    invalid = root / "invalid-compute-dtype.ckpt"
    torch.save({"state_dict": source.state_dict(), "compute_parameter_dtypes": {
        "primary": "bfloat16", "frozen": "arbitrary"}}, invalid)
    invalid_target = ComputeDtypePolicy(checkpoint_path=str(invalid))
    try:
        initialize_for_inference(invalid_target)
    except ValueError as exc:
        check("unsupported" in str(exc), "inference rejects non-allowlisted dtype names")
    else:
        raise AssertionError("inference accepted an arbitrary dtype name")
    check(invalid_target.primary.dtype == torch.float32,
          "invalid dtype maps are rejected before any parameter is cast")

    incomplete = root / "incomplete-compute-dtypes.ckpt"
    torch.save({"state_dict": source.state_dict(), "compute_parameter_dtypes": {"primary": "bfloat16"}},
               incomplete)
    try:
        initialize_for_inference(ComputeDtypePolicy(checkpoint_path=str(incomplete)))
    except ValueError as exc:
        check("does not match" in str(exc), "inference rejects partial dtype maps")
    else:
        raise AssertionError("inference accepted a partial dtype map")
    try:
        initialize_for_inference(ComputeDtypePolicy(checkpoint_path=str(incomplete)), restore_compute_dtypes=False)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("PR10 model-dtype scoring accepted incomplete metadata")

    fsdp_source = ComputeDtypePolicy(dtype=torch.float32, frozen_dtype=torch.float64)
    fsdp_args = SimpleNamespace(strategy="fsdp", precision="bf16", replicate_frozen=["frozen"])
    integer_before = fsdp_source.frozen_integer.detach().clone()
    fsdp_dtypes = _compute_parameter_dtypes(fsdp_source, fsdp_args)
    check(fsdp_dtypes == {"primary": "bfloat16", "frozen": "float64"},
          "FSDP dtype metadata uses execution precision for managed params and source dtype for replicated floats")
    check(fsdp_source.frozen_integer.dtype == torch.int64
          and torch.equal(fsdp_source.frozen_integer, integer_before)
          and "frozen_integer" not in fsdp_dtypes,
          "replicated nonfloating scalar parameters stay outside the compute dtype map")

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

    wrapped = ComputeDtypePolicy(dtype=torch.bfloat16)
    wrapped.primary_layer = checkpoint_wrapper(nn.Linear(2, 2, dtype=torch.float32),
                                                checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    wrapped_dtypes = _compute_parameter_dtypes(wrapped, SimpleNamespace(strategy="ddp", precision="native"))
    check("primary_layer.weight" in wrapped_dtypes
          and all("_checkpoint_wrapped_module" not in name for name in wrapped_dtypes),
          "dtype metadata canonicalizes actual activation-checkpoint wrapper parameter names")

    legacy_trainer = Trainer.__new__(Trainer)
    legacy_trainer.cfg = None
    legacy_payload_path = root / "legacy-trainer-payload.ckpt"
    legacy_trainer.write_ckpt({"sentinel": torch.tensor(1)}, legacy_payload_path)
    legacy_payload = torch.load(legacy_payload_path, map_location="cpu", weights_only=True)
    check(set(legacy_payload) == {"state_dict"},
          "Trainer.__new__ without agent or args keeps the historical checkpoint payload")


def main():
    torch.manual_seed(7)
    check(not torch.cuda.is_initialized(), "all model and checkpoint work starts on CPU")
    start_cpu_process_group()
    try:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            check_sparse_optimizer_resume(root, "sgd")
            check_sparse_optimizer_resume(root, "master_adamw")
            check_inference_compute_dtypes(root)
            check_main_checkpoint_migration(root)
            check_main_stage3_checkpoint_loading(root)
            check_stage3_checkpoint_config_contract(root)
            stage1 = TinyPolicy()
            with torch.no_grad():
                stage1.vlm.weight.fill_(1.25)
                stage1.lora.weight.fill_(-0.75)
                stage1.plan.weight.fill_(0.5)
            stage1_path = root / "stage1.ckpt"
            torch.save({"state_dict": export_model_state(stage1)}, stage1_path)

            stage2 = TinyPolicy(checkpoint_path=str(stage1_path), stage2=True)
            stage2.initialize()
            check(torch.equal(stage2.vlm.weight, stage1.vlm.weight), "stage 2 initializes stage-1 VLM weights")
            check(torch.equal(stage2.lora.weight, stage1.lora.weight), "stage 2 initializes stage-1 LoRA weights")
            check(not stage2.vlm.weight.requires_grad and not stage2.lora.weight.requires_grad,
                  "stage-1 VLM and LoRA are frozen in stage 2")

            with torch.no_grad():
                stage2.plan.weight.add_(2.0)
                stage2.motion.weight.fill_(3.0)
            stage2_path = root / "stage2.ckpt"
            torch.save({"state_dict": export_model_state(stage2)}, stage2_path)
            restored = TinyPolicy(checkpoint_path=str(stage2_path), stage2=True)
            restored.initialize()
            check(all(torch.equal(value, restored.state_dict()[name])
                      for name, value in stage2.state_dict().items()),
                  "actual policy initialize restores all stage-2 weights, including frozen VLM and LoRA")
            restored.initialize(strict=True)
            try:
                TinyPolicy(stage2=True).initialize(strict=True)
            except ValueError:
                pass
            else:
                raise AssertionError("strict inference must not run random weights without a checkpoint")
            for omitted in ("vlm.weight", "lora.weight", "plan.weight", "motion.weight"):
                incomplete = root / f"missing-{omitted.replace('.', '-')}.ckpt"
                torch.save({"state_dict": {name: value for name, value in stage2.state_dict().items()
                                           if name != omitted}}, incomplete)
                try:
                    TinyPolicy(checkpoint_path=str(incomplete), stage2=True).initialize()
                except KeyError as exc:
                    check(omitted in str(exc), f"training initialization rejects missing {omitted}")
                else:
                    raise AssertionError(f"training initialization silently accepted missing {omitted}")

            bad_motion_meta = root / "bad-motion-meta.ckpt"
            torch.save({"state_dict": {name: value for name, value in stage2.state_dict().items()
                                       if not name.startswith("motion.")},
                        "agent_config": {"with_motion": True}}, bad_motion_meta)
            try:
                TinyPolicy(checkpoint_path=str(bad_motion_meta), stage2=True).initialize()
            except ValueError as exc:
                check("motion metadata" in str(exc), "checkpoint motion metadata cannot bless a truncated stage-2 state")
            else:
                raise AssertionError("metadata made a checkpoint with every motion tensor removed look complete")

            optimizer = torch.optim.AdamW((p for p in stage2.parameters() if p.requires_grad), lr=1e-3)
            loss = sum(p.square().sum() for p in stage2.parameters() if p.requires_grad)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            saved = training_state(stage2, optimizer)
            check(set(saved["model"]) == set(stage2.state_dict()),
                  "resume model state includes frozen and trainable tensors")

            checkpointer = ResumeCheckpointer(root / "resume", Context(), async_save=False)
            checkpointer.save(1, stage2, optimizer, {"epoch": 1, "global_step": 1, "best_val": 0.0})
            resume_path = checkpointer.latest()
            saved_extra = torch.load(resume_path / DONE_FILE, map_location="cpu", weights_only=False)
            check(saved_extra["model_state_complete"], "new resume metadata marks the full model state")
            resumed = TinyPolicy(stage2=True)
            resumed_optimizer = torch.optim.AdamW(
                (p for p in resumed.parameters() if p.requires_grad), lr=1e-3)
            checkpointer.load(checkpointer.latest(), resumed, resumed_optimizer)
            check(all(torch.equal(value, resumed.state_dict()[name])
                      for name, value in stage2.state_dict().items()),
                  "DCP resume restores learned frozen weights and trainable weights")

            async_checkpointer = ResumeCheckpointer(root / "async_resume", Context(), async_save=True)
            async_checkpointer.save(2, stage2, optimizer, {"epoch": 2, "global_step": 2, "best_val": 0.0})
            check(async_checkpointer.latest() is None, "async checkpoint stays incomplete until wait finalizes metadata")
            async_checkpointer.wait()
            async_restored = TinyPolicy(stage2=True)
            async_optimizer = torch.optim.AdamW(
                (p for p in async_restored.parameters() if p.requires_grad), lr=1e-3)
            async_checkpointer.load(async_checkpointer.latest(), async_restored, async_optimizer)
            check(all(torch.equal(value, async_restored.state_dict()[name])
                      for name, value in stage2.state_dict().items()),
                  "async DCP round trip completes data before publishing its marker")

            failed_future_checkpointer = ResumeCheckpointer(root / "failed_async", Context(), async_save=True)
            failed_future = Future()
            failed_future.set_exception(CheckpointException("injected async data failure", {}))
            with patch.object(dcp, "async_save", return_value=failed_future):
                failed_future_checkpointer.save(3, stage2, optimizer,
                                                {"epoch": 3, "global_step": 3, "best_val": 0.0})
            try:
                failed_future_checkpointer.wait()
            except RuntimeError as exc:
                check(isinstance(exc.__cause__, CheckpointException), "async DCP CheckpointException propagates from wait")
            else:
                check(False, "async DCP CheckpointException propagates from wait")
            check(failed_future_checkpointer.latest() is None, "failed async data write never gets a completion marker")

            failed_start_checkpointer = ResumeCheckpointer(root / "failed_async_start", Context(), async_save=True)
            with patch.object(dcp, "async_save",
                              side_effect=CheckpointException("injected async start failure", {})):
                failed_start_checkpointer.save(5, stage2, optimizer,
                                               {"epoch": 5, "global_step": 5, "best_val": 0.0})
            try:
                failed_start_checkpointer.wait()
            except RuntimeError as exc:
                check(isinstance(exc.__cause__, CheckpointException), "async start CheckpointException propagates")
            else:
                check(False, "async start CheckpointException propagates")
            check(failed_start_checkpointer.latest() is None,
                  "async start CheckpointException never gets a completion marker")

            failed_sync_checkpointer = ResumeCheckpointer(root / "failed_sync", Context(), async_save=False)
            with patch.object(dcp, "save", side_effect=CheckpointException("injected sync write failure", {})):
                try:
                    failed_sync_checkpointer.save(6, stage2, optimizer,
                                                  {"epoch": 6, "global_step": 6, "best_val": 0.0})
                except RuntimeError as exc:
                    check(isinstance(exc.__cause__, CheckpointException), "sync DCP CheckpointException propagates")
                else:
                    check(False, "sync DCP CheckpointException propagates")
            check(failed_sync_checkpointer.latest() is None,
                  "sync DCP CheckpointException never gets a completion marker")

            failed_marker_checkpointer = ResumeCheckpointer(root / "failed_marker", Context(), async_save=True)
            failed_marker_checkpointer.save(4, stage2, optimizer,
                                            {"epoch": 4, "global_step": 4, "best_val": 0.0})
            def fail_marker(_path, _extra):
                raise OSError("injected completion marker failure")
            failed_marker_checkpointer._write_done = fail_marker
            try:
                failed_marker_checkpointer.wait()
            except RuntimeError as exc:
                check("injected completion marker failure" in str(exc), "marker write errors propagate from wait")
            else:
                check(False, "marker write errors propagate from wait")
            check(failed_marker_checkpointer.latest() is None,
                  "failed marker write never makes an incomplete checkpoint visible")

            model_sd, optim_sd = get_state_dict(
                stage2, optimizer, options=StateDictOptions(ignore_frozen_params=True))
            legacy_path = root / "legacy_resume"
            dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(legacy_path),
                     process_group=Context().gloo_group())
            torch.save({"epoch": 1, "global_step": 1, "best_val": 0.0, "ema": False},
                       legacy_path / DONE_FILE)
            legacy_target = TinyPolicy(checkpoint_path=str(stage1_path), stage2=True)
            legacy_target.initialize()
            legacy_optimizer = torch.optim.AdamW(
                (p for p in legacy_target.parameters() if p.requires_grad), lr=1e-3)
            checkpointer.load(legacy_path, legacy_target, legacy_optimizer)
            check(torch.equal(legacy_target.vlm.weight, stage1.vlm.weight)
                  and torch.equal(legacy_target.lora.weight, stage1.lora.weight),
                  "pre-fix DCP resume keeps frozen VLM and LoRA from its stage-1 initialization")

            corrupt_path = root / "corrupt_marked_resume"
            dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(corrupt_path),
                     process_group=Context().gloo_group())
            torch.save({"epoch": 1, "global_step": 1, "best_val": 0.0, "ema": False,
                        "model_state_complete": True}, corrupt_path / DONE_FILE)
            corrupt_target = TinyPolicy(stage2=True)
            corrupt_optimizer = torch.optim.AdamW(
                (p for p in corrupt_target.parameters() if p.requires_grad), lr=1e-3)
            try:
                checkpointer.load(corrupt_path, corrupt_target, corrupt_optimizer)
            except CheckpointException:
                check(True, "marked complete resume with missing frozen weights fails without fallback")
            else:
                check(False, "marked complete resume with missing frozen weights fails without fallback")
    finally:
        dist.destroy_process_group()
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}")
        sys.exit(1)
