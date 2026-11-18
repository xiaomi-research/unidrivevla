"""CPU check that model registration is FlashAttention-optional but explicit use fails before allocation."""

import importlib.abc
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch


class _BlockFlashAttn(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flash_attn" or fullname.startswith("flash_attn."):
            self.attempts.append(fullname)
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None


def _check_profile_hf_backends():
    from omegaconf import OmegaConf

    from unidrivevla.adapters.unidrivevla import Profile, head_kwargs

    expected = {
        "nuscenes": "flash_attention_2",
        "b2d": "flash_attention_2",
        "navsim": "sdpa",
        "lead": "sdpa",
    }
    for name, backend in expected.items():
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "unidrivevla" / "config" / "profile" / f"{name}.yaml"),
                                     resolve=False)
        profile = Profile.from_config(cfg)
        if profile.hf_attn_impl != backend:
            raise AssertionError(f"profile {name!r} selects HF backend {profile.hf_attn_impl!r}, expected {backend!r}")
        kwargs = head_kwargs(profile, "unused", None, False, False)
        if kwargs.get("hf_attn_impl") != backend:
            raise AssertionError(f"head_kwargs dropped profile {name!r} HF backend {backend!r}")

    if Profile.__dataclass_fields__["hf_attn_impl"].default != "flash_attention_2":
        raise AssertionError("profiles without hf_attn_impl must preserve legacy FlashAttention metadata behavior")
    for name in ("nuscenes_ego", "b2d_ego"):
        legacy = OmegaConf.to_container(
            OmegaConf.load(ROOT / "unidrivevla" / "config" / "profile" / f"{name}.yaml"), resolve=False,
        )
        if "hf_attn_impl" in legacy:
            raise AssertionError(f"legacy profile {name!r} should exercise the missing-field default")
        if Profile.from_config(legacy).hf_attn_impl != "flash_attention_2":
            raise AssertionError(f"old profile metadata {name!r} no longer defaults to flash_attention_2")


def _check_tiny_hf_vision_sdpa():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLVisionConfig(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=2,
        spatial_merge_size=1, temporal_patch_size=1, out_hidden_size=16,
        num_position_embeddings=16, deepstack_visual_indexes=[],
    )
    model = Qwen3VLVisionModel(config).cpu().eval()
    model.config._attn_implementation = "sdpa"
    if any(block.attn.config._attn_implementation != "sdpa" for block in model.blocks):
        raise AssertionError("tiny Qwen3-VL vision blocks did not retain the selected HF SDPA backend")
    patches = torch.arange(4 * 3 * 4, dtype=torch.float32).reshape(4, 12) / 48
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    with torch.no_grad():
        outputs = model(patches, grid_thw)
    if outputs[0].shape != (4, 16) or not torch.isfinite(outputs[0]).all():
        raise AssertionError(f"tiny Qwen3-VL HF vision SDPA forward returned {tuple(outputs[0].shape)}")


def main():
    for name in tuple(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            del sys.modules[name]
    original_find_spec = importlib.util.find_spec

    def find_spec_without_flash(name, *args, **kwargs):
        # Availability probes should see the same result as a genuinely absent package.
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return original_find_spec(name, *args, **kwargs)

    blocker = _BlockFlashAttn()
    sys.meta_path.insert(0, blocker)
    importlib.util.find_spec = find_spec_without_flash
    try:
        from unidrivevla.adapters.unidrivevla import register_modules

        register_modules()
        attention = sys.modules["unidrivevla.models.models.attention"]
        if attention._flash_attention_kernels.cache_info().currsize != 0:
            raise AssertionError("normal model registration loaded optional FlashAttention kernels")
        if any(name == "flash_attn" or name.startswith("flash_attn.") for name in sys.modules):
            raise AssertionError("normal model registration imported the optional flash_attn package")
        attempts_before_select = len(blocker.attempts)

        allocations = []
        original_empty = torch.empty

        def tracked_empty(*args, **kwargs):
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                allocations.append((args, kwargs))
            return original_empty(*args, **kwargs)

        torch.empty = tracked_empty
        try:
            constructors = (
                ("FlashAttention", lambda: attention.FlashAttention()),
                ("MultiheadFlashAttention", lambda: attention.MultiheadFlashAttention(embed_dims=16, num_heads=2)),
            )
            for name, construct in constructors:
                try:
                    construct()
                except ImportError as exc:
                    message = str(exc)
                    if "flash-attn==2.7.4.post1 --no-build-isolation" not in message:
                        raise AssertionError(f"{name} reported no actionable install instruction: {message}") from exc
                else:
                    raise AssertionError(f"{name} silently constructed without the explicitly selected backend")
                if allocations:
                    raise AssertionError(f"{name} allocated tensors before reporting missing flash-attn")

            from unidrivevla.models.unidrivevla.dense_heads.qwenvl3_vla_planning_head import QwenVL3APlanningHead

            try:
                QwenVL3APlanningHead(pretrained_path="unused", hf_attn_impl="unsupported")
            except ValueError as exc:
                if "hf_attn_impl" not in str(exc):
                    raise AssertionError(f"unsupported HF backend reported the wrong error: {exc}") from exc
            else:
                raise AssertionError("head constructor accepted an unsupported HF backend")

            _check_profile_hf_backends()
            attempts_before_sdpa = len(blocker.attempts)
            _check_tiny_hf_vision_sdpa()
            if len(blocker.attempts) != attempts_before_sdpa:
                raise AssertionError("real HF Qwen3-VL vision SDPA attempted to import optional flash-attn")
        finally:
            torch.empty = original_empty
    finally:
        sys.meta_path.remove(blocker)
        importlib.util.find_spec = original_find_spec

    if len(blocker.attempts) <= attempts_before_select:
        raise AssertionError("selecting FlashAttention did not attempt to load its optional kernels")
    print("PASS old/new profile HF backend policy, real tiny Qwen3-VL vision SDPA forward, and explicit FlashAttention loading")


if __name__ == "__main__":
    main()
