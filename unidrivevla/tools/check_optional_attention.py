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
        finally:
            torch.empty = original_empty
    finally:
        sys.meta_path.remove(blocker)
        importlib.util.find_spec = original_find_spec

    if len(blocker.attempts) <= attempts_before_select:
        raise AssertionError("selecting FlashAttention did not attempt to load its optional kernels")
    print("PASS model registration imports without flash-attn; explicit FlashAttention selection fails before tensor allocation")


if __name__ == "__main__":
    main()
