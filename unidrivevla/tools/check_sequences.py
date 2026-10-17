"""python unidrivevla/tools/check_sequences.py: the training order and geometry the official recipes rely on, on the CPU.

1. SequenceBatchSampler: every batch slot of every rank walks consecutive frames of one sequence part, the slots start
   on distinct parts, a part keeps one augmentation, and the epochs visit every frame.
2. The geometric augmentation: a projected 3D point lands where load_bgr moves its pixel.
3. The learning rate: mmcv's cosine with linear warmup, the VLM group at lr * vlm_lr_mult.
Exits 1 on a failure."""

import math
import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.adapters.unidrivevla import Profile, UniDriveVLAPolicy, geometry_matrix, load_bgr  # noqa: E402
from unidrivevla.data.registry import SequenceBatchSampler  # noqa: E402

failures = []


def check(ok, message):
    print(("ok   " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)


# 1. order: 7 sequences of 9-20 frames, cut in two, 2 ranks x 2 slots
rng = np.random.default_rng(0)
sequences, start = [], 0
for length in rng.integers(9, 21, size=7):
    sequences.append(list(range(start, start + length)))
    start += length
draws = iter(range(10 ** 6))
samplers = [SequenceBatchSampler(sequences, 2, rank, 2, split=2, seed=3, draw=lambda: next(draws)) for rank in (0, 1)]
part_of = {i: (s, k) for s, seq in enumerate(sequences) for k, part in enumerate(SequenceBatchSampler.cut(seq, 2))
           for i in part}
firsts = {part[0] for seq in sequences for part in SequenceBatchSampler.cut(seq, 2)}
streams = [[batch for _ in range(3) for batch in sampler] for sampler in samplers]  # three epochs
consecutive, same_augment, last = True, True, {}
for step in range(len(streams[0])):
    for rank in range(2):
        for slot, (index, augment) in enumerate(streams[rank][step]):
            previous = last.get((rank, slot))
            if index not in firsts:  # mid-part: the frame after the slot's previous one, same augmentation
                consecutive &= previous is not None and previous[0] == index - 1
                same_augment &= previous is not None and augment == previous[1]
            last[(rank, slot)] = (index, augment)
starts = [part_of[index] for rank in range(2) for index, _ in streams[rank][0]]
check(consecutive, "a slot walks its sequence part frame by frame")
check(len(set(starts)) == len(starts), "the slots start on distinct sequence parts")
check(same_augment, "a sequence part keeps one augmentation")
seen = {index for stream in streams for batch in stream for index, _ in batch}
check(len(seen) == start, f"three epochs visit {len(seen)} of {start} frames")

# 2. geometry: a bright pixel where a point projects, through flip and rotation
width, height = 960, 544
intrinsic = np.array([[800.0, 0, 480], [0, 800.0, 272], [0, 0, 1]])
point = np.array([1.50625, -0.79375, 10.0, 1.0])  # camera frame (x right, y down, z forward); a pixel centre
view = np.eye(4)
view[:3, :3] = intrinsic
u, v = (view @ point)[:2] / point[2]
image = np.zeros((height, width, 3), np.uint8)
image[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2] = 255
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "dot.png")
    Image.fromarray(image).save(path)
    for geometry in ({"flip": False, "rotate": 4.0}, {"flip": True, "rotate": -5.4}, {"flip": True, "rotate": 0.0}):
        moved = load_bgr(path, height, width, geometry=geometry)[..., 0].astype(np.float64)
        ys, xs = np.nonzero(moved > 128)
        found = np.array([xs.mean(), ys.mean()]) + 0.5  # pixel centres
        projected = geometry_matrix(width, height, geometry) @ view @ point
        expected = projected[:2] / projected[2]
        check(np.abs(found - expected).max() < 1.0, f"{geometry}: pixel {found.round(2)}, projection {expected.round(2)}")

# 3. schedule and parameter groups
profile_dir = Path(__file__).parents[1] / "config" / "profile"
nuscenes_profile = Profile.from_config(OmegaConf.load(profile_dir / "nuscenes.yaml"))
b2d_profile = Profile.from_config(OmegaConf.load(profile_dir / "b2d.yaml"))
legacy_profile = Profile.from_config(OmegaConf.load(profile_dir / "nuscenes_ego.yaml"))
check(nuscenes_profile.lr_microstep and b2d_profile.lr_microstep and not legacy_profile.lr_microstep,
      "only native nuScenes/B2D profiles opt into main micro-step LR timing")
model = torch.nn.Module()
model.head = torch.nn.Module()
model.head.qwen3_vl_with_expert = torch.nn.Module()
model.head.qwen3_vl_with_expert.qwen3_vl = torch.nn.Linear(2, 2)
model.head.expert = torch.nn.Linear(2, 2)
policy = SimpleNamespace(named_parameters=model.named_parameters, profile=SimpleNamespace(lr_microstep=True),
                         lr=2e-4, weight_decay=1e-7, vlm_lr_mult=0.5, warmup_steps=125,
                         warmup_start=1 / 3, min_lr_ratio=1e-3)
optim = UniDriveVLAPolicy.get_optimizers(policy, total_steps=2500, micro_steps=10000, grad_accum=4)
optimizer, factor = optim["optimizer"], optim["lr_scheduler"]["scheduler"].lr_lambdas[0]
cosine = lambda step, total, min_lr=1e-3: min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * step / total))  # noqa: E731
warm = lambda step, start, total: start + (1 - start) * step / total  # noqa: E731
check(optim["lr_scheduler"]["interval"] == "micro", "native main profile schedules on each data micro-batch")
check(abs(factor(0) - cosine(0, 10000) * warm(0, 1 / 3, 500)) < 1e-12,
      f"nuScenes starts at one-third LR ({factor(0):.6f})")
check(abs(factor(3) - cosine(3, 10000) * warm(3, 1 / 3, 500)) < 1e-12
      and abs(factor(3) - 0.337333) < 1e-6,
      f"nuScenes first accumulated update uses main micro t=3 LR ({factor(3):.6f})")
check(abs(factor(499) - cosine(499, 10000) * warm(499, 1 / 3, 500)) < 1e-12
      and abs(factor(500) - cosine(500, 10000)) < 1e-12,
      "nuScenes warmup boundary is 500 mmcv micro-iterations")
check(abs(factor(5000) - cosine(5000, 10000)) < 1e-12
      and abs(factor(10000) - 1e-3) < 1e-12,
      "nuScenes cosine midpoint and final min_lr match main")

b2d_policy = SimpleNamespace(named_parameters=model.named_parameters, profile=SimpleNamespace(lr_microstep=True),
                             lr=2e-4, weight_decay=0.01, vlm_lr_mult=0.5, warmup_steps=250,
                             warmup_start=0.01, min_lr_ratio=0.05)
b2d_optim = UniDriveVLAPolicy.get_optimizers(b2d_policy, total_steps=2500, micro_steps=10000, grad_accum=4)
b2d_factor = b2d_optim["lr_scheduler"]["scheduler"].lr_lambdas[0]
check(abs(b2d_factor(3) - cosine(3, 10000, 0.05) * warm(3, 0.01, 1000)) < 1e-12
      and abs(b2d_factor(3) - 0.01297) < 1e-6,
      f"Bench2Drive first accumulated update matches main warmup ({b2d_factor(3):.6f})")

legacy_policy = SimpleNamespace(named_parameters=model.named_parameters, profile=SimpleNamespace(lr_microstep=False),
                                lr=2e-4, weight_decay=1e-7, vlm_lr_mult=0.5, warmup_steps=125,
                                warmup_start=1 / 3, min_lr_ratio=1e-3)
legacy_optim = UniDriveVLAPolicy.get_optimizers(legacy_policy, total_steps=2500, micro_steps=10000, grad_accum=4)
legacy_factor = legacy_optim["lr_scheduler"]["scheduler"].lr_lambdas[0]
check(legacy_optim["lr_scheduler"]["interval"] == "step"
      and abs(legacy_factor(0) - 1 / 3) < 1e-12
      and abs(legacy_factor(125) - cosine(125, 2500)) < 1e-12,
      "legacy and ego profiles retain optimizer-step schedule timing")

short_policy = SimpleNamespace(named_parameters=model.named_parameters, profile=SimpleNamespace(lr_microstep=True),
                               lr=2e-4, weight_decay=1e-7, vlm_lr_mult=0.5, warmup_steps=1,
                               warmup_start=0.1, min_lr_ratio=0.05)
short = UniDriveVLAPolicy.get_optimizers(short_policy, total_steps=2, micro_steps=8, grad_accum=4)
short_optimizer = short["optimizer"]
short_scheduler = short["lr_scheduler"]["scheduler"]
short_factor = short_scheduler.lr_lambdas[0]
update_lrs = []
short_optimizer.zero_grad(set_to_none=True)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`",
                            category=UserWarning)
    for micro in range(8):
        expected_lrs = [group["initial_lr"] * short_factor(micro) for group in short_optimizer.param_groups]
        check(max(abs(group["lr"] - expected) for group, expected in zip(short_optimizer.param_groups, expected_lrs)) < 1e-12,
              f"production optimizer uses micro-step {micro} LR before its batch")
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        if (micro + 1) % 4 == 0:
            update_lrs.append([group["lr"] for group in short_optimizer.param_groups])
            short_optimizer.step()
            short_optimizer.zero_grad(set_to_none=True)
        short_scheduler.step()
expected_updates = [[group["initial_lr"] * short_factor(t) for group in short_optimizer.param_groups]
                    for t in (3, 7)]
check(all(max(abs(actual - expected) for actual, expected in zip(actual_group, expected_group)) < 1e-12
          for actual_group, expected_group in zip(update_lrs, expected_updates)),
      "production optimizer updates at the final micro-step LR of each 4-batch accumulation window")
try:
    UniDriveVLAPolicy.get_optimizers(short_policy, total_steps=2, micro_steps=0, grad_accum=4)
except ValueError as exc:
    check("micro_steps" in str(exc), "native schedule rejects missing Trainer micro-step budget")
else:
    check(False, "native schedule rejects missing Trainer micro-step budget")

lrs = sorted(group["initial_lr"] for group in optimizer.param_groups)
check(lrs == [1e-4, 2e-4], f"group lrs {lrs}: the VLM's at lr * vlm_lr_mult")

print("OK" if not failures else f"{len(failures)} FAILED")
sys.exit(1 if failures else 0)
