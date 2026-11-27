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
from unidrivevla.data.registry import SequenceBatchSampler, make_loader, sequence_shard  # noqa: E402

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

# 1b. Evaluation must keep whole scenes on one rank, emit every frame once,
# attach a stream ID, and permit a rank with no assigned scenes.
class EvalSequences:
    def __init__(self):
        self.groups = [[0, 1, 2], [3, 4], [5]]
        self.scene_for_index = {index: scene for scene, seq in enumerate(self.groups) for index in seq}

    def __len__(self):
        return 6

    def sequences(self):
        return self.groups

    def __getitem__(self, index):
        pose = np.eye(4)
        pose[0, 3] = index
        return {"img_metas": {"T_global": pose, "T_global_inv": np.linalg.inv(pose)}}, {}, str(index)


eval_source = EvalSequences()
eval_params = {"batch_size": 1, "num_workers": 0}
scene_loaders = [make_loader(eval_source, eval_params, False, rank, 4, sequence_eval=True)
                 for rank in range(4)]
scene_rows = {}
for rank, loader in enumerate(scene_loaders):
    for features, _, tokens in loader:
        token = int(tokens[0])
        scene = eval_source.scene_for_index[token]
        metadata = features["img_metas"][0]
        assert metadata["stream_id"] == f"scene=sequence={scene}"
        scene_rows[token] = metadata["stream_id"]
check(len(scene_rows) == len(eval_source), "scene evaluation covers every frame exactly once")
scene_ranks = {
    scene: {rank for rank, loader in enumerate(scene_loaders)
            if any(eval_source.scene_for_index[int(tokens[0])] == scene
                   for _, _, tokens in loader)}
    for scene in range(3)
}
check(all(len(ranks) == 1 for ranks in scene_ranks.values()),
      "each complete scene is assigned to one rank")
check(len(scene_loaders[3]) == 0, "a rank with no scene assignments has an empty valid loader")
padded_loaders = [make_loader(eval_source, eval_params, False, rank, 4,
                              sequence_eval=True, pad_eval=True) for rank in range(4)]
padded_lengths = [len(loader) for loader in padded_loaders]
check(len(set(padded_lengths)) == 1 and padded_lengths[0] == 3,
      "FSDP scene eval pads each rank to the same number of forwards")
padded_tokens = []
for loader in padded_loaders:
    padded_tokens.extend(tokens[0] for _, _, tokens in loader)
check(sorted(token for token in padded_tokens if token is not None) == [str(i) for i in range(6)]
      and sum(token is None for token in padded_tokens) == 6,
      "FSDP padding tokens are marked out and all real frame tokens remain unique")
from unidrivevla.models.models.instance_bank import InstanceBank

bank = InstanceBank(2, 4, np.zeros((2, 11), np.float32))
bank.cached_anchor = bank.anchor.detach().unsqueeze(0)
bank.cached_feature = torch.zeros((1, 2, 4))
bank.metas = {"timestamp": torch.tensor([0.0], dtype=torch.float64),
              "img_metas": [{"stream_id": scene_rows[0]}]}
next_scene = bank.get(1, {"timestamp": torch.tensor([0.05], dtype=torch.float64),
                          "img_metas": [{"stream_id": scene_rows[3]}]})
check(next_scene[2] is None and bank.cached_anchor is None,
      "the instance bank resets its temporal cache at a scene boundary")
try:
    sequence_shard([[0, 1], [1, 2]], 0, 2, 3)
except ValueError:
    check(True, "scene sharding rejects overlapping frame ownership")
else:
    check(False, "scene sharding rejects overlapping frame ownership")

# The non-opt-in path used by other datasets keeps its previous frame-slice behavior.
legacy_loader = make_loader(eval_source, eval_params, False, 0, 2)
legacy_features = next(iter(legacy_loader))[0]
check("stream_id" not in legacy_features["img_metas"][0],
      "non-nuScenes evaluation keeps its existing metadata behavior")

# Open-loop chooses scene sharding based on the dataset capability, so B2D gets
# it too while a frame-only source retains the old flat slice.
from unidrivevla.eval.openloop import OpenLoopEvaluator


class RaggedB2DSequences(EvalSequences):
    def __init__(self):
        self.groups = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
        self.scene_for_index = {index: scene for scene, seq in enumerate(self.groups) for index in seq}

    def __len__(self):
        return len(self.scene_for_index)


class FrameOnly:
    def __len__(self):
        return 10

    def __getitem__(self, index):
        pose = np.eye(4)
        pose[0, 3] = index
        return {"img_metas": {"T_global": pose, "T_global_inv": np.linalg.inv(pose)}}, {}, str(index)


def openloop_loaders(dataset, world, pad_eval=False):
    evaluator = object.__new__(OpenLoopEvaluator)
    evaluator.source = "b2d"
    return [evaluator.make_eval_loader(dataset, eval_params,
                                      SimpleNamespace(rank=rank, world_size=world), pad_eval=pad_eval)
            for rank in range(world)]


b2d_source = RaggedB2DSequences()
b2d_loaders = openloop_loaders(b2d_source, 3)
b2d_scene_rows = {}
b2d_scene_ranks = {scene: set() for scene in range(4)}
for rank, loader in enumerate(b2d_loaders):
    for features, _, tokens in loader:
        token = int(tokens[0])
        scene = b2d_source.scene_for_index[token]
        b2d_scene_ranks[scene].add(rank)
        b2d_scene_rows[token] = features["img_metas"][0]["stream_id"]
check(len(b2d_scene_rows) == 10 and all(len(ranks) == 1 for ranks in b2d_scene_ranks.values()),
      "B2D open-loop keeps each ragged scene on one rank and covers every frame once")
b2d_padded = openloop_loaders(b2d_source, 3, pad_eval=True)
check([len(loader) for loader in b2d_padded] == [5, 5, 5]
      and sum(token is None for loader in b2d_padded for _, _, tokens in loader for token in tokens) == 5,
      "B2D FSDP padding equalizes uneven scene shards without adding score rows")
flat_loaders = openloop_loaders(FrameOnly(), 3)
check([len(loader) for loader in flat_loaders] == [4, 4, 4]
      and all("stream_id" not in features["img_metas"][0]
              for loader in flat_loaders for features, _, _ in loader),
      "open-loop sources without sequences keep the legacy distributed-sampler path")

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
