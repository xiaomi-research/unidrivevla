"""Fixed-seed comparison for the pinned PR10 nuScenes sequence/RNG path.

Reference: PR10 commit 5fce121f4bf04eaf7e109880d4f26eed62c0eed0,
`GroupInBatchSampler`, `NuScenes3DDataset.get_augmentation`, and
`projects/mmdet3d_plugin/datasets/builder.py::worker_init_fn`.
This CPU check uses a small synthetic sequence list; it does not read NuScenes assets.
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unidrivevla.data.augment import geometry
from unidrivevla.data.registry import (
    SequenceBatchSampler,
    _pr10_worker_init_fn,
    make_loader,
    validate_pr10_sequence_groups,
)


def _pr10_split(sequences, parts):
    """PR10 NuScenes3DDataset._set_sequence_group_flag's ceil-sized split."""
    result = []
    for sequence in sequences:
        size = (len(sequence) + parts - 1) // parts
        bounds = list(range(0, len(sequence), size)) + [len(sequence)]
        result.extend([list(sequence[left:right]) for left, right in zip(bounds, bounds[1:])])
    return result


def _pr10_augmentation():
    """The six NumPy draws in PR10 get_augmentation for fixed nuScenes Stage-2/3 values."""
    np.random.uniform(0.6, 0.6)       # resize_lim
    np.random.uniform(0.0, 0.0)       # bot_pct_lim
    np.random.uniform(0.0, 0.0)       # crop_w, since 1600 * 0.6 == final width 960
    flip = bool(np.random.choice([0, 1]))
    rotate = float(np.random.uniform(-5.4, 5.4))
    np.random.uniform(0.0, 0.0)       # rot3d_range
    return {"flip": flip, "rotate": rotate}


class _OfficialGroupSampler:
    """Small public-source oracle for GroupInBatchSampler's rank/slot and NumPy call order."""

    def __init__(self, groups, batch_size, rank, world, seed):
        self.groups = groups
        self.batch_size = batch_size
        self.rank = rank
        self.world = world
        self.global_batch = batch_size * world
        self.streams = [self._stream(rank * batch_size + slot) for slot in range(batch_size)]
        self.buffers = [[] for _ in range(batch_size)]
        self.augments = [None] * batch_size
        self.batches_per_epoch = sum(map(len, groups)) // self.global_batch
        self.seed = seed

    def _stream(self, global_slot):
        def permutations():
            generator = torch.Generator().manual_seed(self.seed)
            while True:
                yield from torch.randperm(len(self.groups), generator=generator).tolist()
        return itertools.islice(permutations(), global_slot, None, self.global_batch)

    def epoch(self):
        batches = []
        for _ in range(self.batches_per_epoch):
            batch = []
            for slot, buffer in enumerate(self.buffers):
                skip = np.random.uniform() < 0.0 and len(buffer) > 1
                if not buffer:
                    group = next(self.streams[slot])
                    buffer.extend(list(self.groups[group]))
                    if np.random.uniform() < 0.0:
                        buffer.reverse()
                    self.augments[slot] = _pr10_augmentation()
                if skip:
                    buffer.pop(0)
                batch.append((buffer.pop(0), self.augments[slot]))
            batches.append(batch)
        return batches


class _FakeNuScenes:
    pr10_sampler_rng_compatible = True

    def __init__(self, sequences):
        self._sequences = sequences

    def __len__(self):
        return sum(map(len, self._sequences))

    def sequences(self):
        return self._sequences

    def validate_pr10_sampler_groups(self):
        return self._sequences

    def __getitem__(self, index):
        return {"value": torch.tensor(index)}, {}, str(index)


class _FakeOtherDataset(_FakeNuScenes):
    pr10_sampler_rng_compatible = False

    validate_pr10_sampler_groups = None


def _canonical(batch):
    return [
        [(index, augment["flip"], augment["rotate"]) for index, augment in row]
        for row in batch
    ]


def _check_sampler_diff():
    sequences = [list(range(start, start + 6)) for start in range(0, 24, 6)]
    groups = _pr10_split(sequences, parts=2)
    draw = geometry(OmegaConf.create({"geometry": {
        "flip_p": 0.5, "rotate_deg": 5.4, "rng_mode": "pr10_numpy",
    }}))
    assert draw is not None and draw.rng_mode == "pr10_numpy"

    for rank in (0, 1):
        np.random.seed(31415)
        actual_sampler = SequenceBatchSampler(
            sequences, batch_size=4, rank=rank, world=2, split=2, seed=2718, draw=draw,
        )
        actual = []
        for _ in range(2):
            actual.extend(list(actual_sampler))
        actual_tail = float(np.random.uniform())

        np.random.seed(31415)
        reference_sampler = _OfficialGroupSampler(groups, batch_size=4, rank=rank, world=2, seed=2718)
        expected = reference_sampler.epoch() + reference_sampler.epoch()
        expected_tail = float(np.random.uniform())

        assert _canonical(actual) == _canonical(expected), f"PR10 batch/augmentation mismatch at rank {rank}"
        assert actual_tail == expected_tail, f"PR10 NumPy draw order mismatch after rank {rank}'s epochs"


def _check_worker_seed():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        _pr10_worker_init_fn(worker_id=3, num_workers=8, rank=2, seed=31415)
        actual_numpy = [float(np.random.random()) for _ in range(3)]
        actual_python = [random.random() for _ in range(3)]

        worker_seed = 8 * 2 + 3 + 31415
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        assert actual_numpy == [float(np.random.random()) for _ in range(3)]
        assert actual_python == [random.random() for _ in range(3)]
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _check_sequence_boundaries():
    infos = [
        {"scene_token": "scene-a", "sweeps": []},
        {"scene_token": "scene-a", "sweeps": [{"frame": 0}]},
        {"scene_token": "scene-a", "sweeps": [{"frame": 1}]},
        {"scene_token": "scene-b", "sweeps": []},
        {"scene_token": "scene-b", "sweeps": [{"frame": 0}]},
    ]
    scenes = [[0, 1, 2], [3, 4]]
    assert validate_pr10_sequence_groups(infos, scenes) == scenes
    try:
        validate_pr10_sequence_groups(infos, [[0, 1], [2], [3, 4]])
    except ValueError as exc:
        assert "scene_token groups differ" in str(exc)
    else:
        raise AssertionError("inconsistent PR10 sweeps/scene grouping was accepted")


def _check_loader_gate():
    sequences = [list(range(start, start + 6)) for start in range(0, 24, 6)]
    source = _FakeNuScenes(sequences)
    draw = geometry(OmegaConf.create({"geometry": {
        "flip_p": 0.5, "rotate_deg": 5.4, "rng_mode": "pr10_numpy",
    }}))
    loader = make_loader(
        source,
        {"batch_size": 4, "num_workers": 8, "pin_memory": False, "persistent_workers": False,
         "prefetch_factor": 2},
        train=True, rank=1, world=2, sequence_split=2, seed=31415, draw_augment=draw,
    )
    assert loader.num_workers == 8 and loader.persistent_workers is False and loader.pin_memory is False
    assert isinstance(loader.batch_sampler, SequenceBatchSampler)
    assert loader.batch_sampler.rng_mode == "pr10_numpy"
    assert loader.worker_init_fn.func is _pr10_worker_init_fn

    debug = make_loader(
        source,
        {"batch_size": 4, "num_workers": 0, "pin_memory": True},
        train=True, rank=1, world=2, sequence_split=2, seed=31415, draw_augment=draw,
    )
    assert debug.num_workers == 0 and debug.pin_memory is True and debug.worker_init_fn is None

    legacy_draw = geometry(OmegaConf.create({"geometry": {"flip_p": 0.5, "rotate_deg": 5.4}}))
    legacy = make_loader(
        _FakeOtherDataset(sequences),
        {"batch_size": 4, "num_workers": 4, "pin_memory": True},
        train=True, rank=0, world=2, sequence_split=2, seed=31415, draw_augment=legacy_draw,
    )
    assert legacy.num_workers == 4 and legacy.pin_memory is True and legacy.persistent_workers is True
    assert legacy.worker_init_fn is None

    try:
        make_loader(
            _FakeOtherDataset(sequences),
            {"batch_size": 4, "num_workers": 8},
            train=True, rank=0, world=2, sequence_split=2, seed=31415, draw_augment=draw,
        )
    except ValueError as exc:
        assert "dedicated nuScenes" in str(exc)
    else:
        raise AssertionError("PR10 RNG mode was accepted for a non-nuScenes source")


def main():
    _check_sampler_diff()
    _check_worker_seed()
    _check_sequence_boundaries()
    _check_loader_gate()
    print("PASS: PR10 GroupInBatchSampler order/NumPy augmentation draws and rank-worker seed contract")


if __name__ == "__main__":
    main()
