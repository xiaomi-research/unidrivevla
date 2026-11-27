"""Fixed-seed comparison for the pinned PR10 nuScenes sequence/RNG path.

Reference: PR10 commit 5fce121f4bf04eaf7e109880d4f26eed62c0eed0,
`GroupInBatchSampler`, `NuScenes3DDataset.get_augmentation`, and
`projects/mmdet3d_plugin/datasets/builder.py::worker_init_fn`.
This CPU check uses a small synthetic sequence list; it does not read NuScenes assets.
"""

import itertools
import multiprocessing
import os
import pickle
import random
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unidrivevla.data.augment import geometry
from unidrivevla.data.registry import (
    SequenceBatchSampler,
    _pr10_worker_init_fn,
    collate,
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

    def __iter__(self):
        while True:
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
            yield batch

    def epoch(self):
        return list(itertools.islice(self, self.batches_per_epoch))


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


class _WorkerProbe(_FakeNuScenes):
    def __getitem__(self, sample):
        index, augment = sample
        worker = get_worker_info()
        draws = [worker.id if worker else -1, int(augment['flip']), round(augment['rotate'] * 1e7),
                 np.random.randint(2**30), random.randrange(2**30), torch.randint(2**30, ()).item()]
        return {"draws": torch.tensor(draws), "pid": torch.tensor(os.getpid())}, {}, str(index)


def _check_epoch_workers():
    sequences = [list(range(i, i + 6)) for i in range(0, 30, 6)]
    data = _WorkerProbe(sequences)
    seed, rank, world, batch = 17, 1, 2, 2
    draw = geometry(OmegaConf.create({"geometry": {"flip_p": 0.5, "rotate_deg": 5.4, "rng_mode": "pr10_numpy"}}))
    length = len(data) // (world * batch)  # Seven batches: the next epoch starts on a different worker.

    def read(batches, pids):
        rows = []
        for features, _, tokens in batches:
            pids.update(features['pid'].tolist())
            rows.extend(zip(tokens, features['draws'].tolist()))
        return rows

    def reset_rng():
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    for workers in (0, 2):
        params = {"num_workers": workers, "pin_memory": False, "persistent_workers": False}
        if workers:
            params['prefetch_factor'] = 3
        reset_rng()
        oracle = _OfficialGroupSampler(_pr10_split(sequences, 2), batch, rank, world, seed)
        ref = DataLoader(data, batch_sampler=oracle, collate_fn=collate, **params,
                         worker_init_fn=partial(_pr10_worker_init_fn, num_workers=workers, rank=rank, seed=seed))
        stream = iter(ref)
        expected = [read(itertools.islice(stream, length), set()) for _ in range(3)]
        del stream, ref

        reset_rng()
        loader = make_loader(data, {**params, 'batch_size': batch}, True, rank, world,
                             sequence_split=2, seed=seed, draw_augment=draw)
        pids = set()
        try:
            assert len(loader) == length
            for epoch, reference in enumerate(expected):
                loader.batch_sampler.set_epoch(epoch)
                actual = read(itertools.islice(loader, length + 1), pids)
                assert actual == reference, (workers, epoch, 'sample, worker or RNG stream differs from PR10')
            assert len(pids) == max(workers, 1), 'the worker pool was replaced between epochs'
            partial_epoch = iter(loader)
            next(partial_epoch)  # close with prefetch outstanding, as in a max_steps stop.
        finally:
            loader.close_epoch_stream()
        if workers:
            assert pids.isdisjoint(w.pid for w in multiprocessing.active_children()), 'worker left running after close'


def _check_camera_draw_order():
    from PIL import Image
    from unidrivevla.adapters.unidrivevla import Profile, camera_inputs
    from unidrivevla.data.augment import PhotoMetricDistortion
    from unidrivevla.data.nuscenes import CAMERAS, nuscenes_loader
    from unidrivevla.tools.check_native_sources import make_nuscenes

    profile = Profile.from_config(OmegaConf.load(Path(__file__).resolve().parents[1] / 'config/profile/nuscenes.yaml'))
    profile.image_hw, profile.train_resize = (6, 8), None
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        source, _ = make_nuscenes(tmp)
        source.infos[1]['cams'] = dict(reversed(list(source.infos[1]['cams'].items())))
        for split in ('train', 'val'):
            with (tmp / f'nuscenes_infos_{split}.pkl').open('wb') as stream:
                pickle.dump({'infos': source.infos}, stream)
        cfg = OmegaConf.create({'nuscenes': {'info_dir': directory, 'path_root': directory},
                                'augment': {'geometry': {'rng_mode': 'pr10_numpy'}}})
        ds = nuscenes_loader(cfg, None, 'train')
        photo = PhotoMetricDistortion()
        for index, info in enumerate(ds.infos):
            features = ds.features(index)
            np.random.seed(17)
            expected = {}
            for sensor, cam in info['cams'].items():
                with Image.open(tmp / cam['data_path']) as image:
                    expected[CAMERAS[sensor]] = np.asarray(photo(image.convert('RGB')))[:, :, ::-1].transpose(2, 0, 1)
            np.random.seed(17)
            actual = camera_inputs(features, profile, image_augment=photo, train=True)
            np.testing.assert_array_equal(actual['img'].numpy(), np.stack([expected[v] for v in profile.cameras]))
            unaugmented = camera_inputs(features, profile, train=True)
            for key in ('projection_mat', 'focal', 'image_wh'):
                assert torch.equal(actual[key], unaugmented[key]), key
        assert 'camera_augmentation_order' not in nuscenes_loader(cfg, None, 'val').features(0)
        cfg.augment.geometry.rng_mode = 'python'
        assert 'camera_augmentation_order' not in nuscenes_loader(cfg, None, 'train').features(0)


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
    assert len(list(itertools.islice(loader.batch_sampler, len(loader) + 1))) == len(loader)
    assert loader.batch_sampler.rng_mode == "pr10_numpy"
    assert loader.worker_init_fn.func is _pr10_worker_init_fn

    debug = make_loader(
        source,
        {"batch_size": 4, "num_workers": 0, "pin_memory": True},
        train=True, rank=1, world=2, sequence_split=2, seed=31415, draw_augment=draw,
    )
    assert debug.num_workers == 0 and debug.pin_memory is True and debug.worker_init_fn is None

    exact = make_loader(source, {"batch_size": 4, "num_workers": 0}, True, 1, 2, sequence_split=2,
                        seed=31415, draw_augment=draw, exact_resume=True)
    assert type(exact) is DataLoader and type(exact.batch_sampler) is SequenceBatchSampler

    legacy_draw = geometry(OmegaConf.create({"geometry": {"flip_p": 0.5, "rotate_deg": 5.4}}))
    legacy = make_loader(
        _FakeOtherDataset(sequences),
        {"batch_size": 4, "num_workers": 4, "pin_memory": True},
        train=True, rank=0, world=2, sequence_split=2, seed=31415, draw_augment=legacy_draw,
    )
    assert legacy.num_workers == 4 and legacy.pin_memory is True and legacy.persistent_workers is True
    assert type(legacy) is DataLoader
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
    _check_camera_draw_order()
    _check_epoch_workers()
    print("PASS: PR10 camera augmentation order and continuous sample/worker RNG across logical epochs")


if __name__ == "__main__":
    main()
