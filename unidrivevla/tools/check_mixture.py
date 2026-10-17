"""CPU check for weighted mixed sequence sampling and canonical sample integrity."""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.data.mixture import mixture_loader  # noqa: E402
from unidrivevla.data.registry import LOADERS, TransformedDataset, collate, make_loader  # noqa: E402


class TinySource(Dataset):
    def __init__(self, sequences=None, size=0, source="independent"):
        self._sequences = sequences
        self.size = sum(map(len, sequences)) if sequences is not None else size
        self.source = source

    def __len__(self):
        return self.size

    def sequences(self):
        return self._sequences

    def __getitem__(self, index):
        seq_id = index // 8
        token = f"{self.source}-{seq_id}-{index}"
        count = index % 4 + 1
        return ({"frame": torch.tensor(index), "timestamp": float(index)},
                {"gt_boxes": torch.full((count, 7), float(index)), "gt_labels": torch.arange(count),
                 "gt_mask": torch.ones(count)}, token)


class TinyIndependent(Dataset):
    def __init__(self, size, source):
        self.size, self.source = size, source

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        count = index % 4 + 1
        return ({"frame": torch.tensor(index), "timestamp": float(index)},
                {"gt_boxes": torch.full((count, 7), float(index)), "gt_labels": torch.arange(count),
                 "gt_mask": torch.ones(count)}, f"{self.source}-{index}")


def run():
    sequence_source = TinySource([list(range(i * 8, (i + 1) * 8)) for i in range(12)], source="sequence")
    independent_source = TinyIndependent(size=512, source="independent")
    previous = {name: LOADERS.get(name) for name in ("_check_mix_sequence", "_check_mix_independent", "_check_mix_empty")}
    LOADERS["_check_mix_sequence"] = lambda cfg, agent, split: sequence_source
    LOADERS["_check_mix_independent"] = lambda cfg, agent, split: independent_source
    LOADERS["_check_mix_empty"] = lambda cfg, agent, split: TinyIndependent(size=0, source="empty")
    cfg = OmegaConf.create({"data_loader": "mixture", "profile": {"name": "shared", "cameras": ["front"]},
                            "mixture": [{"loader": "_check_mix_sequence", "weight": 1.0},
                                        {"loader": "_check_mix_independent", "weight": 4.0}]})
    dataset = mixture_loader(cfg, None, "train")
    assert len(dataset) == 608
    first = dataset[0]
    assert first[0]["stream_id"] == "source=0:_check_mix_sequence:sequence=0"
    assert first[2] == "source=0:_check_mix_sequence::sequence-0-0"
    assert dataset[96][0]["stream_id"].startswith("source=1:_check_mix_independent:item=")
    assert dataset[96][2].startswith("source=1:_check_mix_independent::")
    assert dataset[0][0]["timestamp"] == 0.0 and dataset[1][0]["timestamp"] == 1.0
    assert dataset[96][0]["timestamp"] == 0.0  # source timestamps stay physical; no synthetic offsets

    ranks = []
    for rank in (0, 1):
        wrapped = TransformedDataset(dataset, lambda features, targets: (features, targets))
        ranks.append(make_loader(wrapped, {"batch_size": 2, "num_workers": 0}, True, rank, 2,
                                 sequence_split=2, seed=23))
    replay = []
    for rank in (0, 1):
        wrapped = TransformedDataset(dataset, lambda features, targets: (features, targets))
        replay.append(make_loader(wrapped, {"batch_size": 2, "num_workers": 0}, True, rank, 2,
                                  sequence_split=2, seed=23))
    replay_batches = [list(loader) for loader in replay]
    epochs = [[list(loader) for loader in ranks]] + [[list(loader) for loader in ranks] for _ in range(7)]
    rank_batches = epochs[0]
    assert len(rank_batches[0]) == len(rank_batches[1])
    for rank in (0, 1):
        assert [b[2] for b in rank_batches[rank]] == [b[2] for b in replay_batches[rank]]
        assert [b[0]["stream_id"] for b in rank_batches[rank]] == [b[0]["stream_id"] for b in replay_batches[rank]]
    visits_by_epoch = [{stream for batches in run for batch in batches for stream in batch[0]["stream_id"]}
                       for run in epochs]
    assert all(visits_by_epoch[i].isdisjoint(visits_by_epoch[i + 1]) for i in range(len(epochs) - 1))

    source_counts = [0, 0]
    last = {}
    for epoch, run in enumerate(epochs):
        for step, (batch0, batch1) in enumerate(zip(*run)):
            stream_ids = batch0[0]["stream_id"] + batch1[0]["stream_id"]
            assert len(stream_ids) == len(set(stream_ids)), f"duplicate active stream at epoch {epoch}, batch {step}"
            for rank, batch in enumerate((batch0, batch1)):
                features, targets, tokens = batch
                for slot, (stream_id, frame, token) in enumerate(zip(features["stream_id"], features["frame"].tolist(), tokens)):
                    source_id = int(token.split("=", 1)[1].split(":", 1)[0])
                    source_counts[source_id] += 1
                    prior = last.get((rank, slot))
                    if prior is not None and prior[0] == stream_id:
                        assert frame == prior[1] + 1, f"nonconsecutive frames in {stream_id}: {prior[1]} -> {frame}"
                    last[(rank, slot)] = stream_id, frame
                assert targets["gt_boxes"].shape[:2] == targets["gt_mask"].shape
                assert targets["gt_labels"].shape == targets["gt_mask"].shape
    independent_share = source_counts[1] / sum(source_counts)
    assert 0.75 < independent_share < 0.85, f"4:1 weights produced source shares {source_counts}"

    ungrouped = [make_loader(TransformedDataset(dataset, lambda features, targets: (features, targets)),
                             {"batch_size": 2, "num_workers": 0}, True, rank, 2, seed=29)
                 for rank in (0, 1)]
    first_epoch = [[batch[0]["stream_id"] for batch in loader] for loader in ungrouped]
    second_epoch = [[batch[0]["stream_id"] for batch in loader] for loader in ungrouped]
    for rank in (0, 1):
        first_visits = {visit for batch in first_epoch[rank] for visit in batch}
        second_visits = {visit for batch in second_epoch[rank] for visit in batch}
        assert first_visits.isdisjoint(second_visits)

    padded = collate([
        ({"stream_id": "a", "frame": torch.tensor(0)},
         {"gt_boxes": torch.ones(1, 7), "gt_labels": torch.tensor([2]), "gt_mask": torch.ones(1)}, "source=0:a::0"),
        ({"stream_id": "b", "frame": torch.tensor(1)},
         {"gt_boxes": torch.ones(3, 7), "gt_labels": torch.tensor([1, 2, 3]), "gt_mask": torch.ones(3)}, "source=1:b::1"),
    ])
    assert padded[1]["gt_boxes"].shape == (2, 3, 7)
    assert padded[1]["gt_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert padded[1]["gt_labels"].shape == padded[1]["gt_mask"].shape
    try:
        collate([({"x": torch.tensor(1)}, {"label": torch.tensor(1)}, "source=0:a::0"),
                 ({"x": torch.tensor(2)}, {"label": torch.tensor(2), "mask": torch.tensor(1)}, "source=1:b::1")])
    except ValueError:
        pass
    else:
        raise AssertionError("mixed batch silently discarded a required target field")

    for bad in (0, -1, float("nan"), float("inf")):
        bad_cfg = OmegaConf.create({"profile": {"name": "shared"},
                                    "mixture": [{"loader": "_check_mix_sequence", "weight": bad}]})
        try:
            mixture_loader(bad_cfg, None, "train")
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid source weight {bad}")
    empty_cfg = OmegaConf.create({"profile": {"name": "shared"},
                                  "mixture": [{"loader": "_check_mix_empty", "weight": 1.0}]})
    try:
        mixture_loader(empty_cfg, None, "train")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an empty source")
    profile_cfg = OmegaConf.create({"profile": {"name": "shared"},
                                    "mixture": [{"loader": "_check_mix_sequence", "weight": 1.0,
                                                 "overrides": {"profile": {"name": "other"}}}]})
    try:
        mixture_loader(profile_cfg, None, "train")
    except ValueError as error:
        assert "mixed profiles" in str(error)
    else:
        raise AssertionError("accepted an incompatible source profile")

    original_b2d = LOADERS.get("b2d")
    calls = []
    LOADERS["b2d"] = lambda cfg, agent, split: calls.append((cfg.profile.name, split)) or sequence_source
    try:
        builtin_cfg = OmegaConf.create({"profile": {"name": "nuscenes"},
                                        "mixture": [{"loader": "b2d", "weight": 1.0}]})
        try:
            mixture_loader(builtin_cfg, None, "train")
        except ValueError as error:
            assert "validated custom shared profile" in str(error)
        else:
            raise AssertionError("accepted b2d under the built-in nuscenes profile")
        assert not calls, "built-in profile mismatch must fail before loading source data"

        custom_cfg = OmegaConf.create({"profile": {"name": "shared_drive"},
                                       "mixture": [{"loader": "b2d", "weight": 1.0}]})
        assert len(mixture_loader(custom_cfg, None, "train")) == len(sequence_source)
        assert calls == [("shared_drive", "train")]
    finally:
        if original_b2d is None:
            LOADERS.pop("b2d", None)
        else:
            LOADERS["b2d"] = original_b2d

    print(f"OK: deterministic two-rank visits over {len(epochs)} epochs, 4:1 weights -> {source_counts}, padded masks kept")
    for name, loader in previous.items():
        if loader is None:
            LOADERS.pop(name, None)
        else:
            LOADERS[name] = loader


if __name__ == "__main__":
    run()
