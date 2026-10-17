"""Dataset registry. A loader returns a torch Dataset whose items are (features, targets, token) in the canonical
sample of unidrivevla.data.sample; new sources register with `@register(name)`, from their own module listed in the
`plugins` config if they live outside this package, and the trainer does not change.

The model's `worker_transform(image_augment=None, train=False)` turns a canonical sample into the model's tensors in the dataloader
workers, so model-specific preprocessing (image decoding included) stays off the training process and out of the
datasets. The training split gets the `augment` config: ego-state dropout before that transform, photometric image
augmentation inside it.

Batching is generic: equal shapes stack, variable-length sequences pad with zeros, anything that is not a tensor becomes
a list, and keys only some samples of a batch have are dropped. Host-side metadata must stay Python objects: every
batch tensor goes to the GPU before forward, and reading a GPU tensor from Python stalls the device.
"""

import importlib
import itertools
import logging
import math
from typing import Any, Callable, Dict

import torch
import torch.nn.utils.rnn as rnn_utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

logger = logging.getLogger(__name__)

POSE_INTERVAL_S = 0.5
LoaderFn = Callable[[Any, Any, str], Any]
LOADERS: Dict[str, LoaderFn] = {}


def register(name: str) -> Callable[[LoaderFn], LoaderFn]:
    def deco(fn: LoaderFn) -> LoaderFn:
        LOADERS[name] = fn
        return fn
    return deco


def import_plugins(cfg: Any) -> None:
    """Imports the modules listed in `plugins`, which register more loaders or evaluators."""
    for module in OmegaConf.select(cfg, "plugins") or []:
        importlib.import_module(str(module))


def horizon(model: Any) -> int:
    """Number of 0.5 s target poses the model predicts (its trajectory_sampling)."""
    sampling = getattr(model, "trajectory_sampling", None)
    if sampling is None:
        raise AttributeError(f"{type(model).__name__} must expose trajectory_sampling (num_poses, interval_length)")
    if abs(float(sampling.interval_length) - POSE_INTERVAL_S) > 1e-6:
        raise ValueError(f"data sources provide poses every {POSE_INTERVAL_S} s, the model samples every "
                         f"{sampling.interval_length} s")
    return int(sampling.num_poses)


def loader_name(cfg: Any) -> str:
    name = OmegaConf.select(cfg, "data_loader") if isinstance(cfg, DictConfig) else getattr(cfg, "data_loader", None)
    return str(name).lower() if name else "navsim"


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform: Callable[[dict, dict], tuple]):
        self.dataset = dataset
        self.transform = transform

    @property
    def sample_weights(self):
        return getattr(self.dataset, "sample_weights", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        visit_id = None
        if isinstance(index, tuple):
            if len(index) == 2:
                index, augment = index  # original SequenceBatchSampler item
            elif len(index) == 3:
                index, augment, visit_id = index
            else:
                raise ValueError(f"unexpected sampled index {index!r}")
        else:
            index, augment = index, None
        features, targets, *rest = self.dataset[index]
        if augment is not None or visit_id is not None:
            features = dict(features)
            if augment is not None:
                features["augment"] = augment
            if visit_id is not None:
                stream_id = features.get("stream_id")
                if stream_id is None:
                    raise ValueError("mixture visit IDs require canonical features['stream_id']")
                features["stream_id"] = f"{stream_id}::visit={visit_id}"
        return (*self.transform(features, targets), *rest)


def build_split(cfg: Any, agent: Any, split: str):
    name = loader_name(cfg)
    fn = LOADERS.get(name)
    if fn is None:
        raise KeyError(
            f"Unknown data loader {name!r}. Registered: {sorted(LOADERS)}. "
            "Register a loader for a new dataset; the trainer does not change."
        )
    augment = OmegaConf.select(cfg, "augment") if split == "train" else None
    return with_worker_transform(fn(cfg, agent, split), agent, augment, train=split == "train")


def with_worker_transform(dataset, agent, augment=None, train: bool = False):
    factory = getattr(dataset, "worker_transform", None)
    transform = factory(agent, train=train) if callable(factory) else build_worker_transform(agent, augment, train)
    return TransformedDataset(dataset, transform)


def build_worker_transform(agent, augment=None, train: bool = False):
    """Build the existing canonical-to-model transform, reusable by each domain's data recipe."""
    from unidrivevla.data.augment import FrameGeometry, ego_dropout, geometry, image_augment

    images = image_augment(augment)
    make = getattr(agent, "worker_transform", None)
    transform = None if make is None else make(image_augment=images, train=train)
    if images is not None and transform is None:
        logger.warning("augment: the model reads no images in the dataloader, image augmentation is skipped")
    draw = geometry(augment)
    steps = [step for step in (ego_dropout(augment), draw and FrameGeometry(draw), transform) if step]
    if not steps:
        return lambda features, targets: (features, targets)

    def run(features, targets):
        for step in steps:
            features, targets = step(features, targets)
        return features, targets

    return run


def dataset_of(dataset):
    """The source dataset under the worker transform."""
    return dataset.dataset if isinstance(dataset, TransformedDataset) else dataset


def collate_field(key: str, values):
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        return list(values)
    first = values[0]
    if not torch.is_tensor(first):
        return list(values)
    if all(v.shape == first.shape for v in values):
        return torch.stack(values, dim=0)
    if first.dim() >= 1 and all(v.dim() == first.dim() and v.shape[1:] == first.shape[1:] for v in values):
        return rnn_utils.pad_sequence(list(values), batch_first=True, padding_value=0)
    raise ValueError(f"cannot batch {key!r} with shapes {[tuple(v.shape) for v in values]}")


def collate_dicts(dicts, require_same_keys: bool = False):
    keys = list(dicts[0])
    if require_same_keys:
        expected = set(keys)
        for index, item in enumerate(dicts[1:], 1):
            current = set(item)
            if current != expected:
                raise ValueError(f"batch item {index} has inconsistent fields: missing={sorted(expected - current)}, "
                                 f"extra={sorted(current - expected)}")
    return {key: collate_field(key, [d[key] for d in dicts]) for key in keys}


def collate(batch):
    columns = list(zip(*batch))
    features_list, targets_list = columns[0], columns[1]
    tokens = columns[2] if len(columns) > 2 else [None] * len(batch)
    mixed = bool(tokens) and all(token is not None and str(token).startswith("source=") for token in tokens)
    return (collate_dicts(features_list, mixed), collate_dicts(targets_list, mixed), list(tokens))


class WeightedDistributedSampler(Sampler):
    """Draws len(weights) indices with replacement, proportional to the weights, identically on every rank from
    (seed, epoch), and yields this rank's share."""

    def __init__(self, weights, rank: int, world: int, seed: int = 0, visit_ids: bool = False):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rank, self.world, self.seed, self.visit_ids = rank, world, seed, visit_ids
        self.per_rank = len(self.weights) // world
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        drawn = torch.multinomial(self.weights, self.per_rank * self.world, replacement=True, generator=generator)
        if not self.visit_ids:
            return iter(drawn[self.rank::self.world].tolist())
        epoch = self.epoch
        self.epoch += 1
        return iter([(int(index), None, f"epoch={epoch}:slot={slot}:visit=0")
                     for slot, index in enumerate(drawn.tolist()) if slot % self.world == self.rank])

    def __len__(self) -> int:
        return self.per_rank


class SequenceBatchSampler(Sampler):
    """Each slot walks consecutive frames. The unweighted path keeps the original GroupInBatchSampler order; the
    mixture path samples source sequences by frame-weight and keeps each sequence's split parts in one slot."""

    def __init__(self, sequences, batch_size: int, rank: int, world: int, split: int = 2, seed: int = 0,
                 draw=None, group_sources=None, source_weights=None, exact_resume: bool = False):
        self.groups, self.group_parents, self.parent_groups = [], [], []
        for parent, sequence in enumerate(sequences):
            parts = self.cut(list(sequence), split)
            self.parent_groups.append(list(range(len(self.groups), len(self.groups) + len(parts))))
            self.groups.extend(parts)
            self.group_parents.extend([parent] * len(parts))
        self.global_batch = batch_size * world
        self.world = world
        if len(self.groups) < self.global_batch:
            raise ValueError(f"{len(self.groups)} sequence parts cannot give each of {self.global_batch} slots its own")
        self.batches = sum(map(len, self.groups)) // self.global_batch
        self.rank, self.batch_size, self.seed, self.epoch = rank, batch_size, seed, 0
        self.exact_resume = exact_resume
        self.group_sources, self.source_sampling_weights, self.source_parents = group_sources, None, None
        if group_sources is not None:
            if len(group_sources) != len(self.parent_groups) or not source_weights:
                raise ValueError("weighted sequence sampling needs one source id per sequence and source weights")
            if len(self.parent_groups) < self.global_batch:
                raise ValueError(f"{len(self.parent_groups)} streams cannot give each of {self.global_batch} slots its own")
            source_parents = {source: [] for source in range(len(source_weights))}
            for parent, source in enumerate(group_sources):
                if source not in source_parents:
                    raise ValueError(f"sequence {parent} references unknown source {source}")
                source_parents[source].append(parent)
            self.source_parents = [source_parents[source] for source in range(len(source_parents))]
            self.source_parents_count = [len(parents) for parents in self.source_parents]
            source_mean_frames = [1.0] * len(source_parents)
            for source, parents in source_parents.items():
                weight = float(source_weights[source])
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"source {source} needs a finite positive weight, got {weight}")
                if parents:
                    mean_frames = sum(sum(len(self.groups[i]) for i in self.parent_groups[p]) for p in parents) / len(parents)
                    source_mean_frames[source] = mean_frames
            self.source_sampling_weights = [float(weight) / source_mean_frames[source]
                                            for source, weight in enumerate(source_weights)]
        else:
            self.slots = [self.order(seed, rank * batch_size + k) for k in range(batch_size)]
        self.buffers = [[] for _ in range(batch_size)]
        self.augments = [None] * batch_size
        self.draw = draw

    @staticmethod
    def cut(seq, parts: int):
        size = math.ceil(len(seq) / parts)
        return [seq[i:i + size] for i in range(0, len(seq), size)]

    def order(self, seed: int, slot: int):
        def stream():
            generator = torch.Generator().manual_seed(seed)
            while True:
                yield from torch.randperm(len(self.groups), generator=generator).tolist()
        return itertools.islice(stream(), slot, None, self.global_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.exact_resume and self.source_sampling_weights is None:
            self.slots = [self.order(self.seed + epoch, self.rank * self.batch_size + k)
                          for k in range(self.batch_size)]
            self.buffers = [[] for _ in range(self.batch_size)]
            self.augments = [None] * self.batch_size

    def __iter__(self):
        if self.source_sampling_weights is None:
            for _ in range(self.batches):
                batch = []
                for k, buffer in enumerate(self.buffers):
                    if not buffer:
                        buffer.extend(self.groups[next(self.slots[k])])
                        self.augments[k] = self.draw() if self.draw is not None else None
                    batch.append((buffer.pop(0), self.augments[k]))
                yield batch
            return

        # Simulate all ranks from one seed so source/group choices agree across processes. One source sequence stays
        # in one slot at a time, and its split parts run in order.
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        slots = self.global_batch
        buffers, active_parents, augments = [[] for _ in range(slots)], [None] * slots, [None] * slots
        visits, visit_counts = [None] * slots, [0] * slots
        next_part = [0] * len(self.parent_groups)
        available = [list(parents) for parents in self.source_parents]

        def choose_parent():
            eligible_sources = [source for source, parents in enumerate(available) if parents]
            source_weights = torch.tensor(
                [self.source_sampling_weights[source] * len(available[source]) / self.source_parents_count[source]
                 for source in eligible_sources], dtype=torch.double)
            source = eligible_sources[int(torch.multinomial(source_weights, 1, generator=generator))]
            choices = available[source]
            index = int(torch.randint(len(choices), (), generator=generator))
            parent = choices[index]
            choices[index] = choices[-1]
            choices.pop()
            return parent

        for _ in range(self.batches):
            batch = []
            for slot, buffer in enumerate(buffers):
                if not buffer:
                    previous = active_parents[slot]
                    if previous is not None:
                        next_part[previous] += 1
                        if next_part[previous] == len(self.parent_groups[previous]):
                            next_part[previous] = 0
                            source = self.group_sources[previous]
                            available[source].append(previous)
                            parent = choose_parent()
                            active_parents[slot] = parent
                            visit_counts[slot] += 1
                            visits[slot] = f"epoch={self.epoch - 1}:slot={slot}:visit={visit_counts[slot]}"
                        else:
                            parent = previous
                    else:
                        parent = choose_parent()
                        active_parents[slot] = parent
                        visits[slot] = f"epoch={self.epoch - 1}:slot={slot}:visit={visit_counts[slot]}"
                    group = self.parent_groups[parent][next_part[parent]]
                    buffer.extend(self.groups[group])
                    augments[slot] = self.draw() if self.draw is not None else None
                batch.append((buffer.pop(0), augments[slot], visits[slot]))
            start = self.rank * self.batch_size
            yield batch[start:start + self.batch_size]

    def __len__(self) -> int:
        return self.batches


def dataloader_kwargs(params) -> dict:
    kwargs = OmegaConf.to_container(params, resolve=True) if isinstance(params, DictConfig) else dict(params)
    kwargs.pop("shuffle", None)
    if int(kwargs.get("num_workers") or 0) > 0:
        kwargs.setdefault("persistent_workers", True)
    else:
        kwargs.pop("prefetch_factor", None)
        kwargs.pop("persistent_workers", None)
    return kwargs


def evaluation_params(cfg) -> dict:
    """Keep official per-frame evaluation independent of the training micro-batch size."""
    values = OmegaConf.to_container(cfg.dataloader.params, resolve=True)
    values["batch_size"] = int(OmegaConf.select(cfg, "dataloader.eval_batch_size", default=1))
    if values["batch_size"] < 1:
        raise ValueError("dataloader.eval_batch_size must be positive")
    return values


def make_loader(dataset, params, train: bool, rank: int, world: int, sequence_split: int = 0, seed: int = 0,
                draw_augment=None, exact_resume: bool = False) -> DataLoader:
    """Training on a source with sequences() and sequence_split > 0 feeds each batch slot consecutive frames
    (SequenceBatchSampler, draw_augment once per sequence); otherwise samples are shuffled."""
    source_dataset = dataset_of(dataset)
    factory = getattr(source_dataset, "make_loader", None)
    if callable(factory):
        return factory(dataset, params, train, rank, world, sequence_split=sequence_split, seed=seed,
                       draw_augment=draw_augment, exact_resume=exact_resume)
    kwargs = dataloader_kwargs(params)
    if exact_resume:
        if int(kwargs.get("num_workers") or 0) != 0:
            raise ValueError("train.exact_resume requires dataloader.params.num_workers=0 so prefetched worker "
                             "batches cannot be lost or replayed")
        kwargs["generator"] = torch.Generator().manual_seed(seed + rank)
    # Fixed batch shapes keep compiled graphs and cuDNN plans valid for the whole epoch.
    drop_last = bool(kwargs.pop("drop_last", train))
    weights = getattr(dataset, "sample_weights", None)
    sequences = getattr(dataset_of(dataset), "sequences", None)
    group_sources = getattr(source_dataset, "sequence_sources", None)
    if train and sequence_split > 0 and sequences is not None and (weights is None or group_sources is not None):
        sampler = SequenceBatchSampler(
            sequences(), int(kwargs.pop("batch_size", 1)), rank, world, sequence_split, seed, draw_augment,
            group_sources=group_sources,
            source_weights=getattr(source_dataset, "source_weights", None), exact_resume=exact_resume)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, **kwargs)
    if not train and sequences is not None:  # frames in time order, a contiguous run of sequences per rank
        order = [i for seq in sequences() for i in seq]
        per = math.ceil(len(order) / world)
        return DataLoader(dataset, sampler=order[rank * per:(rank + 1) * per], collate_fn=collate, drop_last=False,
                          **kwargs)
    if train and weights is not None:
        kwargs["sampler"] = WeightedDistributedSampler(weights, rank, world, seed, visit_ids=group_sources is not None)
    elif world > 1 or (train and exact_resume):
        kwargs["sampler"] = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=train,
                                                drop_last=drop_last, seed=seed)
    else:
        kwargs["shuffle"] = train
    return DataLoader(dataset, collate_fn=collate, drop_last=drop_last, **kwargs)
