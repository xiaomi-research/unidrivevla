"""Source-aware data for whole-profile homogeneous mixed-domain batches."""

import math

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from unidrivevla.data.registry import LOADERS, collate, dataloader_kwargs, register


class MultiProfileDataset(ConcatDataset):
    """Concatenate canonical sources while tagging every sample with its profile."""

    def __init__(self, datasets, names, weights, augments=None):
        if len(datasets) < 2 or len(datasets) != len(names) or len(names) != len(weights):
            raise ValueError("multidomain data needs at least two equally sized datasets, names, and weights")
        for name, dataset, weight in zip(names, datasets, weights):
            if len(dataset) == 0:
                raise ValueError(f"multidomain source {name!r} is empty")
            if not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError(f"multidomain source {name!r} needs a finite positive weight")
        super().__init__(datasets)
        self.source_names = tuple(map(str, names))
        self.source_weights = tuple(map(float, weights))
        self.source_augments = tuple(augments or [None] * len(names))
        if len(self.source_augments) != len(names):
            raise ValueError("multidomain source augmentation count must match source count")
        self.source_offsets = (0, *self.cumulative_sizes[:-1])
        self.source_sizes = tuple(map(len, datasets))

    def source_of(self, index):
        for source, end in enumerate(self.cumulative_sizes):
            if index < end:
                return source
        raise IndexError(index)

    def __getitem__(self, index):
        augment = visit_id = None
        if isinstance(index, tuple):
            if len(index) == 2:
                index, augment = index
            elif len(index) == 3:
                index, augment, visit_id = index
            else:
                raise ValueError(f"unexpected sampled multidomain index {index!r}")
        sample = super().__getitem__(index)
        if not isinstance(sample, (tuple, list)) or len(sample) < 3:
            raise ValueError("multidomain sources must return canonical (features, targets, token) samples")
        features, targets, token, *rest = sample
        if not isinstance(features, dict):
            raise TypeError("canonical sample features must be a dictionary")
        source = self.source_of(index)
        name = self.source_names[source]
        if augment is not None or visit_id is not None:
            features = dict(features)
            if augment is not None:
                features["augment"] = augment
            if visit_id is not None:
                stream_id = features.get("stream_id")
                if stream_id is None:
                    raise ValueError("multidomain visit IDs require canonical features['stream_id']")
                features["stream_id"] = f"{stream_id}::visit={visit_id}"
        return {**features, "profile": name}, targets, f"source={name}::{token}", *rest

    def worker_transform(self, agent, train=False):
        from unidrivevla.data.registry import build_worker_transform

        transforms = {name: build_worker_transform(agent.policy_for_source(name), augment, train)
                      for name, augment in zip(self.source_names, self.source_augments)}

        def run(features, targets):
            name = features.get("profile")
            if name not in transforms:
                raise ValueError(f"multidomain worker transform needs a known features['profile'], got {name!r}")
            features, targets = transforms[name](features, targets)
            features = dict(features)
            features["profile"] = name
            return features, targets

        return run

    def make_loader(self, dataset, params, train, rank, world, sequence_split=0, seed=0,
                    draw_augment=None, exact_resume=False):
        """Factory hook used by data.registry.make_loader; domain selection is shared by all ranks."""
        kwargs = dataloader_kwargs(params)
        batch_size = int(kwargs.pop("batch_size", 1))
        if batch_size < 1 or world < 1 or not 0 <= rank < world:
            raise ValueError("multidomain loader needs positive batch_size/world and a valid rank")
        if train and exact_resume:
            if int(kwargs.get("num_workers") or 0) != 0:
                raise ValueError("train.exact_resume requires dataloader.params.num_workers=0 so prefetched worker "
                                 "batches cannot be lost or replayed")
            # DataLoader consumes this generator when an iterator is created. Keeping that draw private makes
            # replaying already-consumed batches leave the checkpointed model RNG untouched.
            kwargs["generator"] = torch.Generator().manual_seed(seed + rank)
        sampler = (HomogeneousProfileBatchSampler(self, batch_size, rank, world, seed, sequence_split,
                                                   draw_augment, train, exact_resume=exact_resume)
                   if train else ProfileEvalBatchSampler(self, batch_size, rank, world))
        kwargs.pop("drop_last", None)
        kwargs.pop("shuffle", None)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, **kwargs)


class HomogeneousProfileBatchSampler(Sampler):
    """One deterministic domain per global batch; each rank gets its own contiguous shard."""

    def __init__(self, dataset: MultiProfileDataset, batch_size, rank, world, seed=0, sequence_split=0,
                 draw_augment=None, train=True, exact_resume=False):
        self.dataset, self.batch_size, self.rank, self.world = dataset, int(batch_size), int(rank), int(world)
        self.global_batch = self.batch_size * self.world
        self.seed, self.sequence_split, self.draw_augment, self.train = int(seed), int(sequence_split), draw_augment, train
        self.exact_resume = bool(exact_resume)
        self.epoch = 0
        self.batches = max(1, math.ceil(len(dataset) / self.global_batch))
        self._sequence_samplers = []
        for source, child in enumerate(dataset.datasets):
            sequences = getattr(child, "sequences", None)
            if self.sequence_split > 0 and callable(sequences):
                from unidrivevla.data.registry import SequenceBatchSampler

                from unidrivevla.data.augment import geometry

                source_draw = geometry(self.dataset.source_augments[source])
                sampler = SequenceBatchSampler(sequences(), self.global_batch, 0, 1, self.sequence_split,
                                               self.seed + source, source_draw or draw_augment,
                                               exact_resume=self.exact_resume)
                self._sequence_samplers.append(sampler)
            else:
                self._sequence_samplers.append(None)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for sampler in self._sequence_samplers:
            if sampler is not None:
                sampler.set_epoch(self.epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        weights = torch.tensor(self.dataset.source_weights, dtype=torch.double)
        choices = torch.multinomial(weights, self.batches, replacement=True, generator=generator).tolist()
        sequence_iters = [iter(sampler) if sampler is not None else None for sampler in self._sequence_samplers]
        sequence_epochs = [self.epoch] * len(sequence_iters)
        start = self.rank * self.batch_size
        stop = start + self.batch_size
        for source in choices:
            offset, size = self.dataset.source_offsets[source], self.dataset.source_sizes[source]
            sampler = self._sequence_samplers[source]
            if sampler is None:
                local = torch.randint(size, (self.global_batch,), generator=generator).tolist()
                from unidrivevla.data.augment import geometry

                draw = geometry(self.dataset.source_augments[source]) or self.draw_augment
                batch = [(offset + index, draw()) for index in local] if draw else [offset + index for index in local]
            else:
                try:
                    sampled = next(sequence_iters[source])
                except StopIteration:
                    sequence_epochs[source] += 1
                    sampler.set_epoch(sequence_epochs[source])
                    sequence_iters[source] = iter(sampler)
                    sampled = next(sequence_iters[source])
                batch = []
                for item in sampled:
                    if isinstance(item, tuple):
                        index, augment = item[:2]
                        batch.append((offset + index, augment))
                    else:
                        batch.append(offset + item)
            yield batch[start:stop]

    def __len__(self):
        return self.batches


class ProfileEvalBatchSampler(Sampler):
    """Sequential DDP evaluation shards without padding, so every source example is scored exactly once."""

    def __init__(self, dataset, batch_size, rank, world):
        self.dataset, self.batch_size, self.rank, self.world = dataset, int(batch_size), int(rank), int(world)
        self.groups = []
        schedule = []
        for source, (offset, size, name) in enumerate(zip(
                dataset.source_offsets, dataset.source_sizes, dataset.source_names)):
            per_rank = math.ceil(size / self.world)
            start = offset + self.rank * per_rank
            stop = min(offset + size, start + per_rank)
            schedule.append((source, str(name), len(range(start, stop, self.batch_size))))
            self.groups.extend(list(range(chunk, min(chunk + self.batch_size, stop)))
                               for chunk in range(start, stop, self.batch_size))
        self.fsdp_eval_schedule = tuple(schedule)

    def __iter__(self):
        yield from self.groups

    def __len__(self):
        return len(self.groups)


def _profiles(cfg, agent=None):
    # Inference may restore profile metadata from the checkpoint while the active root config names a different
    # head frame. The model that will consume the batch is authoritative; the config remains the plugin/mock fallback.
    profiles = getattr(agent, "profiles", None)
    if profiles:
        return profiles
    profiles = OmegaConf.select(cfg, "profiles") or OmegaConf.select(cfg, "agent.profiles")
    if profiles is None:
        raise ValueError("data_loader=multidomain requires named profiles in profiles or agent.profiles")
    profile_for_source = getattr(agent, "profile_for_source", None)
    if callable(profile_for_source):
        return {str(name): profile_for_source(str(name)) for name in profiles}
    return profiles


class _SourceAgent:
    def __init__(self, profile):
        self.profile = profile


@register("multidomain")
def multidomain_loader(cfg, agent, split: str):
    from unidrivevla.adapters.unidrivevla import Profile

    profiles = _profiles(cfg, agent)
    sources = OmegaConf.select(cfg, "multidomain.sources") or []
    if len(sources) < 2:
        raise ValueError("data_loader=multidomain needs at least two sources")
    datasets, names, weights, augments = [], [], [], []
    for source in sources:
        name = str(source.get("name", "")).strip()
        loader = str(source.get("loader", name)).lower()
        if not name or name not in profiles:
            raise ValueError(f"multidomain source {name!r} has no matching named profile")
        if loader == "multidomain" or loader not in LOADERS:
            raise KeyError(f"source {name!r} uses unknown loader {loader!r}")
        source_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        source_cfg = OmegaConf.merge(source_cfg, source.get("overrides") or {})
        profile = profiles[name] if isinstance(profiles[name], Profile) else Profile.from_config(profiles[name])
        source_cfg.profile = OmegaConf.create(profile.__dict__)
        source_cfg.data_loader = loader
        weight = float(source.get("weight", 1.0))
        dataset = LOADERS[loader](source_cfg, _SourceAgent(profile), split)
        if len(dataset) == 0:
            raise ValueError(f"multidomain source {name!r} is empty for split {split!r}")
        names.append(name)
        weights.append(weight)
        augments.append(source.get("augment"))
        datasets.append(dataset)
    return MultiProfileDataset(datasets, names, weights, augments)
