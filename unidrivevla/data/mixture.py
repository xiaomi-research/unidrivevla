"""Weighted multi-source training for compatible canonical profiles."""

import logging
import math

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from unidrivevla.data.registry import LOADERS, register

logger = logging.getLogger(__name__)
BUILTIN_PROFILES = {"nuscenes", "b2d", "navsim"}


def _profile(cfg):
    profile = OmegaConf.select(cfg, "profile")
    return OmegaConf.to_container(profile, resolve=True) if OmegaConf.is_config(profile) else profile


class MixtureDataset(ConcatDataset):
    def __init__(self, datasets, weights, names):
        if not datasets or len(datasets) != len(weights) or len(datasets) != len(names):
            raise ValueError("mixture needs equally sized, non-empty datasets, weights, and names")
        for name, dataset, weight in zip(names, datasets, weights):
            if len(dataset) == 0:
                raise ValueError(f"mixture source {name!r} is empty")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"mixture source {name!r} needs a finite positive weight, got {weight}")

        super().__init__(datasets)
        self.source_weights = tuple(weights)
        self.source_names = tuple(names)
        # Preserve the old per-sample weights for sequence_split=0.
        self.sample_weights = torch.cat([torch.full((len(d),), w / len(d), dtype=torch.double)
                                         for d, w in zip(datasets, weights)])

        self._streams = [None] * len(self)
        self._sequences = []
        self.sequence_sources = []
        offset = 0
        for source_id, dataset in enumerate(datasets):
            sequence_fn = getattr(dataset, "sequences", None)
            groups = list(sequence_fn()) if callable(sequence_fn) else [[i] for i in range(len(dataset))]
            groups = [list(map(int, seq)) for seq in groups if len(seq)]
            flattened = [index for seq in groups for index in seq]
            if sorted(flattened) != list(range(len(dataset))):
                raise ValueError(f"mixture source {names[source_id]!r}.sequences() must partition every sample once")
            for sequence_id, sequence in enumerate(groups):
                stream_id = (f"source={source_id}:{names[source_id]}:sequence={sequence_id}" if callable(sequence_fn)
                             else f"source={source_id}:{names[source_id]}:item={sequence[0]}")
                self._sequences.append([offset + index for index in sequence])
                self.sequence_sources.append(source_id)
                for index in sequence:
                    self._streams[offset + index] = stream_id
            offset += len(dataset)

    def sequences(self):
        return self._sequences

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        sample = super().__getitem__(index)
        if not isinstance(sample, (tuple, list)) or len(sample) < 3:
            raise ValueError("mixture sources must return canonical (features, targets, token) samples")
        features, targets, token, *rest = sample
        if not isinstance(features, dict):
            raise TypeError("canonical sample features must be a dictionary")
        source_id = next(i for i, end in enumerate(self.cumulative_sizes) if index < end)
        name = self.source_names[source_id]
        namespaced_token = f"source={source_id}:{name}::{token}"
        return {**features, "stream_id": self._streams[index]}, targets, namespaced_token, *rest


@register("mixture")
def mixture_loader(cfg, agent, split: str):
    sources = OmegaConf.select(cfg, "mixture") or []
    if not sources:
        raise ValueError("data_loader=mixture needs a `mixture` list of {loader, weight, overrides}")
    base_profile = _profile(cfg)
    if base_profile is None:
        raise ValueError("mixture needs one shared top-level profile; per-source profiles are unsupported")
    profile_name = base_profile.get("name") if isinstance(base_profile, dict) else None
    datasets, weights, names = [], [], []
    for source_id, source in enumerate(sources):
        name = str(source["loader"]).lower()
        if name == "mixture" or name not in LOADERS:
            raise KeyError(f"mixture source {name!r} is not a registered loader: {sorted(set(LOADERS) - {'mixture'})}")
        if name in BUILTIN_PROFILES and profile_name in BUILTIN_PROFILES and profile_name != name:
            raise ValueError(f"mixture source {source_id}:{name} cannot use built-in profile {profile_name!r}; "
                             "select a validated custom shared profile")
        try:
            weight = float(source.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mixture source {name!r} has an invalid weight {source.get('weight')!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"mixture source {name!r} needs a finite positive weight, got {weight}")
        overrides = source.get("overrides") or {}
        source_cfg = OmegaConf.merge(cfg, overrides)
        if _profile(source_cfg) != base_profile:
            raise ValueError(f"mixture source {source_id}:{name} overrides profile; mixed profiles are unsupported")
        dataset = LOADERS[name](source_cfg, agent, split)
        if len(dataset) == 0:
            raise ValueError(f"mixture source {source_id}:{name} is empty for split {split!r}")
        datasets.append(dataset)
        weights.append(weight)
        names.append(name)
        logger.info("mixture %s: %s %d samples, weight %g", split, name, len(dataset), weight)
    return MixtureDataset(datasets, weights, names)
