from __future__ import annotations

import abc
import logging

import numpy as np
import torch
from beartype import beartype

from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


class AbstractMixedDatasetSampleScheduler(abc.ABC):
    """Abstract class for mixed dataset sample schedulers.
    This class defines the interface for schedulers that determine
    how many samples to draw from each dataset in a mixed dataset training setup.
    """

    @beartype
    def __init__(self, datasets: list[torch.utils.data.Dataset]):
        self.datasets = datasets
        self.num_datasets = len(datasets)

    @beartype
    @abc.abstractmethod
    def get_batches_schedule(self, epoch: int) -> list[float]:
        """Get batch ratio for each dataset for the given epoch. Should sum to to batch size per GPU."""
        pass


class UniformSampleScheduler(AbstractMixedDatasetSampleScheduler):
    """Scheduler that returns equal ratios for all datasets in mixed dataset training."""

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        datasets: list[torch.utils.data.Dataset],
    ):
        self.config = config
        self.datasets = datasets
        self.num_datasets = len(datasets)

    def get_batches_schedule(self, _: int) -> list[float]:
        """Return equal ratios for all datasets."""
        return [
            int(
                1.0
                / self.num_datasets
                * self.config.batch_size
                / torch.cuda.device_count(),
            ),
        ] * self.num_datasets


class Sim2RealSampleScheduler(AbstractMixedDatasetSampleScheduler):
    """Scheduler that gradually shifts sampling from simulation to real-world data over epochs.
    Used in mixed dataset training."""

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        datasets: list[torch.utils.data.Dataset],
    ):
        from data_loader.carla_dataset import CARLAData

        self.config = config
        self.datasets = datasets
        self.num_datasets = len(datasets)
        assert self.num_datasets == 2, (
            "Sim2RealSampleScheduler only supports 2 datasets."
        )
        assert isinstance(datasets[0], CARLAData), "First dataset must be CARLAData."

    def get_batches_schedule(self, epoch: int) -> list[float]:
        """Return batch ratios for sim and real datasets based on epoch."""
        # TODO: make this work with any BS number
        anchors = {
            0: {0: 40, 1: 24},
            1: {0: 32, 1: 32},
            3: {0: 24, 1: 40},
            7: {0: 16, 1: 48},
            15: {0: 8, 1: 56},
            31: {0: 0, 1: 64},
        }
        for anchor_epoch in sorted(anchors.keys(), reverse=True):
            if epoch >= anchor_epoch:
                return [
                    anchors[anchor_epoch][i] // torch.cuda.device_count()
                    for i in range(self.num_datasets)
                ]


class MixedDataset(torch.utils.data.Dataset):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        datasets: list[torch.utils.data.Dataset],
    ):
        self.datasets = datasets
        self.config = config
        self.num_datasets = len(datasets)

        # Store sub-dataset sizes
        self.dataset_sizes = [len(ds) for ds in datasets]
        self.size = sum(self.dataset_sizes)

        assert all(len(ds) == self.dataset_sizes[0] for ds in datasets), (
            "Assumption failed: All datasets must have the same size."
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """
        Get item from the mixed dataset.

        The index comes from MixedSampler in the form:
            index = dataset_idx + dataset_sample_idx * num_datasets

        Where:
            - dataset_idx is which dataset (0, 1, 2, ...)
            - dataset_sample_idx is the actual index in that underlying dataset
        """
        dataset_idx = index % self.num_datasets
        dataset_sample_idx = index // self.num_datasets

        # Access the underlying dataset directly
        return self.datasets[dataset_idx][dataset_sample_idx]

    def shuffle(self, epoch):
        """Shuffle the underlying datasets with custom implemented shuffle function."""
        for dataset in self.datasets:
            dataset.shuffle(epoch)


class MixedSampler(torch.utils.data.BatchSampler):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        samplers: list[torch.utils.data.Sampler],
        sample_scheduler: AbstractMixedDatasetSampleScheduler,
    ):
        """
        Sampler for MixedDataset that samples from each dataset according to sample_scheduler.

        Args:
            config: Training configuration
            samplers: List of samplers, one for each underlying dataset
            sample_scheduler: Scheduler that determines the ratio of samples from each dataset
        """
        self.samplers = samplers
        LOG.info(
            f"MixedSampler using {len(samplers)} samplers. Each sampler size: {[len(s) for s in samplers]}",
        )
        assert all(len(samplers[0]) == len(s) for s in samplers), (
            "All samplers must have the same length."
        )
        self.sample_scheduler = sample_scheduler
        self.drop_last = True
        self.num_datasets = len(samplers)
        self.config = config

        # Calculate batch sizes per dataset based on scheduler ratios
        self.update_batch_sizes(0)

    def update_batch_sizes(self, epoch):
        """Calculate how many samples to take from each dataset per batch."""
        self.batch_sizes = self.sample_scheduler.get_batches_schedule(epoch)
        assert (
            sum(self.batch_sizes) * torch.cuda.device_count() == self.config.batch_size
        ), "Batch sizes must sum to total batch size"

    def __iter__(self):
        """
        Yields batches where each batch contains samples from each dataset according to ratios.

        The MixedDataset interleaves samples: [ds0_s0, ds1_s0, ds0_s1, ds1_s1, ...]
        So we need to generate indices that respect this interleaving pattern.
        """
        iterators = [iter(s) for s in self.samplers]

        try:
            while True:
                batch = []

                # Step 1: Collect indices from each dataset's sampler
                dataset_indices = []
                for dataset_idx, it in enumerate(iterators):
                    indices = []
                    for _ in range(self.batch_sizes[dataset_idx]):
                        indices.append(next(it))
                    dataset_indices.append(indices)

                # Step 2: Convert to MixedDataset global indices
                # MixedDataset structure: index % num_datasets gives dataset_idx
                #                        index // num_datasets gives sample_idx within that dataset
                for dataset_idx in range(self.num_datasets):
                    for local_idx in dataset_indices[dataset_idx]:
                        # Map from (dataset_idx, dataset_sample_idx) to MixedDataset global index
                        mixed_dataset_idx = dataset_idx + local_idx * self.num_datasets
                        batch.append(mixed_dataset_idx)

                yield batch

        except StopIteration:
            return

    def __len__(self):
        """Return the number of batches per epoch."""
        return (
            len(self.samplers[0]) // self.config.batch_size
        ) * torch.cuda.device_count()


def mixed_data_collate_fn(batch):
    """Custom collate function to handle missing keys in mixed dataset batches.
    What happens is that some datasets may not have all keys (e.g., some datasets may not have semantic segmentation labels).
    This function collates the batch while filling in missing keys with default values.
    """
    all_keys = set()
    for b in batch:
        all_keys.update(b.keys())

    collated = {}
    for key in all_keys:
        vals = [b.get(key, None) for b in batch]

        if all(v is None for v in vals):
            collated[key] = None
            continue

        ref = next(v for v in vals if v is not None)

        new_vals = []
        for v in vals:
            if v is None:
                if torch.is_tensor(ref):
                    v = torch.zeros_like(ref)
                elif isinstance(ref, np.ndarray):
                    v = np.zeros_like(ref)
                elif isinstance(ref, str):
                    v = ""
                else:
                    try:
                        v = ref.__class__()
                    except:
                        v = None
            new_vals.append(v)
        try:
            collated[key] = torch.utils.data._utils.collate.default_collate(new_vals)
        except:
            pass

    return collated
