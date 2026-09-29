"""
Perception curriculum for pretraining on CARLA datasets. Just load everything, no filtering, no skipping failed routes.
"""

import logging
import os

from lead.data_buckets import route_filtering
from lead.data_buckets.abstract_bucket_collection import AbstractBucketCollection
from lead.data_buckets.bucket import Bucket
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


class FullPretrainBucketCollection(AbstractBucketCollection):
    def __init__(self, root: str | list[str], config: TrainingConfig):
        self.buckets = [Bucket(config)]
        super().__init__(root, config)
        LOG.info("Using pre-train bucket collection")

    def _build_buckets(self):
        """Build bucket collection from scratch"""
        LOG.info(f"Building buckets from data at: {self.root}")
        for route_path in self.iter_root():
            if route_filtering.route_not_finished(route_path):
                LOG.info(f"Skipping unfinished route {route_path}")
                continue
            if route_filtering.route_failed(route_path):
                LOG.info(f"Skipping invalid route {route_path}")
                continue
            self.trainable_routes += 1
            frame_count = 0
            for seq in self.iter_route(route_path):
                self.buckets[0].add(route_path, seq)
                self.trainable_frames += 1
                frame_count += 1
            LOG.info(f"Added {frame_count} frames from route {route_path}")

    def cache_file_path(self) -> str:
        """Return path for cache file"""
        return os.path.join(
            self.config.bucket_collection_path,
            "full_pretrain_buckets.gz",
        )

    def buckets_mixture_per_epoch(self, _) -> dict[int, float]:
        return {0: 1.0}
