import os
from enum import IntEnum

from lead.data_buckets import route_filtering
from lead.data_buckets.abstract_bucket_collection import AbstractBucketCollection
from lead.data_buckets.bucket import Bucket
from lead.training.config_training import TrainingConfig


class Buckets(IntEnum):
    TOWN13 = 0
    OTHER_TOWNS = 1


class Town13HeldoutPosttrainBucketCollection(AbstractBucketCollection):
    def __init__(self, root: str | list[str], config: TrainingConfig):
        self.buckets = [Bucket(config), Bucket(config)]
        super().__init__(root, config)
        print("Using Town13 held-out post-train bucket collection")

    def _build_buckets(self):
        for route_path in self.iter_root():
            if route_filtering.route_failed(route_path):
                print(f"Skipping invalid route {route_path}")
                continue
            if route_filtering.route_not_finished(route_path):
                print(f"Skipping unfinished route {route_path}")
                continue
            self.trainable_routes += 1
            for seq in self.iter_route(route_path):
                if "Town13" in route_path:
                    self.buckets[Buckets.TOWN13].add(route_path, seq)
                else:
                    self.buckets[Buckets.OTHER_TOWNS].add(route_path, seq)
                    self.trainable_frames += 1

    def cache_file_path(self):
        """Return path for cache file"""
        return os.path.join(
            self.config.bucket_collection_path,
            f"town13_heldout_posttrain_buckets_{self.config.num_way_points_prediction}_{self.config.skip_first}_{self.config.skip_last}_{self.config.waypoints_spacing}.gz",
        )

    def buckets_mixture_per_epoch(self, _):
        total_samples = len(self.buckets[Buckets.TOWN13]) + len(
            self.buckets[Buckets.OTHER_TOWNS],
        )
        return {
            Buckets.TOWN13: 0.0,
            Buckets.OTHER_TOWNS: total_samples
            / max(1, len(self.buckets[Buckets.OTHER_TOWNS])),
        }
