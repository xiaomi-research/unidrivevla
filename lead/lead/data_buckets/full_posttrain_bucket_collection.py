import os

from lead.data_buckets import route_filtering
from lead.data_buckets.abstract_bucket_collection import AbstractBucketCollection
from lead.data_buckets.bucket import Bucket
from lead.training.config_training import TrainingConfig


class FullPosttrainBucketCollection(AbstractBucketCollection):
    def __init__(self, root: str | list[str], config: TrainingConfig):
        self.buckets = [Bucket(config)]
        super().__init__(root, config)
        print("Using full post-train curriculum")

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
                self.buckets[0].add(route_path, seq)
                self.trainable_frames += 1

    def cache_file_path(self):
        """Return path for cache file"""
        return os.path.join(
            self.config.bucket_collection_path,
            f"full_posttrain_buckets_{self.config.num_way_points_prediction}_{self.config.skip_first}_{self.config.skip_last}_{self.config.waypoints_spacing}.gz",
        )

    def buckets_mixture_per_epoch(self, _):
        return {0: 1.0}
