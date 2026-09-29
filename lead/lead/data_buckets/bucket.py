import numpy as np
from beartype import beartype

from lead.training.config_training import TrainingConfig


class Bucket:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.bev_3rd_person_images = []
        self.images = []
        self.images_perturbated = []
        self.semantics = []
        self.semantics_perturbated = []
        self.hdmap = []
        self.hdmap_perturbated = []
        self.depth = []
        self.depth_perturbated = []
        self.lidars = []
        self.radars = []
        self.radars_perturbated = []
        self.bboxes = []
        self.metas = []
        self.sample_start = []
        self.route_dirs = []
        self.route_indices = []
        self.global_indices = []
        self.global_index = 0

    @beartype
    def add(self, route_dir: str, seq: int):
        # Loads the current (and past) frames (if seq_len > 1)
        self.bev_3rd_person_images.append(
            route_dir + "/3rd_person" + (f"/{(seq):04}.jpg"),
        )
        self.images.append(route_dir + "/rgb" + (f"/{(seq):04}.jpg"))
        self.images_perturbated.append(
            route_dir + "/rgb_perturbated" + (f"/{(seq):04}.jpg"),
        )
        self.semantics.append(route_dir + "/semantics" + (f"/{(seq):04}.png"))
        self.semantics_perturbated.append(
            route_dir + "/semantics_perturbated" + (f"/{(seq):04}.png"),
        )
        self.hdmap.append(route_dir + "/hdmap" + (f"/{(seq):04}.png"))
        self.hdmap_perturbated.append(
            route_dir + "/hdmap_perturbated" + (f"/{(seq):04}.png"),
        )
        self.depth.append(route_dir + "/depth" + (f"/{(seq):04}.png"))
        self.depth_perturbated.append(
            route_dir + "/depth_perturbated" + (f"/{(seq):04}.png"),
        )
        self.lidars.append(route_dir + "/lidar" + (f"/{(seq):04}.laz"))
        self.radars.append(route_dir + "/radar" + (f"/{(seq):04}.npz"))
        self.radars_perturbated.append(
            route_dir + "/radar_perturbated" + (f"/{(seq):04}.npz"),
        )
        self.route_dirs.append(route_dir)
        self.route_indices.append(seq)
        self.bboxes.append(route_dir + "/bboxes" + (f"/{(seq):04}.pkl"))
        self.metas.append(route_dir + "/metas" + (f"/{(seq):04}.pkl"))
        self.sample_start.append(seq)
        self.global_indices.append(self.global_index)
        self.global_index += 1

    def finalize(self):
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.bev_3rd_person_images = np.array(self.bev_3rd_person_images).astype(
            np.string_,
        )
        self.images = np.array(self.images).astype(np.string_)
        self.images_perturbated = np.array(self.images_perturbated).astype(np.string_)
        self.semantics = np.array(self.semantics).astype(np.string_)
        self.semantics_perturbated = np.array(self.semantics_perturbated).astype(
            np.string_,
        )
        self.hdmap = np.array(self.hdmap).astype(np.string_)
        self.hdmap_perturbated = np.array(self.hdmap_perturbated).astype(np.string_)
        self.depth = np.array(self.depth).astype(np.string_)
        self.depth_perturbated = np.array(self.depth_perturbated).astype(np.string_)
        self.lidars = np.array(self.lidars).astype(np.string_)
        self.radars = np.array(self.radars).astype(np.string_)
        self.radars_perturbated = np.array(self.radars_perturbated).astype(np.string_)
        self.bboxes = np.array(self.bboxes).astype(np.string_)
        self.metas = np.array(self.metas).astype(np.string_)
        self.route_dirs = np.array(self.route_dirs).astype(np.string_)
        self.route_indices = np.array(self.route_indices).astype(int)
        self.sample_start = np.array(self.sample_start)
        self.global_indices = np.array(self.global_indices)

    def __len__(self):
        return len(self.images)
