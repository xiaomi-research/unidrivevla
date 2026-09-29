import glob
import gzip
import logging
import os
import pickle
import time

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from torch.utils.data import Dataset

from lead.common.constants import (
    NavSimBoundingBoxIndex,
    SourceDataset,
    TransfuserBoundingBoxIndex,
)
from lead.data_loader import carla_dataset_utils, navsim_dataset_utils
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


class NavsimData(Dataset):
    """Data loader for NavSim data"""

    def __init__(
        self,
        root,
        config: TrainingConfig,
        training_session_cache=None,
        random=True,
    ):
        self.root = root
        self.config = config
        self.training_session_cache = training_session_cache
        self.rank = config.rank
        self.data_sampling_generator = default_rng(seed=self.config.seed)

        self._feature = glob.glob(
            os.path.join(self.root, "**/transfuser_feature.gz"),
            recursive=True,
        )
        self._target = glob.glob(
            os.path.join(self.root, "**/transfuser_target.gz"),
            recursive=True,
        )

        self._feature.sort()
        self._target.sort()

        self.random = random
        self.shuffle(0)
        self.size = len(self.feature)

        if self.rank == 0:
            assert len(self._feature) == len(self._target), (
                f"Mismatch in number of files. Found {len(self._feature)} features and {len(self._target)} targets."
            )
            LOG.info(
                f"NavSim: Found {len(self._feature)} samples. Upsampled to {self.size} samples",
            )

    def shuffle(self, epoch):
        # Use epoch as seed for reproducible sampling across epochs
        rng = default_rng(seed=self.config.seed + epoch)

        if self.config.navsim_num_samples > 0:
            target_size: int = self.config.navsim_num_samples
            original_size = len(self._feature)

            if target_size > original_size:
                # Calculate how many times to repeat and how many extras needed
                repeats = target_size // original_size
                remainder = target_size % original_size

                # Create indices for upsampling
                indices = list(range(original_size)) * repeats

                if self.random:
                    # Add random samples for the remainder
                    if remainder > 0:
                        extra_indices = rng.choice(
                            original_size,
                            size=remainder,
                            replace=False,
                        )
                        indices.extend(extra_indices)

                    # Shuffle the indices to mix repeated and original samples
                    rng.shuffle(indices)
                else:
                    # Add the first 'remainder' samples in order
                    if remainder > 0:
                        indices.extend(list(range(remainder)))
            elif self.random:
                # If target size is smaller, randomly subsample without replacement
                indices = rng.choice(original_size, size=target_size, replace=False)
            else:
                # If not random, take the first 'target_size' samples
                indices = list(range(target_size))
            # Apply the sampling indices to all data structures
            self.feature = [self._feature[i] for i in indices]
            self.target = [self._target[i] for i in indices]
        else:
            # If sampling is disabled (0 or negative), use original data
            self.feature = self._feature
            self.target = self._target

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Initialize cache or cache dummy
        before = time.time()
        cache = self.training_session_cache
        if cache is None:
            cache = {}

        # Load data
        target_path = self.target[index]
        if target_path not in cache:
            feature_path = self.feature[index]

            with gzip.open(feature_path, "rb") as f:
                feature = pickle.load(f)

            with gzip.open(target_path, "rb") as f:
                target = pickle.load(f)

            cache[target_path] = (feature, target)

        feature, target = cache[target_path]

        rgb = cv2.imdecode(feature["camera_feature"], cv2.IMREAD_COLOR)
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC to CHW

        agent_states = target["agent_states"]
        agent_labels = target["agent_labels"]
        trajectory = target["trajectory"]

        bev_semantic_map = target["bev_semantic_map"].numpy().astype(np.uint8)
        bev_semantic_map = np.rot90(
            bev_semantic_map,
            1,
        )  # Align NavSim BEV with CARLA BEV
        bev_semantic_map = cv2.imencode(".png", bev_semantic_map)[1]

        data = {
            "sequence_id": self.feature[index].split("/")[-3],
            "frame_id": self.feature[index].split("/")[-1].split(".")[0],
            "rgb": rgb,
            "index": index,
            "trajectory": trajectory,
            "source_dataset": SourceDataset.NAVSIM,
            "future_waypoints": trajectory[:, :2],
            "future_yaws": trajectory[:, 2],
            "command": feature["status_feature"][:4],
            "speed": np.linalg.norm(feature["status_feature"][4:6]),
            "acceleration": np.linalg.norm(feature["status_feature"][6:8]),
            "status_feature": feature["status_feature"],
        }
        if self.config.detect_boxes:
            self._convert_navsim_bb_to_carla(agent_states, agent_labels, data)

        if self.config.use_bev_semantic:
            bev_semantic_map = cv2.imdecode(bev_semantic_map, cv2.IMREAD_UNCHANGED)
            data["navsim_bev_semantic"] = bev_semantic_map.astype(np.uint8)

        data["future_waypoints"][:, 1] = -data["future_waypoints"][
            :,
            1,
        ]  # Flip Y axis to align with CARLA
        data["future_yaws"] = -data["future_yaws"]  # Flip Y axis to align with CARLA

        data["loading_time"] = data["loading_meta_time"] = data[
            "loading_sensor_time"
        ] = time.time() - before
        return data

    def _convert_navsim_bb_to_carla(
        self,
        agent_states: jt.Float[npt.NDArray, "n 5"],
        agent_labels: jt.Bool[npt.NDArray, " n"],
        data: dict,
    ):
        """
        Convert NAVSIM bounding boxes to Transfuser format and add to data dictionary.

        Args:
            agent_states: Agent states from the NavSim dataset. X, Y, YAW, LENGTH, WIDTH
            agent_labels: Agent labels from the NavSim dataset. Boolean array indicating valid boxes.
            data: Data dictionary to add bounding box labels to.
        """
        boxes_array = []
        for i, box in enumerate(agent_states):
            if not agent_labels[i]:
                continue
            box_center_x = box[NavSimBoundingBoxIndex.X]
            box_center_y = -box[
                NavSimBoundingBoxIndex.Y
            ]  # Align NavSim Y axis with CARLA Y axis
            box_yaw = -box[
                NavSimBoundingBoxIndex.HEADING
            ]  # Align NavSim Y axis with CARLA Y axis
            box_length = box[NavSimBoundingBoxIndex.LENGTH] / 2
            box_width = box[NavSimBoundingBoxIndex.WIDTH] / 2
            if (
                box_center_x < self.config.min_x_meter
                or box_center_x > self.config.max_x_meter
            ):
                continue
            if (
                box_center_y < self.config.min_y_meter
                or box_center_y > self.config.max_y_meter
            ):
                continue
            box = {
                TransfuserBoundingBoxIndex.X: box_center_x,
                TransfuserBoundingBoxIndex.Y: box_center_y,
                TransfuserBoundingBoxIndex.W: box_length,
                TransfuserBoundingBoxIndex.H: box_width,
                TransfuserBoundingBoxIndex.YAW: box_yaw,
                TransfuserBoundingBoxIndex.VELOCITY: 0,
                TransfuserBoundingBoxIndex.BRAKE: 0,
                TransfuserBoundingBoxIndex.CLASS: 0,
            }
            # Sort by key and get array from values
            sorted_box = [box[key] for key in sorted(box.keys())]
            boxes_array.append(np.array(sorted_box))

        image_system_boxes = []
        for box in boxes_array:
            image_system_boxes.append(
                carla_dataset_utils.bb_vehicle_to_image_system(
                    box.reshape(1, -1),
                    self.config.pixels_per_meter,
                    self.config.min_x_meter,
                    self.config.min_y_meter,
                ).squeeze(),
            )
        image_system_boxes = np.array(image_system_boxes)

        # Pad with zeros if less than max_num_bbs
        if len(image_system_boxes) < self.config.max_num_bbs:
            padding = np.zeros((self.config.max_num_bbs - len(boxes_array), 8))
            if len(image_system_boxes) > 0:
                image_system_boxes = np.vstack((image_system_boxes, padding))
            else:
                image_system_boxes = padding

        for key, value in navsim_dataset_utils.get_centernet_labels(
            image_system_boxes,
            self.config,
            self.config.navsim_num_bb_classes,
        ).items():
            data["navsim_" + key] = value
