import glob
import gzip
import os
import time

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import ujson
from beartype import beartype
from numpy.random import default_rng
from torch.utils.data import Dataset
from tqdm import tqdm

from lead.common.constants import SourceDataset
from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from lead.training.rfs import compute_rfs


class WODE2EData(Dataset):
    """Data loader for Waymo-Open-Dataset E2E Challenge"""

    @beartype
    def __init__(
        self,
        root,
        config: TrainingConfig,
        training: bool = False,
        val: bool = False,
        test: bool = False,
        training_session_cache=None,
        random: bool = True,
    ):
        self.root = root
        self.config = config
        self.training_session_cache = training_session_cache
        self.training = training
        self.val = val
        self.test = test
        self.random = random
        assert (int(training) + int(val) + int(test)) == 1, (
            "One of training, val or test must be true"
        )

        self.rank = config.rank
        self.data_sampling_generator = default_rng(seed=self.config.seed)

        self._rgb = glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True)
        self._measurements = glob.glob(
            os.path.join(self.root, "**/*.json.gz"),
            recursive=True,
        )
        self._rgb.sort()
        self._measurements.sort()

        self._rgb = self._rgb
        self._measurements = self._measurements

        self.shuffle(0)

        self.size = len(self.rgb)

        if self.rank == 0:
            print(
                f"[Waymo E2E] Found {len(self.rgb)} images, {len(self.measurements)} metas in {self.root}. Size {self.size}",
            )

    def shuffle(self, epoch):
        # Use epoch as seed for reproducible sampling across epochs
        rng = default_rng(seed=self.config.seed + epoch)

        if (
            not (self.val or self.test)
            and self.config.waymo_e2e_num_training_samples > 0
        ):
            target_size: int = self.config.waymo_e2e_num_training_samples
            original_size = len(self._rgb)

            if target_size > original_size:
                # Calculate how many times to repeat and how many extras needed
                repeats = target_size // original_size
                remainder = target_size % original_size

                # Create indices for upsampling
                indices = list(range(original_size)) * repeats

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
                # If target size is smaller, randomly subsample without replacement
                indices = rng.choice(original_size, size=target_size, replace=False)

            # Apply the sampling indices to all data structures
            self.rgb = [self._rgb[i] for i in indices]
            self.measurements = [self._measurements[i] for i in indices]
        else:
            # If sampling is disabled (0 or negative), use original data
            if self.random:
                indices = list(range(len(self._rgb)))
                rng.shuffle(indices)
                self.rgb = [self._rgb[i] for i in indices]
                self.measurements = [self._measurements[i] for i in indices]
            else:
                self.rgb = self._rgb
                self.measurements = self._measurements

            if self.config.waymo_e2e_subsample_factor > 1:
                factor = self.config.waymo_e2e_subsample_factor
                self.rgb = self.rgb[::factor]
                self.measurements = self.measurements[::factor]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.val:
            return self._val(index)
        if self.test:
            return self._test(index)
        return self._load(index)

    def _val(self, index):
        data = self._load(index)
        with gzip.open(self.measurements[index], "rt", encoding="utf-8") as f:
            preferences = ujson.load(f)["preferences"]

        preference_trajectories = {}
        for i in range(3):
            x = preferences[i]["pos_x"]
            y = preferences[i]["pos_y"]
            traj = np.array([x, y]).T
            traj[:, 1] *= -1  # Different coordinate system
            preference_trajectories[i] = traj
        data["preferences"] = preference_trajectories
        data["preference_scores"] = np.array(
            [preferences[i]["preference_score"] for i in range(3)],
        )
        return data

    def _test(self, index):
        data = self._load(index)
        data.update(
            {
                "sequence_id": self.rgb[index].split("/")[-3],
                "frame_id": self.rgb[index].split("/")[-1].split(".")[0],
            },
        )
        return data

    def _load(self, index):
        before = time.time()
        # Initialize cache or cache dummy
        cache = (
            self.training_session_cache
            if self.training_session_cache is not None
            else {}
        )
        # Load data
        measurement_path = self.measurements[index]
        if measurement_path not in cache:
            with gzip.open(measurement_path, "rt", encoding="utf-8") as f:
                measurement = ujson.load(f)
            with open(self.rgb[index], "rb") as f:
                raw_jpg_bytes = f.read()
            cache[measurement_path] = (raw_jpg_bytes, measurement)

        # Decompress jpg
        raw_jpg_bytes, measurement = cache[measurement_path]
        rgb = None
        if raw_jpg_bytes is not None:
            rgb = cv2.imdecode(np.frombuffer(raw_jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.transpose(rgb, (2, 0, 1))

        # Construct
        data = {
            "source_dataset": SourceDataset.WAYMO_E2E_2025,
            "speed": np.linalg.norm(
                np.array(
                    [measurement["past"]["vx"][-1], measurement["past"]["vy"][-1]],
                ),
            ),
            "past_speeds": np.linalg.norm(
                np.array([measurement["past"]["vx"], measurement["past"]["vy"]]).T,
                axis=-1,
            ),
            "past_positions": np.array(
                [measurement["past"]["x"], measurement["past"]["y"]],
            ),
            "command": np.array(
                {
                    0: [1, 0, 0, 0],
                    1: [0, 1, 0, 0],
                    2: [0, 0, 1, 0],
                    3: [0, 0, 0, 1],
                }[int(measurement["command"])],
            ),
        }
        if rgb is not None:
            data["rgb"] = rgb

        # Truncate past data
        data["past_speeds"] = data["past_speeds"][
            -self.config.num_past_samples_used - 1 : -1
        ]
        data["past_positions"] = data["past_positions"][
            :,
            -self.config.num_past_samples_used - 1 : -1,
        ]
        data["past_positions"][1, :] = -data["past_positions"][
            1,
            :,
        ]  # Different coordinate system
        data["past_positions"] = data["past_positions"].T  # Shape (T, 2)
        data["past_speeds"] = data["past_speeds"].reshape(-1, 1)  # Shape (T, 1)
        # Subsample future waypoints
        if "prediction" in measurement:
            data["future_waypoints"] = np.array(
                [measurement["prediction"]["x"], measurement["prediction"]["y"]],
            ).T
            data["future_waypoints"] = data["future_waypoints"][
                ::2
            ]  # Downsample from 4Hz to 2Hz (take indices 0, 2, 4, ...)
            data["future_waypoints"][:, 1] = -data["future_waypoints"][
                :,
                1,
            ]  # Different coordinate system

        data["loading_time"] = data["loading_meta_time"] = data[
            "loading_sensor_time"
        ] = time.time() - before
        return data


@beartype
def upsample_trajectory(
    trajectory: jt.Float[npt.NDArray, "T 2"],
    target_frequency: int = 4,
    source_frequency: int = 2,
) -> jt.Float[npt.NDArray, "T_upsampled 2"]:
    """Upsample trajectory from 2Hz to 4Hz using linear interpolation.

    For 10 points at 2Hz, produces 20 points at 4Hz by inserting interpolated points.
    """
    upsample_factor = target_frequency // source_frequency
    num_source_points = trajectory.shape[0]
    # For even upsampling (e.g., 10->20), we need: (num_source_points - 1) * factor + num_source_points
    # But simpler: just double and append one more if needed
    num_target_points = num_source_points * upsample_factor

    upsampled = np.zeros((num_target_points, 2))

    for i in range(num_source_points - 1):
        start_idx = i * upsample_factor
        upsampled[start_idx] = trajectory[i]

        for j in range(1, upsample_factor):
            alpha = j / upsample_factor
            upsampled[start_idx + j] = (1 - alpha) * trajectory[i] + alpha * trajectory[
                i + 1
            ]

    # Handle the last point(s)
    upsampled[-upsample_factor:] = trajectory[-1]

    return upsampled


@beartype
def evaluate_waymo_e2e(model: TFv6, config: TrainingConfig) -> float:
    """Evaluate the model on the Waymo E2E validation set and return the RFM score."""
    from lead.tfv6.tfv6 import Prediction

    dataset = WODE2EData(
        root=config.waymo_e2e_val_data_root,
        config=config,
        val=True,
    )
    total_rfm_score = 0.0
    num_samples = len(dataset)
    for i in tqdm(range(num_samples)):
        data = dataset[i]

        tensor_data = torch.utils.data._utils.collate.default_collate([data])
        prediction: Prediction = model(tensor_data)

        pred = np.array(
            [
                upsample_trajectory(
                    prediction.pred_future_waypoints[0].detach().cpu().numpy(),
                )[None],
            ],
        )
        preferences = [[data["preferences"][i] for i in range(3)]]
        preferences_scores = np.array(
            [[data["preference_scores"][i]] for i in range(3)],
        ).reshape(1, 3)
        speed = np.array([data["speed"]]).reshape(-1)

        rfm_score = compute_rfs(
            prediction_trajectories=pred,
            rater_specified_trajectories=preferences,
            rater_scores=preferences_scores,
            initial_speed=speed,
            detailed_output=False,
        )
        total_rfm_score += rfm_score
    return float(total_rfm_score / num_samples)


if __name__ == "__main__":
    from lead.tfv6.tfv6 import TFv6

    config = TrainingConfig()
    config.LTF = True
    config.use_planning_decoder = True
    config.carla_root = "data/carla_leaderboad2_v14/results/data/sensor_data"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = Model(device=device, config=config).to(device)

    # rfm_score = evaluate_waymo_e2e(model, config)
    # print(f"Waymo E2E Validation RFM Score: {rfm_score}")
    dataset = WODE2EData(
        root=config.waymo_e2e_val_data_root,
        config=config,
        val=True,
    )
    print(dataset[0]["past_positions"].shape, dataset[0]["past_speeds"].shape)
