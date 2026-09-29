from __future__ import annotations

import logging
import random
import time

import cv2
import diskcache
import jaxtyping as jt
import laspy
import numpy as np
import numpy.typing as npt
import torch
from beartype import beartype
from numpy.random import default_rng
from torch.utils.data import Dataset

import lead.common.constants as constants
from lead.common import common_utils as common_utils
from lead.common.constants import (
    CarlaImageCroppingType,
    SourceDataset,
    TransfuserBEVOccupancyClass,
    TransfuserBEVSemanticClass,
)
from lead.data_loader import carla_dataset_utils
from lead.data_loader.carla_dataset_utils import (
    get_centernet_labels,
    image_augmenter,
    rasterize_lidar,
)
from lead.data_loader.training_cache import CacheKey, PersistentCache, SensorData
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


class CARLAData(Dataset):
    @beartype
    def __init__(
        self,
        root: str | list[str],
        config: TrainingConfig,
        training_session_cache: diskcache.core.Cache | dict | None = None,
        random: bool = True,
        build_cache: bool = False,
        build_buckets: bool = False,
    ):
        """
        CARLA dataset loader constructor.

        Args:
            root: Path to the root directory of the dataset or a list of paths.
            config: Configuration object containing the dataset parameters.
            training_session_cache: Cache for the current training session.
            random: Whether to shuffle the dataset randomly or not.
            build_cache: True will return the data earlier without sensor data to accelerate the cache building.
            build_buckets: True will return the data earlier without loading sensor data to accelerate the bucket building.
        """
        self.config = config
        self.rank = config.rank

        self.training_session_cache = training_session_cache
        self.persistent_cache = (
            PersistentCache(self.config) if self.config.use_persistent_cache else None
        )
        self.memory_cache = {}
        self.semantic_converter = np.uint8(
            list(constants.SEMANTIC_SEGMENTATION_CONVERTER.values()),
        )
        self.hdmap_converter = np.uint8(
            list(constants.CHAFFEURNET_TO_TRANSFUSER_BEV_SEMANTIC_CONVERTER.values()),
        )
        self.image_augmenter_func = image_augmenter(config, config.use_color_aug_prob)
        self.random = random
        self.build_cache = build_cache
        self.build_buckets = build_buckets
        self.sim2real_semantic_converter = np.uint8(
            list(constants.SIM2REAL_SEMANTIC_SEGMENTATION_CONVERTER.values()),
        )
        self.sim2real_bev_semantic_converter = np.uint8(
            list(constants.SIM2REAL_BEV_SEMANTIC_SEGMENTATION_CONVERTER.values()),
        )
        self.sim2real_bev_occupancy_converter = np.uint8(
            list(constants.SIM2REAL_BEV_OCCUPANCY_CLASS_CONVERTER.values()),
        )

        self.bucket_collection = self.config.carla_bucket_collection(root, config)
        self.shuffle(0)
        if self.rank == 0:
            LOG.info(f"All routes: {self.bucket_collection.total_routes}")
            LOG.info(f"Trainable routes: {self.bucket_collection.trainable_routes}")
            LOG.info(f"All frames: {self.bucket_collection.all_frames}")
            LOG.info(f"Trainable frames: {self.bucket_collection.trainable_frames}")

    def __getitem__(self, index):
        # ----------------------------------------------------------------------------------------
        # First part of the dataloader: lightweight meta data
        # ----------------------------------------------------------------------------------------
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        start_loading_time = time.time()
        data = {}

        future_waypoint_indices = [self.config.waypoints_spacing]
        for _ in range(self.config.num_way_points_prediction - 1):
            future_waypoint_indices.append(
                future_waypoint_indices[-1] + self.config.waypoints_spacing,
            )

        past_waypoint_indices = [self.config.waypoints_spacing]
        for _ in range(self.config.num_way_points_prediction - 1):
            past_waypoint_indices.append(
                past_waypoint_indices[-1] + self.config.waypoints_spacing,
            )

        # Determine files of index
        global_index = self.global_indices[index]  # Index in dataset

        # Load meta-data file from disk
        measurement_file = str(self.metas[index], encoding="utf-8")
        if self.memory_cache is not None and measurement_file in self.memory_cache:
            meta = self.memory_cache[measurement_file]
        else:
            meta = common_utils.read_pickle(measurement_file)
            if self.training_session_cache is not None:
                self.training_session_cache[measurement_file] = meta

        # Choosing normal or sensor-perturbated setup
        if (
            np.random.random() <= self.config.use_sensor_perburtation_prob
            and self.config.use_sensor_perburtation
            and not self.build_cache
            and not self.build_buckets
        ):
            perturbate_sensor = True
            perturbation_rotation = meta["perturbation_rotation"]
            perturbation_translation = meta["perturbation_translation"]
        else:
            perturbate_sensor = False
            perturbation_rotation = 0.0
            perturbation_translation = 0.0

        # Identification meta-data
        box_path = str(self.bboxes[index], "utf-8")
        data.update(
            {
                "source_dataset": SourceDataset.CARLA,
                "perturbate_sensor": perturbate_sensor,
                "perturbation_translation": perturbation_translation,
                "perturbation_rotation": perturbation_rotation,
                "index": index,
                "global_index": global_index,
                "bucket_identity": self.bucket_identity[index],
                "route_number": measurement_file.split("/")[-3],
                "frame_number": measurement_file.split("/")[-1].split(".")[0],
            },
        )

        # Meta-data needed for cache or bucket building
        if self.build_buckets or self.build_cache:
            if (not self.config.mixed_data_training and not self.build_cache) or (
                self.build_buckets
            ):
                data.update(meta)

                # Those data has variable length which is hard for collation.
                del data["past_filtered_state"]
                del data["privileged_past_positions"]
                del data["future_speeds"]
                del data["future_yaws"]
                del data["future_positions"]
                del data["past_positions"]
                del data["past_speeds"]
                del data["past_yaws"]
                del data["previous_target_points"]
                del data["next_target_points"]
                del data["previous_commands"]
                del data["next_commands"]

            if self.build_buckets:
                data["boxes"] = common_utils.read_pickle(box_path)

            data.update(
                {
                    "route_dir": "/".join(box_path.split("/")[:-2]),
                    "seq": int(box_path.split("/")[-1].split(".")[0]),
                },
            )

        # Meta-data needed for visualization
        if self.config.visualize_dataset:
            for attr in [
                "scenario_obstacles_ids",
                "scenario_actors_ids",
                "scenario_obstacles_convex_hull",
                "cut_in_actors_ids",
            ]:
                data[f"{attr}_len"] = len(meta[attr])

            json_boxes_i = common_utils.read_pickle(box_path)
            data["vehicles_future_waypoints"] = []
            data["vehicles_future_yaws"] = []
            for box in json_boxes_i:
                vehicle_future_waypoints = box.get("future_positions")
                if vehicle_future_waypoints is not None:
                    vehicle_future_waypoints = np.array(vehicle_future_waypoints)
                    augmented_vehicle_future_waypoints = (
                        carla_dataset_utils.perturbate_waypoints(
                            vehicle_future_waypoints,
                            y_perturbation=perturbation_translation,
                            yaw_perturbation=perturbation_rotation,
                        )
                    )
                    augmented_vehicle_future_yaws = carla_dataset_utils.perturbate_yaws(
                        np.array(box.get("future_yaws")),
                        yaw_perturbation=perturbation_rotation,
                    )
                    augmented_vehicle_future_waypoints = np.array(
                        [
                            augmented_vehicle_future_waypoints[i][:2]
                            for i in future_waypoint_indices
                            if i < len(augmented_vehicle_future_waypoints)
                        ],
                    )
                    augmented_vehicle_future_yaws = np.array(
                        [
                            augmented_vehicle_future_yaws[i]
                            for i in future_waypoint_indices
                            if i < len(augmented_vehicle_future_yaws)
                        ],
                    )
                    data["vehicles_future_waypoints"].append(
                        augmented_vehicle_future_waypoints,
                    )
                    data["vehicles_future_yaws"].append(augmented_vehicle_future_yaws)

        # Non-numeric meta-data
        for attr in [
            "town",
            "current_active_scenario_type",
            "previous_active_scenario_type",
            "changed_route",
            "stop_sign_hazard",
            "walker_hazard",
            "light_hazard",
            "vehicle_hazard",
            "lane_type_str",
            "is_intersection",
            "does_emergency_brake_for_pedestrians",
            "construction_obstacle_two_ways_stuck",
            "accident_two_ways_stuck",
            "parked_obstacle_two_ways_stuck",
            "vehicle_opens_door_two_ways_stuck",
            "vehicle_opened_door",
            "vehicle_door_side",
            "ego_lane_id",
            "rear_danger_8",
            "rear_danger_16",
            "brake_cutin",
            "weather_setting",
            "jpeg_storage_quality",
            "emergency_brake_for_special_vehicle",
            "visual_visibility",
            "num_parking_vehicles_in_proximity",
            "slower_bad_visibility",
            "slower_clutterness",
            "slower_occluded_junction",
            "over_head_traffic_light",
            "europe_traffic_light",
            "vehicle_hazard",
            "light_hazard",
            "walker_hazard",
            "stop_sign_hazard",
            "stop_sign_close",
            "num_dangerous_adversarial",
            "num_safe_adversarial",
            "num_ignored_adversarial",
            "rear_adversarial_id",
        ]:
            if attr in ["vehicle_door_side"] and attr in meta and meta[attr] is None:
                data[attr] = "NA"
            elif attr in ["vehicle_door_side"] and attr in meta:
                if isinstance(meta[attr], list):
                    data[attr] = meta[attr][0]  # List with one element
                else:
                    data[attr] = meta[attr]
            else:
                data[attr] = meta[attr]  # Some scenarios do not have these attributes

        # Numeric meta-data
        for attr in [
            "steer",
            "throttle",
            "brake",
            "dist_to_construction_site",
            "dist_to_accident_site",
            "dist_to_parked_obstacle",
            "dist_to_vehicle_opens_door",
            "dist_to_cutin_vehicle",
            "dist_to_pedestrian",
            "dist_to_biker",
            "distance_to_next_junction",
            "signed_dist_to_lane_change",
            "speed",
            "accel_x",
            "accel_y",
            "accel_z",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
            "speed_limit",
            "steer",
            "throttle",
            "privileged_acceleration",
            "route_curvature",
            "distance_to_next_junction",
            "distance_to_intersection_index_ego",
            "ego_lane_width",
            "route_left_length",
            "distance_ego_to_route",
            "target_speed_limit",
            "target_speed",
            "theta",
            "privileged_yaw",
            "traffic_light_height",
            "second_highest_speed",
            "second_highest_speed_limit",
            "perturbation_rotation",
            "perturbation_translation",
        ]:
            if attr in meta and meta[attr] is None:
                data[attr] = np.inf
            elif attr in meta:
                data[attr] = float(meta[attr])

        # Meta-data specific to scenario types
        for attr in [
            "current_active_scenario_type",
            "previous_active_scenario_type",
        ]:
            if (
                attr
                in ["previous_active_scenario_type", "current_active_scenario_type"]
                and attr in meta
                and meta[attr] is None
            ):
                data[attr] = "NA"
            else:
                data[attr] = meta[attr]  # Some scenarios do not have these attributes
        if (
            data["current_active_scenario_type"] is not None
            and data["current_active_scenario_type"] != "NA"
        ):
            data["scenario_type"] = data["current_active_scenario_type"]
        elif (
            data["previous_active_scenario_type"] is not None
            and data["previous_active_scenario_type"] != "NA"
        ):
            data["scenario_type"] = data["previous_active_scenario_type"]
        else:
            data["scenario_type"] = "NA"
        data["scenario_type_id"] = constants.SCENARIO_TYPES.index(data["scenario_type"])

        # Waypoints
        if self.config.use_planning_decoder or self.config.visualize_dataset:
            future_positions = meta.get("future_positions")
            future_yaws = meta.get("future_yaws")
            future_waypoints = np.array(
                [
                    future_positions[i][:2]
                    for i in future_waypoint_indices
                    if i < len(future_positions)
                ],
            ).reshape(-1, 2)
            future_yaws = np.array(
                [
                    future_yaws[i]
                    for i in future_waypoint_indices
                    if i < len(future_yaws)
                ],
            ).reshape(-1)

            if future_waypoints.shape[0] > 0:
                data["future_waypoints"] = carla_dataset_utils.perturbate_waypoints(
                    future_waypoints,
                    y_perturbation=perturbation_translation,
                    yaw_perturbation=perturbation_rotation,
                )
                data["future_yaws"] = carla_dataset_utils.perturbate_yaws(
                    future_yaws,
                    yaw_perturbation=perturbation_rotation,
                )

        # Route and target speed features
        if (
            self.config.use_planning_decoder
            or self.config.visualize_dataset
            or self.build_buckets
        ):
            # Normal route
            route = meta["route"]
            route = np.array(route[: self.config.num_route_points_smoothing])
            route = carla_dataset_utils.perturbate_route(
                route,
                y_perturbation=perturbation_translation,
                yaw_perturbation=perturbation_rotation,
            )
            if self.config.smooth_route:
                route = carla_dataset_utils.smooth_path(
                    self.config,
                    route,
                    target_first_distance=2.5,
                )

            # Meta data for route
            data["brake"] = meta["brake"]
            data["throttle"] = meta["throttle"]
            data["route"] = route[: self.config.num_route_points_prediction]
            data["route_labels_curvature"] = common_utils.waypoints_curvature(
                torch.from_numpy(data["route"]),
            )

        # Velocity
        if self.config.use_velocity:
            data["speed"] = meta["speed"]

        # Target point and navigation command
        ego_yaw = meta["theta"]
        ego_position = np.array(meta["pos_global"][:2])
        if self.config.use_noisy_tp:
            if self.config.use_kalman_filter_for_gps:
                ego_position = np.array(meta["filtered_pos_global"][:2])
            else:
                ego_position = np.array(meta["noisy_pos_global"][:2])

        def transform_and_augment(point: list[float]) -> jt.Float[npt.NDArray, " 2"]:
            ego_point = common_utils.inverse_conversion_2d(
                np.array(point),
                ego_position,
                ego_yaw,
            )
            return carla_dataset_utils.perturbate_target_point(
                ego_point,
                y_perturbation=perturbation_translation,
                yaw_perturbation=perturbation_rotation,
            )

        noisy_version = ""
        if self.config.use_noisy_tp:
            noisy_version = "_gps"
        next_tp_list = meta[
            f"next{noisy_version}_target_points_{self.config.tp_pop_distance}"
        ]
        next_command_list = meta[
            f"next{noisy_version}_commands_{self.config.tp_pop_distance}"
        ]
        # Merge duplicates target point
        filtered_tp_list = []
        filtered_command_list = []
        for pt, cmd in zip(next_tp_list, next_command_list):
            if (
                len(next_tp_list) == 2
                or not filtered_tp_list
                or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
            ):
                filtered_tp_list.append(pt)
                filtered_command_list.append(cmd)
        next_tp_list = filtered_tp_list
        next_command_list = filtered_command_list

        # Convert to ego and augment
        if len(next_tp_list) > 2:
            data["target_point_next"] = transform_and_augment(next_tp_list[2][:2])
            data["target_point"] = transform_and_augment(next_tp_list[1][:2])
            data["target_point_previous"] = transform_and_augment(next_tp_list[0][:2])
        else:
            assert len(next_tp_list) == 2
            data["target_point"] = transform_and_augment(next_tp_list[1][:2])
            data["target_point_next"] = transform_and_augment(next_tp_list[1][:2])
            data["target_point_previous"] = transform_and_augment(next_tp_list[0][:2])

        if self.config.use_discrete_command or self.config.visualize_dataset:
            data["command"] = carla_dataset_utils.command_to_one_hot(
                next_command_list[0],
            )
            data["next_command"] = carla_dataset_utils.command_to_one_hot(
                next_command_list[1],
            )
            data["previous_command"] = carla_dataset_utils.command_to_one_hot(
                next_command_list[0],
            )

        loading_meta_time = time.time() - start_loading_time
        data["loading_meta_time"] = loading_meta_time
        if self.build_buckets:
            return data

        # PlanT consumes only object tokens + meta -- skip the heavy sensor path
        # (and its persistent-cache round-trip, which is expensive on networked FS).
        if self.config.model_type == "plant":
            return data

        # ----------------------------------------------------------------------------------------
        # Second part of the dataloader: heavy sensor data.
        # Only load this for visualization or training.
        # ----------------------------------------------------------------------------------------
        start_loading_sensor_time = time.time()
        sensor_data = self._load_sensor_data(data, meta, index, perturbate_sensor)
        if self.build_cache:
            return data
        # BEV 3rd person images
        if self.config.load_bev_3rd_person_images:
            data["bev_3rd_person"] = sensor_data.bev_3rd_person_image
        # LiDAR BEV
        if not self.config.LTF:
            data["rasterized_lidar"] = (
                np.array(sensor_data.rasterized_lidar).squeeze()[None]
                if sensor_data.rasterized_lidar is not None
                else None
            )
        # RGB
        if sensor_data.image is None:
            data["rgb"] = None
        else:
            if self.config.use_color_aug:
                processed_image = self.image_augmenter_func(image=sensor_data.image)
            else:
                processed_image = sensor_data.image
            data["rgb"] = np.transpose(processed_image, (2, 0, 1))
        # Radars
        if self.config.use_radars:
            radar_list = carla_dataset_utils.preprocess_radar_input(
                self.config,
                {f"radar{i + 1}": arr for i, arr in enumerate(sensor_data.radars)},
            )
            for i, arr in enumerate(radar_list):
                data[f"radar{i + 1}"] = arr
            data["radar"] = np.concatenate(radar_list, axis=0)
            data["radar_detections"] = sensor_data.radar_detections

        # Semantic segmentation
        if self.config.use_semantic and sensor_data.semantic is not None:
            data["semantic"] = sensor_data.semantic[
                :: self.config.perspective_downsample_factor,
                :: self.config.perspective_downsample_factor,
            ]

        # Depth
        if self.config.use_depth and sensor_data.depth is not None:
            loaded_depth = sensor_data.depth.astype(
                np.float32,
            )  # Use only current frame
            if meta["dataset_information"]["save_depth_resolution_ratio"] != 1.0:
                loaded_depth = cv2.resize(
                    loaded_depth,
                    dsize=(
                        loaded_depth.shape[1]
                        * meta["dataset_information"]["save_depth_resolution_ratio"],
                        loaded_depth.shape[0]
                        * meta["dataset_information"]["save_depth_resolution_ratio"],
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
            if self.config.perspective_downsample_factor > 1:
                loaded_depth = cv2.resize(
                    loaded_depth,
                    dsize=(
                        loaded_depth.shape[1]
                        // self.config.perspective_downsample_factor,
                        loaded_depth.shape[0]
                        // self.config.perspective_downsample_factor,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
            data["depth"] = loaded_depth

        # HD-Map
        if self.config.use_bev_semantic:
            assert self.config.pixels_per_meter == 4.0
            # In data collection we have 2 pixels / meter and in training we have 4 pixels / meter.
            assert sensor_data.hdmap.shape[0] == sensor_data.hdmap.shape[1]
            hdmap_center = sensor_data.hdmap.shape[0] / 2
            hdmap_x_cut = (
                hdmap_center
                + np.array([self.config.min_x_meter, self.config.max_x_meter])
                * self.config.pixels_per_meter_collection
            ).astype(int)  # Cut the BEV around vehicle
            hdmap_y_cut = (
                hdmap_center
                + np.array([self.config.min_y_meter, self.config.max_y_meter])
                * self.config.pixels_per_meter_collection
            ).astype(int)  # Cut the BEV around vehicle
            loaded_hdmap = (
                sensor_data.hdmap[
                    hdmap_y_cut[0] : hdmap_y_cut[1],
                    hdmap_x_cut[0] : hdmap_x_cut[1],
                ]
                .repeat(2, axis=0)
                .repeat(2, axis=1)
            )  # Zoom in around ego and upscale
            data["hdmap"] = data["bev_semantic"] = loaded_hdmap

        # Occupancy
        if self.config.use_bev_semantic:
            assert (
                sensor_data.bev_occupancy.shape[0] == sensor_data.bev_occupancy.shape[1]
            )
            bev_occupancy_center = sensor_data.bev_occupancy.shape[0] / 2
            bev_occupancy_i_x_cut = (
                bev_occupancy_center
                + np.array([self.config.min_x_meter, self.config.max_x_meter]) * 4
            ).astype(int)  # Cut the BEV around vehicle
            bev_occupancy_i_y_cut = (
                bev_occupancy_center
                + np.array([self.config.min_y_meter, self.config.max_y_meter]) * 4
            ).astype(int)  # Cut the BEV around vehicle
            loaded_bev_occupancy = sensor_data.bev_occupancy[
                bev_occupancy_i_y_cut[0] : bev_occupancy_i_y_cut[1],
                bev_occupancy_i_x_cut[0] : bev_occupancy_i_x_cut[1],
            ]  # Zoom in around ego and upscale

            if not self.config.carla_leaderboard_mode:
                loaded_bev_occupancy = self.sim2real_bev_occupancy_converter[
                    loaded_bev_occupancy
                ]

            mask = loaded_bev_occupancy != TransfuserBEVOccupancyClass.UNLABELED
            loaded_hdmap[mask] = loaded_bev_occupancy[mask] + (
                len(TransfuserBEVSemanticClass) - len(TransfuserBEVOccupancyClass)
            )  # Add offset to BEV occupancy classes
            data["bev_semantic"] = loaded_hdmap
            if not self.config.carla_leaderboard_mode:
                data["bev_semantic"] = self.sim2real_bev_semantic_converter[
                    data["bev_semantic"]
                ]

        # 2D bounding boxes
        if self.config.detect_boxes:
            data.update(
                get_centernet_labels(
                    sensor_data.boxes,
                    self.config,
                    self.config.num_bb_classes,
                ),
            )

        # Finish loading. Measure time
        data["loading_time"] = time.time() - start_loading_time
        data["loading_sensor_time"] = time.time() - start_loading_sensor_time

        # Cut cameras down to only used cameras
        if (
            self.config.num_used_cameras != self.config.num_available_cameras
            and data.get("rgb") is not None
        ):
            n = self.config.num_available_cameras
            w = data["rgb"].shape[2] // n

            rgb_slices, depth_slices, semantic_slices = [], [], []
            for i, use in enumerate(self.config.used_cameras):
                if use:
                    s, e = i * w, (i + 1) * w
                    rgb_slices.append(data["rgb"][:, :, s:e])
                    if data.get("depth") is not None:
                        depth_slices.append(data["depth"][:, s:e])
                    if data.get("semantic") is not None:
                        semantic_slices.append(data["semantic"][:, s:e])

            data["rgb"] = np.concatenate(rgb_slices, axis=2)
            if depth_slices:
                data["depth"] = np.concatenate(depth_slices, axis=1)
            if semantic_slices:
                data["semantic"] = np.concatenate(semantic_slices, axis=1)

        # We crop of the image if specified in the config by self.config.crop_height pixels
        if self.config.crop_height > 0:
            if self.config.carla_crop_height_type == CarlaImageCroppingType.BOTTOM:
                if data.get("rgb") is not None:
                    data["rgb"] = data["rgb"][:, : -self.config.crop_height, :]
                if data.get("depth") is not None:
                    data["depth"] = data["depth"][: -self.config.crop_height]
                if data.get("semantic") is not None:
                    data["semantic"] = data["semantic"][: -self.config.crop_height]
            elif self.config.carla_crop_height_type == CarlaImageCroppingType.TOP:
                if data.get("rgb") is not None:
                    data["rgb"] = data["rgb"][:, self.config.crop_height :, :]
                if data.get("depth") is not None:
                    data["depth"] = data["depth"][self.config.crop_height :]
                if data.get("semantic") is not None:
                    data["semantic"] = data["semantic"][self.config.crop_height :]
            else:
                raise ValueError(
                    f"Unknown carla_crop_height_type: {self.config.carla_crop_height_type}",
                )

        # Transform the semantic segmentation to only contain the classes we care about in NavSim
        if data.get("semantic") is not None and not self.config.carla_leaderboard_mode:
            data["semantic"] = self.sim2real_semantic_converter[data["semantic"]]

        # Horizontal FOV reduction: crop left and right, then resize back
        if self.config.horizontal_fov_reduction > 0:
            if data["rgb"] is not None:  # (C, H, W)
                data["rgb"] = common_utils.fov_crop(
                    data["rgb"],
                    self.config.horizontal_fov_reduction,
                    chw=True,
                )
            if data.get("depth") is not None:  # (H, W)
                data["depth"] = common_utils.fov_crop(
                    data["depth"],
                    self.config.horizontal_fov_reduction,
                )
            if data.get("semantic") is not None:  # (H, W)
                data["semantic"] = common_utils.fov_crop(
                    data["semantic"],
                    self.config.horizontal_fov_reduction,
                    interpolation=cv2.INTER_NEAREST,
                )

        return data

    @beartype
    def _load_sensor_data(
        self,
        data: dict,
        meta: dict,
        index: int,
        perturbate_sensor: bool,
    ) -> SensorData:
        """
        Load sensor data for the given index and current meta.

        Args:
            data: Data dictionary to be filled.
            meta: Current meta dictionary containing meta data.
            index: index of the sample in the dataset.
            perturbate_sensor: Whether to load augmented data or not.
        Returns:
            SensorData object with all loaded sensor data.
        """

        image_path_str = str(self.images[index], encoding="utf-8")
        route = image_path_str.split("/")[-3]
        scenario = image_path_str.split("/")[-4]
        frame = image_path_str.split("/")[-1].split(".")[0]
        cache_key_perturbated = CacheKey(
            scenario=scenario,
            route=route,
            frame=frame,
            perturbated=True,
            config=self.config,
        )
        cache_key_normal = CacheKey(
            scenario=scenario,
            route=route,
            frame=frame,
            perturbated=False,
            config=self.config,
        )

        if perturbate_sensor:
            used_cache_key = cache_key_perturbated
        else:
            used_cache_key = cache_key_normal

        # Data in cache. Load them from cache
        if (
            (
                self.training_session_cache is not None
                and used_cache_key in self.training_session_cache
            )
            or (
                self.persistent_cache is not None
                and used_cache_key in self.persistent_cache
            )
        ) and not self.config.force_rebuild_data_cache:
            # Search cache, from fast to slow.
            try:
                cache = None
                if (
                    self.training_session_cache is not None
                    and used_cache_key in self.training_session_cache
                ):
                    cache = self.training_session_cache
                elif (
                    self.persistent_cache is not None
                    and used_cache_key in self.persistent_cache
                ):
                    cache = self.persistent_cache

                # Access cache - now stores CompressedSensorData directly
                cached_compressed_data = cache[used_cache_key]

                if (
                    self.training_session_cache is not None
                    and used_cache_key not in self.training_session_cache
                ):
                    self.training_session_cache[used_cache_key] = cached_compressed_data

                if (
                    self.persistent_cache is not None
                    and used_cache_key not in self.persistent_cache
                ):
                    self.persistent_cache[used_cache_key] = cached_compressed_data

                return cached_compressed_data.decompress()
            except EOFError:
                LOG.warning(
                    f"EOFError when reading cache for key {used_cache_key}. Rebuilding cache for this key.",
                )

        # Data not in cache, load from disk. Do this for all 3 views, since we might need them later.
        sensor_data_normal = self._load_sensor_data_and_build_cache(
            data=data,
            meta=meta,
            index=index,
            perturbated=False,
            perturbation_translation=0.0,
            perturbation_rotation=0.0,
            cache_key=cache_key_normal,
        )

        sensor_data_perturbated = None
        if self.config.use_sensor_perburtation:
            sensor_data_perturbated = self._load_sensor_data_and_build_cache(
                data=data,
                meta=meta,
                index=index,
                perturbated=True,
                perturbation_translation=meta["perturbation_translation"],
                perturbation_rotation=meta["perturbation_rotation"],
                cache_key=cache_key_perturbated,
            )

        # This part is needed to handle special case: bev_3rd_person_image always uses unaugmented version
        if perturbate_sensor and sensor_data_perturbated is not None:
            selected_data = sensor_data_perturbated
        else:
            selected_data = sensor_data_normal
        return SensorData(
            image=selected_data.image,
            rasterized_lidar=selected_data.rasterized_lidar,
            semantic=selected_data.semantic,
            hdmap=selected_data.hdmap,
            depth=selected_data.depth,
            boxes=selected_data.boxes,
            boxes_waypoints=selected_data.boxes_waypoints,
            boxes_num_waypoints=selected_data.boxes_num_waypoints,
            bev_occupancy=selected_data.bev_occupancy,
            bev_3rd_person_image=sensor_data_normal.bev_3rd_person_image,
            radars=selected_data.radars,
            radar_detections=selected_data.radar_detections,
        )

    @beartype
    def _load_sensor_data_and_build_cache(
        self,
        data: dict,
        meta: dict,
        index: int,
        perturbated: bool,
        perturbation_translation: float,
        perturbation_rotation: float,
        cache_key: CacheKey,
    ) -> SensorData:
        """This function does two main things:
        1) Load and preprocess the data from disk. Since this project is a research repo, the stored
        data format is not always optimal for training. Since function does the heavy lifting
        of preprocessing data.
        2) Since the heavy lifting takes long, we also compress the data and store them so that
        the first step is done only once. The compressed data is also much smaller in size,
        which allows us to store more data in the cache.

        Args:
            data: Data dictionary to be filled.
            meta: Current measurement dictionary containing meta data.
            index: index of the sample in the dataset.
            perturbated: Whether to load the perturbated sensor data or normal data.
            perturbation_translation: Translation augmentation to apply.
            perturbation_rotation: Rotation augmentation to apply.
            cache_key: Key to use for caching the processed data.

        Returns:
            SensorData object with all loaded sensor data.
        """
        bev_3rd_person_image_path = self.bev_3rd_person_images[index]
        if perturbated:
            image_path = self.images_perturbated[index]
            semantic_path = self.semantics_perturbated[index]
            bev_semantic_path = self.bev_semantics_perturbated[index]
            depth_path = self.depth_perturbated[index]
            radar_path = self.radars_perturbated[index]
        else:
            image_path = self.images[index]
            semantic_path = self.semantics[index]
            bev_semantic_path = self.bev_semantics[index]
            depth_path = self.depth[index]
            radar_path = self.radars[index]

        lidar_path = self.lidars[index]  # LiDAR is always the same
        boxes_path = self.bboxes[index]  # Boxes are always the same

        bev_3rd_person_images = image = raw_image_bytes = rasterized_lidar = (
            semantic
        ) = hdmap = depth = boxes = boxes_waypoints = boxes_num_waypoints = (
            bev_occupancy
        ) = radars = None

        if self.config.load_bev_3rd_person_images:
            try:
                bev_3rd_person_images = cv2.imread(
                    str(bev_3rd_person_image_path, encoding="utf-8"),
                    cv2.IMREAD_COLOR,
                )
                bev_3rd_person_images = cv2.cvtColor(
                    bev_3rd_person_images,
                    cv2.COLOR_BGR2RGB,
                )
            except Exception:
                bev_3rd_person_images = np.zeros((800, 600, 3), dtype=np.uint8)
        # Read raw bytes of the JPEG image to avoid re-encoding it later aka. double JPEG artifacts.
        if self.config.use_rgb:
            with open(str(image_path, encoding="utf-8"), "rb") as f:
                raw_image_bytes = f.read()
            image = cv2.imdecode(
                np.frombuffer(raw_image_bytes, np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.config.use_radars:
            radars = np.load(str(radar_path, encoding="utf-8"), allow_pickle=True)
            radars1 = radars["radar1"]
            radars2 = radars["radar2"]
            radars3 = radars["radar3"]
            radars4 = radars["radar4"]
            radars = (radars1, radars2, radars3, radars4)

        # Load LiDAR BEV
        if self.config.use_lidar:
            las_object = laspy.read(str(lidar_path, encoding="utf-8"))
            lidar_pc_i = las_object.xyz
            lidar_timestamp = las_object["time"]
            lidar_pc_i = lidar_pc_i[
                lidar_timestamp < self.config.training_used_lidar_steps
            ]
            rasterized_lidar = rasterize_lidar(
                config=self.config,
                lidar=common_utils.align_lidar(
                    lidar_pc_i,
                    np.array([0, perturbation_translation, 0]),
                    np.deg2rad(perturbation_rotation),
                ),
            )

        # Load semantic
        if self.config.use_semantic:
            semantic = cv2.imread(
                str(semantic_path, encoding="utf-8"),
                cv2.IMREAD_UNCHANGED,
            )
            if not self.config.save_grouped_semantic:
                semantic = self.semantic_converter[
                    semantic
                ]  # Convert to TransFuser labeling

        # Load BEV semantic
        if self.config.use_bev_semantic:
            hdmap = cv2.imread(
                str(bev_semantic_path, encoding="utf-8"),
                cv2.IMREAD_UNCHANGED,
            )  # (256, 256)
            hdmap = self.hdmap_converter[hdmap]  # Convert to TransFuser labeling

        # Load depth
        if self.config.use_depth:
            depth = cv2.imread(str(depth_path, encoding="utf-8"), cv2.IMREAD_UNCHANGED)
            depth = common_utils.decode_depth(
                depth,
            )  # Decode 8-bit depth to metric float

        # Load bounding boxes
        if self.config.detect_boxes:
            json_boxes = common_utils.read_pickle(str(boxes_path, encoding="utf-8"))
            boxes, boxes_waypoints, boxes_num_waypoints = (
                carla_dataset_utils.get_bbox_labels(
                    data,
                    self.config,
                    json_boxes,
                    meta,
                    perturbation_translation,
                    perturbation_rotation,
                )
            )

        # Draw BEV occupancy
        if self.config.use_bev_semantic:
            assert self.config.detect_boxes
            bev_occupancy = carla_dataset_utils.build_bev_occupancy(
                data,
                meta,
                json_boxes,
                self.config,
                y_perturbation=perturbation_translation,
                yaw_perturbation=perturbation_rotation,
            )

        # Create SensorData object with loaded data
        sensor_data = SensorData(
            image=image,
            rasterized_lidar=rasterized_lidar,
            semantic=semantic,
            hdmap=hdmap,
            depth=depth,
            boxes=boxes,
            boxes_waypoints=boxes_waypoints,
            boxes_num_waypoints=boxes_num_waypoints,
            bev_occupancy=bev_occupancy,
            bev_3rd_person_image=bev_3rd_person_images,
            radars=radars,
            radar_detections=None,
        )
        # Compute radar detection labels if radars are enabled
        if self.config.use_radars and radars is not None:
            radar_detections = carla_dataset_utils.parse_radar_detection_labels(
                self.config,
                sensor_data,
            )
            sensor_data.radar_detections = radar_detections

        # Store cache if cache directory is available
        if self.training_session_cache is not None or self.persistent_cache is not None:
            # Use our new compression API for cleaner code
            compressed_sensor_data = sensor_data.compress(
                raw_image_bytes,
                self.config,
                meta,
            )

            # Store CompressedSensorData object directly in cache
            if self.training_session_cache is not None:
                self.training_session_cache[cache_key] = compressed_sensor_data

            if self.persistent_cache is not None:
                self.persistent_cache[cache_key] = compressed_sensor_data

        return sensor_data

    def shuffle(self, epoch):
        rng = default_rng(seed=self.config.seed + epoch)
        np.random.seed(self.config.seed + epoch)
        random.seed(self.config.seed + epoch)

        # Build lists by sampling from each bucket
        self.bev_3rd_person_images = []
        self.images = []
        self.images_perturbated = []
        self.semantics = []
        self.semantics_perturbated = []
        self.bev_semantics = []
        self.bev_semantics_perturbated = []
        self.depth = []
        self.depth_perturbated = []
        self.lidars = []
        self.radars = []
        self.radars_perturbated = []
        self.bboxes = []
        self.metas = []
        self.route_dirs = []
        self.route_indices = []
        self.sample_start = []
        self.bucket_identity = []
        self.global_indices = []

        mixture_ratios = self.bucket_collection.buckets_mixture_per_epoch(epoch)
        for bucket_idx, bucket in enumerate(self.bucket_collection.buckets):
            if bucket_idx in mixture_ratios:
                sampling_rate = mixture_ratios[bucket_idx]
                samples_needed = int(len(bucket) * sampling_rate)
                if samples_needed > 0:
                    if self.random:
                        if samples_needed > len(bucket):
                            indices = rng.choice(
                                len(bucket),
                                size=samples_needed,
                                replace=True,
                            )
                        else:
                            indices = rng.choice(
                                len(bucket.images),
                                size=samples_needed,
                                replace=False,
                            )
                    else:
                        indices = np.arange(samples_needed)

                    self.bev_3rd_person_images.extend(
                        bucket.bev_3rd_person_images[indices],
                    )
                    self.images.extend(bucket.images[indices])
                    self.images_perturbated.extend(bucket.images_perturbated[indices])
                    self.semantics.extend(bucket.semantics[indices])
                    self.semantics_perturbated.extend(
                        bucket.semantics_perturbated[indices],
                    )
                    self.bev_semantics.extend(bucket.hdmap[indices])
                    self.bev_semantics_perturbated.extend(
                        bucket.hdmap_perturbated[indices],
                    )
                    self.depth.extend(bucket.depth[indices])
                    self.depth_perturbated.extend(bucket.depth_perturbated[indices])
                    self.lidars.extend(bucket.lidars[indices])
                    self.radars.extend(bucket.radars[indices])
                    self.radars_perturbated.extend(bucket.radars_perturbated[indices])
                    self.bboxes.extend(bucket.bboxes[indices])
                    self.metas.extend(bucket.metas[indices])
                    self.route_dirs.extend(bucket.route_dirs[indices])
                    self.route_indices.extend(bucket.route_indices[indices])
                    self.sample_start.extend(bucket.sample_start[indices])
                    self.bucket_identity.extend([bucket_idx] * len(indices))
                    self.global_indices.extend(bucket.global_indices[indices])

        if self.rank == 0:
            LOG.info(
                f"Loaded {len(self.images)} images from {len(self.bucket_collection.buckets)} buckets.",
            )

        # Convert to numpy arrays and shuffle if needed
        self.bev_3rd_person_images = np.array(self.bev_3rd_person_images).astype(
            np.string_,
        )
        self.images = np.array(self.images)
        self.images_perturbated = np.array(self.images_perturbated)
        self.semantics = np.array(self.semantics)
        self.semantics_perturbated = np.array(self.semantics_perturbated)
        self.bev_semantics = np.array(self.bev_semantics)
        self.bev_semantics_perturbated = np.array(self.bev_semantics_perturbated)
        self.depth = np.array(self.depth)
        self.depth_perturbated = np.array(self.depth_perturbated)
        self.lidars = np.array(self.lidars)
        self.radars = np.array(self.radars)
        self.radars_perturbated = np.array(self.radars_perturbated)
        self.bboxes = np.array(self.bboxes)
        self.metas = np.array(self.metas)
        self.route_dirs = np.array(self.route_dirs)
        self.route_indices = np.array(self.route_indices)
        self.sample_start = np.array(self.sample_start)
        self.bucket_identity = np.array(self.bucket_identity)
        self.global_indices = np.array(self.global_indices)

        if self.random:
            # Shuffle all arrays together
            indices = np.arange(len(self.images))
            rng.shuffle(indices)
            if self.config.carla_num_samples > 0:
                if self.config.carla_num_samples <= len(indices):
                    indices = indices[: self.config.carla_num_samples]
                else:
                    num_repeat = self.config.carla_num_samples // len(indices)
                    remainder = self.config.carla_num_samples % len(indices)
                    indices = np.concatenate(
                        [indices] * num_repeat + [indices[:remainder]],
                    )

            self.bev_3rd_person_images = self.bev_3rd_person_images[indices]
            self.images = self.images[indices]
            self.images_perturbated = self.images_perturbated[indices]
            self.semantics = self.semantics[indices]
            self.semantics_perturbated = self.semantics_perturbated[indices]
            self.bev_semantics = self.bev_semantics[indices]
            self.bev_semantics_perturbated = self.bev_semantics_perturbated[indices]
            self.depth = self.depth[indices]
            self.depth_perturbated = self.depth_perturbated[indices]
            self.lidars = self.lidars[indices]
            self.radars = self.radars[indices]
            self.radars_perturbated = self.radars_perturbated[indices]
            self.bboxes = self.bboxes[indices]
            self.metas = self.metas[indices]
            self.route_dirs = self.route_dirs[indices]
            self.route_indices = self.route_indices[indices]
            self.sample_start = self.sample_start[indices]
            self.bucket_identity = self.bucket_identity[indices]
            self.global_indices = self.global_indices[indices]

    def __len__(self):
        return self.lidars.shape[0]
