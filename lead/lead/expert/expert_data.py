"""Class that provides data collection functionalities for expert agents."""

import copy
import json
import logging
import os
import pathlib
import queue
import random
import threading
from collections import deque

import carla
import cv2
import laspy
import numpy as np
import torch
from agents.navigation.local_planner import LocalPlanner
from beartype import beartype
from leaderboard.utils.statistics_manager_local import RouteRecord

import lead.common.common_utils as common_utils
import lead.expert.expert_utils as expert_utils
from lead.common import constants, ransac, weathers
from lead.common.constants import (
    CameraPointCloudIndex,
    CarlaSemanticSegmentationClass,
    TargetDataset,
    TransfuserSemanticSegmentationClass,
)
from lead.common.pid_controller import ExpertLongitudinalController
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.expert.expert_base import ExpertBase
from lead.expert.hdmap.chauffeurnet import ObsManager
from lead.expert.hdmap.run_stop_sign import RunStopSign
from lead.expert.kinematic_bicycle_model import KinematicBicycleModel

LOG = logging.getLogger(__name__)


class ExpertData(ExpertBase):
    def expert_setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ):
        """
        Set up the autonomous agent for the CARLA simulation.

        Args:
            path_to_conf_file: Path to the configuration file.
            route_index: Index of the route to follow.
            traffic_manager: The traffic manager object.
        """
        super().expert_setup(path_to_conf_file, route_index, traffic_manager)

        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config_expert)
        self.vehicle_model = KinematicBicycleModel(self.config_expert)

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.waiting_ticks_at_stop_sign = 0
        self.ego_blocked_for_ticks = 0

        # Controllers
        self.perturbation_translation = 0
        self.perturbation_rotation = 0

        # Set up the save path if specified
        if os.environ.get("SAVE_PATH", None) is not None:
            self.save_path = (
                pathlib.Path(os.environ["SAVE_PATH"])
                / f"{self.town}_Rep{self.rep}_{self.route_index}"
            )
            self.save_path.mkdir(parents=True, exist_ok=False)

            if self.config_expert.datagen:
                (self.save_path / "metas").mkdir()

        # Store metas to update acceleration with forward difference after finishing route
        self.metas = []
        self.transform_queue = deque(
            maxlen=self.config_expert.ego_num_temporal_data_points_saved + 1,
        )
        self.negative_id_counter = expert_utils.NegativeIdCounter()
        self.traffic_manager = traffic_manager

        self.cutin_vehicle_starting_position = None

        # Threading for async data saving
        self._save_queue = queue.Queue(maxsize=100)  # Large buffer to handle bursts
        self._save_thread = None
        self._save_thread_stop = threading.Event()

        if (
            self.save_path is not None
            and self.config_expert.datagen
            and not self.config_expert.py123d_data_format
        ):
            (self.save_path / "lidar").mkdir()
            (self.save_path / "rgb").mkdir()
            if self.config_expert.save_camera_pc:
                (self.save_path / "camera_pc").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "camera_pc_perturbated").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "rgb_perturbated").mkdir()
            (self.save_path / "semantics").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "semantics_perturbated").mkdir()
            if self.config_expert.save_depth:
                (self.save_path / "depth").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "depth_perturbated").mkdir()
            (self.save_path / "hdmap").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "hdmap_perturbated").mkdir()
            (self.save_path / "bboxes").mkdir()
            if self.config_expert.save_instance_segmentation:
                (self.save_path / "instance").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "instance_perturbated").mkdir()
            if self.config_expert.use_radars:
                (self.save_path / "radar").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "radar_perturbated").mkdir()

        self.weather_setting = "ClearNoon"
        self.semantics_converter = np.uint8(
            list(constants.SEMANTIC_SEGMENTATION_CONVERTER.values()),
        )

        # Start background saving thread if saving is enabled
        if self.save_path is not None and self.config_expert.datagen:
            self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self._save_thread.start()

    def expert_init(self, hd_map: carla.Map | None):
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map: The map object of the CARLA world.
        """
        super().expert_init(hd_map)

        # Set up the longitudinal controller and command planner
        self._longitudinal_controller = ExpertLongitudinalController(self.config_expert)
        self._command_planner = RoutePlanner(
            self.config_expert.route_planner_min_distance,
            self.config_expert.route_planner_max_distance,
        )
        self._command_planner.set_route(self._global_plan_world_coord)

        self._command_planners_dict = {}
        for dist in self.config_expert.tp_distances:
            planner = RoutePlanner(dist, self.config_expert.route_planner_max_distance)
            planner.set_route(self._global_plan_world_coord)
            self._command_planners_dict[dist] = planner

        # Debug camera setup
        if self.config_expert.save_3rd_person_camera and (
            self.config_expert.debug_mode or self.save_path is not None
        ):
            bp_lib = self.carla_world.get_blueprint_library()
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute(
                "image_size_x",
                self.config_expert.camera_3rd_person_calibration["image_size_x"],
            )
            camera_bp.set_attribute(
                "image_size_y",
                self.config_expert.camera_3rd_person_calibration["image_size_y"],
            )
            camera_bp.set_attribute(
                "fov",
                self.config_expert.camera_3rd_person_calibration["fov"],
            )
            self._3rd_person_camera = self.carla_world.spawn_actor(
                camera_bp,
                self.transform_3rd_person_camera,
            )

            def _save_image(image):
                frame = self.step // self.config_expert.data_save_freq

                def _save(img, path):
                    array = np.frombuffer(img.raw_data, dtype=np.uint8)
                    array = copy.deepcopy(array)
                    array = np.reshape(array, (img.height, img.width, 4))
                    bgr = array[:, :, :3]
                    cv2.imwrite(path, bgr)

                if not self.config_expert.debug_mode or self.save_path is not None:
                    save_path_3rd_person = str(self.save_path / "3rd_person")
                    os.makedirs(save_path_3rd_person, exist_ok=True)
                    _save(
                        image,
                        os.path.join(
                            save_path_3rd_person,
                            f"{str(frame).zfill(4)}.jpg",
                        ),
                    )

            self._3rd_person_camera.listen(_save_image)
        if self.config_expert.datagen:
            self.shuffle_weather()
        jpeg_storage_quality_distribution = (
            self.config_expert.weather_jpeg_compression_quality[self.weather_setting]
        )  # key value: quality maps to probability
        if self.config_expert.jpeg_compression:
            self.jpeg_storage_quality = int(
                np.random.choice(
                    list(jpeg_storage_quality_distribution.keys()),
                    p=list(jpeg_storage_quality_distribution.values()),
                ),
            )
        else:
            self.jpeg_storage_quality = 90
        LOG.info(f"[DataAgent] Chose JPEG storage quality {self.jpeg_storage_quality}")

        obs_config = {
            "width_in_pixels": 256,  # self.config.lidar_resolution_width,
            "pixels_ev_to_bottom": 32 * self.config_expert.pixels_per_meter,
            "pixels_per_meter": self.config_expert.pixels_per_meter_collection,
            "history_idx": [-1],
            "scale_bbox": True,
            "scale_mask_col": 1.0,
            "map_folder": "maps_2ppm_cv",
        }
        if obs_config["width_in_pixels"] != self.config_expert.lidar_width_pixel:
            LOG.warning(
                "The BEV resolution is not the same as the LiDAR resolution. This might lead to unexpected results",
            )

        self.stop_sign_criteria = RunStopSign(self.carla_world)
        self.ss_bev_manager = ObsManager(obs_config, self.config_expert)
        self.ss_bev_manager.attach_ego_vehicle(
            self.ego_vehicle,
            criteria_stop=self.stop_sign_criteria,
        )

        if self.config_expert.perturbate_sensors:
            self.ss_bev_manager_perturbated = ObsManager(obs_config, self.config_expert)
            bb_copy = carla.BoundingBox(
                self.ego_vehicle.bounding_box.location,
                self.ego_vehicle.bounding_box.extent,
            )
            transform_copy = carla.Transform(
                self.ego_vehicle.get_transform().location,
                self.ego_vehicle.get_transform().rotation,
            )
            # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
            self.perturbated_vehicle_dummy = expert_utils.CarlaActorDummy(
                self.ego_vehicle.get_world(),
                bb_copy,
                transform_copy,
                self.ego_vehicle.id,
            )
            self.ss_bev_manager_perturbated.attach_ego_vehicle(
                self.perturbated_vehicle_dummy,
                criteria_stop=self.stop_sign_criteria,
            )

        self._local_planner = LocalPlanner(
            self.ego_vehicle,
            opt_dict={},
            map_inst=self.carla_world_map,
        )
        self.bounding_boxes = []
        ransac.remove_ground(
            np.random.rand(1000, 3),
            self.config_expert,
            parallel=True,
        )  # Pre-compile numba code

        self.initialized = True

    @beartype
    def shuffle_weather(self) -> None:
        LOG.info("Shuffling weather settings")
        # change weather for visual diversity
        weather = self.carla_world.get_weather()

        if self.config_expert.shuffle_weather or self.config_expert.nice_weather:
            if self.config_expert.nice_weather:
                self.weather_setting = "ClearNoon"
                LOG.info(f"Chose nice weather {self.weather_setting}")
            else:
                self.weather_setting = random.choice(
                    list(weathers.WEATHER_SETTINGS.keys()),
                )
                LOG.info(f"Chose random weather {self.weather_setting}")
            LOG.info(f"Chose weather {self.weather_setting}")
            self.weather_parameters: dict[str, float] = weathers.WEATHER_SETTINGS[
                self.weather_setting
            ]

            if "Noon" in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(
                    -45.0,
                    45.0,
                )
            elif "Custom" not in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(
                    -15.0,
                    15.0,
                )

            for randomizing_parameter in ["wind_intensity", "fog_density", "wetness"]:
                if self.weather_parameters[randomizing_parameter] < 30:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -5.0,
                        5.0,
                    )
                elif self.weather_parameters[randomizing_parameter] < 80:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -10.0,
                        10.0,
                    )
                else:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -5.0,
                        5.0,
                    )
                self.weather_parameters[randomizing_parameter] = np.clip(
                    self.weather_parameters[randomizing_parameter],
                    0.0,
                    100.0,
                )

            weather = carla.WeatherParameters(**self.weather_parameters)

            self.carla_world.set_weather(weather)

            # night mode
            vehicles = self.carla_world.get_actors().filter("*vehicle*")
            if expert_utils.get_night_mode(weather):
                for vehicle in vehicles:
                    vehicle.set_light_state(
                        carla.VehicleLightState(
                            carla.VehicleLightState.Position
                            | carla.VehicleLightState.LowBeam,
                        ),
                    )
            else:
                for vehicle in vehicles:
                    vehicle.set_light_state(carla.VehicleLightState.NONE)
        else:
            self.weather_setting = expert_utils.get_weather_name(
                weather,
                self.config_expert,
            )
            self.weather_parameters = expert_utils.weather_parameter_to_dict(weather)

        LOG.info(f"Current weather setting: {self.weather_setting}")
        self.visual_visibility = int(
            weathers.WEATHER_VISIBILITY_MAPPING[self.weather_setting],
        )

    @beartype
    def is_actor_inside_bev(self, actor: carla.Actor) -> bool:
        """
        Check if actor is visible in TransFuser++'s planning visible range.
        This is used to filter out actors that are not visible to TransFuser++'s
        planning module even though they might be visible in the camera.
        """
        actor_in_ego = common_utils.get_relative_transform(
            self.ego_matrix,
            np.array(actor.get_transform().get_matrix()),
        )
        x_ego, y_ego, _ = actor_in_ego
        return bool(
            self.config_expert.min_x_meter - 2
            < x_ego
            < self.config_expert.max_x_meter + 2
            and self.config_expert.min_y_meter - 2
            < y_ego
            < self.config_expert.max_y_meter + 2
            and np.linalg.norm(actor_in_ego) < self.config_expert.bb_save_radius,
        )

    def update_3rd_person_camera(self):
        """
        Track ego with 3rd person camera.
        """
        if hasattr(self, "_3rd_person_camera") and self._3rd_person_camera.is_alive:
            self._3rd_person_camera.set_transform(self.transform_3rd_person_camera)

    def sensors(self):
        """
        Returns a list of sensor specifications for the ego vehicle.

        Each sensor specification is a dictionary containing the sensor type,
        reading frequency, position, and other relevant parameters.

        Returns:
            list: A list of sensor specification dictionaries.
        """
        result = []
        if not self.config_expert.datagen:
            result = [
                {
                    "type": "sensor.opendrive_map",
                    "reading_frequency": 1e-6,
                    "id": "hd_map",
                },
                {
                    "type": "sensor.other.imu",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 0.05,
                    "id": "imu",
                },
                {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
                {
                    "type": "sensor.other.gnss",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 0.01,
                    "id": "gps",
                },
            ]

        self.perturbation_translation, self.perturbation_rotation = (
            expert_utils.sample_sensor_perturbation_parameters(
                config=self.config_expert,
                max_speed_limit_route=self.max_speed_limit_route,
                min_lane_width_route=self.min_lane_width_route,
            )
        )

        # --- Set up sensor rig ---
        if self.save_path is not None and self.config_expert.datagen:
            result += av_sensor_setup(
                self.config_expert,
                perturbation_rotation=self.perturbation_rotation,
                perturbation_translation=self.perturbation_translation,
                lidar=True,
                perturbate=self.config_expert.perturbate_sensors,
                sensor_agent=False,
                radar=self.config_expert.use_radars,
            )
        else:
            result.append(
                {
                    "type": "sensor.lidar.ray_cast",
                    "x": self.config_expert.lidar_pos_1[0],
                    "y": self.config_expert.lidar_pos_1[1],
                    "z": self.config_expert.lidar_pos_1[2],
                    "roll": self.config_expert.lidar_rot_1[0],
                    "pitch": self.config_expert.lidar_rot_1[1],
                    "yaw": self.config_expert.lidar_rot_1[2],
                    "id": "lidar1",
                },
            )
            if self.config_expert.use_two_lidars:
                result.append(
                    {
                        "type": "sensor.lidar.ray_cast",
                        "x": self.config_expert.lidar_pos_2[0],
                        "y": self.config_expert.lidar_pos_2[1],
                        "z": self.config_expert.lidar_pos_2[2],
                        "roll": self.config_expert.lidar_rot_2[0],
                        "pitch": self.config_expert.lidar_rot_2[1],
                        "yaw": self.config_expert.lidar_rot_2[2],
                        "id": "lidar2",
                    },
                )
        return result

    @beartype
    def get_nearby_object(self, actors: carla.ActorList, search_radius: float) -> list:
        """
        Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

        Args:
            actors: A list of actors to search through.
            search_radius: The radius (in meters) around the ego vehicle to search for nearby actors.

        Returns:
            A list of actors within the specified search radius.
        """
        nearby_objects = []
        for actor in actors:
            try:
                trigger_box_global_pos = actor.get_transform().transform(
                    actor.trigger_volume.location,
                )
            except:
                LOG.info(
                    "Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)",
                )
                LOG.info("Skipping this object.")
                continue

            # Convert the vector to a carla.Location for distance calculation
            trigger_box_global_pos = carla.Location(
                x=trigger_box_global_pos.x,
                y=trigger_box_global_pos.y,
                z=trigger_box_global_pos.z,
            )

            # Check if the actor's trigger volume is within the search radius
            if trigger_box_global_pos.distance(self.ego_location) < search_radius:
                nearby_objects.append(actor)

        return nearby_objects

    @beartype
    def tick(self, input_data: dict) -> dict:
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data: Input data containing sensor information.

        Returns:
            A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        input_data = super().tick(input_data)
        self.transform_queue.append(self.ego_vehicle.get_transform())
        if self.config_expert.use_radars and self.config_expert.datagen:
            radar_arrays = []
            for i in range(1, self.config_expert.num_radar_sensors + 1):
                radar_arrays.append(input_data[f"radar{i}"])
            input_data["radar"] = np.concatenate(radar_arrays, axis=0)
        if self.save_path is not None and self.config_expert.datagen:
            if self.config_expert.perturbate_sensors:
                # Process perturbated RGB images for each camera
                rgb_perturbated_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    rgb_perturbated = input_data[f"rgb_{camera_idx}_perturbated"][1][
                        :,
                        :,
                        :3,
                    ]
                    input_data[f"rgb_{camera_idx}_perturbated"] = rgb_perturbated
                    rgb_perturbated_cameras.append(rgb_perturbated)
                input_data["rgb_perturbated"] = np.concatenate(
                    rgb_perturbated_cameras,
                    axis=1,
                )

            if self.config_expert.use_radars and self.config_expert.perturbate_sensors:
                radar_perturbated_dict = {}
                for i in range(1, self.config_expert.num_radar_sensors + 1):
                    radar_perturbated = common_utils.radar_points_to_ego(
                        input_data[f"radar{i}_perturbated"][1],
                        sensor_pos=self.config_expert.radar_calibration[str(i)]["pos"],
                        sensor_rot=self.config_expert.radar_calibration[str(i)]["rot"],
                    )
                    radar_perturbated_dict[f"radar{i}_perturbated"] = radar_perturbated

                input_data.update(radar_perturbated_dict)

            # Instance segmentation - flexible camera processing
            instances = []
            converted_instances = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                instance = cv2.cvtColor(
                    input_data[f"instance_{camera_idx}"][1][:, :, :3],
                    cv2.COLOR_BGR2RGB,
                )
                converted_instance = expert_utils.convert_instance_segmentation(
                    instance,
                )

                input_data[f"instance_{camera_idx}"] = instance
                input_data[f"converted_instance_{camera_idx}"] = converted_instance
                instances.append(instance)
                converted_instances.append(converted_instance)

            input_data["instance"] = np.concatenate(instances, axis=1)

            if self.config_expert.perturbate_sensors:
                instances_perturbated = []
                converted_instances_perturbated = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    instance_perturbated = cv2.cvtColor(
                        input_data[f"instance_{camera_idx}_perturbated"][1][:, :, :3],
                        cv2.COLOR_BGR2RGB,
                    )
                    converted_instance_perturbated = (
                        expert_utils.convert_instance_segmentation(instance_perturbated)
                    )

                    input_data[f"instance_{camera_idx}_perturbated"] = (
                        instance_perturbated
                    )
                    input_data[f"converted_instance_{camera_idx}_perturbated"] = (
                        converted_instance_perturbated
                    )
                    instances_perturbated.append(instance_perturbated)
                    converted_instances_perturbated.append(
                        converted_instance_perturbated,
                    )

                input_data["instance_perturbated"] = np.concatenate(
                    instances_perturbated,
                    axis=1,
                )

            # Standard semantics with some details we don't learn but will be useful to enhance the depth map
            semantics_standard_cameras = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                semantics_standard = input_data[f"semantics_{camera_idx}"][1][:, :, 2]
                input_data[f"semantics_{camera_idx}"] = semantics_standard
                semantics_standard_cameras.append(semantics_standard)
            input_data["semantics"] = np.concatenate(semantics_standard_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                semantics_perturbated_standard_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    semantics_perturbated_standard = input_data[
                        f"semantics_{camera_idx}_perturbated"
                    ][1][:, :, 2]
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        semantics_perturbated_standard
                    )
                    semantics_perturbated_standard_cameras.append(
                        semantics_perturbated_standard,
                    )
                input_data["semantics_perturbated"] = np.concatenate(
                    semantics_perturbated_standard_cameras,
                    axis=1,
                )

            # Depth - flexible camera processing
            depth_cameras = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                depth = expert_utils.convert_depth(
                    input_data[f"depth_{camera_idx}"][1][:, :, :3],
                )
                depth = expert_utils.enhance_depth(
                    depth,
                    input_data[f"semantics_{camera_idx}"],
                    input_data[f"converted_instance_{camera_idx}"],
                )
                input_data[f"depth_{camera_idx}"] = depth
                depth_cameras.append(depth)
            input_data["depth"] = np.concatenate(depth_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                depth_perturbated_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    perturbated_depth = expert_utils.convert_depth(
                        input_data[f"depth_{camera_idx}_perturbated"][1][:, :, :3],
                    )
                    perturbated_depth = expert_utils.enhance_depth(
                        perturbated_depth,
                        input_data[f"semantics_{camera_idx}_perturbated"],
                        input_data[f"converted_instance_{camera_idx}_perturbated"],
                    )
                    input_data[f"depth_{camera_idx}_perturbated"] = perturbated_depth
                    depth_perturbated_cameras.append(perturbated_depth)
                input_data["depth_perturbated"] = np.concatenate(
                    depth_perturbated_cameras,
                    axis=1,
                )

            # Semantics segmentation using first channel of instance segmentation
            # After enhancing the depth map, we use the first channel of instance segmentation which has cleaner labels
            semantics_cameras = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                semantics = input_data[f"converted_instance_{camera_idx}"][..., 0]
                input_data[f"semantics_{camera_idx}"] = semantics
                semantics_cameras.append(semantics)
            input_data["semantics"] = np.concatenate(semantics_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                semantics_perturbated_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    semantics_perturbated = input_data[
                        f"converted_instance_{camera_idx}_perturbated"
                    ][..., 0]
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        semantics_perturbated
                    )
                    semantics_perturbated_cameras.append(semantics_perturbated)
                input_data["semantics_perturbated"] = np.concatenate(
                    semantics_perturbated_cameras,
                    axis=1,
                )

            # Camera point cloud
            camera_pcs = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                cam_config = self.config_expert.camera_calibration[camera_idx]
                input_data[f"semantics_camera_pc_{camera_idx}"] = (
                    expert_utils.semantics_camera_pc(
                        input_data[f"depth_{camera_idx}"],
                        instance=input_data[f"converted_instance_{camera_idx}"],
                        camera_fov=cam_config["fov"],
                        camera_width=cam_config["width"],
                        camera_height=cam_config["height"],
                        camera_pos=cam_config["pos"],
                        camera_rot=cam_config["rot"],
                        perturbation_rotation=0.0,
                        perturbation_translation=0.0,
                        config=self.config_expert,
                    )
                )
                camera_pcs.append(input_data[f"semantics_camera_pc_{camera_idx}"])

            if self.config_expert.perturbate_sensors:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    cam_config = self.config_expert.camera_calibration[camera_idx]
                    input_data[f"semantics_camera_pc_{camera_idx}_perturbated"] = (
                        expert_utils.semantics_camera_pc(
                            input_data[f"depth_{camera_idx}_perturbated"],
                            instance=input_data[
                                f"converted_instance_{camera_idx}_perturbated"
                            ],
                            camera_fov=cam_config["fov"],
                            camera_width=cam_config["width"],
                            camera_height=cam_config["height"],
                            camera_pos=cam_config["pos"],
                            camera_rot=cam_config["rot"],
                            perturbation_rotation=self.perturbation_rotation,
                            perturbation_translation=self.perturbation_translation,
                            config=self.config_expert,
                        )
                    )
            # Concatenate the unprojection together
            input_data["semantics_camera_pc"] = (
                torch.cat(camera_pcs, dim=0).cpu().numpy()
            )

            input_data["semantics_camera_pc_all"] = input_data["semantics_camera_pc"]

        # Bounding box
        input_data["bounding_boxes"] = self.get_bounding_boxes(input_data=input_data)
        self.stored_bounding_boxes_of_this_step = input_data["bounding_boxes"]
        self.id2bb_map = {bb["id"]: bb for bb in input_data["bounding_boxes"]}
        self.id2actor_map = {
            actor.id: actor
            for actor in self.carla_world.get_actors()
            if actor.is_alive and actor.id in self.id2bb_map
        }
        # BEV Semantic
        self.stop_sign_criteria.tick(self.ego_vehicle)
        input_data["hdmap"] = self.ss_bev_manager.get_observation(
            self.close_traffic_lights,
        )["hdmap_classes"]
        if self.config_expert.perturbate_sensors:
            input_data["hdmap_perturbated"] = (
                self.ss_bev_manager_perturbated.get_observation(
                    self.close_traffic_lights,
                )["hdmap_classes"]
            )

        # --- Update semantic segmentation to make cones, traffic warning and special vehicles labels ---
        construction_meshes_id_map = (
            expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
                self.ego_matrix,
                CarlaSemanticSegmentationClass.Dynamic,
                [
                    box
                    for box in input_data["bounding_boxes"]
                    if box.get("type_id") in constants.CONSTRUCTION_MESHES
                ],
                input_data["semantics_camera_pc_all"],
            )
        )
        emergency_meshes_id_map = (
            expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
                self.ego_matrix,
                CarlaSemanticSegmentationClass.Car,
                [
                    box
                    for box in input_data["bounding_boxes"]
                    if box.get("type_id") in constants.EMERGENCY_MESHES
                ],
                input_data["semantics_camera_pc_all"],
                penalize_points_outside=True,
            )
        )
        emergency_meshes_id_map.update(
            expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
                self.ego_matrix,
                CarlaSemanticSegmentationClass.Truck,
                [
                    box
                    for box in input_data["bounding_boxes"]
                    if box.get("type_id") in constants.EMERGENCY_MESHES
                ],
                input_data["semantics_camera_pc_all"],
            ),
        )
        stop_sign_meshes_id_map = (
            expert_utils.match_unreal_engine_ids_to_carla_actors_ids(
                self.ego_matrix,
                CarlaSemanticSegmentationClass.TrafficSign,
                self.get_nearby_object(
                    self.carla_world.get_actors().filter("*traffic.stop*"),
                    self.config_expert.light_radius,
                ),
                input_data["semantics_camera_pc_all"],
            )
        )
        if (
            len(construction_meshes_id_map) > 0
            or len(emergency_meshes_id_map) > 0
            or len(stop_sign_meshes_id_map) > 0
        ):
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                input_data[f"semantics_{camera_idx}"] = (
                    expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}"],
                        input_data.get(f"semantics_{camera_idx}"),
                        construction_meshes_id_map,
                        CarlaSemanticSegmentationClass.ConeAndTrafficWarning,
                    )
                )
                input_data[f"semantics_{camera_idx}"] = (
                    expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}"],
                        input_data.get(f"semantics_{camera_idx}"),
                        emergency_meshes_id_map,
                        CarlaSemanticSegmentationClass.SpecialVehicles,
                    )
                )
                input_data[f"semantics_{camera_idx}"] = (
                    expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}"],
                        input_data.get(f"semantics_{camera_idx}"),
                        stop_sign_meshes_id_map,
                        CarlaSemanticSegmentationClass.StopSign,
                    )
                )

            # Concatenate semantics from all cameras
            semantics_cameras = [
                input_data[f"semantics_{i}"]
                for i in range(1, self.config_expert.num_cameras + 1)
            ]
            input_data["semantics"] = np.concatenate(semantics_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        expert_utils.enhance_semantics_segmentation(
                            input_data[f"converted_instance_{camera_idx}_perturbated"],
                            input_data.get(f"semantics_{camera_idx}_perturbated"),
                            construction_meshes_id_map,
                            CarlaSemanticSegmentationClass.ConeAndTrafficWarning,
                        )
                    )
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        expert_utils.enhance_semantics_segmentation(
                            input_data[f"converted_instance_{camera_idx}_perturbated"],
                            input_data.get(f"semantics_{camera_idx}_perturbated"),
                            emergency_meshes_id_map,
                            CarlaSemanticSegmentationClass.SpecialVehicles,
                        )
                    )
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        expert_utils.enhance_semantics_segmentation(
                            input_data[f"converted_instance_{camera_idx}_perturbated"],
                            input_data.get(f"semantics_{camera_idx}_perturbated"),
                            stop_sign_meshes_id_map,
                            CarlaSemanticSegmentationClass.StopSign,
                        )
                    )

                # Concatenate perturbated semantics from all cameras
                semantics_perturbated_cameras = [
                    input_data[f"semantics_{i}_perturbated"]
                    for i in range(1, self.config_expert.num_cameras + 1)
                ]
                input_data["semantics_perturbated"] = np.concatenate(
                    semantics_perturbated_cameras,
                    axis=1,
                )

        self.tick_data = input_data

        return input_data

    @beartype
    def encode_depth(self, depth: np.ndarray) -> np.ndarray:
        if self.config_expert.save_depth_bits == 8:
            return common_utils.encode_depth_8bit(depth)
        return common_utils.encode_depth_16bit(depth)

    @beartype
    def save_sensors(self, tick_data: dict) -> None:
        frame = self.step // self.config_expert.data_save_freq

        if self.config_expert.eval_expert:
            # Just save RGB synchronously for eval
            cv2.imwrite(
                str(self.save_path / "rgb" / (f"{frame:04}.jpg")),
                tick_data["rgb"],
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_storage_quality],
            )
            LOG.info("Evaluation mode: not saving more sensor data.")
            return

        # Queue data for background thread (deep copy to avoid race conditions)
        save_data = {
            "frame": frame,
            "tick_data": copy.deepcopy(tick_data),
            "jpeg_quality": self.jpeg_storage_quality,
            "lidar_points": self.accumulate_lidar(),  # Get lidar points now
        }

        # Block until queue has space - never drop frames
        self._save_queue.put(save_data)

        # Log queue depth occasionally for monitoring
        if frame % 10 == 0:
            queue_size = self._save_queue.qsize()
            if queue_size > 50:  # Warn if queue is getting large
                LOG.warning(
                    f"Save queue depth: {queue_size}/100 - disk may be falling behind",
                )

    def _save_worker(self):
        """Background thread worker for saving sensor data."""
        while not self._save_thread_stop.is_set():
            try:
                # Wait for data with timeout to allow clean shutdown
                save_data = self._save_queue.get(timeout=1.0)
                self._save_sensors_sync(save_data)
                self._save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                LOG.error(f"Error in save worker thread: {e}")

    def _save_sensors_sync(self, save_data: dict) -> None:
        """Synchronous sensor saving - runs in background thread."""
        frame = save_data["frame"]
        tick_data = save_data["tick_data"]
        jpeg_quality = save_data["jpeg_quality"]
        points = save_data["lidar_points"]

        # CARLA images are already in opencv's BGR format.
        cv2.imwrite(
            str(self.save_path / "rgb" / (f"{frame:04}.jpg")),
            tick_data["rgb"],
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if self.config_expert.perturbate_sensors:
            cv2.imwrite(
                str(self.save_path / "rgb_perturbated" / (f"{frame:04}.jpg")),
                tick_data["rgb_perturbated"],
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )

        # Store camera point clouds
        if self.config_expert.save_camera_pc:
            np.savez_compressed(
                str(self.save_path / "camera_pc" / (f"{frame:04}.npz")),
                tick_data["semantics_camera_pc"],
            )

        semantics = tick_data["semantics"]
        if self.config_expert.save_grouped_semantic:
            semantics = self.semantics_converter[semantics]
        cv2.imwrite(
            str(self.save_path / "semantics" / (f"{frame:04}.png")),
            semantics,
            [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                self.config_expert.png_storage_compression_level,
            ],
        )

        if self.config_expert.perturbate_sensors:
            semantics_perturbated = tick_data["semantics_perturbated"]
            if self.config_expert.save_grouped_semantic:
                semantics_perturbated = self.semantics_converter[semantics_perturbated]
            cv2.imwrite(
                str(self.save_path / "semantics_perturbated" / (f"{frame:04}.png")),
                semantics_perturbated,
                [
                    int(cv2.IMWRITE_PNG_COMPRESSION),
                    self.config_expert.png_storage_compression_level,
                ],
            )

        if self.config_expert.save_depth:
            depth = tick_data["depth"]
            if self.config_expert.save_depth_lower_resolution:
                depth = cv2.resize(
                    depth,
                    (
                        depth.shape[1]
                        // self.config_expert.save_depth_resolution_ratio,
                        depth.shape[0]
                        // self.config_expert.save_depth_resolution_ratio,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            depth_encoded = self.encode_depth(depth)
            cv2.imwrite(
                str(self.save_path / "depth" / f"{frame:04}.png"),
                depth_encoded,
                [
                    int(cv2.IMWRITE_PNG_COMPRESSION),
                    self.config_expert.png_storage_compression_level,
                ],
            )

            if self.config_expert.perturbate_sensors:
                depth_aug = tick_data["depth_perturbated"]
                if self.config_expert.save_depth_lower_resolution:
                    depth_aug = cv2.resize(
                        depth_aug,
                        (
                            depth_aug.shape[1]
                            // self.config_expert.save_depth_resolution_ratio,
                            depth_aug.shape[0]
                            // self.config_expert.save_depth_resolution_ratio,
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                depth_aug_encoded = self.encode_depth(depth_aug)
                cv2.imwrite(
                    str(self.save_path / "depth_perturbated" / f"{frame:04}.png"),
                    depth_aug_encoded,
                    [
                        int(cv2.IMWRITE_PNG_COMPRESSION),
                        self.config_expert.png_storage_compression_level,
                    ],
                )

        if self.config_expert.save_instance_segmentation:
            cv2.imwrite(
                str(self.save_path / "instance" / (f"{frame:04}.png")),
                tick_data["instance"],
                [
                    int(cv2.IMWRITE_PNG_COMPRESSION),
                    self.config_expert.png_storage_compression_level,
                ],
            )
            if self.config_expert.perturbate_sensors:
                cv2.imwrite(
                    str(self.save_path / "instance_perturbated" / (f"{frame:04}.png")),
                    tick_data["instance_perturbated"],
                    [
                        int(cv2.IMWRITE_PNG_COMPRESSION),
                        self.config_expert.png_storage_compression_level,
                    ],
                )

        cv2.imwrite(
            str(self.save_path / "hdmap" / (f"{frame:04}.png")),
            tick_data["hdmap"].astype(np.uint8),
            [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                self.config_expert.png_storage_compression_level,
            ],
        )
        if self.config_expert.perturbate_sensors:
            cv2.imwrite(
                str(self.save_path / "hdmap_perturbated" / (f"{frame:04}.png")),
                tick_data["hdmap_perturbated"].astype(np.uint8),
                [
                    int(cv2.IMWRITE_PNG_COMPRESSION),
                    self.config_expert.png_storage_compression_level,
                ],
            )

        if self.config_expert.use_radars:
            # Prepare radar data for saving dynamically
            radar_save_dict = {}
            for i in range(1, self.config_expert.num_radar_sensors + 1):
                radar_save_dict[f"radar{i}"] = tick_data[f"radar{i}"].astype(np.float16)

            np.savez_compressed(
                self.save_path / "radar" / (f"{frame:04}.npz"),
                **radar_save_dict,
            )
            if self.config_expert.perturbate_sensors:
                # Prepare perturbated radar data for saving dynamically
                radar_perturbated_save_dict = {}
                for i in range(1, self.config_expert.num_radar_sensors + 1):
                    radar_perturbated_save_dict[f"radar{i}"] = tick_data[
                        f"radar{i}_perturbated"
                    ].astype(np.float16)

                np.savez_compressed(
                    self.save_path / "radar_perturbated" / (f"{frame:04}.npz"),
                    **radar_perturbated_save_dict,
                )

        # Specialized LiDAR compression format (using pre-computed points)
        header = laspy.LasHeader(point_format=self.config_expert.point_format)
        header.offsets = np.min(points, axis=0)[:3]
        header.scales = np.array(
            [
                self.config_expert.point_precision_x,
                self.config_expert.point_precision_y,
                self.config_expert.point_precision_z,
            ],
        )
        # Add extra dimension for time
        header.add_extra_dim(laspy.ExtraBytesParams(name="time", type=np.uint8))

        with laspy.open(
            self.save_path / "lidar" / (f"{frame:04}.laz"),
            mode="w",
            header=header,
        ) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(
                points.shape[0],
                header=header,
            )
            point_record.x = points[:, 0]
            point_record.y = points[:, 1]
            point_record.z = points[:, 2]
            point_record["time"] = points[:, 3].astype(np.uint8)
            writer.write_points(point_record)

    @beartype
    def destroy(self, results: RouteRecord = None) -> None:
        """
        Save the collected data and statistics to files, and clean up the data structures.
        This method should be called at the end of the data collection process.

        Args:
            results: Any additional results to be processed or saved.
        """
        # Clean shutdown of background save thread
        if hasattr(self, "_save_thread") and self._save_thread is not None:
            LOG.info("Shutting down background save thread...")
            self._save_thread_stop.set()  # Signal the thread to stop

            # Wait for remaining items in queue to be processed
            try:
                self._save_queue.join()  # Wait for queue to be empty
            except:
                pass

            self._save_thread.join(
                timeout=10,
            )  # Wait up to 10 seconds for clean shutdown
            if self._save_thread.is_alive():
                LOG.warning("Save thread did not shutdown cleanly within timeout")

        if (
            not self.config_expert.eval_expert
            and not self.config_expert.py123d_data_format
        ):
            self._offline_process_data()

        if hasattr(self, "_3rd_person_camera"):
            self._3rd_person_camera.stop()
            self._3rd_person_camera.destroy()

        if results is not None and self.save_path is not None:
            with open(
                os.path.join(self.save_path, "results.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(results.__dict__, f, indent=2)

    def _offline_process_data(self) -> None:
        """
        Offline process the collected data for additional annotations or processing.
        This method is called after the data collection is completed.
        """

        # Re-save metas with privileged information for data filtering later
        # This step is necessary so we can obtain higher qualitative data
        # If any logic is changed, this step should be kept an eye on.
        if (
            (self.save_path is not None)
            and self.config_expert.datagen
            and not self.config_expert.eval_expert
        ):
            metas_dir = self.save_path / "metas"
            delta_t = 1.0 / self.config_expert.fps
            N = len(self.metas)

            # Enhance metas
            for i in range(N):
                step, frame, data = self.metas[i]

                # --- Privileged acceleration and angular velocity, mostly for data filtering ---
                if (
                    step % self.config_expert.data_save_freq != 0
                ):  # Very important. Only override data that was saved.
                    continue

                if i < N - 1:
                    _, _, data_next = self.metas[i + 1]

                    speed_now = data.get("speed", 0.0)
                    speed_next = data_next.get("speed", 0.0)
                    accel = (speed_next - speed_now) / delta_t

                    yaw_now = data.get("theta", 0.0)
                    yaw_next = data_next.get("theta", 0.0)
                    dyaw = (yaw_next - yaw_now + np.pi) % (2 * np.pi) - np.pi
                    rot_speed = dyaw / delta_t
                else:
                    # No future data; make final frame zero
                    accel = 0.0
                    rot_speed = 0.0

                data["privileged_acceleration"] = accel
                data["privileged_rotation_speed"] = rot_speed

                # --- Future speeds: from one step after current to furthest future ---
                future_speeds = []
                for offset in range(
                    0,
                    self.config_expert.ego_num_temporal_data_points_saved + 1,
                ):
                    idx = i + offset
                    if idx < N:
                        _, _, future_data = self.metas[idx]
                        future_speeds.append(future_data.get("speed", 0.0))
                data["future_speeds"] = np.array(future_speeds, dtype=np.float32)

                # --- Future yaws: from one step after current to furthest future ---
                future_yaws = []
                for offset in range(
                    0,
                    self.config_expert.ego_num_temporal_data_points_saved + 1,
                ):
                    idx = i + offset
                    if idx < N:
                        _, _, future_data = self.metas[idx]
                        yaw_future = future_data.get("theta", 0.0)
                        dyaw = (yaw_future - yaw_now + np.pi) % (2 * np.pi) - np.pi
                        future_yaws.append(dyaw)
                data["future_yaws"] = np.array(future_yaws, dtype=np.float32)

                # --- Future positions: from one step after current to furthest future ---
                T_world_to_current_ego = np.linalg.inv(np.array(data["ego_matrix"]))
                future_positions = []
                for offset in range(
                    0,
                    self.config_expert.ego_num_temporal_data_points_saved + 1,
                ):
                    idx = i + offset
                    if idx < N:
                        _, _, future_data = self.metas[idx]
                        T_future = np.array(future_data["ego_matrix"])
                        pos_world = np.append(T_future[:3, 3], 1.0)
                        pos_current_ego = T_world_to_current_ego @ pos_world
                        future_positions.append(pos_current_ego[:3].tolist())
                data["future_positions"] = np.array(future_positions, dtype=np.float32)

                # --- Save metas ---
                common_utils.write_pickle(path=metas_dir / f"{frame:04}.pkl", data=data)

            # Enhance bounding boxes with temporal information
            for i in range(N):
                step, frame, bounding_boxes = self.bounding_boxes[i]

                if step % self.config_expert.data_save_freq != 0:
                    continue

                _, _, data = self.metas[i]
                ego_matrix_current = np.array(data["ego_matrix"])
                T_world_to_current_ego = np.linalg.inv(ego_matrix_current)

                for box in bounding_boxes:
                    box_id = box["id"]

                    if box["class"] not in ["car", "walker"]:
                        continue

                    # --- Future positions and yaws: from one step after current to furthest future ---
                    future_positions = []
                    future_yaws = []
                    for offset in range(
                        0,
                        self.config_expert.other_vehicles_num_temporal_data_points_saved
                        + 1,
                    ):
                        idx = i + offset
                        if idx < N:
                            _, _, future_boxes = self.bounding_boxes[idx]
                            future_box = next(
                                (b for b in future_boxes if b["id"] == box_id),
                                None,
                            )
                            if future_box:
                                T_future = np.array(future_box["matrix"])
                                pos_world = np.append(T_future[:3, 3], 1.0)
                                pos_current_ego = T_world_to_current_ego @ pos_world
                                future_positions.append(pos_current_ego[:2].tolist())

                                rot_world = T_future[:3, :3]
                                heading_vector_world = rot_world @ np.array(
                                    [1.0, 0.0, 0.0],
                                )
                                heading_vector_world = np.append(
                                    heading_vector_world,
                                    0.0,
                                )
                                heading_vector_ego = (
                                    T_world_to_current_ego @ heading_vector_world
                                )
                                yaw = np.arctan2(
                                    heading_vector_ego[1],
                                    heading_vector_ego[0],
                                )
                                future_yaws.append(yaw)

                    box["future_positions"] = np.array(
                        future_positions,
                        dtype=np.float16,
                    )
                    box["future_yaws"] = np.array(future_yaws, dtype=np.float16)

                common_utils.write_pickle(
                    path=self.save_path / "bboxes" / f"{frame:04}.pkl",
                    data=bounding_boxes,
                )

    @beartype
    def get_bounding_boxes(self, input_data: dict) -> list[dict]:
        boxes = []

        ego_transform = self.ego_vehicle.get_transform()
        ego_control = self.ego_vehicle.get_control()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_extent = self.ego_vehicle.bounding_box.extent
        ego_speed = self._get_forward_speed(
            transform=ego_transform,
            velocity=ego_velocity,
        )
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_brake = ego_control.brake

        relative_yaw = 0.0
        relative_pos = common_utils.get_relative_transform(ego_matrix, ego_matrix)

        ego_wp = self.carla_world_map.get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Any,
        )

        # how far is next junction
        next_wps = expert_utils.wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []

        next_road_ids_ego = []
        next_next_road_ids_ego = []
        for wp in next_lane_wps_ego:
            next_road_ids_ego.append(wp.road_id)
            next_next_wps = expert_utils.wps_next_until_lane_end(wp)
            try:
                next_next_lane_wps_ego = next_next_wps[-1].next(1)
                if len(next_next_lane_wps_ego) == 0:
                    next_next_lane_wps_ego = [next_next_wps[-1]]
            except:
                next_next_lane_wps_ego = []
            for wp2 in next_next_lane_wps_ego:
                if wp2.road_id not in next_next_road_ids_ego:
                    next_next_road_ids_ego.append(wp2.road_id)

        # Check for possible vehicle obstacles
        # Retrieve all relevant actors
        self._actors = self.carla_world.get_actors()
        vehicle_list = self._actors.filter("*vehicle*")

        try:
            next_action = self.traffic_manager.get_next_action(self.ego_vehicle)[0]
        except:
            next_action = None

        # --- Start iterating through actors ---
        result = {
            "class": "ego_car",
            "transfuser_semantics_id": int(
                TransfuserSemanticSegmentationClass.UNLABELED,
            ),
            "extent": [ego_dx[0], ego_dx[1], ego_dx[2]],
            "position": [0, 0, 0],
            "yaw": 0.0,
            "num_points": -1,
            "distance": 0,
            "speed": ego_speed,
            "brake": ego_brake,
            "id": int(self.ego_vehicle.id),
            "matrix": ego_transform.get_matrix(),
            "visible_pixels": -1,
        }
        boxes.append(result)

        transfuser_camera_semantics_pc = input_data["semantics_camera_pc"].copy()
        transfuser_camera_semantics_pc_semantics_id = np.array(
            list(constants.SEMANTIC_SEGMENTATION_CONVERTER.values()),
        )[
            transfuser_camera_semantics_pc[
                :,
                CameraPointCloudIndex.UNREAL_SEMANTICS_ID,
            ].astype(np.int32)
        ]
        global_camera_pc = {
            TransfuserSemanticSegmentationClass.VEHICLE: transfuser_camera_semantics_pc[
                (
                    transfuser_camera_semantics_pc_semantics_id
                    == TransfuserSemanticSegmentationClass.VEHICLE
                )
                | (
                    transfuser_camera_semantics_pc_semantics_id
                    == TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE
                )
                | (
                    transfuser_camera_semantics_pc_semantics_id
                    == TransfuserSemanticSegmentationClass.BIKER
                )
            ][:, : CameraPointCloudIndex.Z + 1],
            TransfuserSemanticSegmentationClass.PEDESTRIAN: transfuser_camera_semantics_pc[
                transfuser_camera_semantics_pc_semantics_id
                == TransfuserSemanticSegmentationClass.PEDESTRIAN
            ][:, : CameraPointCloudIndex.Z + 1],
            TransfuserSemanticSegmentationClass.OBSTACLE: transfuser_camera_semantics_pc[
                transfuser_camera_semantics_pc_semantics_id
                == TransfuserSemanticSegmentationClass.OBSTACLE
            ][:, : CameraPointCloudIndex.Z + 1],
        }

        for vehicle in vehicle_list:
            if (
                vehicle.get_location().distance(self.ego_vehicle.get_location())
                < self.config_expert.bb_save_radius
            ):
                if vehicle.id != self.ego_vehicle.id:
                    vehicle_transform = vehicle.get_transform()
                    vehicle_rotation = vehicle_transform.rotation
                    vehicle_matrix = np.array(vehicle_transform.get_matrix())
                    vehicle_control: carla.VehicleControl = vehicle.get_control()
                    vehicle_velocity = vehicle.get_velocity()
                    vehicle_extent = vehicle.bounding_box.extent
                    vehicle_id = vehicle.id
                    vehicle_wp = self.carla_world_map.get_waypoint(
                        vehicle.get_location(),
                        project_to_road=True,
                        lane_type=carla.libcarla.LaneType.Any,
                    )

                    next_wps = expert_utils.wps_next_until_lane_end(vehicle_wp)
                    next_lane_wps = next_wps[-1].next(1)
                    if len(next_lane_wps) == 0:
                        next_lane_wps = [next_wps[-1]]

                    next_next_wps = []
                    for wp in next_lane_wps:
                        next_next_wps = expert_utils.wps_next_until_lane_end(wp)

                    try:
                        next_next_lane_wps = next_next_wps[-1].next(1)
                        if len(next_next_lane_wps) == 0:
                            next_next_lane_wps = [next_next_wps[-1]]
                    except:
                        next_next_lane_wps = []

                    next_road_ids = []
                    for wp in next_lane_wps:
                        if wp.road_id not in next_road_ids:
                            next_road_ids.append(wp.road_id)

                    next_next_road_ids = []
                    for wp in next_next_lane_wps:
                        if wp.road_id not in next_next_road_ids:
                            next_next_road_ids.append(wp.road_id)

                    vehicle_extent_list = [
                        vehicle_extent.x,
                        vehicle_extent.y,
                        vehicle_extent.z,
                    ]
                    yaw = np.deg2rad(vehicle_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(
                        ego_matrix,
                        vehicle_matrix,
                    )
                    vehicle_speed = self._get_forward_speed(
                        transform=vehicle_transform,
                        velocity=vehicle_velocity,
                    )
                    vehicle_brake = vehicle_control.brake
                    vehicle_steer = vehicle_control.steer
                    vehicle_throttle = vehicle_control.throttle

                    distance = np.linalg.norm(relative_pos)

                    try:
                        next_action = self.traffic_manager.get_next_action(vehicle)[0]
                    except:
                        next_action = None

                    vehicle_cuts_in = False
                    if (self.scenario_name == "ParkingCutIn") and vehicle.attributes[
                        "role_name"
                    ] == "scenario":
                        if self.cutin_vehicle_starting_position is None:
                            self.cutin_vehicle_starting_position = (
                                vehicle.get_location()
                            )

                        if (
                            vehicle.get_location().distance(
                                self.cutin_vehicle_starting_position,
                            )
                            > 0.2
                            and vehicle.get_location().distance(
                                self.cutin_vehicle_starting_position,
                            )
                            < 8
                        ):  # to make sure the vehicle drives
                            vehicle_cuts_in = True

                    elif (self.scenario_name == "StaticCutIn") and vehicle.attributes[
                        "role_name"
                    ] == "scenario":
                        if vehicle_speed > 1.0 and abs(vehicle_steer) > 0.2:
                            vehicle_cuts_in = True

                    vehicle_extent_list = [
                        vehicle_extent.x,
                        vehicle_extent.y,
                        vehicle_extent.z,
                    ]
                    yaw = np.deg2rad(vehicle_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(
                        ego_matrix,
                        vehicle_matrix,
                    )
                    vehicle_speed = self._get_forward_speed(
                        transform=vehicle_transform,
                        velocity=vehicle_velocity,
                    )
                    vehicle_brake = vehicle_control.brake
                    vehicle_steer = vehicle_control.steer
                    vehicle_throttle = vehicle_control.throttle

                    # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                    if input_data.get("lidar") is not None:
                        num_in_bbox_points = expert_utils.get_num_points_in_actor(
                            self.ego_vehicle,
                            vehicle,
                            input_data["lidar"],
                            pad=True,
                        )
                    else:
                        num_in_bbox_points = -1

                    if input_data.get("radar") is not None:
                        num_in_bb_radar_points = expert_utils.get_num_points_in_actor(
                            self.ego_vehicle,
                            vehicle,
                            input_data["radar"],
                            pad=True,
                        )
                    else:
                        num_in_bb_radar_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "car",
                        "ego_velocity": expert_utils.get_vehicle_velocity_in_ego_frame(
                            self.ego_vehicle,
                            vehicle,
                        ),
                        "next_action": next_action,
                        "vehicle_cuts_in": vehicle_cuts_in,
                        "road_id": vehicle_wp.road_id,
                        "lane_id": vehicle_wp.lane_id,
                        "lane_type_str": str(vehicle_wp.lane_type),
                        "base_type": vehicle.attributes["base_type"],
                        "transfuser_semantics_id": int(
                            TransfuserSemanticSegmentationClass.VEHICLE,
                        ),
                        "extent": vehicle_extent_list,
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points),
                        "num_radar_points": int(num_in_bb_radar_points),
                        "distance": distance,
                        "speed": vehicle_speed,
                        "brake": vehicle_brake,
                        "steer": vehicle_steer,
                        "throttle": vehicle_throttle,
                        "id": int(vehicle_id),
                        "role_name": vehicle.attributes["role_name"],
                        "type_id": vehicle.type_id,
                        "matrix": vehicle_transform.get_matrix(),
                        "speed_limit": vehicle.get_speed_limit(),
                        "visible_pixels": expert_utils.get_num_points_in_actor(
                            self.ego_vehicle,
                            vehicle,
                            global_camera_pc[
                                TransfuserSemanticSegmentationClass.VEHICLE
                            ],
                            pad=True,
                        ),
                    }
                    boxes.append(result)

        walkers = self._actors.filter("*walker*")
        for walker in walkers:
            if (
                walker.get_location().distance(self.ego_vehicle.get_location())
                < self.config_expert.bb_save_radius
            ):
                walker_transform = walker.get_transform()
                walker_velocity = walker.get_velocity()
                walker_rotation = walker.get_transform().rotation
                walker_matrix = np.array(walker_transform.get_matrix())
                walker_id = walker.id
                walker_extent = walker.bounding_box.extent
                walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
                yaw = np.deg2rad(walker_rotation.yaw)

                relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                relative_pos = common_utils.get_relative_transform(
                    ego_matrix,
                    walker_matrix,
                )

                walker_speed = self._get_forward_speed(
                    transform=walker_transform,
                    velocity=walker_velocity,
                )

                # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                if input_data.get("lidar") is not None:
                    num_in_bbox_points = expert_utils.get_num_points_in_actor(
                        self.ego_vehicle,
                        walker,
                        input_data["lidar"],
                        pad=True,
                    )
                else:
                    num_in_bbox_points = -1

                if input_data.get("radar") is not None:
                    num_in_bb_radar_points = expert_utils.get_num_points_in_actor(
                        self.ego_vehicle,
                        walker,
                        input_data["radar"],
                        pad=True,
                    )
                else:
                    num_in_bb_radar_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    "class": "walker",
                    "ego_velocity": expert_utils.get_vehicle_velocity_in_ego_frame(
                        self.ego_vehicle,
                        walker,
                    ),
                    "role_name": walker.attributes["role_name"],
                    "transfuser_semantics_id": int(
                        TransfuserSemanticSegmentationClass.PEDESTRIAN,
                    ),
                    "extent": walker_extent,
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "num_points": int(num_in_bbox_points),
                    "num_radar_points": int(num_in_bb_radar_points),
                    "distance": distance,
                    "speed": walker_speed,
                    "id": int(walker_id),
                    "matrix": walker_transform.get_matrix(),
                    "visible_pixels": expert_utils.get_num_points_in_actor(
                        self.ego_vehicle,
                        walker,
                        global_camera_pc[
                            TransfuserSemanticSegmentationClass.PEDESTRIAN
                        ],
                        pad=True,
                    ),
                }
                boxes.append(result)

        # Note this only saves static actors, which does not include static background objects
        if self.config_expert.eval_expert:
            LOG.info("Skipping static actor bounding boxes in eval_expert mode.")
        else:
            static_list = self._actors.filter("*static*")
            LOG.debug(f"Found {len(static_list)} static actors in the scene.")
            for static in static_list:
                if (
                    static.get_location().distance(self.ego_vehicle.get_location())
                    < self.config_expert.bb_save_radius
                ):
                    static_transform = static.get_transform()
                    static_velocity = static.get_velocity()
                    static_rotation = static.get_transform().rotation
                    static_matrix = np.array(static_transform.get_matrix())
                    static_id = static.id
                    type_id = static.type_id
                    mesh_path = static.attributes.get("mesh_path", None)
                    static_extent = static.bounding_box.extent
                    static_extent = [static_extent.x, static_extent.y, static_extent.z]
                    if mesh_path is not None and mesh_path in constants.LOOKUP_TABLE:
                        static_extent = constants.LOOKUP_TABLE[mesh_path]
                    if type_id == "static.prop.trafficwarning":
                        static_extent[0], static_extent[1] = (
                            self.config_expert.traffic_warning_bb_size[0],
                            self.config_expert.traffic_warning_bb_size[1],
                        )
                    elif type_id == "static.prop.constructioncone":
                        static_extent[0], static_extent[1] = (
                            self.config_expert.construction_cone_bb_size[0],
                            self.config_expert.construction_cone_bb_size[1],
                        )

                    yaw = np.deg2rad(static_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(
                        ego_matrix,
                        static_matrix,
                    )

                    static_speed = self._get_forward_speed(
                        transform=static_transform,
                        velocity=static_velocity,
                    )

                    # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                    if input_data.get("lidar") is not None:
                        num_in_bbox_points = expert_utils.get_num_points_in_actor(
                            self.ego_vehicle,
                            static,
                            input_data["lidar"],
                            pad=True,
                        )
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "static",
                        "transfuser_semantics_id": int(
                            TransfuserSemanticSegmentationClass.UNLABELED,
                        ),
                        "extent": static_extent,
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points),
                        "distance": distance,
                        "speed": static_speed,
                        "id": int(static_id),
                        "matrix": static_transform.get_matrix(),
                        "type_id": type_id,
                        "mesh_path": mesh_path,
                    }
                    if result["mesh_path"] is not None and "Car" in result["mesh_path"]:
                        result["transfuser_semantics_id"] = int(
                            TransfuserSemanticSegmentationClass.VEHICLE,
                        )
                        result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                            self.ego_vehicle,
                            result,
                            global_camera_pc[
                                TransfuserSemanticSegmentationClass.VEHICLE
                            ],
                            pad=True,
                        )
                    else:
                        result["transfuser_semantics_id"] = int(
                            TransfuserSemanticSegmentationClass.OBSTACLE,
                        )
                        result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                            self.ego_vehicle,
                            result,
                            global_camera_pc[
                                TransfuserSemanticSegmentationClass.OBSTACLE
                            ],
                            pad=True,
                        )

                    boxes.append(result)

        # --- Traffic light ---
        # New logic needed to handle some weird traffic lights in Town12 and Town13
        for (
            traffic_light,
            original_traffic_light_bounding_box,
            traffic_light_state,
            traffic_light_id,
            traffic_light_affects_ego,
        ) in self.close_traffic_lights:
            original_waypoint = self.carla_world_map.get_waypoint(
                original_traffic_light_bounding_box.location,
            )
            waypoint_transform_matrix = np.array(
                original_waypoint.transform.get_matrix(),
            )
            traffic_light_transform_matrix = np.array(
                traffic_light.get_transform().get_matrix(),
            )

            traffic_light_in_waypoint = common_utils.get_relative_transform(
                ego_matrix=waypoint_transform_matrix,
                vehicle_matrix=traffic_light_transform_matrix,
            )

            is_over_head_traffic_light = (
                self.town in ["Town11", "Town12", "Town13", "Town15"]
                and abs(traffic_light_in_waypoint[0]) < 4.0
            )
            is_europe_traffic_light = (
                self.town
                in [
                    "Town01",
                    "Town02",
                    "Town03",
                    "Town04",
                    "Town05",
                    "Town06",
                    "Town07",
                    "Town10HD",
                ]
                and abs(traffic_light_in_waypoint[0]) < 4.0
            )
            if traffic_light_affects_ego:
                red_light_stop_waypoints = expert_utils.get_stop_waypoints(
                    self.ego_wp,
                    traffic_light,
                )

                # Create bounding boxes for each additional lane
                for i, red_light_stop_waypoint in enumerate(red_light_stop_waypoints):
                    # Create bounding box for this waypoint
                    duplicated_traffic_light_bounding_box = (
                        expert_utils.create_bounding_box_for_waypoint(
                            original_traffic_light_bounding_box,
                            red_light_stop_waypoint,
                        )
                    )

                    traffic_light_extent = [
                        duplicated_traffic_light_bounding_box.extent.x,
                        duplicated_traffic_light_bounding_box.extent.y,
                        duplicated_traffic_light_bounding_box.extent.z,
                    ]

                    # Keep original rotation/yaw, only change position
                    traffic_light_transform = carla.Transform(
                        duplicated_traffic_light_bounding_box.location,
                        original_traffic_light_bounding_box.rotation,
                    )
                    traffic_light_rotation = traffic_light_transform.rotation
                    traffic_light_matrix = np.array(
                        traffic_light_transform.get_matrix(),
                    )
                    yaw = np.deg2rad(traffic_light_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(
                        ego_matrix,
                        traffic_light_matrix,
                    )

                    distance = np.linalg.norm(relative_pos)

                    # Keep the original distance_to_physical_traffic_light
                    distance_traffic_light_to_bounding_box = np.linalg.norm(
                        np.array(
                            [
                                traffic_light.get_transform().location.x
                                - duplicated_traffic_light_bounding_box.location.x,
                                traffic_light.get_transform().location.y
                                - duplicated_traffic_light_bounding_box.location.y,
                            ],
                        ),
                    )

                    # Duplicated bounding box result
                    if i == 0:
                        result = {
                            "class": "traffic_light",
                            "transfuser_semantics_id": int(
                                TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT,
                            ),
                            "extent": traffic_light_extent,
                            "position": [
                                relative_pos[0],
                                relative_pos[1],
                                relative_pos[2],
                            ],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            "id": int(traffic_light_id),  # Keep original ID
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_transform.get_matrix(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": False,
                            "same_lane_as_ego": red_light_stop_waypoint.lane_id
                            == ego_wp.lane_id,
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    else:
                        result = {
                            "class": "traffic_light",
                            "transfuser_semantics_id": int(
                                TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT,
                            ),
                            "extent": traffic_light_extent,
                            "position": [
                                relative_pos[0],
                                relative_pos[1],
                                relative_pos[2],
                            ],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            "id": self.negative_id_counter(
                                traffic_light.id,
                            ),  # Use negative ID counter for duplicates
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_transform.get_matrix(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": True,
                            "same_lane_as_ego": red_light_stop_waypoint.lane_id
                            == ego_wp.lane_id,
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    boxes.append(result)

        for stop_sign in self.close_stop_signs:
            stop_sign_extent = [
                stop_sign[0].extent.x,
                stop_sign[0].extent.y,
                stop_sign[0].extent.z,
            ]

            stop_sign_transform = carla.Transform(
                stop_sign[0].location,
                stop_sign[0].rotation,
            )
            stop_sign_rotation = stop_sign_transform.rotation
            stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
            yaw = np.deg2rad(stop_sign_rotation.yaw)

            relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
            relative_pos = common_utils.get_relative_transform(
                ego_matrix,
                stop_sign_matrix,
            )

            distance = np.linalg.norm(relative_pos)

            result = {
                "class": "stop_sign",
                "transfuser_semantics_id": int(
                    TransfuserSemanticSegmentationClass.UNLABELED,
                ),
                "extent": stop_sign_extent,
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "distance": distance,
                "id": int(stop_sign[1]),
                "affects_ego": stop_sign[2],
                "matrix": stop_sign_transform.get_matrix(),
            }
            boxes.append(result)

        # Others static meshes that dont belong to an actors but are still relevant for perception
        if self.config_expert.eval_expert:
            LOG.info("Skipping static actor bounding boxes in eval_expert mode.")
        else:
            static_parking_id_start = 1e8
            found = 0
            existed_bboxes_ids = set([box["id"] for box in boxes])
            for bb_class in [carla.CityObjectLabel.Car]:
                for i, vehicle_bounding_box in enumerate(
                    self.carla_world.get_level_bbs(bb_class),
                ):
                    extent = vehicle_bounding_box.extent
                    location = vehicle_bounding_box.location
                    rotation = vehicle_bounding_box.rotation
                    matrix = carla.Transform(location, rotation).get_matrix()
                    relative_pos = common_utils.get_relative_transform(
                        ego_matrix,
                        np.array(matrix),
                    )
                    distance = np.linalg.norm(relative_pos)

                    if distance > self.config_expert.bb_save_radius:
                        continue
                    relative_yaw = common_utils.normalize_angle(
                        np.deg2rad(rotation.yaw) - ego_yaw,
                    )
                    result = {
                        "class": "static_prop_car",
                        "transfuser_semantics_id": int(
                            TransfuserSemanticSegmentationClass.VEHICLE,
                        ),
                        "extent": [extent.x, extent.y, extent.z],
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "distance": distance,
                        "id": int(static_parking_id_start + i),
                        "matrix": matrix,
                    }

                    # Since we iterate every bounding box in the world, we need to make sure that we dont duplicate anything
                    too_close = False
                    for other in list(self._actors.filter("*vehicle*")) + list(
                        self._actors.filter("*static*"),
                    ):
                        if other.id not in existed_bboxes_ids:
                            continue
                        bb = (
                            other.bounding_box
                        )  # local-space BB (relative to actor origin)
                        global_location = other.get_transform().transform(bb.location)
                        global_location = carla.Location(
                            x=global_location.x,
                            y=global_location.y,
                            z=global_location.z,
                        )
                        # copy into a new BoundingBox
                        other_bb = carla.BoundingBox(global_location, bb.extent)
                        other_bb.rotation = bb.rotation
                        if expert_utils.check_obb_intersection(
                            vehicle_bounding_box,
                            other_bb,
                        ):
                            too_close = True
                            break

                    if not too_close:
                        if input_data.get("lidar") is not None:
                            result["num_points"] = expert_utils.get_num_points_in_bbox(
                                self.ego_vehicle,
                                result,
                                input_data["lidar"],
                                pad=True,
                            )
                        else:
                            result["num_points"] = -1

                        result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                            self.ego_vehicle,
                            result,
                            global_camera_pc[
                                TransfuserSemanticSegmentationClass.VEHICLE
                            ],
                            pad=True,
                        )

                        boxes.append(result)
                        found += 1

        # Filter bounding boxes with duplicate id
        bounding_box_ids = set()
        filtered_bounding_boxes = []
        for box in boxes:
            if box["id"] not in bounding_box_ids:
                bounding_box_ids.add(box["id"])
                filtered_bounding_boxes.append(box)
        # Sort by distances to ego
        filtered_bounding_boxes = sorted(
            filtered_bounding_boxes,
            key=lambda x: x["distance"],
        )

        return filtered_bounding_boxes

    @beartype
    def visualize_ego_bb(self, ego_bb_global: carla.BoundingBox):
        ego_vehicle_transform = self.ego_vehicle.get_transform()
        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(
            self.ego_vehicle.bounding_box.location,
        )
        ego_bb_global = carla.BoundingBox(
            center_ego_bb_global,
            self.ego_vehicle.bounding_box.extent,
        )
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.config_expert.visualize_bounding_boxes:
            self.carla_world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=self.config_expert.ego_vehicle_bb_color,
                life_time=self.config_expert.draw_life_time,
            )

    @beartype
    def visualize_lead_and_trailing_vehicles(self):
        if self.config_expert.visualize_internal_data:
            vehicle_list = ...

            leading_vehicle_ids = (
                self.privileged_route_planner.compute_leading_vehicles(
                    vehicle_list,
                    self.ego_vehicle.id,
                )
            )
            trailing_vehicle_ids = (
                self.privileged_route_planner.compute_trailing_vehicles(
                    vehicle_list,
                    self.ego_vehicle.id,
                )
            )

            for vehicle in vehicle_list:
                if vehicle.id in leading_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Leading Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.leading_vehicle_color,
                    )
                elif vehicle.id in trailing_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Trailing Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.trailing_vehicle_color,
                    )

    @beartype
    def visualize_forecasted_bounding_boxes(
        self,
        predicted_bounding_boxes: dict[int, list[carla.BoundingBox]],
    ):
        if self.config_expert.visualize_bounding_boxes:
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for (
                _actor_idx,
                actors_forecasted_bounding_boxes,
            ) in predicted_bounding_boxes.items():
                for bb in actors_forecasted_bounding_boxes:
                    color = self.config_expert.other_vehicles_forecasted_bbs_color
                    if (
                        _actor_idx in dangerous_adversarial_actors_ids
                        or _actor_idx in safe_adversarial_actors_ids
                    ):
                        color = self.config_expert.adversarial_color
                    self.carla_world.debug.draw_box(
                        box=bb,
                        rotation=bb.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=self.config_expert.draw_life_time,
                    )

                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if (
                        vehicle_id in self.leading_vehicle_ids
                        and not self.near_lane_change
                    ):
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.leading_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )
                    elif vehicle_id in self.trailing_vehicle_ids:
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.trailing_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )

    @beartype
    def visualize_pedestrian_bounding_boxes(
        self,
        nearby_pedestrians_bbs: list[list[carla.BoundingBox]],
    ):
        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.config_expert.visualize_bounding_boxes:
            for bbs in nearby_pedestrians_bbs:
                for bbox in bbs:
                    self.carla_world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=self.config_expert.pedestrian_forecasted_bbs_color,
                        life_time=self.config_expert.draw_life_time,
                    )

    @beartype
    def visualize_traffic_lights(
        self,
        traffic_light: carla.TrafficLight,
        wp: carla.Waypoint,
        bounding_box: carla.BoundingBox,
    ):
        if self.config_expert.visualize_traffic_lights_bounding_boxes:
            if traffic_light.state == carla.TrafficLightState.Red:
                color = self.config_expert.red_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Yellow:
                color = self.config_expert.yellow_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Green:
                color = self.config_expert.green_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Off:
                color = self.config_expert.off_traffic_light_color
            else:  # unknown
                color = self.config_expert.unknown_traffic_light_color

            self.carla_world.debug.draw_box(
                box=bounding_box,
                rotation=bounding_box.rotation,
                thickness=0.1,
                color=color,
                life_time=0.051,
            )

            self.carla_world.debug.draw_point(
                wp.transform.location
                + carla.Location(z=traffic_light.trigger_volume.location.z),
                size=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
            )

            self.carla_world.debug.draw_box(
                box=traffic_light.bounding_box,
                rotation=traffic_light.bounding_box.rotation,
                thickness=0.1,
                color=color,
                life_time=0.051,
            )

    @beartype
    def visualize_stop_signs(
        self,
        bounding_box_stop_sign: carla.BoundingBox,
        affects_ego: bool,
    ):
        if self.config_expert.visualize_bounding_boxes:
            color = carla.Color(0, 1, 0) if affects_ego else carla.Color(1, 0, 0)
            self.carla_world.debug.draw_box(
                box=bounding_box_stop_sign,
                rotation=bounding_box_stop_sign.rotation,
                thickness=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
            )

    @expert_utils.step_cached_property
    def visibility_range_camera_1(self):
        if self.config_expert.target_dataset in [
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
        ]:
            return expert_utils.compute_camera_occlusion_score(
                self.tick_data["semantics_camera_pc_1"],
            )
        return 1.0

    @expert_utils.step_cached_property
    def visibility_range_camera_2(self):
        if self.config_expert.target_dataset in [
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
        ]:
            return expert_utils.compute_camera_occlusion_score(
                self.tick_data["semantics_camera_pc_2"],
            )
        return 1.0

    @expert_utils.step_cached_property
    def visibility_range_camera_3(self):
        if self.config_expert.target_dataset in [
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
        ]:
            return expert_utils.compute_camera_occlusion_score(
                self.tick_data["semantics_camera_pc_3"],
            )
        return 1.0
