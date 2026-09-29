import logging
from collections import deque

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype

import lead.common.common_utils as common_utils
from lead.common import ransac
from lead.common.kalman_filter import KalmanFilter
from lead.common.route_planner import RoutePlanner
from lead.expert.config_expert import ExpertConfig
from lead.expert.expert_utils import step_cached_property

LOG = logging.getLogger(__name__)


class BaseAgent:
    """Agent class handle basic sensor processing that both expert and student need."""

    @beartype
    def setup(self, sensor_agent: bool = False):
        self.noisy_lat_ref, self.noisy_lon_ref = common_utils.find_gps_ref(
            self._global_plan_world_coord,
            self._global_plan,
        )
        LOG.info(
            "Noisy lat ref: %s, Noisy lon ref: %s",
            self.noisy_lat_ref,
            self.noisy_lon_ref,
        )
        self.config_expert = ExpertConfig()
        self.kalman_filter = KalmanFilter(self.config_expert)

        self.yaws_queue = deque(
            maxlen=self.config_expert.ego_num_temporal_data_points_saved + 1,
        )
        self.speeds_queue = deque(
            maxlen=self.config_expert.ego_num_temporal_data_points_saved + 1,
        )

        self.lidar_pc_queue = deque(
            maxlen=self.config_expert.lidar_stack_size,
        )  # For stacking LiDAR
        self.radar_pc_queue = deque(
            maxlen=2 * self.config_expert.lidar_stack_size,
        )  # For stacking radar as LiDAR

        self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.previous_compass: float | None = None
        self.compass: float | None = None

        # --- Run this once to save Numba cache ---
        ransac.remove_ground(
            np.random.rand(1000, 3),
            self.config_expert,
            parallel=True,
        )  # Pre-compile numba code

        # --- Route planner ---
        self.sensor_agent = sensor_agent
        route_planner_config = self.config_expert
        if sensor_agent:
            route_planner_config = self.config_closed_loop

        self.gps_waypoint_planners_dict: dict[float, RoutePlanner] = {}
        for dist in route_planner_config.tp_distances:
            planner = RoutePlanner(
                dist,
                route_planner_config.route_planner_max_distance,
            )
            planner.set_route(
                self._global_plan,
                True,
                lat_ref=self.noisy_lat_ref,
                lon_ref=self.noisy_lon_ref,
            )
            self.gps_waypoint_planners_dict[dist] = planner

    @beartype
    def tick(self, input_data: dict, use_kalman_filter: bool = True) -> dict:
        # Get the vehicle's speed from sensor
        speed = input_data["speed"][1]["speed"]
        self.speeds_queue.append(speed)

        # Preprocess the compass data from the IMU
        self.previous_compass = self.compass
        self.compass = common_utils.preprocess_compass(
            input_data["imu"][1][-1],
        )  # Range [-pi,pi]
        if self.previous_compass is not None:
            self.compass = np.unwrap([self.previous_compass, self.compass])[
                1
            ]  # Unbounded range
        self.yaws_queue.append(self.compass)

        # Filter the GPS position with Kalman filter
        noisy_gps_pos = common_utils.convert_gps_to_carla(
            input_data["gps"][1],
            self.noisy_lat_ref,
            self.noisy_lon_ref,
        )
        self.filtered_state = self.kalman_filter.step(
            noisy_position=noisy_gps_pos,
            compass=self.compass,
            speed=speed,
            control=self.control,
        )
        self.smooth_history = self.kalman_filter.smooth()
        self.filtered_history = np.array([self.kalman_filter.history_x]).reshape(-1, 4)[
            :,
            :2,
        ]

        # Use filtered or noisy position based on config (for sensor agents at inference time)
        position_for_planner = (
            self.filtered_state[:2] if use_kalman_filter else noisy_gps_pos[:2]
        )
        for planner in self.gps_waypoint_planners_dict.values():
            planner.run_step(np.append(position_for_planner, noisy_gps_pos[2]))

        # Create a dictionary containing the vehicle's state
        input_data.update(
            {
                "theta": self.compass,
                "filtered_state": self.filtered_state,
                "noisy_state": noisy_gps_pos,
                "accel_x": input_data["imu"][1][0],
                "accel_y": input_data["imu"][1][1],
                "accel_z": input_data["imu"][1][2],
                "angular_velocity_x": input_data["imu"][1][3],
                "angular_velocity_y": input_data["imu"][1][4],
                "angular_velocity_z": input_data["imu"][1][5],
                "speed": speed,
            },
        )

        # --- LiDAR ---
        input_data["lidar"] = common_utils.lidar_to_ego_coordinate(
            self.config_expert.lidar_rot_1,
            self.config_expert.lidar_pos_1,
            input_data["lidar1"],
        )
        if self.config_expert.use_two_lidars:
            input_data["lidar"] = np.concatenate(
                (
                    input_data["lidar"],
                    common_utils.lidar_to_ego_coordinate(
                        self.config_expert.lidar_rot_2,
                        self.config_expert.lidar_pos_2,
                        input_data["lidar2"],
                    ),
                ),
                axis=0,
            )
        lidar_x, lidar_y = input_data["lidar"][:, 0], input_data["lidar"][:, 1]
        # Remove lidar points inside ego bounding boxes. We already need LiDAR for expert.
        input_data["lidar"] = input_data["lidar"][
            (np.abs(lidar_x) > self.config_expert.ego_extent_x)
            & (np.abs(lidar_y) > self.config_expert.ego_extent_y)
        ]
        original_lidar_num_points = input_data["lidar"].shape[0]

        # --- Radar ---
        if self.config_expert.use_radars:
            radar_processed = {}
            radar_points_list = []

            for i in range(1, self.config_expert.num_radar_sensors + 1):
                radar = common_utils.radar_points_to_ego(
                    input_data[f"radar{i}"][1],
                    sensor_pos=self.config_expert.radar_calibration[str(i)]["pos"],
                    sensor_rot=self.config_expert.radar_calibration[str(i)]["rot"],
                )
                radar_processed[f"radar{i}"] = radar
                radar_points_list.append(radar[:, :3])

            self.radar_pc_queue.append(np.concatenate(radar_points_list, axis=0))
            input_data.update(radar_processed)

            # Add radar point clouds to LiDAR point cloud if configured
            if self.config_expert.save_radar_pc_as_lidar:
                input_data["lidar"] = np.concatenate(
                    [input_data["lidar"], self.radar_pc_queue[-1]],
                    axis=0,
                )
                if self.config_expert.duplicate_radar_near_ego:
                    radar_near_ego = self.radar_pc_queue[-1][
                        np.linalg.norm(self.radar_pc_queue[-1][:, :2], axis=1)
                        < self.config_expert.duplicate_radar_radius
                    ]
                    radar_near_ego = np.concatenate(
                        [radar_near_ego] * self.config_expert.duplicate_radar_factor,
                        axis=0,
                    )
                    input_data["lidar"] = np.concatenate(
                        [input_data["lidar"], radar_near_ego],
                        axis=0,
                    )

        # Remove lidar points on ground
        lidar_x, lidar_y = input_data["lidar"][:, 0], input_data["lidar"][:, 1]
        if self.config_expert.save_only_non_ground_lidar:
            ground_mask = ransac.remove_ground(
                input_data["lidar"],
                self.config_expert,
                parallel=True,
            )
            input_data["lidar"] = input_data["lidar"][~ground_mask]
            # Also remove ground points from radar_pc_queue if radar points are appended to LiDAR
            if (
                self.config_expert.use_radars
                and self.config_expert.save_radar_pc_as_lidar
            ):
                radar_ground_mask = ground_mask[original_lidar_num_points:]
                self.radar_pc_queue[-1] = self.radar_pc_queue[-1][
                    ~radar_ground_mask[: self.radar_pc_queue[-1].shape[0]]
                ]  # Filter out ground points from the latest radar pc

        if self.config_expert.save_lidar_only_inside_bev:
            lidar_x, _lidar_y, lidar_z = (
                input_data["lidar"][:, 0],
                input_data["lidar"][:, 1],
                input_data["lidar"][:, 2],
            )
            inside_mask = (
                (self.config_expert.min_x_meter <= lidar_x)
                & (lidar_x <= self.config_expert.max_x_meter)
                & (self.config_expert.min_y_meter <= _lidar_y)
                & (_lidar_y <= self.config_expert.max_y_meter)
                & (self.config_expert.min_height_lidar <= lidar_z)
                & (lidar_z <= self.config_expert.max_height_lidar)
            )
            input_data["lidar"] = input_data["lidar"][inside_mask]
        self.lidar_pc_queue.append(input_data["lidar"])

        # --- Process camera images ---
        # CARLA cameras: stitch all configured cameras
        rgb_cameras = []
        config_camera = self.config_expert
        if self.sensor_agent:
            config_camera = self.training_config
        for camera_idx in range(1, config_camera.num_cameras + 1):
            rgb_camera = input_data[f"rgb_{camera_idx}"][1][:, :, :3]
            input_data[f"rgb_{camera_idx}"] = rgb_camera  # individual camera
            rgb_cameras.append(rgb_camera)

        input_data["rgb"] = np.concatenate(rgb_cameras, axis=1)  # stitched version
        return input_data

    @beartype
    @step_cached_property
    def ego_past_positions(self) -> tuple[tuple[float, float], ...]:
        """Return ego past position in current ego frame. Goes from the oldest (i = 0) to current step (i = -1)."""
        smoothed_states = self.smooth_history[:, :2]
        if len(smoothed_states) == 0:
            return []
        current_pos = smoothed_states[-1]
        current_yaw = self.compass
        R_world_to_current = np.array(
            [
                [np.cos(-current_yaw), -np.sin(-current_yaw)],
                [np.sin(-current_yaw), np.cos(-current_yaw)],
            ],
        )
        return tuple(((smoothed_states - current_pos) @ R_world_to_current.T).tolist())

    @beartype
    @step_cached_property
    def ego_past_filtered_state(self) -> tuple[tuple[float, float], ...]:
        """Return ego past position in current ego frame. Goes from the oldest (i = 0) to current step (i = -1)."""
        past_states = self.filtered_history[:, :2]
        if len(past_states) == 0:
            return []
        current_pos = past_states[-1]
        current_yaw = self.compass
        R_world_to_current = np.array(
            [
                [np.cos(-current_yaw), -np.sin(-current_yaw)],
                [np.sin(-current_yaw), np.cos(-current_yaw)],
            ],
        )
        return tuple(((past_states - current_pos) @ R_world_to_current.T).tolist())

    @beartype
    @step_cached_property
    def ego_past_yaws(self) -> tuple[float, ...]:
        """Return ego past yaws in current ego frame. Goes from the oldest (i = 0) to current step (i = -1)."""
        if len(self.yaws_queue) == 0:
            return []
        past_yaws = []
        yaw_now = self.yaws_queue[-1]
        for yaw in self.yaws_queue:
            dyaw = (yaw - yaw_now + np.pi) % (2 * np.pi) - np.pi
            past_yaws.append(dyaw)
        return tuple(past_yaws)

    @beartype
    def accumulate_lidar(self) -> jt.Float[npt.NDArray, "n 4"]:
        """Accumulate LiDAR point clouds over time. Go from the most recent (i = 0) to the oldest (i = num_past - 1).

        Returns:
            The accumulated 3D LiDAR point clouds, where the last dimension is the time stamp.
            The time stamp is 0 for the most recent point cloud, 1 for the previous point cloud, and so on.
        """
        lidar_queue = list(self.lidar_pc_queue)[::-1]
        radar_queue = list(self.radar_pc_queue)[::-1]
        past_positions = self.ego_past_positions[::-1]
        past_yaws = self.ego_past_yaws[::-1]

        # Stacking LiDAR
        lidar_accumulated = []
        for i, lidar_pc in enumerate(lidar_queue):
            if i > 0 and not self.config_expert.lidar_accumulation:
                break
            dx, dy = past_positions[i]
            dyaw = past_yaws[i]
            lidar_pc = common_utils.align_lidar(
                lidar_pc,
                np.array([-dx, -dy, 0.0]),
                -dyaw,
            )

            # Add time stamp as the 4th dimension
            lidar_pc = np.concatenate(
                [lidar_pc, np.ones((lidar_pc.shape[0], 1), dtype=lidar_pc.dtype) * i],
                axis=1,
            )
            lidar_accumulated.append(lidar_pc)
        lidar_accumulated = (
            np.concatenate(lidar_accumulated, axis=0)
            if len(lidar_accumulated) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        # Stacking and densifying radar PC and treat them as LiDAR detections
        radar_accumulated = []
        for i, radar_pc in enumerate(radar_queue):
            if i > 0 and not self.config_expert.lidar_accumulation:
                break
            dx, dy = past_positions[i]
            dyaw = past_yaws[i]

            if self.config_expert.save_radar_pc_as_lidar:
                radar_pc = common_utils.align_lidar(
                    radar_pc,
                    np.array([-dx, -dy, 0.0]),
                    -dyaw,
                )
                if self.config_expert.duplicate_radar_near_ego:
                    radar_near_ego = radar_pc[
                        np.linalg.norm(radar_pc[:, :2], axis=1)
                        < self.config_expert.duplicate_radar_radius
                    ]
                    radar_near_ego = np.concatenate(
                        [radar_near_ego] * self.config_expert.duplicate_radar_factor,
                        axis=0,
                    )
                    radar_pc = np.concatenate([radar_pc, radar_near_ego], axis=0)

            # Add time stamp as the 4th dimension
            radar_pc = np.concatenate(
                [radar_pc, np.ones((radar_pc.shape[0], 1), dtype=radar_pc.dtype) * i],
                axis=1,
            )
            radar_accumulated.append(radar_pc)

        radar_accumulated = (
            np.concatenate(radar_accumulated, axis=0)
            if len(radar_accumulated) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        return np.concatenate([lidar_accumulated, radar_accumulated], axis=0)
