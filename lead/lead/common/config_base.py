import os
import sys
from typing import Any

from omegaconf import OmegaConf

from lead.common import constants
from lead.common.constants import WAYMO_E2E_INTRINSIC, TargetDataset


class BaseConfig:
    """Base configuration class that provides default settings for both data collection and model training.

    See more details at https://ln2697.github.io/lead/docs/config_system.html
    """

    # If true, run code in debug mode with settings that allow for
    # faster iteration and easier debugging (e.g., lower resolution, fewer data points saved, etc.)
    debug_mode = False

    @property
    def target_dataset(self):
        raise NotImplementedError(
            "Subclasses must implement the target_dataset property.",
        )

    @property
    def lead_project_root(self):
        """Get the LEAD project root directory from environment variable or default to current working directory."""
        return os.getenv("LEAD_PROJECT_ROOT")

    # --- Autopilot ---
    # Frame rate used for the bicycle models in the autopilot
    bicycle_frame_rate = 20
    # Number of future route points we save per step
    num_route_points_saved = 50
    # Points sampled per meter when interpolating route
    points_per_meter = 10
    # Pixels per meter used in the semantic segmentation map during data collection
    # On Town 13 2.0 is the highest that opencv can handle
    pixels_per_meter_collection = 2.0
    # Maximum acceleration in meters per tick (1.9 m/tick)
    longitudinal_max_accelerations = 1.89

    # --- Kinematic Bicycle Model ---
    # Time step for the model (20 frames per second)
    time_step = 1.0 / 20.0
    # Distance from the rear axle to the front axle of the vehicle
    front_wheel_base = -0.090769015
    # Distance from the rear axle to the center of the rear wheels
    rear_wheel_base = 1.4178275
    # Gain factor for steering angle to wheel angle conversion
    steering_gain = 0.36848336
    # Deceleration rate when braking (m/s^2) of other vehicles
    brake_acceleration = -4.952399
    # Acceleration rate when throttling (m/s^2) of other vehicles
    throttle_acceleration = 0.5633837
    # Minimum throttle value that has an affect during forecasting the ego vehicle
    throttle_threshold_during_forecasting = 0.3

    # --- Augmentation and Misc ---
    # Frequency (in steps) at which data is saved during data collection
    data_save_freq = 5
    # If true enable camera perturbation during data collection
    perturbate_sensors = True
    # Safety translation perturbation penalty for default scenarios
    default_safety_translation_perturbation_penalty = 0.25
    # Safety translation perturbation penalty for urban scenarios with low speed limits
    urban_safety_translation_perturbation_penalty = 0.4

    # Minimum value by which the perturbated camera is shifted left and right
    camera_translation_perturbation_min = 0.1
    # Maximum value by which the perturbated camera is shifted left and right
    camera_translation_perturbation_max = 1.0

    # Minimum value by which the perturbated camera is rotated around the yaw (degrees)
    camera_rotation_perturbation_min = 5.0
    # Maximum value by which the perturbated camera is rotated around the yaw (degrees)
    camera_rotation_perturbation_max = 12.5
    # Epsilon threshold to ignore rotation augmentation around 0.0 degrees
    camera_rotation_epsilon = 0.5

    # --- LiDAR Compression ---
    # LARS point format used for storing LiDAR data
    point_format = 0
    # Precision up to which LiDAR points are stored (x, y, z coordinates)
    point_precision_x = point_precision_y = point_precision_z = 0.1
    # Maximum height threshold for LiDAR points (meters, points above are discarded)
    max_height_lidar = 10.0
    # Minimum height threshold for LiDAR points (meters, points below are discarded)
    min_height_lidar = -4.0

    # --- Sensor Configuration ---
    # If true use two LiDARs or one
    use_two_lidars = True

    # x, y, z mounting position of the first LiDAR
    @property
    def lidar_pos_1(self):
        return [0.0, 0.0, 2.5]

    # Roll, pitch, yaw rotation of first LiDAR (degrees)
    @property
    def lidar_rot_1(self):
        return [0.0, 0.0, -90.0]

    # x, y, z mounting position of the second LiDAR
    @property
    def lidar_pos_2(self):
        return [0.0, 0.0, 2.5]

    # Roll, pitch, yaw rotation of second LiDAR (degrees)
    @property
    def lidar_rot_2(self):
        return [0.0, 0.0, -270.0]

    # If true accumulate LiDAR data over multiple frames
    @property
    def lidar_accumulation(self):
        return True

    # --- Camera Configuration ---
    @property
    def num_cameras(self):
        """Number of cameras based on the target dataset."""
        return {
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS: 6,
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS: 3,
            TargetDataset.CARLA_LEADERBOARD2_1CAMERA: 1,
            TargetDataset.CARLA_PY123D_6CAMERAS: 6,
            TargetDataset.NAVSIM_4CAMERAS: 4,
            TargetDataset.WAYMO_E2E_2025_3CAMERAS: 3,
        }[self.target_dataset]

    @property
    def camera_calibration(self):
        """Camera calibration configuration with positions, rotations, and sensor parameters"""
        if self.target_dataset == TargetDataset.CARLA_LEADERBOARD2_6CAMERAS:
            return {
                1: {
                    "pos": [0.0, -0.3, 2.25],
                    "rot": [0.0, 0.0, -57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                2: {
                    "pos": [0.25, 0.0, 2.25],
                    "rot": [0.0, 0.0, 0.0],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                3: {
                    "pos": [0.0, 0.3, 2.25],
                    "rot": [0.0, 0.0, 57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                4: {
                    "pos": [-0.30, 0.3, 2.25],
                    "rot": [0.0, 0.0, 180 - 57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                5: {
                    "pos": [-0.55, 0.0, 2.25],
                    "rot": [0.0, 0.0, 180.0],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                6: {
                    "pos": [-0.30, -0.3, 2.25],
                    "rot": [0.0, 0.0, -180 + 57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
            }
        elif self.target_dataset == TargetDataset.CARLA_LEADERBOARD2_3CAMERAS:
            return {
                1: {
                    "pos": [0.1, -0.35, 2.25],
                    "rot": [0.0, 0.0, -54.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                2: {
                    "pos": [0.35, 0.0, 2.25],
                    "rot": [0.0, 0.0, 0.0],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                3: {
                    "pos": [0.1, 0.35, 2.25],
                    "rot": [0.0, 0.0, 54.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
            }
        elif self.target_dataset == TargetDataset.CARLA_LEADERBOARD2_1CAMERA:
            return {
                1: {
                    "pos": [-1.5, 0.0, 2.0],
                    "rot": [0.0, 0.0, 0.0],
                    "width": 1024,
                    "height": 512,
                    "cropped_height": 384,
                    "fov": 110,
                },
            }
        elif self.target_dataset == TargetDataset.CARLA_PY123D_6CAMERAS:
            # NOTE: Use six cameras inspired by nuScenes rig. In contract
            # to the nuScenes resolution 1600x900, we use a downsampled 800x450 resolution
            return {
                # CAM_FRONT
                1: {
                    "pos": [0.32, 0.0, 1.6],
                    "rot": [0.0, 0.0, 0.0],
                    "width": 800,
                    "height": 450,
                    "cropped_height": 450,
                    "fov": 65,
                },
                # CAM_FRONT_LEFT
                2: {
                    "pos": [0.15, -0.49, 1.6],
                    "rot": [0.0, 0.0, -55.0],
                    "width": 800,
                    "height": 450,
                    "cropped_height": 450,
                    "fov": 65,
                },
                # CAM_BACK_LEFT
                3: {
                    "pos": [-0.35, -0.48, 1.6],
                    "rot": [0.0, 0.0, -110.0],
                    "width": 800,
                    "height": 450,
                    "cropped_height": 450,
                    "fov": 65,
                },
                # CAM_FRONT_RIGHT
                4: {
                    "pos": [0.15, 0.49, 1.6],
                    "rot": [0.0, 0.0, 55.0],
                    "width": 800,
                    "height": 450,
                    "cropped_height": 450,
                    "fov": 65,
                },
                # CAM_BACK_RIGHT
                5: {
                    "pos": [-0.35, 0.48, 1.6],
                    "rot": [0.0, 0.0, 110.0],
                    "width": 800,
                    "height": 450,
                    "cropped_height": 450,
                    "fov": 65,
                },
                # CAM_BACK
                6: {
                    "pos": [-1.36, 0.0, 1.6],
                    "rot": [0.0, 0.0, -180.0],
                    "width": 800,
                    "height": 450,
                    "cropped_height": 450,
                    "fov": 90,
                },
            }
        elif self.target_dataset == TargetDataset.NAVSIM_4CAMERAS:
            return {
                1: constants.NUPLAN_CAMERA_CALIBRATION["CAM_L0"],
                2: constants.NUPLAN_CAMERA_CALIBRATION["CAM_F0"],
                3: constants.NUPLAN_CAMERA_CALIBRATION["CAM_R0"],
                4: constants.NUPLAN_CAMERA_CALIBRATION["CAM_B0"],
            }
        elif self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            from lead.expert.expert_utils import waymo_e2e_camera_setting_to_carla

            return {
                1: waymo_e2e_camera_setting_to_carla(
                    WAYMO_E2E_INTRINSIC,
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"]["extrinsic"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"]["width"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"]["height"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"][
                        "cropped_height"
                    ],
                ),
                2: waymo_e2e_camera_setting_to_carla(
                    WAYMO_E2E_INTRINSIC,
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["extrinsic"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["width"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["height"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["cropped_height"],
                ),
                3: waymo_e2e_camera_setting_to_carla(
                    WAYMO_E2E_INTRINSIC,
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"]["extrinsic"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"]["width"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"]["height"],
                    constants.WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"][
                        "cropped_height"
                    ],
                ),
            }
        raise ValueError(f"Unsupported target dataset: {self.target_dataset}")

    # --- Radar Configuration ---
    num_radar_sensors = 4

    @property
    def radar_calibration(self):
        """Radar sensor calibration configuration with positions and orientations.

        Currently supports only 4 radar sensors. For other numbers, raises an error.
        """
        return {
            "1": {
                "pos": [2.6, 0, 0.60],  # front-left
                "rot": [0.0, 0.0, -45.0],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
            "2": {
                "pos": [2.6, 0, 0.60],  # front
                "rot": [0.0, 0.0, 45.0],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
            "3": {
                "pos": [-2.6, 0, 0.60],  # front-right
                "rot": [0.0, 0.0, 135],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
            "4": {
                "pos": [-2.6, 0, 0.60],  # rear
                "rot": [0.0, 0.0, 225],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
        }

    # If true use radar sensors
    use_radars = True
    # If true save radar point cloud as LiDAR format
    save_radar_pc_as_lidar = True
    # If true save LiDAR data only inside bird's eye view area
    save_lidar_only_inside_bev = True
    # If true duplicate radar points near ego vehicle for better detection
    duplicate_radar_near_ego = True
    # Radius around ego vehicle for radar point duplication
    duplicate_radar_radius = 32
    # Multiplication factor for radar point duplication
    duplicate_radar_factor = 5

    # --- Data Storage ---
    # If true save depth images at lower resolution
    save_depth_lower_resolution = True

    @property
    def save_depth_resolution_ratio(self):
        """Resolution reduction ratio for depth image storage."""
        if not self.debug_mode:
            return 4
        return 4

    # Number of bits used for saving depth images
    save_depth_bits = 8
    # If true save only non-ground LiDAR points
    save_only_non_ground_lidar = True
    # If true save semantic segmentation in grouped format
    save_grouped_semantic = True

    # --- Temporal Data ---
    # Number of temporal data points saved for ego vehicle
    ego_num_temporal_data_points_saved = 60
    # Number of temporal data points saved for other vehicles
    other_vehicles_num_temporal_data_points_saved = 40

    # --- Agent Configuration ---
    # Simulator frames per second
    carla_fps = 20
    # CARLA frame rate in seconds
    carla_frame_rate = 1.0 / carla_fps
    # IoU threshold used for non-maximum suppression on bounding box predictions
    iou_threshold_nms = 0.2
    # Minimum distance to route planner waypoints
    route_planner_min_distance = 7.5
    # Maximum distance to route planner waypoints
    route_planner_max_distance = 50.0
    # Minimum distance to waypoint in dense route that expert follows
    dense_route_planner_min_distance = 2.4
    # Initial frames delay for CARLA initialization
    inital_frames_delay = 1

    # Target point distances for route planning (in meters). From 3.0m to 10.0m with step of 0.25m
    tp_distances = [i / 100 for i in range(300, 1001, 25)]
    # Extent of the ego vehicle's bounding box in x direction
    ego_extent_x = 2.4508416652679443
    # Extent of the ego vehicle's bounding box in y direction
    ego_extent_y = 1.0641621351242065
    # Extent of the ego vehicle's bounding box in z direction
    ego_extent_z = 0.7553732395172119

    # Minimum z coordinate of the safety box
    safety_box_z_min = 0.5
    # Maximum z coordinate of the safety box
    safety_box_z_max = 1.5

    # --- Safety Box Properties ---
    @property
    def safety_box_y_min(self):
        """Minimum y coordinate of the safety box relative to ego vehicle."""
        return -self.ego_extent_y * 0.8

    @property
    def safety_box_y_max(self):
        """Maximum y coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_y * 0.8

    @property
    def safety_box_x_min(self):
        """Minimum x coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_x

    @property
    def safety_box_x_max(self):
        """Maximum x coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_x + 2.5

    @property
    def is_on_tcml(self):
        """Check if running on Training Center for Machine Learning of Tübingen."""
        return "TCML" in os.environ and os.environ["TCML"] is not None

    # --- Configuration Parsing Methods ---
    def load_from_args(self, loaded_config: Any, raise_error_on_missing_key: bool):
        """Load configuration from command-line arguments.

        Args:
            loaded_config: Configuration dict from file (second priority)
            raise_error_on_missing_key: Whether to raise error on unknown keys
        """
        # --- Parameters coming from CLI arguments (sys.argv[1:]), highest priority
        args_params = sys.argv[1:]
        if not args_params:
            args_params = {}
        else:
            parsed = OmegaConf.create(OmegaConf.from_dotlist(args_params))
            args_params = OmegaConf.to_container(parsed, resolve=True)

        # --- Parameters from loaded file, second priority
        if loaded_config is None:
            loaded_config = {}

        # --- Overwrite parameters from loaded file with parameters from CLI arguments
        for key, value in args_params.items():
            loaded_config[key] = value

        # --- Update config object with loaded config and CLI arguments
        for key, value in loaded_config.items():
            if hasattr(self, key) and not callable(getattr(self, key)):
                try:
                    setattr(self, key, value)
                except Exception as _:
                    pass
            elif raise_error_on_missing_key:
                raise AttributeError(f"Unknown configuration key: {key}")

        self._loaded_config = loaded_config

    def load_from_environment(
        self,
        loaded_config,
        env_key: str,
        raise_error_on_missing_key: bool,
    ):
        # --- Parameters coming from environment variables, highest priority
        env_params = os.getenv(env_key, "").strip()
        if not env_params:
            env_params = {}
        else:
            parsed = OmegaConf.create(OmegaConf.from_dotlist(env_params.split()))
            env_params = OmegaConf.to_container(parsed, resolve=True)

        # --- Parameters from loaded file, second priority.
        if loaded_config is None:
            loaded_config = {}

        # ---  We overwrite parameters from loaded file with parameters from environment variables
        for key, value in env_params.items():
            loaded_config[key] = value

        # --- Update config object with loaded config and environment variables
        for key, value in loaded_config.items():
            if hasattr(self, key) and not callable(getattr(self, key)):
                try:
                    setattr(self, key, value)
                except Exception as _:
                    pass
            elif raise_error_on_missing_key:
                raise AttributeError(f"Unknown configuration key: {key}")

        self._loaded_config = loaded_config

    def __setattr__(self, name, value):
        # Override __setattr__ to avoid error where we set an attribute that is not defined in the class.
        # This could happen for example when we refactor and rename avariable but the renaming is not done everywhere.
        # Check if the attribute is allowed to be set
        allowed = set()
        for cls in self.__class__.__mro__:
            allowed.update(getattr(cls, "__dict__", {}).keys())
        allowed.update(self.__dict__.keys())

        if name not in allowed and not name.startswith("_"):
            raise AttributeError(
                f"Can't set unknown attribute '{name}'. Please check if this variable might have been renamed.",
            )
        super().__setattr__(name, value)


def overridable_property(fn):
    attr_name = fn.__name__

    @property
    def wrapper(self):
        try:
            if attr_name in self._loaded_config:
                return type(fn(self))(self._loaded_config[attr_name])
        except:
            pass
        return fn(self)

    return wrapper
