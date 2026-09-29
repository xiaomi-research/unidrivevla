import logging
import os
from typing import Any

import torch
from beartype import beartype

from lead.common.config_base import BaseConfig, overridable_property
from lead.common.constants import (
    SOURCE_DATASET_NAME_MAP,
    CarlaImageCroppingType,
    NavSimBBClass,
    NavSimBEVSemanticClass,
    SourceDataset,
    TargetDataset,
    TransfuserBEVSemanticClass,
    TransfuserBoundingBoxClass,
    TransfuserSemanticSegmentationClass,
)

LOG = logging.getLogger(__name__)


class TrainingConfig(BaseConfig):
    def __init__(
        self,
        loaded_config: dict = None,
        raise_error_on_missing_key: bool = False,
    ):
        """Constructor for training config."""
        super().__init__()
        self.load_from_environment(
            loaded_config=loaded_config,
            env_key="LEAD_TRAINING_CONFIG",
            raise_error_on_missing_key=raise_error_on_missing_key,
        )
        self.load_from_args(
            loaded_config=self._loaded_config,
            raise_error_on_missing_key=raise_error_on_missing_key,
        )

    @property
    def target_dataset(self):
        if self.use_waymo_e2e_data:
            return TargetDataset.WAYMO_E2E_2025_3CAMERAS
        elif self.use_navsim_data:
            return TargetDataset.NAVSIM_4CAMERAS
        elif "carla_leaderboard2" in self.carla_root:
            return TargetDataset.CARLA_LEADERBOARD2_3CAMERAS
        elif "carla_leaderboad2_v3" in self.carla_root:
            return TargetDataset.CARLA_LEADERBOARD2_3CAMERAS
        elif "carla_leaderboad2_v8" in self.carla_root:
            return TargetDataset.CARLA_LEADERBOARD2_3CAMERAS
        elif "carla_leaderboad2_v10" in self.carla_root:
            return TargetDataset.CARLA_LEADERBOARD2_6CAMERAS
        raise ValueError(
            f"Unknown CARLA root path: {self.carla_root}. Please register it in the config.",
        )

    @property
    def num_available_cameras(self):
        """Number of available cameras based on the target dataset."""
        return {
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS: 3,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS: 6,
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS: 3,
            TargetDataset.NAVSIM_4CAMERAS: 4,
            TargetDataset.WAYMO_E2E_2025_3CAMERAS: 3,
        }[self.target_dataset]

    @overridable_property
    def used_cameras(self):
        """List indicating which cameras are used based on the target dataset.
        Can be overridden, if a camera is false it will be ignored during training."""
        return [True] * self.num_available_cameras

    @property
    def num_used_cameras(self):
        """Number of cameras used during training."""
        return sum(int(use) for use in self.used_cameras)

    # --- Planning Area ---
    # How many pixels make up 1 meter in BEV grids.
    pixels_per_meter = 4.0

    @property
    def min_x_meter(self):
        """Back boundary of the planning area in meters."""
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return 0
        return -32

    @property
    def max_x_meter(self):
        """Front boundary of the planning area in meters."""
        if self.carla_leaderboard_mode:
            return 64
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return 64
        return 32

    @property
    def min_y_meter(self):
        """Left boundary of the planning area in meters."""
        if self.carla_leaderboard_mode:
            return -40
        return -32

    @property
    def max_y_meter(self):
        """Right boundary of the planning area in meters."""
        if self.carla_leaderboard_mode:
            return 40
        return 32

    @property
    def lidar_width_pixel(self):
        """Width resolution of LiDAR BEV representation in pixels."""
        return int((self.max_x_meter - self.min_x_meter) * self.pixels_per_meter)

    @property
    def lidar_height_pixel(self):
        """Height resolution of LiDAR BEV representation in pixels."""
        return int((self.max_y_meter - self.min_y_meter) * self.pixels_per_meter)

    @property
    def lidar_width_meter(self):
        """Width of LiDAR coverage area in meters."""
        return int(self.max_x_meter - self.min_x_meter)

    @property
    def lidar_height_meter(self):
        """Height of LiDAR coverage area in meters."""
        return int(self.max_y_meter - self.min_y_meter)

    # Model type: "tfv6" (default) or "plant".
    # Drives model creation in training_utils.create_model().
    model_type = "tfv6"

    # Flag to visualize the dataset and deactivate randomization and augmentation.
    visualize_dataset = False
    # Flag to visualize the failed scenarios and deactivate randomization and augmentation.
    visualize_failed_scenarios = False
    # Flag to load the BEV 3rd person images from the dataset for debugging.
    load_bev_3rd_person_images = False

    # --- Training ID, Logging setting ---
    # Training Seed
    seed = 0
    # WandB ID for the experiment. If None, it will be generated automatically.
    wandb_id: str = None
    # must, allow, never
    wandb_resume = "never"

    @property
    def wandb_project_name(self):
        """Name of the WandB project based on the training phase."""
        if self.is_pretraining:
            return "lead_pretrain"
        return "lead_posttrain"

    # Description of the experiment.
    description = "An example experiment description."

    @overridable_property
    def visualize_training(self):
        """If true produce images during training for visualization."""
        return True

    # Unique experiment identifier.
    id = "Experiment 1"
    # File to continue training from
    load_file: str = None
    # If true continue the training from a failed training checkpoint.
    continue_failed_training = False

    @property
    def epoch_checkpoints_keep(self):
        """Number of checkpoints to keep during training."""
        if self.carla_leaderboard_mode and not self.mixed_data_training:
            return []
        return [sum([1 * 2**i for i in range(n)]) for n in range(3, 10)]

    # --- Training cache ---
    # If true use training session cache. This cache reduces data loading time.
    use_training_session_cache = True
    # If true use persistent cache for training. This cache reduces heavy feature building.
    use_persistent_cache = True
    # If true force rebuild the cache for each training run.
    force_rebuild_data_cache = False

    @property
    def carla_cache_path(self):
        """Tuple of cache characteristics used to identify cached data compatibility."""
        return (
            str(self.image_width_before_camera_subselection),
            str(self.final_image_height),
            str(self.min_x_meter),
            str(self.max_x_meter),
            str(self.min_y_meter),
            str(self.max_y_meter),
            str(self.detect_boxes),
            str(self.use_depth),
            str(self.use_semantic),
            str(self.use_bev_semantic),
            str(self.load_bev_3rd_person_images),
            str(self.training_used_lidar_steps),
        )

    @property
    def training_session_cache_path(self):
        """Path to SSD cache directory."""
        tmp_folder = "/scratch/" + str(os.environ.get("SLURM_JOB_ID"))
        if not self.is_on_tcml:
            tmp_folder = str(os.environ.get("SCRATCH", f"/tmp/{os.environ['USER']}"))
        return tmp_folder

    # Root directory for CARLA sensor data.
    carla_root = "data/carla_leaderboard2"

    @property
    def carla_data(self):
        """Path to CARLA data directory."""
        return os.path.join(self.carla_root, "data")

    # --- Training ---
    # Directory to log data to.
    logdir = None
    # PNG compression level for storing images
    training_png_compression_level = 6
    # Minimum number of LiDAR points for a vehicle to be considered valid.
    vehicle_min_num_lidar_points = 1
    # Minimum number of visible pixels for a vehicle to be considered valid.
    vehicle_min_num_visible_pixels = 1
    # Minimum number of LiDAR points for a pedestrian to be considered valid.
    pedestrian_min_num_lidar_points = 5
    # Minimum number of visible pixels for a pedestrian to be considered valid.
    pedestrian_min_num_visible_pixels = 15
    # Minimum number of LiDAR points for a parking vehicle to be considered valid.
    parking_vehicle_min_num_lidar_points = 3
    # Minimum number of visible pixels for a parking vehicle to be considered valid.
    parking_vehicle_min_num_visible_pixels = 5
    # First scale we use for the gradient scaler.
    grad_scaler_init_scale = 1024
    # Factor by which we grow the gradient scaler.
    grad_scaler_growth_factor = 2
    # Factor by which we backoff the gradient scaler if the gradients are too large.
    grad_scaler_backoff_factor = 0.5
    # Number of steps after which we grow the gradient scaler.
    grad_scaler_growth_interval = 256
    # Maximum gradient scale we use for the gradient scaler.
    grad_scaler_max_grad_scale = 2**16

    @property
    def sync_batchnorm(self) -> bool:
        """If true synchronize batch normalization across distributed processes."""
        return False

    @overridable_property
    def epochs(self):
        """Total number of training epochs."""
        if self.carla_leaderboard_mode:
            return 31
        if self.target_dataset == TargetDataset.NAVSIM_4CAMERAS:
            return 61
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return 20
        raise ValueError(
            "Unknown target dataset. Not sure how many epochs to train for.",
        )

    @overridable_property
    def batch_size(self):
        """Batch size for training."""
        if self.debug_mode:
            return 2
        return 64

    @property
    def torch_float_type(self):
        """PyTorch float precision type for training."""
        if self.use_mixed_precision_training:
            return torch.bfloat16
        return torch.float32

    # If true use mixed precision training with float16. This can speed up training and reduce memory usage on compatible hardware.
    use_mixed_precision_training = True

    @property
    def need_grad_scaler(self):
        """If true gradient scaling is needed for mixed precision training."""
        return (
            self.use_mixed_precision_training and self.torch_float_type == torch.float16
        )

    # If true use ZeRO redundancy optimizer for distributed training.
    use_zero_redundancy = False

    @overridable_property
    def fused_optimizer(self):
        return True

    @property
    def save_model_checkpoint(self):
        """If true save model checkpoints during training."""
        return True

    @property
    def is_pretraining(self):
        """If true indicates pretraining phase."""
        return not self.use_planning_decoder

    # --- Training speed and memory optimization ---
    # Number of data loader workers to prefetch batches.
    prefetch_factor = 16

    @overridable_property
    def compile(self):
        """If true compile the model for optimization."""
        return True

    @property
    def channel_last(self):
        """If true use channel last memory format for input tensors."""
        return True

    # --- Learning rate and epochs ---
    # Base learning rate for the model.
    lr = 3e-4

    # --- Model input ---
    @property
    def skip_first(self):
        """Number of frames to skip at the beginning of sequences."""
        if self.is_pretraining and not self.mixed_data_training:
            return 1
        return self.num_way_points_prediction

    @property
    def skip_last(self):
        """Number of frames to skip at the end of sequences."""
        if self.is_pretraining:
            return 1
        if self.carla_leaderboard_mode:
            return self.num_way_points_prediction
        if self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return self.num_way_points_prediction * 2
        raise ValueError(
            "Unknown target dataset. Not sure how many frames to skip at the end of sequences.",
        )

    # --- RaDAR ---
    @overridable_property
    def radar_detection(self):
        """If true use radar points as additional input to the model."""
        return self.carla_leaderboard_mode

    @overridable_property
    def use_radar_detection(self):
        """If true use radar points as additional input to the model."""
        return self.carla_leaderboard_mode

    # Fixed number of radar points per sensor.
    num_radar_points_per_sensor = 75
    # Number of radar queries in the transformer.
    num_radar_queries = 20
    # Hidden dimension for radar tokenizer.
    radar_hidden_dim_tokenizer = 1024
    # Dimension of radar tokens.
    radar_token_dim = 256
    # Feed-forward dimension in radar transformer.
    radar_tf_dim_ff = 1024
    # Dropout rate for radar components.
    radar_dropout = 0.1
    # Number of attention heads in radar transformer.
    radar_num_heads = 8
    # Number of transformer layers for radar processing.
    radar_num_layers = 4
    # Hidden dimension for radar decoder.
    radar_hidden_dim_decoder = 1024
    # Loss weight for radar classification.
    radar_classification_loss_weight = 1.0
    # Loss weight for radar regression.
    radar_regression_loss_weight = 5.0
    # Total number of radar sensors.
    num_radar_sensors = 4

    # --- Data filtering and bucket system ---
    # If true rebuild buckets collection from scratch.
    force_rebuild_bucket = False
    # If true randomize routes in bucket
    randomize_route_order = False
    # If true then we skip Town13 routes during training
    hold_out_town13_routes = False

    @property
    def carla_bucket_collection(self):
        """Name of the bucket collection to use for training data."""
        from lead.data_buckets.failed_bucket_collection import FailedBucketCollection
        from lead.data_buckets.full_posttrain_bucket_collection import (
            FullPosttrainBucketCollection,
        )
        from lead.data_buckets.full_pretrain_bucket_collection import (
            FullPretrainBucketCollection,
        )
        from lead.data_buckets.navsim_bucket_collection import NavSimBucketCollection
        from lead.data_buckets.town13_heldout_posttrain_bucket_collection import (
            Town13HeldoutPosttrainBucketCollection,
        )
        from lead.data_buckets.town13_heldout_pretrain_bucket_collection import (
            Town13HeldOutPretrainBucketCollection,
        )
        from lead.data_buckets.waymo_bucket_collection import WaymoBucketCollection

        if (
            self.use_carla_data
            and self.use_waymo_e2e_data
            and not self.force_rebuild_bucket
            and not self.force_rebuild_data_cache
        ):
            return WaymoBucketCollection
        if (
            self.use_carla_data
            and self.use_navsim_data
            and not self.force_rebuild_bucket
            and not self.force_rebuild_data_cache
        ):
            return NavSimBucketCollection
        if self.visualize_failed_scenarios:
            return FailedBucketCollection
        if self.is_pretraining and self.hold_out_town13_routes:
            return Town13HeldOutPretrainBucketCollection
        if (
            self.is_pretraining
            or self.visualize_dataset
            or self.force_rebuild_data_cache
        ):
            return FullPretrainBucketCollection
        if not self.is_pretraining and self.hold_out_town13_routes:
            return Town13HeldoutPosttrainBucketCollection
        return FullPosttrainBucketCollection

    @property
    def bucket_collection_path(self):
        """Path to bucket collection directory."""
        return os.path.join(self.carla_root, "buckets")

    # --- Training to recover from drift ---
    # If true use rotation and translation perburtation.
    use_sensor_perburtation = True

    # Probability of the perburtated sample being used.
    @overridable_property
    def use_sensor_perburtation_prob(self):
        if not self.use_sensor_perburtation:
            return 0.0
        if not self.carla_leaderboard_mode:
            return 0.8
        return 0.5

    # --- Regularization ---
    @property
    def use_color_aug(self):
        """If true apply image color based augmentations."""
        # If true apply image color based augmentations
        return not self.visualize_dataset

    @property
    def use_color_aug_prob(self):
        """Probability to apply the different image color augmentations."""
        if self.carla_leaderboard_mode:
            return 0.2
        return 0.1

    # Weight decay for regularization.
    weight_decay = 0.01

    # If true use cosine learning rate scheduler with restart, else only one cycle
    @overridable_property
    def use_cosine_annealing_with_restarts(self):
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return False
        return True

    # --- RGB ---
    # If true load RGB camera images from disk. Disable for models that don't use RGB (e.g., PlanT).
    use_rgb = True

    # --- Depth ---
    @overridable_property
    def use_depth(self):
        """If true use depth prediction as auxiliary task."""
        return self.carla_leaderboard_mode

    # --- LiDAR setting ---
    # If true load LiDAR points from disk and rasterize to BEV. Disable for models that don't use LiDAR (e.g., PlanT).
    use_lidar = True

    @property
    def training_used_lidar_steps(self):
        """We stack lidar frames for motion cues. Number of past frames we stack for the model input."""
        return 10

    # Minimum Z coordinate for LiDAR points.
    min_z = -4
    # Maximum Z coordinate for LiDAR points.
    max_z = 4
    # Max number of LiDAR points per pixel in voxelized LiDAR.
    hist_max_per_pixel = 5

    @property
    def lidar_vert_anchors(self):
        """Number of vertical anchors for LiDAR feature maps."""
        return self.lidar_height_pixel // 32

    @property
    def lidar_horz_anchors(self):
        """Number of horizontal anchors for LiDAR feature maps."""
        return self.lidar_width_pixel // 32

    # --- Bounding boxes detection ---
    # If true use the bounding box auxiliary task.
    detect_boxes = True
    # List of static object types to include in bounding box detection.
    data_bb_static_types_white_list = [
        "static.prop.constructioncone",
        "static.prop.trafficwarning",
    ]
    # Confidence of a bounding box that is needed for the detection to be accepted.
    bb_confidence_threshold = 0.3
    # Maximum number of bounding boxes our system can detect.
    max_num_bbs = 90
    # Number of direction bins for object orientation.
    num_dir_bins = 12
    # Top K center keypoints to consider during detection.
    top_k_center_keypoints = 100
    # Kernel size for CenterNet max pooling operation.
    center_net_max_pooling_kernel = 3
    # Number of input channels for bounding box detection head.
    bb_input_channel = 64
    # Extra width to add when car doors are open for safety.
    car_open_door_extra_width = 1.2
    # Total number of bounding box classes to detect.
    num_bb_classes = len(TransfuserBoundingBoxClass)

    # --- Context and statuses ---
    @overridable_property
    def use_discrete_command(self):
        """If true use discrete command input to the network."""
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return False
        return True

    @property
    def discrete_command_dim(self):
        """Dimension of discrete command input."""
        if self.carla_leaderboard_mode:
            return 6
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 4
        elif self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS,
        ]:
            return 4
        raise ValueError(
            "Unknown target dataset. Not sure how many discrete commands there are.",
        )

    # If true train model with noisy target points. The level of noise depends on use_kalman_filter_for_gps.
    use_noisy_tp = False
    # If true, use the Kalman filter for less noisy ego state estimation.
    use_kalman_filter_for_gps = True
    # If true use the velocity as input to the network.
    use_velocity = True

    @property
    def max_speed(self):
        """Maximum speed limit for the vehicle in m/s."""
        if self.carla_leaderboard_mode:
            return 25.0
        if self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 15.0
        if self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS,
        ]:
            return 33.33
        raise ValueError("Unknown target dataset. Not sure what max speed to use.")

    @property
    def use_acceleration(self):
        """If true use the acceleration as input to the network."""
        return not self.carla_leaderboard_mode and self.target_dataset not in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS,
        ]

    @property
    def max_acceleration(self):
        """Maximum acceleration for normalization."""
        if self.carla_leaderboard_mode:
            return 10.0
        if self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 4.0

    @property
    def use_previous_tp(self):
        """If true use the previous/visited target point as input to the network."""
        if self.carla_leaderboard_mode:
            return True
        return False

    @property
    def use_next_tp(self):
        """If true use the next/subsequent target point as input to the network."""
        if self.carla_leaderboard_mode:
            return True
        return False

    @property
    def use_tp(self):
        """If true use the current target point as input to the network."""
        if self.carla_leaderboard_mode:
            return True
        return False

    @property
    def target_points_normalization_constants(self):
        """Normalization constants for target points [x_norm, y_norm]."""
        return [[200.0, 50.0]]

    @property
    def tp_pop_distance(self):
        """Distance threshold for popping target points from route."""
        return 3.25

    @overridable_property
    def use_past_positions(self):
        """If true use past positions as input to the network."""
        return not self.carla_leaderboard_mode and self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS,
        ]

    @overridable_property
    def use_past_speeds(self):
        """If true use past speeds as input to the network."""
        return not self.carla_leaderboard_mode and self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS,
        ]

    @overridable_property
    def num_past_samples_used(self):
        """Number of past samples to use as input to the network."""
        if self.use_past_positions or self.use_past_speeds:
            return 6
        return 0

    # --- Planning decoder configuration ---
    # Number of BEV cross-attention layers in TransFuser.
    transfuser_num_bev_cross_attention_layers = 6
    # Number of attention heads in BEV cross-attention.
    transfuser_num_bev_cross_attention_heads = 8
    # Dimension of tokens in the transformer.
    transfuser_token_dim = 256

    @property
    def predict_target_speed(self):
        """If true predict target speed."""
        return self.carla_leaderboard_mode

    @property
    def predict_spatial_path(self):
        """If true predict spatial path."""
        return self.carla_leaderboard_mode

    # If true predict temporal spatial waypoints.
    predict_temporal_spatial_waypoints = True

    # If true model will use the planning decoder.
    use_planning_decoder = False

    @property
    def target_speed_classes(self):
        """Carla target speed prediction classes in m/s."""
        return [
            0.0,
            4.0,
            8.0,
            10.0,
            13.88888888,
            16.0,
            17.77777777,
            20.0,
        ]

    @property
    def target_speeds(self):
        return self.target_speed_classes

    # If true smooth the route points with a spline.
    smooth_route = True
    # Number of route points we use for smoothing.
    num_route_points_smoothing = 20
    # Number of route checkpoints to predict. Needs to be smaller than num_route_points_smoothing!
    num_route_points_prediction = 10
    # Assume maximum distance between two future waypoints in meters.
    max_distance_future_waypoint = 10.0

    @property
    def num_way_points_prediction(self):
        """Number of waypoints to predict."""
        if self.carla_leaderboard_mode:
            return 8  # 4Hz and 2 seconds
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 8  # 2Hz and 4 seconds
        elif self.target_dataset in [TargetDataset.WAYMO_E2E_2025_3CAMERAS]:
            return 10  # 2Hz and 5 seconds
        raise ValueError(
            "Unknown target dataset. Not sure how long is the planning horizon.",
        )

    @property
    def waypoints_spacing(self):
        """Spacing between predicted waypoints. For example: spacing 5 = 4Hz prediction."""
        if self.carla_leaderboard_mode:
            return 5  # 4Hz
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 10  # 2Hz
        elif self.target_dataset in [TargetDataset.WAYMO_E2E_2025_3CAMERAS]:
            return 10  # 2Hz
        raise ValueError(
            "Unknown target dataset. Not sure which planning frequency to use.",
        )

    # --- Image config ---
    # Horizontal FOV reduction: number of pixels to crop from each side (left and right)
    horizontal_fov_reduction = 0

    @property
    def crop_height(self):
        """The amount of pixels cropped from the bottom of the image."""
        return (
            self.camera_calibration[1]["height"]
            - self.camera_calibration[1]["cropped_height"]
        )

    @property
    def carla_crop_height_type(self):
        """Type of cropping applied to CARLA images."""
        if self.carla_leaderboard_mode:
            return CarlaImageCroppingType.NONE
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return CarlaImageCroppingType.BOTTOM
        elif self.target_dataset in [TargetDataset.WAYMO_E2E_2025_3CAMERAS]:
            return CarlaImageCroppingType.NONE
        raise ValueError("Unknown target dataset. Not sure how to crop the images.")

    @property
    def image_width_before_camera_subselection(self):
        """Final width of images after loading from disk but before camera sub-selection."""
        return self.num_available_cameras * self.camera_calibration[1]["width"]

    @property
    def final_image_width(self):
        """Final width of images after cropping and camera sub-selection."""
        return self.num_used_cameras * self.camera_calibration[1]["width"]

    @property
    def final_image_height(self):
        """Final height of images after cropping."""
        return self.camera_calibration[1]["cropped_height"]

    @property
    def img_vert_anchors(self):
        """Number of vertical anchors for image feature maps."""
        return self.final_image_height // 32

    @property
    def img_horz_anchors(self):
        """Number of horizontal anchors for image feature maps."""
        return self.num_used_cameras * self.camera_calibration[1]["width"] // 32

    # --- TransFuser backbone ---
    # If true freeze the backbone weights during training.
    freeze_backbone = False
    # Architecture name for image encoder backbone.
    image_architecture = "resnet34"
    # Architecture name for LiDAR encoder backbone.
    lidar_architecture = "resnet34"
    # Latent TF
    LTF = False

    # GPT Encoder
    # Block expansion factor for GPT layers.
    block_exp = 4
    # Number of transformer layers used in the vision backbone.
    n_layer = 2
    # Number of attention heads in transformer.
    n_head = 4
    # Embedding dropout probability.
    embd_pdrop = 0.1
    # Residual connection dropout probability.
    resid_pdrop = 0.1
    # Attention dropout probability.
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    # --- Semantic segmentation ---
    # If true use semantic segmentation as auxiliary loss.
    use_semantic = True
    # Total number of semantic segmentation classes.
    num_semantic_classes = len(TransfuserSemanticSegmentationClass)
    # Resolution at which the perspective auxiliary tasks are predicted
    perspective_downsample_factor = 1
    # Number of channels at the first deconvolution layer
    deconv_channel_num_0 = 128
    # Number of channels at the second deconvolution layer
    deconv_channel_num_1 = 64
    # Number of channels at the third deconvolution layer
    deconv_channel_num_2 = 32
    # Fraction of the down-sampling factor that will be up-sampled in the first Up-sample
    deconv_scale_factor_0 = 4
    # Fraction of the down-sampling factor that will be up-sampled in the second Up-sample.
    deconv_scale_factor_1 = 8

    # --- BEV Semantic ---
    # If true use bev semantic segmentation as auxiliary loss for training.
    use_bev_semantic = True
    # Total number of BEV semantic segmentation classes.
    num_bev_semantic_classes = len(TransfuserBEVSemanticClass)
    # Scale factor for pedestrian BEV semantic size.
    scale_pedestrian_bev_semantic_size = 2.5
    # Minimum extent for pedestrian BEV representation.
    pedestrian_bev_min_extent = 0.4
    # Number of channels for the BEV feature pyramid.
    bev_features_chanels = 64
    # Resolution at which the BEV auxiliary tasks are predicted.
    bev_down_sample_factor = 4
    # Upsampling factor for BEV features.
    bev_upsample_factor = 2

    # --- Mixed data training settings ---
    # If true use CARLA data for training.
    use_carla_data = True
    # Number of CARLA samples to use in mixed data training. -1 = use all data.
    carla_num_samples = -1
    # If true use NavSim data for training.
    use_navsim_data = False
    # NavSim data root directory.
    navsim_data_root = "data/navsim_training_cache/navtrain"
    # Size of NavSim data portion in mixed data training. -1 = use all data.
    navsim_num_samples = -1
    # If true then we also schedule number of samples from CARLA in each batch.
    schedule_carla_num_samples = False
    # If true use Waymo E2E data for training
    use_waymo_e2e_data = False
    # Number of Waymo E2E data from training split
    waymo_e2e_num_training_samples = -1
    # Waymo E2E training data root directory.
    waymo_e2e_training_data_root = (
        "data/waymo_open_dataset_end_to_end_camera_v_1_0_0_training"
    )
    # Waymo E2E validation data root directory.
    waymo_e2e_val_data_root = (
        "data/waymo_open_dataset_end_to_end_camera_v_1_0_0_val_rfm"
    )
    # Waymo E2E test data root directory.
    waymo_e2e_test_data_root = (
        "data/waymo_open_dataset_end_to_end_camera_v_1_0_0_test_submission"
    )
    # Waymo E2E subsample factor for training data.
    waymo_e2e_subsample_factor = 5

    @property
    def navsim_num_bev_semantic_classes(self):
        """Number of BEV semantic classes in NavSim data."""
        return len(NavSimBEVSemanticClass)

    @property
    def navsim_num_bb_classes(self):
        """Number of bb classes in NavSim data."""
        return len(NavSimBBClass)

    @property
    def mixed_data_training(self):
        """If true use mixed data for training."""
        return (
            int(self.use_navsim_data)
            + int(self.use_carla_data)
            + int(self.use_waymo_e2e_data)
            > 1
        )

    @property
    def carla_leaderboard_mode(self):
        """If true use CARLA leaderboard mode settings."""
        return (
            self.target_dataset
            in [
                TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
                TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
            ]
            and not self.mixed_data_training
        )

    @beartype
    def detailed_loss_weights(self, source_dataset: int, _: int) -> dict[str, float]:
        """Computed loss weights for all auxiliary tasks with normalization."""

        weights = {
            "loss_semantic": 1.0,
            "loss_depth": 0.00001,
            "loss_bev_semantic": 1.0,
            "loss_center_net_heatmap": 1.0,
            "loss_center_net_wh": 1.0,
            "loss_center_net_offset": 1.0,
            "loss_center_net_yaw_class": 1.0,
            "loss_center_net_yaw_res": 1.0,
            "loss_center_net_velocity": 1.0,
            "radar_loss": 1.0,
        }

        if source_dataset != SourceDataset.CARLA:
            weights["radar_loss"] = 0.0
            weights["loss_semantic"] = 0.0
            weights["loss_depth"] = 0.0
            weights["loss_center_net_velocity"] = 0.0

        if self.LTF:
            weights["loss_center_net_velocity"] = 0.0

        if not self.use_semantic:
            weights["loss_semantic"] = 0.0

        if not self.use_depth:
            weights["loss_depth"] = 0.0

        if not self.use_bev_semantic:
            weights["loss_bev_semantic"] = 0.0

        if not self.detect_boxes:
            weights["loss_center_net_heatmap"] = 0.0
            weights["loss_center_net_wh"] = 0.0
            weights["loss_center_net_offset"] = 0.0
            weights["loss_center_net_yaw_class"] = 0.0
            weights["loss_center_net_yaw_res"] = 0.0
            weights["loss_center_net_velocity"] = 0.0

        if self.training_used_lidar_steps <= 1:
            weights["loss_center_net_velocity"] = 0.0

        if not self.radar_detection:
            weights["radar_loss"] = 0.0

        # Add prefix to the loss weights based on the source dataset
        prefix = f"{SOURCE_DATASET_NAME_MAP[source_dataset]}_"
        if source_dataset == SourceDataset.CARLA:
            prefix = ""
        weights = {f"{prefix}{k}": v for k, v in weights.items()}

        # Unified planning loss, no source dataset prefix
        weights.update(
            {
                "loss_spatio_temporal_waypoints": 1.0,
                "loss_target_speed": 1.0,
                "loss_spatial_route": 1.0,
            },
        )

        # Disable planning losses during pretraining
        if not self.use_planning_decoder:
            weights["loss_spatio_temporal_waypoints"] = 0.0
            weights["loss_spatial_route"] = 0.0
            weights["loss_target_speed"] = 0.0

        return weights

    @overridable_property
    def log_scalars_frequency(self):
        """How often to log scalar values during training."""
        if self.debug_mode:
            return 1
        try:
            with open(
                os.path.join(
                    self.lead_project_root,
                    "slurm/configs/wandb_log_frequency_training_scalar.txt",
                ),
            ) as f:
                return int(f.readline().strip())
        except Exception as e:
            LOG.error(f"Error reading log frequency file: {e}.")
            return 1

    @property
    def log_images_frequency(self):
        """How often to log images during training."""
        if self.debug_mode:
            return 100
        try:
            with open(
                os.path.join(
                    self.lead_project_root,
                    "slurm/configs/wandb_log_frequency_training_images.txt",
                ),
            ) as f:
                return int(f.readline().strip())
        except Exception as e:
            LOG.error(f"Error reading log frequency file: {e}.")
            return 100

    @overridable_property
    def log_wandb(self):
        """If true log metrics to Weights & Biases."""
        if self.debug_mode:
            return False
        return True

    # --- Hardware configuration ---
    @property
    def rank(self):
        """Current process rank in distributed training."""
        return int(os.environ.get("RANK", "0"))

    @property
    def world_size(self):
        """Total number of processes in distributed training."""
        return int(os.environ.get("WORLD_SIZE", "1"))

    @property
    def local_rank(self):
        """Local rank of current process on the node."""
        return int(os.environ.get("LOCAL_RANK", "0"))

    @property
    def device(self):
        """PyTorch device to use for training."""
        return torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu",
        )

    @overridable_property
    def assigned_cpu_cores(self):
        """Number of CPU cores assigned to this job."""
        if "SLURM_JOB_ID" in os.environ:
            cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
            if cpus_per_task:
                return int(cpus_per_task)
        return 8

    @property
    def workers_per_cpu_cores(self):
        """Number of data loader workers per CPU core."""
        if not self.mixed_data_training and not self.use_carla_data:
            return 1
        return 1

    def training_dict(self) -> dict[str, Any]:
        """Convert training configuration to a dictionary for serialization and logging."""
        out = {}
        cls = self.__class__
        for k, v in cls.__dict__.items():
            if isinstance(v, property):
                try:
                    out[k] = getattr(self, k)
                except Exception:
                    pass
            else:
                if not k.startswith("__") and not callable(v):
                    out[k] = v
        for k, v in self.__dict__.items():
            if isinstance(v, property):
                try:
                    out[k] = getattr(self, k)
                except Exception:
                    pass
            else:
                if not k.startswith("__") and not callable(v):
                    out[k] = v
        return out
