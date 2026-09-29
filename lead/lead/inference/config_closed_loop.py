import os
import pathlib

from lead.common.config_base import overridable_property
from lead.inference.config_open_loop import OpenLoopConfig


class ClosedLoopConfig(OpenLoopConfig):
    """Configuration class for closed-loop inference settings in CARLA.

    See more details at https://ln2697.github.io/lead/docs/config_system.html
    """

    def __init__(self, raise_error_on_missing_key: bool = True):
        super().__init__(raise_error_on_missing_key=raise_error_on_missing_key)
        self.load_from_environment(
            loaded_config=None,
            env_key="LEAD_CLOSED_LOOP_CONFIG",
            raise_error_on_missing_key=raise_error_on_missing_key,
        )

    # --- Kalman Filter Settings ---
    use_kalman_filter = False

    # --- Image Processing ---
    # JPEG quality used in inference (0-100)
    jpeg_quality = 90

    # --- Control which output is used for controlling ---
    # Modality used for steering control
    steer_modality = "route"
    # Modality used for throttle control
    throttle_modality = "target_speed"
    # Modality used for brake control
    brake_modality = "target_speed"

    # --- Steering correction for slow driving ---
    # Steering correction factor
    sensor_agent_steer_correction = True
    # Steering correction parameter
    sensor_agent_steer_correction_param = 3.5

    # --- Route Planner ---
    # Minimum distance for route planner
    route_planner_min_distance = 5.0
    # Maximum distance for route planner
    route_planner_max_distance = 50

    # --- Adaptive target point controller ---
    # Skip next target point if too far away
    sensor_agent_skip_distant_target_point = True
    # Distance threshold for skipping next target point (meters)
    sensor_agent_skip_distant_target_point_threshold = 50
    # Adaptively reduce pop-distance if target points are dense
    sensor_agent_pop_distance_adaptive = True

    # --- Creeping Heuristic ---
    # If true, enable the creeping heuristic that forces the agent to move when stuck
    sensor_agent_creeping = False
    # Number of frames after which the creep controller starts triggering (larger than red light wait time)
    sensor_agent_stuck_threshold = 1100
    # Number of frames to creep forward when stuck
    sensor_agent_stuck_move_duration = 20
    # Throttle value for creeping when stuck
    sensor_agent_stuck_throttle = 0.4

    # --- Stop Sign Heuristic ---
    # If true enable stop sign controller
    slower_for_stop_sign = False
    # Distance threshold for stop sign controller activation
    slower_for_stop_sign_dist_threshold = 1.0
    # Cool down period for stop sign controller (frames)
    slower_for_stop_sign_cool_down = 120
    # Number of frames to apply slower behavior
    slower_for_stop_sign_count = 40
    # Throttle threshold for slower stop sign behavior
    slower_for_stop_sign_throttle_threshold = 0.1

    # --- PID Controller Parameters ---
    # Maximum change in speed input to longitudinal controller
    wp_delta_clip = 0.99
    # Aim distance for fast driving (meters)
    aim_distance_fast = 3.0
    # Aim distance for slow driving (meters)
    aim_distance_slow = 2.25
    # Speed threshold for switching between fast and slow aim distances (m/s)
    aim_distance_threshold = 5.5
    # Turn PID controller proportional gain
    turn_kp = 1.25
    # Turn PID controller integral gain
    turn_ki = 0.75
    # Turn PID controller derivative gain
    turn_kd = 0.3
    # Turn PID controller buffer size
    turn_n = 20
    # Speed PID controller proportional gain
    speed_kp = 1.75
    # Speed PID controller integral gain
    speed_ki = 1.0
    # Speed PID controller derivative gain
    speed_kd = 2.0
    # Speed PID controller buffer size
    speed_n = 20
    # Desired speed below which brake is triggered
    brake_speed = 0.4
    # Ratio of speed to desired speed at which brake is triggered
    brake_ratio = 1.1
    # Lateral PID controller proportional gain
    lateral_k_p = 3.118357247806046
    # Lateral PID controller derivative gain
    lateral_k_d = 1.3782508892109167
    # Lateral PID controller integral gain
    lateral_k_i = 0.6406067986034124
    # Speed scaling factor for lateral control
    lateral_speed_scale = 0.9755321901954155
    # Speed offset for lateral control
    lateral_speed_offset = 1.9152884533402488
    # Default lookahead distance for lateral control
    lateral_default_lookahead = 24
    # Speed threshold for lateral control behavior switching
    lateral_speed_threshold = 23.150102938235136
    # Lateral PID controller buffer size
    lateral_n = 6
    # If true use tuned aim distance for navigation
    tuned_aim_distance = False

    # --- Evaluation Visualization Settings ---
    # If not None, set a custom weather for evaluation
    custom_weather = None  # e.g., "ClearNoon"
    # If true, use random weather
    random_weather = False

    # Frequency of frame production during evaluation
    produce_frame_frequency = 1

    @overridable_property
    def produce_demo_image(self):
        """If true produce demo image output."""
        if not self.debug_mode:
            return False
        return False

    @overridable_property
    def produce_demo_video(self):
        """If true produce demo video output."""
        if not self.debug_mode:
            return False
        return False

    @overridable_property
    def produce_debug_image(self):
        """If true produce debug image output."""
        if not self.debug_mode:
            return False
        return False

    @overridable_property
    def produce_debug_video(self):
        """If true produce debug video output."""
        if not self.debug_mode:
            return False
        return True

    @overridable_property
    def produce_input_image(self):
        """If true produce input image output."""
        if not self.debug_mode:
            return False
        return False

    @overridable_property
    def produce_input_video(self):
        """If true produce input video output."""
        if not self.debug_mode:
            return False
        return False

    @overridable_property
    def produce_grid_image(self):
        """If true produce grid image output (demo + input stacked vertically)."""
        if not self.debug_mode:
            return False
        return False

    @overridable_property
    def produce_grid_video(self):
        """If true produce grid video output (demo + input stacked vertically)."""
        if not self.debug_mode:
            return False
        return False

    @overridable_property
    def produce_input_log(self):
        """If true produce input logging."""
        if not self.debug_mode:
            return False
        return False

    @property
    def save_path(self):
        """Get and create the save path for outputs."""
        save_path = os.environ.get("SAVE_PATH")
        if save_path is not None:
            save_path = pathlib.Path(save_path)
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        return save_path

    @property
    def is_bench2drive(self):
        """Check if running in Bench2Drive environment."""
        return bool(os.getenv("IS_BENCH2DRIVE", "False"))

    @property
    def route_id(self):
        """Get the benchmark route ID from environment."""
        return os.environ["BENCHMARK_ROUTE_ID"]

    @property
    def debug_video_path(self):
        """Get the path for video output."""
        return os.path.join(self.save_path, f"{self.route_id}_debug.mp4")

    @property
    def demo_video_path(self):
        """Get the path for demo video output."""
        return os.path.join(self.save_path, f"{self.route_id}_demo.mp4")

    @property
    def input_log_path(self):
        """Get and create the path for input logging."""
        path = os.path.join(self.save_path, "input_log")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def input_video_path(self):
        """Get the path for input video output."""
        return os.path.join(self.save_path, f"{self.route_id}_input.mp4")

    @property
    def grid_video_path(self):
        """Get the path for grid video output."""
        return os.path.join(self.save_path, f"{self.route_id}_grid.mp4")

    @overridable_property
    def video_fps(self):
        """Calculate video FPS based on frame production frequency."""
        return 20 / self.produce_frame_frequency
