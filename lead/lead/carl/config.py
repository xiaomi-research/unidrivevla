# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Config class that contains all the hyperparameters needed to build any model.
"""

import numpy as np


class GlobalConfig:
    """
    Config class that contains all the hyperparameters needed to build any model.
    """

    def __init__(self):
        self.frame_rate = 10.0
        self.original_frame_rate = 20.0
        self.time_interval = 1.0 / self.frame_rate

        self.pixels_per_meter = 5.0
        self.bev_semantics_width = 192
        self.pixels_ev_to_bottom = 40
        self.bev_semantics_height = 192
        self.light_radius = 15.0
        self.debug = False
        self.logging_freq = 10
        self.logger_region_of_interest = 30.0
        self.route_points = 10

        half_second = int(self.frame_rate * 0.5)
        self.history_idx = [
            -3 * half_second - 1,
            -2 * half_second - 1,
            -1 * half_second - 1,
            -0 * half_second - 1,
        ]

        self.num_route_points_rendered = 80
        self.use_history = False
        self.history_idx_2 = [
            -3 * half_second - 1,
            -2 * half_second - 1,
            -1 * half_second - 1,
        ]

        self.bev_classes_list = (
            (0, 0, 0),
            (150, 150, 150),
            (255, 255, 255),
            (255, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 255, 0),
            (160, 160, 0),
            (0, 255, 0),
        )

        self.use_new_bev_obs = False
        self.route_width = 16
        self.red_light_thickness = 3
        self.use_extra_control_inputs = False
        self.max_avg_steer_angle = 60.0
        self.condition_outside_junction = True
        self.use_target_point = False
        self.use_value_measurements = True
        self.num_value_measurements = 3
        self.render_speed_lines = False
        self.use_positional_encoding = False
        self.render_yellow_time = False
        self.render_shoulder = True
        self.use_shoulder_channel = False

        self.scale_bbox = True
        self.scale_factor_vehicle = 1.0
        self.scale_factor_walker = 2.0
        self.min_ext_bounding_box = 0.8
        self.scale_mask_col = 1.0
        self.map_folder = "maps_low_res"
        self.max_speed_actor = 33.33
        self.min_speed_actor = -2.67

        self.ego_extent_x = 2.44619083404541
        self.ego_extent_y = 0.9183566570281982
        self.ego_extent_z = 0.7451388239860535

        self.reward_type = "roach"
        self.use_exploration_suggest = True
        self.rr_maximum_speed = 6.0
        self.vehicle_distance_threshold = 15
        self.max_vehicle_detection_number = 10
        self.rr_vehicle_proximity_threshold = 9.5
        self.pedestrian_distance_threshold = 15
        self.max_pedestrian_detection_number = 10
        self.rr_pedestrian_proximity_threshold = 9.5
        self.rr_tl_offset = -0.8 * self.ego_extent_x
        self.rr_tl_dist_threshold = 18.0
        self.min_thresh_lat_dist = 3.5
        self.eval_time = 1200.0
        self.n_step_exploration = 100
        self.use_speed_limit_as_max_speed = False

        self.consider_tl = True
        self.terminal_reward = 0.0
        self.terminal_hint = 10.0
        self.normalize_rewards = False
        self.speeding_infraction = False
        self.use_comfort_infraction = False
        self.max_abs_lon_jerk = 30.0
        self.max_abs_mag_jerk = 30.0
        self.min_lon_accel = -20.0
        self.max_lon_accel = 10.0
        self.max_abs_lat_accel = 9.0
        self.max_abs_yaw_rate = 1.0
        self.max_abs_yaw_accel = 3.0
        self.comfort_penalty_ticks = 500
        self.comfort_penalty_factor = 0.5
        self.use_vehicle_close_penalty = False
        self.use_termination_hint = False
        self.ego_forecast_time = 1.0
        self.ego_forecast_min_speed = 2.5
        self.use_perc_progress = False
        self.lane_distance_violation_threshold = 0.0
        self.lane_dist_penalty_softener = 1.0
        self.use_min_speed_infraction = False
        self.use_leave_route_done = True
        self.use_outside_route_lanes = False
        self.use_max_change_penalty = False
        self.max_change = 0.25
        self.penalize_yellow_light = True
        self.use_off_road_term = False
        self.off_road_term_perc = 0.5
        self.use_new_stop_sign_detector = False
        self.use_ttc = False
        self.ttc_resolution = 2
        self.ttc_penalty_ticks = 500
        self.max_overspeed_value_threshold = 2.23
        self.use_single_reward = True
        self.use_rl_termination_hint = False
        self.use_survival_reward = False
        self.survival_reward_magnitude = 0.0001

        self.action_repeat = 1

        self.obs_num_measurements = 8
        self.obs_num_channels = 15

        self.distribution = "beta"
        self.beta_min_a_b_value = 1.0
        self.normal_dist_init = ((0, -2), (0, -2))
        self.normal_dist_action_dep_std = True
        self.uniform_percentage_z = 0.03

        self.action_space_dim = 2
        self.action_space_min = -1.0
        self.action_space_max = 1.0
        self.start_delay_frames = int(2.0 / self.time_interval + 0.5)

        self.exp_name = "PPO_000"
        self.gym_id = "CARLAEnv-v0"
        self.learning_rate = 1.0e-5
        self.seed = 1
        self.total_timesteps = 10000000
        self.torch_deterministic = True
        self.cuda = True
        self.track = False
        self.wandb_project_name = "ppo-roach"
        self.wandb_entity = None
        self.capture_video = False
        self.num_envs = 1
        self.lr_schedule = "kl"
        self.gae = True
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.update_epochs = 4
        self.norm_adv = False
        self.clip_coef = 0.1
        self.clip_vloss = False
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = 0.015
        self.visualize = False
        self.logdir = ""
        self.load_file = None
        self.ports = (5555,)
        self.gpu_ids = (0,)
        self.compile_model = False
        self.total_batch_size = 512
        self.total_minibatch_size = 256
        self.expl_coef = 0.05
        self.lr_schedule_step = 8
        self.current_learning_rate = self.learning_rate
        self.kl_early_stop = 0
        self.adam_eps = 1e-5
        self.allow_tf32 = False
        self.benchmark = False
        self.matmul_precision = "highest"
        self.cpu_collect = False
        self.use_rpo = False
        self.rpo_alpha = 0.5
        self.use_green_wave = False
        self.green_wave_prob = 0.05
        self.image_encoder = "roach"
        self.use_layer_norm = False
        self.use_layer_norm_policy_head = True
        self.features_dim = 256
        self.use_lstm = False
        self.num_lstm_layers = 1

        self.render_green_tl = True
        self.lr_schedule_step_factor = 0.1
        self.lr_schedule_step_perc = (0.5, 0.75)
        self.weight_decay = 0.0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.lr_schedule_cosine_restarts = (0.0, 0.25, 0.50, 0.75, 1.0)
        self.use_dd_ppo_preempt = False
        self.dd_ppo_preempt_threshold = 0.6
        self.dd_ppo_min_perc = 0.25
        self.num_envs_per_proc = 1
        self.eval_intervals = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        self.current_eval_interval_idx = 0
        self.use_temperature = False
        self.min_temperature = 0.1

        self.use_hl_gauss_value_loss = False
        self.hl_gauss_std = 0.75
        self.hl_gauss_vmin = -10.0
        self.hl_gauss_vmax = 30.0
        self.hl_gauss_bucket_size = 1.0
        self.hl_gauss_num_classes = (
            int((self.hl_gauss_vmax - self.hl_gauss_vmin) / self.hl_gauss_bucket_size)
            + 1
        )

        self.global_step = 0
        self.max_training_score = -np.inf
        self.best_iteration = 0
        self.latest_iteration = 0

    def initialize(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
