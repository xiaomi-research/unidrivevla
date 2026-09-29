# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
CaRL agent entry point for the CARLA leaderboard.
Inference-only port of 3rd_party/carl/team_code/eval_agent.py.
"""

import logging
import math
import os
import shutil
from typing import Any

import carla
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch
from gymnasium import spaces
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.carl import rl_utils as rl_u
from lead.carl.bev.bev_observation import ObsManager as ObsManager2
from lead.carl.bev.chauffeurnet import ObsManager
from lead.carl.bev.run_stop_sign import RunStopSign
from lead.carl.config import GlobalConfig
from lead.carl.model import PPOPolicy
from lead.carl.nav_planner import RoutePlanner
from lead.common.logging_config import setup_logging
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.infraction_recorder import InfractionRecorder
from lead.inference.video_recorder import VideoRecorder

jsonpickle_numpy.register_handlers()
setup_logging()
LOG = logging.getLogger(__name__)

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_entry_point():
    return "CarlAgent"


class CarlAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.step = -1
        self.track = autonomous_agent.Track.MAP
        self.config_path = path_to_conf_file
        self.config = GlobalConfig()
        self.config_closed_loop = ClosedLoopConfig(raise_error_on_missing_key=False)

        # Bench2Drive may append route metadata to agent_config using "+...".
        # Keep only the checkpoint directory prefix when loading model files.
        checkpoint_dir = path_to_conf_file.split("+")[0]

        if self.config_closed_loop.save_path is not None and not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is not installed or not found in PATH. "
                "Please install ffmpeg to use video compression.",
            )

        with open(os.path.join(checkpoint_dir, "config.json"), encoding="utf-8") as f:
            json_config = f.read()

        loaded_config = jsonpickle.decode(json_config)
        if isinstance(loaded_config, dict):
            self.config.__dict__.update(loaded_config)
        else:
            self.config.__dict__.update(loaded_config.__dict__)

        self.initialized = False
        self.list_traffic_lights = []

        self.sample_type = os.environ.get("SAMPLE_TYPE", "mean")
        self.high_freq_inference = int(os.environ.get("HIGH_FREQ_INFERENCE", 0))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.meters_travelled = 0.0
        self.infraction_recorder = InfractionRecorder(
            config_closed_loop=self.config_closed_loop,
            agent_name="CarlAgent",
        )
        self.infractions_log = self.infraction_recorder.infractions_log

        self.observation_space = spaces.Dict(
            {
                "bev_semantics": spaces.Box(
                    0,
                    255,
                    shape=(
                        self.config.obs_num_channels,
                        self.config.bev_semantics_height,
                        self.config.bev_semantics_width,
                    ),
                    dtype=np.uint8,
                ),
                "measurements": spaces.Box(
                    -math.inf,
                    math.inf,
                    shape=(self.config.obs_num_measurements,),
                    dtype=np.float32,
                ),
            },
        )
        self.action_space = spaces.Box(
            self.config.action_space_min,
            self.config.action_space_max,
            shape=(self.config.action_space_dim,),
            dtype=np.float32,
        )

        self.agents = []
        self.model_count = 0
        for file in os.listdir(checkpoint_dir):
            if file.startswith("model") and file.endswith(".pth"):
                self.model_count += 1
                print(os.path.join(checkpoint_dir, file))
                agent = PPOPolicy(
                    self.observation_space,
                    self.action_space,
                    config=self.config,
                ).to(self.device)
                state_dict = torch.load(
                    os.path.join(checkpoint_dir, file),
                    map_location=self.device,
                )
                agent.load_state_dict(state_dict, strict=True)
                agent.to(self.device)
                agent.eval()
                self.agents.append(agent)

        if self.high_freq_inference:
            self.total_action_repeat = int(self.config.action_repeat)
        else:
            self.total_action_repeat = int(
                self.config.action_repeat
                * (self.config.original_frame_rate // self.config.frame_rate),
            )

    def sensors(self):
        return []

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # Store the dense (non-downsampled) route before the parent downsamples it.
        self.dense_global_plan_world_coord = global_plan_world_coord
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def set_scenario(self, scenario: Any) -> None:
        """Set the active scenario used for infraction tracking.

        Args:
            scenario: Scenario object that provides get_criteria().
        """
        self.infraction_recorder.set_scenario(scenario)
        LOG.info("[CarlAgent] Scenario reference set for infraction tracking")

    def _update_infraction_tracking(self) -> None:
        """Update travelled distance and log newly detected infractions."""
        velocity = self.vehicle.get_velocity()
        speed = float(np.linalg.norm(np.array([velocity.x, velocity.y, velocity.z])))
        self.meters_travelled += speed * self.config_closed_loop.carla_frame_rate
        self.infraction_recorder.check_infractions(
            step=self.step,
            meters_travelled=self.meters_travelled,
        )

    def _agent_init(self):
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()
        self.world_map = CarlaDataProvider.get_map()
        self.stop_sign_criteria = RunStopSign(self.world, self.world_map)
        self.vehicles_all = []
        self.walkers_all = []

        if self.config.use_new_bev_obs:
            self.bev_semantics_manager = ObsManager2(self.config)
            self.bev_semantics_manager.attach_ego_vehicle(
                self.vehicle,
                self.stop_sign_criteria,
                self.world_map,
                self.dense_global_plan_world_coord,
            )
        else:
            self.bev_semantics_manager = ObsManager(self.config)
            self.bev_semantics_manager.attach_ego_vehicle(
                self.vehicle,
                self.stop_sign_criteria,
                self.world_map,
            )

        all_actors = self.world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = rl_u.get_traffic_light_waypoints(
                    actor,
                    self.world_map,
                )
                self.list_traffic_lights.append((actor, center, waypoints))

        self.route_planner = RoutePlanner()
        self.route_planner.set_route(self.dense_global_plan_world_coord)

        self.last_lstm_states = []
        for _ in range(self.model_count):
            self.last_lstm_states.append(
                (
                    torch.zeros(
                        self.config.num_lstm_layers,
                        1,
                        self.config.features_dim,
                        device=self.device,
                    ),
                    torch.zeros(
                        self.config.num_lstm_layers,
                        1,
                        self.config.features_dim,
                        device=self.device,
                    ),
                ),
            )
        self.done = torch.zeros(1, device=self.device)

        self.video_recorder = VideoRecorder(
            config_closed_loop=self.config_closed_loop,
            vehicle=self.vehicle,
            world=self.world,
            step_counter=self.step,
        )

        self.initialized = True

    def _preprocess_observation(self, waypoint_route):
        self.stop_sign_criteria.tick(self.vehicle)
        actors = self.world.get_actors()
        self.vehicles_all = actors.filter("*vehicle*")
        self.walkers_all = actors.filter("*walker*")
        self.static_all = actors.filter("*static*")

        bev_semantics = self.bev_semantics_manager.get_observation(
            waypoint_route,
            self.vehicles_all,
            self.walkers_all,
            self.static_all,
            debug=False,
        )
        observations = {"bev_semantics": bev_semantics["bev_semantic_classes"]}

        last_control = self.vehicle.get_control()
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()
        forward_vec = transform.get_forward_vector()

        np_vel = np.array([velocity.x, velocity.y, velocity.z])
        np_fvec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
        forward_speed = np.dot(np_vel, np_fvec)

        np_vel_2d = np.array([velocity.x, velocity.y])
        velocity_ego_frame = rl_u.inverse_conversion_2d(
            np_vel_2d,
            np.zeros(2),
            np.deg2rad(transform.rotation.yaw),
        )

        speed_limit = self.vehicle.get_speed_limit()
        if isinstance(speed_limit, float):
            maximum_speed = speed_limit / 3.6
        else:
            maximum_speed = self.config.rr_maximum_speed

        measurements = [
            last_control.steer,
            last_control.throttle,
            last_control.brake,
            float(last_control.gear),
            float(velocity_ego_frame[0]),
            float(velocity_ego_frame[1]),
            float(forward_speed),
            maximum_speed,
        ]

        if self.config.use_extra_control_inputs:
            left_wheel = self.vehicle.get_wheel_steer_angle(
                carla.VehicleWheelLocation.FL_Wheel,
            )
            right_wheel = self.vehicle.get_wheel_steer_angle(
                carla.VehicleWheelLocation.FR_Wheel,
            )
            avg_wheel = 0.5 * (left_wheel + right_wheel)
            avg_wheel /= self.config.max_avg_steer_angle

            measurements.append(avg_wheel)

            last_error = last_control.steer - self.last_wheel_angle

            self.past_wheel_errors.append(last_error)
            error_derivative = self.past_wheel_errors[-1] - self.past_wheel_errors[-2]
            error_integral = sum(self.past_wheel_errors) / len(self.past_wheel_errors)

            measurements.append(last_error)
            measurements.append(error_derivative)
            measurements.append(error_integral)

            self.last_wheel_angle = avg_wheel

        if self.config.use_target_point:
            measurements.append(bev_semantics["target_point"][0])
            measurements.append(bev_semantics["target_point"][1])

        observations["measurements"] = np.array(measurements, dtype=np.float32)

        # Value measurements only feed the value head, not the policy. Zero them out for inference.
        value_measurements = [0.0, 0.0, 0.0]
        if self.config.use_ttc:
            value_measurements.append(0.0)
        if self.config.use_comfort_infraction:
            value_measurements.extend([0.0] * 6)

        observations["value_measurements"] = np.array(
            value_measurements,
            dtype=np.float32,
        )

        return observations

    def _get_waypoint_route(self):
        ego_vehicle_transform = self.vehicle.get_transform()
        pos = ego_vehicle_transform.location
        pos = np.array([pos.x, pos.y])
        waypoint_route = self.route_planner.run_step(pos)
        return waypoint_route

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1

        if not self.initialized:
            self._agent_init()
            if self.config.use_extra_control_inputs:
                from collections import deque

                self.last_wheel_angle = 0.0
                self.past_wheel_errors = deque(
                    [0.0 for _ in range(int(1.0 * self.config.frame_rate))],
                    maxlen=int(1.0 * self.config.frame_rate),
                )
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.last_control = control
            self._update_infraction_tracking()
            return control

        self._update_infraction_tracking()

        self.video_recorder.update_step(self.step)
        self.video_recorder.move_demo_cameras_with_ego()

        if (
            self.config_closed_loop.save_path is not None
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            self.video_recorder.save_demo_cameras()

        if self.step % self.total_action_repeat != 0:
            return self.last_control

        waypoint_route = self._get_waypoint_route()
        obs = self._preprocess_observation(waypoint_route)

        obs_tensor = {
            "bev_semantics": torch.Tensor(obs["bev_semantics"][np.newaxis, ...]).to(
                self.device,
                dtype=torch.float32,
            ),
            "measurements": torch.Tensor(obs["measurements"][np.newaxis, ...]).to(
                self.device,
                dtype=torch.float32,
            ),
            "value_measurements": torch.Tensor(
                obs["value_measurements"][np.newaxis, ...],
            ).to(self.device, dtype=torch.float32),
        }

        actions = []
        for i in range(self.model_count):
            action, _, _, _, _, _, _, _, _, _, self.last_lstm_states[i] = self.agents[
                i
            ].forward(
                obs_tensor,
                sample_type=self.sample_type,
                lstm_state=self.last_lstm_states[i],
                done=self.done,
            )
            actions.append(action)

        action = torch.stack(actions, dim=0).mean(dim=0)[0].cpu().numpy()

        control = self._convert_action_to_control(action)
        self.last_control = control

        return control

    def _convert_action_to_control(self, action):
        if action[1] > 0.0:
            throttle = action[1]
            brake = 0.0
        else:
            throttle = 0.0
            brake = -action[1]
        return carla.VehicleControl(
            steer=float(action[0]),
            throttle=float(throttle),
            brake=float(brake),
        )

    def destroy(self, results=None):
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup_and_compress()
        del self.vehicles_all
        del self.walkers_all
