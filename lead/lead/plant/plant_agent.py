"""PlanT closed-loop CARLA leaderboard agent.

Uses MAP track (ground-truth bounding boxes from CARLA) instead of sensor
inputs.  Tokenizes bounding boxes, runs the PlanT model, and converts
planning outputs to vehicle control via PID controllers.
"""

from __future__ import annotations

import json
import logging
import os

import carla
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.pid_controller import LateralPIDController, PIDController, get_throttle
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.plant.plant_config import PlantConfig
from lead.plant.plant_dataset import plant_collate_fn
from lead.plant.plant_tokenizer import tokenize_bboxes
from lead.plant.plant_variables import PlantVariables
from lead.training.training_utils import create_model

LOG = logging.getLogger(__name__)


def get_entry_point():
    return "PlantAgent"


class PlantAgent(BaseAgent, autonomous_agent.AutonomousAgent):
    """CARLA leaderboard agent for PlanT evaluation on MAP track."""

    @beartype
    def setup(self, path_to_conf_file: str, _=None, __=None):
        self.config_closed_loop = ClosedLoopConfig()
        super().setup(sensor_agent=False)
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")

        # Load training config
        with open(
            os.path.join(path_to_conf_file, "config.json"),
            encoding="utf-8",
        ) as f:
            json_config = json.loads(f.read())

        self.training_config = PlantConfig(json_config)

        # Load model
        self.net = create_model(self.training_config)
        model_path = os.path.join(path_to_conf_file, "model_0.pth")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self.net.cuda(device=self.device).eval()

        # PID controllers
        self.lateral_controller = LateralPIDController(self.config_closed_loop)
        self.longitudinal_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )

        self.speed_cats = PlantVariables.speed_cats
        self.cleared_stop_sign = False

    def sensors(self):
        return [
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
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "speed",
            },
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

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        if not self.initialized:
            self.initialized = True

        speed = input_data["speed"][1]["speed"]
        compass = common_utils.preprocess_compass(input_data["imu"][1][-1])
        yaw = common_utils.normalize_angle(compass)

        ego_transform = self._vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        gps = np.array(
            [
                self._vehicle.get_location().x,
                self._vehicle.get_location().y,
            ],
        )

        # Get route waypoints
        self._waypoint_planner.load()
        (_, _, _, _, _, _, _, speed_limit) = self._waypoint_planner.run_step(gps)
        route_points = self._waypoint_planner.original_route_points[
            self._waypoint_planner.route_index :
        ][:20, :2]
        self._waypoint_planner.save()

        route_ego = np.array(
            [common_utils.inverse_conversion_2d(p, gps, yaw) for p in route_points],
        )

        # Get bounding boxes from CARLA ground truth
        raw_boxes = self._get_bounding_boxes(ego_matrix, ego_yaw)

        # Tokenize
        plant_objects = tokenize_bboxes(
            boxes=[{}] + raw_boxes,  # prepend dummy ego entry (gets skipped)
            plant_range=self.training_config.plant_range,
            plant_range_factor_front=self.training_config.plant_range_factor_front,
            plant_input_static_cars=self.training_config.plant_input_static_cars,
        )

        # Build batch
        sample = {
            "plant_objects": plant_objects,
            "route_original": torch.tensor(
                self._pad_route(
                    route_ego.tolist(),
                    self.training_config.plant_num_route_points,
                ),
                dtype=torch.float32,
            ),
            "speed_limit": torch.tensor(
                self.speed_cats.get(round(speed_limit * 3.6), 0),
                dtype=torch.int,
            ),
            "input_ego_speed": torch.tensor(speed, dtype=torch.float32),
            "future_waypoints": torch.zeros(
                1,
                self.training_config.num_way_points_prediction,
                2,
            ),
            "route": torch.zeros(
                1,
                self.training_config.num_route_points_prediction,
                2,
            ),
            "target_speed": torch.zeros(1),
            "brake": torch.zeros(1, dtype=torch.bool),
            "speed": torch.tensor(speed, dtype=torch.float32),
        }
        batch = plant_collate_fn([sample])
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()
        }

        # Forward pass
        prediction = self.net(batch)

        # Extract controls from prediction
        steer, throttle, brake = self._prediction_to_control(prediction, speed)

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 1.0))

        # Hold brake for initial frames
        if self.step < 40:
            control = carla.VehicleControl(0.0, 0.0, 1.0)

        return control

    def _prediction_to_control(self, prediction, speed):
        """Convert model prediction to (steer, throttle, brake)."""
        # Target speed
        if prediction.pred_target_speed_distribution is not None:
            speed_dist = F.softmax(
                prediction.pred_target_speed_distribution[0].float().cpu(),
                dim=-1,
            )
            target_speeds = torch.tensor(
                self.training_config.target_speed_classes,
                dtype=torch.float32,
            )
            desired_speed = float((speed_dist * target_speeds).sum())
        elif prediction.pred_target_speed_scalar is not None:
            desired_speed = float(prediction.pred_target_speed_scalar[0].cpu())
        else:
            desired_speed = 0.0

        brake = (
            desired_speed < 0.01
            or (speed / max(desired_speed, 0.01)) > self.config_closed_loop.brake_ratio
        )

        throttle, brake = get_throttle(
            brake,
            desired_speed,
            speed,
            self.config_expert,
        )

        # Steering from route prediction
        if prediction.pred_route is not None:
            route_points = prediction.pred_route[0].detach().cpu().float().numpy()
            steer = self.lateral_controller.step(
                route_points,
                speed,
                0.0,
                0.0,
                sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
            )
        elif prediction.pred_future_waypoints is not None:
            waypoints = (
                prediction.pred_future_waypoints[0].detach().cpu().float().numpy()
            )
            steer = self.lateral_controller.step(
                waypoints,
                speed,
                0.0,
                0.0,
                sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
            )
        else:
            steer = 0.0

        return steer, throttle, float(brake)

    def _get_bounding_boxes(self, ego_matrix, ego_yaw):
        """Get bounding boxes from CARLA ground truth (MAP track)."""
        vehicles = CarlaDataProvider.get_world().get_actors().filter("*vehicle*")
        walkers = CarlaDataProvider.get_world().get_actors().filter("*walker*")

        result = []

        for actor in list(vehicles) + list(walkers):
            if actor.id == self._vehicle.id:
                continue

            actor_transform = actor.get_transform()
            bb = actor.bounding_box

            # Relative position
            relative_pos = common_utils.get_relative_transform(
                ego_matrix,
                np.array(actor_transform.get_matrix()),
            )
            relative_yaw = common_utils.normalize_angle(
                np.deg2rad(actor_transform.rotation.yaw) - ego_yaw,
            )

            actor_velocity = actor.get_velocity()
            actor_speed = np.sqrt(
                actor_velocity.x**2 + actor_velocity.y**2 + actor_velocity.z**2,
            )

            actor_class = "car" if "vehicle" in actor.type_id else "walker"

            result.append(
                {
                    "class": actor_class,
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "speed": actor_speed,
                    "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                    "type_id": actor.type_id,
                    "id": actor.id,
                },
            )

        return result

    @staticmethod
    def _pad_route(route, target_len):
        while len(route) < target_len:
            route.append(route[-1] if route else [0.0, 0.0])
        return route[:target_len]

    def destroy(self, results=None):
        del self.net
