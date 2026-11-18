"""LEAD Leaderboard front end for a separate UniDriveVLA HTTP server.

Protocol source pinned at LEAD main 3764f79698922896638638df8062f2e064034610.
This file uses LEAD's BaseAgent sensor/localization path and WaypointTracker;
the official evaluator is launched by scripts/infer/run_lead.sh.
"""

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from unidrivevla.closedloop.lead_protocol import (
    LEAD_CAMERA_COUNT,
    LEAD_HORIZON,
    LEAD_INPUT_HZ,
    LEAD_POSE_INTERVAL_S,
    LEAD_CAMERA_VIEWS,
    LeadHTTPClient,
)

try:
    import carla
    from leaderboard.autoagents import autonomous_agent
    from lead.api.abstract_driving_agent import AbstractDrivingAgent
    from lead.api.py123d_log_api import (
        CAMERA_ID_BY_LEAD_INDEX,
        CARLA_LINCOLN_MKZ_2020_METADATA,
        localized_pose_to_se3_matrix,
    )
    from lead.common import carla_to_123d
    from lead.common.base_agent import BaseAgent
    from lead.config import load_lead_config
    from lead.evaluation.inference.trackers import WaypointTracker
    from py123d.datatypes.sensors.base_camera import CameraID

    _LEAD_IMPORT_ERROR = None
except ImportError as exc:  # Permit CPU protocol tests without a CARLA/LEAD installation.
    carla = None
    autonomous_agent = SimpleNamespace(Track=SimpleNamespace(SENSORS="SENSORS"))
    AbstractDrivingAgent = object
    _LEAD_IMPORT_ERROR = exc


CAMERA_IDS = tuple(CAMERA_ID_BY_LEAD_INDEX[index] for index in range(1, 7)) if _LEAD_IMPORT_ERROR is None else ()


def _agent_config_path(value):
    raw = str(value or "")
    path = Path(raw.split("+", 1)[0]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"LEAD agent-config must be an existing YAML file: {path}")
    return path.resolve()


class UniDriveVLALEADAgent(AbstractDrivingAgent):
    """Use official LEAD sensor/localization primitives and a CPU PID frontend."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        if _LEAD_IMPORT_ERROR is not None:
            raise RuntimeError("LEAD/CARLA Python API is unavailable in this frontend environment") from _LEAD_IMPORT_ERROR
        import yaml

        from unidrivevla.data.lead import lead_inference_features

        config_path = _agent_config_path(path_to_conf_file)
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict) or set(config) - {"server_url", "timeout"}:
            raise ValueError("LEAD agent-config may contain only server_url and timeout")
        if not isinstance(config.get("server_url"), str):
            raise ValueError("LEAD agent-config requires server_url")

        self.lead_config = load_lead_config()
        if len(self.lead_config.expert.sensor_rig.cameras) != LEAD_CAMERA_COUNT:
            raise ValueError("current UniDriveVLA LEAD profile requires the official six-camera rig")
        if len(CAMERA_IDS) != LEAD_CAMERA_COUNT or len(LEAD_CAMERA_VIEWS) != LEAD_CAMERA_COUNT:
            raise RuntimeError("LEAD camera API mapping is incomplete")
        if LEAD_HORIZON * LEAD_POSE_INTERVAL_S != 2.0:
            raise RuntimeError("LEAD profile plan must span exactly 2.0 seconds at 4 Hz")
        fps = self.lead_config.expert.simulation.carla_fps
        save_ticks = self.lead_config.expert.data_collection.data_save_freq
        if fps != LEAD_INPUT_HZ or save_ticks != 5:
            raise ValueError("LEAD adapter requires the pinned 20 Hz input and 5-tick / 4 Hz training cadence")
        self.policy = SimpleNamespace(input_cameras=list(CAMERA_IDS))
        self.step = -1
        self.initialized = False
        self.track = autonomous_agent.Track.SENSORS
        BaseAgent.setup(self, self.lead_config, sensor_agent=True, past_window_num_iterations=30)
        self.lidar_sweep_queue = deque(maxlen=0)
        self.radar_sweep_queue = deque(maxlen=0)
        self._camera_metadatas = carla_to_123d.build_pinhole_camera_metadatas(
            self.lead_config.expert,
            CARLA_LINCOLN_MKZ_2020_METADATA,
            perturbation_translation=0.0,
            perturbation_rotation=0.0,
        )
        self.waypoint_tracker = WaypointTracker(self.lead_config)
        self.client = LeadHTTPClient(config["server_url"], config.get("timeout", 60.0))
        self.client.reset()
        self._lead_inference_features = lead_inference_features

    def _init(self):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self.initialized = True

    def set_scenario(self, scenario):
        # The official evaluator owns score/infraction accounting; this adapter does not shadow it.
        self._scenario = scenario

    def compute_control(self, prediction, features):
        import torch

        waypoints = np.asarray(prediction, dtype=np.float32)
        if waypoints.shape != (LEAD_HORIZON, 2) or not np.isfinite(waypoints).all():
            raise ValueError(f"UniDriveVLA LEAD prediction must be finite {LEAD_HORIZON}x2 at 0.25 s intervals")
        tensor = torch.from_numpy(np.ascontiguousarray(waypoints)).unsqueeze(0)
        speed = torch.tensor([[float(features["speed"])]], dtype=torch.float32)
        control = self.waypoint_tracker.step(tensor, speed)
        return carla.VehicleControl(steer=float(control.steer), throttle=float(control.throttle),
                                    brake=float(control.brake))

    def run_step(self, sensor_data, timestamp, vehicle=None):
        self.step += 1
        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.tick(sensor_data)
            return self.control

        sensor_data = self.tick(sensor_data)
        scene_data = self.build_scene_data(sensor_data)
        required_history = round(1.5 * self.lead_config.expert.simulation.carla_fps) + 1
        observed_history_length = len(scene_data.past_ego_positions)
        if observed_history_length < required_history:
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            return self.control
        current_pose = localized_pose_to_se3_matrix(sensor_data["localized_position"], self.compass)
        timestamp_s = self.step / self.lead_config.expert.simulation.carla_fps
        # This is explicitly the current GNSS/IMU-localized pose; the privileged imu_se3 label is never read.
        scene_data.driving_meta = {"localized_ego_state_se3": current_pose}
        features = self._lead_inference_features(
            scene_data,
            ego2global=current_pose,
            timestamp_seconds=timestamp_s,
        )
        if features.get("command") != "unknown":
            raise ValueError("LEAD inference features must use the canonical 'unknown' command")
        if (not np.isclose(float(features["timestamp"]), timestamp_s)
                or not np.allclose(features["ego2global"], current_pose)):
            raise ValueError("LEAD feature builder changed the supplied localized pose or tick timestamp")
        if np.shape(features.get("history_trajectory")) != (4, 3):
            raise ValueError("LEAD feature builder must provide the trained 4x3 localized history")
        plan = self.client.predict(features, current_pose, timestamp_s, command="unknown")
        control = self.compute_control(plan, {"speed": sensor_data["speed"]})
        if self.step < self.lead_config.expert.simulation.inital_frames_delay:
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.control = control
        return control

    def destroy(self, results=None):
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.reset()
            except Exception:
                pass


def get_entry_point():
    return "UniDriveVLALEADAgent"
