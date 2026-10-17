"""CPU-only B2D protocol regression; this does not start CARLA or score a benchmark."""

import os
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unidrivevla.closedloop.bench2drive as b2d
from unidrivevla.closedloop.bench2drive import (
    CAMERA_SHAPE,
    PID_ARGUMENTS,
    RING_SIZE,
    UniDriveVLAB2DClosedLoopAgent,
    _camera_features,
    _canonical_pose,
    _ego_pose,
    _gps_to_location,
    _load_devkit,
    _pid_coordinates,
    _route_command,
    load_main_contract,
)
from unidrivevla.closedloop.runtime import PolicyRuntime, _TemporalStateRing
from unidrivevla.adapters.unidrivevla import Profile, prepare_sample


class _StatefulPolicy:
    def __init__(self):
        self.state = None
        self.resets = 0

    def reset(self):
        self.state = None
        self.resets += 1

    def save(self):
        return self.state

    def load(self, state):
        self.state = state


def _check_temporal_ring():
    policy = _StatefulPolicy()
    ring = _TemporalStateRing(RING_SIZE, policy.reset, policy.save, policy.load)
    expected = [0] * RING_SIZE
    for tick in range(35):
        slot = ring.begin()
        if policy.state is None:
            policy.state = {"ticks": 0}
        policy.state["ticks"] += 1  # mutate in place, matching cached bank tensors
        expected[slot] += 1
        assert policy.state["ticks"] == expected[slot]
        ring.finish(slot)
    assert ring.step == 35
    assert [state["ticks"] for state in ring.slots] == expected
    ring.reset()
    assert ring.step == 0 and ring.slots == [None] * RING_SIZE and policy.state is None
    slot = ring.begin()
    policy.state = {"ticks": 1}
    ring.finish(slot)
    assert ring.slots[0] == {"ticks": 1}

    from torch import nn

    # PolicyRuntime expects an nn.Module for device/eval lifecycle; its ring wiring is checked without forward().
    class ModulePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.profile = SimpleNamespace(name="b2d", cameras=[])
            self.head = SimpleNamespace(reset_temporal=lambda: None)

        def training_runtime_state_dict(self):
            return {}

        def load_training_runtime_state_dict(self, _state, device=None):
            pass

    runtime = PolicyRuntime(ModulePolicy(), device="cpu")
    runtime.enable_temporal_ring(RING_SIZE)
    assert runtime.temporal_ring_size == RING_SIZE and runtime._ring.size == RING_SIZE
    runtime.reset()

    # Real bank hooks must restore ten-frame-old tensors, metadata and track ids, not the last simulator frame.
    import torch
    from unidrivevla.models.models.instance_bank import InstanceBank
    from unidrivevla.models.models.ego.instance_bank import EgoInstanceBank

    for bank in (InstanceBank(2, 4, np.zeros((2, 11), np.float32)),
                 EgoInstanceBank(4, feature_map_scale=(4, 4), with_instance_feat=True)):
        ring = _TemporalStateRing(RING_SIZE, bank.reset, bank.training_runtime_state_dict,
                                  bank.load_training_runtime_state_dict)
        for tick in range(25):
            slot = ring.begin()
            metas = {"timestamp": torch.tensor([tick / 20], dtype=torch.float64),
                     "img_metas": [{"stream_id": "b2d-route", "T_global": np.eye(4),
                                    "T_global_inv": np.eye(4)}]}
            kwargs = {"feature_maps": []} if isinstance(bank, EgoInstanceBank) else {}
            result = bank.get(1, metas, **kwargs)
            if tick < RING_SIZE:
                assert result[2] is None
            else:
                assert torch.all(result[2] == tick - RING_SIZE)
                if isinstance(bank, InstanceBank):
                    assert bank.prev_id == tick - RING_SIZE
                    assert torch.all(bank.instance_id == tick - RING_SIZE)
                assert abs(float(metas["timestamp"] - bank.metas["timestamp"]) - 0.5) < 1e-9
            bank.cached_feature = torch.full((1, bank.num_anchor, 4), float(tick))
            bank.cached_anchor = bank.anchor.detach().clone().unsqueeze(0)
            bank.metas = metas
            if isinstance(bank, InstanceBank):
                bank.prev_id = tick
                bank.instance_id = torch.full((1, bank.num_anchor), tick, dtype=torch.long)
            ring.finish(slot)
        ring.reset()
        assert bank.cached_feature is None and bank.cached_anchor is None and bank.metas is None
        if isinstance(bank, InstanceBank):
            assert bank.instance_id is None and bank.prev_id == 0


def _check_source_contract(root):
    with patch.dict(os.environ, {}, clear=True):
        base = load_main_contract(root)
    assert base["CAMERA"] == ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
                              "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    ids = [sensor["id"] for sensor in base["sensors"]]
    assert ids == [*base["CAMERA"], "IMU", "GPS", "SPEED"]
    for camera in base["sensors"][:6]:
        assert (camera["width"], camera["height"]) == (1600, 900)
        assert camera["type"] == "sensor.camera.rgb"
    mounts = [(s["x"], s["y"], s["z"], s["yaw"]) for s in base["sensors"][:6]]
    assert mounts == [(0.8, 0.0, 1.6, 0.0), (0.27, -0.55, 1.6, -55.0),
                      (0.27, 0.55, 1.6, 55.0), (-2.0, 0.0, 1.6, 180.0),
                      (-0.32, -0.55, 1.6, -110.0), (-0.32, 0.55, 1.6, 110.0)]
    assert base["sensors"][3]["fov"] == 110
    assert all(camera["fov"] == 70 for i, camera in enumerate(base["sensors"][:6]) if i != 3)
    assert base["sensors"][6]["sensor_tick"] == 0.05
    assert base["sensors"][7]["sensor_tick"] == 0.01
    assert base["sensors"][8]["reading_frequency"] == 20
    with patch.dict(os.environ, {"IS_BENCH2DRIVE": "1"}):
        with_bev = load_main_contract(root)
    assert [sensor["id"] for sensor in with_bev["sensors"]][-1] == "bev"
    assert with_bev["sensors"][-1] == {
        "type": "sensor.camera.rgb", "x": 0.0, "y": 0.0, "z": 60.0,
        "roll": 0.0, "pitch": -90.0, "yaw": 0.0, "width": 512, "height": 512,
        "fov": 50.0, "id": "bev",
    }
    assert np.allclose(base["LIDAR2EGO"], [[0, 1, 0, -0.39], [-1, 0, 0, 0],
                                           [0, 0, 1, 1.84], [0, 0, 0, 1]])
    assert np.allclose(base["CAM2IMG"]["CAM_FRONT"][:3, :3],
                       [[1142.51841, 0, 800], [0, 1142.51841, 450], [0, 0, 1]])
    assert np.allclose(base["CAM2IMG"]["CAM_BACK"][:3, :3],
                       [[560.166031, 0, 800], [0, 560.166031, 450], [0, 0, 1]])
    return base


def _check_calibrated_images(base):
    import cv2
    from PIL import Image

    package = Path(__file__).resolve().parents[1]
    profile_cfg = OmegaConf.load(package / "config" / "profile" / "b2d_ego.yaml")
    profile = Profile.from_config(OmegaConf.to_container(profile_cfg, resolve=True))
    y, x = np.indices(CAMERA_SHAPE)
    rgb = np.stack((x % 256, y % 256, (x // 3 + y // 2) % 256), axis=-1).astype(np.uint8)
    input_data = {}
    for index, name in enumerate(base["CAMERA"]):
        frame = np.empty((*CAMERA_SHAPE, 4), dtype=np.uint8)
        frame[..., :3] = (rgb.astype(np.uint16) + index).astype(np.uint8)
        frame[..., 3] = 255
        input_data[name] = (0.0, frame)
    cameras = _camera_features(input_data, base)
    assert list(cameras) == ["front", "front_left", "front_right", "back", "back_left", "back_right"]
    axes = np.eye(4)
    axes[:3, :3] = base["LIDAR2EGO"][:3, :3]
    for raw in base["CAMERA"]:
        view = raw.removeprefix("CAM_").lower()
        assert cameras[view]["image"].shape == (*CAMERA_SHAPE, 3)
        assert np.allclose(cameras[view]["intrinsic"], base["CAM2IMG"][raw][:3, :3])
        assert np.allclose(cameras[view]["sensor2ego"], axes @ np.linalg.inv(base["LIDAR2CAM"][raw]))

    # Compare model-ready pixels with the literal old agent path, after its second channel swap and resize/crop.
    tensors, _ = prepare_sample({
        "cameras": [{"view": view, **cameras[view], "size": (1600, 900)} for view in profile.cameras],
        "history_trajectory": np.zeros((3, 3), dtype=np.float32),
        "ego_velocity": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "ego_acceleration": np.zeros(3, dtype=np.float32),
        "ego_angular_velocity": np.zeros(3, dtype=np.float32),
        "ego_steer": 0.0, "command": "lane_follow", "timestamp": 0.0,
    }, {}, profile)
    height, width = profile.image_hw
    scale = max(height / CAMERA_SHAPE[0], width / CAMERA_SHAPE[1])
    resize_dims = (int(CAMERA_SHAPE[1] * scale), int(CAMERA_SHAPE[0] * scale))
    crop_w = int(max(0, resize_dims[0] - width) / 2)
    crop = (crop_w, 0, crop_w + width, height)
    source = Path(__file__).resolve().parents[2] / "Bench2Drive/projects/mmdet3d_plugin/datasets/pipelines/augment.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ResizeCropFlipImage")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_img_transform")
    scope = {"np": np, "Image": Image}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    for raw in base["CAMERA"]:
        view = raw.removeprefix("CAM_").lower()
        main_rgb = cv2.cvtColor(input_data[raw][1][:, :, :3], cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".jpg", main_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        assert ok
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        main_pipeline_input = cv2.cvtColor(decoded_bgr, cv2.COLOR_RGB2BGR)
        expected, _ = scope["_img_transform"](None, main_pipeline_input,
                                               {"resize": scale, "crop": crop, "flip": False, "rotate": 0})
        prepared = tensors["img"][profile.cameras.index(view)].numpy().transpose(1, 2, 0)
        assert np.array_equal(prepared, expected), f"{raw} differs from the main JPEG/channel/resize/crop path"

    pos = np.array([12.5, -4.0])
    compass = 0.23
    ego = _ego_pose(pos, compass)
    actual = _canonical_pose(ego, base["LIDAR2EGO"])
    lidar_global = ego @ base["LIDAR2EGO"]
    axes = np.eye(4)
    axes[:3, :3] = base["LIDAR2EGO"][:3, :3]
    assert np.allclose(actual, lidar_global @ np.linalg.inv(axes))
    assert not np.allclose(actual[:2, 3], [pos[0], -pos[1]])  # preserve the main lidar-origin offset


class _RoutePlanner:
    def run_step(self, _pos):
        return np.zeros(2), SimpleNamespace(value=3)


class _PID:
    def __init__(self):
        self.call = None

    def control_pid(self, waypoints, speed, target):
        self.call = (waypoints.copy(), speed, target.copy())
        return 0.4, 0.2, 0.1, {"pid": "main"}


class _Runtime:
    def __init__(self):
        self.profile = SimpleNamespace(commands=["left", "right", "straight", "lane_follow",
                                                 "change_lane_left", "change_lane_right"], horizon=6)
        self.call = None

    def predict(self, **kwargs):
        self.call = kwargs
        return np.array([[1, 0], [2, 0.1], [3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]], dtype=np.float64)


class _VehicleControl:
    def __init__(self):
        self.steer = self.throttle = self.brake = 0.0


def _check_frame_to_control(base):
    agent = UniDriveVLAB2DClosedLoopAgent.__new__(UniDriveVLAB2DClosedLoopAgent)
    agent.contract = base
    agent._route_planner = _RoutePlanner()
    agent.lat_ref, agent.lon_ref = 42.0, 2.0
    agent.step = -1
    agent.runtime = _Runtime()
    agent.pidcontroller = _PID()
    agent.pid_metadata = {"steer": 0.35}
    input_data = {}
    for index, name in enumerate(base["CAMERA"]):
        frame = np.full((*CAMERA_SHAPE, 4), [20 + index, 80 + index, 180 - index, 255], dtype=np.uint8)
        input_data[name] = (0.0, frame)
    input_data["GPS"] = (0.0, np.array([42.0, 2.0, 0.0]))
    input_data["SPEED"] = (0.0, {"speed": 8.0})
    input_data["IMU"] = (0.0, np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.23]))
    with patch.dict(sys.modules, {"carla": SimpleNamespace(VehicleControl=_VehicleControl)}):
        control = agent.run_step(input_data, timestamp=999.0)
    call = agent.runtime.call
    assert call["timestamp"] == 0.0  # simulator contract is fixed 20 Hz, independent of callback timestamp
    assert call["command"] == "straight"
    assert np.allclose(call["ego_velocity"], [8, 0, 0])
    assert np.allclose(call["ego_acceleration"], [1, -2, 3])
    assert np.allclose(call["ego_angular_velocity"], [-0.1, -0.2, -0.3])
    assert call["ego_steer"] == 0.35
    assert np.allclose(call["ego2global"], _canonical_pose(_ego_pose([0, 0], 0.23), base["LIDAR2EGO"]))
    assert list(call["cameras"]) == ["front", "front_left", "front_right", "back", "back_left", "back_right"]
    pid_waypoints, speed, target = agent.pidcontroller.call
    assert np.allclose(pid_waypoints, [[0, 1], [-0.1, 2], [-0.2, 3], [-0.3, 4], [-0.4, 5], [-0.5, 6]])
    assert np.allclose(target, pid_waypoints[-1]) and float(speed) == 8.0
    assert (control.steer, control.throttle, control.brake) == (0.4, 0.2, 0.0)
    assert agent.pid_metadata["steer"] == 0.4
    input_data["IMU"] = (0.05, np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, np.nan]))
    with patch.dict(sys.modules, {"carla": SimpleNamespace(VehicleControl=_VehicleControl)}):
        agent.run_step(input_data, timestamp=999.05)
    assert agent.runtime.call["timestamp"] == 0.05
    assert np.allclose(agent.runtime.call["ego_acceleration"], [0, 0, 0])
    assert np.allclose(agent.runtime.call["ego_angular_velocity"], [0, 0, 0])


def _check_setup_runtime_options():
    class Runtime:
        def __init__(self):
            self.ring_size = None

        def enable_temporal_ring(self, size):
            self.ring_size = size

    for environment, expected_frame, expected_motion in (
        ({}, None, None),
        ({"UNIDRIVE_CHECKPOINT_FRAME": "lidar", "UNIDRIVE_WITH_MOTION": "1"}, "lidar", True),
        ({"UNIDRIVE_CHECKPOINT_FRAME": "ego", "UNIDRIVE_WITH_MOTION": "0"}, "ego", False),
    ):
        runtime = Runtime()
        with patch.dict(os.environ, {"UNIDRIVE_DEVICE": "cpu", **environment}, clear=True), \
                patch.object(b2d, "LEADERBOARD_AVAILABLE", True), \
                patch.object(b2d, "load_main_contract", return_value={"leaderboard_root": Path("/devkit")}), \
                patch.object(b2d, "_load_devkit", return_value=(object, object)), \
                patch.object(b2d, "_checkpoint_path", return_value="/checkpoint.ckpt"), \
                patch.object(b2d, "load_runtime", return_value=runtime) as load_runtime:
            agent = b2d.UniDriveVLAB2DClosedLoopAgent.__new__(b2d.UniDriveVLAB2DClosedLoopAgent)
            agent.setup("b2d+checkpoint.ckpt")
            load_runtime.assert_called_once_with(
                "b2d", "/checkpoint.ckpt", "cpu", with_motion=expected_motion,
                checkpoint_frame=expected_frame,
            )
            assert runtime.ring_size == RING_SIZE

    for variable, value in (("UNIDRIVE_CHECKPOINT_FRAME", "sensor"),
                            ("UNIDRIVE_WITH_MOTION", "true")):
        with patch.dict(os.environ, {variable: value}, clear=True), \
                patch.object(b2d, "LEADERBOARD_AVAILABLE", True), \
                patch.object(b2d, "load_runtime") as load_runtime:
            agent = b2d.UniDriveVLAB2DClosedLoopAgent.__new__(b2d.UniDriveVLAB2DClosedLoopAgent)
            try:
                agent.setup("b2d+checkpoint.ckpt")
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid {variable} was accepted")
            load_runtime.assert_not_called()


def _check_legacy_utilities(root):
    planner_class, pid_class = _load_devkit(root)
    pid = pid_class(**PID_ARGUMENTS)
    assert pid.use_target_to_aim is False and pid.max_throttle == 0.75
    left, right = SimpleNamespace(value=1), SimpleNamespace(value=2)
    route = [({"lat": 42.0, "lon": 2.0}, left), ({"lat": 42.0, "lon": 2.001}, right)]
    planner = planner_class(4.0, 50.0, lat_ref=42.0, lon_ref=2.0)
    planner.set_route(route, True)
    _, command = planner.run_step(_gps_to_location([42.0, 2.0], 42.0, 2.0))
    assert command is right


def main():
    root = Path(__file__).resolve().parents[2] / "Bench2Drive" / "bench2drive" / "leaderboard"
    base = _check_source_contract(root)
    _check_legacy_utilities(root)
    _check_calibrated_images(base)
    assert [_route_command(value, ["left", "right", "straight", "lane_follow", "change_lane_left",
                                   "change_lane_right"]) for value in (1, 2, 3, 4, 5, 6, -1, 0)] == [
        "left", "right", "straight", "lane_follow", "change_lane_left", "change_lane_right",
        "lane_follow", "change_lane_right",
    ]
    assert _pid_coordinates([[1, 2], [3, 4]]).tolist() == [[-2.0, 1.0], [-4.0, 3.0]]
    assert PID_ARGUMENTS == {
        "turn_KP": 1.0, "turn_KI": 0.75, "turn_KD": 0.0, "turn_n": 10,
        "speed_KP": 3.0, "speed_KI": 0.5, "speed_KD": 1.0, "speed_n": 10,
        "brake_speed": 0.4, "brake_ratio": 1.1, "clip_delta": 0.25, "max_throttle": 0.75,
        "aim_dist_low": 2.0, "aim_dist_high": 8.0, "anti_windup": True, "use_target_to_aim": False,
    }
    _check_frame_to_control(base)
    _check_temporal_ring()
    _check_setup_runtime_options()
    print("PASS: B2D static sensors/calibration, JPEG RGB path, 20 Hz status/command, PID frames, "
          "runtime frame/motion options, and 10-slot state ring")
    print("CPU protocol regression only; no CARLA rollout or benchmark score was produced.")


if __name__ == "__main__":
    main()
