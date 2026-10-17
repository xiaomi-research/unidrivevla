"""Thin CARLA Leaderboard adapter for the bundled UniDriveVLA Bench2Drive baseline."""

import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from unidrivevla.serve import load_runtime

try:
    from leaderboard.autoagents import autonomous_agent
    LEADERBOARD_AVAILABLE = True
except ImportError:  # Keep calibration helpers importable for offline CPU checks.
    autonomous_agent = SimpleNamespace(AutonomousAgent=object, Track=SimpleNamespace(SENSORS="SENSORS"))
    LEADERBOARD_AVAILABLE = False


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEADERBOARD_ROOT = CODE_ROOT / "Bench2Drive" / "bench2drive" / "leaderboard"
MAIN_AGENT = Path("team_code") / "unidrivevla_b2d_agent.py"
CAMERA_SHAPE = (900, 1600)
FRAME_RATE = 20
RING_SIZE = 10  # main decoder: 20 Hz close loop / 2 Hz training cadence
PID_ARGUMENTS = {
    "turn_KP": 1.0, "turn_KI": 0.75, "turn_KD": 0.0, "turn_n": 10,
    "speed_KP": 3.0, "speed_KI": 0.5, "speed_KD": 1.0, "speed_n": 10,
    "brake_speed": 0.4, "brake_ratio": 1.1, "clip_delta": 0.25, "max_throttle": 0.75,
    "aim_dist_low": 2.0, "aim_dist_high": 8.0, "anti_windup": True, "use_target_to_aim": False,
}


def _safe_literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_safe_literal(key): _safe_literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_literal(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _safe_literal(node.left) * _safe_literal(node.right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "np"
            and node.func.attr == "array" and len(node.args) == 1 and not node.keywords):
        return np.asarray(_safe_literal(node.args[0]))
    raise ValueError(f"unsupported expression in the legacy B2D contract: {ast.dump(node, include_attributes=False)}")


def _assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return _safe_literal(node.value)
    raise ValueError(f"legacy B2D agent is missing static {name}")


def _sensors_from_main(tree, include_bev):
    owner = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "UniDriveVLAB2DAgent"), None)
    method = next((node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "sensors"), None) \
        if owner else None
    if method is None:
        raise ValueError("legacy B2D agent is missing UniDriveVLAB2DAgent.sensors")
    base = next((statement.value for statement in method.body
                 if isinstance(statement, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "sensors" for target in statement.targets)), None)
    if base is None:
        raise ValueError("legacy B2D sensor list is not a literal assignment")
    result = _safe_literal(base)
    optional = []
    for statement in method.body:
        if (isinstance(statement, ast.If) and isinstance(statement.test, ast.Name)
                and statement.test.id == "IS_BENCH2DRIVE"):
            for child in statement.body:
                if (isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name)
                        and child.target.id == "sensors" and isinstance(child.op, ast.Add)):
                    optional.extend(_safe_literal(child.value))
    if include_bev:
        result.extend(optional)
    return result


def load_main_contract(leaderboard_root=None):
    """Read calibration and sensor literals from the existing main B2D agent without importing it."""
    root = Path(leaderboard_root or os.environ.get("B2D_LEADERBOARD_ROOT") or DEFAULT_LEADERBOARD_ROOT).expanduser()
    root = root.resolve()
    source = root / MAIN_AGENT
    if not source.is_file():
        raise FileNotFoundError(f"B2D main agent not found at {source}; set B2D_LEADERBOARD_ROOT to its leaderboard directory")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values = {name: _assignment(tree, name) for name in ("LIDAR2CAM", "CAM2IMG", "LIDAR2EGO", "CAMERA")}
    cameras = values["CAMERA"]
    if len(cameras) != 6 or len(set(cameras)) != 6:
        raise ValueError("legacy B2D camera contract must contain six unique cameras")
    if set(values["LIDAR2CAM"]) != set(cameras) or set(values["CAM2IMG"]) != set(cameras):
        raise ValueError("legacy B2D camera calibration keys do not match its camera list")
    if any(values["LIDAR2CAM"][name].shape != (4, 4) or values["CAM2IMG"][name].shape != (4, 4)
           for name in cameras) or values["LIDAR2EGO"].shape != (4, 4):
        raise ValueError("legacy B2D calibration matrices must be 4x4")
    values["sensors"] = _sensors_from_main(tree, bool(os.environ.get("IS_BENCH2DRIVE")))
    values["leaderboard_root"] = root
    values["source"] = source
    return values


def _load_legacy_class(path, name, module_name):
    if not path.is_file():
        raise FileNotFoundError(f"B2D devkit utility is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load B2D devkit utility: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"{path} does not define {name}") from exc


def _load_devkit(leaderboard_root):
    team_code = leaderboard_root / "team_code"
    suffix = str(abs(hash(str(leaderboard_root))))
    planner = _load_legacy_class(team_code / "planner.py", "RoutePlanner", f"_unidrivevla_b2d_planner_{suffix}")
    pid = _load_legacy_class(team_code / "pid_controller.py", "PIDController", f"_unidrivevla_b2d_pid_{suffix}")
    return planner, pid


def _checkpoint_path(value):
    raw = str(value or "")
    candidates = [raw]
    if "+" in raw:
        candidates.extend(raw.split("+"))
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
    raise FileNotFoundError(f"agent-config must name an existing UniDriveVLA checkpoint; got {raw!r}")


def _closed_loop_options():
    checkpoint_frame = os.environ.get("UNIDRIVE_CHECKPOINT_FRAME")
    if checkpoint_frame not in (None, "ego", "lidar"):
        raise ValueError("UNIDRIVE_CHECKPOINT_FRAME must be 'ego' or 'lidar'")
    motion = os.environ.get("UNIDRIVE_WITH_MOTION")
    if motion not in (None, "0", "1"):
        raise ValueError("UNIDRIVE_WITH_MOTION must be '0' or '1'")
    return checkpoint_frame, None if motion is None else motion == "1"


def _payload(input_data, name):
    if name not in input_data or not isinstance(input_data[name], (tuple, list)) or len(input_data[name]) < 2:
        raise ValueError(f"missing B2D sensor payload {name!r}")
    return input_data[name][1]


def _jpeg20_rgb(frame):
    import cv2

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
        raise ValueError("CARLA camera frame must be uint8 HxWx3/4")
    image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
    if not ok:
        raise RuntimeError("OpenCV could not encode the CARLA camera frame")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("OpenCV could not decode its JPEG camera frame")
    # Preserve the decoded values: main performs RGB2BGR once more, and prepare_sample
    # performs that same final swap when converting this RGB camera input to model BGR.
    return decoded_bgr


def _camera_features(input_data, contract):
    axes = np.eye(4, dtype=np.float64)
    axes[:3, :3] = contract["LIDAR2EGO"][:3, :3]
    cameras = {}
    for name in contract["CAMERA"]:
        frame = np.asarray(_payload(input_data, name))
        if frame.shape[:2] != CAMERA_SHAPE:
            raise ValueError(f"{name} must be a 1600x900 CARLA image, got {frame.shape[:2]}")
        view = name.removeprefix("CAM_").lower()
        cameras[view] = {
            "image": _jpeg20_rgb(frame),
            "intrinsic": contract["CAM2IMG"][name][:3, :3].copy(),
            "sensor2ego": axes @ np.linalg.inv(contract["LIDAR2CAM"][name]),
        }
    return cameras


def _gps_to_location(gps, lat_ref, lon_ref):
    earth_radius = 6378137.0
    lat, lon = np.asarray(gps, dtype=np.float64).reshape(-1)[:2]
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (earth_radius * scale)
    mx = lon * (math.pi * earth_radius * scale) / 180.0
    y = scale * earth_radius * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my
    x = mx - scale * lon_ref * math.pi * earth_radius / 180.0
    return np.array([x, y], dtype=np.float64)


def _ego_pose(pos, compass):
    x, y = float(pos[0]), -float(pos[1])
    theta = -float(compass) + math.pi / 2.0
    c, s = math.cos(theta), math.sin(theta)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    pose[:2, 3] = (x, y)
    return pose


def _canonical_pose(ego2global, lidar2ego):
    axes = np.eye(4, dtype=np.float64)
    axes[:3, :3] = lidar2ego[:3, :3]
    return ego2global @ lidar2ego @ np.linalg.inv(axes)


def _route_command(command, profile_commands):
    value = int(getattr(command, "value", command))
    if value < 0:
        value = 4
    index = value - 1  # preserve the main agent's raw 0 -> one-hot index -1 wraparound
    try:
        return profile_commands[index]
    except IndexError as exc:
        raise ValueError(f"unsupported B2D route command {value}") from exc


def _pid_coordinates(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2 or not np.isfinite(trajectory[:, :2]).all():
        raise ValueError("policy trajectory must be a finite N x 2 array")
    # UniDriveVLA: x forward / y left. VAD PID: x right / y forward.
    return np.column_stack((-trajectory[:, 1], trajectory[:, 0]))


class UniDriveVLAB2DClosedLoopAgent(autonomous_agent.AutonomousAgent):
    """Leaderboard agent; calibration and policy inputs match the existing main B2D agent."""

    def sensors(self):
        root = getattr(self, "leaderboard_root", None)
        return load_main_contract(root)["sensors"]

    def setup(self, path_to_conf_file):
        checkpoint_frame, with_motion = _closed_loop_options()
        if not LEADERBOARD_AVAILABLE:
            raise RuntimeError("CARLA Leaderboard is not importable; set PYTHONPATH via the official B2D launcher")
        self.contract = load_main_contract()
        self.leaderboard_root = self.contract["leaderboard_root"]
        self.route_planner_class, self.pid_class = _load_devkit(self.leaderboard_root)
        self.track = autonomous_agent.Track.SENSORS
        checkpoint = _checkpoint_path(path_to_conf_file)
        self.device = os.environ.get("UNIDRIVE_DEVICE", "cuda")
        self.runtime = load_runtime("b2d", checkpoint, self.device, with_motion=with_motion,
                                    checkpoint_frame=checkpoint_frame)
        self.runtime.enable_temporal_ring(RING_SIZE)
        self.pid_metadata = None
        self._route_planner = None
        self.step = -1

    def init(self):
        if not getattr(self, "_global_plan", None) or not getattr(self, "_global_plan_world_coord", None):
            raise RuntimeError("Leaderboard did not provide a B2D global plan")
        self.runtime.reset()
        self.step = -1
        self.pid_metadata = None
        self.pidcontroller = self.pid_class(**PID_ARGUMENTS)
        self.lat_ref, self.lon_ref = 42.0, 2.0
        try:
            from scipy.optimize import fsolve

            loc = self._global_plan_world_coord[0][0].location
            lon = self._global_plan[0][0]["lon"]
            lat = self._global_plan[0][0]["lat"]
            earth_radius = 6378137.0

            def equations(values):
                x, y = values
                eq1 = (lon * math.cos(x * math.pi / 180.0) - (loc.x * x * 180.0) /
                       (math.pi * earth_radius) - math.cos(x * math.pi / 180.0) * y)
                eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius *
                       math.cos(x * math.pi / 180.0) + loc.y - math.cos(x * math.pi / 180.0) *
                       earth_radius * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
                return [eq1, eq2]

            self.lat_ref, self.lon_ref = fsolve(equations, [0.0, 0.0])
        except Exception:
            self.lat_ref, self.lon_ref = 0.0, 0.0
        self._route_planner = self.route_planner_class(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)

    def run_step(self, input_data, timestamp):
        if self._route_planner is None:
            self.init()
        self.step += 1
        cameras = _camera_features(input_data, self.contract)
        gps = np.asarray(_payload(input_data, "GPS"), dtype=np.float64).reshape(-1)
        if gps.size < 2 or not np.isfinite(gps[:2]).all():
            raise ValueError("GPS must contain finite latitude and longitude")
        speed_data = _payload(input_data, "SPEED")
        try:
            speed = np.float32(speed_data["speed"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError("SPEED payload must contain a numeric 'speed'") from exc
        if not np.isfinite(speed):
            raise ValueError("SPEED payload is not finite")
        imu = np.asarray(_payload(input_data, "IMU"), dtype=np.float64).reshape(-1)
        if imu.size < 7:
            raise ValueError("IMU payload must contain acceleration, angular velocity, and compass")
        acceleration, angular_velocity = imu[:3].copy(), imu[3:6].copy()
        compass = float(imu[-1])
        if math.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)

        pos = _gps_to_location(gps, self.lat_ref, self.lon_ref)
        _, route_command = self._route_planner.run_step(pos)
        command = _route_command(route_command, self.runtime.profile.commands)
        ego2global = _canonical_pose(_ego_pose(pos, compass), self.contract["LIDAR2EGO"])
        native_to_ego = np.eye(4, dtype=np.float64)
        native_to_ego[:3, :3] = self.contract["LIDAR2EGO"][:3, :3]
        ego_steer = self.pid_metadata["steer"] if self.pid_metadata is not None else 0.0
        plan = self.runtime.predict(
            cameras=cameras,
            ego2global=ego2global,
            timestamp=self.step / FRAME_RATE,
            command=command,
            ego_velocity=np.array([speed, 0.0, 0.0], dtype=np.float32),
            ego_acceleration=np.array([acceleration[0], -acceleration[1], acceleration[2]], dtype=np.float32),
            ego_angular_velocity=(-angular_velocity).astype(np.float32),
            ego_steer=ego_steer,
            native_to_ego=native_to_ego,
        )
        waypoints = _pid_coordinates(plan)
        if len(waypoints) != self.runtime.profile.horizon or len(waypoints) < 2:
            raise RuntimeError(f"B2D policy must return its configured {self.runtime.profile.horizon}-step horizon")
        target = waypoints[-1]  # PID use_target_to_aim=False in the main configuration.
        steer, throttle, brake, metadata = self.pidcontroller.control_pid(waypoints, speed, target)
        brake = float(brake)
        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0
        import carla

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = float(np.clip(throttle, 0.0, 0.75))
        control.brake = float(np.clip(brake, 0.0, 1.0))
        self.pid_metadata = dict(metadata, steer=control.steer)
        return control

    def destroy(self):
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.reset()


def get_entry_point():
    return "UniDriveVLAB2DClosedLoopAgent"
