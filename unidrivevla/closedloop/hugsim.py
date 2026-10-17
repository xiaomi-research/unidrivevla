"""HUGSIM FIFO observation adapter using the official camera and vehicle calibration fields.

Protocol evidence: hyzhou404/HUGSIM `closed_loop.py` exchanges pickle((obs, info)) and plans over two FIFOs;
`sim/hugsim_env/envs/hug_sim.py` publishes RGB plus ego_box/cam_params; `sim/utils/sim_utils.py` defines the
LiDAR-frame x-right/y-forward plan consumed by `traj2control`.
"""

import math

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from unidrivevla.closedloop.runtime import _matrix, _rigid
from unidrivevla.closedloop.neuro_ncap import CAMERA_ORDER, CAMERA_VIEWS

# HUGSIM's world/body coordinates are x-right, y-up, z-forward. This proper rotation maps them to the
# training convention x-forward, y-left, z-up, matching HUGSIM's own ego_box=[z, -x, -y, ..., -yaw].
HUGSIM_TO_EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def load_calibration(camera_config, kinematic_config):
    camera_cfg = OmegaConf.to_container(OmegaConf.load(camera_config), resolve=True)
    kinematic = OmegaConf.to_container(OmegaConf.load(kinematic_config), resolve=True)
    if set(camera_cfg.get("cams", {})) != set(CAMERA_ORDER):
        raise ValueError("HUGSIM camera config must describe all six nuScenes cameras")
    if "cam_rect" not in camera_cfg or not {"rot", "trans"} <= set(camera_cfg["cam_rect"]):
        raise ValueError("HUGSIM camera config must include its actual cam_rect.rot and cam_rect.trans")
    if not isinstance(kinematic, dict) or not all(key in kinematic for key in ("Lr", "Lf")):
        raise ValueError("HUGSIM kinematic config must include Lr and Lf")
    wheelbase = float(kinematic["Lr"]) + float(kinematic["Lf"])
    if not math.isfinite(wheelbase) or wheelbase <= 0:
        raise ValueError("HUGSIM wheelbase Lr + Lf must be finite and positive")
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = Rotation.from_euler("XYZ", camera_cfg["cam_rect"]["rot"], degrees=True).as_matrix()
    rect[:3, 3] = _vector(camera_cfg["cam_rect"]["trans"], "cam_rect.trans")
    expected = {}
    for name, values in camera_cfg["cams"].items():
        extrinsics, intrinsic = values["extrinsics"], values["intrinsics"]
        expected[name] = {
            "v2c": _from_euler(extrinsics["v2c_rot"], extrinsics["v2c_trans"], name + ".v2c"),
            "l2c": _from_euler(extrinsics["l2c_rot"], extrinsics["l2c_trans"], name + ".l2c"),
            "intrinsic": {"H": int(intrinsic["H"]), "W": int(intrinsic["W"]),
                          "cx": float(intrinsic["cx"]), "cy": float(intrinsic["cy"]),
                          "fovx": math.radians(float(intrinsic["fovx"])),
                          "fovy": math.radians(float(intrinsic["fovy"]))},
        }
    return rect, wheelbase, expected


class HUGSIMInputAdapter:
    def __init__(self, cam_rect, wheelbase, expected_cameras):
        self.cam_rect = _rigid(cam_rect, "cam_rect")
        self.wheelbase = float(wheelbase)
        self.expected_cameras = expected_cameras
        if self.wheelbase <= 0 or not math.isfinite(self.wheelbase):
            raise ValueError("wheelbase must be finite and positive")

    def convert(self, observation, info):
        if not isinstance(observation, dict) or not isinstance(info, dict):
            raise ValueError("HUGSIM FIFO item must be (observation, info) dictionaries")
        required = ("ego_pos", "ego_rot", "ego_velo", "ego_steer", "accelerate", "timestamp", "command",
                    "ego_box", "cam_params")
        missing = [key for key in required if key not in info]
        if missing:
            raise ValueError(f"HUGSIM info is missing required fields: {missing}")
        rgb = observation.get("rgb")
        params = info["cam_params"]
        if not isinstance(rgb, dict) or not isinstance(params, dict):
            raise ValueError("HUGSIM requires observation.rgb and info.cam_params mappings")
        if set(rgb) != set(CAMERA_ORDER) or set(params) != set(CAMERA_ORDER):
            raise ValueError("HUGSIM nuScenes split requires all six calibrated cameras")

        timestamp = float(info["timestamp"])
        speed, steer, acceleration = (float(info[key]) for key in ("ego_velo", "ego_steer", "accelerate"))
        if not all(math.isfinite(value) for value in (timestamp, speed, steer, acceleration)) or timestamp < 0:
            raise ValueError("HUGSIM timestamp and ego state must be finite")
        if steer < -math.pi / 2 or steer > math.pi / 2:
            raise ValueError("HUGSIM ego_steer must be radians in (-pi/2, pi/2)")
        command = _command(info["command"])

        body_pose_h = np.eye(4, dtype=np.float64)
        body_pose_h[:3, :3] = Rotation.from_euler("XYZ", _vector(info["ego_rot"], "ego_rot")).as_matrix()
        body_pose_h[:3, 3] = _vector(info["ego_pos"], "ego_pos")
        basis = np.eye(4, dtype=np.float64)
        basis[:3, :3] = HUGSIM_TO_EGO
        body_pose_ego = basis @ body_pose_h @ basis.T
        _check_ego_box(info["ego_box"], body_pose_ego)

        front_v2c = _rigid(params["CAM_FRONT"].get("v2c"), "CAM_FRONT.v2c")
        cameras = {}
        lidar_to_body_h = None
        for name in CAMERA_ORDER:
            camera_params = params[name]
            if not isinstance(camera_params, dict) or not {"intrinsic", "v2c", "l2c"} <= camera_params.keys():
                raise ValueError(f"HUGSIM info.cam_params[{name}] needs intrinsic, v2c, and l2c")
            v2c = _rigid(camera_params["v2c"], f"{name}.v2c")
            l2c = _rigid(camera_params["l2c"], f"{name}.l2c")
            _check_camera_calibration(name, camera_params, v2c, l2c, self.expected_cameras[name])
            camera_to_body_h = front_v2c @ np.linalg.inv(v2c) @ self.cam_rect
            lidar_to_body = camera_to_body_h @ l2c
            if lidar_to_body_h is None:
                lidar_to_body_h = lidar_to_body
            elif not np.allclose(lidar_to_body, lidar_to_body_h, atol=0.12, rtol=0):
                raise ValueError("HUGSIM cameras disagree on LiDAR-to-body calibration; check matching camera config")

            intrinsic = camera_params["intrinsic"]
            height, width = int(intrinsic["H"]), int(intrinsic["W"])
            image = np.asarray(rgb[name])
            if image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise ValueError(f"HUGSIM {name} RGB must be uint8 {(height, width, 3)}, got {image.shape}/{image.dtype}")
            fovx, fovy = float(intrinsic["fovx"]), float(intrinsic["fovy"])
            cx, cy = float(intrinsic["cx"]), float(intrinsic["cy"])
            if not (0 < fovx < math.pi and 0 < fovy < math.pi and math.isfinite(cx) and math.isfinite(cy)):
                raise ValueError(f"HUGSIM {name} intrinsics have invalid FOV or principal point")
            matrix = np.array([[width / (2 * math.tan(fovx / 2)), 0, cx],
                               [0, height / (2 * math.tan(fovy / 2)), cy], [0, 0, 1]], dtype=np.float64)
            cameras[CAMERA_VIEWS[name]] = {
                "image": image,
                "intrinsic": matrix,
                "camera_to_body_h": camera_to_body_h,
                "lidar_to_body_h": lidar_to_body,
            }

        lidar_to_body_ego = basis @ lidar_to_body_h
        axes = np.eye(4, dtype=np.float64)
        axes[:3, :3] = lidar_to_body_ego[:3, :3]
        pose = body_pose_ego @ lidar_to_body_ego @ np.linalg.inv(axes)
        for item in cameras.values():
            camera_to_body_ego = basis @ item.pop("camera_to_body_h")
            item.pop("lidar_to_body_h")
            item["sensor2ego"] = axes @ np.linalg.inv(lidar_to_body_ego) @ camera_to_body_ego

        # HUGSIM reports speed, applied acceleration, and steering. Its own bicycle update gives
        # yaw_rate=-v*tan(steer)/(Lr+Lf) after the x-forward/y-left axis conversion.
        return {
            "cameras": cameras,
            "ego2global": pose,
            "native_to_ego": axes,
            "timestamp": timestamp,
            "command": command,
            "ego_velocity": np.array([speed, 0.0, 0.0], dtype=np.float32),
            "ego_acceleration": np.array([acceleration, 0.0, 0.0], dtype=np.float32),
            "ego_angular_velocity": np.array([0.0, 0.0, -speed * math.tan(steer) / self.wheelbase], dtype=np.float32),
            "ego_steer": steer,
        }


def _command(value):
    if isinstance(value, bool):
        raise ValueError("HUGSIM command must be right/left/straight or its official integer id")
    if isinstance(value, (int, np.integer)):
        if int(value) not in (0, 1, 2):
            raise ValueError(f"unsupported HUGSIM command id {value!r}")
        return ("right", "left", "straight")[int(value)]
    if isinstance(value, str) and value in ("right", "left", "straight"):
        return value
    raise ValueError(f"unsupported HUGSIM command {value!r}")


def trajectory_to_hugsim(trajectory):
    """Convert x-forward/y-left points to HUGSIM's LiDAR x-right/y-forward ordering."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) == 0 or not np.isfinite(trajectory).all():
        raise ValueError("HUGSIM plan must be finite nonempty (T, 2) points")
    return np.column_stack((-trajectory[:, 1], trajectory[:, 0]))


def hugsim_to_canonical_plan(plan):
    """Mirror HUGSIM closed_loop.py's plan[:, [1, 0]] then y *= -1 transform."""
    plan = np.asarray(plan, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] != 2:
        raise ValueError("HUGSIM plan must have shape (T, 2)")
    return np.column_stack((plan[:, 1], -plan[:, 0]))


def _check_ego_box(value, body_pose_ego):
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all():
        raise ValueError("HUGSIM ego_box must contain seven finite values")
    if not np.allclose(box[:3], body_pose_ego[:3, 3], atol=1e-4, rtol=0):
        raise ValueError("HUGSIM ego_pos/ego_rot and ego_box use inconsistent coordinate frames")


def _vector(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3-vector") from exc
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


def _from_euler(rotation, translation, name):
    return _rigid(np.block([[Rotation.from_euler("XYZ", rotation, degrees=True).as_matrix(),
                             _vector(translation, name + ".translation").reshape(3, 1)],
                            [np.zeros((1, 3)), np.ones((1, 1))]]), name)


def _check_camera_calibration(name, actual, v2c, l2c, expected):
    if not np.allclose(v2c, expected["v2c"], atol=1e-5, rtol=0) or not np.allclose(l2c, expected["l2c"], atol=1e-5, rtol=0):
        raise ValueError(f"HUGSIM info.cam_params[{name}] does not match the supplied simulator camera config")
    for key, tolerance in (("cx", 1e-5), ("cy", 1e-5), ("fovx", 1e-6), ("fovy", 1e-6)):
        if not math.isclose(float(actual["intrinsic"][key]), expected["intrinsic"][key], rel_tol=0, abs_tol=tolerance):
            raise ValueError(f"HUGSIM {name} {key} does not match the supplied simulator camera config")
    if any(int(actual["intrinsic"][key]) != expected["intrinsic"][key] for key in ("H", "W")):
        raise ValueError(f"HUGSIM {name} image size does not match the supplied simulator camera config")
