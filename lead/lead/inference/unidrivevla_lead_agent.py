import importlib
import json
import math
import os
import pathlib
import shutil
import sys
import time

import carla
import cv2
import numpy as np
import torch
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmcv.runner import wrap_fp16_model
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from pyquaternion import Quaternion

from leaderboard.autoagents import autonomous_agent


LEAD_ROOT = os.environ.get(
    "LEAD_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)
PROJECT_DIR = os.environ.get(
    "PROJECT_DIR",
    os.path.join(LEAD_ROOT, "Bench2Drive"),
)
TEAM_CODE_DIR = os.environ.get(
    "TEAM_CODE_DIR", os.path.join(PROJECT_DIR, "bench2drive", "leaderboard")
)
F2D_MAP_ROOT = os.environ.get(
    "LEAD_F2D_MAP_ROOT",
    os.path.join(
        LEAD_ROOT,
        "3rd_party",
        "fail2drive",
        "team_code",
        "birds_eye_view",
        "maps_4ppm_cv",
    ),
)
F2D_CARLA_ROOT = os.environ.get("CARLA_ROOT", "")

for path in [PROJECT_DIR, TEAM_CODE_DIR, LEAD_ROOT]:
    if path and path not in sys.path:
        sys.path.insert(0, path)

from team_code.pid_controller import PIDController as PIDControllerVAD  # noqa: E402
from team_code.planner import RoutePlanner as RoutePlannerVAD  # noqa: E402
from lead.inference.hipad_controller import (  # noqa: E402
    HiPADDualPlanningController,
)


SAVE_PATH = os.environ.get("SAVE_PATH", None)
CAMERA_IDS = ["rgb_1", "rgb_2", "rgb_3"]
LEAD_COMMAND_MAP = {
    0: "TURN LEFT",
    1: "TURN RIGHT",
    2: "GO STRAIGHT",
    3: "LANE FOLLOW",
    4: "CHANGE LANE LEFT",
    5: "CHANGE LANE RIGHT",
}
B2D_ACTION_DELTA_MU = [-0.0222, 2.0249]
B2D_ACTION_DELTA_STD = [0.6720, 1.8586]
LEAD_CONTROL_MODES = (
    "lead_waypoint",
    "b2d_pid",
    "tfv6_route_speed",
    "hybrid_route_speed",
    "hipad_dual",
)
LEAD_ROUTE_PLANNER_MODES = ("lead", "b2d")
LEAD_TEMPORAL_MODES = ("close_loop", "reset_each_step")
LEAD_METRIC_FLUSH_INTERVAL = int(os.environ.get("UNIDRIVEVLA_LEAD_METRIC_FLUSH_INTERVAL", "200"))
LEAD_SPEED_SWEEP = os.environ.get(
    "UNIDRIVEVLA_LEAD_SPEED_SWEEP", "0"
).lower() in {"1", "true", "yes"}
LEAD_SPEED_SWEEP_SEED = int(
    os.environ.get("UNIDRIVEVLA_LEAD_SPEED_SWEEP_SEED", "20260730")
)
LEAD_REMOVE_PROMPT_SPEED = os.environ.get(
    "UNIDRIVEVLA_LEAD_REMOVE_PROMPT_SPEED", "0"
).lower() in {"1", "true", "yes"}
LEAD_USE_KALMAN = os.environ.get(
    "UNIDRIVEVLA_LEAD_USE_KALMAN", "1"
).lower() in {"1", "true", "yes"}
LEAD_CREEPING = os.environ.get(
    "UNIDRIVEVLA_LEAD_CREEPING", "1"
).lower() in {"1", "true", "yes"}
LEAD_CREEP_POINT_SOURCE = os.environ.get(
    "UNIDRIVEVLA_LEAD_CREEP_POINT_SOURCE", "single_lidar"
).lower()
if LEAD_CREEP_POINT_SOURCE not in {"single_lidar", "official_tfv6"}:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_CREEP_POINT_SOURCE must be 'single_lidar' or "
        f"'official_tfv6', got {LEAD_CREEP_POINT_SOURCE!r}"
    )
if LEAD_CREEP_POINT_SOURCE == "official_tfv6":
    from lead.common import ransac as tfv6_ransac
    from lead.expert.config_expert import ExpertConfig
else:
    tfv6_ransac = None
    ExpertConfig = None

LEAD_B2D_NEAR_SPEED = os.environ.get(
    "UNIDRIVEVLA_LEAD_B2D_NEAR_SPEED", "0"
).lower() in {"1", "true", "yes"}
LEAD_STOP_SIGN = os.environ.get(
    "UNIDRIVEVLA_LEAD_STOP_SIGN", "0"
).lower() in {"1", "true", "yes"}
LEAD_RECORD_COMFORT = os.environ.get(
    "UNIDRIVEVLA_LEAD_RECORD_COMFORT", "1"
).lower() in {"1", "true", "yes"}
LEAD_RECORD_VISUALIZATION = os.environ.get(
    "UNIDRIVEVLA_LEAD_RECORD_VISUALIZATION", "0"
).lower() in {"1", "true", "yes"}
LEAD_RECORD_CAMERA = os.environ.get("UNIDRIVEVLA_LEAD_RECORD_CAMERA", "0").lower() in {"1", "true", "yes"}
LEAD_CAMERA_FRAME_FREQUENCY = int(os.environ.get("UNIDRIVEVLA_LEAD_CAMERA_FRAME_FREQUENCY", "5"))
if LEAD_CAMERA_FRAME_FREQUENCY <= 0:
    raise ValueError("UNIDRIVEVLA_LEAD_CAMERA_FRAME_FREQUENCY must be positive")
LEAD_RECORD_LIDAR = os.environ.get(
    "UNIDRIVEVLA_LEAD_RECORD_LIDAR", "0"
).lower() in {"1", "true", "yes"}
LEAD_LIDAR_FRAME_FREQUENCY = int(
    os.environ.get("UNIDRIVEVLA_LEAD_LIDAR_FRAME_FREQUENCY", "0")
)
if LEAD_LIDAR_FRAME_FREQUENCY < 0:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_LIDAR_FRAME_FREQUENCY must be non-negative"
    )
LEAD_LIDAR_ROUTE_STEPS_RAW = os.environ.get(
    "UNIDRIVEVLA_LEAD_LIDAR_ROUTE_STEPS", "{}"
)
try:
    _lidar_route_steps = json.loads(LEAD_LIDAR_ROUTE_STEPS_RAW)
except json.JSONDecodeError as exc:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_LIDAR_ROUTE_STEPS must be a JSON object"
    ) from exc
if not isinstance(_lidar_route_steps, dict):
    raise ValueError("UNIDRIVEVLA_LEAD_LIDAR_ROUTE_STEPS must be a JSON object")
LEAD_LIDAR_ROUTE_STEPS = {}
for _route_id, _steps in _lidar_route_steps.items():
    if not isinstance(_route_id, str) or not _route_id:
        raise ValueError("LiDAR route-step keys must be non-empty route IDs")
    if not isinstance(_steps, list) or not _steps:
        raise ValueError(f"LiDAR steps for {_route_id} must be a non-empty list")
    try:
        _normalized_steps = frozenset(int(step) for step in _steps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LiDAR steps for {_route_id} must be integers") from exc
    if any(step < 0 for step in _normalized_steps):
        raise ValueError(f"LiDAR steps for {_route_id} must be non-negative")
    LEAD_LIDAR_ROUTE_STEPS[_route_id] = _normalized_steps
if (
    LEAD_RECORD_LIDAR
    and not LEAD_LIDAR_ROUTE_STEPS
    and LEAD_LIDAR_FRAME_FREQUENCY == 0
):
    raise ValueError(
        "UNIDRIVEVLA_LEAD_RECORD_LIDAR requires "
        "UNIDRIVEVLA_LEAD_LIDAR_ROUTE_STEPS or a positive "
        "UNIDRIVEVLA_LEAD_LIDAR_FRAME_FREQUENCY"
    )
LEAD_RECORD_BEV_CAMERA = os.environ.get(
    "UNIDRIVEVLA_LEAD_RECORD_BEV_CAMERA", "0"
).lower() in {"1", "true", "yes"}
LEAD_BEV_FRAME_FREQUENCY = int(
    os.environ.get("UNIDRIVEVLA_LEAD_BEV_FRAME_FREQUENCY", "0")
)
if LEAD_BEV_FRAME_FREQUENCY < 0:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_BEV_FRAME_FREQUENCY must be non-negative"
    )
LEAD_BEV_ROUTE_STEPS_RAW = os.environ.get(
    "UNIDRIVEVLA_LEAD_BEV_ROUTE_STEPS", "{}"
)
try:
    _bev_route_steps = json.loads(LEAD_BEV_ROUTE_STEPS_RAW)
except json.JSONDecodeError as exc:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_BEV_ROUTE_STEPS must be a JSON object"
    ) from exc
if not isinstance(_bev_route_steps, dict):
    raise ValueError("UNIDRIVEVLA_LEAD_BEV_ROUTE_STEPS must be a JSON object")
LEAD_BEV_ROUTE_STEPS = {}
for _route_id, _steps in _bev_route_steps.items():
    if not isinstance(_route_id, str) or not _route_id:
        raise ValueError("BEV route-step keys must be non-empty route IDs")
    if not isinstance(_steps, list) or not _steps:
        raise ValueError(f"BEV steps for {_route_id} must be a non-empty list")
    try:
        _normalized_steps = frozenset(int(step) for step in _steps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BEV steps for {_route_id} must be integers") from exc
    if any(step < 0 for step in _normalized_steps):
        raise ValueError(f"BEV steps for {_route_id} must be non-negative")
    LEAD_BEV_ROUTE_STEPS[_route_id] = _normalized_steps
if (
    LEAD_RECORD_BEV_CAMERA
    and not LEAD_BEV_ROUTE_STEPS
    and LEAD_BEV_FRAME_FREQUENCY == 0
):
    raise ValueError(
        "UNIDRIVEVLA_LEAD_RECORD_BEV_CAMERA requires "
        "UNIDRIVEVLA_LEAD_BEV_ROUTE_STEPS or a positive "
        "UNIDRIVEVLA_LEAD_BEV_FRAME_FREQUENCY"
    )
LEAD_VIS_DET_SCORE_THRESHOLD = float(
    os.environ.get("UNIDRIVEVLA_LEAD_VIS_DET_SCORE_THRESHOLD", "0.3")
)
LEAD_VIS_MAP_SCORE_THRESHOLD = float(
    os.environ.get("UNIDRIVEVLA_LEAD_VIS_MAP_SCORE_THRESHOLD", "0.3")
)
LEAD_VIS_ACTOR_RADIUS_M = float(
    os.environ.get("UNIDRIVEVLA_LEAD_VIS_ACTOR_RADIUS_M", "65.0")
)
LEAD_VIS_MAX_PRED_BOXES = int(
    os.environ.get("UNIDRIVEVLA_LEAD_VIS_MAX_PRED_BOXES", "100")
)
LEAD_VIS_MAX_MAP_VECTORS = int(
    os.environ.get("UNIDRIVEVLA_LEAD_VIS_MAX_MAP_VECTORS", "100")
)
LEAD_DET_CLASSES = (
    "car", "van", "truck", "bicycle", "traffic_sign",
    "traffic_cone", "traffic_light", "pedestrian", "others",
)
LEAD_MAP_CLASSES = ("Broken", "Solid", "SolidSolid", "Center")
if not 0.0 <= LEAD_VIS_DET_SCORE_THRESHOLD <= 1.0:
    raise ValueError("UNIDRIVEVLA_LEAD_VIS_DET_SCORE_THRESHOLD must be in [0, 1]")
if not 0.0 <= LEAD_VIS_MAP_SCORE_THRESHOLD <= 1.0:
    raise ValueError("UNIDRIVEVLA_LEAD_VIS_MAP_SCORE_THRESHOLD must be in [0, 1]")
if LEAD_VIS_ACTOR_RADIUS_M <= 0.0:
    raise ValueError("UNIDRIVEVLA_LEAD_VIS_ACTOR_RADIUS_M must be positive")
if LEAD_STOP_SIGN:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_STOP_SIGN=1 is unsupported: this checkpoint merges "
        "stop_sign into traffic_sign and cannot apply the official heuristic safely"
    )

LEAD_COMMAND_SOURCE = os.environ.get("UNIDRIVEVLA_LEAD_COMMAND_SOURCE", "current").lower()
if LEAD_COMMAND_SOURCE not in {"current", "next"}:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_COMMAND_SOURCE must be 'current' or 'next', "
        f"got {LEAD_COMMAND_SOURCE!r}"
    )
LEAD_ROUTE_PLANNER_MODE = os.environ.get("UNIDRIVEVLA_LEAD_ROUTE_PLANNER", "lead").lower()
if LEAD_ROUTE_PLANNER_MODE not in LEAD_ROUTE_PLANNER_MODES:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_ROUTE_PLANNER must be one of "
        f"{LEAD_ROUTE_PLANNER_MODES}, got {LEAD_ROUTE_PLANNER_MODE!r}"
    )
LEAD_TEMPORAL_MODE = os.environ.get("UNIDRIVEVLA_LEAD_TEMPORAL_MODE", "close_loop").lower()
if LEAD_TEMPORAL_MODE in {"reset", "none", "no_cache", "disable", "disabled"}:
    LEAD_TEMPORAL_MODE = "reset_each_step"
if LEAD_TEMPORAL_MODE not in LEAD_TEMPORAL_MODES:
    raise ValueError(
        "UNIDRIVEVLA_LEAD_TEMPORAL_MODE must be one of "
        f"{LEAD_TEMPORAL_MODES}, got {LEAD_TEMPORAL_MODE!r}"
    )

LEAD_WP_DELTA_CLIP = 0.99
LEAD_WP_AIM_DISTANCE_FAST = 3.0
LEAD_WP_AIM_DISTANCE_SLOW = 2.25
LEAD_WP_AIM_DISTANCE_THRESHOLD = 5.5
LEAD_WP_BRAKE_SPEED = 0.4
LEAD_WP_BRAKE_RATIO = 1.1
TFV6_ROUTE_BRAKE_RATIO = 1.1
TFV6_STOP_PROBABILITY_THRESHOLD = 0.9
TFV6_LATERAL_KP = 3.118357247806046
TFV6_LATERAL_KD = 1.3782508892109167
TFV6_LATERAL_KI = 0.6406067986034124
TFV6_LATERAL_SPEED_SCALE = 0.9755321901954155
TFV6_LATERAL_SPEED_OFFSET = 1.9152884533402488
TFV6_LATERAL_WINDOW = 6
TFV6_STEER_CORRECTION = 3.5
TFV6_LONGITUDINAL_MAX_ACCELERATION = 1.89
TFV6_LONGITUDINAL_PARAMS = np.asarray(
    [
        1.1990342347353184, -0.8057602384167799, 1.710818710950062,
        0.921890257450335, 1.556497522998393, -0.7013479734904027,
        1.031266635497984,
    ],
    dtype=np.float64,
)
HYBRID_STOP_RELEASE_PROBABILITY = 0.5
HYBRID_TARGET_SPEED_RISE_PER_STEP = 0.1
HYBRID_MAX_TARGET_SPEED = 10.0
HYBRID_TRAJECTORY_SPEED_MARGIN = 0.5
HYBRID_MIN_CREEP_SPEED = 1.5
HYBRID_CURVATURE_SPEED_FACTOR = 1.5
HYBRID_MIN_CURVATURE_SPEED = 2.0
HYBRID_STEER_DISAGREEMENT_THRESHOLD = 0.6
HYBRID_DISAGREEMENT_SPEED_CAP = 2.5
HYBRID_MAX_STEER_DELTA = 0.08
HYBRID_MAX_THROTTLE = 0.6
HYBRID_MAX_THROTTLE_RISE_PER_STEP = 0.05
HYBRID_SPEED_DEADBAND = 0.2
HYBRID_FULL_BRAKE_SPEED_ERROR = 2.0
_LEAD_JPEG_QUALITY_RAW = os.environ.get("UNIDRIVEVLA_LEAD_JPEG_QUALITY", "auto")
if _LEAD_JPEG_QUALITY_RAW.lower() == "auto":
    LEAD_JPEG_QUALITY = None
else:
    LEAD_JPEG_QUALITY = int(_LEAD_JPEG_QUALITY_RAW)
    if not 0 <= LEAD_JPEG_QUALITY <= 100:
        raise ValueError(
            "UNIDRIVEVLA_LEAD_JPEG_QUALITY must be in [0, 100] or auto, "
            f"got {LEAD_JPEG_QUALITY!r}"
        )
_LEAD_WAYPOINT_HZ_RAW = os.environ.get("UNIDRIVEVLA_LEAD_WAYPOINT_HZ", "auto")
if _LEAD_WAYPOINT_HZ_RAW.lower() == "auto":
    LEAD_WAYPOINT_HZ = None
else:
    LEAD_WAYPOINT_HZ = float(_LEAD_WAYPOINT_HZ_RAW)
    if LEAD_WAYPOINT_HZ <= 0:
        raise ValueError(
            "UNIDRIVEVLA_LEAD_WAYPOINT_HZ must be positive or 'auto', "
            f"got {LEAD_WAYPOINT_HZ!r}"
        )
LEAD_ROUTE_POP_DISTANCE = float(os.environ.get("UNIDRIVEVLA_LEAD_ROUTE_POP_DISTANCE", "5.0"))
LEAD_ROUTE_ADAPTIVE_POP_DISTANCE = float(os.environ.get("UNIDRIVEVLA_LEAD_ROUTE_ADAPTIVE_POP_DISTANCE", "4.0"))
LEAD_ROUTE_ADAPTIVE_POP = os.environ.get("UNIDRIVEVLA_LEAD_ROUTE_ADAPTIVE_POP", "1") not in {"0", "false", "False"}
LEAD_SKIP_DISTANT_TARGET_POINT = os.environ.get(
    "UNIDRIVEVLA_LEAD_SKIP_DISTANT_TARGET_POINT", "1"
) not in {"0", "false", "False"}
LEAD_DISTANT_TARGET_POINT_THRESHOLD = float(
    os.environ.get("UNIDRIVEVLA_LEAD_DISTANT_TARGET_POINT_THRESHOLD", "50.0")
)
LEAD_CREEP_STUCK_THRESHOLD = int(
    os.environ.get("UNIDRIVEVLA_LEAD_CREEP_STUCK_THRESHOLD", "1100")
)
LEAD_CREEP_MOVE_DURATION = int(
    os.environ.get("UNIDRIVEVLA_LEAD_CREEP_MOVE_DURATION", "20")
)
LEAD_CREEP_THROTTLE = float(
    os.environ.get("UNIDRIVEVLA_LEAD_CREEP_THROTTLE", "0.4")
)
if LEAD_CREEP_STUCK_THRESHOLD < 0 or LEAD_CREEP_MOVE_DURATION <= 0:
    raise ValueError("Invalid creeping threshold or move duration")
if not 0.0 <= LEAD_CREEP_THROTTLE <= 1.0:
    raise ValueError("UNIDRIVEVLA_LEAD_CREEP_THROTTLE must be in [0, 1]")

LEAD_LIDAR_POSITION = np.asarray([0.0, 0.0, 2.5], dtype=np.float64)
LEAD_LIDAR_ROTATION_DEG = np.asarray([0.0, 0.0, -90.0], dtype=np.float64)
LEAD_CREEP_SAFETY_Z = (0.5, 1.5)
LEAD_CREEP_SAFETY_Y = (-1.0641621351242065 * 0.8, 1.0641621351242065 * 0.8)
LEAD_CREEP_SAFETY_X = (2.4508416652679443, 2.4508416652679443 + 2.5)

LEFT2RIGHT = np.eye(4, dtype=np.float32)
LEFT2RIGHT[1, 1] = -1.0
LEFTHAND_EGO_TO_LIDAR = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def _right_hand_lidar2ego():
    """Map Sparse4D (lateral-right, forward) into the right-handed ego frame."""
    return (LEFT2RIGHT @ np.linalg.inv(LEFTHAND_EGO_TO_LIDAR)).astype(np.float32)

LEAD_CAMERA_CALIBRATION = {
    1: {
        "pos": [0.1, -0.35, 2.25],
        "rot": [0.0, 0.0, -54.5],
        "width": 384,
        "height": 384,
        "cropped_height": 384,
        "fov": 60,
    },
    2: {
        "pos": [0.35, 0.0, 2.25],
        "rot": [0.0, 0.0, 0.0],
        "width": 384,
        "height": 384,
        "cropped_height": 384,
        "fov": 60,
    },
    3: {
        "pos": [0.1, 0.35, 2.25],
        "rot": [0.0, 0.0, 54.5],
        "width": 384,
        "height": 384,
        "cropped_height": 384,
        "fov": 60,
    },
}


def get_entry_point():
    return "UniDriveVLALEADAgent"


def _install_qwen3_vl_replacement():
    import transformers

    dst = os.path.join(os.path.dirname(transformers.__file__), "models", "qwen3_vl")
    marker = "image_embeds, deepstack_image_embeds, _ = self.get_image_features"
    model_file = os.path.join(dst, "modeling_qwen3_vl.py")
    if os.path.exists(model_file):
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                if marker in f.read():
                    return
        except OSError:
            pass

    candidates = [
        os.environ.get("QWEN3_VL_SOURCE", ""),
        os.path.join(
            os.path.dirname(PROJECT_DIR),
            "qwenvl3",
            "transformers_replace",
            "models",
            "qwen3_vl",
        ),
    ]
    src = next((path for path in candidates if path and os.path.isdir(path)), None)
    if src is None:
        raise FileNotFoundError(
            "Qwen3-VL replacement source not found; set QWEN3_VL_SOURCE"
        )

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    with open(model_file, "r", encoding="utf-8") as f:
        if marker not in f.read():
            raise RuntimeError(f"Qwen3-VL replacement check failed: {model_file}")


def _yaw_to_matrix(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return mat


def _camera_to_ego_matrix(yaw, pos):
    # Match LeadDataset._camera_to_ego_matrix used during training.
    # CARLA ego: x forward, y right/lateral, z up.
    # Pinhole camera: x right, y down, z forward.
    c = math.cos(yaw)
    s = math.sin(yaw)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.array(
        [
            [-s, 0.0, c],
            [c, 0.0, s],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    mat[:3, 3] = pos
    return mat


def _invert_pose(pose):
    inv_pose = np.eye(4, dtype=np.float32)
    inv_pose[:3, :3] = np.transpose(pose[:3, :3])
    inv_pose[:3, 3] = -inv_pose[:3, :3] @ pose[:3, 3]
    return inv_pose


def _camera_intrinsic(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0
    return intrinsic


def _camera_mats():
    lidar2img, lidar2cam, cam_intrinsic = [], [], []
    sparse4d_to_ego = LEFTHAND_EGO_TO_LIDAR
    for cam_id in sorted(LEAD_CAMERA_CALIBRATION.keys()):
        calib = LEAD_CAMERA_CALIBRATION[cam_id]
        width = float(calib.get("width", 384))
        height = float(calib.get("cropped_height", calib.get("height", 384)))
        fov = float(calib.get("fov", 60))
        pos = np.array(calib.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        yaw_deg = float(calib.get("rot", [0.0, 0.0, 0.0])[2])

        cam2ego = _camera_to_ego_matrix(math.radians(yaw_deg), pos)
        ego2cam = _invert_pose(cam2ego)
        intrinsic = _camera_intrinsic(width, height, fov)
        ego2img = intrinsic @ ego2cam
        sparse4d_lidar2img = ego2img @ sparse4d_to_ego
        sparse4d_lidar2cam = ego2cam @ sparse4d_to_ego

        lidar2img.append(sparse4d_lidar2img.astype(np.float32))
        lidar2cam.append(sparse4d_lidar2cam.T.astype(np.float32))
        cam_intrinsic.append(intrinsic.astype(np.float32))
    return lidar2img, lidar2cam, cam_intrinsic


def _unwrap_data_containers(inputs, device):
    for key, value in list(inputs.items()):
        if isinstance(value, DataContainer):
            value = value.data[0]
        elif isinstance(value, list) and value and isinstance(value[0], DataContainer):
            value = value[0].data
        inputs[key] = _to_device(value, device)
    return inputs


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _state_dict_from_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "module", "model"]:
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {path}")


def _planning_config_signature(cfg):
    planning = cfg.model.planning_head
    action_horizon = int(planning.get("action_horizon", 6))
    waypoint_hz = float(
        planning.get("waypoint_hz", 4.0 if action_horizon == 8 else 2.0)
    )
    loss_planning = planning.get("loss_planning", {})
    decoder = planning.get("unified_decoder_cfg", {})
    delta_mu = planning.get("action_delta_mu", loss_planning.get("delta_mu", None))
    delta_std = planning.get("action_delta_std", loss_planning.get("delta_std", None))
    planning_status_layout = str(
        planning.get("planning_status_layout", "legacy_fused")
    )
    prefix_max_len = planning.get("prefix_max_len", 3670)

    return {
        "action_dim": int(planning.get("action_dim", 2)),
        "action_horizon": action_horizon,
        "waypoint_hz": waypoint_hz,
        "num_sample_steps": int(planning.get("num_sample_steps", 10)),
        "action_delta_mu": None if delta_mu is None else [float(x) for x in delta_mu],
        "action_delta_std": None if delta_std is None else [float(x) for x in delta_std],
        "open_loop_hz": float(decoder.get("open_loop_hz", 2.0)),
        "close_loop_hz": float(decoder.get("close_loop_hz", 20.0)),
        "system_prompt": str(planning.get("system_prompt", "")),
        "user_prompt_template": str(planning.get("user_prompt_template", "")),
        "strict_user_prompt_fields": (
            None
            if planning.get("strict_user_prompt_fields", None) is None
            else list(planning.get("strict_user_prompt_fields"))
        ),
        "aux_route_loss_weight": float(planning.get("aux_route_loss_weight", 0.0)),
        "aux_target_speed_loss_weight": float(
            planning.get("aux_target_speed_loss_weight", 0.0)
        ),
        "planning_query_layout": str(
            planning.get("planning_query_layout", "perception_shared")
        ),
        "planning_status_layout": planning_status_layout,
        "num_status_tokens": 5 if planning_status_layout == "tfv6_5token" else 1,
        "use_deterministic_waypoint_queries": bool(
            planning.get("use_deterministic_waypoint_queries", False)
        ),
        "deterministic_waypoint_loss_weight": float(
            planning.get("deterministic_waypoint_loss_weight", 0.0)
        ),
        "spatial_waypoint_queries": int(
            planning.get("spatial_waypoint_queries", 6)
        ),
        "spatial_waypoint_loss_weight": float(
            planning.get("spatial_waypoint_loss_weight", 0.0)
        ),
        "inject_navigation_into_ego": bool(
            planning.get("inject_navigation_into_ego", True)
        ),
        "velocity_normalization": float(planning.get("velocity_normalization", 25.0)),
        "prefix_max_len": None if prefix_max_len is None else int(prefix_max_len),
        "route_points_prediction": int(planning.get("route_points_prediction", 10)),
        "target_speed_classes": [
            float(x)
            for x in planning.get(
                "target_speed_classes",
                [0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0],
            )
        ],
        "target_points_normalization_constants": [
            float(x)
            for x in planning.get(
                "target_points_normalization_constants", [200.0, 50.0]
            )
        ],
        "loss_planning_type": loss_planning.get("type", None),
        "use_min_snr_loss": bool(loss_planning.get("use_min_snr_loss", False)),
        "min_snr_gamma": float(loss_planning.get("min_snr_gamma", 0.0)),
        "hybrid_loss_weight": float(loss_planning.get("hybrid_loss_weight", 0.0)),
        "detach_window_size": int(loss_planning.get("detach_window_size", 0)),
    }


def _find_checkpoint_training_config(ckpt_path):
    checkpoint = pathlib.Path(ckpt_path).resolve()
    project_root = pathlib.Path(PROJECT_DIR).resolve()
    for parent in checkpoint.parents:
        candidates = sorted(parent.glob("*.py"))
        preferred = [
            path for path in candidates
            if "unidrivevla" in path.name
        ]
        if preferred:
            return preferred[0]
        if parent == project_root:
            break
    return None


def _planning_config_values_equal(runtime_value, training_value):
    if isinstance(runtime_value, list) or isinstance(training_value, list):
        if not (
            isinstance(runtime_value, list)
            and isinstance(training_value, list)
            and len(runtime_value) == len(training_value)
        ):
            return False
        if runtime_value == training_value:
            return True
        try:
            return bool(
                np.allclose(
                    runtime_value,
                    training_value,
                    rtol=0.0,
                    atol=1e-6,
                )
            )
        except (TypeError, ValueError):
            return False
    return runtime_value == training_value


def _validate_checkpoint_training_config(runtime_cfg, ckpt_path):
    allow_mismatch = os.environ.get(
        "UNIDRIVEVLA_LEAD_ALLOW_CONFIG_MISMATCH", "0"
    ) in {"1", "true", "True"}
    training_config_path = _find_checkpoint_training_config(ckpt_path)
    if training_config_path is None:
        message = (
            "Checkpoint training config not found; planning semantics cannot be "
            f"verified for {ckpt_path}. Keep the dumped *.py config beside the checkpoint."
        )
        if not allow_mismatch:
            raise RuntimeError(
                message
                + " Set UNIDRIVEVLA_LEAD_ALLOW_CONFIG_MISMATCH=1 only for diagnostics."
            )
        print(f"[UniDriveVLALEADAgent] {message}", flush=True)
        return

    training_cfg = Config.fromfile(str(training_config_path))
    runtime_signature = _planning_config_signature(runtime_cfg)
    training_signature = _planning_config_signature(training_cfg)
    mismatches = {}
    for key, runtime_value in runtime_signature.items():
        training_value = training_signature[key]
        equal = _planning_config_values_equal(runtime_value, training_value)
        if not equal:
            mismatches[key] = (training_value, runtime_value)
    if mismatches and not allow_mismatch:
        raise RuntimeError(
            "Checkpoint/config planning semantics mismatch. "
            f"training_config={training_config_path} mismatches={mismatches}. "
            "Use the checkpoint's frozen config or explicitly set "
            "UNIDRIVEVLA_LEAD_ALLOW_CONFIG_MISMATCH=1 only for diagnostics."
        )
    print(
        f"[UniDriveVLALEADAgent] checkpoint training config={training_config_path}",
        flush=True,
    )


def _load_checkpoint(model, ckpt_path):
    state_dict = _state_dict_from_checkpoint(ckpt_path)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if (missing or unexpected) and os.environ.get(
        "UNIDRIVEVLA_LEAD_ALLOW_PARTIAL_CHECKPOINT", "0"
    ) not in {"1", "true", "True"}:
        raise RuntimeError(
            "Checkpoint is not architecture-complete. Set "
            "UNIDRIVEVLA_LEAD_ALLOW_PARTIAL_CHECKPOINT=1 only for legacy diagnostics."
        )
    print(
        f"[UniDriveVLALEADAgent] loaded checkpoint {ckpt_path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[UniDriveVLALEADAgent] first missing keys: {missing[:20]}", flush=True)
    if unexpected:
        print(f"[UniDriveVLALEADAgent] first unexpected keys: {unexpected[:20]}", flush=True)


def _convert_plan_for_pid(plan, coord_mode):
    if coord_mode in ("pid_lateral_forward", "identity", "none"):
        return plan
    if coord_mode == "lead_forward_lateral":
        return np.stack([plan[:, 1], plan[:, 0]], axis=-1)
    if coord_mode == "lead_forward_right":
        return np.stack([-plan[:, 1], plan[:, 0]], axis=-1)
    raise ValueError(
        "Unsupported UNIDRIVEVLA_LEAD_PLAN_COORD="
        f"{coord_mode!r}; expected lead_forward_lateral, lead_forward_right, "
        "or pid_lateral_forward"
    )


class _LeadPIDController:
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.window = [0.0 for _ in range(int(n))]

    def step(self, error):
        self.window.pop(0)
        self.window.append(float(error))
        integral = sum(self.window) / len(self.window)
        derivative = self.window[-1] - self.window[-2] if len(self.window) >= 2 else 0.0
        return self.k_p * error + self.k_i * integral + self.k_d * derivative

    def reset(self):
        self.window = [0.0 for _ in self.window]


def _make_lead_waypoint_controllers():
    return (
        _LeadPIDController(k_p=1.25, k_i=0.75, k_d=0.3, n=20),
        _LeadPIDController(k_p=1.75, k_i=1.0, k_d=2.0, n=20),
    )


def _convert_gps_to_carla(gps, lat_ref, lon_ref):
    gps = np.asarray(gps, dtype=np.float64).reshape(-1)
    if gps.shape[0] == 2:
        gps = np.array([gps[0], gps[1], 0.0], dtype=np.float64)
    earth_radius = 6378137.0
    lat, lon = gps[:2]
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (earth_radius * scale)
    mx = (lon * (math.pi * earth_radius * scale)) / 180.0
    y = scale * earth_radius * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my
    x = mx - scale * lon_ref * math.pi * earth_radius / 180.0
    return np.array([x, y, gps[2]], dtype=np.float64)


def _inverse_conversion_2d(point, translation, yaw):
    point = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)[:2]
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return (rotation.T @ (point - translation)).astype(np.float32)


def _navigation_points_to_ego(route_points, ego_position, ego_yaw):
    points = [np.asarray(point, dtype=np.float64).reshape(-1)[:2] for point in route_points]
    if not points:
        points = [np.asarray(ego_position, dtype=np.float64).reshape(-1)[:2]]
    previous_global = points[0]
    current_global = points[1] if len(points) > 1 else previous_global
    next_global = points[2] if len(points) > 2 else current_global
    return (
        _inverse_conversion_2d(previous_global, ego_position, ego_yaw),
        _inverse_conversion_2d(current_global, ego_position, ego_yaw),
        _inverse_conversion_2d(next_global, ego_position, ego_yaw),
    )


def _find_gps_ref(global_plan_world_coord, global_plan):
    try:
        locx = global_plan_world_coord[0][0].location.x
        locy = global_plan_world_coord[0][0].location.y
        lon = global_plan[0][0]["lon"]
        lat = global_plan[0][0]["lat"]
        earth_radius = 6378137.0

        from scipy.optimize import fsolve

        def equations(vars_):
            x, y = vars_
            eq1 = (
                lon * math.cos(x * math.pi / 180.0)
                - (locx * x * 180.0) / (math.pi * earth_radius)
                - math.cos(x * math.pi / 180.0) * y
            )
            eq2 = (
                math.log(math.tan((lat + 90.0) * math.pi / 360.0))
                * earth_radius
                * math.cos(x * math.pi / 180.0)
                + locy
                - math.cos(x * math.pi / 180.0)
                * earth_radius
                * math.log(math.tan((90.0 + x) * math.pi / 360.0))
            )
            return [eq1, eq2]

        solution = fsolve(equations, [0.0, 0.0])
        return float(solution[0]), float(solution[1])
    except Exception as exc:
        print(f"[UniDriveVLALEADAgent] route ref solve failed: {exc}", flush=True)
        return 0.0, 0.0


class _LeadRoutePlanner:
    def __init__(self, min_distance, max_distance, lat_ref=0.0, lon_ref=0.0):
        self.route = []
        self.route_distances = []
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.lat_ref = float(lat_ref)
        self.lon_ref = float(lon_ref)
        self.is_last = False

    def set_route(self, global_plan, gps=False):
        self.route = []
        for pos, cmd in global_plan:
            if gps:
                pos = _convert_gps_to_carla(
                    np.array([pos["lat"], pos["lon"], pos.get("z", 0.0)], dtype=np.float64),
                    self.lat_ref,
                    self.lon_ref,
                )
            else:
                pos = np.array([pos.location.x, pos.location.y, pos.location.z], dtype=np.float64)
            self.route.append((pos, cmd))

        self.route_distances = [0.0]
        for i in range(1, len(self.route)):
            diff = self.route[i][0] - self.route[i - 1][0]
            self.route_distances.append(float(np.linalg.norm(diff[:2])))

    def run_step(self, gps):
        gps = np.asarray(gps, dtype=np.float64).reshape(-1)
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += self.route_distances[i]
            distance = float(np.linalg.norm(self.route[i][0][:2] - gps[:2]))
            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.pop(0)
                self.route_distances.pop(0)
        return self.route


def _command_to_int(command):
    if hasattr(command, "value"):
        command = command.value
    try:
        return int(command)
    except Exception:
        return 4


def _dedupe_route_points_and_commands(route):
    points, commands = [], []
    route_list = list(route)
    for point, command in route_list:
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        if len(route_list) == 2 or not points or not np.allclose(point[:2], points[-1][:2]):
            points.append(point)
            commands.append(_command_to_int(command))
    return points, commands


def _lead_route_is_dense(points, ego_xy):
    if len(points) <= 2:
        return False
    rel = [np.asarray(point[:2], dtype=np.float32) - ego_xy for point in points[:3]]
    prev_pt, target_pt, next_pt = rel
    close_next = np.linalg.norm(target_pt - next_pt) < 10.0
    close_prev = np.linalg.norm(prev_pt - target_pt) < 10.0
    near_prev_or_target = min(np.linalg.norm(prev_pt), np.linalg.norm(target_pt)) < 10.0
    return bool((close_next and near_prev_or_target) or (close_prev and near_prev_or_target))


def _control_with_lead_waypoints(
    waypoints,
    speed,
    turn_controller,
    speed_controller,
    waypoint_hz,
):
    # LEAD official waypoint control expects waypoints as (forward, lateral).
    waypoints = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
    speed = float(speed)
    if waypoints.shape[0] < 2:
        raise ValueError(f"Expected at least 2 waypoints, got {waypoints.shape}")

    one_second = max(1, min(waypoints.shape[0], int(round(waypoint_hz))))
    half_second = max(1, min(one_second, int(round(0.5 * waypoint_hz))))
    # These indices always span 0.5 seconds, so converting displacement to m/s
    # requires a fixed factor of 2 even when the waypoint sampling rate changes.
    desired_speed = float(
        np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
    )
    delta_speed = float(np.clip(desired_speed - speed, 0.0, LEAD_WP_DELTA_CLIP))
    brake = bool(
        desired_speed < LEAD_WP_BRAKE_SPEED
        or (speed / max(desired_speed, 1e-3)) > LEAD_WP_BRAKE_RATIO
    )
    throttle = float(np.clip(speed_controller.step(delta_speed), 0.0, 1.0))
    throttle = throttle if not brake else 0.0

    if desired_speed < LEAD_WP_AIM_DISTANCE_THRESHOLD:
        aim_distance = LEAD_WP_AIM_DISTANCE_SLOW
    else:
        aim_distance = LEAD_WP_AIM_DISTANCE_FAST

    aim_index = waypoints.shape[0] - 1
    for index, predicted_waypoint in enumerate(waypoints):
        if np.linalg.norm(predicted_waypoint) >= aim_distance:
            aim_index = index
            break

    aim = waypoints[aim_index]
    angle = float(np.degrees(np.arctan2(aim[1], aim[0])) / 90.0)
    if speed < 0.01 or brake:
        angle = 0.0

    steer = float(np.clip(turn_controller.step(angle), -1.0, 1.0))
    metadata = {
        "lead_waypoint_speed": speed,
        "lead_waypoint_steer": steer,
        "lead_waypoint_throttle": throttle,
        "lead_waypoint_brake": float(brake),
        "lead_waypoint_desired_speed": desired_speed,
        "lead_waypoint_delta_speed": delta_speed,
        "lead_waypoint_angle": angle,
        "lead_waypoint_aim": tuple(aim.astype(np.float64)),
        "lead_waypoint_aim_index": int(aim_index),
        "lead_waypoint_aim_distance": float(aim_distance),
    }
    return steer, throttle, float(brake), metadata


class _TFV6LateralController:
    """Python 3.8-compatible equivalent of LEAD's TFv6 lateral controller."""

    def __init__(self):
        self._window = []

    def step(
        self,
        route,
        current_speed,
        ego_vehicle_location=0.0,
        ego_vehicle_rotation=0.0,
        sensor_agent_steer_correction=True,
    ):
        route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
        current_speed_kmh = float(current_speed) * 3.6
        n_lookahead = (
            np.clip(
                TFV6_LATERAL_SPEED_SCALE * current_speed_kmh
                + TFV6_LATERAL_SPEED_OFFSET,
                24,
                105,
            )
            / 10
        )
        n_lookahead = int(min(n_lookahead - 2, route.shape[0] - 1))
        curvature = float(np.mean(np.abs(np.arctan2(route[:, 1], route[:, 0]))))
        if sensor_agent_steer_correction:
            n_lookahead += int(
                np.clip(int(curvature * TFV6_STEER_CORRECTION), 0, 2)
            )

        desired_heading_vec = route[n_lookahead] - ego_vehicle_location
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = (yaw_path - ego_vehicle_rotation) % (2 * np.pi)
        if heading_error >= np.pi:
            heading_error -= 2 * np.pi
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(float(heading_error))
        self._window = self._window[-TFV6_LATERAL_WINDOW:]
        derivative = (
            0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        )
        integral = float(np.mean(self._window))
        steering = np.clip(
            TFV6_LATERAL_KP * heading_error
            + TFV6_LATERAL_KD * derivative
            + TFV6_LATERAL_KI * integral,
            -1.0,
            1.0,
        )
        return round(float(steering), 3)


def _tfv6_get_throttle(brake, target_speed, speed):
    """Python 3.8-compatible equivalent of LEAD's TFv6 longitudinal control."""
    target_speed = float(target_speed)
    speed = float(speed)
    if target_speed < 1e-5 or brake:
        return 0.0, True
    if target_speed < 1.0 / 3.6:
        target_speed = 1.0 / 3.6

    speed_kmh = speed * 3.6
    target_speed_kmh = target_speed * 3.6
    speed_error = target_speed_kmh - speed_kmh
    if speed_error > TFV6_LONGITUDINAL_MAX_ACCELERATION:
        return 1.0, False
    if speed_kmh / target_speed_kmh > TFV6_LONGITUDINAL_PARAMS[-1]:
        return 0.0, True

    speed_error_clipped = np.clip(speed_error, 0.0, np.inf) / 100.0
    normalized_speed = speed_kmh / 100.0
    features = np.asarray(
        [
            normalized_speed,
            normalized_speed**2,
            100 * speed_error_clipped,
            speed_error_clipped**2,
            normalized_speed * speed_error_clipped,
            normalized_speed**2 * speed_error_clipped,
        ],
        dtype=np.float64,
    )
    throttle = np.clip(features @ TFV6_LONGITUDINAL_PARAMS[:-1], 0.0, 1.0)
    return float(throttle), False


def _decode_tfv6_target_speed(
    pred_target_speed,
    pred_target_speed_distribution=None,
):
    """Apply TFv6's high-confidence stop override to the decoded speed."""
    raw_target_speed = float(pred_target_speed)
    stop_probability = None
    if pred_target_speed_distribution is not None:
        distribution = np.asarray(
            pred_target_speed_distribution,
            dtype=np.float32,
        ).reshape(-1)
        if distribution.size == 0 or not np.isfinite(distribution).all():
            raise ValueError("Invalid TFv6 target-speed distribution")
        stop_probability = float(distribution[0])
    target_speed = raw_target_speed
    if (
        stop_probability is not None
        and stop_probability > TFV6_STOP_PROBABILITY_THRESHOLD
    ):
        target_speed = 0.0
    return target_speed, raw_target_speed, stop_probability


def _control_with_tfv6_route_speed(
    pred_route,
    pred_target_speed,
    pred_target_speed_distribution,
    speed,
    lateral_controller,
):
    """Apply LEAD's official TFV6 route and target-speed controller."""
    pred_route = np.asarray(pred_route, dtype=np.float32).reshape(-1, 2)
    (
        pred_target_speed,
        raw_target_speed,
        stop_probability,
    ) = _decode_tfv6_target_speed(
        pred_target_speed,
        pred_target_speed_distribution,
    )
    speed = float(speed)
    if pred_route.shape[0] < 2:
        raise ValueError(f"Expected at least 2 route points, got {pred_route.shape}")
    if not np.isfinite(pred_route).all() or not np.isfinite(pred_target_speed):
        raise ValueError("Non-finite TFV6 route or target-speed prediction")

    brake = bool(
        pred_target_speed < 0.01
        or (speed / pred_target_speed) > TFV6_ROUTE_BRAKE_RATIO
    )
    steer = lateral_controller.step(
        pred_route,
        speed,
        0.0,
        0.0,
        sensor_agent_steer_correction=True,
    )
    throttle, brake = _tfv6_get_throttle(
        brake,
        pred_target_speed,
        speed,
    )
    if brake and speed < 0.01:
        steer = 0.0
    metadata = {
        "tfv6_route_speed_speed": speed,
        "tfv6_route_speed_steer": float(steer),
        "tfv6_route_speed_throttle": float(throttle),
        "tfv6_route_speed_brake": float(brake),
        "tfv6_route_speed_target_speed": pred_target_speed,
        "tfv6_route_speed_raw_target_speed": raw_target_speed,
        "tfv6_route_speed_stop_probability": stop_probability,
        "tfv6_route_speed_route": pred_route.tolist(),
    }
    return float(steer), float(throttle), float(brake), metadata


class _HybridLongitudinalController:
    """Smooth target-speed tracking with asymmetric safety behavior."""

    def __init__(self):
        self.pid = _LeadPIDController(k_p=0.35, k_i=0.05, k_d=0.1, n=10)
        self.previous_throttle = 0.0

    def step(self, target_speed, speed, force_stop=False):
        target_speed = max(0.0, float(target_speed))
        speed = max(0.0, float(speed))
        if force_stop or target_speed <= 0.05:
            self.pid.reset()
            self.previous_throttle = 0.0
            return 0.0, 1.0

        speed_error = target_speed - speed
        if speed_error < -HYBRID_SPEED_DEADBAND:
            self.pid.reset()
            self.previous_throttle = 0.0
            brake = np.clip(
                -speed_error / HYBRID_FULL_BRAKE_SPEED_ERROR,
                0.1,
                1.0,
            )
            return 0.0, float(brake)
        if speed_error <= HYBRID_SPEED_DEADBAND:
            self.previous_throttle = 0.0
            return 0.0, 0.0

        throttle = float(
            np.clip(self.pid.step(speed_error), 0.0, HYBRID_MAX_THROTTLE)
        )
        throttle = min(
            throttle,
            self.previous_throttle + HYBRID_MAX_THROTTLE_RISE_PER_STEP,
        )
        self.previous_throttle = throttle
        return throttle, 0.0


class _HybridRouteSpeedController:
    """Route steering with conservative, confidence-aware longitudinal control."""

    def __init__(self):
        self.lateral_controller = _TFV6LateralController()
        self.longitudinal_controller = _HybridLongitudinalController()
        self.filtered_target_speed = 0.0
        self.stop_latched = False
        self.previous_steer = 0.0

    def step(
        self,
        pred_route,
        pred_target_speed,
        pred_target_speed_distribution,
        speed,
        trajectory_desired_speed,
        trajectory_steer,
    ):
        pred_route = np.asarray(pred_route, dtype=np.float32).reshape(-1, 2)
        if pred_route.shape[0] < 2 or not np.isfinite(pred_route).all():
            raise ValueError(f"Invalid hybrid route prediction: {pred_route.shape}")

        (
            decoded_target_speed,
            raw_target_speed,
            stop_probability,
        ) = _decode_tfv6_target_speed(
            pred_target_speed,
            pred_target_speed_distribution,
        )
        if stop_probability is not None:
            if stop_probability > TFV6_STOP_PROBABILITY_THRESHOLD:
                self.stop_latched = True
            elif stop_probability < HYBRID_STOP_RELEASE_PROBABILITY:
                self.stop_latched = False

        raw_steer = self.lateral_controller.step(
            pred_route,
            speed,
            0.0,
            0.0,
            sensor_agent_steer_correction=True,
        )
        steer = float(
            np.clip(
                raw_steer,
                self.previous_steer - HYBRID_MAX_STEER_DELTA,
                self.previous_steer + HYBRID_MAX_STEER_DELTA,
            )
        )
        self.previous_steer = steer

        route_curvature = float(
            np.mean(np.abs(np.arctan2(pred_route[:, 1], pred_route[:, 0])))
        )
        curvature_speed_cap = float(
            np.clip(
                HYBRID_CURVATURE_SPEED_FACTOR / max(route_curvature, 0.15),
                HYBRID_MIN_CURVATURE_SPEED,
                HYBRID_MAX_TARGET_SPEED,
            )
        )
        trajectory_desired_speed = max(0.0, float(trajectory_desired_speed))
        trajectory_speed_cap = max(
            HYBRID_MIN_CREEP_SPEED,
            trajectory_desired_speed + HYBRID_TRAJECTORY_SPEED_MARGIN,
        )
        steer_disagreement = abs(float(raw_steer) - float(trajectory_steer))

        requested_target_speed = min(
            max(0.0, decoded_target_speed),
            HYBRID_MAX_TARGET_SPEED,
            curvature_speed_cap,
            trajectory_speed_cap,
        )
        if steer_disagreement > HYBRID_STEER_DISAGREEMENT_THRESHOLD:
            requested_target_speed = min(
                requested_target_speed,
                HYBRID_DISAGREEMENT_SPEED_CAP,
            )
        if self.stop_latched:
            requested_target_speed = 0.0

        if requested_target_speed <= self.filtered_target_speed:
            self.filtered_target_speed = requested_target_speed
        else:
            self.filtered_target_speed = min(
                requested_target_speed,
                self.filtered_target_speed + HYBRID_TARGET_SPEED_RISE_PER_STEP,
            )

        throttle, brake = self.longitudinal_controller.step(
            self.filtered_target_speed,
            speed,
            force_stop=self.stop_latched,
        )
        metadata = {
            "hybrid_route_speed_speed": float(speed),
            "hybrid_route_speed_steer": steer,
            "hybrid_route_speed_raw_steer": float(raw_steer),
            "hybrid_route_speed_trajectory_steer": float(trajectory_steer),
            "hybrid_route_speed_steer_disagreement": steer_disagreement,
            "hybrid_route_speed_throttle": float(throttle),
            "hybrid_route_speed_brake": float(brake),
            "hybrid_route_speed_raw_target_speed": raw_target_speed,
            "hybrid_route_speed_decoded_target_speed": decoded_target_speed,
            "hybrid_route_speed_requested_target_speed": requested_target_speed,
            "hybrid_route_speed_target_speed": self.filtered_target_speed,
            "hybrid_route_speed_stop_probability": stop_probability,
            "hybrid_route_speed_stop_latched": self.stop_latched,
            "hybrid_route_speed_route_curvature": route_curvature,
            "hybrid_route_speed_curvature_speed_cap": curvature_speed_cap,
            "hybrid_route_speed_trajectory_desired_speed": trajectory_desired_speed,
            "hybrid_route_speed_trajectory_speed_cap": trajectory_speed_cap,
            "hybrid_route_speed_route": pred_route.tolist(),
        }
        return steer, float(throttle), float(brake), metadata


class _TFV6KalmanFilter:
    """Python 3.8-compatible online UKF matching LEAD BaseAgent."""

    def __init__(self):
        self.points = MerweScaledSigmaPoints(
            n=4, alpha=0.00001, beta=2, kappa=0, subtract=self._residual
        )
        self.ukf = UKF(
            dim_x=4,
            dim_z=4,
            fx=self._bicycle_model_forward,
            hx=self._measurement_function,
            dt=0.05,
            points=self.points,
            x_mean_fn=self._state_mean,
            z_mean_fn=self._state_mean,
            residual_x=self._residual,
            residual_z=self._residual,
        )
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        self.ukf.R = np.diag([0.5, 0.5, 1e-15, 1e-15])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
        self.filter_initialized = False
        self.start_x = None
        self.start_y = None

    
    @staticmethod
    def _normalize_angle(angle):
        angle = float(angle) % (2.0 * np.pi)
        if angle > np.pi:
            angle -= 2.0 * np.pi
        return angle

    def step(self, noisy_position, compass, speed, control):
        noisy_position = np.asarray(noisy_position, dtype=np.float64).reshape(-1)
        if self.start_x is None:
            self.start_x = float(noisy_position[0])
            self.start_y = float(noisy_position[1])
        measurement = np.asarray(
            [
                noisy_position[0] - self.start_x,
                noisy_position[1] - self.start_y,
                self._normalize_angle(compass),
                float(speed),
            ],
            dtype=np.float64,
        )
        if not self.filter_initialized:
            self.ukf.x = measurement
            self.filter_initialized = True
        self.ukf.predict(
            steer=float(control.steer),
            throttle=float(control.throttle),
            brake=float(control.brake),
        )
        self.ukf.update(measurement)
        prediction = self.ukf.x.copy()
        prediction[0] += self.start_x
        prediction[1] += self.start_y
        return prediction

    
    @staticmethod
    def _bicycle_model_forward(state, dt, steer, throttle, brake):
        front_wb = -0.090769015
        rear_wb = 1.4178275
        steer_gain = 0.36848336
        brake_accel = -4.952399
        throttle_accel = 0.5633837
        accel = brake_accel if brake else throttle_accel * throttle
        wheel = steer_gain * steer
        beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
        x, y, yaw, speed = state
        next_speed = max(0.0, speed + accel * dt)
        return np.asarray(
            [
                float(x) + speed * math.cos(yaw + beta) * dt,
                float(y) + speed * math.sin(yaw + beta) * dt,
                yaw + speed / rear_wb * math.sin(beta) * dt,
                next_speed,
            ],
            dtype=np.float64,
        )

    
    @staticmethod
    def _measurement_function(state):
        return state

    
    @staticmethod
    def _state_mean(states, weights):
        mean = np.zeros(4, dtype=np.float64)
        mean[0] = np.dot(states[:, 0], weights)
        mean[1] = np.dot(states[:, 1], weights)
        mean[2] = math.atan2(
            np.dot(np.sin(states[:, 2]), weights),
            np.dot(np.cos(states[:, 2]), weights),
        )
        mean[3] = np.dot(states[:, 3], weights)
        return mean

    
    @classmethod
    def _residual(cls, left, right):
        residual = left - right
        residual[2] = cls._normalize_angle(residual[2])
        return residual


def _lidar_to_ego_for_creep(raw_lidar):
    points = np.asarray(raw_lidar, dtype=np.float64)[:, :3]
    yaw = np.deg2rad(LEAD_LIDAR_ROTATION_DEG[2])
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points = (rotation @ points.T).T + LEAD_LIDAR_POSITION
    points[:, 2] -= LEAD_LIDAR_POSITION[2] / 2.0
    return points


def _euler_deg_to_mat_for_creep(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    rotation_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    rotation_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return rotation_z @ rotation_y @ rotation_x


def _radar_points_to_ego_for_creep(raw_radar, sensor_pos, sensor_rot):
    raw_radar = np.asarray(raw_radar)
    radius = raw_radar[:, 0]
    altitude = raw_radar[:, 1]
    azimuth = raw_radar[:, 2]
    velocity = raw_radar[:, 3]
    points = np.stack(
        [
            radius * np.cos(azimuth) * np.cos(altitude),
            radius * np.sin(azimuth) * np.cos(altitude),
            radius * np.sin(altitude),
        ],
        axis=1,
    ).astype(np.float32)
    rotation = _euler_deg_to_mat_for_creep(*sensor_rot)
    position = np.asarray(sensor_pos, dtype=np.float32).reshape(1, 3)
    points = (rotation @ points.T).T + position
    points[:, 2] -= sensor_pos[-1] / 2.0
    return np.concatenate([points, velocity.reshape(-1, 1)], axis=1)


def _sensor_lidar_to_ego_for_creep(raw_lidar, position, rotation_deg):
    points = np.asarray(raw_lidar, dtype=np.float64)[:, :3]
    rotation = _euler_deg_to_mat_for_creep(
        float(rotation_deg[0]),
        float(rotation_deg[1]),
        float(rotation_deg[2]),
    )
    position = np.asarray(position, dtype=np.float64)
    points = (rotation @ points.T).T + position
    points[:, 2] -= position[2] / 2.0
    return points


def _official_tfv6_creep_lidar(input_data, config):
    """Build the current-frame safety cloud like the official SensorAgent."""
    required_lidars = ["lidar1"]
    if config.use_two_lidars:
        required_lidars.append("lidar2")
    missing = [sensor_id for sensor_id in required_lidars if sensor_id not in input_data]
    if missing:
        raise RuntimeError(
            "Official TFV6 creeping is missing LiDAR sensor data: "
            + ", ".join(missing)
        )

    lidar_parts = [
        _sensor_lidar_to_ego_for_creep(
            input_data["lidar1"][1], config.lidar_pos_1, config.lidar_rot_1
        )
    ]
    if config.use_two_lidars:
        lidar_parts.append(
            _sensor_lidar_to_ego_for_creep(
                input_data["lidar2"][1], config.lidar_pos_2, config.lidar_rot_2
            )
        )
    points = np.concatenate(lidar_parts, axis=0)
    points = points[
        (np.abs(points[:, 0]) > config.ego_extent_x)
        & (np.abs(points[:, 1]) > config.ego_extent_y)
    ]

    if config.use_radars:
        radar_parts = []
        missing_radars = []
        for sensor_index, sensor_config in config.radar_calibration.items():
            sensor_id = f"radar{sensor_index}"
            if sensor_id not in input_data:
                missing_radars.append(sensor_id)
                continue
            radar = _radar_points_to_ego_for_creep(
                input_data[sensor_id][1],
                sensor_pos=sensor_config["pos"],
                sensor_rot=sensor_config["rot"],
            )
            radar_parts.append(radar[:, :3])
        if missing_radars:
            raise RuntimeError(
                "Official TFV6 creeping is missing radar sensor data: "
                + ", ".join(missing_radars)
            )
        radar_points = np.concatenate(radar_parts, axis=0)
        if config.save_radar_pc_as_lidar:
            points = np.concatenate([points, radar_points], axis=0)
            if config.duplicate_radar_near_ego:
                radar_near_ego = radar_points[
                    np.linalg.norm(radar_points[:, :2], axis=1)
                    < config.duplicate_radar_radius
                ]
                points = np.concatenate(
                    [
                        points,
                        np.concatenate(
                            [radar_near_ego] * config.duplicate_radar_factor,
                            axis=0,
                        ),
                    ],
                    axis=0,
                )

    if config.save_only_non_ground_lidar and points.shape[0] > 0:
        ground_mask = tfv6_ransac.remove_ground(points, config, parallel=True)
        points = points[~ground_mask]

    if config.save_lidar_only_inside_bev:
        inside_mask = (
            (config.min_x_meter <= points[:, 0])
            & (points[:, 0] <= config.max_x_meter)
            & (config.min_y_meter <= points[:, 1])
            & (points[:, 1] <= config.max_y_meter)
            & (config.min_height_lidar <= points[:, 2])
            & (points[:, 2] <= config.max_height_lidar)
        )
        points = points[inside_mask]
    return points


class _TFV6ForceMovePostProcessor:
    """Optional official creeping heuristic with a LiDAR-only safety gate."""

    def __init__(self):
        self.stuck_detector = 0
        self.force_move = 0

    def adjust(self, ego_speed, throttle, brake, lidar_points):
        if float(ego_speed) < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
        if self.stuck_detector > LEAD_CREEP_STUCK_THRESHOLD:
            self.force_move = LEAD_CREEP_MOVE_DURATION

        applied = False
        safety_blocked = False
        safety_count = 0
        if self.force_move > 0:
            if lidar_points is None:
                raise RuntimeError("Creeping is enabled but lidar1 data is missing")
            points = np.asarray(lidar_points, dtype=np.float64).reshape(-1, 3)
            mask = (
                (points[:, 2] > LEAD_CREEP_SAFETY_Z[0])
                & (points[:, 2] < LEAD_CREEP_SAFETY_Z[1])
                & (points[:, 1] > LEAD_CREEP_SAFETY_Y[0])
                & (points[:, 1] < LEAD_CREEP_SAFETY_Y[1])
                & (points[:, 0] > LEAD_CREEP_SAFETY_X[0])
                & (points[:, 0] < LEAD_CREEP_SAFETY_X[1])
            )
            safety_count = int(mask.sum())
            safety_blocked = safety_count > 0
            if safety_blocked:
                throttle, brake = 0.0, 1.0
                self.force_move = LEAD_CREEP_MOVE_DURATION
            else:
                throttle = max(LEAD_CREEP_THROTTLE, float(throttle))
                brake = 0.0
                self.force_move -= 1
                applied = True

        metadata = {
            "creeping_enabled": True,
            "creeping_applied": applied,
            "creeping_safety_blocked": safety_blocked,
            "creeping_safety_point_count": safety_count,
            "creeping_stuck_ticks": int(self.stuck_detector),
            "creeping_force_move_remaining": int(self.force_move),
        }
        return float(throttle), float(brake), metadata


class UniDriveVLALEADAgent(autonomous_agent.AutonomousAgent):
    def sensors(self):
        sensors = []
        for cam_id, calib in LEAD_CAMERA_CALIBRATION.items():
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": calib["pos"][0],
                    "y": calib["pos"][1],
                    "z": calib["pos"][2],
                    "roll": calib["rot"][0],
                    "pitch": calib["rot"][1],
                    "yaw": calib["rot"][2],
                    "width": calib["width"],
                    "height": calib["height"],
                    "fov": calib["fov"],
                    "id": f"rgb_{cam_id}",
                }
            )
        sensors.extend(
            [
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
                {
                    "type": "sensor.speedometer",
                    "reading_frequency": 20,
                    "id": "speed",
                },
            ]
        )
        if LEAD_CREEPING:
            if LEAD_CREEP_POINT_SOURCE == "official_tfv6":
                config = getattr(self, "creep_expert_config", None) or ExpertConfig()
                lidar_configs = [("lidar1", config.lidar_pos_1, config.lidar_rot_1)]
                if config.use_two_lidars:
                    lidar_configs.append(
                        ("lidar2", config.lidar_pos_2, config.lidar_rot_2)
                    )
                for sensor_id, position, rotation in lidar_configs:
                    sensors.append(
                        {
                            "type": "sensor.lidar.ray_cast",
                            "x": float(position[0]),
                            "y": float(position[1]),
                            "z": float(position[2]),
                            "roll": float(rotation[0]),
                            "pitch": float(rotation[1]),
                            "yaw": float(rotation[2]),
                            "id": sensor_id,
                        }
                    )
                if config.use_radars:
                    for sensor_index, sensor_config in config.radar_calibration.items():
                        sensors.append(
                            {
                                "type": "sensor.other.radar",
                                "x": sensor_config["pos"][0],
                                "y": sensor_config["pos"][1],
                                "z": sensor_config["pos"][2],
                                "roll": sensor_config["rot"][0],
                                "pitch": sensor_config["rot"][1],
                                "yaw": sensor_config["rot"][2],
                                "horizontal_fov": sensor_config["horz_fov"],
                                "vertical_fov": sensor_config["vert_fov"],
                                "id": f"radar{sensor_index}",
                            }
                        )
            else:
                sensors.append(
                    {
                        "type": "sensor.lidar.ray_cast",
                        "x": float(LEAD_LIDAR_POSITION[0]),
                        "y": float(LEAD_LIDAR_POSITION[1]),
                        "z": float(LEAD_LIDAR_POSITION[2]),
                        "roll": float(LEAD_LIDAR_ROTATION_DEG[0]),
                        "pitch": float(LEAD_LIDAR_ROTATION_DEG[1]),
                        "yaw": float(LEAD_LIDAR_ROTATION_DEG[2]),
                        "id": "lidar1",
                    }
                )
        if LEAD_RECORD_BEV_CAMERA:
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 60.0,
                    "roll": 0.0,
                    "pitch": -90.0,
                    "yaw": 0.0,
                    "width": 512,
                    "height": 512,
                    "fov": 50,
                    "id": "bev",
                }
            )
        return sensors

    def setup(self, path_to_conf_file):
        os.environ.setdefault("PROJECT_DIR", PROJECT_DIR)
        os.environ.setdefault(
            "VLM_PRETRAINED_PATH",
            os.environ.get("LEAD_QWEN_BASE", ""),
        )
        os.environ.setdefault("VLM_VARIANT", "2b")
        _install_qwen3_vl_replacement()

        parts = path_to_conf_file.split("+")
        if len(parts) < 2:
            raise ValueError(
                "TEAM_CONFIG must be '<config.py>+<checkpoint.pt>[+save_name]'"
            )
        self.config_path = parts[0]
        self.ckpt_path = parts[1]
        self.save_name = parts[2] if len(parts) > 2 else "unidrivevla_lead"

        self.device = "cuda"
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.frame_rate = 20
        self.initialized = False
        self.creep_expert_config = (
            ExpertConfig()
            if LEAD_CREEPING and LEAD_CREEP_POINT_SOURCE == "official_tfv6"
            else None
        )
        if self.creep_expert_config is not None:
            tfv6_ransac.remove_ground(
                np.random.rand(1000, 3),
                self.creep_expert_config,
                parallel=True,
            )
        self.data_aug_conf = None
        self.use_bgr_img = True
        self.wall_start = time.time()
        self.plan_coord_mode = os.environ.get(
            "UNIDRIVEVLA_LEAD_PLAN_COORD",
            "lead_forward_lateral",
        )
        self.delta_stats_mode = os.environ.get(
            "UNIDRIVEVLA_LEAD_DELTA_STATS",
            "config",
        )
        self.control_mode = os.environ.get(
            "UNIDRIVEVLA_LEAD_CONTROL",
            "b2d_pid",
        )
        if self.control_mode not in LEAD_CONTROL_MODES:
            raise ValueError(
                "Unsupported UNIDRIVEVLA_LEAD_CONTROL="
                f"{self.control_mode!r}; expected one of {LEAD_CONTROL_MODES}"
            )

        self.b2d_pidcontroller = PIDControllerVAD(
            turn_KP=1.0,
            turn_KI=0.75,
            turn_KD=0.0,
            turn_n=10,
            speed_KP=3.0,
            speed_KI=0.5,
            speed_KD=1.0,
            speed_n=10,
            brake_speed=0.4,
            brake_ratio=1.1,
            clip_delta=0.25,
            max_throttle=0.75,
            waypoint_hz=LEAD_WAYPOINT_HZ or 2.0,
            aim_dist_low=2.0,
            aim_dist_high=8.0,
            anti_windup=True,
            near_speed=LEAD_B2D_NEAR_SPEED,
        )
        self.lead_turn_controller, self.lead_speed_controller = _make_lead_waypoint_controllers()
        self.tfv6_lateral_controller = _TFV6LateralController()
        self.hybrid_route_speed_controller = _HybridRouteSpeedController()
        self.hipad_pidcontroller = HiPADDualPlanningController(temporal_hz=5.0)

        cfg = Config.fromfile(self.config_path)
        if self.ckpt_path:
            _validate_checkpoint_training_config(cfg, self.ckpt_path)
        if cfg.get("custom_imports", None):
            from mmcv.utils import import_modules_from_strings

            import_modules_from_strings(**cfg["custom_imports"])
        if hasattr(cfg, "model") and hasattr(cfg.model, "planning_head"):
            cfg.model.planning_head.dataset_type = "lead"
            if self.delta_stats_mode == "b2d_compat":
                cfg.model.planning_head.action_delta_mu = B2D_ACTION_DELTA_MU
                cfg.model.planning_head.action_delta_std = B2D_ACTION_DELTA_STD
            elif self.delta_stats_mode != "config":
                raise ValueError(
                    "Unsupported UNIDRIVEVLA_LEAD_DELTA_STATS="
                    f"{self.delta_stats_mode!r}; expected b2d_compat or config"
                )
            if hasattr(cfg.model.planning_head, "unified_decoder_cfg"):
                cfg.model.planning_head.unified_decoder_cfg.with_close_loop = True
        configured_waypoint_hz = None
        if hasattr(cfg, "model") and hasattr(cfg.model, "planning_head"):
            configured_waypoint_hz = cfg.model.planning_head.get("waypoint_hz", None)
            if configured_waypoint_hz is None:
                action_horizon = int(cfg.model.planning_head.get("action_horizon", 6))
                configured_waypoint_hz = 4.0 if action_horizon == 8 else 2.0
        self.waypoint_hz = float(
            configured_waypoint_hz if LEAD_WAYPOINT_HZ is None else LEAD_WAYPOINT_HZ
        )
        if self.waypoint_hz <= 0:
            raise ValueError(f"Resolved waypoint_hz must be positive, got {self.waypoint_hz}")
        self.b2d_pidcontroller.waypoint_hz = self.waypoint_hz
        self.hipad_pidcontroller.temporal_hz = self.waypoint_hz
        if hasattr(cfg, "data_aug_conf"):
            self.data_aug_conf = cfg.data_aug_conf
        configured_jpeg_quality = int(cfg.get("lead_jpeg_quality", 30))
        self.jpeg_quality = int(
            configured_jpeg_quality if LEAD_JPEG_QUALITY is None else LEAD_JPEG_QUALITY
        )
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError(
                f"Resolved JPEG quality must be in [0, 100], got {self.jpeg_quality}"
            )

        if hasattr(cfg, "plugin") and hasattr(cfg, "plugin_dir") and cfg.plugin:
            plugin_dir = cfg.plugin_dir
            module_dir = os.path.dirname(plugin_dir).split("/")
            module_path = module_dir[0]
            for module in module_dir[1:]:
                module_path += "." + module
            importlib.import_module(module_path)

        self.model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        if self.ckpt_path:
            _load_checkpoint(self.model, self.ckpt_path)
        self.model.cuda()
        self.model.eval()

        if LEAD_REMOVE_PROMPT_SPEED:
            planning_head = getattr(self.model, "planning_head", None)
            if planning_head is None or not hasattr(planning_head, "user_prompt_template"):
                raise RuntimeError("Prompt-speed ablation requires user_prompt_template")
            original_prompt = planning_head.user_prompt_template
            ablated_prompt = original_prompt.replace(
                "ego speed {speed:.2f} m/s, ", ""
            )
            if ablated_prompt == original_prompt:
                raise RuntimeError(
                    "Could not remove the expected ego-speed field from the prompt"
                )
            planning_head.user_prompt_template = ablated_prompt
            print(
                "[UniDriveVLALEADAgent] removed ego speed from the text prompt; "
                "numeric velocity token remains enabled",
                flush=True,
            )

        pipeline = []
        for step in cfg.inference_only_pipeline:
            if step["type"] not in [
                "LoadMultiViewImageFromFilesInCeph",
                "LoadMultiViewImageFromFiles",
                "LoadLeadMultiViewImageFromFiles",
            ]:
                pipeline.append(step)
        self.inference_only_pipeline = Compose(pipeline)

        self.lidar2img, self.lidar2cam, self.cam_intrinsic = _camera_mats()
        self.lidar2ego = _right_hand_lidar2ego()
        self.lat_ref, self.lon_ref = 42.0, 2.0
        self.metric_info = {}
        self._last_metric_flush_step = -1
        self.hero_actor = getattr(self, "hero_actor", None)
        self._hero_actor_source = None
        self.prev_control_cache = []

        control = carla.VehicleControl()
        control.steer = 0.0
        control.brake = 1.0
        control.throttle = 0.0
        self.prev_control = control
        self.previous_compass = None
        self.kalman_filter = _TFV6KalmanFilter() if LEAD_USE_KALMAN else None
        self.force_move_postprocessor = (
            _TFV6ForceMovePostProcessor() if LEAD_CREEPING else None
        )

        save_root = pathlib.Path(SAVE_PATH or os.getcwd())
        save_root.mkdir(parents=True, exist_ok=True)
        self.save_path = save_root
        self.route_id = os.environ.get("BENCHMARK_ROUTE_ID", "")
        self.lidar_record_steps = LEAD_LIDAR_ROUTE_STEPS.get(
            self.route_id, frozenset()
        )
        if (
            LEAD_RECORD_LIDAR
            and not self.lidar_record_steps
            and LEAD_LIDAR_FRAME_FREQUENCY == 0
        ):
            raise ValueError(
                "LiDAR recording is enabled but no target steps are configured for "
                f"BENCHMARK_ROUTE_ID={self.route_id!r}"
            )
        self.bev_record_steps = LEAD_BEV_ROUTE_STEPS.get(
            self.route_id, frozenset()
        )
        if (
            LEAD_RECORD_BEV_CAMERA
            and not self.bev_record_steps
            and LEAD_BEV_FRAME_FREQUENCY == 0
        ):
            raise ValueError(
                "BEV camera recording is enabled but no target steps are configured "
                f"for BENCHMARK_ROUTE_ID={self.route_id!r}"
            )
        planning_head = getattr(self.model, "planning_head", None)
        self.resolved_settings = {
            "config": str(pathlib.Path(self.config_path).resolve()),
            "checkpoint": str(pathlib.Path(self.ckpt_path).resolve()),
            "save_name": self.save_name,
            "control_mode": self.control_mode,
            "b2d_near_speed": LEAD_B2D_NEAR_SPEED,
            "use_kalman": LEAD_USE_KALMAN,
            "record_comfort": LEAD_RECORD_COMFORT,
            "record_visualization": LEAD_RECORD_VISUALIZATION,
            "record_camera": LEAD_RECORD_CAMERA,
            "camera_frame_frequency": LEAD_CAMERA_FRAME_FREQUENCY,
            "record_lidar": LEAD_RECORD_LIDAR,
            "lidar_record_steps": sorted(self.lidar_record_steps),
            "lidar_frame_frequency": LEAD_LIDAR_FRAME_FREQUENCY,
            "lidar_record_mode": (
                "interval"
                if LEAD_LIDAR_FRAME_FREQUENCY > 0
                else "selected_steps"
            ),
            "lidar_format": "compressed_npz_float32",
            "lidar_sensor_id": "lidar1",
            "record_bev_camera": LEAD_RECORD_BEV_CAMERA,
            "bev_record_steps": sorted(self.bev_record_steps),
            "bev_frame_frequency": LEAD_BEV_FRAME_FREQUENCY,
            "bev_record_mode": (
                "interval"
                if LEAD_BEV_FRAME_FREQUENCY > 0
                else "selected_steps"
            ),
            "bev_camera_format": "png_bgr8",
            "bev_camera_sensor": {
                "id": "bev",
                "type": "sensor.camera.rgb",
                "position_m": [0.0, 0.0, 60.0],
                "rotation_deg": [0.0, -90.0, 0.0],
                "width": 512,
                "height": 512,
                "fov_deg": 50.0,
                "model_input": False,
                "purpose": "qualitative_visualization_only",
            },
            "visualization_det_score_threshold": LEAD_VIS_DET_SCORE_THRESHOLD,
            "visualization_map_score_threshold": LEAD_VIS_MAP_SCORE_THRESHOLD,
            "visualization_actor_radius_m": LEAD_VIS_ACTOR_RADIUS_M,
            "creeping": LEAD_CREEPING,
            "creep_point_source": LEAD_CREEP_POINT_SOURCE,
            "stop_sign": LEAD_STOP_SIGN,
            "creep_stuck_threshold": LEAD_CREEP_STUCK_THRESHOLD,
            "creep_move_duration": LEAD_CREEP_MOVE_DURATION,
            "creep_throttle": LEAD_CREEP_THROTTLE,
            "creep_lidar_safety_only": (
                LEAD_CREEPING and LEAD_CREEP_POINT_SOURCE == "single_lidar"
            ),
            "temporal_mode": LEAD_TEMPORAL_MODE,
            "jpeg_quality": self.jpeg_quality,
            "waypoint_hz": self.waypoint_hz,
            "command_source": LEAD_COMMAND_SOURCE,
            "route_planner": LEAD_ROUTE_PLANNER_MODE,
            "route_pop_distance": LEAD_ROUTE_POP_DISTANCE,
            "user_prompt_template": getattr(planning_head, "user_prompt_template", None),
            "strict_user_prompt_fields": getattr(
                planning_head, "strict_user_prompt_fields", None
            ),
        }
        with open(self.save_path / "evaluation_settings.json", "w", encoding="utf-8") as f:
            json.dump(self.resolved_settings, f, indent=2)
        print(
            f"[UniDriveVLALEADAgent] config={self.config_path} ckpt={self.ckpt_path}",
            flush=True,
        )
        print(
            "[UniDriveVLALEADAgent] "
            f"plan_coord_mode={self.plan_coord_mode} "
            f"delta_stats_mode={self.delta_stats_mode} "
            f"control_mode={self.control_mode} "
            f"command_source={LEAD_COMMAND_SOURCE} "
            f"route_planner={LEAD_ROUTE_PLANNER_MODE} "
            f"route_pop={LEAD_ROUTE_POP_DISTANCE} "
            f"route_adaptive_pop={LEAD_ROUTE_ADAPTIVE_POP} "
            f"skip_distant_target_point={LEAD_SKIP_DISTANT_TARGET_POINT} "
            f"distant_target_point_threshold={LEAD_DISTANT_TARGET_POINT_THRESHOLD} "
            f"temporal_mode={LEAD_TEMPORAL_MODE} "
            f"jpeg_quality={self.jpeg_quality} "
            f"waypoint_hz={self.waypoint_hz} "
            f"b2d_near_speed={LEAD_B2D_NEAR_SPEED} "
            f"use_kalman={LEAD_USE_KALMAN} "
            f"creeping={LEAD_CREEPING} "
            f"creep_point_source={LEAD_CREEP_POINT_SOURCE} "
            f"stop_sign={LEAD_STOP_SIGN} "
            f"record_comfort={LEAD_RECORD_COMFORT} "
            f"record_visualization={LEAD_RECORD_VISUALIZATION} "
            f"record_camera={LEAD_RECORD_CAMERA} "
            f"camera_frame_frequency={LEAD_CAMERA_FRAME_FREQUENCY}",
            flush=True,
        )

    def init(self):
        try:
            self.lat_ref, self.lon_ref = _find_gps_ref(
                self._global_plan_world_coord,
                self._global_plan,
            )
        except Exception as exc:
            print(f"[UniDriveVLALEADAgent] route ref solve failed: {exc}", flush=True)
            self.lat_ref, self.lon_ref = 0, 0

        if LEAD_ROUTE_PLANNER_MODE == "lead":
            pop_distances = [LEAD_ROUTE_POP_DISTANCE]
            if LEAD_ROUTE_ADAPTIVE_POP and LEAD_ROUTE_ADAPTIVE_POP_DISTANCE not in pop_distances:
                pop_distances.append(LEAD_ROUTE_ADAPTIVE_POP_DISTANCE)
            self._route_planners = {}
            for pop_distance in pop_distances:
                planner = _LeadRoutePlanner(pop_distance, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
                planner.set_route(self._global_plan, True)
                self._route_planners[pop_distance] = planner
            self._route_planner = self._route_planners[LEAD_ROUTE_POP_DISTANCE]
        else:
            self._route_planners = {}
            self._route_planner = RoutePlannerVAD(
                LEAD_ROUTE_ADAPTIVE_POP_DISTANCE,
                50.0,
                lat_ref=self.lat_ref,
                lon_ref=self.lon_ref,
            )
            self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}
        self._last_metric_flush_step = -1

    def destroy(self, results=None):
        if LEAD_RECORD_VISUALIZATION:
            try:
                self._attach_gt_future_ego()
            except Exception as exc:
                print(
                    f"[UniDriveVLALEADAgent] failed to attach actual future ego GT: {exc}",
                    flush=True,
                )
        self._flush_metric_info(force=True)
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()

    def _attach_gt_future_ego(self):
        if not self.metric_info:
            return
        frames = sorted(
            ((int(step), frame) for step, frame in self.metric_info.items()),
            key=lambda item: item[0],
        )
        by_step = {step: frame for step, frame in frames}
        sample_stride = max(1, int(round(float(self.frame_rate) / float(self.waypoint_hz))))
        horizon = int(getattr(getattr(self.model, "planning_head", None), "action_horizon", 8))
        for step, frame in frames:
            location = frame.get("location")
            rotation = frame.get("rotation")
            if not isinstance(location, list) or len(location) < 2 or not isinstance(rotation, list) or len(rotation) < 3:
                continue
            yaw = math.radians(float(rotation[2]))
            forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
            right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
            origin = np.asarray(location[:2], dtype=np.float64)
            points = []
            valid = []
            future_steps = []
            for horizon_index in range(1, horizon + 1):
                future_step = step + horizon_index * sample_stride
                future = by_step.get(future_step)
                future_location = future.get("location") if isinstance(future, dict) else None
                if isinstance(future_location, list) and len(future_location) >= 2:
                    delta = np.asarray(future_location[:2], dtype=np.float64) - origin
                    points.append([float(delta @ right), float(delta @ forward)])
                    valid.append(True)
                else:
                    points.append([0.0, 0.0])
                    valid.append(False)
                future_steps.append(future_step)
            gt_future_payload = {
                "points": points,
                "valid": valid,
                "future_steps": future_steps,
                "coordinate_system": "lateral_right_forward",
                "sample_hz": float(self.waypoint_hz),
                "horizon": horizon,
                "description": "actual future ego rollout under this model, not expert planning",
            }
            frame["gt_future_ego_closed_loop"] = gt_future_payload
            # Keep an explicit planning-named alias for downstream BEV tools.
            frame["gt_planning"] = dict(gt_future_payload)

    def _flush_metric_info(self, force=False):
        if not hasattr(self, "save_path") or not getattr(self, "metric_info", None):
            return
        if not force:
            if LEAD_METRIC_FLUSH_INTERVAL <= 0:
                return
            if self.step <= 0 or self.step % LEAD_METRIC_FLUSH_INTERVAL != 0:
                return
            if self._last_metric_flush_step == self.step:
                return

        metric_path = self.save_path / "metric_info.json"
        tmp_path = self.save_path / "metric_info.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.metric_info, f)
        tmp_path.replace(metric_path)
        self._last_metric_flush_step = self.step

    def _reset_temporal_banks(self):
        decoder = getattr(getattr(self.model, "planning_head", None), "unified_decoder", None)
        if decoder is None:
            return
        for attr in (
            "det_instance_bank_list",
            "map_instance_bank_list",
            "ego_instance_bank_list",
            "plan_instance_bank_list",
        ):
            for bank in getattr(decoder, attr, []) or []:
                if hasattr(bank, "reset"):
                    bank.reset()
        decoder.run_step = 0

    @torch.no_grad()
    def _run_speed_sweep(self, inputs):
        planning_head = getattr(self.model, "planning_head", None)
        if planning_head is None or getattr(planning_head, "velocity_encoder", None) is None:
            raise RuntimeError("Speed sweep requires the TFv6-style velocity token")

        original_prompt = planning_head.user_prompt_template
        no_text_speed_prompt = original_prompt.replace(
            "ego speed {speed:.2f} m/s, ", ""
        )
        if no_text_speed_prompt == original_prompt:
            raise RuntimeError(
                "Could not remove the expected ego-speed field from the prompt"
            )

        original_seed = planning_head.inference_seed
        original_ego_status = inputs["ego_status"].clone()
        actual_speed = float(original_ego_status.reshape(-1)[0].item())
        speed_values = np.arange(0.0, 8.0 + 0.25, 0.5, dtype=np.float32)
        results = {
            "route_id": os.environ.get("BENCHMARK_ROUTE_ID"),
            "step": int(self.step),
            "fixed_noise_seed": int(LEAD_SPEED_SWEEP_SEED),
            "actual_speed": actual_speed,
            "speed_values": speed_values.tolist(),
            "arms": {},
        }

        np.savez_compressed(
            self.save_path / "speed_sweep_input.npz",
            img=inputs["img"].detach().cpu().float().numpy(),
            ego_status=original_ego_status.detach().cpu().float().numpy(),
            command=inputs["gt_ego_fut_cmd"].detach().cpu().float().numpy(),
            target_point_previous=inputs["target_point_previous"].detach().cpu().float().numpy(),
            target_point=inputs["target_point"].detach().cpu().float().numpy(),
            target_point_next=inputs["target_point_next"].detach().cpu().float().numpy(),
            timestamp=inputs["timestamp"].detach().cpu().float().numpy(),
        )

        def run_arm(name, prompt_template, *, zero_numeric_velocity=False,
                    fixed_prompt_speed=None):
            planning_head.user_prompt_template = prompt_template
            arm_results = []
            hook = None
            if zero_numeric_velocity:
                hook = planning_head.velocity_encoder.register_forward_hook(
                    lambda _module, _args, output: torch.zeros_like(output)
                )
            try:
                for speed in speed_values:
                    self._reset_temporal_banks()
                    planning_head.inference_seed = LEAD_SPEED_SWEEP_SEED
                    inputs["ego_status"] = original_ego_status.clone()
                    inputs["ego_status"].reshape(-1)[0] = float(speed)
                    if fixed_prompt_speed is not None:
                        planning_head.user_prompt_template = prompt_template.replace(
                            "{speed:.2f}", f"{fixed_prompt_speed:.2f}"
                        )

                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs = self.model(
                        img=inputs["img"],
                        img_metas=inputs["img_metas"],
                        projection_mat=inputs["projection_mat"],
                        ego_status=inputs.get("ego_status"),
                        gt_ego_fut_cmd=inputs["gt_ego_fut_cmd"],
                        target_point_previous=inputs.get("target_point_previous"),
                        target_point=inputs.get("target_point"),
                        target_point_next=inputs.get("target_point_next"),
                        image_wh=inputs["image_wh"],
                        timestamp=inputs["timestamp"],
                        rescale=True,
                        return_loss=False,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start

                    prediction = outputs[0]["img_bbox"]
                    distribution = prediction[
                        "pred_target_speed_distribution"
                    ].detach().cpu().float().reshape(-1)
                    raw_target_speed = prediction[
                        "pred_target_speed_scalar"
                    ].detach().cpu().float().reshape(-1)[0]
                    arm_results.append(
                        {
                            "speed": float(speed),
                            "stop_probability": float(distribution[0].item()),
                            "raw_target_speed": float(raw_target_speed.item()),
                            "distribution": distribution.tolist(),
                            "elapsed_seconds": float(elapsed),
                        }
                    )
                    print(
                        "[SpeedSweep] "
                        f"arm={name} speed={float(speed):.2f} "
                        f"p_stop={float(distribution[0]):.6f} "
                        f"raw_target_speed={float(raw_target_speed):.6f}",
                        flush=True,
                    )
            finally:
                if hook is not None:
                    hook.remove()
            results["arms"][name] = arm_results

        try:
            run_arm("full", original_prompt)
            run_arm("no_text_speed", no_text_speed_prompt)
            run_arm(
                "fixed_text_speed_numeric_sweep",
                original_prompt,
                fixed_prompt_speed=actual_speed,
            )
            run_arm(
                "no_numeric_velocity",
                original_prompt,
                zero_numeric_velocity=True,
            )
        finally:
            planning_head.user_prompt_template = original_prompt
            planning_head.inference_seed = original_seed
            inputs["ego_status"] = original_ego_status
            self._reset_temporal_banks()

        output_path = self.save_path / "speed_sweep.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[SpeedSweep] wrote {output_path}", flush=True)
        raise RuntimeError(f"Speed sweep complete: {output_path}")

    def gps_to_location(self, gps):
        gps = np.asarray(gps, dtype=np.float64).reshape(-1)
        if gps.shape[0] == 2:
            gps = np.array([gps[0], gps[1], 0.0], dtype=np.float64)
        return _convert_gps_to_carla(gps, self.lat_ref, self.lon_ref)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        h, w = self.data_aug_conf["H"], self.data_aug_conf["W"]
        final_h, final_w = self.data_aug_conf["final_dim"]
        resize = max(final_h / h, final_w / w)
        resize_dims = (int(w * resize), int(h * resize))
        new_w, new_h = resize_dims
        crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * new_h) - final_h
        crop_w = int(max(0, new_w - final_w) / 2)
        return {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": (crop_w, crop_h, crop_w + final_w, crop_h + final_h),
            "flip": False,
            "rotate": 0,
            "rotate_3d": 0,
        }

    def tick(self, input_data):
        self.step += 1
        if LEAD_RECORD_CAMERA and self.step % LEAD_CAMERA_FRAME_FREQUENCY == 0:
            self._save_camera_frame(input_data)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        imgs = {}
        for cam in CAMERA_IDS:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img

        gps_full = np.asarray(input_data["gps"][1], dtype=np.float64).reshape(-1)
        gps = gps_full[:2]
        speed = float(input_data["speed"][1]["speed"])
        raw_compass = float(input_data["imu"][1][-1])
        acceleration = np.asarray(input_data["imu"][1][:3], dtype=np.float64)
        angular_velocity = np.asarray(input_data["imu"][1][3:6], dtype=np.float64)
        if math.isnan(raw_compass):
            raw_compass = 0.0
            acceleration = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)

        carla_yaw = (raw_compass - np.pi / 2.0 + np.pi) % (2.0 * np.pi) - np.pi
        if self.previous_compass is not None:
            carla_yaw = float(np.unwrap([self.previous_compass, carla_yaw])[1])
        self.previous_compass = carla_yaw

        raw_pos = self.gps_to_location(gps_full)
        pos = raw_pos.copy()
        navigation_yaw = carla_yaw
        if self.kalman_filter is not None:
            filtered_state = self.kalman_filter.step(
                raw_pos, carla_yaw, speed, self.prev_control
            )
            pos[:2] = filtered_state[:2]
            navigation_yaw = float(filtered_state[2])

        creep_lidar = None
        if LEAD_CREEPING:
            if LEAD_CREEP_POINT_SOURCE == "official_tfv6":
                creep_lidar = _official_tfv6_creep_lidar(
                    input_data, self.creep_expert_config
                )
            else:
                if "lidar1" not in input_data:
                    raise RuntimeError(
                        "Creeping is enabled but lidar1 sensor data is missing"
                    )
                creep_lidar = _lidar_to_ego_for_creep(input_data["lidar1"][1])
        save_bev_frame = (
            self.step in self.bev_record_steps
            or (
                LEAD_BEV_FRAME_FREQUENCY > 0
                and self.step % LEAD_BEV_FRAME_FREQUENCY == 0
            )
        )
        if LEAD_RECORD_BEV_CAMERA and save_bev_frame:
            self._save_bev_camera_frame(input_data)
        save_lidar_frame = (
            self.step in self.lidar_record_steps
            or (
                LEAD_LIDAR_FRAME_FREQUENCY > 0
                and self.step % LEAD_LIDAR_FRAME_FREQUENCY == 0
            )
        )
        if LEAD_RECORD_LIDAR and save_lidar_frame:
            self._save_lidar_frame(input_data)
        if LEAD_ROUTE_PLANNER_MODE == "lead":
            for planner in self._route_planners.values():
                planner.run_step(pos)
            planner = self._route_planners[LEAD_ROUTE_POP_DISTANCE]
            route_points, route_commands = _dedupe_route_points_and_commands(planner.route)
            if (
                LEAD_ROUTE_ADAPTIVE_POP
                and LEAD_ROUTE_ADAPTIVE_POP_DISTANCE in self._route_planners
                and _lead_route_is_dense(route_points, pos[:2])
            ):
                planner = self._route_planners[LEAD_ROUTE_ADAPTIVE_POP_DISTANCE]
                route_points, route_commands = _dedupe_route_points_and_commands(planner.route)
            self._route_planner = planner
            if route_points:
                near_node = route_points[1][:2] if len(route_points) > 1 else route_points[0][:2]
            else:
                near_node = pos[:2]
            if route_commands:
                cmd_idx = 1 if LEAD_COMMAND_SOURCE == "next" and len(route_commands) > 1 else 0
                near_command = route_commands[cmd_idx]
            else:
                near_command = 4
        else:
            near_node, near_command = self._route_planner.run_step(pos[:2])
            route_points, _ = _dedupe_route_points_and_commands(self._route_planner.route)
            if LEAD_COMMAND_SOURCE == "current" and len(self._route_planner.route) > 0:
                near_command = self._route_planner.route[0][1]
            near_command = _command_to_int(near_command)
        target_point_previous, target_point, target_point_next = _navigation_points_to_ego(
            route_points,
            pos[:2],
            navigation_yaw,
        )
        if (
            LEAD_SKIP_DISTANT_TARGET_POINT
            and np.linalg.norm(target_point_next) > LEAD_DISTANT_TARGET_POINT_THRESHOLD
        ):
            target_point_next = target_point.copy()

        return {
            "imgs": imgs,
            "gps": gps,
            "pos": pos,
            "raw_pos": raw_pos,
            "speed": speed,
            "compass": raw_compass,
            "carla_yaw": navigation_yaw,
            "creep_lidar": creep_lidar,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "target_xy": near_node,
            "target_point_previous": target_point_previous,
            "target_point": target_point,
            "target_point_next": target_point_next,
            "reference_route_world": [
                np.asarray(point, dtype=np.float64).reshape(-1)[:3].tolist()
                for point in route_points
            ],
            "command": near_command,
        }

    def _save_camera_frame(self, input_data):
        """Persist uncompressed CARLA camera views for qualitative review."""
        try:
            frames = []
            for camera_id in CAMERA_IDS:
                raw = np.asarray(input_data[camera_id][1])
                if raw.ndim != 3 or raw.shape[2] < 3:
                    raise ValueError(f"camera {camera_id} has unexpected shape {raw.shape}")
                # CARLA supplies BGRA; drop alpha and retain BGR for PNG output.
                frames.append(np.ascontiguousarray(raw[:, :, :3]))
            height = frames[0].shape[0]
            if any(frame.shape[0] != height for frame in frames):
                raise ValueError("camera views have different heights: " + ", ".join(str(frame.shape) for frame in frames))
            concatenated = np.concatenate(frames, axis=1)
            output_dir = self.save_path / "input_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.step:05d}.png"
            temporary_path = self.save_path / f".{self.step:05d}.tmp.png"
            written = cv2.imwrite(str(temporary_path), concatenated, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            if not written:
                raise IOError(f"cv2.imwrite returned false for {temporary_path}")
            temporary_path.replace(output_path)
        except Exception as exc:
            # Image logging must not alter driving; report a write failure once.
            if not getattr(self, "_camera_save_error_reported", False):
                self._camera_save_error_reported = True
                print(f"[UniDriveVLALEADAgent] camera frame save failed: {exc}", flush=True)

    def _save_bev_camera_frame(self, input_data):
        """Persist a visualization-only top-down CARLA RGB frame."""
        temporary_path = None
        try:
            if "bev" not in input_data:
                raise KeyError("bev camera data is missing")
            raw = np.asarray(input_data["bev"][1])
            if raw.ndim != 3 or raw.shape[2] < 3:
                raise ValueError(f"bev camera has unexpected shape {raw.shape}")
            if raw.shape[:2] != (512, 512):
                raise ValueError(
                    f"bev camera has unexpected resolution {raw.shape[:2]}"
                )
            # CARLA supplies BGRA; cv2.imwrite expects BGR.
            image = np.ascontiguousarray(raw[:, :, :3])
            output_dir = self.save_path / "input_bev"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.step:05d}.png"
            temporary_path = (
                output_dir / f".{self.step:05d}.{os.getpid()}.tmp.png"
            )
            written = cv2.imwrite(
                str(temporary_path),
                image,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
            )
            if not written:
                raise IOError(f"cv2.imwrite returned false for {temporary_path}")
            temporary_path.replace(output_path)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if not getattr(self, "_bev_camera_save_error_reported", False):
                self._bev_camera_save_error_reported = True
                print(
                    f"[UniDriveVLALEADAgent] BEV camera frame save failed: {exc}",
                    flush=True,
                )

    def _save_lidar_frame(self, input_data):
        """Persist the selected lidar1 frame in sensor and physical ego coordinates."""
        temporary_path = None
        try:
            if "lidar1" not in input_data:
                raise KeyError("lidar1 sensor data is missing")
            points_sensor = np.asarray(input_data["lidar1"][1], dtype=np.float32)
            if points_sensor.ndim != 2 or points_sensor.shape[1] < 3:
                raise ValueError(
                    f"lidar1 has unexpected shape {points_sensor.shape}; expected Nx3+"
                )
            if points_sensor.shape[0] == 0:
                raise ValueError("lidar1 point cloud is empty")

            rotation = _euler_deg_to_mat_for_creep(*LEAD_LIDAR_ROTATION_DEG)
            ego_xyz = (
                rotation @ points_sensor[:, :3].astype(np.float64, copy=False).T
            ).T + LEAD_LIDAR_POSITION
            points_ego = np.concatenate(
                [ego_xyz.astype(np.float32), points_sensor[:, 3:]], axis=1
            )

            output_dir = self.save_path / "input_lidar"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.step:05d}.npz"
            temporary_path = output_dir / f".{self.step:05d}.{os.getpid()}.tmp.npz"
            extra_columns = ["intensity"] + [
                f"feature_{index}" for index in range(4, points_sensor.shape[1])
            ]
            np.savez_compressed(
                temporary_path,
                points_sensor=points_sensor,
                points_ego=points_ego,
                step=np.asarray(self.step, dtype=np.int32),
                route_id=np.asarray(self.route_id),
                sensor_id=np.asarray("lidar1"),
                sensor_position_m=LEAD_LIDAR_POSITION.astype(np.float32),
                sensor_rotation_deg=LEAD_LIDAR_ROTATION_DEG.astype(np.float32),
                sensor_coordinate_system=np.asarray("carla_lidar_local_xyz"),
                ego_coordinate_system=np.asarray("carla_ego_forward_right_up_xyz"),
                point_columns=np.asarray(
                    ["x", "y", "z"] + extra_columns[: points_sensor.shape[1] - 3]
                ),
            )
            temporary_path.replace(output_path)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if not getattr(self, "_lidar_save_error_reported", False):
                self._lidar_save_error_reported = True
                print(
                    f"[UniDriveVLALEADAgent] lidar frame save failed: {exc}",
                    flush=True,
                )

    def _make_command_onehot(self, command):
        if command < 0:
            command = 4
        command -= 1
        if command < 0 or command >= 6:
            command = 3
        command_onehot = np.zeros(6, np.float32)
        command_onehot[command] = 1.0
        return command, command_onehot

    @staticmethod
    def _visualization_array(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().float().numpy()
        return np.asarray(value)

    @staticmethod
    def _actor_category(type_id):
        type_id = str(type_id).lower()
        if type_id.startswith("vehicle."):
            return "vehicle"
        if type_id.startswith("walker."):
            return "animal" if "animal" in type_id else "pedestrian"
        if "traffic_light" in type_id:
            return "traffic_light"
        if "trafficcone" in type_id or "traffic_cone" in type_id:
            return "traffic_cone"
        if type_id.startswith("traffic.") or "sign" in type_id:
            return "traffic_sign"
        return "other"

    @staticmethod
    def _carla_vector(vector, rotation=False):
        if rotation:
            return [float(vector.roll), float(vector.pitch), float(vector.yaw)]
        return [float(vector.x), float(vector.y), float(vector.z)]

    @staticmethod
    def _is_live_actor(actor):
        if actor is None:
            return False
        try:
            return bool(getattr(actor, "is_alive", True))
        except (RuntimeError, AttributeError, TypeError):
            return False

    def _resolve_hero_actor(self):
        hero = getattr(self, "hero_actor", None)
        if self._is_live_actor(hero):
            return hero

        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        source = None
        get_hero_actor = getattr(CarlaDataProvider, "get_hero_actor", None)
        if callable(get_hero_actor):
            try:
                hero = get_hero_actor()
            except (RuntimeError, AttributeError, KeyError, TypeError):
                hero = None
            if self._is_live_actor(hero):
                source = "CarlaDataProvider.get_hero_actor()"

        if source is None:
            try:
                world = CarlaDataProvider.get_world()
                actors = world.get_actors() if world is not None else []
                for actor in actors:
                    attributes = getattr(actor, "attributes", {}) or {}
                    if (
                        self._is_live_actor(actor)
                        and hasattr(attributes, "get")
                        and attributes.get("role_name") == "hero"
                    ):
                        hero = actor
                        source = "CarlaDataProvider.get_world()"
                        break
            except (RuntimeError, AttributeError, KeyError, TypeError):
                hero = None

        if source is None:
            self.hero_actor = None
            return None

        self.hero_actor = hero
        if getattr(self, "_hero_actor_source", None) != source:
            self._hero_actor_source = source
            print(
                "[UniDriveVLALEADAgent] resolved hero actor "
                f"id={getattr(hero, 'id', 'unknown')} via {source}",
                flush=True,
            )
        return hero

    def _require_hero_actor(self, purpose):
        hero = self._resolve_hero_actor()
        if hero is None:
            raise RuntimeError(
                f"Cannot {purpose}: no live CARLA actor with role_name='hero' "
                "is available from CarlaDataProvider"
            )
        return hero

    def get_metric_info(self):
        """Return the six official Bench2Drive comfort input fields."""
        hero = self._require_hero_actor("record comfort metrics")
        transform = hero.get_transform()
        return {
            "acceleration": self._carla_vector(hero.get_acceleration()),
            "angular_velocity": self._carla_vector(hero.get_angular_velocity()),
            "forward_vector": self._carla_vector(transform.get_forward_vector()),
            "right_vector": self._carla_vector(transform.get_right_vector()),
            "location": self._carla_vector(transform.location),
            "rotation": self._carla_vector(transform.rotation, rotation=True),
        }

    def _serialize_pred_detections(self, detection):
        output = {
            "available": detection is not None,
            "coordinate_system": "sparse4d_lidar_lateral_right_forward",
            "box_layout": [
                "lateral_right", "forward", "z", "width", "length", "height",
                "yaw", "vx", "vy", "vz",
            ],
            "score_threshold": float(LEAD_VIS_DET_SCORE_THRESHOLD),
            "boxes": [],
        }
        if not isinstance(detection, dict):
            return output
        try:
            raw_boxes = self._visualization_array(detection.get("boxes_3d", []))
            if raw_boxes.size == 0:
                return output
            boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])
            scores = self._visualization_array(detection.get("scores_3d", [])).reshape(-1)
            labels = self._visualization_array(detection.get("labels_3d", [])).reshape(-1)
        except (TypeError, ValueError, IndexError):
            return output
        instance_ids = detection.get("instance_ids")
        if instance_ids is not None:
            instance_ids = self._visualization_array(instance_ids).reshape(-1)
        count = min(len(boxes), len(scores), len(labels))
        valid_indices = [
            index for index in range(count)
            if np.isfinite(float(scores[index])) and np.isfinite(boxes[index]).all()
        ]
        order = sorted(valid_indices, key=lambda index: float(scores[index]), reverse=True)
        for index in order:
            score = float(scores[index])
            box = boxes[index].astype(np.float64, copy=False)
            if score < LEAD_VIS_DET_SCORE_THRESHOLD or not np.isfinite(box).all():
                continue
            if box.shape[0] >= 2 and float(np.linalg.norm(box[:2])) > LEAD_VIS_ACTOR_RADIUS_M:
                continue
            label_id = int(labels[index])
            item = {
                "box": box.tolist(),
                "score": score,
                "label_id": label_id,
                "label": LEAD_DET_CLASSES[label_id] if 0 <= label_id < len(LEAD_DET_CLASSES) else "unknown",
            }
            if instance_ids is not None and index < len(instance_ids):
                item["instance_id"] = int(instance_ids[index])
            output["boxes"].append(item)
            if len(output["boxes"]) >= LEAD_VIS_MAX_PRED_BOXES:
                break
        return output

    def _serialize_pred_map(self, map_prediction):
        output = {
            "available": map_prediction is not None,
            "coordinate_system": "sparse4d_lidar_lateral_right_forward",
            "score_threshold": float(LEAD_VIS_MAP_SCORE_THRESHOLD),
            "vectors": [],
        }
        if not isinstance(map_prediction, dict):
            return output
        vectors = map_prediction.get("vectors", [])
        try:
            scores = self._visualization_array(map_prediction.get("scores", [])).reshape(-1)
            labels = self._visualization_array(map_prediction.get("labels", [])).reshape(-1)
        except (TypeError, ValueError):
            return output
        count = min(len(vectors), len(scores), len(labels))
        valid_indices = [
            index for index in range(count)
            if np.isfinite(float(scores[index]))
        ]
        order = sorted(valid_indices, key=lambda index: float(scores[index]), reverse=True)
        for index in order:
            score = float(scores[index])
            if score < LEAD_VIS_MAP_SCORE_THRESHOLD:
                continue
            try:
                vector = self._visualization_array(vectors[index]).astype(np.float64, copy=False).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(vector).all():
                continue
            label_id = int(labels[index])
            output["vectors"].append(
                {
                    "points": vector.tolist(),
                    "score": score,
                    "label_id": label_id,
                    "label": LEAD_MAP_CLASSES[label_id] if 0 <= label_id < len(LEAD_MAP_CLASSES) else "unknown",
                }
            )
            if len(output["vectors"]) >= LEAD_VIS_MAX_MAP_VECTORS:
                break
        return output

    def _serialize_gt_actors(self):
        hero = self._require_hero_actor("serialize GT actors")
        ego_transform = hero.get_transform()
        ego_location = ego_transform.location
        yaw = math.radians(float(ego_transform.rotation.yaw))
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        output = []
        for actor in hero.get_world().get_actors():
            try:
                if actor.id == hero.id or not getattr(actor, "is_alive", True):
                    continue
                type_id = str(actor.type_id)
                if type_id.startswith("sensor.") or type_id.startswith("controller.") or type_id == "spectator":
                    continue
                transform = actor.get_transform()
                delta = np.array(
                    [transform.location.x - ego_location.x, transform.location.y - ego_location.y],
                    dtype=np.float64,
                )
                if float(np.linalg.norm(delta)) > LEAD_VIS_ACTOR_RADIUS_M:
                    continue
                bbox = actor.bounding_box
                world_vertices = bbox.get_world_vertices(transform)
                if not world_vertices:
                    continue
                world_corners = np.array(
                    [[vertex.x, vertex.y, vertex.z] for vertex in world_vertices],
                    dtype=np.float64,
                )
                corner_delta = world_corners[:, :2] - np.array(
                    [ego_location.x, ego_location.y], dtype=np.float64
                )
                ego_corners = np.stack(
                    [corner_delta @ right, corner_delta @ forward], axis=-1
                )
                ego_center = ego_corners.mean(axis=0)
                ego_yaw = math.degrees(
                    math.atan2(
                        math.sin(math.radians(float(transform.rotation.yaw)) - yaw),
                        math.cos(math.radians(float(transform.rotation.yaw)) - yaw),
                    )
                )
                attributes = getattr(actor, "attributes", {}) or {}
                if not hasattr(attributes, "get"):
                    attributes = {}

                def _optional_carla_vector(method_name):
                    method = getattr(actor, method_name, None)
                    if method is None:
                        return [0.0, 0.0, 0.0]
                    try:
                        return self._carla_vector(method())
                    except (RuntimeError, AttributeError, TypeError, ValueError):
                        return [0.0, 0.0, 0.0]

                output.append(
                    {
                        "actor_id": int(actor.id),
                        "type_id": type_id,
                        "category": self._actor_category(type_id),
                        "role_name": str(attributes.get("role_name", "")),
                        "semantic_tags": [int(value) for value in getattr(actor, "semantic_tags", [])],
                        "location": self._carla_vector(transform.location),
                        "rotation": self._carla_vector(transform.rotation, rotation=True),
                        "velocity": _optional_carla_vector("get_velocity"),
                        "acceleration": _optional_carla_vector("get_acceleration"),
                        "angular_velocity": _optional_carla_vector("get_angular_velocity"),
                        "bbox_location": self._carla_vector(bbox.location),
                        "bbox_rotation": self._carla_vector(bbox.rotation, rotation=True),
                        "bbox_extent": self._carla_vector(bbox.extent),
                        "bbox_ego_center_lateral_right_forward": ego_center.tolist(),
                        "bbox_ego_yaw_deg": float(ego_yaw),
                        "bbox_world_corners": world_corners.tolist(),
                        "bbox_ego_corners_lateral_right_forward": ego_corners.tolist(),
                    }
                )
            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue
        output.sort(key=lambda item: item["actor_id"])
        return output

    def _visualization_payload(self, img_bbox, tick_data, timestamp):
        try:
            timestamp_value = float(timestamp)
        except (TypeError, ValueError):
            timestamp_value = None
        scenario_metadata = {
            "route_id": os.environ.get("BENCHMARK_ROUTE_ID"),
            "route_file": os.environ.get("ROUTES"),
            "town": None,
            "weather": {},
        }
        hero = self._require_hero_actor("record visualization metadata")
        try:
            world = hero.get_world()
            scenario_metadata["town"] = str(world.get_map().name)
            weather = world.get_weather()
            for field in (
                "cloudiness", "precipitation", "precipitation_deposits",
                "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
                "fog_density", "fog_distance", "wetness", "scattering_intensity",
                "dust_storm",
            ):
                if hasattr(weather, field):
                    scenario_metadata["weather"][field] = float(getattr(weather, field))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        reference_route = tick_data.get("reference_route_world", [])
        pred_route = img_bbox.get("pred_route")
        if pred_route is not None:
            try:
                pred_route = self._visualization_array(pred_route).astype(np.float64, copy=False).reshape(-1, 2).tolist()
            except (TypeError, ValueError):
                pred_route = None
        return {
            "visualization_schema_version": 1,
            "simulation_timestamp": timestamp_value,
            "scenario_metadata": scenario_metadata,
            "ego_coordinate_system": "lateral_right_forward",
            "model_coordinate_system": "sparse4d_lidar_lateral_right_forward",
            "reference_route_world": reference_route,
            "pred_route": {
                "points": pred_route,
                "coordinate_system": "lead_forward_lateral",
            },
            "gt_map": {
                "town": "Town13",
                "h5": str(pathlib.Path(F2D_MAP_ROOT) / "Town13.h5"),
                "opendrive": str(
                    pathlib.Path(F2D_CARLA_ROOT)
                    / "CarlaUE4/Content/Carla/Maps/Town13/OpenDrive/Town13.xodr"
                ),
            },
            "pred_detections": self._serialize_pred_detections(
                img_bbox.get("_visualization_det")
            ),
            "pred_map": self._serialize_pred_map(
                img_bbox.get("_visualization_map")
            ),
            "gt_actors": self._serialize_gt_actors(),
        }

    def _metric_payload(self, control, pred_traj, command, speed, raw_pred_traj=None):
        return {
            "agent": "unidrivevla_lead",
            "step": int(self.step),
            "speed": float(speed),
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "command": int(command),
            "command_text": LEAD_COMMAND_MAP.get(int(command), "LANE FOLLOW"),
            "plan_temp": pred_traj.tolist() if isinstance(pred_traj, np.ndarray) else pred_traj,
            "plan_coord_mode": self.plan_coord_mode,
            "route_planner_mode": LEAD_ROUTE_PLANNER_MODE,
            "route_pop_distance": float(LEAD_ROUTE_POP_DISTANCE),
            "route_adaptive_pop": bool(LEAD_ROUTE_ADAPTIVE_POP),
            "temporal_mode": LEAD_TEMPORAL_MODE,
            "use_kalman": bool(LEAD_USE_KALMAN),
            "creeping_enabled": bool(LEAD_CREEPING),
            "stop_sign_enabled": bool(LEAD_STOP_SIGN),
            "waypoint_hz": float(self.waypoint_hz),
            "b2d_near_speed": bool(LEAD_B2D_NEAR_SPEED),
            "plan_temp_raw": raw_pred_traj.tolist() if isinstance(raw_pred_traj, np.ndarray) else raw_pred_traj,
            "plan_temp_coordinate_system": "lateral_right_forward",
            "plan_temp_raw_coordinate_system": self.plan_coord_mode,
            "pred_planning": {
                "points": pred_traj.tolist() if isinstance(pred_traj, np.ndarray) else pred_traj,
                "coordinate_system": "lateral_right_forward",
                "raw_points": raw_pred_traj.tolist() if isinstance(raw_pred_traj, np.ndarray) else raw_pred_traj,
                "raw_coordinate_system": self.plan_coord_mode,
            },
        }

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        first_frame = not self.initialized
        if first_frame:
            self.init()
        tick_data = self.tick(input_data)

        # Match LEAD SensorAgent: CARLA cannot move the ego vehicle on the
        # first frame, so apply a full brake before the first model forward.
        if first_frame:
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.prev_control = control
            self.prev_control_cache.append(control)
            return control

        inputs = {
            "img": [],
            "folder": "",
            "scene_token": "",
            "lidar2img": [],
            "lidar2cam": [],
            "cam_intrinsic": [],
            "frame_idx": 0,
            "timestamp": self.step / self.frame_rate,
            "aug_config": self.get_augmentation(),
        }
        for idx, cam in enumerate(CAMERA_IDS):
            img = tick_data["imgs"][cam]
            if self.use_bgr_img:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            inputs["img"].append(img.astype(np.float32))
            inputs["lidar2img"].append(self.lidar2img[idx])
            inputs["lidar2cam"].append(self.lidar2cam[idx])
            inputs["cam_intrinsic"].append(self.cam_intrinsic[idx])

        ego_theta = -float(tick_data["carla_yaw"])
        ego_pos = [tick_data["pos"][0], -tick_data["pos"][1]]

        ego_status = np.zeros(6, dtype=np.float32)
        ego_status[0] = tick_data["speed"]
        # LEAD training infos store raw IMU x/y values from lead.common.base_agent.
        # Do not apply the B2D right-handed y-axis flip here.
        ego_status[1:3] = tick_data["acceleration"][:2]
        ego_status[3:5] = tick_data["angular_velocity"][:2]
        ego_status[5] = self.pid_metadata["steer"] if hasattr(self, "pid_metadata") else 0.0
        inputs["ego_status"] = ego_status

        command, command_onehot = self._make_command_onehot(tick_data["command"])
        inputs["command"] = command
        inputs["gt_ego_fut_cmd"] = command_onehot
        inputs["target_point_previous"] = tick_data["target_point_previous"]
        inputs["target_point"] = tick_data["target_point"]
        inputs["target_point_next"] = tick_data["target_point_next"]

        ego2world = np.eye(4, dtype=np.float32)
        ego2world[:3, :3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[:2, 3] = ego_pos
        inputs["lidar2global"] = (ego2world @ self.lidar2ego).astype(np.float32)

        final_h, final_w = self.data_aug_conf["final_dim"]
        inputs["img_shape"] = [(final_h, final_w, 3) for _ in CAMERA_IDS]
        inputs["ori_shape"] = [(final_h, final_w, 3) for _ in CAMERA_IDS]
        inputs["pad_shape"] = [(final_h, final_w, 3) for _ in CAMERA_IDS]

        inputs = self.inference_only_pipeline(inputs)
        inputs = mm_collate_to_batch_form([inputs], samples_per_gpu=1)
        inputs = _unwrap_data_containers(inputs, self.device)

        if LEAD_SPEED_SWEEP:
            self._run_speed_sweep(inputs)

        if LEAD_TEMPORAL_MODE == "reset_each_step":
            self._reset_temporal_banks()

        outputs = self.model(
            img=inputs["img"],
            img_metas=inputs["img_metas"],
            projection_mat=inputs["projection_mat"],
            ego_status=inputs.get("ego_status"),
            gt_ego_fut_cmd=inputs["gt_ego_fut_cmd"],
            target_point_previous=inputs.get("target_point_previous"),
            target_point=inputs.get("target_point"),
            target_point_next=inputs.get("target_point_next"),
            image_wh=inputs["image_wh"],
            timestamp=inputs["timestamp"],
            rescale=True,
            return_loss=False,
        )

        img_bbox = outputs[0]["img_bbox"]
        if "final_planning" in img_bbox:
            pred_traj = img_bbox["final_planning"].detach().cpu().numpy()
        elif "plan_speed_5hz" in img_bbox:
            pred_traj = img_bbox["plan_speed_5hz"].detach().cpu().numpy()
        else:
            raise KeyError(f"planning output missing, got keys: {list(img_bbox.keys())}")

        if pred_traj.ndim == 3:
            pred_traj = pred_traj[0]

        pred_route = img_bbox.get("pred_route")
        if torch.is_tensor(pred_route):
            pred_route = pred_route.detach().cpu().float().numpy()
        if pred_route is not None:
            pred_route = np.asarray(pred_route, dtype=np.float32).reshape(-1, 2)

        pred_target_speed = img_bbox.get("pred_target_speed_scalar")
        if torch.is_tensor(pred_target_speed):
            pred_target_speed = float(
                pred_target_speed.detach().cpu().float().reshape(-1)[0]
            )
        elif pred_target_speed is not None:
            pred_target_speed = float(np.asarray(pred_target_speed).reshape(-1)[0])

        pred_target_speed_distribution = img_bbox.get(
            "pred_target_speed_distribution"
        )
        if torch.is_tensor(pred_target_speed_distribution):
            pred_target_speed_distribution = (
                pred_target_speed_distribution.detach().cpu().float().numpy()
            )
        if pred_target_speed_distribution is not None:
            pred_target_speed_distribution = np.asarray(
                pred_target_speed_distribution, dtype=np.float32
            ).reshape(-1)

        plan_speed_5hz = img_bbox.get("plan_speed_5hz")
        plan_spat_2m = img_bbox.get("plan_spat_2m")
        if torch.is_tensor(plan_speed_5hz):
            plan_speed_5hz = plan_speed_5hz.detach().cpu().float().numpy()
        if torch.is_tensor(plan_spat_2m):
            plan_spat_2m = plan_spat_2m.detach().cpu().float().numpy()
        if plan_speed_5hz is not None:
            plan_speed_5hz = np.asarray(
                plan_speed_5hz, dtype=np.float32
            ).reshape(-1, 2)
        if plan_spat_2m is not None:
            plan_spat_2m = np.asarray(
                plan_spat_2m, dtype=np.float32
            ).reshape(-1, 2)

        raw_pred_traj = pred_traj.copy()
        pred_traj = _convert_plan_for_pid(pred_traj, self.plan_coord_mode)
        dummy_target = pred_traj[-1]
        b2d_speed = np.float64(tick_data["speed"])

        b2d_steer, b2d_throttle, b2d_brake, b2d_metadata = self.b2d_pidcontroller.control_pid(
            pred_traj,
            b2d_speed,
            dummy_target,
        )
        lead_steer, lead_throttle, lead_brake, lead_metadata = _control_with_lead_waypoints(
            raw_pred_traj,
            tick_data["speed"],
            self.lead_turn_controller,
            self.lead_speed_controller,
            self.waypoint_hz,
        )
        tfv6_metadata = {"tfv6_route_speed_available": False}
        tfv6_control = None
        if pred_route is not None and pred_target_speed is not None:
            (
                tfv6_steer,
                tfv6_throttle,
                tfv6_brake,
                tfv6_metadata,
            ) = _control_with_tfv6_route_speed(
                pred_route,
                pred_target_speed,
                pred_target_speed_distribution,
                tick_data["speed"],
                self.tfv6_lateral_controller,
            )
            tfv6_control = (tfv6_steer, tfv6_throttle, tfv6_brake)
            tfv6_metadata["tfv6_route_speed_available"] = True
            if pred_target_speed_distribution is not None:
                tfv6_metadata["tfv6_target_speed_distribution"] = (
                    pred_target_speed_distribution.tolist()
                )
        elif self.control_mode == "tfv6_route_speed":
            raise KeyError(
                "tfv6_route_speed control requires pred_route and "
                "pred_target_speed_scalar outputs"
            )

        hybrid_metadata = {"hybrid_route_speed_available": False}
        hybrid_control = None
        if (
            pred_route is not None
            and pred_target_speed is not None
            and pred_target_speed_distribution is not None
        ):
            (
                hybrid_steer,
                hybrid_throttle,
                hybrid_brake,
                hybrid_metadata,
            ) = self.hybrid_route_speed_controller.step(
                pred_route,
                pred_target_speed,
                pred_target_speed_distribution,
                tick_data["speed"],
                b2d_metadata["desired_speed"],
                b2d_steer,
            )
            hybrid_control = (hybrid_steer, hybrid_throttle, hybrid_brake)
            hybrid_metadata["hybrid_route_speed_available"] = True
        elif self.control_mode == "hybrid_route_speed":
            raise KeyError(
                "hybrid_route_speed control requires pred_route, "
                "pred_target_speed_scalar, and pred_target_speed_distribution"
            )

        hipad_metadata = {"hipad_dual_available": False}
        hipad_control = None
        if plan_speed_5hz is not None and plan_spat_2m is not None:
            hipad_temporal_pid = _convert_plan_for_pid(
                plan_speed_5hz, self.plan_coord_mode
            )
            hipad_spatial_pid = _convert_plan_for_pid(
                plan_spat_2m, self.plan_coord_mode
            )
            (
                hipad_steer,
                hipad_throttle,
                hipad_brake,
                hipad_metadata,
            ) = self.hipad_pidcontroller.control(
                hipad_temporal_pid,
                hipad_spatial_pid,
                tick_data["speed"],
            )
            hipad_control = (hipad_steer, hipad_throttle, hipad_brake)
            hipad_metadata["hipad_dual_available"] = True
        elif self.control_mode == "hipad_dual":
            raise KeyError(
                "hipad_dual control requires plan_speed_5hz and plan_spat_2m"
            )

        if self.control_mode == "b2d_pid":
            steer, throttle, brake = b2d_steer, b2d_throttle, b2d_brake
            metadata = dict(b2d_metadata)
        elif self.control_mode == "lead_waypoint":
            steer, throttle, brake = lead_steer, lead_throttle, lead_brake
            metadata = {
                "speed": lead_metadata["lead_waypoint_speed"],
                "steer": lead_metadata["lead_waypoint_steer"],
                "throttle": lead_metadata["lead_waypoint_throttle"],
                "brake": lead_metadata["lead_waypoint_brake"],
                "desired_speed": lead_metadata["lead_waypoint_desired_speed"],
                "angle": lead_metadata["lead_waypoint_angle"],
                "aim": lead_metadata["lead_waypoint_aim"],
                "delta": lead_metadata["lead_waypoint_delta_speed"],
                **lead_metadata,
            }
        elif self.control_mode == "tfv6_route_speed":
            steer, throttle, brake = tfv6_control
            metadata = {
                "speed": float(tick_data["speed"]),
                "steer": float(steer),
                "throttle": float(throttle),
                "brake": float(brake),
                "desired_speed": float(
                    tfv6_metadata["tfv6_route_speed_target_speed"]
                ),
                "route": pred_route.tolist(),
                **tfv6_metadata,
            }
        elif self.control_mode == "hipad_dual":
            steer, throttle, brake = hipad_control
            metadata = dict(hipad_metadata)
        else:
            steer, throttle, brake = hybrid_control
            metadata = {
                "speed": float(tick_data["speed"]),
                "steer": float(steer),
                "throttle": float(throttle),
                "brake": float(brake),
                "desired_speed": float(
                    hybrid_metadata["hybrid_route_speed_target_speed"]
                ),
                "route": pred_route.tolist(),
                **hybrid_metadata,
            }

        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0

        creep_metadata = {
            "creeping_enabled": bool(LEAD_CREEPING),
            "creeping_applied": False,
            "creeping_safety_blocked": False,
            "creeping_safety_point_count": 0,
            "creeping_stuck_ticks": 0,
            "creeping_force_move_remaining": 0,
        }
        if (
            self.force_move_postprocessor is not None
            and self.control_mode != "hipad_dual"
        ):
            throttle, brake, creep_metadata = self.force_move_postprocessor.adjust(
                tick_data["speed"], throttle, brake, tick_data["creep_lidar"]
            )

        control = carla.VehicleControl()
        control.steer = np.clip(float(steer), -1, 1)
        control.throttle = np.clip(float(throttle), 0, 1.0)
        control.brake = np.clip(float(brake), 0, 1)

        self.pid_metadata = dict(metadata)
        self.pid_metadata["control_mode"] = self.control_mode
        self.pid_metadata.update({f"b2d_pid_{key}": value for key, value in b2d_metadata.items()})
        self.pid_metadata.update(lead_metadata)
        self.pid_metadata.update(tfv6_metadata)
        self.pid_metadata.update(hybrid_metadata)
        self.pid_metadata.update(hipad_metadata)
        self.pid_metadata.update(creep_metadata)
        self.pid_metadata.update(
            self._metric_payload(
                control,
                pred_traj,
                command,
                tick_data["speed"],
                raw_pred_traj=raw_pred_traj,
            )
        )
        if LEAD_RECORD_COMFORT or LEAD_RECORD_VISUALIZATION:
            comfort_metrics = self.get_metric_info()
            required_comfort_fields = {
                "acceleration",
                "angular_velocity",
                "forward_vector",
                "right_vector",
                "location",
                "rotation",
            }
            missing_comfort_fields = required_comfort_fields.difference(comfort_metrics)
            if missing_comfort_fields:
                raise RuntimeError(
                    "Bench2Drive comfort metric payload is missing fields: "
                    + ", ".join(sorted(missing_comfort_fields))
                )

            metric_entry = dict(self.pid_metadata)
            metric_entry.update(comfort_metrics)
            if LEAD_RECORD_VISUALIZATION:
                metric_entry.update(
                    self._visualization_payload(img_bbox, tick_data, timestamp)
                )
            self.metric_info[self.step] = metric_entry
            self._flush_metric_info()

        self.prev_control = control
        if len(self.prev_control_cache) == 10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control
