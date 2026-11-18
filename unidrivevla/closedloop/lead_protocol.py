"""Torch-free JSON/HTTP boundary for the official LEAD CARLA agent and model server."""

import base64
import ipaddress
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

LEAD_CAMERA_VIEWS = ("front_left", "front", "front_right", "back_right", "back", "back_left")
LEAD_PROTOCOL = "unidrivevla.lead.v1"
MAX_REQUEST_BYTES = 128 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_TOTAL_IMAGE_PIXELS = 24 * 1024 * 1024
LEAD_CAMERA_COUNT = 6
LEAD_HORIZON = 8
LEAD_POSE_INTERVAL_S = 0.25
LEAD_HISTORY_SHAPE = (4, 3)
LEAD_INPUT_HZ = 20
LEAD_TEMPORAL_RING_SIZE = int(LEAD_INPUT_HZ * LEAD_POSE_INTERVAL_S)
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if (array.shape != shape or not np.isfinite(array).all()
            or np.any(np.abs(array) > _FLOAT32_MAX)):
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} array")
    return array


def _rigid(value, name):
    matrix = _matrix(value, (4, 4), name)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or np.linalg.det(rotation) < 0.99):
        raise ValueError(f"{name} is not a rigid transform")
    return matrix


def _json_numbers(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.isinf(array).any() or np.any(np.abs(array[np.isfinite(array)]) > _FLOAT32_MAX):
        raise ValueError("LEAD status contains infinite or out-of-range values")
    return [float(value) if math.isfinite(float(value)) else None for value in array]


def _status_vector(features, name, required_indices):
    array = np.asarray(features[name], dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if any(not math.isfinite(float(array[index])) for index in required_indices):
        raise ValueError(f"observed {name} fields {required_indices} must be finite")
    return _json_numbers(array)


def _history_array(value):
    try:
        history = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("LEAD history_trajectory must be a numeric 4x3 array") from exc
    if (history.shape != LEAD_HISTORY_SHAPE or not np.isfinite(history).all()
            or np.any(np.abs(history) > _FLOAT32_MAX)):
        raise ValueError("LEAD history_trajectory must be a finite float32-safe 4x3 array")
    return history


def encode_request(features, ego2global, timestamp, command="unknown"):
    """Encode only current localized/sensor features; future labels never enter this protocol."""
    if command != "unknown":
        raise ValueError("the LEAD profile command is fixed to 'unknown'")
    timestamp = float(timestamp)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite nonnegative seconds")
    pose = _rigid(ego2global, "ego2global")
    history = _history_array(features.get("history_trajectory"))
    velocity = _status_vector(features, "ego_velocity", (0, 1))
    acceleration = _status_vector(features, "ego_acceleration", ())
    angular_velocity = _status_vector(features, "ego_angular_velocity", (2,))
    cameras = features.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(LEAD_CAMERA_VIEWS):
        raise ValueError(f"LEAD requires calibrated cameras {list(LEAD_CAMERA_VIEWS)}")

    packed = {}
    total_pixels = 0
    for view in LEAD_CAMERA_VIEWS:
        camera = cameras[view]
        image = np.asarray(camera["image"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"LEAD {view} image must be RGB uint8 HxWx3")
        height, width = image.shape[:2]
        if height < 1 or width < 1 or height * width > MAX_IMAGE_PIXELS:
            raise ValueError(f"LEAD {view} image dimensions exceed the protocol bound")
        total_pixels += height * width
        if total_pixels > MAX_TOTAL_IMAGE_PIXELS:
            raise ValueError("LEAD total camera pixels exceed the protocol bound")
        intrinsic = _matrix(camera["intrinsic"], (3, 3), f"{view} intrinsic")
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-5)):
            raise ValueError(f"LEAD {view} intrinsic is not a calibrated pinhole matrix")
        sensor2ego = _rigid(camera["sensor2ego"], f"{view} sensor2ego")
        packed[view] = {
            "shape": [height, width, 3],
            "rgb": base64.b64encode(np.ascontiguousarray(image).tobytes()).decode("ascii"),
            "intrinsic": intrinsic.tolist(),
            "sensor2ego": sensor2ego.tolist(),
        }
    payload = {
        "protocol": LEAD_PROTOCOL,
        "pose_interval_s": LEAD_POSE_INTERVAL_S,
        "timestamp": timestamp,
        "command": command,
        "ego2global": pose.tolist(),
        "history_trajectory": history.tolist(),
        "ego_velocity": velocity,
        "ego_acceleration": acceleration,
        "ego_angular_velocity": angular_velocity,
        "ego_steer": _json_numbers([features.get("ego_steer", 0.0)])[0],
        "cameras": packed,
    }
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError("LEAD JSON request exceeds 128 MiB")
    return body


def _vector(value, name, required_indices):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers or nulls")
    decoded = []
    for item in value:
        if item is None:
            decoded.append(float("nan"))
            continue
        number = float(item)
        if not math.isfinite(number) or abs(number) > _FLOAT32_MAX:
            raise ValueError(f"{name} contains an infinite or out-of-range value")
        decoded.append(number)
    if any(not math.isfinite(decoded[index]) for index in required_indices):
        raise ValueError(f"observed {name} fields {required_indices} must be finite")
    return np.asarray(decoded, dtype=np.float32)


def decode_request(payload):
    """Strictly validate the server-side LEAD schema and decode lossless RGB bytes."""
    if not isinstance(payload, dict) or set(payload) != {
        "protocol", "pose_interval_s", "timestamp", "command", "ego2global", "history_trajectory",
        "ego_velocity", "ego_acceleration", "ego_angular_velocity", "ego_steer", "cameras",
    }:
        raise ValueError("LEAD request has missing or unsupported fields")
    if (payload["protocol"] != LEAD_PROTOCOL or payload["command"] != "unknown"
            or float(payload["pose_interval_s"]) != LEAD_POSE_INTERVAL_S):
        raise ValueError("unsupported LEAD protocol or command")
    timestamp = float(payload["timestamp"])
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite nonnegative seconds")
    pose = _rigid(payload["ego2global"], "ego2global")
    raw_cameras = payload["cameras"]
    if not isinstance(raw_cameras, dict) or set(raw_cameras) != set(LEAD_CAMERA_VIEWS):
        raise ValueError("LEAD request must contain all six calibrated camera views")

    cameras = {}
    total_pixels = 0
    for view in LEAD_CAMERA_VIEWS:
        item = raw_cameras[view]
        if not isinstance(item, dict) or set(item) != {"shape", "rgb", "intrinsic", "sensor2ego"}:
            raise ValueError(f"LEAD {view} camera has missing or unsupported fields")
        shape = item["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
                or any(type(dim) is not int or dim < 1 for dim in shape[:2])
                or shape[0] * shape[1] > MAX_IMAGE_PIXELS):
            raise ValueError(f"LEAD {view} image shape is invalid or oversized")
        total_pixels += shape[0] * shape[1]
        if total_pixels > MAX_TOTAL_IMAGE_PIXELS:
            raise ValueError("LEAD total camera pixels exceed the protocol bound")
        encoded = item["rgb"]
        if not isinstance(encoded, str) or len(encoded) > ((shape[0] * shape[1] * 3 + 2) // 3) * 4 + 8:
            raise ValueError(f"LEAD {view} image encoding exceeds its shape bound")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"LEAD {view} image is not valid base64") from exc
        if len(raw) != shape[0] * shape[1] * 3:
            raise ValueError(f"LEAD {view} image byte count does not match its shape")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        intrinsic = _matrix(item["intrinsic"], (3, 3), f"{view} intrinsic")
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-5)):
            raise ValueError(f"LEAD {view} intrinsic is invalid")
        cameras[view] = {
            "image": image,
            "intrinsic": intrinsic,
            "sensor2ego": _rigid(item["sensor2ego"], f"{view} sensor2ego"),
            "size": (shape[1], shape[0]),
        }
    steer = payload["ego_steer"]
    steer = float("nan") if steer is None else float(steer)
    if not math.isfinite(steer) and not math.isnan(steer):
        raise ValueError("ego_steer must be finite or null")
    return {
        "cameras": cameras,
        "ego2global": pose,
        "history_trajectory": _history_array(payload["history_trajectory"]).astype(np.float32),
        "timestamp": timestamp,
        "command": "unknown",
        "ego_velocity": _vector(payload["ego_velocity"], "ego_velocity", (0, 1)),
        "ego_acceleration": _vector(payload["ego_acceleration"], "ego_acceleration", ()),
        "ego_angular_velocity": _vector(payload["ego_angular_velocity"], "ego_angular_velocity", (2,)),
        "ego_steer": steer,
    }


def encode_trajectory(trajectory):
    plan = np.asarray(trajectory, dtype=np.float64)
    if (plan.shape != (LEAD_HORIZON, 2) or not np.isfinite(plan).all()
            or np.any(np.abs(plan) > _FLOAT32_MAX)):
        raise ValueError(f"LEAD policy must return a finite {LEAD_HORIZON}x2 trajectory")
    return json.dumps({"trajectory": plan.tolist(), "pose_interval_s": LEAD_POSE_INTERVAL_S},
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_trajectory(payload):
    if (not isinstance(payload, dict) or set(payload) != {"trajectory", "pose_interval_s"}
            or float(payload["pose_interval_s"]) != LEAD_POSE_INTERVAL_S):
        raise ValueError("LEAD inference response has missing or unsupported fields")
    plan = np.asarray(payload["trajectory"], dtype=np.float64)
    if (plan.shape != (LEAD_HORIZON, 2) or not np.isfinite(plan).all()
            or np.any(np.abs(plan) > _FLOAT32_MAX)):
        raise ValueError(f"LEAD server returned an invalid {LEAD_HORIZON}x2 trajectory")
    return plan


class LeadHTTPClient:
    """Single-client synchronous HTTP transport; inference requests are never retried."""

    def __init__(self, server_url, timeout=60.0):
        parts = urlsplit(str(server_url))
        if (parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password
                or parts.path not in {"", "/"} or parts.query or parts.fragment):
            raise ValueError("LEAD server_url must be an http(s) origin without credentials, path, query, or fragment")
        host = parts.hostname or ""
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host.lower() == "localhost"
        if not is_loopback and parts.scheme != "https":
            raise ValueError("non-loopback LEAD inference servers require HTTPS")
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("LEAD HTTP timeout must be positive and finite")

    def _post(self, path, body=b""):
        request = Request(self.origin + path, data=body, method="POST",
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_REQUEST_BYTES + 1)
                if len(raw) > MAX_REQUEST_BYTES:
                    raise ValueError("LEAD server response exceeds 128 MiB")
                if response.status != 200:
                    raise RuntimeError(f"LEAD server returned HTTP {response.status}")
                if response.headers.get_content_type().lower() != "application/json":
                    raise ValueError("LEAD server response Content-Type must be application/json")
                return json.loads(raw, parse_constant=_reject_json_constant)
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"LEAD server returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"could not reach LEAD inference server: {exc.reason}") from exc

    def reset(self):
        response = self._post("/reset")
        if response is not True:
            raise RuntimeError("LEAD server did not confirm episode reset")

    def predict(self, features, ego2global, timestamp, command="unknown"):
        body = encode_request(features, ego2global, timestamp, command)
        return decode_trajectory(self._post("/infer", body))


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")
