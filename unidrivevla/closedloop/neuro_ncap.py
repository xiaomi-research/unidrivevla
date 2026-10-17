"""NeuroNCAP Model Node protocol and its calibrated nuScenes-to-policy input adapter.

The endpoint and field names follow atonderski/neuro-ncap's Model Node API and
wljungbergh/UniAD's inference/server.py. The renderer payload is a torch-saved uint8 tensor, as decoded by that
reference server; it is deliberately not treated as an ordinary PNG or an untrusted pickle.
"""

import base64
import binascii
import io
import zipfile

import numpy as np
import torch

from unidrivevla.closedloop.runtime import _matrix, _rigid

CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
CAMERA_VIEWS = {
    "CAM_FRONT": "front", "CAM_FRONT_LEFT": "front_left", "CAM_FRONT_RIGHT": "front_right",
    "CAM_BACK": "back", "CAM_BACK_LEFT": "back_left", "CAM_BACK_RIGHT": "back_right",
}
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_194_304


def decode_image_tensor(payload):
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(f"each NeuroNCAP image payload must be 1..{MAX_IMAGE_BYTES} bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > 64 or sum(item.file_size for item in entries) > MAX_IMAGE_BYTES:
                raise ValueError("NeuroNCAP tensor archive exceeds the decoded size bound")
    except zipfile.BadZipFile as exc:
        raise ValueError("NeuroNCAP images must use the renderer's torch.save ZIP format") from exc
    try:
        image = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("NeuroNCAP image is not a weights-only torch-saved tensor") from exc
    if not torch.is_tensor(image) or image.dtype != torch.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("NeuroNCAP images must decode to uint8 HxWx3 RGB tensors")
    height, width = image.shape[:2]
    if min(height, width) < 32 or height * width > MAX_IMAGE_PIXELS:
        raise ValueError("NeuroNCAP image dimensions are outside the accepted bounds")
    return image.contiguous().numpy()


def _image_bytes(value, name):
    if not isinstance(value, str):
        raise ValueError(f"images[{name}] must be base64 text")
    if len(value) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError(f"images[{name}] exceeds the encoded size bound")
    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"images[{name}] is not valid base64") from exc
    return payload


def canonical_input(request):
    """Validate the official request and return calibrated arguments for PolicyRuntime.predict()."""
    if not isinstance(request, dict):
        raise ValueError("NeuroNCAP request must be a JSON object")
    required = {"images", "ego2world", "canbus", "timestamp", "command", "calibration"}
    missing = required - request.keys()
    if missing:
        raise ValueError(f"NeuroNCAP request is missing fields: {sorted(missing)}")
    images = request["images"]
    if not isinstance(images, dict) or set(images) != set(CAMERA_ORDER):
        raise ValueError(f"images must contain exactly {list(CAMERA_ORDER)}")
    timestamp = request["timestamp"]
    command = request["command"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("timestamp must be a nonnegative integer in microseconds")
    if isinstance(command, bool) or command not in (0, 1, 2):
        raise ValueError("command must be 0 (right), 1 (left), or 2 (straight)")
    canbus = np.asarray(request["canbus"], dtype=np.float64)
    if canbus.shape != (16,) or not np.isfinite(canbus).all():
        raise ValueError("canbus must contain the 16 finite values used by the NeuroNCAP reference runner")

    calibration = request["calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object")
    camera2image = calibration.get("camera2image")
    camera2ego = calibration.get("camera2ego")
    if (not isinstance(camera2image, dict) or not isinstance(camera2ego, dict)
            or set(camera2image) != set(CAMERA_ORDER) or set(camera2ego) != set(CAMERA_ORDER)):
        raise ValueError("calibration must provide camera2image and camera2ego for all six nuScenes cameras")
    ego2world = _rigid(request["ego2world"], "ego2world")
    if "lidar2ego" not in calibration:
        raise ValueError("calibration.lidar2ego is required; identity calibration is not assumed")
    lidar2ego = _rigid(calibration["lidar2ego"], "calibration.lidar2ego")
    axes = np.eye(4, dtype=np.float64)
    axes[:3, :3] = lidar2ego[:3, :3]
    pose = ego2world @ lidar2ego @ np.linalg.inv(axes)

    cameras, camera_calibration = {}, {}
    for name in CAMERA_ORDER:
        intrinsic = _matrix(camera2image[name], (3, 3), f"calibration.camera2image[{name}]")
        sensor2body = _rigid(camera2ego[name], f"calibration.camera2ego[{name}]")
        camera_calibration[CAMERA_VIEWS[name]] = {
            "intrinsic": intrinsic,
            "sensor2ego": axes @ np.linalg.inv(lidar2ego) @ sensor2body,
        }

    payloads = {name: _image_bytes(images[name], name) for name in CAMERA_ORDER}
    if sum(map(len, payloads.values())) > 96 * 1024 * 1024:
        raise ValueError("combined NeuroNCAP image payload exceeds 96 MiB")
    decoded = {name: decode_image_tensor(payloads[name]) for name in CAMERA_ORDER}
    for name in CAMERA_ORDER:
        view = CAMERA_VIEWS[name]
        cameras[view] = {"image": decoded[name], **camera_calibration[view]}

    # The nuScenes profile reads acceleration and rotation rate. NeuroNCAP's public API does not provide steering.
    return {
        "cameras": cameras,
        "ego2global": pose,
        "native_to_ego": axes,
        "timestamp": timestamp / 1_000_000.0,
        "command": ("right", "left", "straight")[command],
        "ego_acceleration": canbus[7:10].astype(np.float32),
        "ego_angular_velocity": canbus[10:13].astype(np.float32),
        "ego_velocity": canbus[13:16].astype(np.float32),
        "ego_steer": float("nan"),
    }


def trajectory_to_ncap(trajectory):
    """The official API consumes x-forward/y-left ego-frame points unchanged."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or not np.isfinite(trajectory).all():
        raise ValueError("NeuroNCAP trajectory must be finite (T, 2) ego-frame points")
    return trajectory.tolist()
