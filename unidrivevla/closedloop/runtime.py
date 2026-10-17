"""Load a policy without the training loop and run one calibrated simulator frame."""

import math
import threading
import uuid

import numpy as np
import torch

from unidrivevla.adapters.unidrivevla import _rotation_only, prepare_sample
from unidrivevla.closedloop.history import PoseHistory


class _TemporalStateRing:
    """Rotate one model runtime snapshot per simulator cadence phase."""

    def __init__(self, size, reset, save, load):
        self.size, self.reset_policy, self.save_policy, self.load_policy = int(size), reset, save, load
        if self.size < 1:
            raise ValueError("temporal_ring_size must be positive")
        self.reset()

    def reset(self):
        self.slots = [None] * self.size
        self.step = 0
        self.reset_policy()

    def begin(self):
        slot = self.step % self.size
        state = self.slots[slot]
        # Remove before restoring because instance-bank tensors can be updated in place.
        self.slots[slot] = None
        if state is None:
            self.reset_policy()
        else:
            self.load_policy(state)
        return slot

    def finish(self, slot):
        self.slots[slot] = self.save_policy()
        self.step += 1


def _matrix(value, shape, name):
    try:
        value = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric with shape {shape}") from exc
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return value


def _rigid(value, name):
    value = _matrix(value, (4, 4), name)
    rotation = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or np.linalg.det(rotation) < 0.99):
        raise ValueError(f"{name} is not a rigid transform")
    return value


class PolicyRuntime:
    def __init__(self, policy, device="cuda", temporal_ring_size=1):
        self.policy = policy.to(torch.device(device)).eval()
        self.device = torch.device(device)
        self.profile = policy.profile
        self._lock = threading.RLock()
        self.temporal_ring_size = int(temporal_ring_size)
        if self.temporal_ring_size < 1:
            raise ValueError("temporal_ring_size must be positive")
        self._ring = None
        if self.temporal_ring_size > 1:
            self._ring = self._make_temporal_ring(self.temporal_ring_size)
        self.reset()

    def _make_temporal_ring(self, size):
        save = getattr(self.policy, "training_runtime_state_dict", None)
        load = getattr(self.policy, "load_training_runtime_state_dict", None)
        if not callable(save) or not callable(load):
            raise ValueError("temporal ring needs policy training-runtime state save/load hooks")
        return _TemporalStateRing(size, self._reset_policy_temporal, save,
                                  lambda state: load(state, device=self.device))

    def enable_temporal_ring(self, size):
        """Use independent decoder state per simulator cadence phase; resets the episode."""
        size = int(size)
        if size < 1:
            raise ValueError("temporal_ring_size must be positive")
        with self._lock:
            self.temporal_ring_size = size
            self._ring = self._make_temporal_ring(size) if size > 1 else None
            self.reset()

    def _reset_policy_temporal(self):
        reset = getattr(getattr(self.policy, "head", None), "reset_temporal", None)
        if reset is not None:
            reset()

    def reset(self):
        with self._lock:
            self.history = PoseHistory()
            self.stream_id = f"closedloop-{uuid.uuid4()}"
            if self._ring is None:
                self._reset_policy_temporal()
            else:
                self._ring.reset()

    def predict(self, cameras, ego2global, timestamp, command, ego_velocity, ego_acceleration,
                ego_angular_velocity, ego_steer=float("nan"), native_to_ego=None):
        """Camera transforms and pose use canonical EGO axes; native profiles require their actual LiDAR axes."""
        with self._lock:
            try:
                slot = self._ring.begin() if self._ring is not None else None
                pose = _rigid(ego2global, "ego2global")
                frame = getattr(self.profile, "frame", "ego")
                if frame == "lidar":
                    if native_to_ego is None:
                        raise ValueError("LiDAR-frame inference requires the source's actual native_to_ego rotation")
                    axes = _rotation_only(native_to_ego)
                elif frame == "ego":
                    axes = np.eye(4, dtype=np.float64)
                else:
                    raise ValueError(f"unsupported policy frame {frame!r}")
                required = set(self.profile.cameras)
                if not isinstance(cameras, dict) or set(cameras) != required:
                    raise ValueError(f"profile {self.profile.name} requires cameras {sorted(required)}")
                history = self.history.history(timestamp, pose)
                if frame == "lidar":
                    history = _poses_to_native(history, axes)
                prepared = self._prepare(cameras, pose, timestamp, command, ego_velocity, ego_acceleration,
                                         ego_angular_velocity, ego_steer, history, axes)
                with torch.inference_mode():
                    output = self.policy(prepared)["pred_traj"][0, :, :2].float().cpu().numpy()
                if self._ring is not None:
                    self._ring.finish(slot)
            except Exception:
                self.reset()
                raise
            if output.ndim != 2 or output.shape[1] != 2 or not np.isfinite(output).all():
                self.reset()
                raise RuntimeError(f"policy returned invalid trajectory shape/values: {output.shape}")
            return output.astype(np.float64, copy=False)

    def _prepare(self, cameras, pose, timestamp, command, velocity, acceleration, angular_velocity, steer, history,
                 native_to_ego=None):
        axes = np.eye(4, dtype=np.float64) if native_to_ego is None else _rotation_only(native_to_ego)
        frame = getattr(self.profile, "frame", "ego")
        if frame == "lidar" and native_to_ego is None:
            raise ValueError("LiDAR-frame inference requires native_to_ego")
        if frame == "ego" and not np.allclose(axes, np.eye(4), atol=1e-5):
            raise ValueError("ego-frame inference must use identity native_to_ego")
        features = {
            "cameras": [],
            "history_trajectory": history,
            "ego_velocity": np.asarray(velocity, dtype=np.float32).reshape(3),
            "ego_acceleration": np.asarray(acceleration, dtype=np.float32).reshape(3),
            "ego_angular_velocity": np.asarray(angular_velocity, dtype=np.float32).reshape(3),
            "ego_steer": float(steer),
            "command": command,
            "timestamp": float(timestamp),
            "ego2global": pose @ axes,
            "stream_id": self.stream_id,
            "coordinate_frame": frame,
            "native_to_ego": axes,
        }
        for view in self.profile.cameras:
            item = cameras[view]
            image = np.asarray(item["image"])
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"camera {view} must be an RGB uint8 HxWx3 image")
            intrinsic = _matrix(item["intrinsic"], (3, 3), f"camera {view} intrinsic")
            if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
                raise ValueError(f"camera {view} intrinsic is invalid")
            sensor2ego = _rigid(item["sensor2ego"], f"camera {view} sensor2ego")
            if frame == "lidar":
                sensor2ego = np.linalg.inv(axes) @ sensor2ego
            features["cameras"].append({"view": view, "image": image, "intrinsic": intrinsic,
                                        "sensor2ego": sensor2ego, "size": image.shape[1::-1]})
        tensors, _ = prepare_sample(features, {}, self.profile)
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in tensors.items()
                 if torch.is_tensor(value)}
        if "img_metas" in tensors:
            batch["img_metas"] = [tensors["img_metas"]]
        batch["coordinate_frame"] = [frame]
        batch["native_to_ego"] = tensors["native_to_ego"].unsqueeze(0).to(self.device)
        return batch


def _poses_to_native(poses, native_to_ego):
    """Convert canonical EGO history into native axes while keeping its LiDAR-centered origin."""
    poses = np.asarray(poses, dtype=np.float64)
    rotation = native_to_ego[:3, :3]
    xyz = np.concatenate((poses[:, :2], np.zeros((len(poses), 1))), axis=1) @ rotation
    headings = []
    for angle in poses[:, 2]:
        c, s = math.cos(float(angle)), math.sin(float(angle))
        relative = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        native_relative = rotation.T @ relative @ rotation
        headings.append(math.atan2(native_relative[1, 0], native_relative[0, 0]))
    return np.column_stack((xyz[:, :2], headings))
