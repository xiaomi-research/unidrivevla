"""CPU regression for native LiDAR head inputs and canonical public trajectories."""

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unidrivevla.adapters.unidrivevla import (Profile, UniDriveVLAPolicy, _check_checkpoint_frames,
                                               head_kwargs, prepare_sample)
from unidrivevla.closedloop.runtime import PolicyRuntime, _poses_to_native


def _rotation(angle, translation=(0.0, 0.0, 0.0)):
    c, s = math.cos(angle), math.sin(angle)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    value[:3, 3] = translation
    return value


def _axes(angle):
    value = _rotation(angle)
    return value


def _profile():
    source = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "profile" / "nuscenes.yaml")
    return replace(Profile.from_config(source), frame="lidar", image_hw=[8, 8])


class CaptureHead(nn.Module):
    def __init__(self, trajectory):
        super().__init__()
        self.trajectory = torch.as_tensor(trajectory, dtype=torch.float32)
        self.last = None

    def forward_test(self, **kwargs):
        self.last = kwargs
        batch = kwargs["img"].shape[0]
        return {"traj": self.trajectory.to(kwargs["img"].device).unsqueeze(0).expand(batch, -1, -1)}


def _policy(profile, trajectory):
    policy = UniDriveVLAPolicy.__new__(UniDriveVLAPolicy)
    nn.Module.__init__(policy)
    policy.profile = profile
    policy.head = CaptureHead(trajectory)
    return policy.eval()


def _cameras(profile, sensor2ego=None):
    sensor2ego = sensor2ego or _rotation(0.13, (0.3, -0.2, 1.5))
    return [
        {"view": view, "image": np.full((8, 8, 3), (17, 41, 83), dtype=np.uint8),
         "intrinsic": np.array([[6.0, 0.0, 4.0], [0.0, 7.0, 4.0], [0.0, 0.0, 1.0]]),
         "sensor2ego": sensor2ego.copy(), "size": (8, 8)}
        for view in profile.cameras
    ]


def _batch(prepared):
    features, targets = prepared
    batched = {key: (value.unsqueeze(0) if torch.is_tensor(value) else value)
               for key, value in features.items()}
    batched["coordinate_frame"] = [features["coordinate_frame"]]
    batched["img_metas"] = [features["img_metas"]]
    target_batch = {key: (value.unsqueeze(0) if torch.is_tensor(value) else value)
                    for key, value in targets.items()}
    return batched, target_batch


def main():
    profile = _profile()
    assert profile.official_attention_mask and profile.prefix_max_len == 3740
    assert head_kwargs(profile, "vlm", None, False, False)["official_attention_mask"] is True
    axes = _axes(0.61)
    trajectory = np.array([[1.0, 0.0], [2.0, 0.5], [3.0, 1.0], [4.0, 1.5],
                          [5.0, 2.0], [6.0, 2.5]], dtype=np.float32)
    features = {
        "cameras": _cameras(profile),
        "coordinate_frame": "lidar",
        "native_to_ego": axes,
        "history_trajectory": np.zeros((4, 3), dtype=np.float32),
        "ego_velocity": np.array([4.0, 0.2, 0.0], dtype=np.float32),
        "ego_acceleration": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "ego_angular_velocity": np.array([0.01, 0.02, 0.03], dtype=np.float32),
        "ego_steer": 0.0,
        "command": "straight",
        "timestamp": 1.0,
        "ego2global": _rotation(0.2, (10.0, 20.0, 0.0)),
    }
    task_valid = {"planning": True, "det": False, "map": False, "motion": False,
                  "occupancy": False, "geometry": False}
    targets = {"trajectory": np.column_stack((trajectory, np.zeros(len(trajectory), dtype=np.float32))),
               "trajectory_mask": np.ones(len(trajectory), dtype=np.float32), "task_valid": task_valid}
    prepared = prepare_sample(features, targets, profile)
    feature_batch, target_batch = _batch(prepared)
    policy = _policy(profile, trajectory)
    output = policy(feature_batch, target_batch)
    head = policy.head.last
    assert "coordinate_frame" not in head and "native_to_ego" not in head
    intrinsic4 = np.eye(4, dtype=np.float64)
    intrinsic4[:3, :3] = features["cameras"][0]["intrinsic"]
    assert torch.allclose(head["projection_mat"][0, 0],
                          torch.from_numpy(intrinsic4 @ np.linalg.inv(features["cameras"][0]["sensor2ego"])).float(),
                          atol=1e-5)
    assert torch.allclose(head["ego_status"][0], torch.tensor([0.1, 0.2, 0.3, 0.01, 0.02, 0.03]))
    assert torch.allclose(head["hist_traj"][0], torch.zeros((4, 2)))
    expected_public = torch.from_numpy(trajectory) @ torch.from_numpy(axes[:2, :2].T).float()
    assert torch.allclose(output["pred_traj"][0, :, :2], expected_public, atol=1e-5)
    assert torch.allclose(output["pred_traj_head"][0, :, :2], torch.from_numpy(trajectory), atol=1e-5)
    assert policy.compute_loss(feature_batch, target_batch, output).item() < 1e-6

    for bad in (
        {**features, "coordinate_frame": "ego"},
        {key: value for key, value in features.items() if key != "native_to_ego"},
    ):
        try:
            prepare_sample(bad, targets, profile)
        except ValueError:
            pass
        else:
            raise AssertionError("native sample with mismatched or missing frame metadata was accepted")

    try:
        _check_checkpoint_frames({"agent_config": {"profile": {"name": profile.name, "frame": "ego"}}},
                                 {profile.name: profile})
    except ValueError as exc:
        assert "does not match target" in str(exc)
    else:
        raise AssertionError("canonical checkpoint was silently accepted for a native profile")
    _check_checkpoint_frames({}, {profile.name: profile}, checkpoint_frame="lidar")
    try:
        _check_checkpoint_frames({}, {profile.name: profile})
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError("legacy native checkpoint without explicit frame provenance was accepted")
    history = np.array([[0.0, 0.0, 0.4]], dtype=np.float64)
    assert np.allclose(_poses_to_native(history, axes)[0, 2], 0.4, atol=1e-6)

    # Exercise the production closed-loop adapter with a nontrivial actual LiDAR mounting rotation.
    runtime_policy = _policy(profile, trajectory)
    runtime = PolicyRuntime(runtime_policy, "cpu")
    canonical_cameras = {view: {key: value for key, value in camera.items() if key != "view"}
                         for view, camera in ((item["view"], item) for item in _cameras(profile))}
    ego_pose = _rotation(0.2, (2.0, 3.0, 0.0))
    args = dict(cameras=canonical_cameras, ego2global=ego_pose, timestamp=0.0, command="straight",
                ego_velocity=features["ego_velocity"], ego_acceleration=features["ego_acceleration"],
                ego_angular_velocity=features["ego_angular_velocity"], native_to_ego=axes)
    closedloop = runtime.predict(**args)
    assert np.allclose(closedloop, expected_public.numpy(), atol=1e-5)
    assert np.allclose(runtime_policy.head.last["img_metas"][0]["T_global"], ego_pose @ axes, atol=1e-5)
    expected_native_camera = np.linalg.inv(axes) @ canonical_cameras[profile.cameras[0]]["sensor2ego"]
    intrinsic4 = np.eye(4, dtype=np.float64)
    intrinsic4[:3, :3] = canonical_cameras[profile.cameras[0]]["intrinsic"]
    assert np.allclose(runtime_policy.head.last["projection_mat"][0, 0],
                       intrinsic4 @ np.linalg.inv(expected_native_camera), atol=1e-5)
    moved_pose = ego_pose.copy()
    moved_pose[:2, 3] += 0.5 * ego_pose[:2, 0]
    runtime.predict(**{**args, "ego2global": moved_pose, "timestamp": 0.5})
    expected_history = np.array([-0.5, 0.0]) @ axes[:2, :2]
    assert np.allclose(runtime_policy.head.last["hist_traj"][0, 2].numpy(), expected_history, atol=1e-5)

    try:
        runtime.predict(**{key: value for key, value in args.items() if key != "native_to_ego"})
    except ValueError as exc:
        assert "requires" in str(exc)
    else:
        raise AssertionError("native closed-loop inference guessed missing LiDAR axes")
    print("OK: native head coordinates, canonical output, validation loss, calibration, checkpoint guards, closed-loop")


if __name__ == "__main__":
    main()
