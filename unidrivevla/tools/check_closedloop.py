"""Offline CPU regression for NeuroNCAP and HUGSIM request, calibration, output, reset, and FIFO contracts.

Official schemas under test: github.com/atonderski/neuro-ncap/docs/framework.md,
github.com/wljungbergh/UniAD/inference/server.py, and github.com/hyzhou404/HUGSIM/closed_loop.py plus
sim/hugsim_env/envs/hug_sim.py and sim/utils/sim_utils.py. No simulator or benchmark score is generated.
"""

import base64
from dataclasses import replace
import http.client
import io
import json
import os
import pickle
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torch import nn
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unidrivevla.adapters.unidrivevla import Profile
from unidrivevla.closedloop.hugsim import (HUGSIMInputAdapter, HUGSIM_TO_EGO, hugsim_to_canonical_plan,
                                           load_calibration, trajectory_to_hugsim)
from unidrivevla.closedloop.neuro_ncap import (CAMERA_ORDER, CAMERA_VIEWS, MAX_IMAGE_BYTES, canonical_input,
                                               decode_image_tensor)
from unidrivevla.closedloop.runtime import PolicyRuntime
from unidrivevla.closedloop import hugsim_client
from unidrivevla.serve import _handler
from unidrivevla.serve import load_runtime
from unidrivevla.serve import main as serve_main


class TinyPolicy(nn.Module):
    def __init__(self, profile):
        super().__init__()
        self.profile = profile
        self.head = SimpleNamespace(reset_count=0)
        self.head.reset_temporal = lambda: setattr(self.head, "reset_count", self.head.reset_count + 1)
        self.last = None

    def forward(self, batch):
        self.last = batch
        assert batch["img"].shape == (1, 6, 3, 544, 960)
        assert batch["projection_mat"].shape == (1, 6, 4, 4)
        xy = torch.tensor([[1.0, 0.0], [2.0, 0.25], [3.0, 0.5],
                           [4.0, 0.75], [5.0, 1.0], [6.0, 1.25]], device=batch["img"].device)
        return {"pred_traj": torch.cat([xy, torch.zeros(6, 1, device=xy.device)], dim=1).unsqueeze(0)}


class TinyCheckpointPolicy(nn.Module):
    """Small policy shell whose loader delegates to the production UniDriveVLAPolicy initializer."""

    def __init__(self, profile, checkpoint_path, checkpoint_frame=None, with_motion=False):
        super().__init__()
        from unidrivevla.adapters.unidrivevla import Profile

        self.profile = Profile.from_config(profile)
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_frame = checkpoint_frame
        self.with_motion = bool(with_motion)
        self.weight = nn.Parameter(torch.zeros(()))
        if self.with_motion:
            self.motion = nn.Linear(1, 1, bias=False)

    def initialize(self, strict=False):
        from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy

        return UniDriveVLAPolicy.initialize(self, strict=strict)


def _rigid(yaw, translation):
    c, s = np.cos(yaw), np.sin(yaw)
    value = np.eye(4)
    value[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    value[:3, 3] = translation
    return value


def _encoded_rgb(value, size=(544, 960)):
    h, w = size
    image = torch.empty((h, w, 3), dtype=torch.uint8)
    image[..., 0], image[..., 1], image[..., 2] = value
    output = io.BytesIO()
    torch.save(image, output)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _ncap_request(images):
    camera2image, camera2ego = {}, {}
    for index, name in enumerate(CAMERA_ORDER):
        camera2image[name] = [[500.0 + index, 0.0, 480.0], [0.0, 510.0 + index, 272.0], [0.0, 0.0, 1.0]]
        camera2ego[name] = _rigid(index * 0.15, [0.8, (index - 2.5) * 0.1, 1.4]).tolist()
    canbus = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03,
              4.0, 0.0, 0.0]
    return {
        "images": images,
        "ego2world": _rigid(0.2, [10.0, 20.0, 0.0]).tolist(),
        "canbus": canbus,
        "timestamp": 1_000_000,
        "command": 2,
        "calibration": {"camera2image": camera2image, "camera2ego": camera2ego,
                        "lidar2ego": _rigid(0.1, [1.2, 0.1, 1.8]).tolist()},
    }


def _check_neuro_ncap(profile):
    images = {name: _encoded_rgb((11, 23, 37)) for name in CAMERA_ORDER}
    request = _ncap_request(images)
    policy = TinyPolicy(profile)
    runtime = PolicyRuntime(policy, "cpu")
    decoded = canonical_input(request)
    result = runtime.predict(**decoded)
    assert np.allclose(result[0], [1.0, 0.0])

    lidar2ego = np.asarray(request["calibration"]["lidar2ego"])
    axes = np.eye(4)
    axes[:3, :3] = lidar2ego[:3, :3]
    expected_pose = np.asarray(request["ego2world"]) @ lidar2ego @ np.linalg.inv(axes)
    assert np.allclose(policy.last["img_metas"][0]["T_global"], expected_pose)
    name = CAMERA_ORDER[0]
    view = CAMERA_VIEWS[name]
    sensor2body = np.asarray(request["calibration"]["camera2ego"][name])
    sensor2ego = axes @ np.linalg.inv(lidar2ego) @ sensor2body
    projection = np.eye(4)
    projection[:3, :3] = request["calibration"]["camera2image"][name]
    assert np.allclose(policy.last["projection_mat"][0, profile.cameras.index(view)], projection @ np.linalg.inv(sensor2ego), atol=1e-5)
    assert policy.last["img"][0, profile.cameras.index(view), :, 0, 0].tolist() == [37, 23, 11]
    assert policy.last["ego_status"].shape == (1, 10)
    assert np.allclose(policy.last["hist_traj"][0, -1].numpy(), [0, 0])

    request["timestamp"] += 500_000
    request["ego2world"] = _rigid(0.2, [10.0 + 0.5 * np.cos(0.2), 20.0 + 0.5 * np.sin(0.2), 0.0]).tolist()
    runtime.predict(**canonical_input(request))
    assert np.allclose(policy.last["hist_traj"][0, -2].numpy(), [-0.5, 0], atol=1e-5)
    resets = policy.head.reset_count
    runtime.reset()
    assert policy.head.reset_count == resets + 1
    request["timestamp"] = 1_000_000
    runtime.predict(**canonical_input(request))
    assert np.allclose(policy.last["hist_traj"][0, -1].numpy(), [0, 0])

    try:
        canonical_input({**request, "images": {"CAM_FRONT": images["CAM_FRONT"]}})
    except ValueError:
        pass
    else:
        raise AssertionError("missing camera accepted")
    try:
        decode_image_tensor(b"x" * (MAX_IMAGE_BYTES + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("oversized tensor payload accepted")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as compressed:
        compressed.writestr("archive/data/0", b"\0" * (MAX_IMAGE_BYTES + 1))
    with patch("torch.load", side_effect=AssertionError("archive bounds must be checked before torch.load")):
        try:
            decode_image_tensor(archive.getvalue())
        except ValueError:
            pass
        else:
            raise AssertionError("oversized decoded tensor archive accepted")
    payload = io.BytesIO()
    torch.save({"not": "an image tensor"}, payload)
    try:
        decode_image_tensor(payload.getvalue())
    except ValueError:
        pass
    else:
        raise AssertionError("non-tensor payload accepted")

    runtime.reset()
    api = ThreadingHTTPServer(("127.0.0.1", 0), _handler(runtime))
    thread = threading.Thread(target=api.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection("127.0.0.1", api.server_port, timeout=30)
        client.request("GET", "/alive")
        response = client.getresponse()
        assert response.status == 200 and json.loads(response.read()) is True
        client.request("POST", "/infer", json.dumps(request), {"Content-Type": "application/json"})
        response = client.getresponse()
        result = json.loads(response.read())
        assert response.status == 200 and np.allclose(result["trajectory"][0], [1.0, 0.0])
        assert result["aux_outputs"] == {}
        client.request("POST", "/reset", "{}", {"Content-Type": "application/json"})
        response = client.getresponse()
        assert response.status == 200 and json.loads(response.read()) is True
        client.close()
    finally:
        api.shutdown()
        api.server_close()
        thread.join(timeout=2)
    print("OK: NeuroNCAP calibration, tensor bounds, history, reset, and canonical output")


def _check_hugsim(profile):
    cam_rect = np.eye(4)
    cam_rect[:3, 3] = [0, -0.3, 0]
    front_v2c = _rigid(0.05, [0.05, 1.4, 0.02])
    lidar2body = _rigid(0.03, [0.1, 0.2, -0.5])
    cam_params, cams_yaml, rgb = {}, {}, {}
    for index, name in enumerate(CAMERA_ORDER):
        # Build six non-identity camera extrinsics that agree on the same calibrated LiDAR/body transform.
        desired_cam2body = cam_rect if index == 0 else _rigid(index * 0.18, [0.1 * index, 0.05, 0.2])
        v2c = cam_rect @ np.linalg.inv(desired_cam2body) @ front_v2c
        l2c = np.linalg.inv(desired_cam2body) @ lidar2body
        cam_params[name] = {
            "v2c": v2c, "l2c": l2c,
            "intrinsic": {"H": 450, "W": 800, "cx": 400.0 + index, "cy": 225.0,
                          "fovx": 1.1, "fovy": 0.75},
        }
        def degrees(matrix):
            return Rotation.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True).tolist()

        cams_yaml[name] = {
            "extrinsics": {"v2c_rot": degrees(v2c), "v2c_trans": v2c[:3, 3].tolist(),
                           "l2c_rot": degrees(l2c), "l2c_trans": l2c[:3, 3].tolist()},
            "intrinsics": {"H": 450, "W": 800, "cx": 400.0 + index, "cy": 225.0,
                           "fovx": float(np.degrees(1.1)), "fovy": float(np.degrees(0.75))},
        }
        rgb[name] = np.full((450, 800, 3), [41, 83, 127], dtype=np.uint8)

    camera_path = Path(tempfile.gettempdir()) / f"closedloop-camera-{os.getpid()}.yaml"
    kinematic_path = Path(tempfile.gettempdir()) / f"closedloop-kinematic-{os.getpid()}.yaml"
    OmegaConf.save(OmegaConf.create({"cams": cams_yaml, "cam_rect": {"rot": [0, 0, 0], "trans": [0, -0.3, 0]}}), camera_path)
    OmegaConf.save(OmegaConf.create({"Lr": 1.35, "Lf": 1.35}), kinematic_path)
    try:
        rect, wheelbase, expected_cameras = load_calibration(camera_path, kinematic_path)
        adapter = HUGSIMInputAdapter(rect, wheelbase, expected_cameras)
        ego_pos, ego_yaw = [1.5, 2.0, 3.0], 0.2
        ego_box = [3.0, -1.5, -2.0, 1.6, 3.0, 1.5, -ego_yaw]
        info = {"ego_pos": ego_pos, "ego_rot": [0.0, ego_yaw, 0.0], "ego_velo": 3.0,
                "ego_steer": 0.1, "accelerate": 0.4, "steer_rate": 0.0, "timestamp": 0.0,
                "command": 0, "ego_box": ego_box, "cam_params": cam_params}
        converted = adapter.convert({"rgb": rgb}, info)
        policy = TinyPolicy(profile)
        runtime = PolicyRuntime(policy, "cpu")
        trajectory = runtime.predict(**converted)
        offset = HUGSIM_TO_EGO @ lidar2body[:3, 3]
        yaw_rotation = _rigid(-ego_yaw, [0, 0, 0])[:3, :3]
        assert np.allclose(converted["ego2global"][:3, 3],
                           np.array([3.0, -1.5, -2.0]) + yaw_rotation @ offset, atol=1e-5)
        view = "front"
        camera_index = profile.cameras.index(view)
        camera_to_body_h = front_v2c @ np.linalg.inv(cam_params["CAM_FRONT"]["v2c"]) @ rect
        lidar_to_body_ego = np.eye(4)
        lidar_to_body_ego[:3, :3] = HUGSIM_TO_EGO
        lidar_to_body_ego = lidar_to_body_ego @ camera_to_body_h @ cam_params["CAM_FRONT"]["l2c"]
        axes = np.eye(4)
        axes[:3, :3] = lidar_to_body_ego[:3, :3]
        camera_to_body_ego = np.eye(4)
        camera_to_body_ego[:3, :3] = HUGSIM_TO_EGO
        camera_to_body_ego = camera_to_body_ego @ camera_to_body_h
        sensor2ego = axes @ np.linalg.inv(lidar_to_body_ego) @ camera_to_body_ego
        intrinsic = cam_params["CAM_FRONT"]["intrinsic"]
        scale = 544 / 450
        left = int((int(800 * scale) - 960) / 2)
        K4 = np.eye(4)
        K4[:3, :3] = [[(800 / (2 * np.tan(1.1 / 2))) * scale, 0, intrinsic["cx"] * scale - left],
                      [0, (450 / (2 * np.tan(0.75 / 2))) * scale, intrinsic["cy"] * scale], [0, 0, 1]]
        assert np.allclose(policy.last["projection_mat"][0, camera_index], K4 @ np.linalg.inv(sensor2ego), atol=1e-4)
        sent = trajectory_to_hugsim(trajectory)
        assert np.allclose(sent[0], [0.0, 1.0])  # model [x=1,y=0] -> HUGSIM [x-right=0,y-forward=1]
        assert np.allclose(hugsim_to_canonical_plan(sent), trajectory)

        # Drive the FIFO loop with a trusted in-memory stream on platforms without os.mkfifo (e.g. Windows).
        with tempfile.TemporaryDirectory(prefix="unidrivevla-fifo-check-") as output_dir:
            output = Path(output_dir)
            obs_values = [pickle.dumps(({"rgb": rgb}, info), protocol=4), pickle.dumps("Done", protocol=4)]
            messages = [pickle.loads(value) for value in obs_values]
            writes = []

            class FakeReader:
                def __init__(self, path, timeout):
                    pass

                def read(self):
                    return messages.pop(0)

                def close(self):
                    pass

            with patch.object(hugsim_client, "wait_for_fifos"), \
                    patch.object(hugsim_client, "PickleFIFOReader", FakeReader), \
                    patch.object(hugsim_client, "write_pickle_fifo", lambda _path, value, timeout=30: writes.append(value)):
                runtime.reset()
                assert hugsim_client.run_fifo(output, runtime, adapter, timeout=0.1) == 0
            assert len(writes) == 1
            assert np.allclose(hugsim_to_canonical_plan(writes[0]), trajectory)

            # A regular file at an IPC path must be rejected without truncation.
            bad_path = output / "obs_pipe"
            bad_path.write_bytes(b"keep-existing-content")
            try:
                hugsim_client.wait_for_fifos(bad_path, output / "plan_pipe", 0.01)
            except ValueError:
                pass
            else:
                raise AssertionError("regular file accepted as a FIFO")
            assert bad_path.read_bytes() == b"keep-existing-content"
            try:
                hugsim_client.write_pickle_fifo(bad_path, np.zeros((6, 2)))
            except ValueError:
                pass
            else:
                raise AssertionError("regular file accepted as a FIFO writer")
            assert bad_path.read_bytes() == b"keep-existing-content"

            if hasattr(os, "mkfifo"):
                obs_pipe, plan_pipe = output / "live_obs", output / "live_plan"
                os.mkfifo(obs_pipe)
                os.mkfifo(plan_pipe)
                reader = hugsim_client.PickleFIFOReader(obs_pipe, timeout=1)
                writer = threading.Thread(target=lambda: _write_once(obs_pipe, pickle.dumps("Done", protocol=4)))
                writer.start()
                assert reader.read() == "Done"
                reader.close()
                writer.join(timeout=1)
                assert not writer.is_alive()
                received = []
                plan_reader = threading.Thread(target=lambda: _read_once(plan_pipe, received))
                plan_reader.start()
                hugsim_client.write_pickle_fifo(plan_pipe, np.array([[2.0, 1.0]]), timeout=1)
                plan_reader.join(timeout=1)
                assert not plan_reader.is_alive() and np.allclose(received[0], [[2.0, 1.0]])
    finally:
        camera_path.unlink(missing_ok=True)
        kinematic_path.unlink(missing_ok=True)
    print("OK: HUGSIM calibrated input, bicycle state, output axes, Done drain, and FIFO safety")


def _write_once(path, payload):
    with open(path, "wb") as pipe:
        pipe.write(payload)


def _read_once(path, output):
    with open(path, "rb") as pipe:
        output.append(pickle.load(pipe))


def _check_checkpoint_config(profile):
    from dataclasses import asdict
    from unidrivevla.adapters import initialize_for_inference
    from unidrivevla.infer import _requested_vlm_path, restore_checkpoint_agent_config
    from unidrivevla.train.trainer import Trainer

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_dir = root / "vlm"
        model_dir.mkdir()
        checkpoint = root / "model.ckpt"
        config = {"_target_": "unidrivevla.adapters.unidrivevla.UniDriveVLAPolicy", "profile": asdict(profile),
                  "vlm_path": str(model_dir), "with_motion": True}
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = OmegaConf.create({"agent": config})
        trainer.write_ckpt({}, checkpoint)
        assert torch.load(checkpoint, weights_only=True)["agent_config"] == config
        assert not checkpoint.with_suffix(".ckpt.tmp").exists()
        infer_cfg = OmegaConf.create({
            "data_loader": "nuscenes", "profile": {"name": "nuscenes"},
            "agent": {"_target_": config["_target_"], "profile": asdict(profile), "with_motion": False,
                      "train_vlm": True, "gradient_checkpointing": True, "vlm_path": "implicit-default"},
        })
        assert restore_checkpoint_agent_config(infer_cfg, checkpoint)
        assert infer_cfg.agent.with_motion and not infer_cfg.agent.train_vlm
        assert not infer_cfg.agent.gradient_checkpointing and not infer_cfg.agent._recursive_
        assert infer_cfg.agent.checkpoint_path == str(checkpoint) and infer_cfg.agent.vlm_path == str(model_dir)
        assert _requested_vlm_path(infer_cfg, [], "") is None
        assert _requested_vlm_path(infer_cfg, [], str(root / "environment-vlm")) == str(root / "environment-vlm")
        infer_cfg.agent.vlm_path = str(root / "cli-vlm")
        cli_vlm = _requested_vlm_path(infer_cfg, ["agent.vlm_path=/cli-vlm"], str(root / "environment-vlm"))
        assert cli_vlm == str(root / "cli-vlm")
        restore_checkpoint_agent_config(infer_cfg, checkpoint, cli_vlm)
        assert infer_cfg.agent.vlm_path == str(root / "cli-vlm")

        multi_checkpoint = root / "multi.ckpt"
        multi_config = {"_target_": "unidrivevla.adapters.multidomain.MultiProfilePolicy",
                        "profiles": {name: {"name": name} for name in ("nuscenes", "b2d", "navsim")},
                        "vlm_path": str(model_dir), "with_motion": True}
        torch.save({"state_dict": {}, "agent_config": multi_config}, multi_checkpoint)
        infer_cfg.data_loader = "navsim"
        assert restore_checkpoint_agent_config(infer_cfg, multi_checkpoint)
        assert infer_cfg.agent._target_ == multi_config["_target_"]
        assert infer_cfg.agent.exact_resume is False and infer_cfg.agent._recursive_ is False

        legacy = root / "legacy.ckpt"
        torch.save({"state_dict": {}}, legacy)
        previous = OmegaConf.to_container(infer_cfg.agent, resolve=True)
        assert not restore_checkpoint_agent_config(infer_cfg, legacy)
        assert OmegaConf.to_container(infer_cfg.agent, resolve=True) == previous

        policy = TinyPolicy(profile)
        policy.initialize = lambda strict=False: setattr(policy, "strict", strict)
        with patch("unidrivevla.serve.instantiate", return_value=policy) as make:
            runtime = load_runtime("nuscenes", str(checkpoint), "cpu")
            assert runtime.profile.name == "nuscenes"
            assert policy.strict is True
            loaded = make.call_args.args[0]
            assert loaded.with_motion and not loaded.train_vlm
            assert make.call_args.kwargs["_recursive_"] is False
        try:
            load_runtime("nuscenes", str(checkpoint), "cpu", with_motion=False)
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint architecture override was silently accepted")
        legacy_policy = TinyPolicy(profile)
        legacy_policy.initialize = lambda: setattr(legacy_policy, "legacy_initialized", True)
        initialize_for_inference(legacy_policy)
        assert legacy_policy.legacy_initialized
        config["_target_"] = "os.system"
        torch.save({"state_dict": {}, "agent_config": config}, checkpoint)
        with patch("unidrivevla.serve.instantiate", side_effect=AssertionError("untrusted target must not execute")):
            try:
                load_runtime("nuscenes", str(checkpoint), "cpu")
            except ValueError:
                pass
            else:
                raise AssertionError("untrusted checkpoint target accepted")


def _check_checkpoint_frame_compatibility(profile):
    from dataclasses import asdict
    from unidrivevla.adapters.multidomain import MultiProfilePolicy
    from unidrivevla.adapters.unidrivevla import Profile

    package = Path(__file__).resolve().parents[1]
    profile_dir = package / "config" / "profile"
    model_dir = None
    calls = []

    def config_for(selected_profile, with_motion=False):
        return {"_target_": "unidrivevla.adapters.unidrivevla.UniDriveVLAPolicy",
                "profile": asdict(selected_profile), "vlm_path": str(model_dir), "with_motion": with_motion}

    def tiny_instantiate(agent_cfg, _recursive_):
        assert _recursive_ is False
        values = OmegaConf.to_container(agent_cfg, resolve=True)
        calls.append(values)
        selected = values.get("profile") or values["profiles"]["b2d"]
        return TinyCheckpointPolicy(selected, values["checkpoint_path"], values.get("checkpoint_frame"),
                                    values.get("with_motion", False))

    with tempfile.TemporaryDirectory(prefix="unidrivevla-frame-check-") as directory:
        root = Path(directory)
        model_dir = root / "vlm"
        model_dir.mkdir()
        legacy = root / "legacy-stage2.ckpt"
        torch.save({"state_dict": {"weight": torch.tensor(1.0), "motion.weight": torch.tensor([[2.0]])}}, legacy)

        # The loader must refuse an ambiguous legacy nuScenes frame before constructing any model.
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir)}), \
                patch("unidrivevla.serve.instantiate", side_effect=tiny_instantiate) as instantiate_model:
            try:
                load_runtime("nuscenes", str(legacy), "cpu")
            except ValueError as exc:
                assert "verified --checkpoint-frame" in str(exc)
            else:
                raise AssertionError("metadata-less native nuScenes checkpoint was accepted without frame provenance")
            assert instantiate_model.call_count == 0

            # A verified ego-axis stage-2 checkpoint selects the actual legacy preset and keeps its motion head.
            runtime = load_runtime("nuscenes", str(legacy), "cpu", True, None, "ego")
            assert runtime.profile.frame == "ego" and runtime.profile.prefix_max_len == 0
            assert not runtime.profile.official_attention_mask
            assert runtime.policy.with_motion and runtime.policy.motion.weight.item() == 2.0
            assert calls[-1]["with_motion"] is True and calls[-1]["checkpoint_frame"] == "ego"

        native_legacy = root / "legacy-native.ckpt"
        torch.save({"state_dict": {"weight": torch.tensor(3.0)}}, native_legacy)
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir)}), \
                patch("unidrivevla.serve.instantiate", side_effect=tiny_instantiate):
            native_runtime = load_runtime("nuscenes", str(native_legacy), "cpu", None, None, "lidar")
            assert native_runtime.profile.frame == "lidar" and native_runtime.profile.prefix_max_len == 3740
            assert native_runtime.profile.official_attention_mask

            legacy_b2d = root / "legacy-b2d.ckpt"
            torch.save({"state_dict": {"weight": torch.tensor(3.5)}}, legacy_b2d)
            try:
                load_runtime("b2d", str(legacy_b2d), "cpu")
            except ValueError as exc:
                assert "verified --checkpoint-frame" in str(exc)
            else:
                raise AssertionError("metadata-less native Bench2Drive checkpoint was accepted without frame provenance")
            b2d_ego_runtime = load_runtime("b2d", str(legacy_b2d), "cpu", None, None, "ego")
            assert b2d_ego_runtime.profile.frame == "ego" and b2d_ego_runtime.profile.prefix_max_len == 0
            assert not b2d_ego_runtime.profile.official_attention_mask

        # New profile metadata wins over a user override, both in load_runtime and in the real initializer.
        saved_native = Profile.from_config(OmegaConf.load(profile_dir / "nuscenes.yaml"))
        metadata_checkpoint = root / "native-with-metadata.ckpt"
        torch.save({"state_dict": {"weight": torch.tensor(4.0)},
                    "agent_config": config_for(saved_native)}, metadata_checkpoint)
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir)}), \
                patch("unidrivevla.serve.instantiate", side_effect=tiny_instantiate):
            positional_vlm_runtime = load_runtime("nuscenes", str(metadata_checkpoint), "cpu", False, str(model_dir))
            assert positional_vlm_runtime.profile.frame == "lidar"
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir)}), \
                patch("unidrivevla.serve.instantiate", side_effect=AssertionError("mismatched frame must fail early")):
            try:
                load_runtime("nuscenes", str(metadata_checkpoint), "cpu", None, None, "ego")
            except ValueError as exc:
                assert "disagrees with checkpoint profile metadata" in str(exc)
            else:
                raise AssertionError("explicit frame override replaced authoritative checkpoint metadata")

        ego_cfg = OmegaConf.load(profile_dir / "nuscenes_ego.yaml")
        mismatched_shell = TinyCheckpointPolicy(ego_cfg, metadata_checkpoint, "ego", False)
        try:
            mismatched_shell.initialize(strict=True)
        except ValueError as exc:
            assert "frame" in str(exc)
        else:
            raise AssertionError("UniDriveVLAPolicy.initialize accepted a frame-mismatched full checkpoint")

        # NAVSIM has only the standard ego-frame profile; no separate legacy preset is selected.
        navsim = root / "legacy-navsim.ckpt"
        torch.save({"state_dict": {"weight": torch.tensor(5.0)}}, navsim)
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir)}), \
                patch("unidrivevla.serve.instantiate", side_effect=tiny_instantiate):
            nav_runtime = load_runtime("navsim", str(navsim), "cpu")
            assert nav_runtime.profile.name == "navsim" and nav_runtime.profile.frame == "ego"
            try:
                load_runtime("navsim", str(navsim), "cpu", None, None, "lidar")
            except ValueError as exc:
                assert "NAVSIM" in str(exc)
            else:
                raise AssertionError("NAVSIM accepted an unsupported LiDAR-frame checkpoint")

        # A selected-profile mapping survives the loader and the production multi-profile validator ignores
        # the other profiles' correctly recorded, different frames.
        profiles = {
            "nuscenes": asdict(Profile.from_config(OmegaConf.load(profile_dir / "nuscenes.yaml"))),
            "b2d": asdict(Profile.from_config(OmegaConf.load(profile_dir / "b2d_ego.yaml"))),
            "navsim": asdict(Profile.from_config(OmegaConf.load(profile_dir / "navsim.yaml"))),
        }
        multi_config = {"_target_": "unidrivevla.adapters.multidomain.MultiProfilePolicy", "profiles": profiles,
                        "vlm_path": str(model_dir), "with_motion": False}
        multi_checkpoint = root / "multi.ckpt"
        torch.save({"state_dict": {"shared_core.placeholder": torch.tensor(1.0)},
                    "agent_config": multi_config}, multi_checkpoint)
        multi_calls = []

        def capture_multi(agent_cfg, _recursive_):
            assert _recursive_ is False
            value = OmegaConf.to_container(agent_cfg, resolve=True)
            multi_calls.append(value)
            selected = Profile.from_config(value["profiles"]["b2d"])
            policy = TinyPolicy(selected)
            policy.initialize = lambda strict=False: None
            return policy

        with patch("unidrivevla.serve.instantiate", side_effect=capture_multi):
            multi_runtime = load_runtime("b2d", str(multi_checkpoint), "cpu", None, None, "ego")
        assert multi_runtime.profile.name == "b2d"
        assert multi_calls[-1]["checkpoint_frame"] == {"b2d": "ego"}

        profile_objects = {name: Profile.from_config(value) for name, value in profiles.items()}

        class MultiFrameShell:
            checkpoint_path = str(multi_checkpoint)
            checkpoint_frame = {"b2d": "ego"}
            profiles = profile_objects
            loaded = False

            def _load_multidomain_state(self, state, checkpoint, strict=False):
                self.loaded = True

        multi_validator = MultiFrameShell()
        MultiProfilePolicy.initialize(multi_validator, strict=True)
        assert multi_validator.loaded

        # HUGSIM's public entrypoint uses the shared loader: an environment frame default remains scoped to this
        # profile, while a CLI choice wins and a legacy stage-2 motion setting reaches the real initializer.
        output = root / "hugsim-output"
        output.mkdir()
        camera_cfg, kinematic_cfg = root / "cams.yaml", root / "vehicle.yaml"
        camera_cfg.write_text("stub", encoding="utf-8")
        kinematic_cfg.write_text("stub", encoding="utf-8")
        fifo_calls = []
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir),
                                    "UNIDRIVE_CHECKPOINT_FRAME": "ego", "UNIDRIVE_WITH_MOTION": "1"}), \
                patch("unidrivevla.serve.instantiate", side_effect=tiny_instantiate), \
                patch.object(hugsim_client, "load_calibration", return_value=(np.eye(4), 2.7, set(CAMERA_ORDER))), \
                patch.object(hugsim_client, "HUGSIMInputAdapter", return_value=object()), \
                patch.object(hugsim_client, "run_fifo", side_effect=lambda out, runtime, adapter, timeout:
                             fifo_calls.append((out, runtime, adapter, timeout)) or 0):
            result = hugsim_client.main(["--output", str(output), "--checkpoint", str(legacy), "--profile", "nuscenes",
                                         "--device", "cpu", "--camera-config", str(camera_cfg),
                                         "--kinematic-config", str(kinematic_cfg)])
        assert result == 0 and len(fifo_calls) == 1
        assert fifo_calls[-1][1].profile.frame == "ego" and fifo_calls[-1][1].policy.with_motion
        assert fifo_calls[-1][1].policy.motion.weight.item() == 2.0

        # Explicit CLI frame takes precedence over the environment default and selects the native preset.
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir),
                                    "UNIDRIVE_CHECKPOINT_FRAME": "ego", "UNIDRIVE_WITH_MOTION": ""}), \
                patch("unidrivevla.serve.instantiate", side_effect=tiny_instantiate), \
                patch.object(hugsim_client, "load_calibration", return_value=(np.eye(4), 2.7, set(CAMERA_ORDER))), \
                patch.object(hugsim_client, "HUGSIMInputAdapter", return_value=object()), \
                patch.object(hugsim_client, "run_fifo", side_effect=lambda out, runtime, adapter, timeout:
                             fifo_calls.append((out, runtime, adapter, timeout)) or 0):
            result = hugsim_client.main(["--output", str(output), "--checkpoint", str(native_legacy), "--profile", "nuscenes",
                                         "--device", "cpu",
                                         "--checkpoint-frame", "lidar", "--camera-config", str(camera_cfg),
                                         "--kinematic-config", str(kinematic_cfg)])
        assert result == 0 and fifo_calls[-1][1].profile.frame == "lidar"
        assert not fifo_calls[-1][1].policy.with_motion

        # Saved metadata remains authoritative when HUGSIM environment defaults request another frame or stage.
        saved_ego = Profile.from_config(OmegaConf.load(profile_dir / "nuscenes_ego.yaml"))
        metadata_ego = root / "ego-with-metadata.ckpt"
        torch.save({"state_dict": {"weight": torch.tensor(6.0)},
                    "agent_config": config_for(saved_ego, with_motion=False)}, metadata_ego)
        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir),
                                    "UNIDRIVE_CHECKPOINT_FRAME": "ego", "UNIDRIVE_WITH_MOTION": ""}), \
                patch("unidrivevla.serve.instantiate", side_effect=AssertionError("metadata mismatch must fail early")), \
                patch.object(hugsim_client, "load_calibration", return_value=(np.eye(4), 2.7, set(CAMERA_ORDER))), \
                patch.object(hugsim_client, "HUGSIMInputAdapter", return_value=object()), \
                patch.object(hugsim_client, "run_fifo", return_value=0):
            try:
                hugsim_client.main(["--output", str(output), "--checkpoint", str(metadata_ego), "--profile", "nuscenes",
                                    "--device", "cpu",
                                    "--checkpoint-frame", "lidar", "--camera-config", str(camera_cfg),
                                    "--kinematic-config", str(kinematic_cfg)])
            except ValueError as exc:
                assert "disagrees with checkpoint profile metadata" in str(exc)
            else:
                raise AssertionError("HUGSIM CLI frame override replaced authoritative checkpoint metadata")

        with patch.dict(os.environ, {"VLM_PRETRAINED_PATH": str(model_dir),
                                    "UNIDRIVE_CHECKPOINT_FRAME": "ego", "UNIDRIVE_WITH_MOTION": "1"}), \
                patch("unidrivevla.serve.instantiate", side_effect=AssertionError("motion mismatch must fail early")), \
                patch.object(hugsim_client, "load_calibration", return_value=(np.eye(4), 2.7, set(CAMERA_ORDER))), \
                patch.object(hugsim_client, "HUGSIMInputAdapter", return_value=object()), \
                patch.object(hugsim_client, "run_fifo", return_value=0):
            try:
                hugsim_client.main(["--output", str(output), "--checkpoint", str(metadata_ego), "--profile", "nuscenes",
                                    "--device", "cpu",
                                    "--camera-config", str(camera_cfg), "--kinematic-config", str(kinematic_cfg)])
            except ValueError as exc:
                assert "requested motion architecture differs" in str(exc)
            else:
                raise AssertionError("HUGSIM motion default overrode saved checkpoint architecture")

        # The public server CLI carries both frame and the existing stage-2 architecture choice to the loader.
        serve_config = root / "serve.yaml"
        OmegaConf.save(OmegaConf.create({"profile": "nuscenes", "checkpoint": str(native_legacy),
                                         "with_motion": None, "checkpoint_frame": None, "device": "cpu",
                                         "host": "127.0.0.1", "port": 9000}), serve_config)
        runtime_marker = object()
        with patch("unidrivevla.serve.load_runtime", return_value=runtime_marker) as load, \
                patch("unidrivevla.serve.serve") as run_server:
            serve_main(["--config", str(serve_config), "--checkpoint-frame", "ego", "--with-motion"])
        assert load.call_args.args == ("nuscenes", str(native_legacy), "cpu", True, None, "ego")
        run_server.assert_called_once_with(runtime_marker, "127.0.0.1", 9000)
        assert OmegaConf.select(OmegaConf.load(package / "config" / "serve.yaml"), "checkpoint_frame") is None
    print("OK: legacy frame selection, metadata authority, stage-2 serving, multi-profile frame scope, and both public CLIs")


def main():
    profile_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "profile" / "nuscenes.yaml")
    profile = Profile.from_config(profile_cfg)
    _check_checkpoint_config(profile)
    _check_checkpoint_frame_compatibility(profile)
    # Keep this check explicit about canonical EGO mode; native LiDAR conversion has its own regression.
    canonical_profile = replace(profile, frame="ego")
    _check_neuro_ncap(canonical_profile)
    _check_hugsim(canonical_profile)


if __name__ == "__main__":
    main()
