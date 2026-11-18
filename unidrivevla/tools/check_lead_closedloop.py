"""CPU-only LEAD protocol and canonical-feature regression; no CARLA score is generated."""

import copy
import http.client
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import tempfile
from threading import Thread
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unidrivevla.closedloop.history import PoseHistory
from unidrivevla.closedloop.lead_protocol import (
    LEAD_CAMERA_VIEWS,
    LEAD_HORIZON,
    LEAD_POSE_INTERVAL_S,
    LEAD_TEMPORAL_RING_SIZE,
    LeadHTTPClient,
    decode_request,
    decode_trajectory,
    encode_request,
    encode_trajectory,
)


def _pose(x, y, z, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    qz, qw = np.sin(yaw / 2), np.cos(yaw / 2)
    return SimpleNamespace(array=np.array([x, y, z, 0, 0, qz, qw]), rotation_matrix=rotation)


class _CameraMetadata:
    def __init__(self, index):
        self.k = np.array([[310.0 + index, 0, 11.5], [0, 320.0 + index, 7.5], [0, 0, 1]], dtype=np.float64)
        self.camera_to_imu_se3 = _pose(0.5 + index * 0.1, -0.2 + index * 0.03, 1.7, index * 0.08)

    def project_to_image(self, points):
        points = np.asarray(points, dtype=np.float64)
        pixels = points[:, :2] / points[:, 2:3] @ self.k[:2, :2].T + self.k[:2, 2]
        return pixels, np.ones(len(points), dtype=bool)


class _OnlineScene:
    @property
    def ego_state(self):
        raise AssertionError("inference must never read LEAD's privileged ego_state/imu_se3")

    @property
    def future_ego_states(self):
        raise AssertionError("inference must never read future LEAD labels")


def _scene_fixture():
    scene = _OnlineScene()
    current_pose = np.eye(4, dtype=np.float64)
    current_pose[:2, 3] = [3.0, -1.0]
    scene.driving_meta = {"localized_ego_state_se3": current_pose.copy()}
    scene.log_metadata = SimpleNamespace(log_name="synthetic-online")
    scene.cameras = []
    y, x = np.indices((16, 24))
    for index, view in enumerate(LEAD_CAMERA_VIEWS):
        image = np.stack(((x + index) % 256, (y + 30 * index) % 256,
                          (x + y + 60 * index) % 256), axis=-1).astype(np.uint8)
        scene.cameras.append(SimpleNamespace(
            image=image,
            metadata=_CameraMetadata(index),
            timestamp=SimpleNamespace(time_us=1_500_000 + index * 1_000),
        ))
    # Current-to-oldest localized poses at 20 Hz, as BaseAgent.build_scene_data exposes them.
    ages = np.arange(31, dtype=np.float64) * 0.05
    scene.past_ego_positions = np.column_stack((-2.0 * ages, np.zeros_like(ages)))
    scene.past_ego_yaws = np.zeros(31, dtype=np.float64)
    return scene, current_pose


def _check_feature_builder_and_history():
    from unidrivevla.data.lead import lead_inference_features

    scene, pose = _scene_fixture()
    features = lead_inference_features(scene, ego2global=pose, timestamp_seconds=1.5)
    assert tuple(features["cameras"]) == LEAD_CAMERA_VIEWS
    assert features["history_trajectory"].shape == (4, 3)
    assert features["command"] == "unknown" and features["timestamp"] == 1.5
    assert np.allclose(features["ego_velocity"][:2], [2.0, 0.0], atol=1e-5)
    assert np.allclose(features["ego_acceleration"][:2], [0.0, 0.0], atol=1e-5)
    assert abs(float(features["ego_angular_velocity"][2])) < 1e-5
    assert np.allclose(features["ego2global"], pose)

    history = PoseHistory()
    for index in range(31):
        sample_pose = pose.copy()
        sample_pose[0, 3] -= 2.0 * (1.5 - index * 0.05)
        runtime_history = history.history(index * 0.05, sample_pose)
    assert np.allclose(features["history_trajectory"], runtime_history, atol=1e-5)

    for index, view in enumerate(LEAD_CAMERA_VIEWS):
        camera = features["cameras"][view]
        assert np.array_equal(camera["image"], scene.cameras[index].image)
        assert np.allclose(camera["intrinsic"], scene.cameras[index].metadata.k)
        transform = camera["sensor2ego"]
        assert np.allclose(transform[:3, :3], scene.cameras[index].metadata.camera_to_imu_se3.rotation_matrix)
        assert np.allclose(transform[:3, 3], scene.cameras[index].metadata.camera_to_imu_se3.array[:3])
    return features, pose


def _check_warmup_history(scene, pose):
    from unidrivevla.data.lead import lead_inference_features

    full_positions = scene.past_ego_positions
    full_yaws = scene.past_ego_yaws
    positions = np.asarray(full_positions[:2], dtype=np.float64)
    yaws = np.asarray(full_yaws[:2], dtype=np.float64)
    scene.past_ego_positions, scene.past_ego_yaws = positions, yaws
    try:
        lead_inference_features(scene, ego2global=pose, timestamp_seconds=1.5)
    except ValueError as exc:
        assert "1.5 seconds" in str(exc)
    else:
        raise AssertionError("LEAD online features accepted less than the trained history window")
    scene.past_ego_positions, scene.past_ego_yaws = full_positions, full_yaws
    full_features = lead_inference_features(scene, ego2global=pose, timestamp_seconds=1.5)
    assert full_features["history_trajectory"].shape == (4, 3)
    assert np.isfinite(full_features["history_trajectory"]).all()
    # Keep null status serialization covered for fields unavailable to this source.
    unknown_status = dict(full_features)
    unknown_status["ego_acceleration"] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    decoded = decode_request(json.loads(encode_request(unknown_status, pose, 1.5)))
    assert np.isnan(decoded["ego_acceleration"][:2]).all()
    return full_features


def _check_runtime_prepare(features, pose):
    import torch
    from omegaconf import OmegaConf

    from unidrivevla.adapters.unidrivevla import Profile
    from unidrivevla.closedloop.runtime import PolicyRuntime

    package = Path(__file__).resolve().parents[1]
    profile_config = OmegaConf.load(package / "config" / "profile" / "lead.yaml")
    profile = Profile.from_config(OmegaConf.to_container(profile_config, resolve=True))

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.profile = profile
            self.counter = 0
            self.seen = []
            self.batch = None
            self.head = SimpleNamespace(reset_temporal=self.reset_temporal)

        def reset_temporal(self):
            self.counter = 0

        def training_runtime_state_dict(self):
            return {"counter": self.counter}

        def load_training_runtime_state_dict(self, state, device=None):
            self.counter = state["counter"]

        def forward(self, batch):
            self.batch = batch
            assert np.allclose(batch["hist_traj"][0].cpu().numpy(), features["history_trajectory"][:, :2])
            self.counter += 1
            self.seen.append(self.counter)
            trajectory = torch.zeros((1, LEAD_HORIZON, 3), dtype=torch.float32, device=batch["img"].device)
            return {"pred_traj": trajectory}

    policy = TinyPolicy()
    runtime = PolicyRuntime(policy, device="cpu")
    runtime.enable_temporal_ring(LEAD_TEMPORAL_RING_SIZE)
    for step in range(11):
        current_pose = pose.copy()
        current_pose[0, 3] += 2.0 * step * 0.05
        plan = runtime.predict(
            cameras=features["cameras"], ego2global=current_pose, timestamp=1.5 + step * 0.05,
            command="unknown", ego_velocity=features["ego_velocity"],
            ego_acceleration=features["ego_acceleration"], ego_angular_velocity=features["ego_angular_velocity"],
            ego_steer=features["ego_steer"], history_trajectory=features["history_trajectory"],
        )
        assert plan.shape == (LEAD_HORIZON, 2)
    assert policy.seen == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    assert policy.batch["ego_status"].shape == (1, 4)
    assert np.allclose(policy.batch["ego_status"].cpu().numpy()[0], [2.0, 0.0, 0.0, 0.0])
    runtime.reset()
    assert runtime._ring.step == 0 and runtime._ring.slots == [None] * LEAD_TEMPORAL_RING_SIZE
    assert policy.counter == 0
    reset_pose = pose.copy()
    runtime.predict(cameras=features["cameras"], ego2global=reset_pose, timestamp=0.0, command="unknown",
                    ego_velocity=features["ego_velocity"], ego_acceleration=features["ego_acceleration"],
                    ego_angular_velocity=features["ego_angular_velocity"], ego_steer=features["ego_steer"],
                    history_trajectory=features["history_trajectory"])
    assert policy.seen[-1] == 1 and runtime._ring.step == 1


class _Runtime:
    def __init__(self):
        self.profile = SimpleNamespace(name="lead", horizon=LEAD_HORIZON, pose_interval=LEAD_POSE_INTERVAL_S)
        self.last = None
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict(self, **inputs):
        self.last = inputs
        return np.column_stack((np.arange(1, LEAD_HORIZON + 1), np.arange(LEAD_HORIZON) * 0.1))


def _check_http_roundtrip(features, pose):
    from unidrivevla.closedloop.lead_server import _handler

    runtime = _Runtime()
    server = HTTPServer(("127.0.0.1", 0), _handler(runtime))
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    client = LeadHTTPClient(f"http://127.0.0.1:{server.server_port}", timeout=5)
    try:
        client.reset()
        plan = client.predict(features, pose, 1.5, command="unknown")
        assert plan.shape == (LEAD_HORIZON, 2)
        assert np.allclose(plan[:, 0], np.arange(1, LEAD_HORIZON + 1))
        assert runtime.reset_count == 1
        assert runtime.last["command"] == "unknown" and runtime.last["timestamp"] == 1.5
        assert np.allclose(runtime.last["history_trajectory"], features["history_trajectory"])
        assert np.allclose(runtime.last["ego2global"], pose)
        for view in LEAD_CAMERA_VIEWS:
            assert np.array_equal(runtime.last["cameras"][view]["image"], features["cameras"][view]["image"])
            assert np.allclose(runtime.last["cameras"][view]["sensor2ego"], features["cameras"][view]["sensor2ego"])
        invalid = json.loads(encode_request(features, pose, 1.5))
        invalid["ego_acceleration"][0] = 1e100
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("POST", "/infer", json.dumps(invalid, separators=(",", ":")),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
        duplicate = b'{"protocol":"x","protocol":"x"}'
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("POST", "/infer", duplicate, {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        runtime.reset()


def _check_rejections(features, pose):
    body = encode_request(features, pose, 1.5)
    decoded = json.loads(body)
    assert decoded["pose_interval_s"] == LEAD_POSE_INTERVAL_S
    assert decode_trajectory({"trajectory": np.zeros((LEAD_HORIZON, 2)).tolist(),
                              "pose_interval_s": LEAD_POSE_INTERVAL_S}).shape == (LEAD_HORIZON, 2)
    invalid = copy.deepcopy(decoded)
    invalid["ego_velocity"][0] = None
    try:
        decode_request(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("missing observed longitudinal velocity was accepted")
    invalid = copy.deepcopy(decoded)
    invalid["history_trajectory"] = [[0.0, 0.0]]
    try:
        decode_request(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("missing trained history poses were accepted")
    invalid = copy.deepcopy(decoded)
    invalid["history_trajectory"][0][0] = float("nan")
    try:
        decode_request(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite history poses were accepted")
    invalid = copy.deepcopy(decoded)
    invalid["ego_acceleration"][0] = 1e100
    try:
        decode_request(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("float32-overflowing status was accepted")
    invalid = copy.deepcopy(decoded)
    invalid["cameras"].pop(LEAD_CAMERA_VIEWS[0])
    try:
        decode_request(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("missing LEAD camera was accepted")
    try:
        decode_trajectory({"trajectory": [[0.0, 0.0]], "pose_interval_s": LEAD_POSE_INTERVAL_S})
    except ValueError:
        pass
    else:
        raise AssertionError("short-horizon model output was accepted")
    try:
        encode_trajectory(np.full((LEAD_HORIZON, 2), np.inf))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite trajectory was accepted")


def _check_official_runner():
    from unidrivevla.closedloop import lead_runner
    from unidrivevla.closedloop.lead_runner import _verify_lead_checkout

    def as_frontend_python(version, function, **kwargs):
        frontend_sys = SimpleNamespace(version_info=version, executable=sys.executable)
        with patch.object(lead_runner, "sys", frontend_sys):
            return function(**kwargs)

    def git_stub(sha, tracked="", submodules="  abcdef 3rd_party/leaderboard\n"):
        def run(command, **_kwargs):
            if command[-2:] == ["rev-parse", "HEAD"]:
                return SimpleNamespace(returncode=0, stdout=sha)
            if "submodule" in command:
                return SimpleNamespace(returncode=0, stdout=submodules)
            if "status" in command:
                return SimpleNamespace(returncode=0, stdout=tracked)
            return SimpleNamespace(returncode=1, stdout="")
        return run

    assert _verify_lead_checkout("fixture", "standard",
                                 runner=git_stub(lead_runner.LEAD_MAIN_SHA)) == lead_runner.LEAD_MAIN_SHA
    for sha, tracked, submodules in (
        ("wrong", "", "  abcdef 3rd_party/leaderboard\n"),
        (lead_runner.LEAD_MAIN_SHA, " M src/lead/agent.py\n", "  abcdef 3rd_party/leaderboard\n"),
        (lead_runner.LEAD_MAIN_SHA, "", "-abcdef 3rd_party/leaderboard\n"),
    ):
        try:
            _verify_lead_checkout("fixture", "standard", runner=git_stub(sha, tracked, submodules))
        except RuntimeError:
            pass
        else:
            raise AssertionError("unpinned/modified LEAD source or evaluator submodule was accepted")

    with tempfile.TemporaryDirectory(prefix="lead-runner-check-") as temporary:
        root = Path(temporary)
        lead_root, carla_root, code_root = root / "lead", root / "carla", root / "code"
        routes = root / "town13_0.xml"
        agent_config = root / "agent.yaml"
        routes.write_text('<routes><route id="town13_0"><scenario type="NoScenario" /></route></routes>',
                          encoding="utf-8")
        agent_config.write_text("server_url: http://127.0.0.1:9001\ntimeout: 60\n", encoding="utf-8")
        (carla_root / "PythonAPI" / "carla").mkdir(parents=True)
        agent = code_root / "unidrivevla" / "closedloop" / "lead.py"
        agent.parent.mkdir(parents=True)
        agent.write_text("# test fixture\n", encoding="utf-8")
        for variant in ("standard", "bench2drive"):
            evaluator = lead_root / "3rd_party" / "leaderboard" / variant / "leaderboard" \
                / "leaderboard" / "leaderboard_evaluator.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("# test fixture\n", encoding="utf-8")
            (lead_root / "3rd_party" / "leaderboard" / variant / "scenario_runner").mkdir(parents=True)

        shared = dict(lead_root=lead_root, carla_root=carla_root, routes=routes, agent_config=agent_config,
                      code_root=code_root)
        try:
            as_frontend_python((3, 9, 18), lead_runner.build_invocation,
                               **shared, output_dir=root / "py39", mode="standard")
        except RuntimeError as exc:
            assert "Python 3.10-3.12" in str(exc)
        else:
            raise AssertionError("Python 3.9 was accepted for the official LEAD frontend")

        with patch("unidrivevla.closedloop.lead_runner._verify_lead_checkout",
                   return_value=lead_runner.LEAD_MAIN_SHA):
            standard_argv, standard_env, _, _, _ = as_frontend_python(
                (3, 10, 0), lead_runner.build_invocation,
                **shared, output_dir=root / "standard-out", mode="standard")
            assert standard_argv[1].replace("\\", "/").endswith(
                "standard/leaderboard/leaderboard/leaderboard_evaluator.py")
            assert standard_env["IS_BENCH2DRIVE"] == "0" and standard_env["SCENARIO_TYPE"] == "NoScenario"
            assert standard_env["BENCHMARK_ROUTE_ID"] == "town13"
            assert standard_argv[standard_argv.index("--track") + 1] == "SENSORS"
            assert standard_argv[standard_argv.index("--agent") + 1] == str(agent)
            b2d_argv, b2d_env, _, _, _ = as_frontend_python(
                (3, 10, 0), lead_runner.build_invocation,
                **shared, output_dir=root / "b2d-out", mode="bench2drive")
            assert b2d_argv[1].replace("\\", "/").endswith(
                "bench2drive/leaderboard/leaderboard/leaderboard_evaluator.py")
            assert b2d_env["IS_BENCH2DRIVE"] == "1"

            def success_runner(command, cwd, env, check):
                Path(command[command.index("--checkpoint") + 1]).write_text('{"official":true}', encoding="utf-8")
                return SimpleNamespace(returncode=0)

            assert as_frontend_python((3, 10, 0), lead_runner.run_evaluation,
                                      **shared, output_dir=root / "valid", runner=success_runner) == 0

            with patch("unidrivevla.closedloop.lead_runner._verify_lead_checkout",
                       side_effect=RuntimeError("unpinned")):
                try:
                    as_frontend_python((3, 10, 0), lead_runner.build_invocation,
                                       **shared, output_dir=root / "unpinned")
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("unpinned LEAD source revision was accepted")

            def failed_runner(command, cwd, env, check):
                return SimpleNamespace(returncode=17)

            assert as_frontend_python((3, 10, 0), lead_runner.run_evaluation,
                                      **shared, output_dir=root / "failed", runner=failed_runner) == 17

            def missing_score_runner(command, cwd, env, check):
                return SimpleNamespace(returncode=0)

            try:
                as_frontend_python((3, 10, 0), lead_runner.run_evaluation,
                                   **shared, output_dir=root / "missing", runner=missing_score_runner)
            except RuntimeError:
                pass
            else:
                raise AssertionError("zero exit without official score JSON was accepted")

            def invalid_score_runner(command, cwd, env, check):
                Path(command[command.index("--checkpoint") + 1]).write_text("not-json", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            try:
                as_frontend_python((3, 10, 0), lead_runner.run_evaluation,
                                   **shared, output_dir=root / "invalid", runner=invalid_score_runner)
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid official score JSON was accepted")

            reused = root / "existing"
            reused.mkdir()
            (reused / "keep.json").write_text('{"preserve":true}', encoding="utf-8")
            try:
                as_frontend_python((3, 10, 0), lead_runner.run_evaluation,
                                   **shared, output_dir=reused, runner=failed_runner)
            except FileExistsError:
                pass
            else:
                raise AssertionError("existing LEAD outputs were not protected")
            assert (reused / "keep.json").read_text(encoding="utf-8") == '{"preserve":true}'


def main():
    features, pose = _check_feature_builder_and_history()
    scene, pose = _scene_fixture()
    full_features = _check_warmup_history(scene, pose)
    _check_rejections(features, pose)
    _check_http_roundtrip(features, pose)
    _check_runtime_prepare(full_features, pose)
    _check_official_runner()
    print("PASS: LEAD localized feature builder, six calibrated cameras, history padding, strict HTTP RPC, and 8x0.25s plan")
    print("Synthetic CPU protocol check only; no CARLA rollout or official benchmark score was produced.")


if __name__ == "__main__":
    main()
