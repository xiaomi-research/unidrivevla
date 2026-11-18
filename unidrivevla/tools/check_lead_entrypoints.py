"""Exercise the LEAD agent and launcher with fake official modules; no CARLA rollout or weights required."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    module.__path__ = []
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
    return module


def _pose(x, y, z, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    qz, qw = np.sin(yaw / 2), np.cos(yaw / 2)
    return types.SimpleNamespace(
        array=np.array([x, y, z, 0.0, 0.0, qz, qw]), rotation_matrix=rotation,
    )


class _CameraMetadata:
    def __init__(self, index):
        self.k = np.array([[300.0 + index, 0.0, 16.0], [0.0, 310.0 + index, 8.0], [0, 0, 1]], dtype=np.float64)
        self.camera_to_imu_se3 = _pose(0.5 + index * 0.1, -0.2, 1.7, index * 0.02)

    def project_to_image(self, points):
        points = np.asarray(points, dtype=np.float64)
        pixels = points[:, :2] / points[:, 2:3] @ self.k[:2, :2].T + self.k[:2, 2]
        return pixels, np.ones(len(points), dtype=bool)


class _OnlineScene:
    @property
    def ego_state(self):
        raise AssertionError("LEAD inference must not read privileged ego_state/imu_se3")

    @property
    def future_ego_states(self):
        raise AssertionError("LEAD inference must not read future ground-truth states")


def _install_official_stubs():
    controls = []

    class VehicleControl:
        def __init__(self, *, steer, throttle, brake):
            self.steer, self.throttle, self.brake = steer, throttle, brake
            controls.append(self)

    carla = _module("carla", VehicleControl=VehicleControl)

    class _AgentAPI:
        def sensors(self):
            return list(getattr(self.policy, "input_cameras", []))

        def tick(self, sensor_data):
            self.compass = float(sensor_data["compass"])
            self.tick_count += 1
            return sensor_data

        def build_scene_data(self, _sensor_data):
            scene = _OnlineScene()
            scene.log_metadata = types.SimpleNamespace(log_name="entrypoint-fixture")
            ages = np.arange(self.tick_count, dtype=np.float64) * 0.05
            scene.past_ego_positions = np.column_stack((-2.0 * ages, np.zeros(len(ages))))
            scene.past_ego_yaws = np.zeros(self.tick_count, dtype=np.float64)
            scene.cameras = []
            y, x = np.indices((16, 32))
            for index, metadata in enumerate(self._camera_metadatas):
                image = np.stack(((x + index) % 255, (y + 10 * index) % 255,
                                  (x + y + 20 * index) % 255), axis=-1).astype(np.uint8)
                scene.cameras.append(types.SimpleNamespace(
                    image=image,
                    metadata=metadata,
                    timestamp=types.SimpleNamespace(time_us=1_000_000 + index * 1_000),
                ))
            return scene

    class BaseAgent:
        @staticmethod
        def setup(agent, config, *, sensor_agent, past_window_num_iterations):
            agent.base_setup = (config, sensor_agent, past_window_num_iterations)
            agent.compass = 0.0
            agent.tick_count = 0

    class _Hero:
        def get_world(self):
            return "fake-world"

    class CarlaDataProvider:
        @staticmethod
        def get_hero_actor():
            return _Hero()

    srunner = _module("srunner")
    scenariomanager = _module("srunner.scenariomanager")
    provider_module = _module("srunner.scenariomanager.carla_data_provider", CarlaDataProvider=CarlaDataProvider)
    srunner.scenariomanager = scenariomanager
    scenariomanager.carla_data_provider = provider_module

    cfg = types.SimpleNamespace(
        expert=types.SimpleNamespace(
            sensor_rig=types.SimpleNamespace(cameras=[object() for _ in range(6)]),
            simulation=types.SimpleNamespace(carla_fps=20, inital_frames_delay=0),
            data_collection=types.SimpleNamespace(data_save_freq=5),
        )
    )
    camera_views = {
        1: "front_left", 2: "front", 3: "front_right",
        4: "back_right", 5: "back", 6: "back_left",
    }

    def build_camera_metadata(expert, _vehicle_metadata, *, perturbation_translation, perturbation_rotation):
        assert len(expert.sensor_rig.cameras) == 6
        assert perturbation_translation == perturbation_rotation == 0.0
        return [_CameraMetadata(index) for index in range(6)]

    def localized_pose_to_se3_matrix(position, compass):
        x, y, z = map(float, position)
        c, s = np.cos(compass), np.sin(compass)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:2, :2] = [[c, -s], [s, c]]
        matrix[:3, 3] = [x, y, z]
        return matrix

    carla_to_123d = types.SimpleNamespace(build_pinhole_camera_metadatas=build_camera_metadata)
    autonomous_agent = types.SimpleNamespace(Track=types.SimpleNamespace(SENSORS="SENSORS"))
    _module("leaderboard")
    _module("leaderboard.autoagents", autonomous_agent=autonomous_agent)
    _module("lead")
    _module("lead.api")
    _module("lead.api.abstract_driving_agent", AbstractDrivingAgent=_AgentAPI)
    _module(
        "lead.api.py123d_log_api",
        CAMERA_ID_BY_LEAD_INDEX=camera_views,
        CARLA_LINCOLN_MKZ_2020_METADATA=object(),
        localized_pose_to_se3_matrix=localized_pose_to_se3_matrix,
    )
    _module("lead.common", carla_to_123d=carla_to_123d)
    _module("lead.common.base_agent", BaseAgent=BaseAgent)
    _module("lead.config", load_lead_config=lambda: cfg)

    tracker_instances = []

    class WaypointTracker:
        def __init__(self, lead_config):
            assert lead_config is cfg
            self.calls = []
            tracker_instances.append(self)

        def step(self, plan, speed):
            assert plan.shape == (1, 8, 2), f"WaypointTracker plan shape was {plan.shape}"
            assert speed.shape == (1, 1), f"WaypointTracker speed shape was {speed.shape}"
            self.calls.append((plan.array.copy(), speed.array.copy()))
            return types.SimpleNamespace(steer=0.25, throttle=0.4, brake=0.05)

    _module("lead.evaluation")
    _module("lead.evaluation.inference")
    _module("lead.evaluation.inference.trackers", WaypointTracker=WaypointTracker)
    _module("py123d")
    _module("py123d.datatypes")
    _module("py123d.datatypes.sensors")
    _module("py123d.datatypes.sensors.base_camera", CameraID=types.SimpleNamespace())

    return cfg, carla, controls, tracker_instances


def _check_agent_entrypoint():
    cfg, carla, controls, trackers = _install_official_stubs()
    lead_module = importlib.import_module("unidrivevla.closedloop.lead")
    lead_module = importlib.reload(lead_module)
    client_calls = []

    class FakeClient:
        def __init__(self, url, timeout):
            assert url == "http://127.0.0.1:9811"
            assert timeout == 7.0
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def predict(self, features, ego2global, timestamp, *, command):
            assert command == "unknown"
            assert timestamp == 1.5
            assert np.allclose(ego2global, expected_pose)
            assert features["command"] == "unknown" and features["timestamp"] == timestamp
            assert np.allclose(features["ego2global"], expected_pose)
            assert features["history_trajectory"].shape == (4, 3)
            assert np.isfinite(features["history_trajectory"]).all()
            assert np.allclose(
                features["history_trajectory"],
                [[-3.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            )
            assert tuple(features["cameras"]) == tuple(lead_module.LEAD_CAMERA_VIEWS)
            client_calls.append((features, ego2global.copy(), timestamp, command))
            return np.arange(16, dtype=np.float32).reshape(8, 2)

    lead_module.LeadHTTPClient = FakeClient
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda stream: json.load(stream)
    torch_module = types.ModuleType("torch")
    torch_module.float32 = "float32"

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        @property
        def shape(self):
            return self.array.shape

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.array, axis))

    torch_module.from_numpy = lambda array: FakeTensor(array)
    torch_module.tensor = lambda values, dtype=None: FakeTensor(np.asarray(values, dtype=np.float32))
    sys.modules["yaml"] = yaml_module
    sys.modules["torch"] = torch_module

    with tempfile.TemporaryDirectory(prefix="lead-entrypoint-") as temporary:
        config_path = Path(temporary) / "agent.yaml"
        config_path.write_text(json.dumps({"server_url": "http://127.0.0.1:9811", "timeout": 7.0}), encoding="utf-8")
        agent = lead_module.UniDriveVLALEADAgent()
        agent.setup(str(config_path))
        assert agent.base_setup == (cfg, True, 30)
        assert agent.track == "SENSORS" and len(agent.sensors()) == 6
        assert agent.client.reset_calls == 1
        assert agent.waypoint_tracker is trackers[-1]

        first = agent.run_step({"compass": 0.0}, 0.0)
        assert first.brake == 1.0 and first.throttle == 0.0 and first.steer == 0.0
        assert not client_calls and not trackers[-1].calls

        current = (4.0, -2.0, 0.0)
        compass = 0.2
        c, s = np.cos(compass), np.sin(compass)
        expected_pose = np.eye(4, dtype=np.float64)
        expected_pose[:2, :2] = [[c, -s], [s, c]]
        expected_pose[:3, 3] = current
        sensor_data = {"compass": compass, "localized_position": current, "speed": 7.5}

        # The second observed pose is still short of the official 31-pose history; stay braked and offline.
        warmup = agent.run_step(sensor_data, 0.05)
        assert warmup.brake == 1.0 and warmup.throttle == 0.0 and warmup.steer == 0.0
        assert agent.tick_count == 2 and not client_calls and not trackers[-1].calls

        for expected_observed in range(3, 31):
            warmup = agent.run_step(sensor_data, expected_observed / 20.0)
            assert warmup.brake == 1.0 and warmup.throttle == 0.0 and warmup.steer == 0.0
            assert agent.tick_count == expected_observed and not client_calls and not trackers[-1].calls

        control = agent.run_step(sensor_data, 1.5)
        assert len(client_calls) == 1 and len(trackers[-1].calls) == 1
        plan, speed = trackers[-1].calls[0]
        assert plan.shape == (1, 8, 2) and np.array_equal(plan[0], np.arange(16).reshape(8, 2))
        assert speed.shape == (1, 1) and speed[0, 0] == 7.5
        assert (control.steer, control.throttle, control.brake) == (0.25, 0.4, 0.05)
        assert isinstance(control, carla.VehicleControl)
        assert not agent.lead_config is None


def _fake_checkout(root):
    lead_root = root / "lead"
    carla_root = root / "carla"
    lead_root.joinpath("src", "lead").mkdir(parents=True)
    carla_root.joinpath("PythonAPI").mkdir(parents=True)
    routes = root / "town13_0.xml"
    routes.write_text('<routes><route id="town13_0"><scenario type="NoScenario" /></route></routes>', encoding="utf-8")
    agent_config = root / "agent.yaml"
    agent_config.write_text('server_url: "http://127.0.0.1:9001"\ntimeout: 60\n', encoding="utf-8")
    for variant in ("standard", "bench2drive"):
        variant_root = lead_root / "3rd_party" / "leaderboard" / variant
        evaluator = variant_root / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
        evaluator.parent.mkdir(parents=True, exist_ok=True)
        evaluator.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "checkpoint = Path(args[args.index('--checkpoint') + 1])\n"
            "record = {'argv': sys.argv, 'env': {k: os.environ.get(k) for k in ("
            "'IS_BENCH2DRIVE','SCENARIO_TYPE','BENCHMARK_ROUTE_ID','ROUTE_NUMBER','CUDA_VISIBLE_DEVICES',"
            "'CHECKPOINT_DIR','OUTPUT_DIR','EVALUATION_OUTPUT_DIR','SAVE_PATH','LEADERBOARD_ROOT',"
            "'SCENARIO_RUNNER_ROOT','ROUTES','PYTHONPATH')}, 'executable': sys.executable}\n"
            "Path(os.environ['LEAD_FIXTURE_EVALUATOR_RECORD']).write_text(json.dumps(record), encoding='utf-8')\n"
            "rc = int(os.environ.get('LEAD_FIXTURE_EVALUATOR_RC', '0'))\n"
            "if rc: raise SystemExit(rc)\n"
            "mode = os.environ.get('LEAD_FIXTURE_SCORE_MODE', 'valid')\n"
            "checkpoint.write_text('{invalid' if mode == 'malformed' else json.dumps({'fixture': True}), encoding='utf-8')\n",
            encoding="utf-8",
        )
        (variant_root / "scenario_runner").mkdir(parents=True, exist_ok=True)
    return lead_root, carla_root, routes, agent_config


def _fake_git(bin_dir):
    path = bin_dir / "git"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['LEAD_FIXTURE_GIT_LOG'], 'a', encoding='utf-8') as f: f.write(json.dumps(args) + '\\n')\n"
        "if args[-2:] == ['rev-parse', 'HEAD']:\n"
        " print(os.environ['LEAD_FIXTURE_SHA'])\n"
        "elif 'status' in args:\n"
        " print(os.environ.get('LEAD_FIXTURE_STATUS', ''))\n"
        "elif 'submodule' in args:\n"
        " print('  0123456789abcdef0123456789abcdef01234567 ' + args[-1])\n"
        "else:\n"
        " raise SystemExit(1)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _python_frontend(path, version_info):
    path.write_text(
        f"#!{sys.executable}\n"
        "import os, runpy, subprocess, sys\n"
        "from pathlib import Path\n"
        "if os.name == 'nt':\n"
        " # Simulate the paths and executable lookup seen by the native Linux frontend.\n"
        f" sys.path.insert(0, {str(REPO)!r})\n"
        " for name in ('LEAD_ROOT', 'CARLA_ROOT', 'ROUTES', 'LEAD_AGENT_CONFIG'):\n"
        "  value = os.environ[name]\n"
        "  if value.startswith('/') and value[2:3] == '/':\n"
        "   os.environ[name] = value[1] + ':' + value[2:]\n"
        " run_native = subprocess.run\n"
        " def run_fixture(argv, **kwargs):\n"
        "  if argv[0] == 'git':\n"
        "   argv = [sys.executable, str(Path(__file__).parent / 'bin/git'), *argv[1:]]\n"
        "  return run_native(argv, **kwargs)\n"
        " subprocess.run = run_fixture\n"
        f"sys.version_info = {version_info!r}\n"
        "if sys.argv[1:] != ['-m', 'unidrivevla.closedloop.lead_runner']:\n"
        " raise SystemExit('unexpected LEAD_PYTHON invocation')\n"
        "sys.argv = ['unidrivevla.closedloop.lead_runner']\n"
        "runpy.run_module('unidrivevla.closedloop.lead_runner', run_name='__main__', alter_sys=True)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _check_bash_launcher():
    from unidrivevla.closedloop.lead_runner import LEAD_MAIN_SHA

    bash = shutil.which("bash")
    if not bash:
        raise RuntimeError("bash is required to execute scripts/infer/run_lead.sh")
    launcher = REPO / "unidrivevla" / "scripts" / "infer" / "run_lead.sh"
    assert launcher.is_file()
    with tempfile.TemporaryDirectory(prefix="lead-launcher-") as temporary:
        root = Path(temporary)
        lead_root, carla_root, routes, agent_config = _fake_checkout(root)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        _fake_git(fake_bin)
        git_log = root / "git.jsonl"
        evaluator_record = root / "evaluator.json"
        python_310 = _python_frontend(root / "python-sim-310", (3, 10, 0, "final", 0))
        # On the documented navsim host this is the real Python 3.9 executable; other hosts simulate that gate.
        python_39 = sys.executable if sys.version_info[:2] == (3, 9) else _python_frontend(
            root / "python-sim-39", (3, 9, 0, "final", 0),
        )

        def launch(output_dir, mode, *, python_frontend=python_310, **extra_env):
            env = os.environ.copy()
            env.update({
                "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                "LEAD_ROOT": str(lead_root), "CARLA_ROOT": str(carla_root),
                "ROUTES": str(routes), "LEAD_AGENT_CONFIG": str(agent_config),
                "OUTPUT_DIR": str(output_dir), "LEAD_MODE": mode,
                "LEAD_CUDA_VISIBLE_DEVICES": "7", "LEAD_PYTHON": str(python_frontend),
                "LEAD_FIXTURE_SHA": LEAD_MAIN_SHA, "LEAD_FIXTURE_GIT_LOG": str(git_log),
                "LEAD_FIXTURE_EVALUATOR_RECORD": str(evaluator_record),
            })
            env.update(extra_env)
            return subprocess.run([bash, str(launcher)], cwd=REPO, env=env, text=True, capture_output=True)

        for mode, variant, flag in (
            ("standard", "standard", "0"), ("bench2drive", "bench2drive", "1"),
        ):
            output = root / f"out-{variant}"
            result = launch(output, mode)
            assert result.returncode == 0, result.stderr
            record = json.loads(evaluator_record.read_text(encoding="utf-8"))
            argv, env = record["argv"], record["env"]
            assert Path(argv[0]).resolve() == (
                lead_root / "3rd_party" / "leaderboard" / variant / "leaderboard" / "leaderboard"
                / "leaderboard_evaluator.py"
            ).resolve()
            assert argv[argv.index("--routes") + 1] == str(routes.resolve())
            assert argv[argv.index("--track") + 1] == "SENSORS"
            assert argv[argv.index("--agent") + 1] == str((REPO / "unidrivevla" / "closedloop" / "lead.py").resolve())
            assert argv[argv.index("--agent-config") + 1] == str(agent_config.resolve())
            for option, expected in (
                ("--debug", "0"), ("--resume", "0"), ("--port", "2000"),
                ("--traffic-manager-port", "8000"), ("--traffic-manager-seed", "0"),
                ("--repetitions", "1"), ("--timeout", "180.0"), ("--record", "None"),
            ):
                assert argv[argv.index(option) + 1] == expected
            assert record["executable"] == sys.executable
            assert env["IS_BENCH2DRIVE"] == flag and env["SCENARIO_TYPE"] == "NoScenario"
            assert env["BENCHMARK_ROUTE_ID"] == env["ROUTE_NUMBER"] == "town13"
            assert env["CUDA_VISIBLE_DEVICES"] == "7" and env["CHECKPOINT_DIR"] is None
            assert env["OUTPUT_DIR"] == env["EVALUATION_OUTPUT_DIR"] == env["SAVE_PATH"] == str(output.resolve())
            pythonpath = env["PYTHONPATH"].split(os.pathsep)
            assert pythonpath[:6] == [
                str(REPO.resolve()), str((lead_root / "src").resolve()),
                str((carla_root / "PythonAPI").resolve()),
                str((carla_root / "PythonAPI" / "carla").resolve()),
                str((lead_root / "3rd_party" / "leaderboard" / variant / "leaderboard").resolve()),
                str((lead_root / "3rd_party" / "leaderboard" / variant / "scenario_runner").resolve()),
            ]
            assert env["LEADERBOARD_ROOT"] == pythonpath[4]
            assert env["SCENARIO_RUNNER_ROOT"] == pythonpath[5]

        git_commands = [json.loads(line) for line in git_log.read_text(encoding="utf-8").splitlines()]
        checked_submodules = [args[-1] for args in git_commands if "submodule" in args]
        assert checked_submodules == [
            "3rd_party/leaderboard/standard", "3rd_party/leaderboard/bench2drive",
        ]

        failed = launch(root / "out-failed", "standard", LEAD_FIXTURE_EVALUATOR_RC="19")
        assert failed.returncode == 19
        assert not (root / "out-failed" / "checkpoint_endpoint.json").exists()

        malformed = launch(root / "out-malformed", "standard", LEAD_FIXTURE_SCORE_MODE="malformed")
        assert malformed.returncode == 2 and "valid score JSON" in malformed.stderr

        preserved = root / "out-preserved"
        preserved.mkdir()
        keep = preserved / "keep.json"
        keep.write_text('{"keep":true}', encoding="utf-8")
        refusal = launch(preserved, "standard")
        assert refusal.returncode == 2 and "refusing to reuse" in refusal.stderr
        assert keep.read_text(encoding="utf-8") == '{"keep":true}'

        old_python = launch(root / "out-py39", "standard", python_frontend=python_39)
        assert old_python.returncode == 2 and "requires Python 3.10-3.12" in old_python.stderr


def main():
    _check_agent_entrypoint()
    _check_bash_launcher()
    print("PASS: LEAD setup/first brake/next-frame helper and tracker mapping; bash launcher through pinned fake evaluator")
    print("Synthetic CPU frontend and launcher checks only; no CARLA rollout or official score was produced.")


if __name__ == "__main__":
    main()
