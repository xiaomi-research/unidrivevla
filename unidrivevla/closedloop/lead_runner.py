"""Safe launcher helpers for the evaluator selected by LEAD main at the pinned revision."""

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

LEAD_MAIN_SHA = "3764f79698922896638638df8062f2e064034610"


def _checkout_sha(root, runner=subprocess.run):
    result = runner(["git", "-C", str(root), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"could not read LEAD checkout revision at {root}")
    return result.stdout.strip()


def _verify_lead_checkout(root, variant, runner=subprocess.run):
    sha = _checkout_sha(root, runner)
    if sha != LEAD_MAIN_SHA:
        raise RuntimeError(f"LEAD checkout is {sha!r}; expected pinned main {LEAD_MAIN_SHA}")
    tracked = runner(["git", "-C", str(root), "status", "--porcelain", "--", "src/lead"],
                     capture_output=True, text=True, check=False)
    if tracked.returncode != 0:
        raise RuntimeError(f"could not verify tracked LEAD source status at {root}")
    if tracked.stdout.strip():
        raise RuntimeError("refusing modified or untracked LEAD source under src/lead")
    submodule_path = f"3rd_party/leaderboard/{variant}"
    submodules = runner(["git", "-C", str(root), "submodule", "status", "--recursive", "--", submodule_path],
                        capture_output=True, text=True, check=False)
    if submodules.returncode != 0:
        raise RuntimeError("could not verify LEAD's pinned evaluator submodules")
    if any(line[:1] in {"+", "-", "U"} or line.rstrip().endswith("-dirty")
           for line in submodules.stdout.splitlines() if line):
        raise RuntimeError("LEAD evaluator submodule checkout differs from the pinned main commit")
    return sha


def build_invocation(*, lead_root, carla_root, routes, agent_config, output_dir, code_root,
                     mode="standard", python_executable=None, port=2000, traffic_manager_port=8000,
                     traffic_manager_seed=0, repetitions=1, timeout=180, cuda_visible_devices="0"):
    """Return the exact official-evaluator argv/env, refusing output reuse and unpinned LEAD sources."""
    if sys.version_info[:2] < (3, 10) or sys.version_info[:2] >= (3, 13):
        raise RuntimeError("LEAD frontend requires Python 3.10-3.12")
    lead_root = Path(lead_root).expanduser().resolve(strict=True)
    carla_root = Path(carla_root).expanduser().resolve(strict=True)
    routes = Path(routes).expanduser().resolve(strict=True)
    agent_config = Path(agent_config).expanduser().resolve(strict=True)
    code_root = Path(code_root).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse LEAD output directory: {output_dir}")

    if mode == "standard":
        variant = "standard"
        is_bench2drive = "0"
    elif mode == "bench2drive":
        variant = "bench2drive"
        is_bench2drive = "1"
    else:
        raise ValueError("LEAD mode must be 'standard' or 'bench2drive'")
    _verify_lead_checkout(lead_root, variant)
    leaderboard_root = lead_root / "3rd_party" / "leaderboard" / variant / "leaderboard"
    scenario_root = lead_root / "3rd_party" / "leaderboard" / variant / "scenario_runner"
    evaluator = leaderboard_root / "leaderboard" / "leaderboard_evaluator.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"official LEAD evaluator is missing: {evaluator}")
    if not scenario_root.is_dir() or not (carla_root / "PythonAPI").is_dir():
        raise FileNotFoundError("LEAD scenario_runner or CARLA PythonAPI directory is missing")
    agent = code_root / "unidrivevla" / "closedloop" / "lead.py"
    if not agent.is_file():
        raise FileNotFoundError(f"UniDriveVLA LEAD adapter is missing: {agent}")

    try:
        route_root = ET.parse(routes).getroot()
        route_id = routes.stem.split("_", 1)[0]
        scenario = route_root.find(".//scenario")
        scenario_type = scenario.get("type") if scenario is not None and scenario.get("type") else "noScenarios"
    except ET.ParseError as exc:
        raise ValueError(f"LEAD route XML is invalid: {routes}") from exc

    result_path = output_dir / "checkpoint_endpoint.json"
    command = [
        str(python_executable or sys.executable), str(evaluator),
        "--routes", str(routes), "--track", "SENSORS", "--checkpoint", str(result_path),
        "--agent", str(agent), "--agent-config", str(agent_config), "--debug", "0", "--resume", "0",
        "--port", str(int(port)), "--traffic-manager-port", str(int(traffic_manager_port)),
        "--traffic-manager-seed", str(int(traffic_manager_seed)), "--repetitions", str(int(repetitions)),
        "--timeout", str(float(timeout)), "--record", "None",
    ]
    carla_python = carla_root / "PythonAPI"
    pythonpath = [str(code_root), str(lead_root / "src"), str(carla_python), str(carla_python / "carla"),
                  str(leaderboard_root), str(scenario_root)]
    extra_carla_pythonpath = os.environ.get("CARLA_PYTHONPATH", "")
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    if extra_carla_pythonpath:
        pythonpath.append(extra_carla_pythonpath)
    if old_pythonpath:
        pythonpath.append(old_pythonpath)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "SCENARIO_RUNNER_ROOT": str(scenario_root),
        "LEADERBOARD_ROOT": str(leaderboard_root),
        "ROUTES": str(routes),
        "SCENARIO_TYPE": scenario_type,
        "BENCHMARK_ROUTE_ID": route_id,
        "ROUTE_NUMBER": route_id,
        "PYTHONUNBUFFERED": "1",
        "IS_BENCH2DRIVE": is_bench2drive,
        "OUTPUT_DIR": str(output_dir),
        "EVALUATION_OUTPUT_DIR": str(output_dir),
        "SAVE_PATH": str(output_dir),
        "CUDA_VISIBLE_DEVICES": str(cuda_visible_devices),
    })
    env.pop("CHECKPOINT_DIR", None)  # The UniDriveVLA weights live in the separate navsim service.
    return command, env, lead_root, output_dir, result_path


def run_evaluation(*, runner=subprocess.run, **kwargs):
    command, env, cwd, output_dir, result_path = build_invocation(**kwargs)
    output_dir.mkdir(parents=True)
    result = runner(command, cwd=str(cwd), env=env, check=False)
    if result.returncode != 0:
        return result.returncode
    try:
        with result_path.open(encoding="utf-8") as stream:
            score = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"official LEAD evaluator returned 0 without valid score JSON at {result_path}") from exc
    if not isinstance(score, dict) or not score:
        raise RuntimeError(f"official LEAD evaluator wrote an empty or non-object score at {result_path}")
    return 0


def main():
    required = ("LEAD_ROOT", "CARLA_ROOT", "ROUTES", "LEAD_AGENT_CONFIG", "OUTPUT_DIR")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing required environment variables: {', '.join(missing)}")
    try:
        return run_evaluation(
            lead_root=os.environ["LEAD_ROOT"], carla_root=os.environ["CARLA_ROOT"], routes=os.environ["ROUTES"],
            agent_config=os.environ["LEAD_AGENT_CONFIG"], output_dir=os.environ["OUTPUT_DIR"],
            code_root=Path(__file__).resolve().parents[2], mode=os.environ.get("LEAD_MODE", "standard"),
            port=int(os.environ.get("PORT", "2000")),
            traffic_manager_port=int(os.environ.get("TM_PORT", "8000")),
            traffic_manager_seed=int(os.environ.get("TM_SEED", "0")),
            repetitions=int(os.environ.get("REPETITIONS", "1")), timeout=float(os.environ.get("TIMEOUT", "180")),
            cuda_visible_devices=os.environ.get("LEAD_CUDA_VISIBLE_DEVICES", "0"),
        )
    except Exception as exc:
        print(f"LEAD evaluation setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
