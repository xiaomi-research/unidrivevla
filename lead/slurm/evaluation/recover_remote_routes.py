#!/usr/bin/env python3
"""Recover routes abandoned by a broken GPU slot on a remote evaluator."""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from slurm.evaluation.evaluate_utils import did_route_crash


ATTEMPT_RE = re.compile(r"^stdout_(?P<route_id>.+)_(?P<attempt>[0-9]+)\.txt$")
GPU_RE = re.compile(
    r"^Route\s+\S+\s+uses CUDA_VISIBLE_DEVICES=(?P<gpu>[0-9]+),",
    re.MULTILINE,
)
TERMINAL_INFRA_MARKERS = (
    "LowLevelFatalError",
    "Segmentation fault",
    "VK_ERROR_DEVICE_LOST",
    "Failed to connect to CARLA server",
    "load_world failed",
    "CarlaUE4-Linux-Shipping not found",
)


@dataclass(frozen=True)
class RecoveryCandidate:
    route_id: str
    attempts: int
    latest_attempt: int
    age_seconds: float


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{timestamp} UTC] {message}", flush=True)


def load_result(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as result_file:
            return json.load(result_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def is_valid_result(path: Path) -> bool:
    result = load_result(path)
    return result is not None and not did_route_crash(result)


def _read_text(path: Path, max_bytes: int = 512 * 1024) -> str:
    try:
        with path.open("rb") as input_file:
            data = input_file.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _attempt_gpu(stdout_path: Path) -> Optional[int]:
    match = GPU_RE.search(_read_text(stdout_path, max_bytes=4096))
    return int(match.group("gpu")) if match else None


def _has_terminal_infrastructure_failure(stdout_path: Path, stderr_path: Path) -> bool:
    output = _read_text(stdout_path) + "\n" + _read_text(stderr_path)
    return any(marker in output for marker in TERMINAL_INFRA_MARKERS)


def _group_attempts(stdout_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    attempts: Dict[str, List[Tuple[int, Path]]] = {}
    for stdout_path in stdout_dir.glob("stdout_*.txt"):
        match = ATTEMPT_RE.match(stdout_path.name)
        if match is None:
            continue
        route_id = match.group("route_id")
        attempt = int(match.group("attempt"))
        attempts.setdefault(route_id, []).append((attempt, stdout_path))
    for route_attempts in attempts.values():
        route_attempts.sort(key=lambda item: item[0])
    return attempts


def discover_candidates(
    source_output_dir: Path,
    target_gpu: int,
    max_remote_attempts: int,
    stale_seconds: float,
    now: Optional[float] = None,
) -> List[RecoveryCandidate]:
    """Find routes abandoned after terminal infrastructure failures on one GPU."""
    now = time.time() if now is None else now
    stdout_dir = source_output_dir / "stdout"
    stderr_dir = source_output_dir / "stderr"
    eval_dir = source_output_dir / "eval"
    candidates: List[RecoveryCandidate] = []

    for route_id, attempts in _group_attempts(stdout_dir).items():
        if is_valid_result(eval_dir / f"{route_id}.json"):
            continue

        gpu_attempts = [item for item in attempts if _attempt_gpu(item[1]) == target_gpu]
        if len(gpu_attempts) < max_remote_attempts:
            continue

        latest_attempt, latest_stdout = gpu_attempts[-1]
        if latest_attempt < max_remote_attempts:
            continue
        latest_stderr = stderr_dir / f"stderr_{route_id}_{latest_attempt}.txt"
        if not _has_terminal_infrastructure_failure(latest_stdout, latest_stderr):
            continue

        modified_times = [latest_stdout.stat().st_mtime]
        if latest_stderr.exists():
            modified_times.append(latest_stderr.stat().st_mtime)
        age_seconds = now - max(modified_times)
        if age_seconds < stale_seconds:
            continue

        candidates.append(
            RecoveryCandidate(
                route_id=route_id,
                attempts=len(gpu_attempts),
                latest_attempt=latest_attempt,
                age_seconds=age_seconds,
            ),
        )

    return sorted(candidates, key=lambda candidate: int(candidate.route_id))


def _allowed_route_ids(source_output_dir: Path) -> set:
    route_dirs = [source_output_dir / "routes_all", source_output_dir / "routes"]
    for route_dir in route_dirs:
        route_ids = {path.stem for path in route_dir.glob("*.xml")}
        if route_ids:
            return route_ids
    return set()


def promote_valid_results(
    recovery_output_dir: Path,
    source_output_dir: Path,
) -> List[str]:
    """Atomically install valid local recovery results into the source shard."""
    recovery_eval_dir = recovery_output_dir / "eval"
    source_eval_dir = source_output_dir / "eval"
    source_eval_dir.mkdir(parents=True, exist_ok=True)
    allowed_route_ids = _allowed_route_ids(source_output_dir)
    promoted: List[str] = []

    for recovery_result in sorted(recovery_eval_dir.glob("*.json")):
        route_id = recovery_result.stem
        if route_id == "merged" or (allowed_route_ids and route_id not in allowed_route_ids):
            continue
        if not is_valid_result(recovery_result):
            continue

        source_result = source_eval_dir / recovery_result.name
        if is_valid_result(source_result):
            continue

        temporary_result = source_eval_dir / f".{route_id}.recovery-{os.getpid()}.tmp"
        shutil.copyfile(recovery_result, temporary_result)
        os.replace(temporary_result, source_result)
        promoted.append(route_id)

    return promoted


def _load_state(state_path: Path) -> dict:
    state = load_result(state_path)
    if not isinstance(state, dict):
        return {"local_launches": {}}
    state.setdefault("local_launches", {})
    return state


def _write_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(f".tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)
    os.replace(temporary_path, state_path)


def run_local_route(args: argparse.Namespace, route_id: str) -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(args.python_bin),
            "CARLA_ROOT": str(args.carla_root),
            "CUDA_VISIBLE_DEVICES": str(args.local_gpu),
            "LOCAL_GPUS_PER_NODE": "1",
            "LEAD_LOCAL_GPU_IDS": "0",
            "LEAD_LOCAL_MAX_PARALLEL_JOBS": "1",
            "CARLA_VK_ICD": str(args.vulkan_icd),
            "CARLA_XDG_RUNTIME_DIR": str(args.xdg_runtime_dir),
            "BENCH2DRIVE_NUM_SHARDS": "1",
            "BENCH2DRIVE_SHARD_INDEX": "0",
            "ROUTE_IDS": route_id,
            "ALLOW_BENCH2DRIVE_SUBSET": "1",
            "ROUTE_STARTUP_STAGGER": "0",
            "EXPERIMENT_RUN_ID": args.recovery_run_id,
            "USE_B2D_NONROOT_CARLA": "0",
            "LEAD_FORCE_LOCAL": "1",
            "PYTHONUNBUFFERED": "1",
        },
    )
    command = ["bash", str(args.lead_root / "run_unidrivevla_lead_stage1_bench2drive_eval_8gpu.sh")]
    log(f"Launching local recovery for route {route_id}")
    return subprocess.run(command, cwd=args.lead_root, env=env, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--recovery-run-id", required=True)
    parser.add_argument("--lead-root", type=Path, default=Path.cwd())
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--local-gpu", type=int, default=0)
    parser.add_argument("--remote-gpu", type=int, default=0)
    parser.add_argument("--remote-max-attempts", type=int, default=3)
    parser.add_argument("--max-local-launches", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=180.0)
    parser.add_argument("--vulkan-icd", type=Path, default=Path("/etc/vulkan/icd.d/nvidia_icd.json"))
    parser.add_argument("--xdg-runtime-dir", type=Path, default=Path("/tmp/xdg-lead-b2d-recovery-watcher"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.lead_root = args.lead_root.resolve()
    args.source_output_dir = args.source_output_dir.resolve()
    recovery_output_dir = args.lead_root / "outputs" / "evaluation" / args.recovery_run_id
    state_path = recovery_output_dir / "recovery_watcher_state.json"

    for required_path in (
        args.source_output_dir,
        args.python_bin,
        args.carla_root,
        args.lead_root / "run_unidrivevla_lead_stage1_bench2drive_eval_8gpu.sh",
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    if args.local_gpu != 0:
        raise ValueError("This watcher exposes one local GPU as logical GPU 0; use --local-gpu 0")
    if args.poll_interval <= 0 or args.stale_seconds < 0:
        raise ValueError("poll interval must be positive and stale seconds must be non-negative")

    state = _load_state(state_path)
    log(
        f"Watching {args.source_output_dir} GPU {args.remote_gpu}; "
        f"local results use {recovery_output_dir}",
    )

    while True:
        if not args.dry_run:
            promoted = promote_valid_results(recovery_output_dir, args.source_output_dir)
            if promoted:
                log(f"Promoted valid local results into shard 0: {' '.join(promoted)}")

        candidates = discover_candidates(
            args.source_output_dir,
            target_gpu=args.remote_gpu,
            max_remote_attempts=args.remote_max_attempts,
            stale_seconds=args.stale_seconds,
        )
        pending = [
            candidate
            for candidate in candidates
            if not is_valid_result(recovery_output_dir / "eval" / f"{candidate.route_id}.json")
        ]
        if candidates:
            log(
                "Detected remote recovery candidates: "
                + " ".join(candidate.route_id for candidate in candidates),
            )

        if not args.dry_run:
            for candidate in pending:
                launches = int(state["local_launches"].get(candidate.route_id, 0))
                if launches >= args.max_local_launches:
                    log(
                        f"Route {candidate.route_id} exhausted {launches} local launches; "
                        "leaving it for manual inspection",
                    )
                    continue
                state["local_launches"][candidate.route_id] = launches + 1
                _write_state(state_path, state)
                return_code = run_local_route(args, candidate.route_id)
                log(f"Local route {candidate.route_id} launcher exited with code {return_code}")
                promoted = promote_valid_results(recovery_output_dir, args.source_output_dir)
                if candidate.route_id in promoted:
                    log(f"Route {candidate.route_id} recovered and promoted")
                elif is_valid_result(recovery_output_dir / "eval" / f"{candidate.route_id}.json"):
                    log(f"Route {candidate.route_id} already has a valid promoted result")
                else:
                    log(f"Route {candidate.route_id} did not produce a valid result")

        if args.once:
            return 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
