#!/usr/bin/env python3
"""Run explicit local recovery routes without racing an existing watcher."""

import argparse
import fcntl
import os
import shutil
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

from slurm.evaluation.recover_remote_routes import is_valid_result, log, run_local_route


EVALUATION_PROCESS_MARKERS = (
    "slurm/evaluation/evaluate.py",
    "leaderboard_evaluator.py",
)
CARLA_PROCESS_MARKER = "CarlaUE4-Linux-Shipping"


def parse_route_destination(value: str) -> tuple:
    try:
        route_id, destination = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("route must use ROUTE_ID=SHARD_OUTPUT_DIR") from exc
    if not route_id.isdigit() or not destination:
        raise argparse.ArgumentTypeError("route must use numeric ROUTE_ID=SHARD_OUTPUT_DIR")
    return route_id, Path(destination)


def process_exists(run_id: str, exclude_pid: int) -> bool:
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit() or int(proc_dir.name) == exclude_pid:
            continue
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if CARLA_PROCESS_MARKER in command:
            return True
        if run_id in command and any(marker in command for marker in EVALUATION_PROCESS_MARKERS):
            return True
    return False


def wait_until_idle(run_id: str, timeout_seconds: float, poll_interval: float) -> None:
    started = time.monotonic()
    next_update = started
    while process_exists(run_id, os.getpid()):
        now = time.monotonic()
        if now - started >= timeout_seconds:
            raise TimeoutError(
                f"Local evaluation for {run_id} did not become idle within {timeout_seconds}s"
            )
        if now >= next_update:
            log(f"Waiting for the current {run_id} route to release local GPU 0")
            next_update = now + 60.0
        time.sleep(poll_interval)


def promote_route(result_path: Path, destination: Path, route_id: str) -> None:
    if not is_valid_result(result_path):
        raise RuntimeError(f"Route {route_id} did not produce a valid result: {result_path}")
    destination_eval = destination / "eval"
    destination_eval.mkdir(parents=True, exist_ok=True)
    target = destination_eval / f"{route_id}.json"
    temporary = destination_eval / f".{route_id}.manual-recovery-{os.getpid()}.tmp"
    shutil.copyfile(result_path, temporary)
    os.replace(temporary, target)
    if not is_valid_result(target):
        raise RuntimeError(f"Promoted result failed validation: {target}")


def watcher_state(watcher_pid: int) -> Optional[str]:
    try:
        for line in Path(f"/proc/{watcher_pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watcher-pid",
        type=int,
        help="Optional recovery watcher to pause while this queue owns local GPU 0.",
    )
    parser.add_argument("--route", action="append", required=True, type=parse_route_destination)
    parser.add_argument("--recovery-run-id", required=True)
    parser.add_argument("--lead-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--vulkan-icd", type=Path, required=True)
    parser.add_argument("--xdg-runtime-dir", type=Path, required=True)
    parser.add_argument("--max-launches", type=int, default=3)
    parser.add_argument("--idle-timeout", type=float, default=10800.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/lead-b2d-manual-recovery.lock"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.lead_root = args.lead_root.resolve()
    route_destinations: Dict[str, Path] = {
        route_id: destination.resolve() for route_id, destination in args.route
    }
    if len(route_destinations) != len(args.route):
        raise ValueError("Duplicate route IDs are not allowed")
    if args.max_launches < 1 or args.poll_interval <= 0 or args.idle_timeout <= 0:
        raise ValueError("launches and timeout values must be positive")
    if args.watcher_pid is not None and watcher_state(args.watcher_pid) is None:
        raise ProcessLookupError(f"Watcher PID {args.watcher_pid} does not exist")
    for route_id, destination in route_destinations.items():
        if not destination.is_dir():
            raise FileNotFoundError(destination)
        route_dirs = (destination / "routes_all", destination / "routes")
        if not any((route_dir / f"{route_id}.xml").is_file() for route_dir in route_dirs):
            raise FileNotFoundError(f"Route {route_id} is not part of {destination}")

    recovery_output = args.lead_root / "outputs" / "evaluation" / args.recovery_run_id
    launch_args = SimpleNamespace(
        lead_root=args.lead_root,
        python_bin=args.python_bin.resolve(),
        carla_root=args.carla_root.resolve(),
        local_gpu=0,
        vulkan_icd=args.vulkan_icd.resolve(),
        xdg_runtime_dir=args.xdg_runtime_dir.resolve(),
        recovery_run_id=args.recovery_run_id,
    )

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another manual recovery queue owns {args.lock_file}") from exc

        watcher_paused = False
        failed_routes = []
        try:
            if args.watcher_pid is not None:
                os.kill(args.watcher_pid, signal.SIGSTOP)
                watcher_paused = True
                log(
                    f"Paused watcher PID {args.watcher_pid}; "
                    "its current route continues running"
                )
            else:
                log("Running standalone manual recovery queue without a watcher")
            wait_until_idle(args.recovery_run_id, args.idle_timeout, args.poll_interval)

            for route_id, destination in route_destinations.items():
                result_path = recovery_output / "eval" / f"{route_id}.json"
                destination_result = destination / "eval" / f"{route_id}.json"
                if is_valid_result(destination_result):
                    log(f"Route {route_id} already has a valid destination result; skipping")
                    continue
                if is_valid_result(result_path):
                    promote_route(result_path, destination, route_id)
                    log(f"Promoted existing local result for route {route_id}")
                    continue

                for launch in range(1, args.max_launches + 1):
                    log(f"Manual recovery route {route_id}, launch {launch}/{args.max_launches}")
                    return_code = run_local_route(launch_args, route_id)
                    log(f"Manual route {route_id} launcher exited with code {return_code}")
                    if is_valid_result(result_path):
                        promote_route(result_path, destination, route_id)
                        log(f"Route {route_id} recovered and promoted into {destination}")
                        break
                    log(f"Route {route_id} launch {launch} did not produce a valid result")
                else:
                    failed_routes.append(route_id)
                    log(
                        f"Route {route_id} exhausted {args.max_launches} manual launches; "
                        "continuing with the remaining routes"
                    )
        finally:
            if (
                watcher_paused
                and args.watcher_pid is not None
                and watcher_state(args.watcher_pid) is not None
            ):
                os.kill(args.watcher_pid, signal.SIGCONT)
                log(f"Resumed watcher PID {args.watcher_pid}")

    if failed_routes:
        log("Manual recovery failures: " + " ".join(failed_routes))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
