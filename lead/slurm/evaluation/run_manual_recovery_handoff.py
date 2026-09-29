#!/usr/bin/env python3
"""Hand off a local watcher to explicit routes, then restart the watcher."""

import argparse
import fcntl
import os
import signal
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from slurm.evaluation.recover_remote_routes import (
    is_valid_result,
    log,
    promote_valid_results,
    run_local_route,
)
from slurm.evaluation.run_manual_recovery_queue import parse_route_destination, promote_route


STACK_MARKERS = (
    "slurm/evaluation/evaluate.py",
    "leaderboard_evaluator.py",
    "run_unidrivevla_lead_eval_8gpu.sh",
    "/scripts/",
)
CARLA_MARKER = "CarlaUE4-Linux-Shipping"


def process_commands() -> Iterable[Tuple[int, str]]:
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        yield int(proc_dir.name), command


def read_process_command(pid: int) -> List[str]:
    try:
        raw_command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError) as exc:
        raise ProcessLookupError(f"Watcher PID {pid} does not exist") from exc
    command = [part.decode("utf-8", errors="strict") for part in raw_command.split(b"\0") if part]
    if not command:
        raise ProcessLookupError(f"Watcher PID {pid} has no executable command")
    return command


def route_is_running(route_id: str) -> bool:
    route_marker = f"/routes/{route_id}.xml"
    return any(
        "leaderboard_evaluator.py" in command and route_marker in command
        for _, command in process_commands()
    )


def wait_for_route_handoff(
    route_id: str,
    result_path: Path,
    timeout_seconds: float,
    poll_interval: float,
) -> None:
    started = time.monotonic()
    next_update = started
    while route_is_running(route_id) and not is_valid_result(result_path):
        now = time.monotonic()
        if now - started >= timeout_seconds:
            raise TimeoutError(f"Route {route_id} did not reach a handoff point")
        if now >= next_update:
            log(f"Waiting for current route {route_id} to finish before watcher handoff")
            next_update = now + 60.0
        time.sleep(poll_interval)


def stack_process_groups(watcher_pid: int, run_id: str, exclude_pgid: int) -> List[int]:
    groups = set()
    try:
        groups.add(os.getpgid(watcher_pid))
    except ProcessLookupError:
        pass
    for pid, command in process_commands():
        if pid == os.getpid():
            continue
        is_carla = CARLA_MARKER in command and "carla-rpc-port=20000" in command
        is_evaluation = run_id in command and any(marker in command for marker in STACK_MARKERS)
        if not is_carla and not is_evaluation:
            continue
        try:
            process_group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if process_group != exclude_pgid:
            groups.add(process_group)
    return sorted(groups)


def stop_old_stack(watcher_pid: int, run_id: str, timeout_seconds: float = 30.0) -> None:
    own_pgid = os.getpgrp()
    groups = stack_process_groups(watcher_pid, run_id, own_pgid)
    log("Stopping old watcher/evaluator process groups: " + " ".join(map(str, groups)))
    for process_group in groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = stack_process_groups(watcher_pid, run_id, own_pgid)
        if not remaining:
            return
        time.sleep(1.0)

    remaining = stack_process_groups(watcher_pid, run_id, own_pgid)
    if remaining:
        log("Force-stopping stale process groups: " + " ".join(map(str, remaining)))
    for process_group in remaining:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watcher-pid", type=int, required=True)
    parser.add_argument("--handoff-route", required=True)
    parser.add_argument("--watcher-source-output-dir", type=Path, required=True)
    parser.add_argument("--route", action="append", required=True, type=parse_route_destination)
    parser.add_argument("--recovery-run-id", required=True)
    parser.add_argument("--lead-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--vulkan-icd", type=Path, required=True)
    parser.add_argument("--xdg-runtime-dir", type=Path, required=True)
    parser.add_argument("--max-launches", type=int, default=3)
    parser.add_argument("--handoff-timeout", type=float, default=10800.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/lead-b2d-manual-recovery.lock"))
    return parser


def run_manual_routes(args: argparse.Namespace, route_destinations: Dict[str, Path]) -> None:
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
            raise RuntimeError(f"Route {route_id} exhausted {args.max_launches} manual launches")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.lead_root = args.lead_root.resolve()
    args.watcher_source_output_dir = args.watcher_source_output_dir.resolve()
    route_destinations = {
        route_id: destination.resolve() for route_id, destination in args.route
    }
    if len(route_destinations) != len(args.route):
        raise ValueError("Duplicate route IDs are not allowed")
    if args.max_launches < 1 or args.poll_interval <= 0 or args.handoff_timeout <= 0:
        raise ValueError("launches and timeout values must be positive")
    for route_id, destination in route_destinations.items():
        route_dirs = (destination / "routes_all", destination / "routes")
        if not destination.is_dir() or not any(
            (route_dir / f"{route_id}.xml").is_file() for route_dir in route_dirs
        ):
            raise FileNotFoundError(f"Route {route_id} is not part of {destination}")

    watcher_command = read_process_command(args.watcher_pid)
    recovery_output = args.lead_root / "outputs" / "evaluation" / args.recovery_run_id
    handoff_result = recovery_output / "eval" / f"{args.handoff_route}.json"
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)

    with args.lock_file.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another manual recovery owns {args.lock_file}") from exc

        old_stack_stopped = False
        try:
            wait_for_route_handoff(
                args.handoff_route,
                handoff_result,
                args.handoff_timeout,
                args.poll_interval,
            )
            old_stack_stopped = True
            stop_old_stack(args.watcher_pid, args.recovery_run_id)
            promoted = promote_valid_results(recovery_output, args.watcher_source_output_dir)
            if promoted:
                log("Promoted handoff results into shard 0: " + " ".join(promoted))
            run_manual_routes(args, route_destinations)
        except Exception:
            log("Manual recovery handoff failed; restarting the automatic watcher")
            traceback.print_exc()
        finally:
            if old_stack_stopped:
                log("Manual queue finished; replacing this process with the original watcher")
                os.chdir(args.lead_root)
                os.execv(watcher_command[0], watcher_command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
