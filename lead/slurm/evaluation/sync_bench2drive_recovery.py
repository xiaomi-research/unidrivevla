#!/usr/bin/env python3
"""Synchronize valid results between benchmark shards and a recovery run."""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from merge_bench2drive_shards import load_valid_result
except ModuleNotFoundError:
    from slurm.evaluation.merge_bench2drive_shards import load_valid_result


def route_ids(route_dir: Path) -> set:
    ids = set()
    for route_path in route_dir.glob("*.xml"):
        routes = list(ET.parse(route_path).getroot().iter("route"))
        if len(routes) != 1:
            raise ValueError(
                f"Expected one route in {route_path}, found {len(routes)}"
            )
        route_id = routes[0].get("id")
        if not route_id:
            raise ValueError(f"Route without id in {route_path}")
        if route_id in ids:
            raise ValueError(f"Duplicate route id in {route_dir}: {route_id}")
        ids.add(route_id)
    return ids


def is_valid(path: Path) -> bool:
    return path.is_file() and load_valid_result(path)[1] is None


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.sync-{os.getpid()}.tmp"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def synchronize(shard_dirs, recovery_dir: Path):
    owners = {}
    for shard_dir in shard_dirs:
        route_dir = shard_dir / "routes_all"
        eval_dir = shard_dir / "eval"
        if not route_dir.is_dir() or not eval_dir.is_dir():
            raise FileNotFoundError(
                f"Shard requires routes_all/ and eval/: {shard_dir}"
            )
        for route_id in route_ids(route_dir):
            if route_id in owners:
                raise ValueError(
                    f"Route {route_id} belongs to both {owners[route_id]} "
                    f"and {shard_dir}"
                )
            owners[route_id] = shard_dir

    if len(owners) != 220:
        raise ValueError(
            f"Expected 220 disjoint shard routes, found {len(owners)}"
        )

    recovery_eval = recovery_dir / "eval"
    recovery_eval.mkdir(parents=True, exist_ok=True)
    seeded = []
    promoted = []
    pending = []

    for route_id in sorted(owners, key=int):
        shard_result = owners[route_id] / "eval" / f"{route_id}.json"
        recovery_result = recovery_eval / f"{route_id}.json"

        # The original shard is canonical when it already has a valid result.
        if is_valid(shard_result):
            if not is_valid(recovery_result) or (
                shard_result.read_bytes() != recovery_result.read_bytes()
            ):
                atomic_copy(shard_result, recovery_result)
                seeded.append(route_id)
            continue

        if is_valid(recovery_result):
            atomic_copy(recovery_result, shard_result)
            promoted.append(route_id)
            continue

        pending.append(route_id)

    return pending, seeded, promoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True, type=Path)
    parser.add_argument("--recovery-dir", required=True, type=Path)
    args = parser.parse_args()

    pending, seeded, promoted = synchronize(
        [path.resolve() for path in args.shard_dir],
        args.recovery_dir.resolve(),
    )
    print(
        f"Recovery sync: seeded={len(seeded)}, promoted={len(promoted)}, "
        f"pending={len(pending)}",
        file=sys.stderr,
    )
    if promoted:
        print("Promoted route IDs: " + " ".join(promoted), file=sys.stderr)
    for route_id in pending:
        print(route_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
