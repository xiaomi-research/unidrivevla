#!/usr/bin/env python
import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from merge_route_json import merge_route_json
except ModuleNotFoundError:
    from slurm.evaluation.merge_route_json import merge_route_json


SOFTWARE_FAILURE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
    "Failed - Agent's sensors were invalid",
}


def record_has_software_failure(record):
    """Keep policy failures, including zero-completion blocked routes, mergeable."""
    status = record.get("status")
    if not isinstance(status, str):
        return True
    return status in SOFTWARE_FAILURE_STATUSES


def load_valid_result(path):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return None, f"invalid JSON: {exc}"

    records = data.get("_checkpoint", {}).get("records", [])
    if data.get("entry_status") != "Finished":
        return None, f"entry_status={data.get('entry_status')!r}"
    if data.get("eligible") is not True:
        return None, f"eligible={data.get('eligible')!r}"
    if not records:
        return None, "no route records"
    if any(record_has_software_failure(record) for record in records):
        return None, "contains a software-failure record"
    return data, None


def expected_route_ids(route_dir):
    route_files = sorted(route_dir.glob("*.xml"))
    route_ids = set()
    for route_file in route_files:
        routes = list(ET.parse(route_file).getroot().iter("route"))
        if len(routes) != 1:
            raise ValueError(
                f"Expected one route in {route_file}, found {len(routes)}"
            )
        route_id = routes[0].get("id")
        if not route_id:
            raise ValueError(f"Route without id in {route_file}")
        if route_id in route_ids:
            raise ValueError(f"Duplicate expected route id: {route_id}")
        route_ids.add(route_id)
    if len(route_ids) != 220:
        raise ValueError(
            f"Expected 220 benchmark routes, found {len(route_ids)} in {route_dir}"
        )
    return route_ids


def collect_results(shard_dirs, expected_ids):
    result_paths = {}
    invalid_results = []
    for shard_dir in shard_dirs:
        eval_dir = shard_dir / "eval"
        if not eval_dir.is_dir():
            raise FileNotFoundError(f"Shard eval directory not found: {eval_dir}")
        for result_path in sorted(eval_dir.glob("*.json")):
            if result_path.name == "merged.json":
                continue
            route_id = result_path.stem
            if route_id not in expected_ids:
                raise ValueError(
                    f"Unexpected route {route_id} in shard result {result_path}"
                )
            if route_id in result_paths:
                raise ValueError(
                    f"Duplicate route {route_id} in {result_paths[route_id]} "
                    f"and {result_path}"
                )
            _, error = load_valid_result(result_path)
            if error is not None:
                invalid_results.append(f"{result_path}: {error}")
                continue
            result_paths[route_id] = result_path
    return result_paths, invalid_results


def merge_shards(shard_dirs, output_dir, route_dir):
    expected_ids = expected_route_ids(route_dir)
    result_paths, invalid_results = collect_results(shard_dirs, expected_ids)
    missing_ids = sorted(expected_ids - set(result_paths), key=int)
    if invalid_results or missing_ids:
        details = []
        if missing_ids:
            details.append(
                f"missing {len(missing_ids)} routes: "
                + " ".join(missing_ids[:20])
                + (" ..." if len(missing_ids) > 20 else "")
            )
        if invalid_results:
            details.append(
                f"invalid {len(invalid_results)} results:\n"
                + "\n".join(invalid_results[:20])
            )
        raise RuntimeError("Cannot merge incomplete shards; " + "; ".join(details))

    resolved_output = output_dir.resolve()
    if any(resolved_output == shard_dir.resolve() for shard_dir in shard_dirs):
        raise ValueError("Output directory must be different from every shard directory")

    combined_eval_dir = output_dir / "eval"
    combined_eval_dir.mkdir(parents=True, exist_ok=True)
    for stale_json in combined_eval_dir.glob("*.json"):
        stale_json.unlink()
    for route_id, source in result_paths.items():
        shutil.copy2(source, combined_eval_dir / f"{route_id}.json")

    os.environ["EVALUATION_DATASET"] = "bench2drive"
    os.environ["BENCH2DRIVE_NUM_SHARDS"] = "1"
    os.environ["BENCH2DRIVE_SHARD_INDEX"] = "0"
    merged = merge_route_json(str(combined_eval_dir))
    if merged.get("eval num") != 220:
        raise RuntimeError(
            f"Merged result contains {merged.get('eval num')} records, expected 220"
        )
    shutil.copy2(combined_eval_dir / "merged.json", output_dir / "merged.json")
    print(f"Merged 220 routes into {output_dir / 'merged.json'}")
    print(f"Driving Score: {merged['driving score']}")
    print(f"Success Rate: {merged['success rate']}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and merge disjoint Bench2Drive evaluation shards."
    )
    parser.add_argument("--shard-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=Path("data/benchmark_routes/bench2drive"),
    )
    args = parser.parse_args()
    merge_shards(args.shard_dir, args.output_dir, args.route_dir)


if __name__ == "__main__":
    main()
