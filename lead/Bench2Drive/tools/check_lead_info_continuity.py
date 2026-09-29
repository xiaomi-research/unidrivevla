#!/usr/bin/env python3
"""Validate LEAD info ordering required by GroupInBatchSampler streaming."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def validate_sequence_index(
    folder_ids,
    frame_idxs,
    *,
    expected_stride: int,
    expected_offset: int,
    sequences_split_num: int = 1,
    min_streaming_groups: int = 0,
):
    folder_ids = np.asarray(folder_ids, dtype=np.int64)
    frame_idxs = np.asarray(frame_idxs, dtype=np.int64)
    if folder_ids.ndim != 1 or frame_idxs.ndim != 1:
        raise ValueError("folder_ids and frame_idxs must be one-dimensional")
    if len(folder_ids) != len(frame_idxs) or len(folder_ids) == 0:
        raise ValueError("folder_ids and frame_idxs must have equal non-zero length")
    if expected_stride <= 0:
        raise ValueError("expected_stride must be positive")
    if not 0 <= expected_offset < expected_stride:
        raise ValueError("expected_offset must be in [0, expected_stride)")
    if sequences_split_num <= 0:
        raise ValueError("sequences_split_num must be positive")
    if np.any(frame_idxs % expected_stride != expected_offset):
        raise ValueError("frame index does not match the requested stride/offset")

    seen_folders = {int(folder_ids[0])}
    route_lengths = []
    current_route_length = 1
    transitions = 0
    exact_transitions = 0
    for index in range(1, len(folder_ids)):
        previous_folder = int(folder_ids[index - 1])
        folder = int(folder_ids[index])
        if folder != previous_folder:
            route_lengths.append(current_route_length)
            current_route_length = 1
            if folder in seen_folders:
                raise ValueError(
                    f"route group {folder} re-enters at manifest index {index}"
                )
            seen_folders.add(folder)
            continue
        current_route_length += 1

        delta = int(frame_idxs[index] - frame_idxs[index - 1])
        if delta <= 0:
            raise ValueError(
                "frame indices must strictly increase within each route: "
                f"index={index}, delta={delta}"
            )
        if delta % expected_stride:
            raise ValueError(
                "within-route frame delta is incompatible with the requested "
                f"stride: index={index}, delta={delta}, stride={expected_stride}"
            )
        transitions += 1
        exact_transitions += int(delta == expected_stride)
    route_lengths.append(current_route_length)
    num_streaming_groups = sum(
        min(length, sequences_split_num) for length in route_lengths
    )
    if num_streaming_groups < min_streaming_groups:
        raise ValueError(
            "not enough streaming sequence groups for the global batch: "
            f"groups={num_streaming_groups}, required={min_streaming_groups}"
        )

    return {
        "num_samples": int(len(folder_ids)),
        "num_route_groups": int(len(seen_folders)),
        "num_streaming_groups": int(num_streaming_groups),
        "within_route_transitions": int(transitions),
        "exact_stride_transitions": int(exact_transitions),
        "exact_stride_ratio": (
            float(exact_transitions / transitions) if transitions else 1.0
        ),
    }


def load_sequence_index(manifest_path: Path):
    with manifest_path.open("rb") as handle:
        manifest = pickle.load(handle)
    if isinstance(manifest, dict) and manifest.get(
        "lead_sharded_infos_version"
    ) == 1:
        if int(manifest.get("num_samples", -1)) != len(manifest["folder_ids"]):
            raise ValueError("sharded manifest num_samples is inconsistent")
        return manifest["folder_ids"], manifest["frame_idxs"]
    if isinstance(manifest, list):
        folders = []
        folder_to_id = {}
        frame_idxs = []
        for info in manifest:
            folder = str(info["folder"])
            folder_id = folder_to_id.setdefault(folder, len(folder_to_id))
            folders.append(folder_id)
            frame_idxs.append(int(info["frame_idx"]))
        return folders, frame_idxs
    raise ValueError(f"unsupported LEAD info manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-stride", type=int, required=True)
    parser.add_argument("--expected-offset", type=int, default=0)
    parser.add_argument("--sequences-split-num", type=int, default=1)
    parser.add_argument("--min-streaming-groups", type=int, default=0)
    args = parser.parse_args()
    folder_ids, frame_idxs = load_sequence_index(args.manifest)
    stats = validate_sequence_index(
        folder_ids,
        frame_idxs,
        expected_stride=args.expected_stride,
        expected_offset=args.expected_offset,
        sequences_split_num=args.sequences_split_num,
        min_streaming_groups=args.min_streaming_groups,
    )
    print(
        "LEAD sequence continuity OK: "
        + ", ".join(f"{key}={value}" for key, value in stats.items())
    )


if __name__ == "__main__":
    main()
