#!/usr/bin/env python3
"""Verify that two LEAD info sets select the same route frames."""

import argparse
import json
import pickle


SELECTION_STATS_KEYS = (
    "stats_version",
    "sensor_perturbation_version",
    "sensor_perturbation_probability",
    "num_sensor_perturbation_samples",
    "perturbed_image_layout",
    "detection_filter_version",
    "detection_class_mapping_version",
    "detection_z_range",
    "detection_visibility_filter",
    "detection_traffic_control_filter",
    "detection_box_dimension_order",
    "detection_box_center",
    "bbox_center_offset_version",
    "frame_sample_stride",
    "frame_sample_offset",
    "route_quality_filter",
    "route_skip_first",
    "route_skip_last",
    "source_sim_hz",
    "data_save_freq",
    "jpeg_storage_quality",
    "perception_sample_hz",
    "planning_step_sim_frames",
    "planning_freq_hz",
    "planning_horizon_steps",
    "planning_horizon_seconds",
    "val_ratio",
    "split_strategy",
    "num_train_samples",
    "num_val_samples",
    "num_total_samples",
    "valid_waypoints",
    "delta_mu",
    "delta_std",
    "delta_min",
    "delta_max",
    "absolute_min",
    "absolute_max",
    "route_zero_ade",
    "route_zero_fde",
    "route_zero_total",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--reference-stats")
    parser.add_argument("--candidate-stats")
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def manifest_keys(data, path):
    if isinstance(data, dict) and data.get("lead_sharded_infos_version") == 1:
        num_samples = int(data["num_samples"])
        folder_names = list(data["folder_names"])
        folder_ids = data["folder_ids"]
        frame_idxs = data["frame_idxs"]
        if len(folder_ids) != num_samples or len(frame_idxs) != num_samples:
            raise RuntimeError(f"Inconsistent sharded manifest lengths: {path}")
        return num_samples, (
            (folder_names[int(folder_id)], int(frame_idx))
            for folder_id, frame_idx in zip(folder_ids, frame_idxs)
        )
    if isinstance(data, list):
        return len(data), (
            (item["folder"], int(item["frame_idx"])) for item in data
        )
    raise RuntimeError(f"Unsupported LEAD info format: {path}")


def compare_stats(reference_path, candidate_path):
    if not reference_path and not candidate_path:
        return
    if not reference_path or not candidate_path:
        raise ValueError("Set both --reference-stats and --candidate-stats")
    with open(reference_path, "r", encoding="utf-8") as stream:
        reference = json.load(stream)
    with open(candidate_path, "r", encoding="utf-8") as stream:
        candidate = json.load(stream)
    mismatches = {
        key: (reference.get(key), candidate.get(key))
        for key in SELECTION_STATS_KEYS
        if reference.get(key) != candidate.get(key)
    }
    if mismatches:
        raise RuntimeError(f"LEAD selection stats differ: {mismatches}")


def main():
    args = parse_args()
    reference_count, reference_keys = manifest_keys(
        load_pickle(args.reference_manifest), args.reference_manifest
    )
    candidate_count, candidate_keys = manifest_keys(
        load_pickle(args.candidate_manifest), args.candidate_manifest
    )
    if reference_count != candidate_count:
        raise RuntimeError(
            "LEAD info sample counts differ: "
            f"reference={reference_count}, candidate={candidate_count}"
        )
    for index, (reference_key, candidate_key) in enumerate(
        zip(reference_keys, candidate_keys)
    ):
        if reference_key != candidate_key:
            raise RuntimeError(
                f"LEAD info identity mismatch at index {index}: "
                f"reference={reference_key}, candidate={candidate_key}"
            )
    compare_stats(args.reference_stats, args.candidate_stats)
    print(
        "LEAD info identity OK: "
        f"num_samples={reference_count}, route/frame sequence is exact"
    )


if __name__ == "__main__":
    main()
