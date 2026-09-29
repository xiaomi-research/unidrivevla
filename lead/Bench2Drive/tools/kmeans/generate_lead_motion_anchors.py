#!/usr/bin/env python3
"""Generate class-conditioned LEAD motion anchors from generated infos."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans


CLASSES = [
    "car",
    "van",
    "truck",
    "bicycle",
    "traffic_sign",
    "traffic_cone",
    "traffic_light",
    "pedestrian",
    "others",
]
DYNAMIC_CLASSES = {"car", "van", "truck", "bicycle", "pedestrian"}
STATIC_CLASSES = set(CLASSES) - DYNAMIC_CLASSES
FALLBACK_GROUPS = {
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bicycle": "vulnerable_road_user",
    "pedestrian": "vulnerable_road_user",
}


class Reservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.values = []

    def add(self, value: np.ndarray) -> None:
        self.seen += 1
        value = np.asarray(value, dtype=np.float32).copy()
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        index = int(self.rng.integers(0, self.seen))
        if index < self.capacity:
            self.values[index] = value

    def array(self, horizon: int) -> np.ndarray:
        if not self.values:
            return np.empty((0, horizon, 2), dtype=np.float32)
        return np.stack(self.values, axis=0)


def iter_infos(info_file: Path) -> Iterable[dict]:
    with info_file.open("rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, list):
        yield from payload
        return
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported LEAD info payload: {type(payload).__name__}")
    if payload.get("lead_sharded_infos_version") == 1:
        for shard in payload["shards"]:
            shard_path = info_file.parent / shard["file"]
            with shard_path.open("rb") as stream:
                values = pickle.load(stream)
            expected_length = int(shard["length"])
            if len(values) != expected_length:
                raise RuntimeError(
                    f"LEAD shard length mismatch for {shard_path}: "
                    f"{len(values)} != {expected_length}"
                )
            yield from values
        return
    if isinstance(payload.get("infos"), list):
        yield from payload["infos"]
        return
    raise ValueError(f"Unsupported LEAD info manifest: {info_file}")


def lidar_offsets_to_agent(
    trajectories: np.ndarray, boxes: np.ndarray
) -> np.ndarray:
    """Convert cumulative lidar-frame offsets to the current agent frame."""
    cumulative = np.cumsum(np.asarray(trajectories, dtype=np.float32), axis=1)
    yaw = -np.asarray(boxes, dtype=np.float32)[:, 6]
    cosine = np.cos(yaw)[:, None]
    sine = np.sin(yaw)[:, None]
    x = cumulative[:, :, 0]
    y = cumulative[:, :, 1]
    return np.stack(
        [cosine * x - sine * y, sine * x + cosine * y], axis=-1
    ).astype(np.float32)


def _class_group(name: str) -> str:
    if name in {"car", "van", "truck"}:
        return "vehicle"
    if name in {"bicycle", "pedestrian"}:
        return "vulnerable_road_user"
    raise ValueError(f"No dynamic fallback group for {name}")


def collect_trajectories(
    info_file: Path,
    *,
    horizon: int,
    distance_threshold: float,
    capacity: int,
    seed: int,
    expected_info_version: int,
    expected_motion_frequency: float,
):
    class_reservoirs = {
        name: Reservoir(capacity, seed + 1009 * (index + 1))
        for index, name in enumerate(CLASSES)
    }
    group_reservoirs = {
        "vehicle": Reservoir(capacity, seed + 100_003),
        "vulnerable_road_user": Reservoir(capacity, seed + 200_003),
    }
    counts = {
        name: {
            "objects": 0,
            "full_valid": 0,
            "within_distance": 0,
            "eligible": 0,
        }
        for name in CLASSES
    }
    info_count = 0
    observed_versions = set()
    observed_horizons = set()
    observed_frequencies = set()

    for info in iter_infos(info_file):
        info_count += 1
        observed_versions.add(int(info.get("lead_stage1_infos_version", -1)))
        observed_horizons.add(int(info.get("agent_motion_horizon_steps", -1)))
        observed_frequencies.add(float(info.get("agent_motion_freq_hz", -1.0)))
        names = np.asarray(info.get("gt_names", []), dtype=object).reshape(-1)
        boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 9)
        raw_trajectories = np.asarray(
            info.get("gt_agent_fut_trajs", []), dtype=np.float32
        )
        raw_masks = np.asarray(
            info.get("gt_agent_fut_masks", []), dtype=np.float32
        )
        if len(boxes):
            trajectories = raw_trajectories.reshape(len(boxes), -1, 2)
            masks = raw_masks.reshape(len(boxes), -1)
        else:
            trajectories = raw_trajectories.reshape(0, horizon, 2)
            masks = raw_masks.reshape(0, horizon)
        if not (len(names) == len(boxes) == len(trajectories) == len(masks)):
            raise RuntimeError(
                f"Detection/motion count mismatch in sample {info.get('folder')} "
                f"frame {info.get('frame_idx')}"
            )
        if trajectories.shape[1] != masks.shape[1]:
            raise RuntimeError("Agent trajectory and mask horizons differ")
        if trajectories.shape[1] < horizon:
            raise RuntimeError(
                f"Info stores {trajectories.shape[1]} motion steps, expected {horizon}"
            )

        trajectories = trajectories[:, :horizon]
        masks = masks[:, :horizon]
        distances = np.linalg.norm(boxes[:, :2], axis=1)
        finite = np.all(np.isfinite(trajectories), axis=(1, 2))
        full_valid = np.all(masks > 0, axis=1)
        nearby = distances < distance_threshold
        agent_trajectories = lidar_offsets_to_agent(trajectories, boxes)

        for class_index, name in enumerate(CLASSES):
            class_mask = names == name
            counts[name]["objects"] += int(class_mask.sum())
            counts[name]["full_valid"] += int((class_mask & full_valid).sum())
            counts[name]["within_distance"] += int((class_mask & nearby).sum())
            eligible = class_mask & full_valid & nearby & finite
            counts[name]["eligible"] += int(eligible.sum())
            if name not in DYNAMIC_CLASSES:
                continue
            for trajectory in agent_trajectories[eligible]:
                class_reservoirs[name].add(trajectory)
                group_reservoirs[_class_group(name)].add(trajectory)

    if observed_versions != {expected_info_version}:
        raise RuntimeError(
            f"Expected LEAD info version {expected_info_version}, got "
            f"{sorted(observed_versions)}"
        )
    if observed_horizons != {horizon}:
        raise RuntimeError(
            f"Expected stored motion horizon {horizon}, got {sorted(observed_horizons)}"
        )
    if observed_frequencies != {expected_motion_frequency}:
        raise RuntimeError(
            f"Expected motion frequency {expected_motion_frequency}Hz, got "
            f"{sorted(observed_frequencies)}"
        )
    return (
        class_reservoirs,
        group_reservoirs,
        counts,
        {
            "num_infos": info_count,
            "observed_info_versions": sorted(observed_versions),
            "observed_motion_horizons": sorted(observed_horizons),
            "observed_motion_frequencies_hz": sorted(observed_frequencies),
        },
    )


def _enough_unique(values: np.ndarray, num_modes: int) -> bool:
    if len(values) < num_modes:
        return False
    return len(np.unique(values.reshape(len(values), -1), axis=0)) >= num_modes


def build_anchors(
    class_reservoirs,
    group_reservoirs,
    *,
    horizon: int,
    num_modes: int,
    seed: int,
    n_init: int,
):
    anchors = np.zeros(
        (len(CLASSES), num_modes, horizon, 2), dtype=np.float32
    )
    strategies = {}
    for class_index, name in enumerate(CLASSES):
        if name in STATIC_CLASSES:
            strategies[name] = {
                "strategy": "static_zero",
                "source": name,
                "samples_used": 0,
            }
            continue

        values = class_reservoirs[name].array(horizon)
        source = name
        strategy = "class"
        if not _enough_unique(values, num_modes):
            source = FALLBACK_GROUPS[name]
            values = group_reservoirs[source].array(horizon)
            strategy = "dynamic_group_fallback"
        if not _enough_unique(values, num_modes):
            raise RuntimeError(
                f"Cannot generate {num_modes} non-degenerate anchors for dynamic "
                f"class {name}; class/group samples are insufficient"
            )
        flattened = values.reshape(len(values), horizon * 2)
        model = KMeans(
            n_clusters=num_modes,
            random_state=seed,
            n_init=n_init,
            algorithm="lloyd",
        )
        centers = model.fit(flattened).cluster_centers_.reshape(
            num_modes, horizon, 2
        )
        anchors[class_index] = centers.astype(np.float32)
        strategies[name] = {
            "strategy": strategy,
            "source": source,
            "samples_used": len(values),
            "inertia": float(model.inertia_),
        }
    return anchors, strategies


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reproducible class-wise LEAD motion anchors."
    )
    parser.add_argument(
        "--info-file",
        default="data/lead/infos_4hz_full_motion4hz8_v15/lead_infos_train.pkl",
    )
    parser.add_argument(
        "--output",
        default="data/kmeans/lead_motion_6_4hz8step.npy",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--distance-threshold", type=float, default=55.0)
    parser.add_argument("--max-samples-per-source", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--expected-info-version", type=int, default=15)
    parser.add_argument("--expected-motion-frequency", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    info_file = Path(args.info_file).resolve()
    output = Path(args.output).resolve()
    if not info_file.is_file():
        raise FileNotFoundError(f"Missing LEAD info manifest: {info_file}")
    if args.horizon <= 0 or args.num_modes <= 0:
        raise ValueError("--horizon and --num-modes must be positive")
    if args.max_samples_per_source < args.num_modes:
        raise ValueError("--max-samples-per-source must be at least --num-modes")

    class_reservoirs, group_reservoirs, counts, observed = collect_trajectories(
        info_file,
        horizon=args.horizon,
        distance_threshold=args.distance_threshold,
        capacity=args.max_samples_per_source,
        seed=args.seed,
        expected_info_version=args.expected_info_version,
        expected_motion_frequency=args.expected_motion_frequency,
    )
    anchors, strategies = build_anchors(
        class_reservoirs,
        group_reservoirs,
        horizon=args.horizon,
        num_modes=args.num_modes,
        seed=args.seed,
        n_init=args.n_init,
    )
    expected_shape = (len(CLASSES), args.num_modes, args.horizon, 2)
    if anchors.shape != expected_shape or not np.all(np.isfinite(anchors)):
        raise RuntimeError(f"Invalid generated anchor tensor: {anchors.shape}")
    for name in DYNAMIC_CLASSES:
        if not np.any(anchors[CLASSES.index(name)]):
            raise RuntimeError(f"Dynamic class {name} received all-zero anchors")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("wb") as stream:
        np.save(stream, anchors)
    temporary_output.replace(output)

    metadata = {
        "format_version": 1,
        "source_manifest": str(info_file),
        "output_anchor": str(output),
        "classes": CLASSES,
        "dynamic_classes": sorted(DYNAMIC_CLASSES),
        "static_classes": sorted(STATIC_CLASSES),
        "shape": list(anchors.shape),
        "info_version": args.expected_info_version,
        "horizon_steps": args.horizon,
        "motion_frequency_hz": args.expected_motion_frequency,
        "horizon_seconds": args.horizon / args.expected_motion_frequency,
        "num_modes": args.num_modes,
        "distance_threshold_m": args.distance_threshold,
        "seed": args.seed,
        "n_init": args.n_init,
        "max_samples_per_source": args.max_samples_per_source,
        "counts": counts,
        "strategies": strategies,
        **observed,
    }
    metadata_path = output.with_suffix(".json")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary_metadata.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_metadata.replace(metadata_path)
    print(f"Saved {anchors.shape} LEAD motion anchors to {output}")
    print(f"Saved generation metadata to {metadata_path}")


if __name__ == "__main__":
    main()
