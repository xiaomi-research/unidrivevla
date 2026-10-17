"""python tools/fit_anchors.py data=navsim|nuscenes|b2d [hydra overrides] [--stride N] [--out DIR]

k-means anchors of a source's training split, written to the profile's frame-specific anchors_dir:
  det_900.npy     (900, 11)                box centres within 55 m; log size 1, heading forward, no velocity
  map_100.npy     (100, 20, 2)             polyline centres, each an 8 m segment along the heading (Bench2Drive: every
                                           100th sample, 8-20 m segments, 70 % along the heading, the rest across)
  motion_6.npy    (classes, 6, steps, 2)   per class, full-length futures within 55 m in the agent's own frame
Native LiDAR mode reads raw main annotations and follows each dataset's official k-means settings. Its output directory
is separate from legacy ego-frame anchors. Existing files are never replaced unless --replace-existing is supplied.
Run it on the full training split: anchors of a subset are not the official ones.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from unidrivevla.adapters.unidrivevla import Profile  # noqa: E402
from unidrivevla.data import LOADERS, loader_name  # noqa: E402

K_DET, K_MAP, K_MOTION = 900, 100, 6
DIST_M = 55.0


def compose(overrides):
    from hydra import compose as hydra_compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PACKAGE / "config"), version_base=None):
        return hydra_compose(config_name="train", overrides=overrides)


def kmeans(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans

    points = np.asarray(points, dtype=np.float64)
    if len(points) < k:
        raise RuntimeError(f"{len(points)} points for {k} clusters")
    # The main scripts use KMeans(n_clusters=k) with sklearn defaults (random_state=None).
    model = KMeans(n_clusters=k) if seed is None else KMeans(n_clusters=k, random_state=seed)
    return model.fit(points).cluster_centers_


def map_segments(centres: np.ndarray, points: int, b2d: bool, seed: int = 0, native: bool = False) -> np.ndarray:
    """(K, points, 2): each centre spread into a polyline, as the official kmeans_map scripts."""
    if not b2d:
        offsets = np.stack([np.zeros(points), np.linspace(-4, 4, points)], -1) if native else \
            np.stack([np.linspace(-4, 4, points), np.zeros(points)], -1)
        return centres[:, None] + offsets[None]
    if native:
        out = []
        for centre in centres:
            length = np.random.uniform(8, 20, 10)[0]
            along = np.random.uniform() < 0.7
            line = np.linspace(-length / 2, length / 2, points)
            out.append(centre + np.stack([np.zeros(points), line] if along else [line, np.zeros(points)], -1))
        return np.stack(out)
    rng = np.random.default_rng(seed)
    out = []
    for centre in centres:
        line = np.linspace(-0.5, 0.5, points) * rng.uniform(8, 20)
        along = rng.uniform() < 0.7
        out.append(centre + np.stack([line, np.zeros(points)] if along else [np.zeros(points), line], -1))
    return np.stack(out)


def _native_map_centres(dataset, index: int, b2d: bool) -> list:
    """The raw polyline means used by main's two map-anchor scripts."""
    info = dataset.infos[index]
    if not b2d:
        return [np.asarray(line, np.float64)[:, :2].mean(axis=0)
                for lines in info["map_annos"].values() for line in lines]

    town = dataset.maps[info["town_name"]]
    world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], np.float64)
    ego_xy = np.linalg.inv(world2lidar)[:2, 3]
    chosen = [i for i, points in enumerate(town["lane_sample_points"])
              if np.min(np.linalg.norm(np.asarray(points)[:, :2] - ego_xy, axis=-1)) < 50]
    kinds = {"Broken", "Solid", "SolidSolid", "Center", "TrafficLight", "StopSign"}
    centres = []
    for i in chosen:
        if town["lane_types"][i] not in kinds:
            continue
        points = np.asarray(town["lane_points"][i], np.float64)
        homogeneous = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)
        points = (homogeneous @ world2lidar.T)[:, :2]
        inside = ((points[:, 0] > -15) & (points[:, 0] < 15)
                  & (points[:, 1] > -30) & (points[:, 1] < 30))
        if inside.sum() > 1:
            centres.append(points[inside].mean(axis=0))
    for points, kind in zip(town["trigger_volumes_points"], town["trigger_volumes_types"]):
        if kind not in kinds:
            continue
        points = np.asarray(points, np.float64)
        homogeneous = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)
        points = (homogeneous @ world2lidar.T)[:, :2]
        inside = ((points[:, 0] > -15) & (points[:, 0] < 15)
                  & (points[:, 1] > -30) & (points[:, 1] < 30))
        if inside.all():
            centres.append(np.concatenate([points, points[:1]], axis=0).mean(axis=0))
    return centres


def _native_motion(dataset, profile: Profile, b2d: bool, stride: int = 1) -> list:
    """Raw LiDAR motion anchors using the official script's eligibility and lidar2agent math."""
    motion = [[] for _ in profile.det_classes]
    infos = dataset.infos
    steps = profile.motion_steps
    for index, info in enumerate(infos):
        if index % stride:
            continue
        boxes = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 9 if b2d else 7)
        names = [str(n) for n in info["gt_names"]]
        if b2d:
            from unidrivevla.data.bench2drive import NAME_MAPPING, STEP_FRAMES

            names = [NAME_MAPPING.get(str(n), str(n)) for n in info["gt_names"]]
            ids = np.asarray(info["gt_ids"])
            track = np.zeros((len(boxes), steps + 1, 2), np.float64)
            seen = np.zeros((len(boxes), steps + 1), np.float64)
            world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], np.float64)
            for step in range(steps + 1):
                other_index = index + step * STEP_FRAMES
                if other_index >= len(infos):
                    break
                other = infos[other_index]
                if other["folder"] != info["folder"]:
                    break
                where = {int(box_id): row for row, box_id in enumerate(other["gt_ids"])}
                for row, box_id in enumerate(ids):
                    other_row = where.get(int(box_id))
                    if other_row is None:
                        continue
                    track[row, step] = (world2lidar @ np.asarray(other["npc2world"][other_row]))[:2, 3]
                    seen[row, step] = 1
            offsets, valid = track[:, 1:] - track[:, :-1], seen[:, 1:]
        else:
            offsets = np.asarray(info["gt_agent_fut_trajs"], np.float64)[:, :steps]
            valid = np.asarray(info["gt_agent_fut_masks"], np.float64)[:, :steps]
        if len(boxes) == 0:
            continue
        near = np.linalg.norm(boxes[:, :2], axis=1) < DIST_M
        complete = valid.sum(axis=1) == steps
        for row, name in enumerate(names):
            label = profile.det_label(name)
            if label is None or not near[row] or not complete[row]:
                continue
            delta = np.concatenate([np.zeros((1, 2)), np.cumsum(offsets[row], axis=0)], axis=0)[1:]
            yaw = boxes[row, 6]
            c, s = math.cos(yaw), math.sin(yaw)
            motion[label].append(np.stack([c * delta[:, 0] + s * delta[:, 1],
                                           -s * delta[:, 0] + c * delta[:, 1]], axis=1).reshape(-1))
    return motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1, help="use every N-th training sample")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replace-existing", action="store_true", help="replace existing anchor files explicitly")
    parser.add_argument("--development-fill-missing-motion", action="store_true",
                        help="fill underpopulated native motion classes with zero anchors for smoke construction only")
    args, overrides = parser.parse_known_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    cfg = compose(overrides)
    profile = Profile.from_config(cfg.profile)
    dataset = LOADERS[loader_name(cfg)](cfg, None, "train")
    steps = profile.motion_steps
    b2d = profile.name == "b2d"
    native = profile.frame == "lidar"
    seed = None if native else 0  # native follows the official KMeans(n_clusters=k) defaults
    centres, lines, motion = [], [], [[] for _ in profile.det_classes]
    out = args.out or Path(profile.anchors_dir)
    names = ("det_900.npy", "map_100.npy", "motion_6.npy")
    existing = [name for name in names if (out / name).exists()]
    if existing and not args.replace_existing:
        raise FileExistsError(f"refusing to replace existing anchor files under {out}: {existing}; pass --replace-existing explicitly")

    if native:
        for index in range(0, len(dataset), args.stride):
            info = dataset.infos[index]
            boxes = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 9 if b2d else 7)
            if len(boxes):
                near = np.linalg.norm(boxes[:, :2], axis=1) < DIST_M
                centres.append(boxes[near, :3])
            if not b2d or index % 100 == 0:  # the official Bench2Drive script reads data_infos[::100]
                lines.extend(_native_map_centres(dataset, index, b2d))
        motion = _native_motion(dataset, profile, b2d, stride=args.stride)
    else:
        for index in range(0, len(dataset), args.stride):
            _, targets, _ = dataset[index]
            boxes = np.asarray(targets["boxes"], np.float64).reshape(-1, 7)
            near = np.hypot(boxes[:, 0], boxes[:, 1]) < DIST_M
            centres.append(boxes[near, :3])
            if not b2d or index % 100 == 0:  # its kmeans_map reads every 100th sample
                lines += [np.asarray(c, np.float64).mean(0) for name, c in targets["map"]
                          if profile.map_label(name) is not None]
            future = np.asarray(targets["agent_future"], np.float64)
            mask = np.asarray(targets["agent_future_mask"])
            if future.ndim != 3 or future.shape[1] < steps:
                continue
            for row in np.where(near & (mask[:, :steps].sum(1) == steps))[0]:
                label = profile.det_label(targets["box_names"][row])
                if label is None:
                    continue
                x, y, yaw = boxes[row, [0, 1, 6]]
                rel = future[row, :steps] - (x, y)
                c, s = math.cos(yaw), math.sin(yaw)
                motion[label].append(np.stack([c * rel[:, 0] + s * rel[:, 1],
                                               -s * rel[:, 0] + c * rel[:, 1]], 1).reshape(-1))

    det = kmeans(np.concatenate(centres), K_DET, seed=seed)
    det_suffix = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0] if native else \
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    det = np.concatenate([det, np.tile(det_suffix, (K_DET, 1))], axis=1).astype(np.float32)

    map_centres = kmeans(np.stack(lines), K_MAP, seed=seed)
    map_anchors = map_segments(map_centres, profile.map_points, b2d, native=native).astype(np.float32)

    clusters = []
    for name, tracks in zip(profile.det_classes, motion):
        if len(tracks) < K_MOTION:
            if native and not args.development_fill_missing_motion:
                raise RuntimeError(f"native motion anchor fit needs {K_MOTION} full tracks for {name}, found {len(tracks)}; "
                                   "no zero modes are fabricated; the explicit development flag is for smoke data only")
            print(f"motion: {name} has {len(tracks)} full tracks, using zero development anchors")
            clusters.append(np.zeros((K_MOTION, steps, 2)))
        else:
            clusters.append(kmeans(np.stack(tracks), K_MOTION, seed=seed).reshape(K_MOTION, steps, 2))
    motion_anchors = np.stack(clusters).astype(np.float32)

    out.mkdir(parents=True, exist_ok=True)
    for name, array in zip(names, (det, map_anchors, motion_anchors)):
        path = out / name
        if path.exists() and not args.replace_existing:
            raise FileExistsError(f"refusing to replace existing anchor file {path}; pass --replace-existing explicitly")
        np.save(path, array)
    print(f"anchors of {len(range(0, len(dataset), args.stride))} samples ({sum(map(len, centres))} boxes, "
          f"{len(lines)} lines, {sum(map(len, motion))} tracks) -> {out}")


if __name__ == "__main__":
    main()
