"""python tools/check_sources.py data=navsim|nuscenes|b2d [hydra overrides] [--split train] [--samples 4]
                                [--token TOKEN --old-cache DIR]

Checks a data source end to end without the network: samples follow the canonical contract, the adapter's
prepare_sample gives the planning head's shapes, and the geometry is consistent (box centres within 30 m project into
some camera). With --old-cache, the token's decoded images and projection matrices must equal the old NAVSIM feature
cache (unidrivevla_feature.gz of the previous builder) exactly. Exits 1 on any failure.
"""

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from unidrivevla.adapters.unidrivevla import Profile, prepare_sample  # noqa: E402
from unidrivevla.data import LOADERS, loader_name  # noqa: E402
from unidrivevla.data.sample import check_sample  # noqa: E402


def compose(overrides):
    from hydra import compose as hydra_compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PACKAGE / "config"), version_base=None):
        return hydra_compose(config_name="train", overrides=overrides)


def expected_shapes(profile: Profile) -> dict:
    views, (height, width), points = len(profile.cameras), profile.image_hw, profile.map_points
    return {
        "img": (views, 3, height, width), "projection_mat": (views, 4, 4), "image_wh": (views, 2), "focal": (views,),
        "ego_status": (len(profile.status_layout),), "hist_traj": (4, 2),
        "gt_ego_fut_trajs": (profile.horizon, 2), "gt_ego_fut_masks": (profile.horizon,),
        "gt_bboxes_3d": (profile.max_boxes, 9), "gt_labels_3d": (profile.max_boxes,),
        "gt_agent_fut_trajs": (profile.max_boxes, profile.motion_steps, 2),
        "gt_map_pts": (profile.max_map, 2 * (points - 1), points, 2), "gt_map_labels": (profile.max_map,),
    }


def visible_fraction(features: dict, targets: dict) -> float:
    """Share of boxes within 30 m whose centre lands inside at least one image in front of its camera."""
    boxes = np.asarray(targets["boxes"], np.float64).reshape(-1, 7)
    boxes = boxes[np.hypot(boxes[:, 0], boxes[:, 1]) < 30.0]
    if not len(boxes):
        return float("nan")
    seen = np.zeros(len(boxes), bool)
    centres = np.concatenate([boxes[:, :3], np.ones((len(boxes), 1))], axis=1)
    for cam in features["cameras"]:
        view = np.eye(4)
        view[:3, :3] = cam["intrinsic"]
        uvz = centres @ (view @ np.linalg.inv(cam["sensor2ego"])).T
        z = uvz[:, 2]
        u, v = uvz[:, 0] / np.maximum(z, 1e-6), uvz[:, 1] / np.maximum(z, 1e-6)
        seen |= (z > 0.5) & (u >= 0) & (u < cam["size"][0]) & (v >= 0) & (v < cam["size"][1])
    return float(seen.mean())


def compare_old_cache(dataset, token: str, old_cache: Path, profile: Profile) -> list:
    files = list(old_cache.glob(f"*/{token}/unidrivevla_feature.gz"))
    if not files:
        return [f"no unidrivevla_feature.gz for {token} under {old_cache}"]
    with gzip.open(files[0], "rb") as f:
        old = pickle.load(f)
    features, _, _ = dataset[list(dataset.tokens).index(token)]
    new, _ = prepare_sample(features, {}, profile)
    problems = []
    if "img" in old:
        if not torch.equal(torch.as_tensor(old["img"]).float(), new["img"].float()):
            diff = (torch.as_tensor(old["img"]).float() - new["img"].float()).abs().max().item()
            problems.append(f"img differs from the old cache (max abs diff {diff})")
    else:
        problems.append("the old cache holds no decoded img (a path cache); pixels not compared")
    if not torch.allclose(torch.as_tensor(old["projection_mat"]).float(), new["projection_mat"], atol=1e-3):
        problems.append("projection_mat differs from the old cache")
    print(f"old cache {files[0]}: {'OK' if not problems else problems}")
    return [p for p in problems if "not compared" not in p]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--token")
    parser.add_argument("--old-cache", type=Path)
    args, overrides = parser.parse_known_args()
    cfg = compose(overrides)
    profile = Profile.from_config(cfg.profile)
    dataset = LOADERS[loader_name(cfg)](cfg, None, args.split)
    print(f"{loader_name(cfg)} {args.split}: {len(dataset)} samples, profile {profile.name}")
    problems, shapes = [], expected_shapes(profile)
    for index in np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset))).astype(int):
        features, targets, token = dataset[int(index)]
        found = check_sample(features, targets)
        head_features, head_targets = prepare_sample(features, targets, profile)
        for key, shape in shapes.items():
            value = {**head_features, **head_targets}.get(key)
            if value is None and key in ("gt_ego_fut_trajs", "gt_bboxes_3d") and not targets:
                continue
            ragged = key in {"gt_bboxes_3d", "gt_labels_3d", "gt_agent_fut_trajs", "gt_map_pts", "gt_map_labels"}
            matches = value is not None and (tuple(value.shape) == shape or (
                ragged and value.ndim == len(shape) and value.shape[0] >= shape[0] and tuple(value.shape[1:]) == shape[1:]))
            if not matches:
                found.append(f"{key}: {None if value is None else tuple(value.shape)}, expected {shape}")
        for key, value in {**head_features, **head_targets}.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                found.append(f"{key} is not finite")
        fraction = visible_fraction(features, targets) if targets else float("nan")
        if fraction == fraction and fraction < 0.5:
            found.append(f"only {fraction:.0%} of the boxes within 30 m project into a camera")
        kept = lambda mask: int(head_targets[mask].sum()) if targets else 0  # noqa: E731
        if targets and kept("map_mask") > profile.max_map:
            print(f"  {token}: retained {kept('map_mask')} map rows beyond the padding minimum {profile.max_map}")
        print(f"  {token}: {len(targets.get('boxes', []))} boxes ({kept('box_mask')} kept), "
              f"{len(targets.get('map', []))} map lines ({kept('map_mask')} in the ROI), command {features['command']}, "
              f"visible {fraction:.0%}{'' if not found else ' -> ' + '; '.join(found)}")
        problems += found
    if args.old_cache and args.token:
        problems += compare_old_cache(dataset, args.token, args.old_cache, profile)
    print("OK" if not problems else f"FAILED: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
