"""python tools/data_stats.py data=navsim|nuscenes|b2d [hydra overrides] [--stride N]

Mean and std of the per-step ego displacement (x, y) over the profile's horizon on the training split: the profile's
delta_mean / delta_std, which normalize the planning head's flow-matching targets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from unidrivevla.adapters.unidrivevla import Profile  # noqa: E402
from unidrivevla.data import LOADERS, loader_name  # noqa: E402


def compose(overrides):
    from hydra import compose as hydra_compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PACKAGE / "config"), version_base=None):
        return hydra_compose(config_name="train", overrides=overrides)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1, help="use every N-th training sample")
    args, overrides = parser.parse_known_args()
    cfg = compose(overrides)
    profile = Profile.from_config(cfg.profile)
    dataset = LOADERS[loader_name(cfg)](cfg, None, "train")
    deltas = []
    for index in range(0, len(dataset), args.stride):
        _, targets, _ = dataset[index]
        xy = np.asarray(targets["trajectory"], np.float64)[: profile.horizon, :2]
        valid = np.asarray(targets["trajectory_mask"])[: profile.horizon] > 0
        step = np.diff(xy, axis=0, prepend=np.zeros((1, 2)))
        deltas.append(step[valid & np.concatenate([[True], valid[:-1]])])
    deltas = np.concatenate(deltas)
    mean, std = deltas.mean(0), deltas.std(0)
    print(f"{len(deltas)} steps of {len(range(0, len(dataset), args.stride))} samples")
    print(f"delta_mean: [{mean[0]:.4f}, {mean[1]:.4f}]\ndelta_std: [{std[0]:.4f}, {std[1]:.4f}]")


if __name__ == "__main__":
    main()
