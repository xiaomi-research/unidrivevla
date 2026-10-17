"""CPU check: verify NAVSIM cache + occ_root produces a loadable occ label."""
import argparse
import gzip
import os
import pickle
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/data/yongkangli/UniDriveVLA")
from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy, prepare_sample
from unidrivevla.data.navsim import normalize_navsim_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--occ-root", required=True)
    parser.add_argument("--policy-cfg", default="/data/yongkangli/UniDriveVLA/unidrivevla/config/agent/unidrivevla.yaml")
    args = parser.parse_args()

    profile_cfg = OmegaConf.load("/data/yongkangli/UniDriveVLA/unidrivevla/config/profile/navsim.yaml")
    agent_cfg = OmegaConf.load(args.policy_cfg)
    agent_cfg.vae_path = "/data/yongkangli/UniDriveVLA/NavSim/data/vae/occvae_navsim.pth"
    agent_cfg.profile = profile_cfg
    agent = UniDriveVLAPolicy(agent_cfg)

    targets = occ_path = None
    token = None
    for root, dirs, files in os.walk(args.cache_dir):
        if "unidrivevla_sample_target.gz" in files and "unidrivevla_sample_feature.gz" in files:
            with gzip.open(os.path.join(root, "unidrivevla_sample_feature.gz"), "rb") as f:
                features = pickle.load(f)
            with gzip.open(os.path.join(root, "unidrivevla_sample_target.gz"), "rb") as f:
                targets = pickle.load(f)
            token = os.path.basename(root)
            if not os.path.isfile(os.path.join(args.occ_root, f"{token}.npz")):
                continue
            break

    if targets is None:
        print("FAIL: no cache sample with occ label found")
        sys.exit(1)

    targets = dict(targets)
    targets["occ_path"] = os.path.join(args.occ_root, f"{token}.npz")
    targets["occ_format"] = "occ3d"
    targets = normalize_navsim_targets(targets)

    try:
        out, labels = prepare_sample(features, targets, agent.profile, image_augment=None, train=True)
    except Exception as exc:
        print(f"FAIL: prepare_sample raised {exc}")
        raise

    occ = labels.get("gt_occ_dense")
    if occ is None:
        print("FAIL: gt_occ_dense is None")
        sys.exit(1)

    print(f"OK: token={token} occ shape={occ.shape} dtype={occ.dtype} unique={torch.unique(occ).tolist()}")


if __name__ == "__main__":
    main()
