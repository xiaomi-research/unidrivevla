"""Check locally generated nuScenes infos and Occ3D labels before training."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np


def check_stage1_data(dataset_root, infos_root, split):
    # These are the trusted, local PKLs written by scripts/create_data.sh.
    with (infos_root / f"nuscenes_infos_{split}.pkl").open("rb") as stream:
        annotations = pickle.load(stream)
    with (infos_root / f"vad_nuscenes_infos_temporal_{split}.pkl").open("rb") as stream:
        histories = {row["token"]: row.get("gt_ego_his_trajs") for row in pickle.load(stream)["infos"]}
    samples = annotations["infos"]
    if not samples:
        raise ValueError(f"No samples in nuscenes_infos_{split}.pkl")
    version = annotations["metadata"]["version"]
    if version not in ("v1.0-trainval", "v1.0-mini"):
        raise ValueError(f"Unexpected nuScenes version for Stage 1: {version}")
    with (dataset_root / version / "scene.json").open(encoding="utf-8") as stream:
        scenes = {row["token"]: row["name"] for row in json.load(stream)}

    failures = Counter()
    for sample in samples:
        token = sample["token"]
        history = np.asarray(histories.get(token), dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 2 or not history.size or not np.isfinite(history).all():
            failures["missing or invalid VAD history"] += 1
            if failures["missing or invalid VAD history"] <= 3:
                print(f"Missing or invalid VAD history for {token}")

        scene = scenes.get(sample["scene_token"])
        if scene is None:
            failures["missing scene metadata"] += 1
            if failures["missing scene metadata"] <= 3:
                print(f"Missing scene metadata for {token}")
            continue
        if any(value in ("", ".", "..") or Path(value).name != value for value in (scene, token)):
            raise ValueError(f"Invalid nuScenes path components for sample {token}")
        label_file = dataset_root / "gts" / scene / token / "labels.npz"
        try:
            with ZipFile(label_file) as archive:
                archive.getinfo("semantics.npy")
        except (OSError, BadZipFile, KeyError):
            failures["missing or invalid Occ3D archive"] += 1
            if failures["missing or invalid Occ3D archive"] <= 3:
                print(f"Missing or invalid Occ3D archive: {label_file}")

    print(f"Checked {len(samples)} nuScenes {split} samples")
    for problem, count in failures.items():
        print(f"{problem}: {count}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--infos-root", type=Path, default=Path("data/infos"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    try:
        return check_stage1_data(args.dataset_root, args.infos_root, args.split)
    except (OSError, ValueError, KeyError, pickle.UnpicklingError, EOFError) as exc:
        parser.exit(1, f"Could not check Stage 1 data: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
