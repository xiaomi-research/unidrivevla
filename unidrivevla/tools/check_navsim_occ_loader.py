"""CPU check for NAVSIM occupancy token injection and target preparation.

Run without arguments for a synthetic end-to-end wrapper check. Optionally pass both --cache-dir and --occ-root to
also check one matching sample from a trusted local NAVSIM cache.
"""

import argparse
import gzip
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from unidrivevla.adapters.unidrivevla import Profile, prepare_sample  # noqa: E402
from unidrivevla.data.navsim import WithTokens, _require_occ_vae  # noqa: E402

OCC_SHAPE = (200, 200, 16)
OCC_CLASSES = 9


class OneSample:
    def __init__(self, features, targets):
        self.features, self.targets = features, targets

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.features, self.targets


def _profile():
    path = PACKAGE / "config" / "profile" / "navsim.yaml"
    profile = Profile.from_config(OmegaConf.load(path))
    # Keep the real profile classes and adapter, with one tiny in-memory camera and no model construction.
    profile.cameras = profile.cameras[:1]
    profile.view_tokens = profile.view_tokens[:1]
    profile.image_hw = [32, 32]
    profile.max_boxes = profile.max_map = 1
    profile.map_points = 2
    profile.occ_format = "not-the-injected-format"
    return profile


def _features():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    return {
        "cameras": [{"view": "front", "image": image, "intrinsic": np.eye(3),
                     "sensor2ego": np.eye(4), "size": (32, 32)}],
        "history_trajectory": np.zeros((4, 3), dtype=np.float32),
        "ego_velocity": np.zeros(3, dtype=np.float32),
        "ego_acceleration": np.zeros(3, dtype=np.float32),
        "ego_angular_velocity": np.zeros(3, dtype=np.float32),
        "ego_steer": np.float32(np.nan),
        "command": "unknown",
    }


def _targets():
    return {
        "trajectory": np.zeros((8, 3), dtype=np.float32),
        "trajectory_mask": np.ones(8, dtype=np.float32),
        "task_valid": {"planning": True, "det": False, "map": False, "motion": False,
                       "occupancy": False, "geometry": False},
    }


def _prepare(features, targets, profile):
    return prepare_sample(features, targets, profile, image_augment=None, train=True)[1]


def _check_grid(labels, token):
    occ = labels.get("gt_occ_dense")
    if occ is None:
        raise AssertionError(f"{token}: gt_occ_dense is None")
    if tuple(occ.shape) != OCC_SHAPE or occ.dtype != torch.int64:
        raise AssertionError(f"{token}: expected int64 {OCC_SHAPE}, got {occ.dtype} {tuple(occ.shape)}")
    values = torch.unique(occ)
    if int(values.min()) < 0 or int(values.max()) >= OCC_CLASSES:
        raise AssertionError(f"{token}: expected class ids 0..{OCC_CLASSES - 1}, got {values.tolist()}")
    print(f"OK: token={token} shape={tuple(occ.shape)} dtype={occ.dtype} classes={values.tolist()}")


def _check_synthetic(profile):
    token = "synthetic-navsim-token"
    with tempfile.TemporaryDirectory(prefix="unidrivevla-navsim-occ-") as temp:
        root = Path(temp)
        occ_root = root / "occ"
        occ_root.mkdir()
        semantics = np.resize(np.arange(OCC_CLASSES, dtype=np.uint8), OCC_SHAPE)
        np.savez(occ_root / f"{token}.npz", semantics=semantics)

        wrapper = WithTokens(OneSample(_features(), _targets()), [token], occ_root, "occ3d")
        features, targets, actual_token = wrapper[0]
        if actual_token != token or targets.get("occ_format") != "occ3d":
            raise AssertionError("WithTokens did not preserve the token and propagate occ_format")
        expected = {"planning": True, "det": False, "map": False, "motion": False,
                    "occupancy": True, "geometry": False}
        if targets["task_valid"] != expected:
            raise AssertionError(f"unexpected task_valid after occ injection: {targets['task_valid']}")
        labels = _prepare(features, targets, profile)
        _check_grid(labels, token)
        if labels["gt_task_valid"] != expected:
            raise AssertionError("prepare_sample changed task masks while loading occupancy")

        missing_root = root / "no-occ"
        missing = WithTokens(OneSample(_features(), _targets()), [token], missing_root, "occ3d")
        missing_features, missing_targets, _ = missing[0]
        if "occ_path" in missing_targets or missing_targets["task_valid"]["occupancy"]:
            raise AssertionError("missing occupancy files must leave the cached supervision mask disabled")
        if _prepare(missing_features, missing_targets, profile)["gt_occ_dense"] is not None:
            raise AssertionError("missing occupancy files must not produce dense labels")

        existing = _targets()
        existing["occ_path"] = str(occ_root / f"{token}.npz")
        disabled = WithTokens(OneSample(_features(), existing), [token], occ_root, "occ3d")
        _, disabled_targets, _ = disabled[0]
        if disabled_targets["task_valid"]["occupancy"]:
            raise AssertionError("an existing explicit false occupancy flag must remain disabled")
        if _prepare(_features(), disabled_targets, profile)["gt_occ_dense"] is not None:
            raise AssertionError("existing explicit false occupancy supervision must not load a target")

    print("OK: occupancy file injection enables only its missing supervision; absent and explicitly disabled labels stay off")


def _check_vae_path_resolution(profile):
    class SourceAgent:
        def __init__(self, vae_path):
            self.profile = profile
            self.vae_path = vae_path

    class LegacyAgent:
        def __init__(self):
            self.profile = profile

    with tempfile.TemporaryDirectory(prefix="unidrivevla-navsim-vae-") as temp:
        root = Path(temp)
        global_path = root / "global-nuscenes.vae"
        source_path = root / "source-navsim.vae"
        global_path.touch()
        source_path.touch()
        cfg = OmegaConf.create({"agent": {"vae_path": str(global_path)}, "profile": {"occ_classes": OCC_CLASSES}})

        # The multidomain policy view's explicit source path wins over the shared/root setting.
        _require_occ_vae(cfg, SourceAgent(source_path), "train", root / "labels")

        # Explicit None disables this profile's checkpoint and must not fall back to the other profile's file.
        try:
            _require_occ_vae(cfg, SourceAgent(None), "train", root / "labels")
        except ValueError as exc:
            if "source-specific" not in str(exc):
                raise
        else:
            raise AssertionError("an explicit source vae_path=None incorrectly used cfg.agent.vae_path")

        # Older policies without a vae_path attribute retain the pre-multidomain config fallback.
        _require_occ_vae(cfg, LegacyAgent(), "val", root / "labels")

        # Inference on the unlabeled test split and disabled occupancy do not require a VAE checkpoint.
        _require_occ_vae(cfg, SourceAgent(None), "test", root / "labels")
        _require_occ_vae(cfg, SourceAgent(None), "train", None)

        try:
            _require_occ_vae(cfg, SourceAgent(root / "missing.vae"), "train", root / "labels")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a nonexistent source VAE path must fail before training")

    print("OK: occupancy preflight uses the per-source VAE, honors explicit None, and keeps legacy fallback")


def _check_real_cache(cache_dir, occ_root, profile):
    cache_dir, occ_root = Path(cache_dir).expanduser().resolve(), Path(occ_root).expanduser().resolve()
    for target_path in cache_dir.rglob("unidrivevla_sample_target.gz"):
        sample_dir = target_path.parent
        token = sample_dir.name
        feature_path = sample_dir / "unidrivevla_sample_feature.gz"
        if not feature_path.is_file():
            continue
        try:
            wrapper = WithTokens(OneSample({}, {}), [token], occ_root, "occ3d")
            if wrapper._occ_path(token) is None:
                continue
        except ValueError:
            continue
        with gzip.open(target_path, "rb") as stream:
            targets = pickle.load(stream)  # trusted local NAVSIM cache only
        if targets.get("occ_path"):
            continue  # this optional check targets labels injected after the original cache was written
        with gzip.open(feature_path, "rb") as stream:
            features = pickle.load(stream)  # trusted local NAVSIM cache only
        wrapper = WithTokens(OneSample(features, targets), [token], occ_root, "occ3d")
        features, targets, _ = wrapper[0]
        labels = _prepare(features, targets, profile)
        _check_grid(labels, token)
        flags = labels["gt_task_valid"]
        if not flags["occupancy"]:
            raise AssertionError(f"{token}: real cache occupancy was not enabled after a matching file was found")
        print(f"OK: real cache occupancy mask={flags}; root={occ_root}")
        return
    raise FileNotFoundError(f"no trusted NAVSIM cache token in {cache_dir} has a flat label under {occ_root}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", help="optional trusted local NAVSIM feature/target cache")
    parser.add_argument("--occ-root", help="optional flat directory of token-named semantics NPZ files")
    args = parser.parse_args(argv)
    if bool(args.cache_dir) != bool(args.occ_root):
        parser.error("--cache-dir and --occ-root must be supplied together")

    profile = _profile()
    _check_synthetic(profile)
    _check_vae_path_resolution(profile)
    if args.cache_dir:
        _check_real_cache(args.cache_dir, args.occ_root, profile)


if __name__ == "__main__":
    main()
