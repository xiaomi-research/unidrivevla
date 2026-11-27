"""Import or verify the public-main nuScenes anchor baseline for PR-10 reproduction.

Original files live under nuScenes/data/kmeans and are intentionally not regenerated here.
This verifies the selected public source; it cannot establish which private experiment files were used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


SCHEMA = "unidrivevla.nuscenes-anchor-manifest/v1"
REFERENCE = "PR-10 public-main baseline 5fce121; see unidrivevla/docs/NUSCENES_ANCHORS.md"
ARRAYS = (
    ("kmeans_det_900.npy", "det_900.npy", (900, 11)),
    ("kmeans_map_100.npy", "map_100.npy", (100, 20, 2)),
    ("kmeans_motion_6.npy", "motion_6.npy", (10, 6, 12, 2)),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash shape, numeric dtype, and C-order values; independent of .npy header metadata."""
    array = np.asarray(array)
    if array.dtype.kind != "f":
        raise ValueError(f"anchors must use a floating dtype, got {array.dtype}")
    little_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(little_dtype, copy=False))
    header = json.dumps({"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header + b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _inspect(path: Path, expected_shape) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing anchor array: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot load anchor array {path}: {type(exc).__name__}: {exc}") from exc
    if array.shape != expected_shape:
        raise ValueError(f"{path.name} has shape {array.shape}; expected {expected_shape}")
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{path.name} must contain only finite floating point values; got {array.dtype}")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "array_sha256": array_sha256(array),
        "file_sha256": _file_sha256(path),
    }


def inspect_directory(directory: Path, *, original_names: bool) -> dict:
    report = {}
    for source_name, target_name, shape in ARRAYS:
        name = source_name if original_names else target_name
        report[target_name] = {"source_name": source_name, **_inspect(directory / name, shape)}
    return report


def _compare(report: dict, expected: dict) -> None:
    arrays = expected.get("arrays")
    if expected.get("schema") != SCHEMA or not isinstance(arrays, dict):
        raise ValueError("anchor manifest has an unsupported or invalid schema")
    for _, target_name, shape in ARRAYS:
        actual = report[target_name]
        baseline = arrays.get(target_name)
        if not isinstance(baseline, dict):
            raise ValueError(f"anchor manifest does not contain {target_name}")
        for key in ("shape", "dtype", "array_sha256"):
            if actual[key] != baseline.get(key):
                raise ValueError(f"{target_name} {key} differs from the expected manifest")
        if baseline.get("file_sha256") is not None and actual["file_sha256"] != baseline["file_sha256"]:
            raise ValueError(f"{target_name} file SHA differs from the expected manifest")


def _promote(staged: Path, destination: Path, *, replace_existing: bool) -> None:
    if replace_existing:
        os.replace(staged, destination)
    else:
        # Atomic exclusive creation: a concurrent file cannot slip past the earlier collision check.
        os.link(staged, destination)


def import_original(source_dir: Path, target_dir: Path, *, replace_existing: bool, expected_manifest=None,
                    expected_manifest_path=None) -> dict:
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    source_report = inspect_directory(source_dir, original_names=True)
    if expected_manifest is not None:
        _compare(source_report, expected_manifest)
    manifest_path = target_dir / "nuscenes_original_anchor_manifest.json"
    existing_report = {}
    preserved_identical = []
    missing = []
    if not replace_existing:
        for _, target_name, shape in ARRAYS:
            path = target_dir / target_name
            if not path.exists():
                missing.append(target_name)
                continue
            actual = _inspect(path, shape)
            expected = source_report[target_name]
            if any(actual[key] != expected[key] for key in ("shape", "dtype", "array_sha256", "file_sha256")):
                raise FileExistsError(
                    f"refusing to overwrite conflicting anchor {path}; its file SHA/content differ from source"
                )
            existing_report[target_name] = {"source_name": expected["source_name"], **actual}
            preserved_identical.append(target_name)
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _compare(source_report, existing_manifest)
        if not missing and manifest_path.exists():
            return {"manifest": str(manifest_path), "arrays": existing_report,
                    "copied": [], "preserved_identical": preserved_identical,
                    "manifest_preserved": True}
    else:
        missing = [target_name for _, target_name, _ in ARRAYS]

    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anchor-import-", dir=target_dir) as temp:
        staging = Path(temp)
        for source_name, target_name, _ in ARRAYS:
            if target_name in missing:
                shutil.copyfile(source_dir / source_name, staging / target_name)
        staged_report = {
            target_name: {"source_name": source_report[target_name]["source_name"],
                          **_inspect(staging / target_name, shape)}
            for _, target_name, shape in ARRAYS if target_name in missing
        }
        target_report = {**existing_report, **staged_report}
        _compare(target_report, {"schema": SCHEMA, "arrays": source_report})
        manifest_exists = manifest_path.is_file()
        if not manifest_exists or replace_existing:
            manifest = {
                "schema": SCHEMA,
                "reference": REFERENCE,
                "source_dir": str(source_dir),
                "verified_against": (str(expected_manifest_path.resolve()) if expected_manifest_path else None),
                "arrays": target_report,
            }
            (staging / manifest_path.name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                       encoding="utf-8")
        for _, target_name, _ in ARRAYS:
            if target_name in missing:
                _promote(staging / target_name, target_dir / target_name, replace_existing=replace_existing)
        if not manifest_exists or replace_existing:
            _promote(staging / manifest_path.name, manifest_path, replace_existing=replace_existing)
    return {"manifest": str(manifest_path), "arrays": target_report,
            "copied": list(missing), "preserved_identical": preserved_identical,
            "manifest_preserved": bool(manifest_exists and not replace_existing)}


def verify_target(target_dir: Path, manifest_path=None) -> dict:
    target_dir = target_dir.expanduser().resolve()
    manifest_path = manifest_path or target_dir / "nuscenes_original_anchor_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no recorded anchor baseline at {manifest_path}; import the trusted original files first or pass --manifest"
        )
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = inspect_directory(target_dir, original_names=False)
    _compare(report, expected)
    return {"verified_against": str(manifest_path.resolve()), "arrays": report}


def self_test() -> None:
    """Exercise shape, hashing, copy, manifest verification, and corruption detection without nuScenes data."""
    rng = np.random.default_rng(2026)
    arrays = (
        rng.standard_normal((900, 11), dtype=np.float32),
        rng.standard_normal((100, 20, 2), dtype=np.float32),
        rng.standard_normal((10, 6, 12, 2), dtype=np.float32),
    )
    if array_sha256(arrays[1]) != array_sha256(np.asfortranarray(arrays[1])):
        raise AssertionError("array hash must ignore memory layout")
    with tempfile.TemporaryDirectory(prefix="unidrivevla-nusc-anchor-check-") as temp:
        root = Path(temp)
        source, target = root / "original", root / "native"
        source.mkdir()
        for (source_name, _, _), array in zip(ARRAYS, arrays):
            np.save(source / source_name, array, allow_pickle=False)
        baseline = {"schema": SCHEMA, "arrays": inspect_directory(source, original_names=True)}
        baseline_path = root / "trusted-reference.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        target.mkdir()
        shutil.copyfile(source / ARRAYS[0][0], target / ARRAYS[0][1])  # interrupted import: one complete file, no manifest
        imported = import_original(source, target, replace_existing=False, expected_manifest=baseline,
                                   expected_manifest_path=baseline_path)
        verified = verify_target(target, baseline_path)
        if imported["arrays"] != verified["arrays"]:
            raise AssertionError("copied arrays do not match their import manifest")
        if imported["copied"] != ["map_100.npy", "motion_6.npy"] or imported["preserved_identical"] != ["det_900.npy"]:
            raise AssertionError(f"interrupted import was not completed idempotently: {imported}")
        repeated = import_original(source, target, replace_existing=False, expected_manifest=baseline)
        if repeated["copied"] or len(repeated["preserved_identical"]) != 3 or not repeated["manifest_preserved"]:
            raise AssertionError(f"identical repeated import did not preserve existing files: {repeated}")
        damaged = arrays[-1].copy()
        damaged[0, 0, 0, 0] += 1
        np.save(target / ARRAYS[-1][1], damaged, allow_pickle=False)
        conflict_sha = _file_sha256(target / ARRAYS[-1][1])
        try:
            import_original(source, target, replace_existing=False, expected_manifest=baseline)
        except FileExistsError:
            pass
        else:
            raise AssertionError("importer silently replaced a conflicting same-name anchor")
        if _file_sha256(target / ARRAYS[-1][1]) != conflict_sha:
            raise AssertionError("a conflicting existing anchor was modified")
        try:
            verify_target(target)
        except ValueError as exc:
            if "sha" not in str(exc).lower():
                raise AssertionError(f"corruption was rejected for the wrong reason: {exc}") from exc
        else:
            raise AssertionError("changed motion anchor values passed the manifest check")
        np.save(source / ARRAYS[-1][0], arrays[-1][:, :, :, :1], allow_pickle=False)
        try:
            inspect_directory(source, original_names=True)
        except ValueError as exc:
            if "shape" not in str(exc):
                raise AssertionError(f"bad motion shape was rejected for the wrong reason: {exc}") from exc
        else:
            raise AssertionError("motion anchor shape other than (10, 6, 12, 2) was accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        help="public-main nuScenes/data/kmeans source (kmeans_*.npy names)")
    parser.add_argument("--target-dir", type=Path,
                        help="native nuScenes anchors directory (det_900.npy, map_100.npy, motion_6.npy)")
    parser.add_argument("--manifest", type=Path,
                        help="independent expected manifest; for imports, compare the source before copying")
    parser.add_argument("--write-manifest", type=Path,
                        help="write a manifest of --source-dir for later independent verification")
    parser.add_argument("--replace-existing", action="store_true", help="allow replacing imported arrays and manifest")
    parser.add_argument("--self-test", action="store_true", help="run the synthetic CPU contract check")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("PASS: nuScenes anchor shape/hash/idempotent-import/conflict-preservation/manifest contract")
        return
    if args.source_dir is None and args.target_dir is None:
        parser.error("provide --source-dir to inspect/import originals or --target-dir to verify an import")
    if args.source_dir is not None:
        source_report = inspect_directory(args.source_dir.expanduser().resolve(), original_names=True)
        expected = (json.loads(args.manifest.read_text(encoding="utf-8"))
                    if args.manifest is not None else None)
        if expected is not None:
            _compare(source_report, expected)
        if args.target_dir is None:
            result = {"source_dir": str(args.source_dir.expanduser().resolve()), "arrays": source_report}
            if args.write_manifest is not None:
                path = args.write_manifest.expanduser().resolve()
                if path.exists() and not args.replace_existing:
                    raise FileExistsError(f"refusing to replace manifest {path}; pass --replace-existing explicitly")
                path.parent.mkdir(parents=True, exist_ok=True)
                baseline = {"schema": SCHEMA, "reference": REFERENCE,
                            "source_dir": str(args.source_dir.expanduser().resolve()),
                            "arrays": source_report}
                mode = "w" if args.replace_existing else "x"
                with path.open(mode, encoding="utf-8") as stream:
                    stream.write(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
                result["manifest"] = str(path)
        else:
            result = import_original(args.source_dir, args.target_dir, replace_existing=args.replace_existing,
                                     expected_manifest=expected, expected_manifest_path=args.manifest)
    else:
        result = verify_target(args.target_dir, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
