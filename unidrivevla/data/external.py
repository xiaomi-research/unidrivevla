"""Safe, SDK-free caches shared by offline dataset converters and the trainer."""

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional

import numpy as np

from unidrivevla.data.sample import check_sample

CACHE_SCHEMA = "unidrivevla.external.v1"
_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")


def _segment(value: str, what: str) -> str:
    value = str(value)
    if not _SEGMENT.fullmatch(value):
        raise ValueError(f"{what} must contain only letters, digits, '_' or '-': {value!r}")
    return value


def _relative_path(value: str) -> PurePosixPath:
    value = str(value)
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(f"cache paths must use POSIX-relative syntax: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"cache paths must be relative and stay inside the split: {value!r}")
    return path


def _inside(root: Path, path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"cache path escapes its root: {path}")
    return path


def _encode(value, arrays: Dict[str, np.ndarray], key: str):
    if isinstance(value, np.ndarray) or isinstance(value, np.generic):
        name = f"a{len(arrays)}"
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"object arrays are not supported in caches at {key}")
        arrays[name] = array
        return {"$array": name}
    if isinstance(value, float) and not math.isfinite(value):
        name = f"a{len(arrays)}"
        arrays[name] = np.asarray(value, dtype=np.float64)
        return {"$array": name}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError(f"cache object keys must be strings at {key}")
        return {k: _encode(v, arrays, f"{key}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, arrays, f"{key}[{i}]") for i, v in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported cache value at {key}: {type(value).__name__}")


class ExternalCacheWriter:
    """Build one split atomically. Existing split directories are never replaced."""

    def __init__(self, cache_root: str, source: str, split: str):
        self.root = Path(cache_root).expanduser().resolve()
        self.source, self.split = _segment(source, "source"), _segment(split, "split")
        self.final = self.root / "splits" / self.split
        if self.final.exists():
            raise FileExistsError(f"cache split already exists: {self.final}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=f".{self.source}-{self.split}-", dir=str(self.root)))
        (self.stage / "samples").mkdir()
        (self.stage / "assets").mkdir()
        self.manifest = open(self.stage / "manifest.jsonl", "x", encoding="utf-8")
        self.tokens = set()
        self.count = 0
        self.closed = False

    def write_sample(self, token: str, features: dict, targets: dict, assets: Optional[Dict[str, bytes]] = None) -> None:
        if self.closed:
            raise RuntimeError("cache writer is closed")
        token = str(token)
        if not token or token in self.tokens:
            raise ValueError(f"empty or duplicate sample token: {token!r}")
        problems = check_sample(features, targets)
        if problems:
            raise ValueError(f"invalid {self.source} sample {token}: " + "; ".join(problems))
        history = np.asarray(features["history_trajectory"], dtype=np.float64)
        if not np.isfinite(history).all():
            raise ValueError(f"invalid {self.source} sample {token}: observed history pose contains non-finite values")
        if features.get("ego2global") is not None:
            pose = np.asarray(features["ego2global"], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"invalid {self.source} sample {token}: ego2global must be a finite 4x4 pose")
        for camera in features["cameras"]:
            for key, shape in (("intrinsic", (3, 3)), ("sensor2ego", (4, 4))):
                if key not in camera:
                    raise ValueError(f"invalid {self.source} sample {token}: camera lacks {key} calibration")
                value = np.asarray(camera[key], dtype=np.float64)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid {self.source} sample {token}: camera {key} must be finite {shape}")
        task_valid = targets.get("task_valid")
        if not isinstance(task_valid, dict) or task_valid.get("planning", True):
            trajectory = np.asarray(targets.get("trajectory", []), dtype=np.float64).reshape(-1, 3)
            mask = np.asarray(targets.get("trajectory_mask", []), dtype=np.float64).reshape(-1)
            if not np.isfinite(mask).all():
                raise ValueError(f"invalid {self.source} sample {token}: planning mask contains non-finite values")
            valid = mask > 0
            if len(valid) != len(trajectory) or not np.isfinite(trajectory[valid]).all():
                raise ValueError(f"invalid {self.source} sample {token}: supervised trajectory coordinates must be finite")
        sample_id = hashlib.sha256(f"{self.source}\0{self.split}\0{token}".encode("utf-8")).hexdigest()
        arrays: Dict[str, np.ndarray] = {}
        row = {"token": token, "features": _encode(features, arrays, "features"),
               "targets": _encode(targets, arrays, "targets"), "arrays": f"samples/{sample_id}.npz"}
        np.savez_compressed(self.stage / row["arrays"], **arrays)
        for name, payload in (assets or {}).items():
            rel = _relative_path(name)
            if not isinstance(payload, bytes):
                raise TypeError(f"cache asset {name!r} must be bytes")
            path = self.stage.joinpath("assets", *rel.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "xb") as f:
                f.write(payload)
        self.manifest.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self.tokens.add(token)
        self.count += 1

    def close(self) -> int:
        if self.closed:
            return self.count
        self.manifest.flush()
        os.fsync(self.manifest.fileno())
        self.manifest.close()
        metadata = {"schema": CACHE_SCHEMA, "source": self.source, "split": self.split, "samples": self.count}
        with open(self.stage / "metadata.json", "x", encoding="utf-8") as f:
            json.dump(metadata, f, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        self.final.parent.mkdir(parents=True, exist_ok=True)
        if self.final.exists():
            raise FileExistsError(f"cache split already exists: {self.final}")
        os.replace(str(self.stage), str(self.final))
        self.closed = True
        return self.count

    def abort(self) -> None:
        if not self.closed:
            self.manifest.close()
            shutil.rmtree(self.stage, ignore_errors=True)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            try:
                self.close()
            except Exception:
                self.abort()
                raise
        else:
            self.abort()


def _decode(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"$array"}:
            name = value["$array"]
            if not isinstance(name, str) or name not in arrays:
                raise ValueError(f"cache row references missing array {name!r}")
            return arrays[name].copy()
        return {k: _decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    return value


class ExternalCacheDataset:
    """Training-side reader; deliberately imports no PhysicalAI or LEAD SDK."""

    def __init__(self, cache_root: str, source: str, split: str):
        self.root = Path(cache_root).expanduser().resolve()
        self.source, self.split = _segment(source, "source"), _segment(split, "split")
        self.split_root = (self.root / "splits" / self.split).resolve()
        if not self.split_root.is_relative_to(self.root):
            raise ValueError("cache split escapes cache_root")
        with open(_inside(self.split_root, self.split_root / "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata != {"schema": CACHE_SCHEMA, "source": self.source, "split": self.split,
                        "samples": metadata.get("samples")} or not isinstance(metadata["samples"], int):
            raise ValueError(f"invalid cache metadata in {self.split_root}")
        self.rows = []
        self.by_token = {}
        with open(_inside(self.split_root, self.split_root / "manifest.jsonl"), encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                    if not all(key in row for key in ("token", "features", "targets", "arrays")):
                        raise ValueError("missing fields")
                    token = row["token"]
                    if not isinstance(token, str) or not token or token in self.by_token:
                        raise ValueError(f"empty or duplicate sample token {token!r}")
                    _relative_path(row["arrays"])
                    self.by_token[token] = len(self.rows)
                    self.rows.append(row)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"invalid manifest row {line_number}: {exc}") from exc
        if len(self.rows) != metadata["samples"]:
            raise ValueError(f"manifest has {len(self.rows)} rows, metadata says {metadata['samples']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        array_path = _inside(self.split_root, self.split_root / _relative_path(row["arrays"]))
        with np.load(array_path, allow_pickle=False) as arrays:
            features = _decode(row["features"], arrays)
            targets = _decode(row["targets"], arrays)
        for cam in features.get("cameras", []):
            for key in ("path", "valid_mask_path"):
                if cam.get(key):
                    path = _inside(self.split_root, self.split_root / _relative_path(cam[key]))
                    if not path.is_file():
                        raise FileNotFoundError(f"invalid or missing cached camera {key}: {cam[key]!r}")
                    cam[key] = str(path)
        return features, targets, row["token"]

    def eval_info(self, token: str):
        """Minimal planning targets for the generic open-loop evaluator."""
        if token not in self.by_token:
            raise KeyError(f"unknown {self.source} evaluation token: {token!r}")
        row = self.rows[self.by_token[token]]
        array_path = _inside(self.split_root, self.split_root / _relative_path(row["arrays"]))
        with np.load(array_path, allow_pickle=False) as arrays:
            targets = _decode(row["targets"], arrays)
        if "trajectory" not in targets or "trajectory_mask" not in targets:
            raise ValueError(f"planning targets missing from cache token {token!r}")
        return {"trajectory": targets["trajectory"], "mask": targets["trajectory_mask"]}

    def sequences(self):
        """0.5s contiguous runs grouped by the converter's stable stream id."""
        groups = {}
        for i, row in enumerate(self.rows):
            features = row["features"]
            stream = features.get("stream_id")
            stamp_ref = features.get("timestamp")
            if not stream or stamp_ref is None or (isinstance(stamp_ref, dict) and "$array" in stamp_ref):
                continue
            stamp = float(stamp_ref)
            groups.setdefault(str(stream), []).append((stamp, i))
        sequences = []
        for values in groups.values():
            values.sort()
            run = []
            previous = None
            for stamp, index in values:
                if previous is not None and abs((stamp - previous) - 0.5) > 1e-3:
                    if run:
                        sequences.append(run)
                    run = []
                run.append(index)
                previous = stamp
            if run:
                sequences.append(run)
        return sequences


def write_png(image: np.ndarray, mode: Optional[str] = None) -> bytes:
    """Encode an RGB or single-channel array as PNG without a new dependency."""
    import io
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError(f"PNG asset must be uint8 HxW, HxWx3 or HxWx4, got {array.shape} {array.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def write_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    """Encode an RGB frame as a compact training asset."""
    import io
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"JPEG asset must be uint8 HxWx3, got {array.shape} {array.dtype}")
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return buffer.getvalue()
