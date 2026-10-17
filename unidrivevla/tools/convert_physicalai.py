"""Offline PhysicalAI-AV to canonical planning cache converter.

Use Python >=3.11 with the licensed PhysicalAI-AV SDK and already-cached data. This command always forces HuggingFace
offline mode; it never downloads, accepts terms, or stores a token. In the Python 3.9 training environment, use
`data=physicalai` to read the resulting cache.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_clip_ids(path):
    with open(path, encoding="utf-8") as f:
        clips = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if not clips or len(set(clips)) != len(clips):
        raise ValueError("clip list must be nonempty and contain no duplicate IDs")
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="local HF dataset files for the official SDK")
    parser.add_argument("--cache-root", required=True, help="canonical output root; existing split is an error")
    parser.add_argument("--clip-ids", required=True, help="authorized local text file, one clip UUID per line")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--revision", default=None, help="official dataset revision or immutable commit hash")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(960, 544))
    parser.add_argument("--fov-deg", action="append", default=[], metavar="CAMERA=DEGREES",
                        help="override a calibrated per-camera rectified horizontal FOV (repeatable)")
    parser.add_argument("--sync-tolerance-us", type=int, default=50_000,
                        help="maximum capture-time offset from the front camera anchor")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("the official PhysicalAI-AV SDK requires Python >=3.11; run this offline converter separately")

    # The official SDK can resolve local hub snapshots while offline; fail instead of fetching anything missing.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from physical_ai_av import PhysicalAIAVDatasetInterface
    except ImportError as exc:
        raise SystemExit("install the official physical_ai_av SDK in this offline conversion environment") from exc

    from unidrivevla.data.physicalai import convert_dataset_interface

    dataset = PhysicalAIAVDatasetInterface(revision=args.revision, local_dir=Path(args.dataset_root).expanduser(),
                                           cache_dir=os.environ.get("HF_HOME"))
    clip_ids = read_clip_ids(args.clip_ids)
    fov_overrides = {}
    for value in args.fov_deg:
        try:
            name, degrees = value.split("=", 1)
            if name in fov_overrides:
                raise ValueError(f"duplicate FOV override for {name}")
            fov_overrides[name] = float(degrees)
        except ValueError as exc:
            raise SystemExit(f"invalid --fov-deg {value!r}; expected CAMERA=DEGREES") from exc
    if args.sync_tolerance_us < 0:
        parser.error("--sync-tolerance-us must be nonnegative")
    result = convert_dataset_interface(dataset, clip_ids, args.cache_root, args.split,
                                       image_size=args.image_size, sync_tolerance_us=args.sync_tolerance_us,
                                       fov_overrides=fov_overrides)
    if not result["written"]:
        raise SystemExit(f"no samples converted; inspect calibration/data coverage (skipped={result['skipped']})")
    print(f"PhysicalAI-AV {args.split}: wrote {result['written']}, skipped {result['skipped']} into "
          f"{Path(args.cache_root).expanduser() / 'splits' / args.split}")


if __name__ == "__main__":
    main()
