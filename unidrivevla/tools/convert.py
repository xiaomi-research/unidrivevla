"""Runs the vendored info converters (tools/data_converter/) unchanged apart from their origin header.
Needs mmcv (no CUDA ops), nuscenes-devkit, pyquaternion, shapely and opencv. The VAD temporal converter
also needs mmdet3d. Image paths in the infos are relative to the working directory (nuScenes) or to
--data-root (Bench2Drive).

  python tools/convert.py nuscenes -- \
      --root-path ./data/nuscenes --canbus ./data/nuscenes --version v1.0 \
      --out-dir ./data/infos --extra-tag nuscenes
  python tools/convert.py vad -- \
      --root-path ./data/nuscenes --canbus ./data/nuscenes --version v1.0 \
      --out-dir ./data/infos --extra-tag vad_nuscenes
  python tools/convert.py b2d --data-root DATA --map-root MAPS --out-dir INFOS \
      --train CLIP ... --val CLIP ...

--code DIR uses a checkout of the official tree instead of the vendored copies (the nuScenes root, or the
Bench2Drive root). Omit it to use tools/data_converter/.

nuscenes: nuscenes_converter.py imports projects.mmdet3d_plugin.datasets.map_utils, and the package __init__
files above it in the official tree import the model stack. Those packages are registered empty, and
map_utils is the vendored copy, so only map_utils loads. The trainer reads nuscenes_infos_{train,val}.pkl
from this converter (--version v1.0 appends -trainval and writes both splits). vad writes
{extra-tag}_infos_temporal_{train,val}.pkl; the current loader does not read those. b2d:
bench2drive_converter.py's __main__ only builds the map (its info calls are commented out) and reads a split
file of the base dataset; this calls generate_infos on the given clips (DATA/v1/CLIP), then generate_map.
"""

import argparse
import os
import runpy
import sys
import types

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_converter")


def _stub_map_utils(datasets_path: str) -> None:
    """Empty stand-ins for projects.mmdet3d_plugin.datasets, whose real __init__ imports the model stack.
    map_utils is loaded from datasets_path (the vendored directory, which contains map_utils/)."""
    for name in ("projects", "projects.mmdet3d_plugin", "projects.mmdet3d_plugin.datasets"):
        package = types.ModuleType(name)
        package.__path__ = [datasets_path] if name.endswith("datasets") else []
        sys.modules[name] = package


def nuscenes(args, rest) -> None:
    if args.code:
        code = os.path.abspath(args.code)
        sys.path.insert(0, code)
        converter = os.path.join(code, "tools", "data_converter", "nuscenes_converter.py")
        datasets_path = os.path.join(code, "projects", "mmdet3d_plugin", "datasets")
        for name in ("projects", "projects.mmdet3d_plugin", "projects.mmdet3d_plugin.datasets"):
            package = types.ModuleType(name)
            package.__path__ = [os.path.join(code, *name.split("."))]
            sys.modules[name] = package
    else:
        converter = os.path.join(VENDOR, "nuscenes_converter.py")
        _stub_map_utils(VENDOR)
    sys.argv = ["nuscenes_converter.py", "nuscenes", *rest]
    runpy.run_path(converter, run_name="__main__")


def vad(args, rest) -> None:
    if args.code:
        code = os.path.abspath(args.code)
        sys.path.insert(0, code)
        converter = os.path.join(code, "tools", "data_converter", "vad_nuscenes_converter.py")
    else:
        converter = os.path.join(VENDOR, "vad_nuscenes_converter.py")
    sys.argv = ["vad_nuscenes_converter.py", "nuscenes", *rest]
    runpy.run_path(converter, run_name="__main__")


def b2d(args) -> None:
    if args.code:
        sys.path.insert(0, os.path.join(os.path.abspath(args.code), "tools", "data_converter"))
    else:
        sys.path.insert(0, VENDOR)
    import bench2drive_converter as converter

    converter.DATAROOT, converter.MAP_ROOT, converter.OUT_DIR = args.data_root, args.map_root, args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    for split, clips in (("train", args.train), ("val", args.val)):
        converter.process_list = []  # generate_infos appends its workers to this module global
        converter.generate_infos([f"v1/{clip}" for clip in clips], max(1, min(args.workers, len(clips))), split, "tmp_data")
    converter.generate_map(args.map_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="source", required=True)
    nus = sub.add_parser("nuscenes")
    nus.add_argument("--code", default="", help="official Code/nuScenes; default: tools/data_converter")
    temporal = sub.add_parser("vad")
    temporal.add_argument("--code", default="", help="official Code/nuScenes; default: tools/data_converter")
    bench = sub.add_parser("b2d")
    bench.add_argument("--code", default="", help="official Code/Bench2Drive; default: tools/data_converter")
    bench.add_argument("--data-root", required=True)
    bench.add_argument("--map-root", required=True)
    bench.add_argument("--out-dir", required=True)
    bench.add_argument("--train", nargs="+", required=True)
    bench.add_argument("--val", nargs="+", required=True)
    bench.add_argument("--workers", type=int, default=4)
    args, rest = parser.parse_known_args()
    rest = [r for r in rest if r != "--"]
    if args.source == "nuscenes":
        nuscenes(args, rest)
    elif args.source == "vad":
        vad(args, rest)
    else:
        b2d(args)


if __name__ == "__main__":
    main()
