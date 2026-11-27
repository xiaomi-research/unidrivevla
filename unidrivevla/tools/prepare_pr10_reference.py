"""Build the portable PR10 CPU scorer bundle from commit 5fce121.

Run from any Python environment with Git available. The runtime scorer uses the checked archive and
does not require a checkout or `.git` directory.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


PR10 = "5fce121f4bf04eaf7e109880d4f26eed62c0eed0"
SOURCE_PATHS = (
    "LICENSE.txt",
    "nuScenes/projects/__init__.py",
    "nuScenes/projects/mmdet3d_plugin/__init__.py",
    "nuScenes/projects/mmdet3d_plugin/core/box3d.py",
    "nuScenes/projects/mmdet3d_plugin/datasets",
    "nuScenes/projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage2.py",
    "nuScenes/projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage3.py",
)
DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "references" / "pr10_nuscenes_scorer.tar.gz"


def build(repo: Path, output: Path) -> dict:
    repo = repo.resolve()
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", PR10 + "^{commit}"], text=True).strip()
    if revision != PR10:
        raise ValueError(f"fixed PR10 commit was not resolved exactly: {revision}")
    archive = subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-c", "tar.umask=0000", "-C", str(repo),
         "archive", "--format=tar", PR10, *SOURCE_PATHS]
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(archive)
    return {
        "revision": PR10,
        "bundle": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "license": "Apache-2.0 (upstream repository LICENSE.txt included)",
        "source_paths": list(SOURCE_PATHS),
        "build": "git archive with core.autocrlf=false and tar.umask=0000; gzip mtime=0",
        "purpose": "PR10 NuScenes3DDataset Stage2 and Stage3 CPU evaluation only; no models or checkpoints",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Git repository containing PR10 commit 5fce121")
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = build(args.repo, args.output)
    manifest_path = args.manifest or args.output.with_suffix(".json")
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
