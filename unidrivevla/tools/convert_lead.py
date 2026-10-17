"""Offline converter for LEAD-main's official 123D Arrow logs.

Use LEAD main's Python 3.10 environment with its pinned py123d v0.7.0. The explicit log list is the split boundary;
provide disjoint lists for train/val/test. No CARLA process, simulator, download, or training dependency is used.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_log_names(path):
    with open(path, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if not names or len(set(names)) != len(names):
        raise ValueError("log list must be nonempty and contain no duplicate names")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="LEAD data root containing logs/ and maps/")
    parser.add_argument("--cache-root", required=True, help="canonical output root; existing split is an error")
    parser.add_argument("--log-list", required=True, help="explicit local text file of disjoint LEAD log names")
    parser.add_argument("--scene-split", action="append", default=None,
                        help="py123d split directory, repeat for multiple; default: normal_view")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="optional positive limit for a local smoke cache")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        parser.error("official LEAD main pins Python 3.10; run this converter in that separate environment")
    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0")

    try:
        from lead.log_reader import SceneLoader
        from py123d.api.scene.scene_filter import SceneFilter
    except ImportError as exc:
        raise SystemExit("run with the official LEAD main and its pinned py123d v0.7.0 installed") from exc

    from unidrivevla.data.lead import HORIZON, INTERVAL_TICKS, TICK_US, exact_label_origin, lead_sample
    from unidrivevla.data.external import ExternalCacheWriter

    log_names = read_log_names(args.log_list)
    future_ticks = [i * INTERVAL_TICKS for i in range(1, HORIZON + 1)]
    scene_filter = SceneFilter(split_names=args.scene_split or ["normal_view"], log_names=log_names,
                               history_num_iterations=3 * INTERVAL_TICKS,
                               future_num_iterations=HORIZON * INTERVAL_TICKS)
    loader = SceneLoader(data_root=Path(args.data_root).expanduser(), scene_filter=scene_filter,
                         perturbation_probability=0.0, past_tick_ages=tuple(range(3 * INTERVAL_TICKS + 1)),
                         tick_duration_us=TICK_US, read_lidar_sweeps=False, read_radar_sweeps=False,
                         read_semantic_cameras=False, read_depth_cameras=False, read_map_api=False)
    if not len(loader):
        raise SystemExit("LEAD SceneLoader returned no camera-anchored scenes for the requested log list")

    phase_counts = defaultdict(int)
    written = skipped = 0
    with ExternalCacheWriter(args.cache_root, "lead", args.split) as writer:
        for index in range(len(loader)):
            if args.max_samples and written >= args.max_samples:
                break
            try:
                scene_data = loader[index]  # normal view only; all camera inputs are 4Hz anchor frames
                log_name = str(scene_data.log_metadata.log_name)
                phase = phase_counts[log_name] % 2
                phase_counts[log_name] += 1
                future = loader.read_future_ego_states(index, future_ticks)
                label_origin_pose = exact_label_origin(scene_data)
                features, targets, assets = lead_sample(scene_data, future, log_name,
                                                        label_origin_pose=label_origin_pose, phase=phase)
                token = f"{log_name}:{int(scene_data.cameras[0].timestamp.time_us)}"
                writer.write_sample(token, features, targets, assets)
                written += 1
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                skipped += 1
                print(f"skip LEAD sample {index}: {exc}")
    if not written:
        raise SystemExit(f"no samples converted (skipped={skipped}); inspect the selected logs and SDK version")
    print(f"LEAD {args.split}: wrote {written}, skipped {skipped} into "
          f"{Path(args.cache_root).expanduser() / 'splits' / args.split}")


if __name__ == "__main__":
    main()
