#!/usr/bin/env python
import argparse
import glob
import json
import os

import numpy as np


def harmonic_mean(first, second):
    if first <= 0 or second <= 0:
        return 0.0
    return 2.0 * first * second / (first + second)


def merge_route_json(folder_path):
    file_paths = glob.glob(f"{folder_path}/*.json")
    evaluation_dataset = os.getenv("EVALUATION_DATASET")
    merged_records = []
    driving_score = []
    success_num = 0
    split_scores = {"Base": [], "Generalization": []}
    split_success = {"Base": 0, "Generalization": 0}
    for file_path in file_paths:
        if "merged.json" in file_path:
            continue
        try:
            with open(file_path) as file:
                data = json.load(file)
                records = data["_checkpoint"]["records"]
                file_stem = os.path.splitext(os.path.basename(file_path))[0]
                split = next(
                    (name for name in split_scores if file_stem.startswith(name + "_")),
                    None,
                )
                for rd in records:
                    rd.pop("index", None)
                    merged_records.append(rd)
                    score = rd["scores"]["score_composed"]
                    driving_score.append(score)
                    ignored = {"min_speed_infractions"}
                    if evaluation_dataset == "fail2drive":
                        ignored.add("outside_route_lanes")
                    success_flag = not any(
                        values
                        for name, values in rd["infractions"].items()
                        if name not in ignored
                    )
                    if evaluation_dataset != "fail2drive":
                        success_flag = success_flag and rd["status"] in {"Completed", "Perfect"}
                    if success_flag:
                        success_num += 1
                        print(rd["route_id"])
                    if evaluation_dataset == "fail2drive" and split is not None:
                        split_scores[split].append(score)
                        split_success[split] += int(success_flag)
        except Exception:
            print(
                f"-----------------------Warning: {file_path} is not a valid json file.",
            )
    merged_records = sorted(merged_records, key=lambda d: d["route_id"], reverse=True)
    _checkpoint = {"records": merged_records}

    if os.getenv("EVALUATION_DATASET") == "Town13":
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / 20,
            "success rate": success_num / 20,
            "eval num": len(driving_score),
            "current_driving_score": np.mean(driving_score),
        }
    elif os.getenv("EVALUATION_DATASET") == "longest6":
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / 36,
            "success rate": success_num / 36,
            "eval num": len(driving_score),
            "current_driving_score": np.mean(driving_score),
        }
    elif evaluation_dataset == "fail2drive":
        split_summary = {}
        for split, scores in split_scores.items():
            expected_routes = 100
            split_ds = sum(scores) / expected_routes
            split_sr_percent = 100.0 * split_success[split] / expected_routes
            split_summary[split] = {
                "driving score": split_ds,
                "success rate": split_sr_percent / 100.0,
                "harmonic mean": harmonic_mean(split_ds, split_sr_percent),
                "eval num": len(scores),
                "current_driving_score": float(np.mean(scores)) if scores else 0.0,
            }
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / 200,
            "success rate": success_num / 200,
            "eval num": len(driving_score),
            "current_driving_score": float(np.mean(driving_score)) if driving_score else 0.0,
            "expected eval num": 200,
            "benchmark total routes": 200,
            "splits": split_summary,
        }
    elif evaluation_dataset == "bench2drive":
        num_shards = int(os.getenv("BENCH2DRIVE_NUM_SHARDS", "1"))
        shard_index = int(os.getenv("BENCH2DRIVE_SHARD_INDEX", "0"))
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError(
                "Invalid Bench2Drive shard configuration: "
                f"index={shard_index}, count={num_shards}"
            )
        expected_routes = (
            220
            if num_shards == 1
            else (220 + num_shards - 1 - shard_index) // num_shards
        )
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / expected_routes,
            "success rate": success_num / expected_routes,
            "eval num": len(driving_score),
            "current_driving_score": np.mean(driving_score),
            "expected eval num": expected_routes,
            "benchmark total routes": 220,
            "shard index": shard_index,
            "num shards": num_shards,
            "official full benchmark": num_shards == 1,
        }
    else:
        raise ValueError("EVALUATION_DATASET is not set or not supported.")

    with open(os.path.join(folder_path, "merged.json"), "w") as file:
        json.dump(merged_data, file, indent=4)
    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="old foo help")
    args = parser.parse_args()
    merge_route_json(args.folder)
