#!/usr/bin/env python3
import argparse
import difflib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


NUSCENES_ROOT = Path(__file__).resolve().parents[1]
DENSE_HEAD_ROOT = (
    NUSCENES_ROOT
    / "projects/mmdet3d_plugin/unidrivevla/dense_heads"
)
VIEW_TOKENS = [
    "<FRONT_VIEW>",
    "<FRONT_LEFT_VIEW>",
    "<FRONT_RIGHT_VIEW>",
    "<BACK_LEFT_VIEW>",
    "<BACK_RIGHT_VIEW>",
    "<BACK_VIEW>",
]


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPT_HELPER = load_source_module(
    "main_coc_canbus_prompt",
    DENSE_HEAD_ROOT / "canbus_prompt.py",
)
PROMPT_CONSTANTS = load_source_module(
    "main_coc_prompt_constants",
    DENSE_HEAD_ROOT / "constants.py",
)


def load_infos(path):
    with open(path, "rb") as stream:
        data = pickle.load(stream)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


def strip_view_prefix(prompt):
    prefix = "".join(f"{view}\n<image>\n" for view in VIEW_TOKENS)
    if not prompt.startswith(prefix):
        raise AssertionError("SFT prompt does not start with the canonical six-view prefix")
    return prompt[len(prefix):]


def build_inputs(vad_info):
    raw_his = np.asarray(vad_info["gt_ego_his_trajs"], dtype=np.float32)[:4]
    cumulative = np.cumsum(raw_his, axis=0)
    hist_traj = (
        cumulative - cumulative[-1:]
        if len(cumulative) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    command = np.asarray(vad_info["gt_ego_fut_cmd"], dtype=np.float32)
    command_index = int(np.argmax(command)) if command.size == 3 else int(command)
    nav_cmd = PROMPT_CONSTANTS._NAV_CMD_FIXED.get(
        command_index,
        PROMPT_CONSTANTS._NAV_CMD_FIXED[2],
    )
    return {
        "nav_cmd": nav_cmd,
        "hist_traj": hist_traj,
        "gt_ego_lcf_feat": np.asarray(vad_info["gt_ego_lcf_feat"], dtype=np.float32),
        "gt_ego_his_trajs": raw_his,
    }


def build_stage2_prompt(vad_info, *, use_coc_language_prompt, include_ego_status_prompt):
    return PROMPT_HELPER.build_driving_user_prompt(
        trajectory_prompt_template=PROMPT_CONSTANTS.NUSCENES_USER_PROMPT_TEMPLATE,
        use_coc_language_prompt=use_coc_language_prompt,
        include_ego_status_prompt=include_ego_status_prompt,
        **build_inputs(vad_info),
    )


def build_image_lookup(vad_infos):
    lookup = {}
    for info in vad_infos:
        image_name = Path(
            info.get("cams", {}).get("CAM_FRONT", {}).get("data_path", "")
        ).name
        if image_name:
            if image_name in lookup:
                raise AssertionError(f"Duplicate CAM_FRONT basename in VAD infos: {image_name}")
            lookup[image_name] = info
    return lookup


def iter_aligned_records(jsonl_path, vad_path, num_samples):
    lookup = build_image_lookup(load_infos(vad_path))
    checked = 0
    with open(jsonl_path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            image_name = Path(record["image"][0]).name
            vad_info = lookup.get(image_name)
            if vad_info is None:
                raise AssertionError(
                    f"Cannot map JSONL line {line_number} to VAD info: {image_name}"
                )
            yield line_number, record, vad_info
            checked += 1
            if checked >= num_samples:
                break
    if checked != num_samples:
        raise AssertionError(f"Requested {num_samples} samples, but checked only {checked}")


def assert_equal(expected, actual, context):
    if actual == expected:
        return
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile="SFT JSONL",
            tofile="Stage2 helper",
        )
    )
    raise AssertionError(f"{context}\n{diff}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--vad-pkl", required=True)
    parser.add_argument("--num-samples", type=int, default=28130)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    prefix = "".join(f"{view}\n<image>\n" for view in VIEW_TOKENS)
    checked = 0
    for line_number, record, vad_info in iter_aligned_records(
        args.jsonl, args.vad_pkl, args.num_samples
    ):
        expected = record["conversations"][0]["value"]
        actual = prefix + build_stage2_prompt(
            vad_info,
            use_coc_language_prompt=False,
            include_ego_status_prompt=True,
        )
        assert_equal(expected, actual, f"Prompt mismatch at JSONL line {line_number}")
        checked += 1
    print(f"Full canbus SFT prompt alignment passed: {checked} samples")


if __name__ == "__main__":
    main()
