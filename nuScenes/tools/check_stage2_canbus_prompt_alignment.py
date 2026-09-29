#!/usr/bin/env python3
import argparse
import ast
import hashlib
from pathlib import Path

from check_canbus_prompt_alignment import (
    PROMPT_CONSTANTS,
    PROMPT_HELPER,
    assert_equal,
    build_inputs,
    build_stage2_prompt,
    iter_aligned_records,
    strip_view_prefix,
)


NUSCENES_ROOT = Path(__file__).resolve().parents[1]
PLANNING_HEAD = (
    NUSCENES_ROOT
    / "projects/mmdet3d_plugin/unidrivevla/dense_heads/qwenvl3_vla_planning_head.py"
)
LEGACY_WITHOUT_EGO_COC_SHA256 = "c797b3fce94bb91402938d0283e7a4aaff0f70aeec6e2743d91c1c09d3685fdd"


def check_planning_head_wiring():
    tree = ast.parse(PLANNING_HEAD.read_text(encoding="utf-8"), filename=str(PLANNING_HEAD))
    constructor_default_is_false = False
    prompt_call_keywords = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            positional = node.args.args
            defaults = node.args.defaults
            default_by_name = {
                argument.arg: default
                for argument, default in zip(positional[-len(defaults):], defaults)
            }
            default = default_by_name.get("include_ego_status_prompt")
            constructor_default_is_false = (
                isinstance(default, ast.Constant) and default.value is False
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_driving_user_prompt"
        ):
            prompt_call_keywords.update(
                keyword.arg for keyword in node.keywords if keyword.arg is not None
            )
    required = {
        "nav_cmd",
        "hist_traj",
        "trajectory_prompt_template",
        "use_coc_language_prompt",
        "include_ego_status_prompt",
        "gt_ego_lcf_feat",
        "gt_ego_his_trajs",
        "ego_status",
    }
    if not constructor_default_is_false:
        raise AssertionError("include_ego_status_prompt must default to False")
    missing = required - prompt_call_keywords
    if missing:
        raise AssertionError(f"Planning head prompt helper call is missing: {sorted(missing)}")


def build_legacy_without_ego_prompt(vad_info):
    inputs = build_inputs(vad_info)
    hist_traj_str = "[PT_HIST, {}]".format(
        ", ".join(
            f"({point[0]:+07.2f}, {point[1]:+07.2f})"
            for point in inputs["hist_traj"]
        )
    )
    return PROMPT_CONSTANTS.NUSCENES_USER_PROMPT_TEMPLATE.format(
        nav_cmd=inputs["nav_cmd"],
        hist_traj_str=hist_traj_str,
    )


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
    check_planning_head_wiring()
    coc_hash = hashlib.sha256(PROMPT_HELPER.COC_LANGUAGE_PROMPT_TEMPLATE.encode()).hexdigest()
    if coc_hash != LEGACY_WITHOUT_EGO_COC_SHA256:
        raise AssertionError(f"Without-ego CoC template changed: {coc_hash}")
    checked = 0
    for line_number, record, vad_info in iter_aligned_records(
        args.jsonl, args.vad_pkl, args.num_samples
    ):
        expected_with_ego = strip_view_prefix(record["conversations"][0]["value"])
        actual_with_ego = build_stage2_prompt(
            vad_info,
            use_coc_language_prompt=False,
            include_ego_status_prompt=True,
        )
        assert_equal(
            expected_with_ego,
            actual_with_ego,
            f"Stage2 with-ego mismatch at JSONL line {line_number}",
        )

        coc_with_ego = build_stage2_prompt(
            vad_info,
            use_coc_language_prompt=True,
            include_ego_status_prompt=True,
        )
        expected_context = PROMPT_HELPER.extract_based_on_context(expected_with_ego)
        actual_context = PROMPT_HELPER.extract_based_on_context(coc_with_ego)
        assert_equal(
            expected_context,
            actual_context,
            f"CoC canbus context mismatch at JSONL line {line_number}",
        )

        expected_without_ego = build_legacy_without_ego_prompt(vad_info)
        actual_without_ego = build_stage2_prompt(
            vad_info,
            use_coc_language_prompt=False,
            include_ego_status_prompt=False,
        )
        assert_equal(
            expected_without_ego,
            actual_without_ego,
            f"Without-ego regression at JSONL line {line_number}",
        )
        if "4. Ego Status:" in actual_without_ego:
            raise AssertionError(
                f"Without-ego prompt contains ego status at JSONL line {line_number}"
            )
        checked += 1
    print(
        "Stage2 prompt checks passed: "
        f"{checked} with-ego exact, CoC-context exact, and without-ego regression samples"
    )


if __name__ == "__main__":
    main()
