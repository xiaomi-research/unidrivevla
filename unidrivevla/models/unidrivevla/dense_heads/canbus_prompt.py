from __future__ import annotations

import gzip
import json

import numpy as np


COC_LANGUAGE_PROMPT_TEMPLATE = (
    "As an autonomous driving system, first generate chain-of-causation reasoning "
    "for future trajectory planning.\n"
    "Based on:\n"
    "1. Surround-view camera images.\n"
    "2. Active navigation command: [{nav_cmd}].\n"
    "3. Historical ego motion in 2D BEV frame:\n"
    "   {hist_traj_str}\n\n"
    "Output requirements:\n"
    "- The next module will use this reasoning to plan 6 future waypoints over 3.0 seconds.\n"
    "- In this text response, output only the concise chain-of-causation reasoning.\n"
    "- Do not write trajectory coordinates, waypoints, or [PT, ...] in text.\n"
    "- Keep the answer as a single short sentence."
)


def format_history_trajectory(hist_traj, aligned_with_canbus_sft: bool) -> str:
    hist = np.asarray(hist_traj, dtype=np.float32)
    if hist.ndim != 2 or hist.shape[-1] < 2:
        raise ValueError(f"hist_traj must have shape [T, >=2], got {hist.shape}")
    if aligned_with_canbus_sft:
        points = [f"({point[0]:.2f}, {point[1]:.2f})" for point in hist]
    else:
        points = [f"({point[0]:+07.2f}, {point[1]:+07.2f})" for point in hist]
    return f"[PT_HIST, {', '.join(points)}]"


def format_ego_status(
    *, gt_ego_lcf_feat, gt_ego_his_trajs, ego_status=None, hist_traj=None
) -> str:
    ax, ay = 0.0, 0.0
    if gt_ego_his_trajs is not None:
        raw_his = np.asarray(gt_ego_his_trajs, dtype=np.float32)
        if raw_his.shape[0] >= 2:
            v_curr = raw_his[-1] / 0.5
            v_prev = raw_his[-2] / 0.5
            ax, ay = ((v_curr - v_prev) / 0.5)[:2].tolist()
    elif hist_traj is not None:
        hist = np.asarray(hist_traj, dtype=np.float32)
        if hist.shape[0] >= 3:
            raw_his = hist[1:] - hist[:-1]
            v_curr = raw_his[-1] / 0.5
            v_prev = raw_his[-2] / 0.5
            ax, ay = ((v_curr - v_prev) / 0.5)[:2].tolist()

    if gt_ego_lcf_feat is not None:
        lcf = np.asarray(gt_ego_lcf_feat, dtype=np.float32).reshape(-1)
        if lcf.size < 9:
            lcf = np.pad(lcf, (0, 9 - lcf.size))
        vx, vy = lcf[0], lcf[1]
        can_bus_x, can_bus_y = lcf[2], lcf[3]
        v_yaw = lcf[4]
        heading_speed = lcf[7] * 0.5
        steering = lcf[8]
    elif ego_status is not None:
        ego = np.asarray(ego_status, dtype=np.float32).reshape(-1)
        if ego.size < 10:
            ego = np.pad(ego, (0, 10 - ego.size))
        can_bus_x, can_bus_y = ego[0], ego[1]
        v_yaw = ego[5]
        vx, vy = ego[6], ego[7]
        heading_speed = float(np.sqrt(vx * vx + vy * vy) * 0.5)
        steering = ego[9]
    else:
        raise ValueError("gt_ego_lcf_feat or ego_status is required for the ego-status prompt")

    return (
        "4. Ego Status:\n"
        f"   - Velocity (vx,vy): ({vx:.2f},{vy:.2f})\n"
        f"   - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})\n"
        f"   - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})\n"
        f"   - Can Bus: ({can_bus_x:.2f},{can_bus_y:.2f})\n"
        f"   - Heading Speed: ({heading_speed:.2f})\n"
        f"   - Steering: ({steering:.2f})"
    )


def build_driving_user_prompt(
    *, nav_cmd, hist_traj, trajectory_prompt_template, use_coc_language_prompt,
    include_ego_status_prompt, gt_ego_lcf_feat=None, gt_ego_his_trajs=None,
    ego_status=None,
) -> str:
    if include_ego_status_prompt:
        nav_cmd = nav_cmd.strip()
    hist_traj_str = format_history_trajectory(
        hist_traj, aligned_with_canbus_sft=include_ego_status_prompt
    )
    template = COC_LANGUAGE_PROMPT_TEMPLATE if use_coc_language_prompt else trajectory_prompt_template
    prompt = template.format(nav_cmd=nav_cmd, hist_traj_str=hist_traj_str)
    if not include_ego_status_prompt:
        return prompt
    ego_status_text = format_ego_status(
        gt_ego_lcf_feat=gt_ego_lcf_feat,
        gt_ego_his_trajs=gt_ego_his_trajs,
        ego_status=ego_status,
        hist_traj=hist_traj,
    )
    marker = "\n\nOutput requirements:"
    if marker not in prompt:
        raise ValueError("Driving prompt is missing the Output requirements marker")
    return prompt.replace(marker, f"\n{ego_status_text}{marker}", 1)


def load_coc_by_token(path: str) -> dict:
    """sample token -> {cot, nav_text}. Empty path yields nothing; a set path that cannot be read fails."""
    if not path:
        return {}
    opener = gzip.open if str(path).endswith(".gz") else open
    table = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            token = item.get("sample_token") or item.get("token")
            if token:
                table[str(token)] = {"cot": item.get("cot") or "", "nav_text": item.get("nav_text") or item.get("nav_command") or ""}
    return table


def coc_language_target(record) -> str:
    """The assistant sentence written after the image. Target format is the chain-of-causation text."""
    if not record:
        return ""
    return (record.get("cot") or "").strip()


def extract_based_on_context(prompt: str) -> str:
    start_marker = "Based on:\n"
    end_marker = "\n\nOutput requirements:"
    start = prompt.find(start_marker)
    end = prompt.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Prompt does not contain a canonical Based on context")
    return prompt[start:end]
