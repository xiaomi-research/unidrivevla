"""
Generate B2D Planning QA JSONL Dataset
Reference format from Bench2Drive official data

Output format:
{
    "id": int,
    "image": [6 camera image paths],
    "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
    ]
}
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np

# B2D Navigation Commands
B2D_COMMAND_MAP = {
    0: "TURN LEFT",
    1: "TURN RIGHT",
    2: "GO STRAIGHT",
    3: "LANE FOLLOW",
    4: "CHANGE LANE LEFT",
    5: "CHANGE LANE RIGHT",
}

# B2D Camera mapping (folder name -> image subfolder)
B2D_CAMERA_MAP = {
    'CAM_FRONT': 'camera/rgb_front',
    'CAM_FRONT_LEFT': 'camera/rgb_front_left',
    'CAM_FRONT_RIGHT': 'camera/rgb_front_right',
    'CAM_BACK_LEFT': 'camera/rgb_back_left',
    'CAM_BACK_RIGHT': 'camera/rgb_back_right',
    'CAM_BACK': 'camera/rgb_back',
}

# B2D View Tokens
B2D_VIEW_TOKENS = {
    'CAM_FRONT': '<FRONT_VIEW>',
    'CAM_FRONT_LEFT': '<FRONT_LEFT_VIEW>',
    'CAM_FRONT_RIGHT': '<FRONT_RIGHT_VIEW>',
    'CAM_BACK_LEFT': '<BACK_LEFT_VIEW>',
    'CAM_BACK_RIGHT': '<BACK_RIGHT_VIEW>',
    'CAM_BACK': '<BACK_VIEW>',
}

# Camera order for output
B2D_CAMERA_ORDER = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
]

# System prompt (Bench2Drive format)
B2D_SYSTEM_PROMPT = """You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on images from front camera, ego vehicle states, and discrete navigation commands. Predictions will be evaluated under the Bench2Drive closed-loop protocol covering 220 short routes across 44 interactive scenarios. Evaluation metrics include Driving Score, Success Rate, Multi-Ability Scores, Driving Efficiency, and Driving Smoothness as defined in the Bench2Drive evaluation toolkit. Ensure your predicted trajectories adhere to the fair and comprehensive assessment standards of Bench2Drive."""

# Human prompt template
B2D_HUMAN_TEMPLATE = """{view_tokens}
You are given 6 synchronized surround-view images.
Predict the ego vehicle's future trajectory based on:
1. Multi-view visual perception (6 views)
2. Ego speed: {speed:.2f} m/s, acceleration: ({acc_x:.2f}, {acc_y:.2f}) m/s^2
3. Active navigation command: [{cmd}]

Output requirements:
- Predict 6 future trajectory points as (x,y)
- Output shape: [6,2]
- Keep 2 decimal places"""

# GPT answer template
B2D_GPT_TEMPLATE = """Here is the planning trajectory [PT, {points}]."""


def format_traj_point_2d(x, y):
    return f"({x:.2f},{y:.2f})"


def format_traj_point_3d(x, y, delta):
    return f"({x:.2f},{y:.2f},{delta:.2f})"


def format_traj_points(traj, with_delta=False):
    """Format trajectory points string."""
    if with_delta:
        return ", ".join([format_traj_point_3d(x, y, d) for x, y, d in traj])
    return ", ".join([format_traj_point_2d(x, y) for x, y in traj])


def traj_to_residual(points_2d):
    """res[0]=pt0, res[i]=pt[i]-pt[i-1] (i=1..5)."""
    if not points_2d:
        return []
    res = [points_2d[0]]
    for i in range(1, len(points_2d)):
        x0, y0 = points_2d[i - 1]
        x1, y1 = points_2d[i]
        res.append((float(x1 - x0), float(y1 - y0)))
    return res


def _ensure_py_scalar(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, (float, int)):
        return float(x)
    if hasattr(x, 'item'):
        try:
            return float(x.item())
        except Exception:
            return default
    return default


def compute_ego_fut_traj_like_dataset(data_infos, idx, future_frames=6, interval=5, with_delta=False):
    """Match Bench2DriveDataset.get_plan_info() temporal 2hz branch.

    In Bench2DriveDataset.get_plan_info:
      interval = int(10 // float('2hz'.split('hz')[0])) == 5
      ego_temporal_trajs, ego_temporal_masks = get_ego_temporal_trajs(idx, future_frames, interval)

    Returns cumulative points (not offsets) in lidar(cur) frame, to match downstream usage:
      gt_ego_fut_trajs_2hz.cumsum(dim=-2)
    """
    cur_frame = data_infos[idx]
    world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']

    # build adj idx list like get_ego_temporal_trajs with split_group == 0
    adj_idx_list = [idx]
    adj_idx = idx
    for _ in range(future_frames * interval):
        adj_idx += 1
        adj_idx_list.append(adj_idx)
    adj_idx_list = adj_idx_list[::interval]

    full_adj_track = np.zeros((future_frames + 1, 2), dtype=np.float32)
    full_adj_adj_mask = np.zeros(future_frames + 1, dtype=np.float32)

    # make sure the carla clip already init
    past_idx = adj_idx_list[0] - 2
    if 0 <= past_idx < len(data_infos):
        past_frame = data_infos[past_idx]
    else:
        past_frame = None

    if past_frame is not None and past_frame['folder'] == cur_frame['folder']:
        for j, aidx in enumerate(adj_idx_list):
            if 0 <= aidx < len(data_infos):
                adj_frame = data_infos[aidx]
                if adj_frame['folder'] != cur_frame['folder']:
                    break
                world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
                adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
                xy = adj2cur_lidar[0:2, 3]
                full_adj_track[j, 0:2] = xy
                full_adj_adj_mask[j] = 1

    offset_track = full_adj_track[1:] - full_adj_track[:-1]

    for j in range(future_frames):
        if full_adj_adj_mask[j + 1] == 0:
            offset_track[j] = 0

    ego_fut_trajs = offset_track.copy()  # (future_frames, 2)
    ego_fut_masks = full_adj_adj_mask[-future_frames:].copy()  # (future_frames,)

    if ego_fut_masks.sum() < future_frames:
        return [], False

    cum = np.cumsum(ego_fut_trajs, axis=0)
    pts = [(float(cum[i, 0]), float(cum[i, 1])) for i in range(future_frames)]
    return pts, True


def parse_from_info_like_dataset(data_infos, idx):
    """Extract ego speed/accel/command in a way consistent with the produced infos pkl."""
    info = data_infos[idx]

    # speed: converter sets ego_vel = [speed, 0, 0]
    speed = 0.0
    if 'ego_vel' in info and info['ego_vel'] is not None:
        try:
            speed = float(np.linalg.norm(np.array(info['ego_vel'])[0:2]))
        except Exception:
            speed = _ensure_py_scalar(info['ego_vel'][0], 0.0) if hasattr(info['ego_vel'], '__len__') else 0.0

    acc_x, acc_y = 0.0, 0.0
    if 'ego_accel' in info and info['ego_accel'] is not None:
        a = np.array(info['ego_accel'])
        if a.shape[0] >= 2:
            acc_x = float(a[0])
            acc_y = float(a[1])

    # Command processing matching Bench2DriveDataset.command2hot
    # "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"
    # Raw: 1=Left, 2=Right, 3=Straight, 4=Follow, 5=ChangeLeft, 6=ChangeRight
    raw_command = int(info.get('command_near', 4))
    if raw_command < 0:
        raw_command = 4
    command = raw_command - 1

    return speed, acc_x, acc_y, command


def build_view_tokens(image_paths):
    """Build view tokens with image placeholders."""
    parts = []
    for cam_name, img_path in zip(B2D_CAMERA_ORDER, image_paths):
        parts.append(f"{B2D_VIEW_TOKENS[cam_name]}:")
        parts.append("<image>")
    return "\n".join(parts)


def generate_qa_sample(
    sample_id,
    scenario,
    frame_idx,
    data_root,
    future_traj=None,
    speed=0.0,
    acc_x=0.0,
    acc_y=0.0,
    command=3,
    include_answer=True,
    traj_repr='waypoint',
):
    """
    Generate a single QA sample.

    Args:
        sample_id: Unique ID
        scenario: Scenario folder name (e.g., InterurbanActorFlow_Town12_Route1296_Weather7)
        frame_idx: Frame index
        data_root: Root path to B2D data
        future_traj: (6, 3) trajectory or None
        speed: Ego speed in m/s
        acc_x, acc_y: Acceleration
        command: Navigation command (0-5)
        include_answer: Whether to include GPT answer
    """
    # Build image paths (always local paths under data_root)
    image_paths = []
    path_root = data_root
    scenario_rel = scenario
    # Some infos store folder like "v1/<scenario>" while data_root already ends with ".../v1".
    if scenario_rel.startswith('v1' + os.sep):
        scenario_rel = scenario_rel.split(os.sep, 1)[1]

    for cam_name in B2D_CAMERA_ORDER:
        cam_folder = B2D_CAMERA_MAP[cam_name]
        img_path = os.path.join(
            path_root,
            scenario_rel,
            cam_folder,
            f"{frame_idx:05d}.jpg"
        )
        image_paths.append(img_path)

    # Get command text
    cmd_text = B2D_COMMAND_MAP.get(command, "LANE FOLLOW")

    # Build conversation
    conversations = []

    # System message
    conversations.append({
        "from": "system",
        "value": B2D_SYSTEM_PROMPT
    })

    # Human message
    view_tokens = build_view_tokens(image_paths)
    human_value = B2D_HUMAN_TEMPLATE.format(
        view_tokens=view_tokens,
        speed=speed,
        acc_x=acc_x,
        acc_y=acc_y,
        cmd=cmd_text
    )
    conversations.append({
        "from": "human",
        "value": human_value
    })

    # GPT answer
    if include_answer and future_traj is not None:
        if traj_repr == 'residual':
            future_traj = traj_to_residual(future_traj)
        points_str = format_traj_points(future_traj, with_delta=False)
        gpt_value = B2D_GPT_TEMPLATE.format(points=points_str)
        conversations.append({
            "from": "gpt",
            "value": gpt_value
        })

    return {
        "id": sample_id,
        "image": image_paths,
        "conversations": conversations
    }


def _load_infos(infos_path):
    """Load infos pkl the same way mmcv.load/joblib would.

    Prefer joblib when available (common for large pickles), fallback to pickle.
    """
    try:
        import joblib

        return joblib.load(infos_path)
    except Exception:
        import pickle

        with open(infos_path, 'rb') as f:
            return pickle.load(f)


def generate_qa_from_infos(
    data_root,
    infos_path,
    output_path,
    start_id=0,
    max_samples=None,
    past_frames=2,
    future_frames=6,
    sample_rate=1,
    traj_repr='waypoint',
):
    """Generate QA JSONL using infos pkl, aligning with Bench2DriveDataset.get_ego_trajs."""
    data_infos = _load_infos(infos_path)

    if isinstance(data_infos, dict) and 'infos' in data_infos:
        data_infos = data_infos['infos']

    total_avail = len(data_infos)
    total = min(max_samples, total_avail) if max_samples else total_avail
    iterator = tqdm(range(total), desc='Generating QA samples')

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    missing_traj = 0
    wrote_with_traj = 0

    with open(output_path, 'w') as f:
        for i in iterator:
            sample_id = start_id + i
            info = data_infos[i]

            scenario = info['folder']
            frame_idx = int(info['frame_idx'])

            speed, acc_x, acc_y, command = parse_from_info_like_dataset(data_infos, i)

            # Align with dataset default planning GT: temporal 2hz (interval=5 at 10Hz)
            future_traj, ok = compute_ego_fut_traj_like_dataset(
                data_infos,
                i,
                future_frames=future_frames,
                interval=5,
            )
            if not ok:
                missing_traj += 1
                continue

            sample = generate_qa_sample(
                sample_id=sample_id,
                scenario=scenario,
                frame_idx=frame_idx,
                data_root=data_root,
                future_traj=future_traj if ok else None,
                speed=speed,
                acc_x=acc_x,
                acc_y=acc_y,
                command=command,
                include_answer=ok,
                traj_repr=traj_repr,
            )

            if ok:
                wrote_with_traj += 1

            f.write(json.dumps(sample) + '\n')

    print(
        f"Generated {total} QA samples to {output_path} | "
        f"with_traj={wrote_with_traj}, missing_traj={missing_traj}"
    )


def main():
    parser = argparse.ArgumentParser(description='Generate B2D QA JSONL Dataset')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root path to B2D data (contains scenario folders)')
    parser.add_argument('--infos-path', type=str, default=None,
                        help='Path to infos pkl (e.g., data/infos/b2d_infos_train.pkl). If set, generation aligns with Bench2DriveDataset.get_ego_trajs().')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to generate')
    parser.add_argument('--start-id', type=int, default=0,
                        help='Starting sample ID')

    # keep same defaults as dataset
    parser.add_argument('--future-frames', type=int, default=6)
    parser.add_argument('--traj-repr', type=str, default='waypoint', choices=['waypoint', 'residual'],
                        help='Trajectory representation in gpt answer: waypoint=[(x,y)...], residual=[pt0, d1, d2, ...]')

    args = parser.parse_args()

    if not args.infos_path:
        raise ValueError('--infos-path is required for aligned trajectory generation')

    generate_qa_from_infos(
        data_root=args.data_root,
        infos_path=args.infos_path,
        output_path=args.output,
        start_id=args.start_id,
        max_samples=args.max_samples,
        past_frames=2,
        future_frames=args.future_frames,
        sample_rate=1,
        traj_repr=args.traj_repr,
    )


if __name__ == '__main__':
    main()
