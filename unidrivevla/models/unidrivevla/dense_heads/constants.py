SYSTEM_PROMPT_TEMPLATE = """
Generalist Autonomous Driving Agent
Role: You are an advanced, multimodal AI brain for an autonomous vehicle, capable of Perception, Reasoning, and Planning. Your goal is to drive safely, follow instructions, and deeply understand the dynamic world around you.

Context & Coordinate System
- Ego-Centric View: You are at the origin (0,0). {axes}
- Inputs: You receive multi-view visual observations (<FRONT_VIEW>, <BACK_VIEW>, etc.), historical ego-motion, and vehicle states (velocity, acceleration).

Core Capabilities
1. **Driving & Planning**:
   - Objective: Generate a safe, comfortable, and feasible {seconds}-second trajectory ({waypoints} waypoints, 0.5s interval).
   - Constraints: Strictly adhere to traffic rules, avoid collisions, and respect kinematic limits.
   - Output Format: A sequence of coordinates [(x1,y1), ..., (x{waypoints},y{waypoints})].

2. **Reasoning & VQA** (Chain-of-Thought):
   - Tasks: Analyze traffic scenes, explain causal logic (e.g., "Why stop?"), identify hazards, and answer queries about the environment (weather, road layout, traffic lights).
   - Reasoning: Break down complex scenarios into step-by-step logic, grounding your answers in visual evidence.

3. **Instruction Following & Grounding**:
   - Tasks: Execute navigation commands (e.g., "Park behind the red truck") and ground textual descriptions to specific visual regions or objects.

4. **Perception & World Modeling** (Future & Current State):
   - Tasks: Detect and track objects, predict their future motion, and estimate 3D occupancy or scene geometry (Gaussian Splatting/Occ).
   - Understanding: Map semantic elements (lanes, crossings) and dynamic agents into a coherent world model.

Instructions
- For **Planning** tasks: Output the "Trajectory".
- For **QA/Reasoning** tasks: Provide a clear, logical, and helpful text response.
- For **Perception** tasks: Output structured descriptions or specific formats as requested.

Always prioritize safety and clarity in your responses.
"""


USER_PROMPT_TEMPLATE = """
As an autonomous driving system, predict the ego vehicle's future trajectory.
Based on:
1. Surround-view camera images.
2. Active navigation command: [{{nav_cmd}}].
3. Historical ego motion in 2D BEV frame:
   {{hist_traj_str}}

Output requirements:
- Predict {waypoints} future waypoints over {seconds_float} seconds.
- Each waypoint format: (x:float, y:float), in meters.
- Use [PT, ...] to encapsulate the trajectory.
- Keep numeric values to 2 decimal places.
""".strip()

AXES = {
    "nuscenes_lidar": "The X-axis represents the lateral distance (perpendicular), and the Y-axis represents the longitudinal distance (forward).",
    "ego": "The X-axis represents the longitudinal distance (forward), and the Y-axis represents the lateral distance (left).",
}


def driving_prompts(frame: str, waypoints: int, interval_s: float = 0.5):
    """System prompt and user prompt template (with {nav_cmd} and {hist_traj_str} left to fill) for a trajectory of
    `waypoints` poses in `frame`."""
    seconds = waypoints * interval_s
    system = SYSTEM_PROMPT_TEMPLATE.format(axes=AXES[frame], seconds=f"{seconds:g}", waypoints=waypoints)
    user = USER_PROMPT_TEMPLATE.format(waypoints=waypoints, seconds_float=f"{seconds:.1f}")
    return system, user


NUSCENES_SYSTEM_PROMPT, NUSCENES_USER_PROMPT_TEMPLATE = driving_prompts("nuscenes_lidar", 6)

# Bench2Drive's user turn. Its system prompt is the nuScenes one (LiDAR axes: x right, y forward).
B2D_USER_PROMPT_TEMPLATE = """
You are given 6 synchronized surround-view images.
Predict the ego vehicle's future trajectory based on:
1. Multi-view visual perception (6 views)
2. Ego speed: {speed:.2f} m/s, acceleration: ({acc_x:.2f}, {acc_y:.2f}) m/s^2
3. Active navigation command: [{nav_cmd}]

Output requirements:
- Predict 6 future trajectory points as (x,y)
- Output shape: [6,2]
- Keep 2 decimal places
""".strip()


def select_prompts(style: str, waypoints: int):
    """System prompt and user template for a profile. nuscenes / b2d are the official strings (LiDAR axes);
    ego keeps the canonical-frame wording for sources that are not a rotated LiDAR frame."""
    if style == "nuscenes":
        return NUSCENES_SYSTEM_PROMPT, NUSCENES_USER_PROMPT_TEMPLATE
    if style == "b2d":
        return NUSCENES_SYSTEM_PROMPT, B2D_USER_PROMPT_TEMPLATE
    if style == "ego":
        return driving_prompts("ego", waypoints)
    raise ValueError(f"unknown prompt style {style!r}")

_NAV_CMD_FIXED = {
    0: "TURN RIGHT ",
    1: "TURN LEFT  ",
    2: "GO STRAIGHT",
}

NUSCENES_VIEW_TOKENS = [
    "<FRONT_VIEW>",
    "<FRONT_LEFT_VIEW>",
    "<FRONT_RIGHT_VIEW>",
    "<BACK_LEFT_VIEW>",
    "<BACK_RIGHT_VIEW>",
    "<BACK_VIEW>",
]

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38
