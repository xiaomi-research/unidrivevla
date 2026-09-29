# LEAD motion training with native 4Hz, 8-step, 2-second targets.
# This is a separate experiment; the original Stage1, HiP-AD, and 2Hz/4-step
# motion configurations remain unchanged.

_base_ = "./unidrivevla_lead_stage1_unified_2b.py"

import json
import os

import numpy as np


project_dir = os.environ.get(
    "PROJECT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)
lead_motion_fut_ts = 8
lead_motion_fut_mode = 6
lead_motion_anchor = os.environ.get(
    "LEAD_MOTION_ANCHOR",
    os.path.join(project_dir, "data/kmeans/lead_motion_6_4hz8step.npy"),
)
lead_motion_anchor_metadata = os.path.splitext(lead_motion_anchor)[0] + ".json"
lead_motion_stats_path = os.environ.get(
    "LEAD_TRAJECTORY_STATS_PATH",
    os.path.join(
        project_dir,
        "data/lead/infos_4hz_full_motion4hz8_v15/lead_trajectory_stats.json",
    ),
)
default_stage1_checkpoint = os.path.join(
    project_dir,
    "work_dirs2",
    "unidrivevla_lead_stage1_unified_2b_2node16gpu_bs16_accum2_3ep_"
    "full4hz_actionqueries19_stableflow_posttrainfilter_sensorperturb50_"
    "b2dzoo_eager",
    "iter_11358",
    "global_step5679",
    "mp_rank_00_model_states.pt",
)
# An explicitly empty value selects fresh training from the Qwen3-VL base.
# Keeping the environment variable unset preserves the original Stage1-init
# behavior for existing launchers.
load_from = os.environ.get(
    "LEAD_MOTION_LOAD_FROM", default_stage1_checkpoint
).strip() or None
resume_from = os.environ.get("LEAD_MOTION_RESUME_FROM", "").strip() or None

if resume_from:
    # Stage2 resume restores optimizer/scheduler state and must not also apply
    # the Stage1 model-only initialization.
    load_from = None

if resume_from and not os.path.isdir(resume_from):
    raise FileNotFoundError(
        "LEAD motion resume path must be a DeepSpeed checkpoint directory: "
        + repr(resume_from)
    )
if load_from and not os.path.isfile(load_from):
    raise FileNotFoundError(
        "LEAD motion initialization checkpoint does not exist: "
        + repr(load_from)
    )
if not os.path.isfile(lead_motion_anchor):
    raise FileNotFoundError(
        "Missing LEAD 4Hz/8-step motion anchor. Run "
        "prepare_lead_motion_4hz8_data.sh: " + lead_motion_anchor
    )
if not os.path.isfile(lead_motion_anchor_metadata):
    raise FileNotFoundError(
        "Missing LEAD motion anchor metadata: " + lead_motion_anchor_metadata
    )
motion_anchor_array = np.load(lead_motion_anchor, mmap_mode="r")
motion_anchor_shape = tuple(motion_anchor_array.shape)
expected_motion_anchor_shape = (
    9,
    lead_motion_fut_mode,
    lead_motion_fut_ts,
    2,
)
if motion_anchor_shape != expected_motion_anchor_shape:
    raise RuntimeError(
        f"Invalid LEAD motion anchor shape {motion_anchor_shape}; "
        f"expected {expected_motion_anchor_shape}"
    )
if not np.isfinite(motion_anchor_array).all():
    raise RuntimeError("LEAD motion anchor contains NaN or Inf: " + lead_motion_anchor)
# Static/control classes may legitimately have zero anchors. The five dynamic
# LEAD classes must each contain at least one non-zero trajectory.
dynamic_motion_class_indices = [0, 1, 2, 3, 7]
all_zero_dynamic_classes = [
    index
    for index in dynamic_motion_class_indices
    if not np.any(motion_anchor_array[index])
]
if all_zero_dynamic_classes:
    raise RuntimeError(
        "LEAD motion anchor has all-zero dynamic classes "
        + repr(all_zero_dynamic_classes)
        + ": "
        + lead_motion_anchor
    )

with open(lead_motion_anchor_metadata, "r", encoding="utf-8") as metadata_file:
    lead_motion_anchor_metadata_dict = json.load(metadata_file)
del metadata_file
if lead_motion_anchor_metadata_dict.get("shape") != list(expected_motion_anchor_shape):
    raise RuntimeError(
        "LEAD motion anchor metadata shape mismatch: "
        + repr(lead_motion_anchor_metadata_dict.get("shape"))
        + ", expected "
        + repr(list(expected_motion_anchor_shape))
    )
del motion_anchor_array

with open(lead_motion_stats_path, "r", encoding="utf-8") as stats_file:
    lead_motion_stats = json.load(stats_file)
del stats_file
expected_motion_stats = {
    "lead_stage1_infos_version": 15,
    "agent_motion_step_sim_frames": 5,
    "agent_motion_freq_hz": 4.0,
    "agent_motion_horizon_steps": lead_motion_fut_ts,
    "agent_motion_horizon_seconds": 2.0,
    "planning_step_sim_frames": 5,
    "planning_freq_hz": 4.0,
    "planning_horizon_steps": 8,
    "planning_horizon_seconds": 2.0,
}
motion_stats_mismatches = {
    key: (lead_motion_stats.get(key), expected)
    for key, expected in expected_motion_stats.items()
    if lead_motion_stats.get(key) != expected
}
if motion_stats_mismatches:
    raise RuntimeError(
        "LEAD 4Hz/8-step motion stats mismatch: "
        + repr(motion_stats_mismatches)
        + " in "
        + lead_motion_stats_path
    )

task_select = ["det", "map", "ego", "motion"]
# Motion queries are created from matched detection queries, as in HiP-AD.
query_select = ["det", "map", "ego"]

data = dict(
    train=dict(agent_fut_ts=lead_motion_fut_ts, expected_infos_version=15),
    val=dict(agent_fut_ts=lead_motion_fut_ts, expected_infos_version=15),
    test=dict(agent_fut_ts=lead_motion_fut_ts, expected_infos_version=15),
)

model = dict(
    planning_head=dict(
        unified_decoder_cfg=dict(
            task_select=task_select,
            query_select=query_select,
            motion_anchor_encoder=dict(
                type="MotionAnchorSineEncoder",
                embed_dims=256,
            ),
            motion_refine_layer=dict(
                type="SparseMotionRefinementModule",
                embed_dims=256,
                fut_ts=lead_motion_fut_ts,
                fut_mode=lead_motion_fut_mode,
            ),
            motion_sampler=dict(type="SparseMotionTarget"),
            loss_motion_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2,
            ),
            loss_motion_reg=dict(type="L1Loss", loss_weight=0.2),
            motion_anchor=lead_motion_anchor,
        ),
    ),
)

# With neither load nor resume set, the Qwen3-VL weights come from the base
# config and all task-specific modules start from their configured
# initialization. LEAD_MOTION_RESUME_FROM restores model and optimizer state.
lead_motion_lr = float(os.environ.get("LEAD_MOTION_LR", "1e-4"))
if lead_motion_lr <= 0.0:
    raise ValueError(f"LEAD_MOTION_LR must be positive, got {lead_motion_lr}")
optimizer = dict(lr=lead_motion_lr)

# The closed-loop LEAD evaluator does not yet decode an agent-motion metric.
evaluation = dict(eval_mode=dict(with_motion=False))
