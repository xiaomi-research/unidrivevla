#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEAD_ROOT="${LEAD_RELEASE_ROOT:-${SCRIPT_DIR}}"
B2D_ROOT="${BENCH2DRIVE_ROOT:-${LEAD_ROOT}/Bench2Drive}"
RUN_NAME=unidrivevla_lead_stage3_motion_4hz8_2b_4node32gpu_bs16_accum1_9ep_from_stage2_ds87_iter11358_keep9
CONFIG="${LEAD_CONFIG:-${B2D_ROOT}/projects/configs/unidrivevla_lead_stage2_motion_4hz8_2b.py}"
CHECKPOINT="${LEAD_CHECKPOINT:?Set LEAD_CHECKPOINT to the external Epoch9 checkpoint}"
QWEN_BASE="${LEAD_QWEN_BASE:?Set LEAD_QWEN_BASE to the external Qwen3-VL-2B-Instruct directory}"
MOTION_ANCHOR="$B2D_ROOT/data/kmeans/lead_motion_6_4hz8step.npy"
PYTHON_BIN="${LEAD_PYTHON_BIN:?Set LEAD_PYTHON_BIN to the evaluation Python executable}"
CARLA_ROOT="${LEAD_CARLA_ROOT:?Set LEAD_CARLA_ROOT to the external CARLA installation}"
CARLA_PYTHON_BINDING="${LEAD_CARLA_PYTHON_BINDING:-}"
RUN_ID="${LEAD_STAGE3_DS87_EPOCH3_FULL220_RUN_ID:-latest_4hz_stage3_ds87_epoch9_comfort_full220_v1}"
TRANSFORMERS_SRC="${LEAD_TRANSFORMERS_SRC:-${LEAD_ROOT}/../third_party/transformers-4.57.1/src}"
MMCV_ROOT="${LEAD_MMCV_ROOT:-${LEAD_ROOT}/../third_party/mmcv-1.7.2}"
MMDET3D_ROOT="${LEAD_MMDET3D_ROOT:-${LEAD_ROOT}/../third_party/mmdetection3d-1.0.0rc6}"
ROUTE_SOURCE="${LEAD_ROUTE_SOURCE:?Set LEAD_ROUTE_SOURCE to the external 220-route XML directory}"
LEAD_EXPECTED_INFOS_VERSION="${LEAD_EXPECTED_INFOS_VERSION:-15}"
LEAD_TRAJECTORY_STATS_PATH="${LEAD_TRAJECTORY_STATS_PATH:?Set LEAD_TRAJECTORY_STATS_PATH to the external version-15 Lead trajectory stats JSON}"
export PROJECT_DIR="$B2D_ROOT"
export VLM_PRETRAINED_PATH="$QWEN_BASE"
export CARLA_PYTHON_BINDING
export LEAD_MOTION_ANCHOR="$MOTION_ANCHOR"
export LEAD_EXPECTED_INFOS_VERSION
export LEAD_TRAJECTORY_STATS_PATH
export LEAD_INFO_FRAME_STRIDE=1
export LEAD_INFO_FRAME_OFFSET=0
export LEAD_NUM_TRAIN_SAMPLES="${LEAD_NUM_TRAIN_SAMPLES:-968710}"
export LEAD_VAL_RATIO=0
export LEAD_SENSOR_PERTURBATION=1
export LEAD_ROUTE_QUALITY_FILTER=tfv6_posttrain
export LEAD_ROUTE_SKIP_FIRST=8
export LEAD_ROUTE_SKIP_LAST=8
if [[ "$LEAD_EXPECTED_INFOS_VERSION" != "15" ]]; then
    echo "LEAD evaluation requires LEAD_EXPECTED_INFOS_VERSION=15." >&2
    exit 1
fi
export LEAD_MOTION_LOAD_FROM=""
export LEAD_MOTION_RESUME_FROM=""
REQUESTED_GPU_IDS="${LEAD_STAGE3_DS87_EPOCH9_GPU_IDS:-0,1,2,3,4,5,6,7}"
CARLA_NONROOT_MODE="${LEAD_STAGE3_DS87_EPOCH3_USE_B2D_NONROOT_CARLA:-0}"
CARLA_HOME="${LEAD_STAGE3_DS87_EPOCH3_CARLA_HOME:-/tmp/carla-home-1001}"
CARLA_XDG_DIR="${LEAD_STAGE3_DS87_EPOCH3_CARLA_XDG_RUNTIME_DIR:-/tmp/xdg-lead-eval-stage3-ds87-epoch3-${UID:-0}}"
export MMCV_ROOT MMDET3D_ROOT
export PYTHONPATH="$TRANSFORMERS_SRC:$B2D_ROOT:$MMCV_ROOT:$MMDET3D_ROOT:${PYTHONPATH:-}"

if [[ "$CARLA_NONROOT_MODE" != "0" && "$CARLA_NONROOT_MODE" != "1" ]]; then
    echo "LEAD_STAGE3_DS87_EPOCH3_USE_B2D_NONROOT_CARLA must be 0 or 1" >&2
    exit 1
fi

for required_path in \
    "$CONFIG" \
    "$CHECKPOINT" \
    "$QWEN_BASE" \
    "$MOTION_ANCHOR" \
    "$PYTHON_BIN" \
    "$CARLA_ROOT/CarlaUE4.sh"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path is missing: $required_path" >&2
        exit 1
    fi
done
if [[ ! -f "$LEAD_TRAJECTORY_STATS_PATH" ]]; then
    echo "Required Lead trajectory stats file is missing: $LEAD_TRAJECTORY_STATS_PATH" >&2
    exit 1
fi

checkpoint_size_before=$(stat -c %s "$CHECKPOINT")
sleep "${LEAD_STAGE3_DS87_EPOCH3_CHECKPOINT_STABILITY_SECONDS:-5}"
checkpoint_size_after=$(stat -c %s "$CHECKPOINT")
if [[ "$checkpoint_size_before" != "$checkpoint_size_after" ]]; then
    echo "Epoch 9 checkpoint is still being written: $CHECKPOINT" >&2
    echo "size changed from $checkpoint_size_before to $checkpoint_size_after bytes" >&2
    exit 1
fi
echo "Epoch 9 checkpoint is present and size-stable: $CHECKPOINT ($checkpoint_size_after bytes)"

# Reject a wrong epoch, non-motion config, or partial model state before
# starting CARLA or creating route scripts.
"$PYTHON_BIN" - "$CONFIG" "$CHECKPOINT" "$QWEN_BASE" "$MOTION_ANCHOR" <<'PY'
import pathlib
import sys

import torch
from mmcv import Config

config_path = pathlib.Path(sys.argv[1]).resolve()
checkpoint_path = pathlib.Path(sys.argv[2]).resolve()
qwen_base_path = pathlib.Path(sys.argv[3]).resolve()
motion_anchor_path = pathlib.Path(sys.argv[4]).resolve()

cfg = Config.fromfile(str(config_path))
planning = cfg.model.planning_head
decoder = planning.unified_decoder_cfg

expected_tasks = ["det", "map", "ego", "motion"]
if list(decoder.get("task_select", [])) != expected_tasks:
    raise SystemExit(
        "Evaluation config is not the trained motion config: "
        f"task_select={decoder.get('task_select')}"
    )
expected_queries = ["det", "map", "ego"]
if list(decoder.get("query_select", [])) != expected_queries:
    raise SystemExit(
        "Unexpected motion query_select: "
        f"{decoder.get('query_select')}"
    )
if planning.get("planning_query_layout") != "action_expert":
    raise SystemExit(
        "Unexpected planning_query_layout: "
        f"{planning.get('planning_query_layout')}"
    )
if int(planning.get("action_horizon", -1)) != 8:
    raise SystemExit(f"Unexpected action_horizon: {planning.get('action_horizon')}")
if int(planning.get("route_points_prediction", -1)) != 10:
    raise SystemExit(
        "Unexpected route_points_prediction: "
        f"{planning.get('route_points_prediction')}"
    )

motion_refine = decoder.get("motion_refine_layer", {})
if int(motion_refine.get("fut_ts", -1)) != 8:
    raise SystemExit(f"Unexpected motion fut_ts: {motion_refine.get('fut_ts')}")
if int(motion_refine.get("fut_mode", -1)) != 6:
    raise SystemExit(f"Unexpected motion fut_mode: {motion_refine.get('fut_mode')}")

configured_anchor = pathlib.Path(decoder.get("motion_anchor", "")).resolve()
if configured_anchor != motion_anchor_path:
    raise SystemExit(
        "Motion anchor mismatch: "
        f"config={configured_anchor} expected={motion_anchor_path}"
    )
configured_base = pathlib.Path(planning.get("pretrained_path", "")).resolve()
if configured_base != qwen_base_path:
    raise SystemExit(
        "Qwen base initialization mismatch: "
        f"config={configured_base} expected={qwen_base_path}"
    )

if planning.get("train_vlm") is not True:
    raise SystemExit(f"Expected train_vlm=True for this run, got {planning.get('train_vlm')}")
lora_cfg = planning.get("lora_cfg", {})
if int(lora_cfg.get("r", -1)) != 64 or int(lora_cfg.get("lora_alpha", -1)) != 128:
    raise SystemExit(f"Unexpected ordinary LoRA config: {lora_cfg}")
if planning.get("peft_cfg") is not None or planning.get("visual_wasp_cfg") is not None:
    raise SystemExit("WASP adapters must be disabled for this ordinary motion run")

checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
if not isinstance(checkpoint, dict):
    raise SystemExit(f"Unsupported checkpoint object: {type(checkpoint).__name__}")
state_dict = None
for key in ("module", "state_dict", "model"):
    candidate = checkpoint.get(key)
    if isinstance(candidate, dict):
        state_dict = candidate
        break
if state_dict is None:
    raise SystemExit("Checkpoint does not contain a model state dictionary")

expected_metadata = {
    "iter": 17036,
    "global_steps": 17037,
    "global_samples": 8722944,
    "dp_world_size": 32,
    "mp_world_size": 1,
    "skipped_steps": 0,
}
metadata_mismatches = {
    key: (checkpoint.get(key), expected)
    for key, expected in expected_metadata.items()
    if checkpoint.get(key) != expected
}
if metadata_mismatches:
    raise SystemExit(
        "Checkpoint is not the Stage3 DS87 Epoch 9 model: "
        f"{metadata_mismatches}"
    )

lora_keys = [key for key in state_dict if "lora_" in key.lower()]
adapter_tensor_counts = {
    "vision": sum(
        ".model.visual.base_model.model.blocks." in key for key in lora_keys
    ),
    "projector_merger": sum(
        ".model.visual.base_model.model.merger." in key for key in lora_keys
    ),
    "projector_deepstack": sum(
        ".model.visual.base_model.model.deepstack_merger_list." in key
        for key in lora_keys
    ),
    "llm": sum(
        ".model.language_model.layers." in key for key in lora_keys
    ),
}
motion_keys = sorted(key for key in state_dict if "motion" in key.lower())
required_motion_keys = {
    "planning_head.motion_proj_up.weight",
    "planning_head.motion_proj_down.weight",
    "planning_head.unified_decoder.motion_anchor",
    "planning_head.unified_decoder.motion_anchor_encoder.encoder.0.weight",
    "planning_head.unified_decoder.motion_refine.0.motion_cls_branch.0.weight",
    "planning_head.unified_decoder.motion_refine.0.motion_reg_branch.0.weight",
}
missing_motion_keys = sorted(required_motion_keys.difference(state_dict))
if (
    len(state_dict) != 2763
    or len(lora_keys) != 392
    or len(motion_keys) != 75
    or missing_motion_keys
):
    raise SystemExit(
        "Checkpoint architecture mismatch: "
        f"model_keys={len(state_dict)} lora_tensors={len(lora_keys)} "
        f"adapter_groups={adapter_tensor_counts} motion_keys={len(motion_keys)} "
        f"missing_motion={missing_motion_keys}"
    )

print(
    "Stage3 DS87 Epoch 9 ordinary motion checkpoint preflight passed: "
    f"global_steps={checkpoint['global_steps']} "
    f"global_samples={checkpoint['global_samples']} "
    f"model_keys={len(state_dict)} lora_tensors={len(lora_keys)} "
    f"motion_keys={len(motion_keys)}"
)
print(f"CONFIG={config_path}")
print(f"CHECKPOINT={checkpoint_path}")
print(f"QWEN_BASE={qwen_base_path}")
print(f"MOTION_ANCHOR={motion_anchor_path}")
PY

if [[ "${LEAD_EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "Stage3 DS87 Epoch 9 preflight completed; evaluation was not started."
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to discover visible evaluation GPUs" >&2
    exit 1
fi

declare -A VISIBLE_GPUS=()
while IFS= read -r gpu_id; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        VISIBLE_GPUS["$gpu_id"]=1
    fi
done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)

IFS=',' read -r -a requested_gpu_array <<< "$REQUESTED_GPU_IDS"
selected_gpu_array=()
for gpu_id in "${requested_gpu_array[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if ! [[ "$gpu_id" =~ ^[0-7]$ ]]; then
        echo "Invalid LEAD_STAGE3_DS87_EPOCH9_GPU_IDS entry '$gpu_id'; expected physical ids 0 through 7" >&2
        exit 1
    fi
    if [[ -n "${VISIBLE_GPUS[$gpu_id]:-}" ]]; then
        selected_gpu_array+=("$gpu_id")
    else
        echo "GPU $gpu_id is not visible on this node; skipping its CARLA probe." >&2
    fi
done
if [[ "${#selected_gpu_array[@]}" -eq 0 ]]; then
    echo "None of the requested GPUs are visible: requested=$REQUESTED_GPU_IDS" >&2
    exit 1
fi
SELECTED_GPU_IDS=$(IFS=,; printf '%s' "${selected_gpu_array[*]}")
SELECTED_GPU_COUNT=${#selected_gpu_array[@]}

echo "Stage3 DS87 Epoch 9 evaluation will probe visible GPUs: $SELECTED_GPU_IDS"
echo "CARLA preflight runs before route scripts are generated or dispatched."
echo "Only healthy CARLA GPU slots will receive routes."
echo "CARLA non-root launch mode: $CARLA_NONROOT_MODE"
echo "Resume mode will skip completed non-crash route JSON files under run $RUN_ID."
echo "Evaluation output: $LEAD_ROOT/outputs/evaluation/$RUN_ID"
echo "Stage3 DS87 Epoch 9 checkpoint: $CHECKPOINT"

mkdir -p "$CARLA_HOME" "$CARLA_XDG_DIR"
chmod 700 "$CARLA_HOME" "$CARLA_XDG_DIR" || true

cd "$LEAD_ROOT"
HOME="$CARLA_HOME" \
CARLA_XDG_RUNTIME_DIR="$CARLA_XDG_DIR" \
LEAD_PROJECT_ROOT="$LEAD_ROOT" \
PROJECT_DIR="$B2D_ROOT" \
CONFIG="$CONFIG" \
CHECKPOINT="$CHECKPOINT" \
VLM_PRETRAINED_PATH="$QWEN_BASE" \
TRANSFORMERS_SRC="$TRANSFORMERS_SRC" \
PYTHON_BIN="$PYTHON_BIN" \
CARLA_ROOT="$CARLA_ROOT" \
CARLA_PYTHON_BINDING="$CARLA_PYTHON_BINDING" \
CUDA_VISIBLE_DEVICES="$SELECTED_GPU_IDS" \
LOCAL_GPUS_PER_NODE=8 \
LEAD_LOCAL_MAX_PARALLEL_JOBS="$SELECTED_GPU_COUNT" \
LEAD_LOCAL_GPU_IDS="$SELECTED_GPU_IDS" \
BENCH2DRIVE_NUM_SHARDS=1 \
BENCH2DRIVE_SHARD_INDEX=0 \
LEAD_FORCE_LOCAL=1 \
RUN_EVAL=1 \
USE_B2D_NONROOT_CARLA="$CARLA_NONROOT_MODE" \
CLEAN_EXISTING_CARLA=0 \
CARLA_SEQUENTIAL_PREFLIGHT=1 \
CARLA_SEQUENTIAL_STARTUP=1 \
CARLA_PREFLIGHT_TIMEOUT=90 \
CARLA_STARTUP_SLEEP=60 \
    ROUTE_STARTUP_STAGGER=65 \
PORT_BASE=33000 \
TM_PORT_BASE=43000 \
PORT_STRIDE=10 \
LEAD_RESUME_SKIP_FINISHED=1 \
LEAD_RESUME_REQUIRE_COMFORT_METRICS=1 \
    SAVE_NAME=latest_4hz_stage3_ds87_epoch9_comfort \
    ROUTE_SOURCE="$ROUTE_SOURCE" \
EXPERIMENT_RUN_ID="$RUN_ID" \
UNIDRIVEVLA_LEAD_ALLOW_CONFIG_MISMATCH=0 \
UNIDRIVEVLA_LEAD_ALLOW_PARTIAL_CHECKPOINT=0 \
UNIDRIVEVLA_LEAD_REMOVE_PROMPT_SPEED=0 \
UNIDRIVEVLA_LEAD_USE_KALMAN=1 \
UNIDRIVEVLA_LEAD_CREEPING=1 \
UNIDRIVEVLA_LEAD_CREEP_POINT_SOURCE=single_lidar \
UNIDRIVEVLA_LEAD_STOP_SIGN=0 \
UNIDRIVEVLA_LEAD_TEMPORAL_MODE=close_loop \
UNIDRIVEVLA_LEAD_CONTROL=tfv6_route_speed \
UNIDRIVEVLA_LEAD_DELTA_STATS=config \
UNIDRIVEVLA_LEAD_PLAN_COORD=lead_forward_lateral \
UNIDRIVEVLA_LEAD_JPEG_QUALITY=auto \
UNIDRIVEVLA_LEAD_WAYPOINT_HZ=auto \
UNIDRIVEVLA_LEAD_COMMAND_SOURCE=current \
UNIDRIVEVLA_LEAD_ROUTE_PLANNER=lead \
UNIDRIVEVLA_LEAD_ROUTE_POP_DISTANCE=5.0 \
UNIDRIVEVLA_LEAD_ROUTE_ADAPTIVE_POP=1 \
WANDB_MODE=disabled \
bash run_unidrivevla_lead_stage1_bench2drive_eval_8gpu.sh
