#!/usr/bin/env bash
set -euo pipefail

# Official TFV6-style Fail2Drive evaluation for the Stage3 DS87 Epoch9 model.
# This is intentionally single-seed (traffic-manager seed 0), with one local
# route worker per requested GPU.  Set *_RUN_EVAL=0 for route/job generation
# smoke without requiring the CARLA_F2D executable to be present yet.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEAD_ROOT="${LEAD_RELEASE_ROOT:-$SCRIPT_DIR}"
B2D_ROOT="${BENCH2DRIVE_ROOT:-$LEAD_ROOT/Bench2Drive}"
CONFIG="${LEAD_CONFIG:-$B2D_ROOT/projects/configs/unidrivevla_lead_stage2_motion_4hz8_2b.py}"
CHECKPOINT="${LEAD_CHECKPOINT:?Set LEAD_CHECKPOINT to the external Stage3 Epoch9 checkpoint}"
QWEN_BASE="${LEAD_QWEN_BASE:?Set LEAD_QWEN_BASE to the external Qwen3-VL-2B-Instruct directory}"
MOTION_ANCHOR="$B2D_ROOT/data/kmeans/lead_motion_6_4hz8step.npy"
PYTHON_BIN="${LEAD_PYTHON_BIN:-python}"
TRANSFORMERS_SRC="${LEAD_TRANSFORMERS_SRC:-$LEAD_ROOT/../third_party/transformers-4.57.1/src}"
MMCV_ROOT="${LEAD_MMCV_ROOT:-$LEAD_ROOT/../third_party/mmcv-1.7.2}"
MMDET3D_ROOT="${LEAD_MMDET3D_ROOT:-$LEAD_ROOT/../third_party/mmdetection3d-1.0.0rc6}"
export MMCV_ROOT MMDET3D_ROOT
ROUTE_SOURCE="${LEAD_STAGE3_DS87_EPOCH9_F2D_ROUTE_SOURCE:-${LEAD_ROUTE_SOURCE:-}}"
CARLA_ROOT="${LEAD_STAGE3_DS87_EPOCH9_F2D_CARLA_ROOT:-${LEAD_CARLA_ROOT:-}}"
CARLA_PYTHON_BINDING="${LEAD_STAGE3_DS87_EPOCH9_F2D_CARLA_PYTHON_BINDING:-${LEAD_CARLA_PYTHON_BINDING:-}}"
LEAD_EXPECTED_INFOS_VERSION="${LEAD_EXPECTED_INFOS_VERSION:-15}"
LEAD_TRAJECTORY_STATS_PATH="${LEAD_TRAJECTORY_STATS_PATH:?Set LEAD_TRAJECTORY_STATS_PATH to the external version-15 Lead trajectory stats JSON}"

RUN_ID="${LEAD_STAGE3_DS87_EPOCH9_F2D_RUN_ID:-latest_4hz_stage3_ds87_epoch9_fail2drive_seed0_v1}"
REQUESTED_GPU_IDS="${LEAD_STAGE3_DS87_EPOCH9_F2D_GPU_IDS:-0,1,2,3,4,5,6,7}"
ROUTE_IDS="${LEAD_STAGE3_DS87_EPOCH9_F2D_ROUTE_IDS:-}"
RUN_EVAL="${LEAD_STAGE3_DS87_EPOCH9_F2D_RUN_EVAL:-1}"
CREEP_POINT_SOURCE="${LEAD_STAGE3_DS87_EPOCH9_F2D_CREEP_POINT_SOURCE:-single_lidar}"
CARLA_NONROOT_MODE="${LEAD_STAGE3_DS87_EPOCH9_F2D_USE_B2D_NONROOT_CARLA:-0}"
CARLA_HOME="${LEAD_STAGE3_DS87_EPOCH9_F2D_CARLA_HOME:-/tmp/carla-home-1001}"
CARLA_XDG_DIR="${LEAD_STAGE3_DS87_EPOCH9_F2D_CARLA_XDG_RUNTIME_DIR:-/tmp/xdg-lead-eval-stage3-ds87-epoch9-f2d-${UID:-0}}"
export PYTHONPATH="$TRANSFORMERS_SRC:$LEAD_ROOT:$B2D_ROOT:$MMCV_ROOT:$MMDET3D_ROOT:${PYTHONPATH:-}"
export LEAD_EXPECTED_INFOS_VERSION LEAD_TRAJECTORY_STATS_PATH
export PROJECT_DIR="$B2D_ROOT"
export VLM_PRETRAINED_PATH="$QWEN_BASE"
export LEAD_MOTION_ANCHOR="$MOTION_ANCHOR"
export LEAD_MOTION_LOAD_FROM=""
export LEAD_MOTION_RESUME_FROM=""
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

if [[ "$RUN_EVAL" != "0" && "$RUN_EVAL" != "1" ]]; then
    echo "LEAD_STAGE3_DS87_EPOCH9_F2D_RUN_EVAL must be 0 or 1" >&2
    exit 1
fi
if [[ "$CARLA_NONROOT_MODE" != "0" && "$CARLA_NONROOT_MODE" != "1" ]]; then
    echo "LEAD_STAGE3_DS87_EPOCH9_F2D_USE_B2D_NONROOT_CARLA must be 0 or 1" >&2
    exit 1
fi
if [[ "$CREEP_POINT_SOURCE" != "single_lidar" ]]; then
    echo "LEAD_STAGE3_DS87_EPOCH9_F2D_CREEP_POINT_SOURCE must be single_lidar" >&2
    exit 1
fi

for required_value in ROUTE_SOURCE CARLA_ROOT CARLA_PYTHON_BINDING; do
    if [[ -z "${!required_value}" ]]; then
        echo "$required_value must be supplied as an external Fail2Drive prerequisite" >&2
        exit 1
    fi
done

for required_path in "$LEAD_ROOT" "$B2D_ROOT" "$CONFIG" "$CHECKPOINT" "$QWEN_BASE" "$MOTION_ANCHOR" "$PYTHON_BIN" "$ROUTE_SOURCE" "$CARLA_PYTHON_BINDING"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path is missing: $required_path" >&2
        exit 1
    fi
done
if [[ "$RUN_EVAL" == "1" && ! -f "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "Fail2Drive CARLA executable is missing: $CARLA_ROOT/CarlaUE4.sh" >&2
    echo "Extract fail2drive_simulator.tar.gz there or set LEAD_STAGE3_DS87_EPOCH9_F2D_CARLA_ROOT." >&2
    exit 1
fi
if [[ ! -f "$LEAD_TRAJECTORY_STATS_PATH" ]]; then
    echo "Required Lead trajectory stats file is missing: $LEAD_TRAJECTORY_STATS_PATH" >&2
    exit 1
fi

checkpoint_size_before=$(stat -c %s "$CHECKPOINT")
sleep "${LEAD_STAGE3_DS87_EPOCH9_F2D_CHECKPOINT_STABILITY_SECONDS:-2}"
checkpoint_size_after=$(stat -c %s "$CHECKPOINT")
if [[ "$checkpoint_size_before" != "$checkpoint_size_after" ]]; then
    echo "Checkpoint is still being written: $CHECKPOINT" >&2
    exit 1
fi

"$PYTHON_BIN" - "$ROUTE_SOURCE" "$CONFIG" "$CHECKPOINT" "$QWEN_BASE" "$MOTION_ANCHOR" <<'PY'
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

route_dir = pathlib.Path(sys.argv[1]).resolve()
config_path = pathlib.Path(sys.argv[2]).resolve()
checkpoint_path = pathlib.Path(sys.argv[3]).resolve()
qwen_path = pathlib.Path(sys.argv[4]).resolve()
motion_anchor = pathlib.Path(sys.argv[5]).resolve()
route_files = sorted(route_dir.glob("*.xml"))
if len(route_files) != 200:
    raise SystemExit(f"Fail2Drive route count mismatch: expected 200, found {len(route_files)}")

pattern = re.compile(r"^(Base|Generalization)_(.+)_(\d{4})\.xml$")
seen = {"Base": {}, "Generalization": {}}
classes = set()
for path in route_files:
    match = pattern.fullmatch(path.name)
    if match is None:
        raise SystemExit(f"Unexpected Fail2Drive route filename: {path.name}")
    split, scenario, suffix = match.groups()
    route_id = int(suffix)
    expected_range = range(0, 100) if split == "Base" else range(1000, 1100)
    if route_id not in expected_range:
        raise SystemExit(f"Unexpected {split} route ID in {path.name}: {route_id}")
    if route_id in seen[split]:
        raise SystemExit(f"Duplicate {split} route ID {route_id}")
    root = ET.parse(path).getroot()
    routes = list(root.iter("route"))
    if len(routes) != 1:
        raise SystemExit(f"Expected one route in {path}, found {len(routes)}")
    xml_id = routes[0].get("id")
    if xml_id != str(route_id):
        raise SystemExit(f"Route ID mismatch in {path.name}: filename={route_id} XML={xml_id!r}")
    seen[split][route_id] = scenario
    classes.add(scenario)

if set(seen["Base"]) != set(range(100)) or set(seen["Generalization"]) != set(range(1000, 1100)):
    raise SystemExit("Fail2Drive route IDs are not complete 0..99 and 1000..1099 splits")
for base_id, scenario in seen["Base"].items():
    if seen["Generalization"].get(base_id + 1000) != scenario:
        raise SystemExit(f"Base/Generalization scenario pair mismatch for route {base_id}")
if len(classes) != 17:
    raise SystemExit(f"Fail2Drive scenario class count mismatch: expected 17, found {len(classes)}")

# Keep the checkpoint identity check strict enough to prevent accidentally
# evaluating a different Stage3 epoch or a non-motion model.
import torch
from mmcv import Config
cfg = Config.fromfile(str(config_path))
planning = cfg.model.planning_head
decoder = planning.unified_decoder_cfg
if list(decoder.get("task_select", [])) != ["det", "map", "ego", "motion"]:
    raise SystemExit(f"Unexpected task_select: {decoder.get('task_select')}")
if list(decoder.get("query_select", [])) != ["det", "map", "ego"]:
    raise SystemExit(f"Unexpected query_select: {decoder.get('query_select')}")
if planning.get("planning_query_layout") != "action_expert":
    raise SystemExit(f"Unexpected planning_query_layout: {planning.get('planning_query_layout')}")
if int(planning.get("action_horizon", -1)) != 8 or int(planning.get("route_points_prediction", -1)) != 10:
    raise SystemExit("Unexpected action horizon or route point count")
motion_refine = decoder.get("motion_refine_layer", {})
if int(motion_refine.get("fut_ts", -1)) != 8 or int(motion_refine.get("fut_mode", -1)) != 6:
    raise SystemExit("Unexpected motion refinement dimensions")
if pathlib.Path(decoder.get("motion_anchor", "")).resolve() != motion_anchor:
    raise SystemExit("Motion anchor mismatch")
if pathlib.Path(planning.get("pretrained_path", "")).resolve() != qwen_path:
    raise SystemExit("Qwen base initialization mismatch")

checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
if not isinstance(checkpoint, dict) or checkpoint.get("global_steps") != 17037 or checkpoint.get("global_samples") != 8722944:
    raise SystemExit("Checkpoint metadata is not the expected Stage3 DS87 Epoch9 model")
state_dict = next((checkpoint[key] for key in ("module", "state_dict", "model") if isinstance(checkpoint.get(key), dict)), None)
if state_dict is None:
    raise SystemExit("Checkpoint does not contain a model state dictionary")
if len(state_dict) != 2763 or len([k for k in state_dict if "motion" in k.lower()]) != 75:
    raise SystemExit("Checkpoint architecture does not match the expected motion model")
print(f"Fail2Drive preflight passed: routes=200 classes=17 checkpoint_steps={checkpoint['global_steps']} motion_keys=75")
PY

IFS=',' read -r -a requested_gpu_array <<< "$REQUESTED_GPU_IDS"
selected_gpu_array=()
if [[ "$RUN_EVAL" == "1" ]]; then
    declare -A visible_gpus=()
    while IFS= read -r gpu_id; do
        gpu_id="${gpu_id//[[:space:]]/}"
        [[ "$gpu_id" =~ ^[0-9]+$ ]] && visible_gpus["$gpu_id"]=1
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
    visible_gpus=()
fi
for gpu_id in "${requested_gpu_array[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if ! [[ "$gpu_id" =~ ^[0-7]$ ]]; then
        echo "Invalid GPU id '$gpu_id'; expected 0 through 7" >&2
        exit 1
    fi
    if [[ "$RUN_EVAL" == "0" || -n "${visible_gpus[$gpu_id]:-}" ]]; then
        selected_gpu_array+=("$gpu_id")
    fi
done
if [[ "${#selected_gpu_array[@]}" -eq 0 ]]; then
    echo "No requested evaluation GPU is available: $REQUESTED_GPU_IDS" >&2
    exit 1
fi
SELECTED_GPU_IDS=$(IFS=,; printf '%s' "${selected_gpu_array[*]}")
MAX_PARALLEL_JOBS="${LEAD_STAGE3_DS87_EPOCH9_F2D_MAX_PARALLEL_JOBS:-${#selected_gpu_array[@]}}"
if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "LEAD_STAGE3_DS87_EPOCH9_F2D_MAX_PARALLEL_JOBS must be a positive integer" >&2
    exit 1
fi
if (( MAX_PARALLEL_JOBS > ${#selected_gpu_array[@]} )); then
    MAX_PARALLEL_JOBS=${#selected_gpu_array[@]}
fi


echo "Stage3 DS87 Epoch9 Fail2Drive evaluation"
echo "Checkpoint: $CHECKPOINT"
echo "Maximum parallel routes: $MAX_PARALLEL_JOBS"
echo "Routes: $ROUTE_SOURCE (Base=100, Generalization=100)"
if [[ -n "$ROUTE_IDS" ]]; then
    echo "Route subset: $ROUTE_IDS"
fi
echo "Seed: 0 (single seed, official TFV6 direct-script behavior)"
echo "Controller: tfv6_route_speed; Kalman=1; creeping=$CREEP_POINT_SOURCE; stop-sign=0"
echo "CARLA_ROOT: $CARLA_ROOT"
echo "CARLA Python binding: $CARLA_PYTHON_BINDING"
echo "Requested/selected GPUs: $REQUESTED_GPU_IDS / $SELECTED_GPU_IDS"
echo "Automatic resume: enabled"
echo "Output: $LEAD_ROOT/outputs/evaluation/$RUN_ID/seed0"

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
EVALUATION_DATASET=fail2drive \
ROUTE_SOURCE="$ROUTE_SOURCE" \
ROUTE_IDS="$ROUTE_IDS" \
EVALUATION_OUTPUT_DIR="$LEAD_ROOT/outputs/evaluation/$RUN_ID/seed0" \
EXPERIMENT_RUN_ID="${RUN_ID}_seed0" \
EXPERIMENT_SEED=0 \
CARLA_PREFLIGHT_MAX_HEALTHY="${LEAD_STAGE3_DS87_EPOCH9_F2D_PREFLIGHT_MAX_HEALTHY:-0}" \
LOCAL_GPUS_PER_NODE=8 \
LEAD_LOCAL_MAX_PARALLEL_JOBS="$MAX_PARALLEL_JOBS" \
LEAD_LOCAL_GPU_IDS="$SELECTED_GPU_IDS" \
LEAD_FORCE_LOCAL=1 \
RUN_EVAL="$RUN_EVAL" \
USE_B2D_NONROOT_CARLA="$CARLA_NONROOT_MODE" \
CLEAN_EXISTING_CARLA=0 \
CARLA_SEQUENTIAL_PREFLIGHT=1 \
CARLA_SEQUENTIAL_STARTUP=1 \
CARLA_PREFLIGHT_TIMEOUT=90 \
CARLA_STARTUP_SLEEP=60 \
LEAD_RESUME_REQUIRE_BEV_CAMERA="${LEAD_STAGE3_DS87_EPOCH9_F2D_REQUIRE_BEV_CAMERA:-0}" \
LEAD_RESUME_REQUIRE_LIDAR="${LEAD_STAGE3_DS87_EPOCH9_F2D_REQUIRE_LIDAR:-0}" \
ROUTE_STARTUP_STAGGER=65 \
LEADERBOARD_TIMEOUT=600 \
RECORD_PATH=None \
PORT_BASE="${LEAD_STAGE3_DS87_EPOCH9_F2D_PORT_BASE:-35000}" \
TM_PORT_BASE="${LEAD_STAGE3_DS87_EPOCH9_F2D_TM_PORT_BASE:-45000}" \
STREAMING_PORT_BASE="${LEAD_STAGE3_DS87_EPOCH9_F2D_STREAMING_PORT_BASE:-55000}" \
PORT_STRIDE=10 \
LEAD_RESUME_SKIP_FINISHED=1 \
LEAD_RESUME_REQUIRE_COMFORT_METRICS="${LEAD_STAGE3_DS87_EPOCH9_F2D_REQUIRE_COMFORT_METRICS:-0}" \
LEAD_RESUME_REQUIRE_VISUALIZATION="${LEAD_STAGE3_DS87_EPOCH9_F2D_REQUIRE_VISUALIZATION:-0}" \
LEAD_RESUME_REQUIRE_CAMERA="${LEAD_STAGE3_DS87_EPOCH9_F2D_REQUIRE_CAMERA:-0}" \
LEAD_PRESERVE_ROUTE_XML_ID=1 \
UNIDRIVEVLA_LEAD_RECORD_LIDAR="${LEAD_STAGE3_DS87_EPOCH9_F2D_RECORD_LIDAR:-0}" \
UNIDRIVEVLA_LEAD_LIDAR_FRAME_FREQUENCY="${LEAD_STAGE3_DS87_EPOCH9_F2D_LIDAR_FRAME_FREQUENCY:-0}" \
UNIDRIVEVLA_LEAD_LIDAR_ROUTE_STEPS="${LEAD_STAGE3_DS87_EPOCH9_F2D_LIDAR_ROUTE_STEPS:-}" \
UNIDRIVEVLA_LEAD_RECORD_BEV_CAMERA="${LEAD_STAGE3_DS87_EPOCH9_F2D_RECORD_BEV_CAMERA:-0}" \
UNIDRIVEVLA_LEAD_BEV_FRAME_FREQUENCY="${LEAD_STAGE3_DS87_EPOCH9_F2D_BEV_FRAME_FREQUENCY:-0}" \
UNIDRIVEVLA_LEAD_BEV_ROUTE_STEPS="${LEAD_STAGE3_DS87_EPOCH9_F2D_BEV_ROUTE_STEPS:-}" \
SAVE_NAME=latest_4hz_stage3_ds87_epoch9_fail2drive_seed0 \
UNIDRIVEVLA_LEAD_ALLOW_CONFIG_MISMATCH=0 \
UNIDRIVEVLA_LEAD_ALLOW_PARTIAL_CHECKPOINT=0 \
UNIDRIVEVLA_LEAD_REMOVE_PROMPT_SPEED=0 \
UNIDRIVEVLA_LEAD_USE_KALMAN=1 \
UNIDRIVEVLA_LEAD_CREEPING=1 \
UNIDRIVEVLA_LEAD_CREEP_POINT_SOURCE="$CREEP_POINT_SOURCE" \
UNIDRIVEVLA_LEAD_STOP_SIGN=0 \
UNIDRIVEVLA_LEAD_RECORD_COMFORT="${LEAD_STAGE3_DS87_EPOCH9_F2D_RECORD_COMFORT:-0}" \
UNIDRIVEVLA_LEAD_RECORD_VISUALIZATION="${LEAD_STAGE3_DS87_EPOCH9_F2D_RECORD_VISUALIZATION:-0}" \
UNIDRIVEVLA_LEAD_RECORD_CAMERA="${LEAD_STAGE3_DS87_EPOCH9_F2D_RECORD_CAMERA:-0}" \
UNIDRIVEVLA_LEAD_CAMERA_FRAME_FREQUENCY="${LEAD_STAGE3_DS87_EPOCH9_F2D_CAMERA_FRAME_FREQUENCY:-5}" \
UNIDRIVEVLA_LEAD_METRIC_FLUSH_INTERVAL="${LEAD_STAGE3_DS87_EPOCH9_F2D_METRIC_FLUSH_INTERVAL:-200}" \
UNIDRIVEVLA_LEAD_VIS_DET_SCORE_THRESHOLD="${LEAD_STAGE3_DS87_EPOCH9_F2D_VIS_DET_SCORE_THRESHOLD:-0.3}" \
UNIDRIVEVLA_LEAD_VIS_MAP_SCORE_THRESHOLD="${LEAD_STAGE3_DS87_EPOCH9_F2D_VIS_MAP_SCORE_THRESHOLD:-0.3}" \
UNIDRIVEVLA_LEAD_VIS_ACTOR_RADIUS_M="${LEAD_STAGE3_DS87_EPOCH9_F2D_VIS_ACTOR_RADIUS_M:-65.0}" \
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
bash "$LEAD_ROOT/run_unidrivevla_lead_eval_8gpu.sh"
