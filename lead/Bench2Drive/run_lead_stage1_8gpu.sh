#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export LEAD_ZIP_ROOT="${LEAD_ZIP_ROOT:?Set LEAD_ZIP_ROOT to the external LEAD route/data archive}"
export LEAD_DATA_ROOT="${LEAD_DATA_ROOT:?Set LEAD_DATA_ROOT to the external prepared LEAD data root}"
export LEAD_INFO_DIR="${LEAD_INFO_DIR:-${LEAD_DATA_ROOT}/infos}"
export LEAD_TRAJECTORY_STATS_PATH="${LEAD_TRAJECTORY_STATS_PATH:-${LEAD_INFO_DIR}/lead_trajectory_stats.json}"
export LEAD_MAP_INFO_PATH="${LEAD_MAP_INFO_PATH:-${LEAD_DATA_ROOT}/infos/lead_official_map_infos.pkl}"

export VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:?Set VLM_PRETRAINED_PATH to the external VLM checkpoint}"
export VLM_VARIANT="${VLM_VARIANT:-2b}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_DIR}/zero_configs/adam_zero1_bf16_bs16_accum2_16gpu_b2dzoo.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-28651}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export LEAD_EXPECTED_INFOS_VERSION="${LEAD_EXPECTED_INFOS_VERSION:-14}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-14}"

export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
export LEAD_STAGE1_EPOCHS="${LEAD_STAGE1_EPOCHS:-3}"
export LEAD_STAGE1_LR="${LEAD_STAGE1_LR:-2e-4}"
export LEAD_SEQUENCES_SPLIT_NUM="${LEAD_SEQUENCES_SPLIT_NUM:-2}"
export LEAD_STAGE1_CKPT_INTERVAL="${LEAD_STAGE1_CKPT_INTERVAL:-}"
export LEAD_STAGE1_MAX_KEEP_CKPTS="${LEAD_STAGE1_MAX_KEEP_CKPTS:-1}"
export LEAD_INFO_FRAME_STRIDE="${LEAD_INFO_FRAME_STRIDE:-4}"
export LEAD_INFO_FRAME_OFFSET="${LEAD_INFO_FRAME_OFFSET:-0}"
export LEAD_INFO_SHARD_SIZE="${LEAD_INFO_SHARD_SIZE:-0}"
export LEAD_INFO_SHARD_CACHE_SIZE="${LEAD_INFO_SHARD_CACHE_SIZE:-32}"
export LEAD_INFO_NUM_WORKERS="${LEAD_INFO_NUM_WORKERS:-16}"
export ENABLE_VALIDATE="${ENABLE_VALIDATE:-0}"
if [[ -z "${LEAD_VAL_RATIO:-}" ]]; then
    if [[ "${ENABLE_VALIDATE}" == "1" ]]; then
        export LEAD_VAL_RATIO="0.02"
    else
        export LEAD_VAL_RATIO="0"
    fi
fi
export LEAD_SPLIT_STRATEGY="${LEAD_SPLIT_STRATEGY:-hash}"
export LEAD_ROUTE_QUALITY_FILTER="${LEAD_ROUTE_QUALITY_FILTER:-tfv6_posttrain}"
export LEAD_ROUTE_SKIP_FIRST="${LEAD_ROUTE_SKIP_FIRST:-8}"
export LEAD_ROUTE_SKIP_LAST="${LEAD_ROUTE_SKIP_LAST:-8}"
export LEAD_EGO_PLAN_HORIZON="${LEAD_EGO_PLAN_HORIZON:-8}"
export LEAD_EGO_PLAN_STEP="${LEAD_EGO_PLAN_STEP:-5}"
export LEAD_SAMPLE_STRIDE="${LEAD_SAMPLE_STRIDE:-1}"
export LEAD_SAMPLE_OFFSET="${LEAD_SAMPLE_OFFSET:-0}"
export LEAD_NUM_TRAIN_SAMPLES="${LEAD_NUM_TRAIN_SAMPLES:-}"
export LEAD_AUX_ROUTE_LOSS_WEIGHT="${LEAD_AUX_ROUTE_LOSS_WEIGHT:-1.0}"
export LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT="${LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT:-1.0}"
export LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT="${LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT:-1.0}"
export LEAD_PLANNING_QUERY_LAYOUT="${LEAD_PLANNING_QUERY_LAYOUT:-action_expert}"
export LEAD_STAGE1_LOAD_FROM="${LEAD_STAGE1_LOAD_FROM:-}"
export LEAD_STAGE1_RESUME_FROM="${LEAD_STAGE1_RESUME_FROM:-}"
export LEAD_SENSOR_PERTURBATION="${LEAD_SENSOR_PERTURBATION:-1}"
export LEAD_SENSOR_PERTURBATION_PROB="${LEAD_SENSOR_PERTURBATION_PROB:-0.5}"
export LEAD_PREFLIGHT_ONLY="${LEAD_PREFLIGHT_ONLY:-0}"
export LEAD_PREPROCESS_ONLY="${LEAD_PREPROCESS_ONLY:-0}"
export LEAD_REQUIRE_PREGENERATED_INFOS="${LEAD_REQUIRE_PREGENERATED_INFOS:-0}"

if [[ -n "${LEAD_STAGE1_LOAD_FROM}" && -n "${LEAD_STAGE1_RESUME_FROM}" ]]; then
    echo "Set only one of LEAD_STAGE1_LOAD_FROM and LEAD_STAGE1_RESUME_FROM." >&2
    exit 1
fi

if [[ "${LEAD_EGO_PLAN_HORIZON}" != "8" || "${LEAD_EGO_PLAN_STEP}" != "5" ]]; then
    echo "This LEAD config requires 4Hz/8-step/2s labels: LEAD_EGO_PLAN_HORIZON=8 and LEAD_EGO_PLAN_STEP=5." >&2
    exit 1
fi
if [[ ("${LEAD_INFO_FRAME_STRIDE}" != "1" && "${LEAD_INFO_FRAME_STRIDE}" != "4") ||
      "${LEAD_INFO_FRAME_OFFSET}" != "0" ||
      "${LEAD_ROUTE_QUALITY_FILTER}" != "tfv6_posttrain" ||
      "${LEAD_ROUTE_SKIP_FIRST}" != "8" || "${LEAD_ROUTE_SKIP_LAST}" != "8" ]]; then
    echo "LEAD requires frame stride 1 or 4, offset=0, tfv6_posttrain filtering, and first/last skip=8." >&2
    exit 1
fi
if [[ "${LEAD_SAMPLE_STRIDE}" != "1" || "${LEAD_SAMPLE_OFFSET}" != "0" ]]; then
    echo "Version-14 infos are sampled during generation; LEAD_SAMPLE_STRIDE must be 1 and LEAD_SAMPLE_OFFSET must be 0." >&2
    exit 1
fi
if ! [[ "${LEAD_INFO_SHARD_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "LEAD_INFO_SHARD_SIZE must be a non-negative integer." >&2
    exit 1
fi
if ! [[ "${LEAD_INFO_SHARD_CACHE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LEAD_INFO_SHARD_CACHE_SIZE must be a positive integer." >&2
    exit 1
fi
if ! [[ "${LEAD_INFO_NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LEAD_INFO_NUM_WORKERS must be a positive integer." >&2
    exit 1
fi
if (( LEAD_INFO_SHARD_SIZE > 0 )) && [[ "${LEAD_VAL_RATIO}" != "0" && "${LEAD_VAL_RATIO}" != "0.0" ]]; then
    echo "Sharded LEAD infos require LEAD_VAL_RATIO=0." >&2
    exit 1
fi
if [[ "${LEAD_PLANNING_QUERY_LAYOUT}" != "action_expert" &&
      "${LEAD_PLANNING_QUERY_LAYOUT}" != "action_expert_hipad" ]]; then
    echo "LEAD_PLANNING_QUERY_LAYOUT must be action_expert or action_expert_hipad." >&2
    exit 1
fi
if [[ "${LEAD_SENSOR_PERTURBATION}" != "0" && "${LEAD_SENSOR_PERTURBATION}" != "1" ]]; then
    echo "LEAD_SENSOR_PERTURBATION must be 0 or 1." >&2
    exit 1
fi
if [[ "${LEAD_PREFLIGHT_ONLY}" != "0" && "${LEAD_PREFLIGHT_ONLY}" != "1" ]]; then
    echo "LEAD_PREFLIGHT_ONLY must be 0 or 1." >&2
    exit 1
fi
if [[ "${LEAD_PREPROCESS_ONLY}" != "0" && "${LEAD_PREPROCESS_ONLY}" != "1" ]]; then
    echo "LEAD_PREPROCESS_ONLY must be 0 or 1." >&2
    exit 1
fi
if [[ "${LEAD_REQUIRE_PREGENERATED_INFOS}" != "0" &&
      "${LEAD_REQUIRE_PREGENERATED_INFOS}" != "1" ]]; then
    echo "LEAD_REQUIRE_PREGENERATED_INFOS must be 0 or 1." >&2
    exit 1
fi
if [[ "${LEAD_PREFLIGHT_ONLY}" == "1" && "${LEAD_PREPROCESS_ONLY}" == "1" ]]; then
    echo "Set at most one of LEAD_PREFLIGHT_ONLY and LEAD_PREPROCESS_ONLY." >&2
    exit 1
fi
if [[ "${LEAD_SENSOR_PERTURBATION}" == "1" ]]; then
    export LEAD_EFFECTIVE_SENSOR_PERTURBATION_PROB="${LEAD_SENSOR_PERTURBATION_PROB}"
else
    export LEAD_EFFECTIVE_SENSOR_PERTURBATION_PROB="0.0"
fi

if [[ "${PYTHON_BIN}" == */* ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
        exit 1
    fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "PYTHON_BIN was not found on PATH: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ -n "${LOCAL_TRANSFORMERS_SRC:-}" &&
      ! -f "${LOCAL_TRANSFORMERS_SRC}/transformers/__init__.py" ]]; then
    echo "LOCAL_TRANSFORMERS_SRC is incomplete: ${LOCAL_TRANSFORMERS_SRC}" >&2
    exit 1
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ &&
        "${MLP_WORKER_NUM}" =~ ^[1-9][0-9]*$ &&
        "${MLP_WORKER_GPU}" =~ ^[1-9][0-9]*$ &&
        "${MLP_ROLE_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "NUM_GPUS, MLP_WORKER_NUM, MLP_WORKER_GPU, and MLP_ROLE_INDEX must be integers." >&2
    exit 1
fi
if (( MLP_WORKER_NUM * MLP_WORKER_GPU != NUM_GPUS )); then
    echo "Distributed layout mismatch: ${MLP_WORKER_NUM} nodes x ${MLP_WORKER_GPU} GPUs != NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi
if (( MLP_ROLE_INDEX >= MLP_WORKER_NUM )); then
    echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX} is outside [0, $((MLP_WORKER_NUM - 1))]." >&2
    exit 1
fi
if (( MLP_WORKER_NUM > 1 )) &&
   [[ "${MLP_WORKER_0_HOST}" == "127.0.0.1" ||
      "${MLP_WORKER_0_HOST}" == "localhost" ||
      "${MLP_WORKER_0_HOST}" == "::1" ]]; then
    echo "Multi-node training requires a node-0 address reachable by every node; got MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST}." >&2
    exit 1
fi

QWEN3_TRANSFORMERS_REPLACE="${QWEN3_TRANSFORMERS_REPLACE:-${PROJECT_DIR}/../../qwenvl3/transformers_replace/models}"
EARLY_REQUIRED_FILES=(
    "${LEAD_MAP_INFO_PATH}"
    "${PROJECT_DIR}/data/kmeans/lead_map_100.npy"
    "${PROJECT_DIR}/data/kmeans/lead_ego_anchor.npy"
    "${DEEPSPEED_CONFIG}"
)
for path in "${EARLY_REQUIRED_FILES[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required LEAD Stage1 file before preprocessing: ${path}" >&2
        exit 1
    fi
done
if [[ ! -d "${LEAD_ZIP_ROOT}" ]]; then
    echo "Missing LEAD ZIP root: ${LEAD_ZIP_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${QWEN3_TRANSFORMERS_REPLACE}/qwen3_vl" ]]; then
    echo "Missing Qwen3-VL transformers replacement: ${QWEN3_TRANSFORMERS_REPLACE}/qwen3_vl" >&2
    exit 1
fi
if [[ ! -f "${VLM_PRETRAINED_PATH}/model.safetensors" &&
      ! -f "${VLM_PRETRAINED_PATH}/model.safetensors.index.json" &&
      ! -f "${VLM_PRETRAINED_PATH}/pytorch_model.bin.index.json" ]]; then
    echo "Missing HF-format VLM checkpoint under ${VLM_PRETRAINED_PATH}" >&2
    exit 1
fi
"${PYTHON_BIN}" - <<'PY'
import transformers
import deepspeed
import mmcv
import torch

print(
    "Training environment imports OK: "
    f"torch={torch.__version__}, mmcv={mmcv.__version__}, "
    f"transformers={transformers.__version__}, deepspeed={deepspeed.__version__}"
)
PY

if [[ "${LEAD_PREFLIGHT_ONLY}" == "1" ]]; then
    echo "LEAD static preflight passed; skipping infos/anchor generation and torchrun."
    exit 0
fi


CONFIG="${LEAD_STAGE1_CONFIG:-projects/configs/unidrivevla_lead_stage1_unified_2b.py}"
WORK_DIR="${WORK_DIR:-work_dirs2/unidrivevla_lead_stage1_unified_2b_8gpu_actionqueries19_sensorperturb50}"
LOG_FILE="${LOG_FILE:-work_dirs2/train-lead-stage1-8gpu-actionqueries19-sensorperturb50-$(date +%Y%m%d-%H%M%S).log}"

NEED_GENERATE_INFOS=0
if [[ ! -s "${LEAD_INFO_DIR}/lead_infos_train.pkl" ||
      ! -s "${LEAD_INFO_DIR}/lead_infos_val.pkl" ||
      ! -s "${LEAD_TRAJECTORY_STATS_PATH}" ]]; then
    NEED_GENERATE_INFOS=1
else
    if ! "${PYTHON_BIN}" - <<'PY'
import json
import math
import os
import pickle

with open(os.environ["LEAD_TRAJECTORY_STATS_PATH"], "r", encoding="utf-8") as f:
    stats = json.load(f)
infos_index = None
if int(os.environ["LEAD_INFO_SHARD_SIZE"]) > 0:
    with open(os.path.join(os.environ["LEAD_INFO_DIR"], "lead_infos_train.pkl"), "rb") as f:
        infos_index = pickle.load(f)

sensor_perturbation = os.environ["LEAD_SENSOR_PERTURBATION"] == "1"
configured_probability = float(os.environ["LEAD_SENSOR_PERTURBATION_PROB"])
if not 0.0 <= configured_probability <= 1.0:
    raise SystemExit(1)
effective_probability = configured_probability if sensor_perturbation else 0.0

frame_stride = int(os.environ["LEAD_INFO_FRAME_STRIDE"])
frame_offset = int(os.environ["LEAD_INFO_FRAME_OFFSET"])
planning_horizon = int(os.environ["LEAD_EGO_PLAN_HORIZON"])
planning_step = int(os.environ["LEAD_EGO_PLAN_STEP"])
planning_hz = 20.0 / planning_step
planning_seconds = planning_horizon * planning_step / 20.0
num_train_samples = int(stats.get("num_train_samples", 0))

expected_stats = {
    "stats_version": 7,
    "lead_stage1_infos_version": int(os.environ["LEAD_EXPECTED_INFOS_VERSION"]),
    "sensor_perturbation_version": 1,
    "sensor_perturbation_probability": effective_probability,
    "detection_filter_version": 4,
    "detection_class_mapping_version": 2,
    "detection_z_range": [-4.0, 4.0],
    "detection_visibility_filter": "lead_b2d_visibility_v1",
    "detection_traffic_control_filter": "tfv6_affects_ego_non_green_v1",
    "detection_box_dimension_order": "length_width_height",
    "detection_box_center": "carla_bbox_center",
    "bbox_center_offset_version": 1,
    "num_sensor_perturbation_samples": (
        num_train_samples if effective_probability > 0.0 else 0
    ),
    "perturbed_image_layout": "rgb_perturbated_strip",
    "frame_sample_stride": frame_stride,
    "frame_sample_offset": frame_offset,
    "route_quality_filter": os.environ["LEAD_ROUTE_QUALITY_FILTER"],
    "route_skip_first": int(os.environ["LEAD_ROUTE_SKIP_FIRST"]),
    "route_skip_last": int(os.environ["LEAD_ROUTE_SKIP_LAST"]),
    "source_sim_hz": 20,
    "data_save_freq": 5,
    "jpeg_storage_quality": 30,
    "perception_sample_hz": 4.0 / frame_stride,
    "planning_step_sim_frames": planning_step,
    "planning_freq_hz": planning_hz,
    "planning_horizon_steps": planning_horizon,
    "planning_horizon_seconds": planning_seconds,
    "val_ratio": float(os.environ["LEAD_VAL_RATIO"]),
    "split_strategy": os.environ["LEAD_SPLIT_STRATEGY"],
}
ok = num_train_samples > 0
ok = ok and all(stats.get(key) == value for key, value in expected_stats.items())
shard_size = int(os.environ["LEAD_INFO_SHARD_SIZE"])
if shard_size > 0:
    ok = ok and isinstance(infos_index, dict)
    ok = ok and infos_index.get("lead_sharded_infos_version") == 1
    ok = ok and infos_index.get("num_samples") == num_train_samples
    ok = ok and stats.get("infos_storage") == "sharded_pickle_v1"
    ok = ok and stats.get("infos_shard_size") == shard_size
    ok = ok and all(
        os.path.isfile(os.path.join(os.environ["LEAD_INFO_DIR"], shard["file"]))
        for shard in infos_index.get("shards", [])
    )
delta_mu = stats.get("delta_mu", [])
delta_std = stats.get("delta_std", [])
ok = (
    ok
    and len(delta_mu) == 2
    and len(delta_std) == 2
    and all(math.isfinite(value) for value in delta_mu + delta_std)
    and all(value > 0.0 for value in delta_std)
)
raise SystemExit(0 if ok else 1)
PY
    then
        NEED_GENERATE_INFOS=1
    fi
fi

lead_infos_ready() {
    [[ -s "${LEAD_INFO_DIR}/lead_infos_train.pkl" &&
       -s "${LEAD_INFO_DIR}/lead_infos_val.pkl" &&
       -s "${LEAD_TRAJECTORY_STATS_PATH}" ]] || return 1
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import pickle

with open(os.environ["LEAD_TRAJECTORY_STATS_PATH"], "r", encoding="utf-8") as f:
    stats = json.load(f)
infos_index = None
if int(os.environ["LEAD_INFO_SHARD_SIZE"]) > 0:
    with open(os.path.join(os.environ["LEAD_INFO_DIR"], "lead_infos_train.pkl"), "rb") as f:
        infos_index = pickle.load(f)

ok = (
    stats.get("stats_version") == 7
    and stats.get("lead_stage1_infos_version") == int(os.environ["LEAD_EXPECTED_INFOS_VERSION"])
    and stats.get("detection_filter_version") == 4
    and stats.get("detection_traffic_control_filter") == "tfv6_affects_ego_non_green_v1"
    and stats.get("detection_box_dimension_order") == "length_width_height"
    and stats.get("detection_box_center") == "carla_bbox_center"
    and stats.get("bbox_center_offset_version") == 1
    and stats.get("frame_sample_stride") == int(os.environ["LEAD_INFO_FRAME_STRIDE"])
    and stats.get("frame_sample_offset") == int(os.environ["LEAD_INFO_FRAME_OFFSET"])
    and stats.get("route_quality_filter") == os.environ["LEAD_ROUTE_QUALITY_FILTER"]
    and stats.get("route_skip_first") == int(os.environ["LEAD_ROUTE_SKIP_FIRST"])
    and stats.get("route_skip_last") == int(os.environ["LEAD_ROUTE_SKIP_LAST"])
    and stats.get("planning_horizon_steps") == int(os.environ["LEAD_EGO_PLAN_HORIZON"])
    and stats.get("planning_step_sim_frames") == int(os.environ["LEAD_EGO_PLAN_STEP"])
    and stats.get("sensor_perturbation_probability")
        == float(os.environ["LEAD_EFFECTIVE_SENSOR_PERTURBATION_PROB"])
    and int(stats.get("num_train_samples", 0)) > 0
)
if int(os.environ["LEAD_INFO_SHARD_SIZE"]) > 0:
    ok = (
        ok
        and isinstance(infos_index, dict)
        and infos_index.get("lead_sharded_infos_version") == 1
        and infos_index.get("num_samples") == int(stats["num_train_samples"])
        and stats.get("infos_storage") == "sharded_pickle_v1"
        and all(
            os.path.isfile(os.path.join(os.environ["LEAD_INFO_DIR"], shard["file"]))
            for shard in infos_index.get("shards", [])
        )
    )
raise SystemExit(0 if ok else 1)
PY
}

if [[ "${NEED_GENERATE_INFOS}" == "1" ]]; then
    if [[ "${LEAD_REQUIRE_PREGENERATED_INFOS}" == "1" ]]; then
        echo "LEAD infos are missing or incompatible under ${LEAD_INFO_DIR}; run the dedicated preparation script first." >&2
        exit 1
    fi
    if [[ "${MLP_ROLE_INDEX}" == "0" ]]; then
        echo "Generating version-${LEAD_EXPECTED_INFOS_VERSION} LEAD Stage1 infos under ${LEAD_INFO_DIR} on rank 0."
        INFO_ARGS=(
            --lead-zip-root "${LEAD_ZIP_ROOT}"
            --image-root "${LEAD_DATA_ROOT}"
            --out-dir "${LEAD_INFO_DIR}"
            --frame-stride "${LEAD_INFO_FRAME_STRIDE}"
            --frame-offset "${LEAD_INFO_FRAME_OFFSET}"
            --val-ratio "${LEAD_VAL_RATIO}"
            --split-strategy "${LEAD_SPLIT_STRATEGY}"
            --route-quality-filter "${LEAD_ROUTE_QUALITY_FILTER}"
            --route-skip-first "${LEAD_ROUTE_SKIP_FIRST}"
            --route-skip-last "${LEAD_ROUTE_SKIP_LAST}"
            --sensor-perturbation-probability "${LEAD_EFFECTIVE_SENSOR_PERTURBATION_PROB}"
            --ego-plan-horizon "${LEAD_EGO_PLAN_HORIZON}"
            --ego-plan-step "${LEAD_EGO_PLAN_STEP}"
            --num-workers "${LEAD_INFO_NUM_WORKERS}"
        )
        if [[ -n "${LEAD_MAX_SAMPLES:-}" ]]; then
            INFO_ARGS+=(--max-samples "${LEAD_MAX_SAMPLES}")
        fi
        if (( LEAD_INFO_SHARD_SIZE > 0 )); then
            INFO_ARGS+=(--shard-size "${LEAD_INFO_SHARD_SIZE}")
        fi
        "${PYTHON_BIN}" tools/generate_lead_stage1_infos.py "${INFO_ARGS[@]}"
        if ! lead_infos_ready; then
            echo "Rank 0 generated LEAD infos, but version-${LEAD_EXPECTED_INFOS_VERSION} validation failed." >&2
            exit 1
        fi
    else
        echo "Rank ${MLP_ROLE_INDEX} waiting for rank 0 to generate version-${LEAD_EXPECTED_INFOS_VERSION} LEAD infos."
        deadline=$(( $(date +%s) + ${LEAD_PREPROCESS_WAIT_TIMEOUT:-7200} ))
        until lead_infos_ready; do
            if (( $(date +%s) >= deadline )); then
                echo "Timed out waiting for version-${LEAD_EXPECTED_INFOS_VERSION} LEAD infos." >&2
                exit 1
            fi
            sleep 15
        done
    fi
fi

if [[ -z "${LEAD_NUM_TRAIN_SAMPLES}" ]]; then
    export LEAD_NUM_TRAIN_SAMPLES=$(
        "${PYTHON_BIN}" - <<'PY'
import json
import os
with open(os.environ["LEAD_TRAJECTORY_STATS_PATH"], "r", encoding="utf-8") as f:
    print(int(json.load(f)["num_train_samples"]))
PY
    )
fi

"${PYTHON_BIN}" tools/check_lead_info_continuity.py \
    --manifest "${LEAD_INFO_DIR}/lead_infos_train.pkl" \
    --expected-stride "${LEAD_INFO_FRAME_STRIDE}" \
    --expected-offset "${LEAD_INFO_FRAME_OFFSET}" \
    --sequences-split-num "${LEAD_SEQUENCES_SPLIT_NUM}" \
    --min-streaming-groups "$((NUM_GPUS * SAMPLES_PER_GPU))"

if [[ -z "${LEAD_STAGE1_CKPT_INTERVAL}" ]]; then
    GLOBAL_MICRO_BATCH=$(( NUM_GPUS * SAMPLES_PER_GPU ))
    export LEAD_STAGE1_CKPT_INTERVAL=$((
        (LEAD_NUM_TRAIN_SAMPLES + GLOBAL_MICRO_BATCH - 1) / GLOBAL_MICRO_BATCH ))
fi

LEAD_DET_ANCHOR="${PROJECT_DIR}/data/kmeans/lead_det_900.npy"
LEAD_DET_ANCHOR_MARKER="${LEAD_DET_ANCHOR}.sparse4d_lateral_forward.ok"

lead_det_anchor_v14_ready() {
    [[ -s "${LEAD_DET_ANCHOR}" && -s "${LEAD_DET_ANCHOR_MARKER}" ]] &&
    grep -Eq '^perception_coord_system=sparse4d_lateral_forward$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^detection_z_range=-4[.]0,4[.]0$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^lead_stage1_infos_version=14$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^detection_filter_version=4$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^detection_traffic_control_filter=tfv6_affects_ego_non_green_v1$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^detection_class_mapping_version=2$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^detection_box_center=carla_bbox_center$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^bbox_center_offset_version=1$' "${LEAD_DET_ANCHOR_MARKER}" &&
    grep -Eq '^detection_box_dimension_order=length_width_height$' "${LEAD_DET_ANCHOR_MARKER}"
}

NEED_GENERATE_DET_ANCHOR=0
if [[ "${LEAD_REGENERATE_DET_ANCHOR:-0}" == "1" ]] || ! lead_det_anchor_v14_ready; then
    echo "LEAD det anchor is missing or stale for the version-14 schema."
    NEED_GENERATE_DET_ANCHOR=1
fi
if [[ "${NEED_GENERATE_DET_ANCHOR}" == "1" ]]; then
    if [[ "${MLP_ROLE_INDEX}" == "0" ]]; then
        echo "Generating LEAD Sparse4D det anchors on rank 0: ${LEAD_DET_ANCHOR}"
        DET_ANCHOR_ARGS=(
            --lead-zip-root "${LEAD_ZIP_ROOT}"
            --image-root "${LEAD_DATA_ROOT}"
            --out "${LEAD_DET_ANCHOR}"
            --num-clusters 900
            --distance-threshold 55
            --z-min -4.0
            --z-max 4.0
            --sample-stride "${LEAD_DET_ANCHOR_SAMPLE_STRIDE:-1}"
            --frame-stride "${LEAD_INFO_FRAME_STRIDE}"
            --frame-offset "${LEAD_INFO_FRAME_OFFSET}"
            --route-quality-filter "${LEAD_ROUTE_QUALITY_FILTER}"
            --route-skip-first "${LEAD_ROUTE_SKIP_FIRST}"
            --route-skip-last "${LEAD_ROUTE_SKIP_LAST}"
            --sensor-perturbation-probability "${LEAD_EFFECTIVE_SENSOR_PERTURBATION_PROB}"
            --ego-plan-horizon "${LEAD_EGO_PLAN_HORIZON}"
            --ego-plan-step "${LEAD_EGO_PLAN_STEP}"
            --num-workers "${LEAD_DET_ANCHOR_WORKERS:-16}"
        )
        "${PYTHON_BIN}" tools/kmeans/kmeans_lead_det.py "${DET_ANCHOR_ARGS[@]}"
        if ! lead_det_anchor_v14_ready; then
            echo "Rank 0 generated the detection anchor, but its v14 marker is invalid." >&2
            exit 1
        fi
    else
        echo "Rank ${MLP_ROLE_INDEX} waiting for rank 0 to generate the v14 detection anchor."
        deadline=$(( $(date +%s) + ${LEAD_PREPROCESS_WAIT_TIMEOUT:-7200} ))
        until lead_det_anchor_v14_ready; do
            if (( $(date +%s) >= deadline )); then
                echo "Timed out waiting for the v14 LEAD detection anchor." >&2
                exit 1
            fi
            sleep 15
        done
    fi
fi

REQUIRED_FILES=(
    "${LEAD_INFO_DIR}/lead_infos_train.pkl"
    "${LEAD_INFO_DIR}/lead_infos_val.pkl"
    "${LEAD_TRAJECTORY_STATS_PATH}"
    "${LEAD_MAP_INFO_PATH}"
    "${LEAD_DET_ANCHOR}"
    "${LEAD_DET_ANCHOR_MARKER}"
    "${PROJECT_DIR}/data/kmeans/lead_map_100.npy"
    "${PROJECT_DIR}/data/kmeans/lead_ego_anchor.npy"
)
for path in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required LEAD Stage1 file: ${path}" >&2
        exit 1
    fi
done

if [[ "${LEAD_PREPROCESS_ONLY}" == "1" ]]; then
    echo "LEAD preprocessing completed and validated; skipping Qwen replacement and torchrun."
    exit 0
fi

if [[ ! -f "${VLM_PRETRAINED_PATH}/model.safetensors" && ! -f "${VLM_PRETRAINED_PATH}/model.safetensors.index.json" && ! -f "${VLM_PRETRAINED_PATH}/pytorch_model.bin.index.json" ]]; then
    echo "Warning: VLM_PRETRAINED_PATH may not be an HF-format checkpoint: ${VLM_PRETRAINED_PATH}" >&2
fi

echo "Starting LEAD Stage1 training"
echo "CONFIG=${CONFIG}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "LEAD_ZIP_ROOT=${LEAD_ZIP_ROOT}"
echo "LEAD_INFO_DIR=${LEAD_INFO_DIR}"
echo "LEAD_TRAJECTORY_STATS_PATH=${LEAD_TRAJECTORY_STATS_PATH}"
echo "LEAD_MAP_INFO_PATH=${LEAD_MAP_INFO_PATH}"
echo "VLM_PRETRAINED_PATH=${VLM_PRETRAINED_PATH}"
echo "VLM_VARIANT=${VLM_VARIANT}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "MLP_WORKER_NUM=${MLP_WORKER_NUM}"
echo "MLP_WORKER_GPU=${MLP_WORKER_GPU}"
echo "SAMPLES_PER_GPU=${SAMPLES_PER_GPU}"
echo "WORKERS_PER_GPU=${WORKERS_PER_GPU}"
echo "LEAD_STAGE1_EPOCHS=${LEAD_STAGE1_EPOCHS}"
echo "LEAD_STAGE1_LR=${LEAD_STAGE1_LR}"
echo "LEAD_SEQUENCES_SPLIT_NUM=${LEAD_SEQUENCES_SPLIT_NUM}"
echo "LEAD_INFO_FRAME_STRIDE=${LEAD_INFO_FRAME_STRIDE}"
echo "LEAD_INFO_FRAME_OFFSET=${LEAD_INFO_FRAME_OFFSET}"
echo "LEAD_INFO_SHARD_SIZE=${LEAD_INFO_SHARD_SIZE}"
echo "LEAD_INFO_SHARD_CACHE_SIZE=${LEAD_INFO_SHARD_CACHE_SIZE}"
echo "LEAD_INFO_NUM_WORKERS=${LEAD_INFO_NUM_WORKERS}"
echo "LEAD_ROUTE_QUALITY_FILTER=${LEAD_ROUTE_QUALITY_FILTER}"
echo "LEAD_ROUTE_SKIP_FIRST=${LEAD_ROUTE_SKIP_FIRST}"
echo "LEAD_ROUTE_SKIP_LAST=${LEAD_ROUTE_SKIP_LAST}"
echo "LEAD_EGO_PLAN_HORIZON=${LEAD_EGO_PLAN_HORIZON}"
echo "LEAD_EGO_PLAN_STEP=${LEAD_EGO_PLAN_STEP}"
echo "LEAD_SAMPLE_STRIDE=${LEAD_SAMPLE_STRIDE}"
echo "LEAD_SAMPLE_OFFSET=${LEAD_SAMPLE_OFFSET}"
echo "LEAD_NUM_TRAIN_SAMPLES=${LEAD_NUM_TRAIN_SAMPLES}"
echo "LEAD_AUX_ROUTE_LOSS_WEIGHT=${LEAD_AUX_ROUTE_LOSS_WEIGHT}"
echo "LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT=${LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT}"
echo "LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT=${LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT}"
echo "LEAD_PLANNING_QUERY_LAYOUT=${LEAD_PLANNING_QUERY_LAYOUT}"
echo "LEAD_STAGE1_ATTN_IMPLEMENTATION=${LEAD_STAGE1_ATTN_IMPLEMENTATION:-eager}"
echo "LEAD_STAGE1_LOAD_FROM=${LEAD_STAGE1_LOAD_FROM:-<none>}"
echo "LEAD_STAGE1_RESUME_FROM=${LEAD_STAGE1_RESUME_FROM:-<none>}"
echo "WORK_DIR=${WORK_DIR}"
echo "LEAD_SENSOR_PERTURBATION=${LEAD_SENSOR_PERTURBATION}"
echo "LEAD_SENSOR_PERTURBATION_PROB=${LEAD_SENSOR_PERTURBATION_PROB}"
echo "Log: ${PROJECT_DIR}/${LOG_FILE}"

TRANSFORMERS_DIR=$("${PYTHON_BIN}" - <<'PY'
import os
import transformers
print(os.path.dirname(transformers.__file__))
PY
)
qwen3_replacement_ready() {
    "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import inspect
from transformers.models.qwen3_vl import modeling_qwen3_vl as m

src = inspect.getsource(m.Qwen3VLModel.forward)
assert "image_embeds, deepstack_image_embeds, _ = self.get_image_features" in src
PY
}
QWEN_REPLACEMENT_LOCK_DIR="${PROJECT_DIR}/work_dirs2/.locks"
mkdir -p "${QWEN_REPLACEMENT_LOCK_DIR}"
(
    flock -x 9
    if qwen3_replacement_ready; then
        echo "Qwen3-VL replacement already validated at ${TRANSFORMERS_DIR}/models"
    else
        echo "Installing Qwen3-VL transformers replacement into ${TRANSFORMERS_DIR}/models"
        cp -rf "${QWEN3_TRANSFORMERS_REPLACE}" "${TRANSFORMERS_DIR}/"
        if ! qwen3_replacement_ready; then
            echo "Qwen3-VL replacement check failed." >&2
            exit 1
        fi
        echo "Qwen3-VL replacement check passed"
    fi
) 9>"${QWEN_REPLACEMENT_LOCK_DIR}/qwen3_transformers_replace.lock"

TRAIN_EXTRA_ARGS=()
if [[ "${ENABLE_VALIDATE}" != "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--no-validate)
fi

CFG_OPTIONS=(
    "model.planning_head.vlm_variant=${VLM_VARIANT}"
)
if [[ -n "${LEAD_STAGE1_ATTN_IMPLEMENTATION:-}" ]]; then
    CFG_OPTIONS+=("model.planning_head.attn_implementation=${LEAD_STAGE1_ATTN_IMPLEMENTATION}")
fi

mkdir -p "${WORK_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

NPROC_PER_NODE="${MLP_WORKER_GPU:-${NUM_GPUS}}"
NNODES="${MLP_WORKER_NUM:-1}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-127.0.0.1}"
MASTER_PORT="${MLP_WORKER_0_PORT:-28651}"
TORCHRUN_RDZV_ARGS=()
if [[ -n "${LEAD_TORCHRUN_RDZV_TIMEOUT:-}" ]]; then
    if ! [[ "${LEAD_TORCHRUN_RDZV_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "LEAD_TORCHRUN_RDZV_TIMEOUT must be a positive integer number of seconds." >&2
        exit 1
    fi
    TORCHRUN_RDZV_ARGS+=(--rdzv_conf "timeout=${LEAD_TORCHRUN_RDZV_TIMEOUT}")
fi

echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "LEAD_TORCHRUN_RDZV_TIMEOUT=${LEAD_TORCHRUN_RDZV_TIMEOUT:-<torch-default>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TORCHRUN_RDZV_ARGS[@]}" \
    tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch \
    --cfg-options "${CFG_OPTIONS[@]}" \
    "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
