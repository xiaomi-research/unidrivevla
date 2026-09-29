#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
export LEAD_STAGE1_ENV="${LEAD_STAGE1_ENV:-b2dzoo}"
if [[ "${LEAD_STAGE1_ENV}" != "b2dzoo" ]]; then
    echo "The full-4Hz launcher currently requires LEAD_STAGE1_ENV=b2dzoo." >&2
    exit 1
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
export VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:?Set VLM_PRETRAINED_PATH to the external Qwen3-VL-2B-Instruct directory}"
export LOCAL_TRANSFORMERS_SRC="${LOCAL_TRANSFORMERS_SRC:-${PROJECT_DIR}/../third_party/transformers-4.57.1/src}"
export MMCV_ROOT="${MMCV_ROOT:-${PROJECT_DIR}/../third_party/mmcv-1.7.2}"
export MMDET3D_ROOT="${MMDET3D_ROOT:-${PROJECT_DIR}/../third_party/mmdetection3d-1.0.0rc6}"
export PYTHONPATH="${LOCAL_TRANSFORMERS_SRC}:${PROJECT_DIR}:${MMCV_ROOT}:${MMDET3D_ROOT}:${PYTHONPATH:-}"

export NUM_GPUS="${NUM_GPUS:-16}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-2}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-16}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_DIR}/zero_configs/adam_zero1_bf16_bs16_accum2_16gpu_b2dzoo.json}"

export LEAD_INFO_DIR="${LEAD_INFO_DIR:-${PROJECT_DIR}/data/lead/infos_4hz_full}"
export LEAD_TRAJECTORY_STATS_PATH="${LEAD_TRAJECTORY_STATS_PATH:-${LEAD_INFO_DIR}/lead_trajectory_stats.json}"
export LEAD_MAP_INFO_PATH="${LEAD_MAP_INFO_PATH:-${PROJECT_DIR}/data/lead/infos/lead_official_map_infos.pkl}"
export LEAD_INFO_FRAME_STRIDE=1
export LEAD_INFO_FRAME_OFFSET=0
export LEAD_INFO_SHARD_SIZE="${LEAD_INFO_SHARD_SIZE:-2048}"
export LEAD_INFO_SHARD_CACHE_SIZE="${LEAD_INFO_SHARD_CACHE_SIZE:-32}"
export LEAD_INFO_NUM_WORKERS="${LEAD_INFO_NUM_WORKERS:-32}"
export LEAD_VAL_RATIO=0
export ENABLE_VALIDATE=0
export LEAD_ROUTE_SKIP_FIRST=8
export LEAD_ROUTE_SKIP_LAST=8
export LEAD_SAMPLE_STRIDE=1
export LEAD_SAMPLE_OFFSET=0
export LEAD_PREPROCESS_WAIT_TIMEOUT="${LEAD_PREPROCESS_WAIT_TIMEOUT:-43200}"

export LEAD_STAGE1_EPOCHS="${LEAD_STAGE1_EPOCHS:-3}"
export LEAD_STAGE1_LR="${LEAD_STAGE1_LR:-2e-4}"
export LEAD_SEQUENCES_SPLIT_NUM="${LEAD_SEQUENCES_SPLIT_NUM:-2}"
export LEAD_STAGE1_MAX_KEEP_CKPTS="${LEAD_STAGE1_MAX_KEEP_CKPTS:-1}"
export LEAD_STAGE1_CKPT_INTERVAL="${LEAD_STAGE1_CKPT_INTERVAL:-}"
export LEAD_STAGE1_ATTN_IMPLEMENTATION="${LEAD_STAGE1_ATTN_IMPLEMENTATION:-eager}"
export LEAD_PLANNING_QUERY_LAYOUT="${LEAD_PLANNING_QUERY_LAYOUT:-action_expert}"
export LEAD_AUX_ROUTE_LOSS_WEIGHT="${LEAD_AUX_ROUTE_LOSS_WEIGHT:-1.0}"
export LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT="${LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT:-1.0}"
export LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT="${LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT:-1.0}"
export LEAD_STAGE1_LOAD_FROM="${LEAD_STAGE1_LOAD_FROM:-}"
export LEAD_STAGE1_RESUME_FROM="${LEAD_STAGE1_RESUME_FROM:-}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-28681}"
export LEAD_PREFLIGHT_ONLY="${LEAD_PREFLIGHT_ONLY:-0}"

export WORK_DIR="${WORK_DIR:-work_dirs2/unidrivevla_lead_stage1_unified_2b_2node16gpu_bs16_accum2_3ep_full4hz_actionqueries19_stableflow_posttrainfilter_sensorperturb50_b2dzoo_eager}"
export LOG_FILE="${LOG_FILE:-work_dirs2/train-lead-stage1-full4hz-16gpu-bs16-accum2-node${MLP_ROLE_INDEX:-0}-$(date +%Y%m%d-%H%M%S).log}"

if [[ "${NUM_GPUS}" != "16" || "${MLP_WORKER_NUM}" != "2" ||
      "${MLP_WORKER_GPU}" != "8" || "${SAMPLES_PER_GPU}" != "16" ]]; then
    echo "This launcher requires 16 GPUs as 2 nodes x 8 GPUs and SAMPLES_PER_GPU=16." >&2
    exit 1
fi
if [[ "${LEAD_STAGE1_EPOCHS}" != "3" || "${LEAD_STAGE1_LR}" != "2e-4" ||
      "${LEAD_SEQUENCES_SPLIT_NUM}" != "2" || "${LEAD_ROUTE_SKIP_FIRST}" != "8" ||
      "${LEAD_ROUTE_SKIP_LAST}" != "8" ]]; then
    echo "This launcher requires 3 epochs, LR=2e-4, route skip=8/8, and two streaming route splits." >&2
    exit 1
fi
if [[ "${LEAD_PREFLIGHT_ONLY}" != "0" && "${LEAD_PREFLIGHT_ONLY}" != "1" ]]; then
    echo "LEAD_PREFLIGHT_ONLY must be 0 or 1." >&2
    exit 1
fi

export LEAD_NUM_TRAIN_SAMPLES="${LEAD_NUM_TRAIN_SAMPLES:-$(
    "${PYTHON_BIN}" - "${LEAD_TRAJECTORY_STATS_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(int(json.load(f)["num_train_samples"]))
PY
)}"
LEAD_GRADIENT_ACCUMULATION_STEPS=2
LEAD_GLOBAL_MICRO_BATCH=$((NUM_GPUS * SAMPLES_PER_GPU))
LEAD_RAW_ITERS_PER_EPOCH=$((
    (LEAD_NUM_TRAIN_SAMPLES + LEAD_GLOBAL_MICRO_BATCH - 1) /
    LEAD_GLOBAL_MICRO_BATCH
))
LEAD_ALIGNED_ITERS_PER_EPOCH=$((
    (LEAD_RAW_ITERS_PER_EPOCH + LEAD_GRADIENT_ACCUMULATION_STEPS - 1) /
    LEAD_GRADIENT_ACCUMULATION_STEPS * LEAD_GRADIENT_ACCUMULATION_STEPS
))
export LEAD_STAGE1_CKPT_INTERVAL="${LEAD_STAGE1_CKPT_INTERVAL:-${LEAD_ALIGNED_ITERS_PER_EPOCH}}"
export LEAD_STAGE1_MAX_ITERS="${LEAD_STAGE1_MAX_ITERS:-$((LEAD_STAGE1_CKPT_INTERVAL * LEAD_STAGE1_EPOCHS))}"
if (( LEAD_STAGE1_CKPT_INTERVAL % LEAD_GRADIENT_ACCUMULATION_STEPS != 0 ||
      LEAD_STAGE1_MAX_ITERS % LEAD_GRADIENT_ACCUMULATION_STEPS != 0 )); then
    echo "Checkpoint and final iterations must align with gradient accumulation 2." >&2
    exit 1
fi
if (( LEAD_STAGE1_MAX_ITERS != LEAD_STAGE1_CKPT_INTERVAL * LEAD_STAGE1_EPOCHS )); then
    echo "Formal training requires exactly three aligned checkpoint intervals." >&2
    exit 1
fi

"${PYTHON_BIN}" - "${DEEPSPEED_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = json.load(f)
expected = {
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 2,
}
mismatches = {
    key: (config.get(key), value)
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"Invalid full-4Hz 16GPU DeepSpeed batch configuration: {mismatches}")
PY

echo "LEAD Stage1 full native-4Hz training: 3 epochs, LR=2e-4, 16 GPUs, micro=16, accum=2, global_batch=512"
echo "Streaming sampler: sequences_split_num=${LEAD_SEQUENCES_SPLIT_NUM}"
echo "INFO_STORAGE=sharded_pickle_v1 shard_size=${LEAD_INFO_SHARD_SIZE}"
echo "LEAD_INFO_NUM_WORKERS=${LEAD_INFO_NUM_WORKERS}"
echo "ITERATIONS: raw_per_epoch=${LEAD_RAW_ITERS_PER_EPOCH} aligned_per_epoch=${LEAD_STAGE1_CKPT_INTERVAL} max=${LEAD_STAGE1_MAX_ITERS}"

bash run_lead_stage1_8gpu.sh
