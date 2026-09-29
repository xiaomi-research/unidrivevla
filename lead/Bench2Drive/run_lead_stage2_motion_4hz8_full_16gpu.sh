#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export LOCAL_TRANSFORMERS_SRC="${LOCAL_TRANSFORMERS_SRC:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export NUM_GPUS="${NUM_GPUS:-16}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-2}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-16}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_DIR}/zero_configs/adam_zero1_bf16_bs16_accum2_16gpu_b2dzoo.json}"

export LEAD_INFO_DIR="${LEAD_INFO_DIR:-${PROJECT_DIR}/data/lead/infos_4hz_full_motion4hz8_v15}"
export LEAD_TRAJECTORY_STATS_PATH="${LEAD_TRAJECTORY_STATS_PATH:-${LEAD_INFO_DIR}/lead_trajectory_stats.json}"
export LEAD_MAP_INFO_PATH="${LEAD_MAP_INFO_PATH:-${PROJECT_DIR}/data/lead/infos/lead_official_map_infos.pkl}"
export LEAD_STAGE1_REFERENCE_INFO="${LEAD_STAGE1_REFERENCE_INFO:-${PROJECT_DIR}/data/lead/infos_4hz_full/lead_infos_train.pkl}"
export LEAD_STAGE1_REFERENCE_STATS="${LEAD_STAGE1_REFERENCE_STATS:-${PROJECT_DIR}/data/lead/infos_4hz_full/lead_trajectory_stats.json}"
export LEAD_EXPECTED_INFOS_VERSION=15
export LEAD_REQUIRE_PREGENERATED_INFOS=1
export LEAD_INFO_FRAME_STRIDE=1
export LEAD_INFO_FRAME_OFFSET=0
export LEAD_INFO_SHARD_SIZE="${LEAD_INFO_SHARD_SIZE:-2048}"
export LEAD_INFO_SHARD_CACHE_SIZE="${LEAD_INFO_SHARD_CACHE_SIZE:-32}"
export LEAD_VAL_RATIO=0
export ENABLE_VALIDATE=0
export LEAD_ROUTE_QUALITY_FILTER=tfv6_posttrain
export LEAD_ROUTE_SKIP_FIRST=8
export LEAD_ROUTE_SKIP_LAST=8
export LEAD_SAMPLE_STRIDE=1
export LEAD_SAMPLE_OFFSET=0

export LEAD_STAGE1_CONFIG="${LEAD_STAGE1_CONFIG:-projects/configs/unidrivevla_lead_stage2_motion_4hz8_2b.py}"
export LEAD_STAGE1_EPOCHS="${LEAD_STAGE1_EPOCHS:-3}"
export LEAD_STAGE1_LR="${LEAD_STAGE1_LR:-1e-4}"
export LEAD_MOTION_LR="${LEAD_MOTION_LR:-1e-4}"
export LEAD_SEQUENCES_SPLIT_NUM="${LEAD_SEQUENCES_SPLIT_NUM:-2}"
export LEAD_STAGE1_MAX_KEEP_CKPTS="${LEAD_STAGE1_MAX_KEEP_CKPTS:-1}"
export LEAD_STAGE1_CKPT_INTERVAL="${LEAD_STAGE1_CKPT_INTERVAL:-}"
export LEAD_STAGE1_ATTN_IMPLEMENTATION="${LEAD_STAGE1_ATTN_IMPLEMENTATION:-eager}"
export LEAD_PLANNING_QUERY_LAYOUT=action_expert
export LEAD_AUX_ROUTE_LOSS_WEIGHT="${LEAD_AUX_ROUTE_LOSS_WEIGHT:-1.0}"
export LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT="${LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT:-1.0}"
export LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT="${LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT:-1.0}"
export LEAD_STAGE1_LOAD_FROM=""
export LEAD_STAGE1_RESUME_FROM=""
if [[ "${LEAD_MOTION_FROM_QWEN_BASE:-0}" == "1" ]]; then
    export LEAD_MOTION_LOAD_FROM=""
    export LEAD_MOTION_RESUME_FROM=""
else
    export LEAD_MOTION_LOAD_FROM="${LEAD_MOTION_LOAD_FROM:-${LEAD_STAGE1_CHECKPOINT:?Set LEAD_STAGE1_CHECKPOINT to the external Stage1 checkpoint}}"
fi
export LEAD_MOTION_RESUME_FROM="${LEAD_MOTION_RESUME_FROM:-}"
export LEAD_MOTION_ANCHOR="${LEAD_MOTION_ANCHOR:-${PROJECT_DIR}/data/kmeans/lead_motion_6_4hz8step.npy}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-28701}"
export LEAD_PREFLIGHT_ONLY="${LEAD_PREFLIGHT_ONLY:-0}"

export WORK_DIR="${WORK_DIR:-work_dirs2/unidrivevla_lead_stage2_motion_4hz8_2b_2node16gpu_bs16_accum2_3ep_from_iter11358}"
export LOG_FILE="${LOG_FILE:-work_dirs2/train-lead-stage2-motion-4hz8-full-16gpu-bs16-accum2-node${MLP_ROLE_INDEX:-0}-$(date +%Y%m%d-%H%M%S).log}"

LEAD_EXPECTED_NUM_GPUS="${LEAD_EXPECTED_NUM_GPUS:-16}"
LEAD_EXPECTED_WORKER_NUM="${LEAD_EXPECTED_WORKER_NUM:-2}"
LEAD_EXPECTED_WORKER_GPU="${LEAD_EXPECTED_WORKER_GPU:-8}"
LEAD_EXPECTED_SAMPLES_PER_GPU="${LEAD_EXPECTED_SAMPLES_PER_GPU:-16}"
LEAD_GRADIENT_ACCUMULATION_STEPS="${LEAD_GRADIENT_ACCUMULATION_STEPS:-2}"
if [[ "${NUM_GPUS}" != "${LEAD_EXPECTED_NUM_GPUS}" ||
      "${MLP_WORKER_NUM}" != "${LEAD_EXPECTED_WORKER_NUM}" ||
      "${MLP_WORKER_GPU}" != "${LEAD_EXPECTED_WORKER_GPU}" ||
      "${SAMPLES_PER_GPU}" != "${LEAD_EXPECTED_SAMPLES_PER_GPU}" ]]; then
    echo "This launcher requires ${LEAD_EXPECTED_NUM_GPUS} GPUs as ${LEAD_EXPECTED_WORKER_NUM} nodes x ${LEAD_EXPECTED_WORKER_GPU} GPUs and SAMPLES_PER_GPU=${LEAD_EXPECTED_SAMPLES_PER_GPU}." >&2
    exit 1
fi
if ! [[ "${LEAD_STAGE1_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LEAD_STAGE1_EPOCHS must be a positive integer." >&2
    exit 1
fi
if [[ "${LEAD_STAGE1_LR}" != "1e-4" || "${LEAD_MOTION_LR}" != "1e-4" ||
      "${LEAD_SEQUENCES_SPLIT_NUM}" != "2" ]]; then
    echo "This launcher requires LR=1e-4 and two streaming route splits." >&2
    exit 1
fi

if [[ -n "${LEAD_MOTION_RESUME_FROM}" && ! -d "${LEAD_MOTION_RESUME_FROM}" ]]; then
    echo "LEAD_MOTION_RESUME_FROM is not a DeepSpeed checkpoint directory: ${LEAD_MOTION_RESUME_FROM}" >&2
    exit 1
fi

required_files=(
    "${LEAD_INFO_DIR}/lead_infos_train.pkl"
    "${LEAD_INFO_DIR}/lead_infos_val.pkl"
    "${LEAD_TRAJECTORY_STATS_PATH}"
    "${LEAD_STAGE1_REFERENCE_INFO}"
    "${LEAD_STAGE1_REFERENCE_STATS}"
    "${LEAD_MOTION_ANCHOR}"
    "${LEAD_MOTION_ANCHOR%.npy}.json"
    "${DEEPSPEED_CONFIG}"
)
if [[ -z "${LEAD_MOTION_RESUME_FROM}" && -n "${LEAD_MOTION_LOAD_FROM}" ]]; then
    required_files+=("${LEAD_MOTION_LOAD_FROM}")
fi
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required LEAD 4Hz/8-step motion file: ${required_file}" >&2
        echo "Run prepare_lead_motion_4hz8_data.sh before launching training." >&2
        exit 1
    fi
done

"${PYTHON_BIN}" tools/check_lead_info_identity.py \
    --reference-manifest "${LEAD_STAGE1_REFERENCE_INFO}" \
    --candidate-manifest "${LEAD_INFO_DIR}/lead_infos_train.pkl" \
    --reference-stats "${LEAD_STAGE1_REFERENCE_STATS}" \
    --candidate-stats "${LEAD_TRAJECTORY_STATS_PATH}"

export LEAD_NUM_TRAIN_SAMPLES="${LEAD_NUM_TRAIN_SAMPLES:-$(
    "${PYTHON_BIN}" - "${LEAD_TRAJECTORY_STATS_PATH}" "${LEAD_MOTION_ANCHOR}" "${LEAD_MOTION_ANCHOR%.npy}.json" "${DEEPSPEED_CONFIG}" "${NUM_GPUS}" "${SAMPLES_PER_GPU}" "${LEAD_GRADIENT_ACCUMULATION_STEPS}" <<'PY'
import json
import sys

import numpy as np

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    stats = json.load(stream)
expected_stats = {
    "lead_stage1_infos_version": 15,
    "agent_motion_step_sim_frames": 5,
    "agent_motion_freq_hz": 4.0,
    "agent_motion_horizon_steps": 8,
    "agent_motion_horizon_seconds": 2.0,
    "planning_step_sim_frames": 5,
    "planning_freq_hz": 4.0,
    "planning_horizon_steps": 8,
    "planning_horizon_seconds": 2.0,
}
mismatches = {
    key: (stats.get(key), expected)
    for key, expected in expected_stats.items()
    if stats.get(key) != expected
}
if mismatches:
    raise SystemExit(f"Invalid LEAD 4Hz/8-step stats: {mismatches}")

anchor = np.load(sys.argv[2], mmap_mode="r")
if anchor.shape != (9, 6, 8, 2):
    raise SystemExit(f"Invalid LEAD motion anchor shape: {anchor.shape}")
dynamic_indices = [0, 1, 2, 3, 7]
if any(not np.any(anchor[index]) for index in dynamic_indices):
    raise SystemExit("A dynamic LEAD class has all-zero motion anchors")
with open(sys.argv[3], "r", encoding="utf-8") as stream:
    metadata = json.load(stream)
if metadata.get("shape") != [9, 6, 8, 2]:
    raise SystemExit("LEAD motion anchor metadata shape is invalid")

with open(sys.argv[4], "r", encoding="utf-8") as stream:
    deepspeed = json.load(stream)
expected_batch = {
    "train_batch_size": int(sys.argv[5]) * int(sys.argv[6]) * int(sys.argv[7]),
    "train_micro_batch_size_per_gpu": int(sys.argv[6]),
    "gradient_accumulation_steps": int(sys.argv[7]),
}
batch_mismatches = {
    key: (deepspeed.get(key), expected)
    for key, expected in expected_batch.items()
    if deepspeed.get(key) != expected
}
if batch_mismatches:
    raise SystemExit(f"Invalid DeepSpeed batch configuration: {batch_mismatches}")
print(int(stats["num_train_samples"]))
PY
)}"

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
    echo "Checkpoint and final iterations must align with gradient accumulation ${LEAD_GRADIENT_ACCUMULATION_STEPS}." >&2
    exit 1
fi

echo "LEAD Stage2 motion: native 4Hz x 8 steps x 2s, 6 modes, ${LEAD_STAGE1_EPOCHS} epochs"
if [[ -n "${LEAD_MOTION_RESUME_FROM}" ]]; then
    echo "RESUME_FROM=${LEAD_MOTION_RESUME_FROM}"
else
    echo "LOAD_FROM=${LEAD_MOTION_LOAD_FROM:-<none; Qwen3-VL-2B base>}"
fi
echo "BATCH: micro_per_gpu=${SAMPLES_PER_GPU} accumulation=${LEAD_GRADIENT_ACCUMULATION_STEPS} effective_global=$((NUM_GPUS * SAMPLES_PER_GPU * LEAD_GRADIENT_ACCUMULATION_STEPS))"
echo "ITERATIONS: raw_per_epoch=${LEAD_RAW_ITERS_PER_EPOCH} aligned_per_epoch=${LEAD_STAGE1_CKPT_INTERVAL} max=${LEAD_STAGE1_MAX_ITERS}"

bash run_lead_stage1_8gpu.sh
