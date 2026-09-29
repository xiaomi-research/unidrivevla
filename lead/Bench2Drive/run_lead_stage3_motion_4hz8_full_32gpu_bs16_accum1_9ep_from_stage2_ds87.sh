#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"

readonly STAGE2_DS87_CHECKPOINT="${LEAD_STAGE1_CHECKPOINT:?Set LEAD_STAGE1_CHECKPOINT to the external 3-epoch Stage1 checkpoint}"

if [[ ! -f "${STAGE2_DS87_CHECKPOINT}" || ! -s "${STAGE2_DS87_CHECKPOINT}" ]]; then
    echo "Missing or empty DS=87.0619 Stage 1 checkpoint file: ${STAGE2_DS87_CHECKPOINT}" >&2
    exit 1
fi

# Fix the experiment identity here so inherited shell variables cannot select a
# different checkpoint, batch size, schedule, or retention policy.
export LEAD_STAGE1_ENV=b2dzoo
export NUM_GPUS=32
export MLP_WORKER_NUM=4
export MLP_WORKER_GPU=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SAMPLES_PER_GPU=16
export LEAD_EXPECTED_NUM_GPUS=32
export LEAD_EXPECTED_WORKER_NUM=4
export LEAD_EXPECTED_WORKER_GPU=8
export LEAD_EXPECTED_SAMPLES_PER_GPU=16
export LEAD_GRADIENT_ACCUMULATION_STEPS=1
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_DIR}/zero_configs/adam_zero1_bf16_bs16_accum1_32gpu_b2dzoo.json}"

export LEAD_STAGE1_CONFIG=projects/configs/unidrivevla_lead_stage2_motion_4hz8_2b.py
export LEAD_STAGE1_EPOCHS=9
export LEAD_STAGE1_LR=1e-4
export LEAD_MOTION_LR=1e-4
export LEAD_SEQUENCES_SPLIT_NUM=2
export LEAD_STAGE1_CKPT_INTERVAL=""
export LEAD_STAGE1_MAX_ITERS=""
export LEAD_STAGE1_MAX_KEEP_CKPTS=9
export ENABLE_VALIDATE=0
export LEAD_STAGE1_ATTN_IMPLEMENTATION=eager
export LEAD_PLANNING_QUERY_LAYOUT=action_expert
export LEAD_AUX_ROUTE_LOSS_WEIGHT=1.0
export LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT=1.0
export LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT=1.0

# LOAD_FROM initializes model weights only. RESUME_FROM remains empty so the
# Stage 2 optimizer state and iteration counter are not restored.
export LEAD_MOTION_FROM_QWEN_BASE=0
export LEAD_MOTION_LOAD_FROM="${STAGE2_DS87_CHECKPOINT}"
export LEAD_MOTION_RESUME_FROM=""
export LEAD_STAGE1_LOAD_FROM=""
export LEAD_STAGE1_RESUME_FROM=""

export WORK_DIR=work_dirs2/unidrivevla_lead_stage3_motion_4hz8_2b_4node32gpu_bs16_accum1_9ep_from_stage2_ds87_iter11358_keep9
export LOG_FILE="work_dirs2/train-lead-stage3-motion-4hz8-32gpu-bs16-accum1-9ep-from-stage2-ds87-node${MLP_ROLE_INDEX:-0}-$(date +%Y%m%d-%H%M%S).log"

exec bash run_lead_stage2_motion_4hz8_full_16gpu.sh
