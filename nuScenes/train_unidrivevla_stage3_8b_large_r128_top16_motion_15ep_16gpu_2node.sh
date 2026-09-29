#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT="${NUSCENES_ROOT:-${SCRIPT_DIR}}"
DATA_CWD="${NUSCENES_DATA_CWD:-${CODE_ROOT}}"
cd "${DATA_CWD}"
export NUSCENES_ROOT="${CODE_ROOT}"

export PYTHON_BIN=${PYTHON_BIN:-python}
export MMCV_ROOT=${MMCV_ROOT:-${CODE_ROOT}/../third_party/mmcv-1.7.2}
export MMDET3D_ROOT=${MMDET3D_ROOT:-${CODE_ROOT}/../third_party/mmdetection3d-1.0.0rc6}
export PYTHONPATH="${MMCV_ROOT}:${MMDET3D_ROOT}:${CODE_ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-${CODE_ROOT}/projects/configs/UniDriveVLA/unidrivevla_nusc_8b_stage3.py}"
STAGE2_EXP="${CODE_ROOT}/work_dirs2/UniDriveVLA/unidrivevla_stage2_8b_nusc_large_stage1_coc_weight01_visual_wasp_beta0005_r128a256_top16_projector_lora_r128a256_llm_wasp_beta0003_r128a256_top16_2node16gpu_bs2_accum4_zero2_vgc"

export NUM_GPUS="${NUM_GPUS:-16}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-2}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-127.0.0.1}"

if [[ "${NUM_GPUS}" == "8" ]]; then
    DEFAULT_DS_CONFIG="${CODE_ROOT}/zero_configs/adam_zero2_bf16_bs2_accum8.json"
    DEFAULT_WORK_DIR="${CODE_ROOT}/work_dirs2/UniDriveVLA/unidrivevla_stage3_8b_large_r128a256_top16_motion_15ep_from_stage2_epoch20_bs2_accum8_8gpu_zero2_vgc_noema_eager"
    DEFAULT_BATCH_SUMMARY="2/GPU x 8 GPUs x accumulation 8 = 128"
    DEFAULT_SCHEDULE_SUMMARY="1758 iter/epoch x 15 epochs = 26370 iter"
elif [[ "${NUM_GPUS}" == "16" ]]; then
    DEFAULT_DS_CONFIG="${CODE_ROOT}/zero_configs/adam_zero2_bf16_bs2_accum4_16gpu.json"
    DEFAULT_WORK_DIR="${CODE_ROOT}/work_dirs2/UniDriveVLA/unidrivevla_stage3_8b_large_r128a256_top16_motion_15ep_from_stage2_epoch20_bs2_accum4_2node16gpu_zero2_vgc_noema_eager"
    DEFAULT_BATCH_SUMMARY="2/GPU x 16 GPUs x accumulation 4 = 128"
    DEFAULT_SCHEDULE_SUMMARY="879 iter/epoch x 15 epochs = 13185 iter"
else
    echo "ERROR: supported NUM_GPUS values are 8 and 16, got ${NUM_GPUS}." >&2
    exit 1
fi

WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"

# Use the requested Stage 2 epoch-20 checkpoint by default.
export STAGE1_CHECKPOINT="${STAGE2_CHECKPOINT:-${STAGE2_EXP}/iter_17580/global_step4395/mp_rank_00_model_states.pt}"
export VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-/path/to/UniDriveVLA_Nusc_Large_Stage1}"
export OCCWORLD_VAE_PATH="${OCCWORLD_VAE_PATH:-/path/to/occvae_latest.pth}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DEFAULT_DS_CONFIG}}"
export DATA_INFOS_ROOT="${DATA_INFOS_ROOT:-${DATA_CWD}/data/infos}"
export NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-${DATA_CWD}/data/nuscenes}"
export NUSCENES_ANCHOR_ROOT="${NUSCENES_ANCHOR_ROOT:-${DATA_CWD}/data/kmeans}"
export MOTION_ANCHOR="${MOTION_ANCHOR:-${DATA_CWD}/data/kmeans/kmeans_motion_6.npy}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-28851}"
export EVAL_PORT="${EVAL_PORT:-28852}"
export RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
export MOTION_THRESHOLD="${MOTION_THRESHOLD:-0.20}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export UNIDRIVE_DISABLE_COMPILED_FLEX="${UNIDRIVE_DISABLE_COMPILED_FLEX:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export UNIDRIVE_WASP_SVD_CACHE_DIR="${UNIDRIVE_WASP_SVD_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/unidrivevla/wasp_svd}"
export UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING="${UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING:-1}"
export VLA_ATTN_IMPLEMENTATION="${VLA_ATTN_IMPLEMENTATION:-sdpa}"
BATCH_SUMMARY="${BATCH_SUMMARY:-${DEFAULT_BATCH_SUMMARY}}"
SCHEDULE_SUMMARY="${SCHEDULE_SUMMARY:-${DEFAULT_SCHEDULE_SUMMARY}}"

for required_file in \
    "${CONFIG}" \
    "${STAGE1_CHECKPOINT}" \
    "${OCCWORLD_VAE_PATH}" \
    "${DEEPSPEED_CONFIG}" \
    "${NUSCENES_ANCHOR_ROOT}/kmeans_det_900.npy" \
    "${NUSCENES_ANCHOR_ROOT}/kmeans_map_100.npy" \
    "${MOTION_ANCHOR}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file does not exist: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -f "${VLM_PRETRAINED_PATH}/config.json" ]]; then
    echo "ERROR: VLM_PRETRAINED_PATH is invalid: ${VLM_PRETRAINED_PATH}" >&2
    exit 1
fi

if ! [[ "${MLP_WORKER_NUM}" =~ ^[1-9][0-9]*$ && "${MLP_WORKER_GPU}" =~ ^[1-9][0-9]*$ && "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid distributed layout variables." >&2
    exit 1
fi
if (( MLP_WORKER_NUM * MLP_WORKER_GPU != NUM_GPUS || NODE_RANK >= MLP_WORKER_NUM )); then
    echo "ERROR: NUM_GPUS=${NUM_GPUS}, nodes=${MLP_WORKER_NUM}, GPUs/node=${MLP_WORKER_GPU}, node_rank=${NODE_RANK} are inconsistent." >&2
    exit 1
fi
if (( MLP_WORKER_NUM > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "::1" ]]; then
    echo "ERROR: multi-node training needs MLP_WORKER_0_HOST reachable from both nodes." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import mmcv, peft, torch" >/dev/null 2>&1; then
    echo "ERROR: the active conda environment is missing mmcv, peft, or torch." >&2
    exit 1
fi

"${PYTHON_BIN}" - "${MOTION_ANCHOR}" <<'PY'
import sys

import numpy as np

anchor = np.load(sys.argv[1], mmap_mode="r")
if anchor.shape != (10, 6, 12, 2):
    raise SystemExit(
        f"ERROR: expected motion anchor shape (10, 6, 12, 2), got {anchor.shape}"
    )
if not np.isfinite(anchor).all():
    raise SystemExit("ERROR: motion anchor contains NaN or Inf")
if not np.any(anchor, axis=(1, 2, 3)).all():
    raise SystemExit("ERROR: at least one nuScenes class has an all-zero anchor")
print(f"Motion anchor validated: {anchor.shape} from {sys.argv[1]}")
PY

AVAILABLE_GPUS="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${AVAILABLE_GPUS}" -lt "${MLP_WORKER_GPU}" ]]; then
    echo "ERROR: ${MLP_WORKER_GPU} local GPUs are required, but only ${AVAILABLE_GPUS} are visible." >&2
    exit 1
fi

case "${UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING,,}" in
    1|true|yes|on) VISUAL_GC_CFG=True ;;
    0|false|no|off) VISUAL_GC_CFG=False ;;
    *)
        echo "ERROR: UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING must be 0 or 1." >&2
        exit 1
        ;;
esac

RESUME_FROM="${RESUME_FROM:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
if [[ -z "${RESUME_FROM}" && "${AUTO_RESUME}" == "1" ]]; then
    LATEST_LINK="${WORK_DIR%/}/latest"
    if [[ -L "${LATEST_LINK}" ]]; then
        RESUME_FROM="$(readlink -f "${LATEST_LINK}")"
    elif [[ -f "${LATEST_LINK}" ]]; then
        LATEST_TAG="$(<"${LATEST_LINK}")"
        RESUME_FROM="${WORK_DIR%/}/${LATEST_TAG}"
    elif [[ -d "${WORK_DIR}" ]]; then
        RESUME_FROM="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'iter_*' -print 2>/dev/null | sort -V | tail -n 1)"
    fi
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    if [[ ! -d "${RESUME_FROM}" ]]; then
        echo "ERROR: RESUME_FROM is not a DeepSpeed checkpoint directory: ${RESUME_FROM}" >&2
        exit 1
    fi
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_motion_15ep_${NUM_GPUS}gpu.${TIMESTAMP}.node${NODE_RANK}.log"

echo "============================================================"
echo "UniDriveVLA Large Stage 3 motion training: 15 epochs"
echo "CONFIG=${CONFIG}"
echo "STAGE2_CHECKPOINT=${STAGE1_CHECKPOINT}"
echo "MOTION_ANCHOR=${MOTION_ANCHOR}"
echo "VLM_PRETRAINED_PATH=${VLM_PRETRAINED_PATH}"
echo "WORK_DIR=${WORK_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Distributed: nodes=${MLP_WORKER_NUM}, GPUs/node=${MLP_WORKER_GPU}, node_rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "Batch: ${BATCH_SUMMARY}"
echo "Schedule: ${SCHEDULE_SUMMARY}"
echo "Adapters: load r128/a256 top16/16 from Stage 2; frozen in Stage 3"
echo "WASP regularization: disabled (loss weights are zero)"
echo "EMA: disabled (evaluation uses raw trained weights)"
echo "torch.compile: disabled"
echo "Attention: ${VLA_ATTN_IMPLEMENTATION}"
echo "Visual gradient checkpointing: ${VISUAL_GC_CFG}"
echo "Final evaluation: motion only, threshold ${MOTION_THRESHOLD}"
if [[ -n "${RESUME_FROM}" ]]; then
    echo "Resume: ${RESUME_FROM}"
else
    echo "Resume: disabled; initialize from STAGE2_CHECKPOINT"
fi
echo "Log: ${LOG_FILE}"
echo "============================================================"

torchrun \
    --nproc_per_node="${MLP_WORKER_GPU}" \
    --nnodes="${MLP_WORKER_NUM}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${CODE_ROOT}/tools/train.py" \
    "${CONFIG}" \
    --launcher pytorch \
    --deterministic \
    --no-validate \
    --work-dir "${WORK_DIR}" \
    "${RESUME_ARGS[@]}" \
    --cfg-options \
        model.planning_head.attn_implementation="${VLA_ATTN_IMPLEMENTATION}" \
        model.planning_head.visual_gradient_checkpointing="${VISUAL_GC_CFG}" \
        model.planning_head.unified_decoder_cfg.motion_anchor="${MOTION_ANCHOR}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

if [[ "${RUN_FINAL_EVAL}" != "1" ]]; then
    echo "Training finished. Motion-only evaluation skipped (RUN_FINAL_EVAL=${RUN_FINAL_EVAL})."
    exit 0
fi

EVAL_CKPT="${EVAL_CKPT:-}"
if [[ -z "${EVAL_CKPT}" ]]; then
    LATEST_ITER_DIR="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'iter_*' -print | sort -V | tail -n 1)"
    if [[ -n "${LATEST_ITER_DIR}" ]]; then
        EVAL_CKPT="$(find "${LATEST_ITER_DIR}" -path '*/mp_rank_00_model_states.pt' -type f -print | sort -V | tail -n 1)"
    fi
fi

if [[ -z "${EVAL_CKPT}" || ! -f "${EVAL_CKPT}" ]]; then
    echo "ERROR: no checkpoint found for motion-only evaluation under ${WORK_DIR}." >&2
    exit 1
fi

ITER_TAG="$(basename "$(dirname "$(dirname "${EVAL_CKPT}")")")"
THRESHOLD_TAG="${MOTION_THRESHOLD/./p}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${WORK_DIR%/}/eval_${ITER_TAG}_motion_only_thresh_${THRESHOLD_TAG}_8gpu}"
EVAL_LOG_DIR="${EVAL_OUT_DIR}/logs"
mkdir -p "${EVAL_LOG_DIR}"
EVAL_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_LOG_FILE="${EVAL_LOG_DIR}/eval_${ITER_TAG}_motion_only_thresh_${THRESHOLD_TAG}.${EVAL_TIMESTAMP}.node${NODE_RANK}.log"

echo "============================================================"
echo "UniDriveVLA Large Stage 3 final motion-only evaluation"
echo "CONFIG=${CONFIG}"
echo "EVAL_CKPT=${EVAL_CKPT}"
echo "EVAL_OUT_DIR=${EVAL_OUT_DIR}"
echo "MOTION_THRESHOLD=${MOTION_THRESHOLD}"
echo "Log: ${EVAL_LOG_FILE}"
echo "============================================================"

torchrun \
    --nproc_per_node="${MLP_WORKER_GPU}" \
    --nnodes="${MLP_WORKER_NUM}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${EVAL_PORT}" \
    "${CODE_ROOT}/tools/test.py" \
    "${CONFIG}" \
    "${EVAL_CKPT}" \
    --launcher pytorch \
    --eval bbox \
    --out "${EVAL_OUT_DIR}/results.pkl" \
    --show-dir "${EVAL_OUT_DIR}" \
    --cfg-options \
        model.planning_head.attn_implementation="${VLA_ATTN_IMPLEMENTATION}" \
        model.planning_head.visual_gradient_checkpointing="${VISUAL_GC_CFG}" \
        model.planning_head.unified_decoder_cfg.motion_anchor="${MOTION_ANCHOR}" \
        evaluation.eval_mode.with_det=False \
        evaluation.eval_mode.with_tracking=False \
        evaluation.eval_mode.with_map=False \
        evaluation.eval_mode.with_motion=True \
        evaluation.eval_mode.with_planning=False \
        evaluation.eval_mode.motion_threshhold="${MOTION_THRESHOLD}" \
    2>&1 | tee "${EVAL_LOG_FILE}"
