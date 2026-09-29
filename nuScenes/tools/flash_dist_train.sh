#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG GPUS EXP_NAME [train.py arguments ...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CFG="$1"
GPUS="$2"
EXP_NAME="$3"
shift 3

if [[ "${CFG}" != /* ]]; then
    CFG="${ROOT}/${CFG}"
fi
if [[ "${EXP_NAME}" = /* ]]; then
    WORK_DIR="${EXP_NAME}"
else
    WORK_DIR="${ROOT}/work_dirs/${EXP_NAME}"
fi

if [[ ! -f "${CFG}" ]]; then
    echo "Flash config does not exist: ${CFG}" >&2
    exit 1
fi

GPUS_PER_NODE=$((GPUS < 8 ? GPUS : 8))
NPROC_PER_NODE="${MLP_WORKER_GPU:-${GPUS_PER_NODE}}"
NNODES="${MLP_WORKER_NUM:-1}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-127.0.0.1}"
MASTER_PORT="${MLP_WORKER_0_PORT:-29500}"

mkdir -p "${WORK_DIR}/logs/baseline"

RESUME_ARG=()
LATEST_LINK="${WORK_DIR}/latest"
if [[ -L "${LATEST_LINK}" ]]; then
    LATEST_TARGET="$(readlink -f "${LATEST_LINK}")"
    [[ -d "${LATEST_TARGET}" ]] && RESUME_ARG=(--resume-from "${LATEST_TARGET}")
elif [[ -f "${LATEST_LINK}" ]]; then
    LATEST_TAG="$(<"${LATEST_LINK}")"
    LATEST_TARGET="${WORK_DIR}/${LATEST_TAG}"
    [[ -d "${LATEST_TARGET}" ]] && RESUME_ARG=(--resume-from "${LATEST_TARGET}")
else
    LATEST_TARGET="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'iter_*' -printf '%f\n' 2>/dev/null | sort -t_ -k2,2n | tail -1 || true)"
    [[ -n "${LATEST_TARGET}" ]] && RESUME_ARG=(--resume-from "${WORK_DIR}/${LATEST_TARGET}")
fi

if [[ ${#RESUME_ARG[@]} -gt 0 ]]; then
    echo "[Flash] resuming from ${RESUME_ARG[*]}"
fi

export PYTHONPATH="${ROOT}:${FLASH_MMCV_ROOT}:${FLASH_MMDET3D_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

cd "${ROOT}"
python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    tools/flash_train.py "${CFG}" \
    --launcher pytorch \
    --deterministic \
    --work-dir "${WORK_DIR}" \
    "${RESUME_ARG[@]}" \
    "$@" \
    > "${WORK_DIR}/logs/baseline/train-baseline-all-train.txt" 2>&1
