#!/usr/bin/env bash

T=`date +%m%d%H%M`

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

CFG=$1
GPUS=$2
EXP_NAME=$3

GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

export RANK=$MLP_ROLE_INDEX
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export WORLD_SIZE=$MLP_WORKER_NUM

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}
export TORCH_NCCL_ENABLE_TIMING=1

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "GPUS: ${GPUS}"

if [ -n "$EXP_NAME" ]; then
    WORK_DIR="work_dirs/${EXP_NAME}/"
    PY_ARGS=${@:4}
else
    WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
    PY_ARGS=${@:3}
fi

LOG_DIR="${WORK_DIR}logs/baseline"

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

NPROC_PER_NODE=${MLP_WORKER_GPU:-${GPUS_PER_NODE}}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR}}
MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT}}

DISTRIBUTED_ARGS=(
    --nproc_per_node ${NPROC_PER_NODE}
    --nnodes ${NNODES}
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)
# Prefer MLP-provided topology; fall back to local single-node defaults.

export NCCL_DEBUG=ERROR                # 正常训练只显示错误日志

# Auto-detect latest DeepSpeed checkpoint for resume
RESUME_ARG=""
LATEST_LINK="${WORK_DIR%/}/latest"
if [ -L "$LATEST_LINK" ]; then
    LATEST_TARGET=$(readlink -f "$LATEST_LINK")
    if [ -d "$LATEST_TARGET" ]; then
        RESUME_ARG="--resume-from ${LATEST_TARGET}"
        echo "AutoResume: resuming from ${LATEST_TARGET}"
    fi
elif [ -f "$LATEST_LINK" ]; then
    LATEST_TAG=$(cat "$LATEST_LINK")
    LATEST_TARGET="${WORK_DIR%/}/${LATEST_TAG}"
    if [ -d "$LATEST_TARGET" ]; then
        RESUME_ARG="--resume-from ${LATEST_TARGET}"
        echo "AutoResume: resuming from ${LATEST_TARGET}"
    fi
else
    # Fallback: find iter_* dir with highest number
    LATEST_TARGET=$(ls -d ${WORK_DIR%/}/iter_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$LATEST_TARGET" ] && [ -d "$LATEST_TARGET" ]; then
        RESUME_ARG="--resume-from ${LATEST_TARGET}"
        echo "AutoResume: resuming from ${LATEST_TARGET}"
    fi
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun "${DISTRIBUTED_ARGS[@]}" \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch \
    --deterministic \
    --work-dir ${WORK_DIR} \
    ${RESUME_ARG} \
    $PY_ARGS \
    > ${LOG_DIR}/train-baseline-all-train.txt 2>&1