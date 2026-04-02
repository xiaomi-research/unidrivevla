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

DISTRIBUTED_ARGS=(
    --nproc_per_node $MLP_WORKER_GPU
    --nnodes $MLP_WORKER_NUM
    --node_rank $MLP_ROLE_INDEX
    --master_addr $MLP_WORKER_0_HOST
    --master_port $MLP_WORKER_0_PORT
)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=1 \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch $PY_ARGS \
    --deterministic \
    --work-dir ${WORK_DIR} #> ${LOG_DIR}/train-baseline-all-train.txt 2>&1