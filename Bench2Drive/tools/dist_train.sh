#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
CFG=$3
PORT=${PORT:-28651}

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

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

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# Prefer MLP-provided topology; fall back to local single-node defaults.
NPROC_PER_NODE=${MLP_WORKER_GPU:-${GPUS_PER_NODE}}
NNODES=${MLP_WORKER_NUM:-6}
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

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun "${DISTRIBUTED_ARGS[@]}" \
    $(dirname "$0")/train.py $CONFIG --work-dir ${WORK_DIR} --launcher pytorch ${@:4} > train-baseline-all-train-final.txt 2>&1
