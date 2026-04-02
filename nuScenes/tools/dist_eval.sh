#!/usr/bin/env bash

T=`date +%m%d%H%M`
# export NCCL_DEBUG=NONE
# export NCCL_NET_PLUGIN=none
# export NCCL_SOCKET_NTHREADS=8
# export NCCL_SOCKET_IFNAME=bond0
# export GLOO_SOCKET_IFNAME=bond0
# export UCX_NET_DEVICES=bond0
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_RETRY_CNT=13
# export NCCL_IB_GID_INDEX=3
# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T