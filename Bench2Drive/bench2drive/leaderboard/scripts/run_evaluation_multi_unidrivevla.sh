#!/bin/bash

# ===== User Configuration =====
WORK_DIR=${WORK_DIR:-$(pwd)}           # root of unidrivevla/Bench2Drive, or set via env
CHECKPOINT=${CHECKPOINT:-"/path/to/UniDriveVLA_Stage3_Bench2Drive_2B.pt"}
SAVE_PATH=${SAVE_PATH:-"evaluation/unidrivevla_b2d"}
# ==============================

BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True

CONFIG_NAME=unidrivevla_b2d_stage2_unified_2b

TEAM_AGENT=bench2drive/leaderboard/team_code/unidrivevla_b2d_agent.py
TEAM_CONFIG=${WORK_DIR}/projects/configs/${CONFIG_NAME}.py+${CHECKPOINT}

PLANNER_TYPE=traj
BASE_ROUTES=bench2drive/leaderboard/data/splits8/bench2drive220
BASE_CHECKPOINT_ENDPOINT=$SAVE_PATH/$CONFIG_NAME
GPU_RANK_LIST=(0 1 2 3 4 5 6 7)
TASK_LIST=(0 1 2 3 4 5 6 7)

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p "$SAVE_PATH"
fi

echo -e "Final 2B model — PID v19b (speed_KP=3.0)"
echo -e "TASK_LIST: $TASK_LIST"
echo -e "GPU_RANK_LIST: $GPU_RANK_LIST"
echo -e "\033[36m***********************************************************************************\033[0m"

length=${#TASK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 200))
    TM_PORT=$((BASE_TM_PORT + i * 200))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}.xml"
    CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}

    echo -e "TASK_ID: $i  PORT: $PORT  GPU: $GPU_RANK"
    echo -e "\033[36m***********************************************************************************\033[0m"
    PID_ABLATION=v19b IS_VISUALIZE=1 bash -e bench2drive/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.log &
    sleep 5
done
wait
