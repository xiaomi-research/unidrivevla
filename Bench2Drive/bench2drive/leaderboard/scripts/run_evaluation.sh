#!/bin/bash
# Must set WORK_DIR and CARLA_ROOT before running, e.g.:
#   export WORK_DIR=/path/to/unidrivevla/Bench2Drive
#   export CARLA_ROOT=/path/to/carla_0.9.15
export WORK_DIR=${WORK_DIR:-"/path/to/unidrivevla/Bench2Drive"}
export CARLA_ROOT=${CARLA_ROOT:-"/path/to/carla_0.9.15"}
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/bench2drive
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/bench2drive/leaderboard
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/bench2drive/scenario_runner
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/bench2drive/scenario_runner

export LEADERBOARD_ROOT=${WORK_DIR}/bench2drive/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
