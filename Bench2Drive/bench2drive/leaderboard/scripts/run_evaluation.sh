#!/bin/bash
set -e
if [[ $# -ne 10 ]]; then
    echo "Usage: $0 PORT TM_PORT IS_BENCH2DRIVE ROUTES AGENT AGENT_CONFIG CHECKPOINT_FILE SAVE_PATH PLANNER GPU" >&2
    exit 2
fi

export WORK_DIR="${WORK_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd -- "$WORK_DIR"
export PROJECT_DIR="${PROJECT_DIR:-$WORK_DIR}"
export CARLA_ROOT="${CARLA_ROOT:-$WORK_DIR/carla}"
export CARLA_SERVER="$CARLA_ROOT/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="$WORK_DIR/bench2drive/scenario_runner"
export LEADERBOARD_ROOT="$WORK_DIR/bench2drive/leaderboard"
evaluator="$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py"

if [[ ! -f "$evaluator" ]]; then
    echo "Bench2Drive evaluator missing at $evaluator. Set WORK_DIR to the repository's Bench2Drive directory." >&2
    exit 1
fi
if [[ ! -x "$CARLA_SERVER" ]]; then
    echo "CARLA executable missing at $CARLA_SERVER. Set CARLA_ROOT to your CARLA 0.9.15 installation." >&2
    exit 1
fi
export PYTHONPATH="$WORK_DIR:$WORK_DIR/bench2drive:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"

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

CUDA_VISIBLE_DEVICES="$GPU_RANK" exec python "$evaluator" \
    --routes="$ROUTES" \
    --repetitions="$REPETITIONS" \
    --track="$CHALLENGE_TRACK_CODENAME" \
    --checkpoint="$CHECKPOINT_ENDPOINT" \
    --agent="$TEAM_AGENT" \
    --agent-config="$TEAM_CONFIG" \
    --debug="$DEBUG_CHALLENGE" \
    --record="${RECORD_PATH:-}" \
    --resume="$RESUME" \
    --port="$PORT" \
    --traffic-manager-port="$TM_PORT" \
    --gpu-rank="$GPU_RANK"
