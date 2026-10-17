#!/usr/bin/env bash
set -euo pipefail

: "${WORK_DIR:?Set WORK_DIR to the Bench2Drive checkout root (contains bench2drive/leaderboard)}"
: "${CARLA_ROOT:?Set CARLA_ROOT to the CARLA 0.9.15 install root}"
: "${ROUTES:?Set ROUTES to the official route XML}"
: "${CHECKPOINT:?Set CHECKPOINT to an existing UniDriveVLA checkpoint}"
: "${SAVE_PATH:?Set SAVE_PATH to a fresh writable output directory}"

WORK_DIR="$(realpath "$WORK_DIR")"
CARLA_ROOT="$(realpath "$CARLA_ROOT")"
CHECKPOINT="$(realpath "$CHECKPOINT")"
ROUTES="$(realpath "$ROUTES")"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEADERBOARD_ROOT="$WORK_DIR/bench2drive/leaderboard"
EVALUATOR="$LEADERBOARD_ROOT/scripts/run_evaluation.sh"
TEAM_AGENT="$CODE_ROOT/unidrivevla/closedloop/bench2drive.py"
test -f "$EVALUATOR" && test -f "$LEADERBOARD_ROOT/team_code/unidrivevla_b2d_agent.py" \
    && test -f "$TEAM_AGENT" && test -f "$CHECKPOINT" && test -f "$ROUTES"

export WORK_DIR CARLA_ROOT
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export B2D_LEADERBOARD_ROOT="$LEADERBOARD_ROOT"
export IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-1}"
export RECORD_PATH="${RECORD_PATH:-$SAVE_PATH/records}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
PLANNER_TYPE="${PLANNER_TYPE:-}"
GPU_RANK="${GPU_RANK:-0}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-$SAVE_PATH/leaderboard_results.json}"

if [[ -e "$SAVE_PATH" || -e "$RECORD_PATH" || -e "$CHECKPOINT_ENDPOINT" ]]; then
    echo "Refusing to reuse existing B2D outputs: SAVE_PATH, RECORD_PATH, and CHECKPOINT_ENDPOINT must be fresh." >&2
    exit 2
fi
mkdir -p "$SAVE_PATH" "$RECORD_PATH"
exec bash "$EVALUATOR" "$PORT" "$TM_PORT" "$IS_BENCH2DRIVE" "$ROUTES" "$TEAM_AGENT" \
    "$CHECKPOINT" "$CHECKPOINT_ENDPOINT" "$SAVE_PATH" "$PLANNER_TYPE" "$GPU_RANK"
