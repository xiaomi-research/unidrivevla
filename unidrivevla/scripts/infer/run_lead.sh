#!/usr/bin/env bash
set -euo pipefail

: "${LEAD_ROOT:?Set LEAD_ROOT to the pinned kesai-labs/lead checkout}"
: "${CARLA_ROOT:?Set CARLA_ROOT to CARLA 0.9.15/0.9.16}"
: "${ROUTES:?Set ROUTES to an official LEAD route XML}"
: "${LEAD_AGENT_CONFIG:?Set LEAD_AGENT_CONFIG to YAML with server_url and optional timeout}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh official-result directory}"

LEAD_ROOT="$(realpath "$LEAD_ROOT")"
CARLA_ROOT="$(realpath "$CARLA_ROOT")"
ROUTES="$(realpath "$ROUTES")"
LEAD_AGENT_CONFIG="$(realpath "$LEAD_AGENT_CONFIG")"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEAD_PYTHON="${LEAD_PYTHON:-python}"
test -f "$ROUTES" && test -f "$LEAD_AGENT_CONFIG"

# LEAD's convenience CLI derives an in-tree model agent from a LEAD checkpoint.
# This adapter uses the same pinned official Leaderboard evaluator, while UniDriveVLA
# inference remains in its separate navsim service process.
export LEAD_ROOT CARLA_ROOT ROUTES LEAD_AGENT_CONFIG
export OUTPUT_DIR
export LEAD_MODE="${LEAD_MODE:-standard}"
export LEAD_CUDA_VISIBLE_DEVICES="${LEAD_CUDA_VISIBLE_DEVICES:-0}"
export LEAD_PYTHON
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$LEAD_PYTHON" -m unidrivevla.closedloop.lead_runner
