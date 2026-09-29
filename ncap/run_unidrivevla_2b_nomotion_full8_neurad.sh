#!/usr/bin/env bash
set -euo pipefail

# Short 8-GPU NCAP launcher. MODEL_KEY is only a run label; the model itself is
# already running on the A server and is selected there through CONFIG and
# CHECKPOINT.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT="${RELEASE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_KEY="${MODEL_KEY:-custom}"
NCAP_ROOT="${NCAP_ROOT:?Set NCAP_ROOT to the NeuroNCAP checkout}"
RUNNER="${RUNNER:-${RELEASE_ROOT}/ncap/run_unidrivevla_iter17580_full_neurad_parallel8.sh}"
MODEL_HOST="${MODEL_HOST:?Set MODEL_HOST to the model-server IP}"
MODEL_PORTS="${MODEL_PORTS:-9000,9001,9002,9003,9004,9005,9006,9007}"
RENDERER_GPUS="${RENDERER_GPUS:-0,1,2,3,4,5,6,7}"
RENDERER_BASE_PORT="${RENDERER_BASE_PORT:-8100}"
NUSCENES_ROOT="${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes metadata root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NEURAD_PYTHON_BIN="${NEURAD_PYTHON_BIN:-python}"
RUNS="${RUNS:-100}"
RESUME="${RESUME:-1}"
RUN_ID="${RUN_ID:-${MODEL_KEY}}"
RESULT_ROOT="${RESULT_ROOT:-${NCAP_ROOT}/outputs/unidrivevla_${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${NCAP_ROOT}/logs/unidrivevla_${RUN_ID}}"
TIMESTAMP="${TIMESTAMP:-${RUN_ID}}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "NCAP runner not found: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -d "${NUSCENES_ROOT}/v1.0-trainval" ]]; then
  echo "nuScenes metadata not found: ${NUSCENES_ROOT}/v1.0-trainval" >&2
  exit 1
fi

echo "Launching NCAP for ${MODEL_KEY}"
echo "Model: ${MODEL_HOST}:{${MODEL_PORTS}}"
echo "Renderer GPUs: ${RENDERER_GPUS}, base port: ${RENDERER_BASE_PORT}"
echo "Results: ${RESULT_ROOT}"

export NCAP_ROOT MODEL_HOST MODEL_PORTS RENDERER_GPUS RENDERER_BASE_PORT
export NUSCENES_ROOT RESULT_ROOT LOG_ROOT TIMESTAMP RUNS RESUME MODEL_KEY RUN_ID
export PYTHON_BIN NEURAD_PYTHON_BIN

exec bash "${RUNNER}"
