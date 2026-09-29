#!/usr/bin/env bash
set -euo pipefail

# A-server helper: launch one UniDriveVLA NeuroNCAP model server per GPU.
# Default ports are 9000..9007, mapped to GPU 0..7. Use this only if you want
# to run multiple NeuroNCAP scenarios in parallel from B.
# Install runtime dependencies once before launching this process group.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT="${RELEASE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-${RELEASE_ROOT}}"
SCRIPT="${SCRIPT:-${RELEASE_ROOT}/ncap/run_ncap_model_server_iter17580.sh}"
NUM_SERVERS="${NUM_SERVERS:-8}"
BASE_PORT="${BASE_PORT:-9000}"
GPU_IDS="${GPU_IDS:-}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ncap_model_server_multi8}"

if ! [[ "${NUM_SERVERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SERVERS must be a positive integer, got: ${NUM_SERVERS}" >&2
  exit 2
fi

if ! [[ "${BASE_PORT}" =~ ^[0-9]+$ ]]; then
  echo "BASE_PORT must be an integer, got: ${BASE_PORT}" >&2
  exit 2
fi

if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
else
  gpu_ids=()
  for idx in $(seq 0 $((NUM_SERVERS - 1))); do
    gpu_ids+=("${idx}")
  done
fi

if [[ "${#gpu_ids[@]}" -ne "${NUM_SERVERS}" ]]; then
  echo "GPU_IDS must contain exactly NUM_SERVERS=${NUM_SERVERS} comma-separated entries; got: ${GPU_IDS}" >&2
  exit 2
fi

for gpu_id in "${gpu_ids[@]}"; do
  if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Each GPU_IDS entry must be a non-negative integer, got: ${gpu_id}" >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}"

if [[ ! -x "${SCRIPT}" && ! -f "${SCRIPT}" ]]; then
  echo "A-side single-server script not found: ${SCRIPT}" >&2
  exit 1
fi

echo "Starting ${NUM_SERVERS} UniDriveVLA NeuroNCAP model servers"
echo "  repo:      ${REPO_ROOT}"
echo "  script:    ${SCRIPT}"
echo "  host:      ${HOST}"
echo "  base port: ${BASE_PORT}"
echo "  gpu ids:   ${gpu_ids[*]}"
echo "  log dir:   ${LOG_DIR}"
echo

pids=()

cleanup() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo
    echo "Stopping UniDriveVLA NeuroNCAP model servers..."
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
    done
  fi
}

trap cleanup INT TERM

for idx in $(seq 0 $((NUM_SERVERS - 1))); do
  port=$((BASE_PORT + idx))
  gpu_id="${gpu_ids[idx]}"
  log_file="${LOG_DIR}/launcher_gpu${gpu_id}_port${port}.log"
  echo "Launching gpu=${gpu_id} port=${port} log=${log_file}"
  (
    cd "${REPO_ROOT}"
    HOST="${HOST}" \
    GPU_ID="${gpu_id}" \
    PORT="${port}" \
    LOG_DIR="${LOG_DIR}" \
    RELEASE_ROOT="${RELEASE_ROOT}" \
    bash "${SCRIPT}"
  ) > "${log_file}" 2>&1 &
  pids+=("$!")
done

echo
echo "Launched all requested servers. This script will stay in the foreground."
echo "Watch logs with:"
echo "  tail -f ${LOG_DIR}/latest_gpu${gpu_ids[0]}_port${BASE_PORT}.log"
echo "  tail -f ${LOG_DIR}/latest_gpu${gpu_ids[NUM_SERVERS - 1]}_port$((BASE_PORT + NUM_SERVERS - 1)).log"
echo
echo "From B, use MODEL_PORTS=${BASE_PORT},$((BASE_PORT + 1)),...,$((BASE_PORT + NUM_SERVERS - 1))"

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
