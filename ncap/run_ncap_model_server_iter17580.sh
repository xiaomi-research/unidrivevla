#!/usr/bin/env bash
set -euo pipefail

# A-server script: start the UniDriveVLA model HTTP server for NeuroNCAP.
#
# Run this on the machine that has the UniDriveVLA environment, checkpoint,
# config, nuScenes infos, and GPU access.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT="${RELEASE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${RELEASE_ROOT}/ncap}"
MODEL_ROOT="${MODEL_ROOT:-${RELEASE_ROOT}}"
CONFIG="${CONFIG:?Set CONFIG to the UniDriveVLA config for the model being evaluated}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the UniDriveVLA checkpoint for the model being evaluated}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MMCV_ROOT="${MMCV_ROOT:-${MODEL_ROOT}/third_party/mmcv-1.7.2}"
MMDET3D_ROOT="${MMDET3D_ROOT:-${MODEL_ROOT}/third_party/mmdetection3d-1.0.0rc6}"
UNIDRIVE_WASP_SVD_CACHE_DIR="${UNIDRIVE_WASP_SVD_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/unidrivevla/wasp_svd}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${MODEL_ROOT}/nuScenes}"
DATA_INFOS_ROOT="${DATA_INFOS_ROOT:-${NUSCENES_ROOT}/data/infos}"
NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-${NUSCENES_ROOT}/data/nuscenes}"
NUSCENES_ANCHOR_ROOT="${NUSCENES_ANCHOR_ROOT:-${NUSCENES_ROOT}/data/kmeans}"
VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:?Set VLM_PRETRAINED_PATH to the base HF model directory}"
OCCWORLD_VAE_PATH="${OCCWORLD_VAE_PATH:?Set OCCWORLD_VAE_PATH to the OccWorld VAE checkpoint}"
ENABLE_COC="${ENABLE_COC:-1}"
ENABLE_COC_GENERATE="${ENABLE_COC_GENERATE:-1}"
COC_EVAL_SOURCE="${COC_EVAL_SOURCE:-pred}"
COC_EVAL_INJECTION="${COC_EVAL_INJECTION:-answer}"
COC_JSONL_PATH="${COC_JSONL_PATH:-${NUSCENES_DATA_ROOT}/alpamayo_coc_val_with_nav.jsonl}"
HUGSIM_USE_GT_EGO_STATUS="${HUGSIM_USE_GT_EGO_STATUS:-1}"
HUGSIM_CHECKPOINT_MIN_PARAMETER_COVERAGE="${HUGSIM_CHECKPOINT_MIN_PARAMETER_COVERAGE:-0.99}"
VLA_ATTN_IMPLEMENTATION="${VLA_ATTN_IMPLEMENTATION:-sdpa}"
LOG_DIR="${LOG_DIR:-${MODEL_ROOT}/logs/ncap_model_server}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/iter17580_gpu${GPU_ID}_port${PORT}_${TIMESTAMP}.log}"
LATEST_LOG="${LOG_DIR}/latest_gpu${GPU_ID}_port${PORT}.log"

mkdir -p "${LOG_DIR}"

cd "${MODEL_ROOT}"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sfn "${LOG_FILE}" "${LATEST_LOG}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${ADAPTER_ROOT}/inference/server.py" ]]; then
  echo "NeuroNCAP adapter not found: ${ADAPTER_ROOT}/inference/server.py" >&2
  exit 1
fi

if [[ ! -d "${DATA_INFOS_ROOT}" ]]; then
  echo "nuScenes infos directory not found: ${DATA_INFOS_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${NUSCENES_DATA_ROOT}" ]]; then
  echo "nuScenes data directory not found: ${NUSCENES_DATA_ROOT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export UNIDRIVEVLA_MODEL_ROOT="${MODEL_ROOT}"
export UNIDRIVE_WASP_SVD_CACHE_DIR
export NUSCENES_ROOT NUSCENES_DATA_ROOT NUSCENES_ANCHOR_ROOT DATA_INFOS_ROOT VLM_PRETRAINED_PATH OCCWORLD_VAE_PATH
export ENABLE_COC ENABLE_COC_GENERATE COC_EVAL_SOURCE COC_EVAL_INJECTION
export COC_JSONL_PATH HUGSIM_USE_GT_EGO_STATUS
export HUGSIM_CHECKPOINT_MIN_PARAMETER_COVERAGE VLA_ATTN_IMPLEMENTATION
export MMCV_ROOT MMDET3D_ROOT
export PYTHONPATH="${MODEL_ROOT}/nuScenes:${ADAPTER_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${MMCV_ROOT}:${PYTHONPATH}"
export PYTHONPATH="${MMDET3D_ROOT}:${PYTHONPATH}"

echo "Starting UniDriveVLA NeuroNCAP model server"
echo "  adapter:    ${ADAPTER_ROOT}"
echo "  model repo: ${MODEL_ROOT}"
echo "  config:     ${CONFIG}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  host:       ${HOST}"
echo "  port:       ${PORT}"
echo "  gpu id:     ${GPU_ID}"
echo "  python bin: ${PYTHON_BIN}"
echo "  mmcv root:  ${MMCV_ROOT}"
echo "  mmdet3d:    ${MMDET3D_ROOT}"
echo "  svd cache:  ${UNIDRIVE_WASP_SVD_CACHE_DIR}"
echo "  coc jsonl:  ${COC_JSONL_PATH}"
echo "  coc mode:   generate=${ENABLE_COC_GENERATE}, source=${COC_EVAL_SOURCE}, injection=${COC_EVAL_INJECTION}"
echo "  gt ego:     ${HUGSIM_USE_GT_EGO_STATUS}"
echo "  log file:   ${LOG_FILE}"
echo "  latest:     ${LATEST_LOG}"
echo
echo "Checking Python dependencies..."
"${PYTHON_BIN}" - <<'PY'
import sys

print("python executable:", sys.executable)
print("python version:", sys.version.replace("\n", " "))

for name in ["torch", "mmcv", "mmdet", "mmdet3d"]:
    try:
        module = __import__(name)
        print(f"{name}: {getattr(module, '__version__', 'unknown')} ({getattr(module, '__file__', 'built-in')})")
    except Exception as exc:
        print(f"{name}: IMPORT FAILED: {repr(exc)}")
        raise
PY
echo
echo "From the B server, test connectivity with:"
echo "  curl -X POST http://<A_SERVER_IP>:${PORT}/reset -H 'Content-Type: application/json' -d '{}'"
echo

"${PYTHON_BIN}" "${ADAPTER_ROOT}/inference/server.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device cuda:0

STATUS=$?
echo "[$(date '+%F %T')] UniDriveVLA NeuroNCAP model server exited with status ${STATUS}"
exit "${STATUS}"
