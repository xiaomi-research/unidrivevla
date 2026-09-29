#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export LEAD_ZIP_ROOT="${LEAD_ZIP_ROOT:?Set LEAD_ZIP_ROOT to the external LEAD route/data archive}"
export LEAD_DATA_ROOT="${LEAD_DATA_ROOT:?Set LEAD_DATA_ROOT to the external prepared LEAD data root}"
export LEAD_MOTION_INFO_DIR="${LEAD_MOTION_INFO_DIR:-${LEAD_DATA_ROOT}/infos_4hz_full_motion4hz8_v15}"
export LEAD_MOTION_ANCHOR="${LEAD_MOTION_ANCHOR:-${PROJECT_DIR}/data/kmeans/lead_motion_6_4hz8step.npy}"
export LEAD_STAGE1_REFERENCE_INFO="${LEAD_STAGE1_REFERENCE_INFO:-${LEAD_DATA_ROOT}/infos_4hz_full/lead_infos_train.pkl}"
export LEAD_STAGE1_REFERENCE_STATS="${LEAD_STAGE1_REFERENCE_STATS:-${LEAD_DATA_ROOT}/infos_4hz_full/lead_trajectory_stats.json}"
export LEAD_INFO_NUM_WORKERS="${LEAD_INFO_NUM_WORKERS:-32}"
export LEAD_INFO_SHARD_SIZE="${LEAD_INFO_SHARD_SIZE:-2048}"

if [[ "${PYTHON_BIN}" == */* ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
        exit 1
    fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "PYTHON_BIN was not found on PATH: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${LEAD_ZIP_ROOT}" ]]; then
    echo "Missing LEAD ZIP root: ${LEAD_ZIP_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${LEAD_STAGE1_REFERENCE_INFO}" ||
      ! -f "${LEAD_STAGE1_REFERENCE_STATS}" ]]; then
    echo "Missing full-4Hz Stage1 reference infos or stats." >&2
    echo "INFO=${LEAD_STAGE1_REFERENCE_INFO}" >&2
    echo "STATS=${LEAD_STAGE1_REFERENCE_STATS}" >&2
    exit 1
fi
if [[ -e "${LEAD_MOTION_INFO_DIR}/lead_infos_train.pkl" ||
      -e "${LEAD_MOTION_ANCHOR}" ]]; then
    echo "Refusing to overwrite existing 4Hz/8-step motion data. Move the existing output or select new LEAD_MOTION_INFO_DIR and LEAD_MOTION_ANCHOR paths." >&2
    exit 1
fi

echo "Preparing LEAD agent motion: 4Hz x 8 steps x 2s, info version 15"
echo "INFO_DIR=${LEAD_MOTION_INFO_DIR}"
echo "ANCHOR=${LEAD_MOTION_ANCHOR}"

"${PYTHON_BIN}" tools/generate_lead_stage1_infos.py \
    --lead-zip-root "${LEAD_ZIP_ROOT}" \
    --image-root "${LEAD_DATA_ROOT}" \
    --out-dir "${LEAD_MOTION_INFO_DIR}" \
    --val-ratio 0 \
    --split-strategy hash \
    --shard-size "${LEAD_INFO_SHARD_SIZE}" \
    --frame-stride 1 \
    --frame-offset 0 \
    --route-quality-filter tfv6_posttrain \
    --route-skip-first 8 \
    --route-skip-last 8 \
    --ego-plan-horizon 8 \
    --ego-plan-step 5 \
    --agent-motion-horizon 8 \
    --agent-motion-step 5 \
    --sensor-perturbation-probability 0.5 \
    --num-workers "${LEAD_INFO_NUM_WORKERS}"

"${PYTHON_BIN}" tools/check_lead_info_identity.py \
    --reference-manifest "${LEAD_STAGE1_REFERENCE_INFO}" \
    --candidate-manifest "${LEAD_MOTION_INFO_DIR}/lead_infos_train.pkl" \
    --reference-stats "${LEAD_STAGE1_REFERENCE_STATS}" \
    --candidate-stats "${LEAD_MOTION_INFO_DIR}/lead_trajectory_stats.json"

"${PYTHON_BIN}" tools/kmeans/generate_lead_motion_anchors.py \
    --info-file "${LEAD_MOTION_INFO_DIR}/lead_infos_train.pkl" \
    --output "${LEAD_MOTION_ANCHOR}" \
    --horizon 8 \
    --num-modes 6 \
    --distance-threshold 55 \
    --seed 0 \
    --n-init 10 \
    --expected-info-version 15

echo "LEAD 4Hz/8-step motion infos and anchors are ready."
