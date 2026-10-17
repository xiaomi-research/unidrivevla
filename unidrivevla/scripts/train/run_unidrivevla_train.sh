#!/usr/bin/env bash
# UniDriveVLA training on one data source, inside the training env (its python and torchrun on PATH).
#   DATA=navsim|nuscenes|b2d  NAVSIM_TREE=1.1|2.0  GPUS=1  MAX_STEPS=0  MASTER_PORT=29517
#   CACHE=1 (navsim) builds the sample cache first with NAVSIM's run_dataset_caching.py; CACHE_OVERRIDES adds to it
#   TRAIN_TEST_SPLIT (navtrain)  CACHE_PATH  METRIC_CACHE_PATH (navsim PDMS)
#   NAVSIM_DEVKIT_ROOT  OPENSCENE_DATA_ROOT  NUPLAN_MAPS_ROOT  NUPLAN_MAP_VERSION  NAVSIM_EXP_ROOT
#     author defaults: OPENSCENE_DATA_ROOT=/data/yongkangli/NAVSIM/dataset
#                      NAVSIM_EXP_ROOT=/data/yongkangli/NAVSIM/exp
#                      NAVSIM_DEVKIT_ROOT=<repo>/../NavSim/navsim${NAVSIM_TREE} when that directory exists
#   bash scripts/train/run_unidrivevla_train.sh [hydra overrides]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # the unidrivevla package
PKG_PARENT="$(dirname "${ROOT}")"
DATA="${DATA:-navsim}"
NAVSIM_TREE="${NAVSIM_TREE:-1.1}"
case "${DATA}" in navsim|nuscenes|b2d|multidomain|physicalai|lead) ;; *) echo "unsupported DATA=${DATA}" >&2; exit 1 ;; esac
case "${NAVSIM_TREE}" in 1.1|2.0) ;; *) echo "NAVSIM_TREE must be 1.1 or 2.0, got ${NAVSIM_TREE}" >&2; exit 1 ;; esac

# NAVSIM_DEVKIT_ROOT, else a sibling checkout ../NavSim/navsim${NAVSIM_TREE}. nuScenes and Bench2Drive do not need it.
if [[ -z "${NAVSIM_DEVKIT_ROOT:-}" && -d "${ROOT}/../NavSim/navsim${NAVSIM_TREE}" ]]; then
  NAVSIM_DEVKIT_ROOT="$(cd "${ROOT}/../NavSim/navsim${NAVSIM_TREE}" && pwd)"
fi
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-}"
if [[ ( "${DATA}" == "navsim" || "${DATA}" == "multidomain" ) && -z "${NAVSIM_DEVKIT_ROOT}" ]]; then
  echo "DATA=navsim needs NAVSIM_DEVKIT_ROOT pointing at a navsim${NAVSIM_TREE} devkit" >&2
  exit 1
fi
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT:+${NAVSIM_DEVKIT_ROOT}:}${PKG_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/yongkangli/NAVSIM/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/yongkangli/NAVSIM/exp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# One ID is inherited by every torchrun rank and gives every invocation its own Hydra provenance directory.
UNIDRIVE_RUN_ID="${UNIDRIVE_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex[:16])')}"
export UNIDRIVE_RUN_ID
UNIDRIVE_INVOCATION_ID="$(python -c 'import uuid; print(uuid.uuid4().hex[:16])')"
export UNIDRIVE_INVOCATION_ID
export UNIDRIVE_RUN_MODE=train

GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29517}"
ARGS=("data=${DATA}")
if [[ "${DATA}" == "multidomain" ]]; then
  ARGS+=("agent=multidomain")
fi
if [[ "${MAX_STEPS:-0}" != "0" ]]; then
  ARGS+=("train.max_steps=${MAX_STEPS}")
fi

if [[ "${DATA}" == "navsim" || "${DATA}" == "multidomain" ]]; then
  TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
  CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/unidrivevla_sample_cache_navsim${NAVSIM_TREE}}"
  ARGS+=("train_test_split=${TRAIN_TEST_SPLIT}" "cache_path=${CACHE_PATH}")
  if [[ -n "${METRIC_CACHE_PATH:-}" ]]; then
    ARGS+=("+evaluator.metric_cache_path=${METRIC_CACHE_PATH}")
  fi
  if [[ "${CACHE:-0}" == "1" ]]; then
    # shellcheck disable=SC2086
    python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching.py" \
      "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://${ROOT}/config]" \
      agent=unidrivevla_navsim experiment_name=unidrivevla_caching \
      train_test_split="${TRAIN_TEST_SPLIT}" cache_path="${CACHE_PATH}" force_cache_computation=false \
      ${CACHE_OVERRIDES:-}
  fi
fi

torchrun --nnodes=1 --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" "${ROOT}/train.py" "${ARGS[@]}" "$@"
