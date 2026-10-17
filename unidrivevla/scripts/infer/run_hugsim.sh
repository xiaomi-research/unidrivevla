#!/usr/bin/env bash
set -eu

# HUGSIM invokes its AD shell as: <shell> <cuda_id> <output_dir>.
cuda_id="${1:?HUGSIM CUDA id required}"
output_dir="${2:?HUGSIM scenario output directory required}"
code_root="$(cd "$(dirname "$0")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$cuda_id"
export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
: "${UNIDRIVE_CHECKPOINT:?set the UniDriveVLA checkpoint path}"
: "${HUGSIM_CAMERA_CONFIG:?set the same camera YAML used by HUGSIM}"
: "${HUGSIM_KINEMATIC_CONFIG:?set the same kinematic YAML used by HUGSIM}"
python_bin="${UNIDRIVE_PYTHON:-python}"
exec "$python_bin" -m unidrivevla.closedloop.hugsim_client --output "$output_dir"
