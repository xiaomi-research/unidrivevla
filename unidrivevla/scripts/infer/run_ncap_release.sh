#!/usr/bin/env bash
# PR 10 ncap/run_ncap_model_server_iter17580_multi8.sh: one Model Node per GPU, ports 9000.. .
# Each server is this repo's serve.py (POST /reset and POST /infer).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
NUM_SERVERS="${NUM_SERVERS:-8}"
BASE_PORT="${BASE_PORT:-9000}"
CKPT="${CKPT:?set CKPT to the release checkpoint}"
for idx in $(seq 0 $((NUM_SERVERS - 1))); do
  port=$((BASE_PORT + idx))
  CUDA_VISIBLE_DEVICES="$idx" python -m unidrivevla.serve --profile nuscenes --checkpoint "$CKPT" --port "$port" &
done
wait
