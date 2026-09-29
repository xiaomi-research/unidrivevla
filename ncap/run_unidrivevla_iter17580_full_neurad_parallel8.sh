#!/usr/bin/env bash
set -euo pipefail

# B-server script: full NeuroNCAP evaluation for UniDriveVLA iter_17580 with
# real NeuRAD rendering. Start the 8 UniDriveVLA model servers on A first, then
# run this script on B. B will start one local NeuRAD renderer per active task.

NCAP_ROOT="${NCAP_ROOT:?Set NCAP_ROOT to the NeuroNCAP checkout}"
NEURAD_ROOT="${NEURAD_ROOT:?Set NEURAD_ROOT to the NeuRAD checkout}"
NEURAD_CHECKPOINT_ROOT="${NEURAD_CHECKPOINT_ROOT:-${NEURAD_ROOT}/checkpoints}"
NUSCENES_ROOT="${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes metadata root}"

MODEL_HOST="${MODEL_HOST:?Set MODEL_HOST to the A-server VPC IP, for example MODEL_HOST=10.96.58.196}"
MODEL_PORTS="${MODEL_PORTS:-9000,9001,9002,9003,9004,9005,9006,9007}"

RUNS="${RUNS:-100}"
RESUME="${RESUME:-0}"
START_TASK_INDEX="${START_TASK_INDEX:-0}"
SELECTED_TASK_RUNS="${SELECTED_TASK_RUNS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NEURAD_PYTHON_BIN="${NEURAD_PYTHON_BIN:-python}"
VISER_ZIP="${VISER_ZIP:-${NCAP_ROOT}/viser-main.zip}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-${HOME}/.cache/torch/hub/checkpoints}"
RENDERER_HOST="${RENDERER_HOST:-127.0.0.1}"
RENDERER_BASE_PORT="${RENDERER_BASE_PORT:-8100}"
RENDERER_GPUS="${RENDERER_GPUS:-0,1,2,3,4,5,6,7}"
ADJUST_POSE="${ADJUST_POSE:-1}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${NCAP_ROOT}/outputs/unidrivevla_iter17580_full_neurad_${TIMESTAMP}}"
LOG_ROOT="${LOG_ROOT:-${NCAP_ROOT}/logs/unidrivevla_iter17580_full_neurad_${TIMESTAMP}}"
MASTER_LOG="${LOG_ROOT}/parallel_full_neurad_eval.log"
SUMMARY_LOG="${RESULT_ROOT}/aggregate_summary.txt"

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"
cd "${NCAP_ROOT}"

exec > >(tee -a "${MASTER_LOG}") 2>&1

export PYTHONUNBUFFERED=1

cache_torch_weight() {
  local filename="$1"
  local source_path="${NCAP_ROOT}/${filename}"
  local target_path="${TORCH_CACHE_DIR}/${filename}"
  if [[ -f "${source_path}" ]]; then
    mkdir -p "${TORCH_CACHE_DIR}"
    if [[ ! -f "${target_path}" ]] || [[ "$(stat -c %s "${source_path}")" != "$(stat -c %s "${target_path}")" ]]; then
      echo "Caching torch weight: ${source_path} -> ${target_path}"
      cp "${source_path}" "${target_path}"
    fi
  fi
}

cache_torch_weight "vgg19-dcbb9e9d.pth"
cache_torch_weight "alexnet-owt-7be5be79.pth"

IFS=',' read -r -a model_ports <<< "${MODEL_PORTS}"
IFS=',' read -r -a renderer_gpus <<< "${RENDERER_GPUS}"

tasks=(
  "stationary 0099"
  "stationary 0101"
  "stationary 0103"
  "stationary 0106"
  "stationary 0108"
  "stationary 0278"
  "stationary 0331"
  "stationary 0783"
  "stationary 0796"
  "stationary 0966"
  "frontal 0103"
  "frontal 0106"
  "frontal 0110"
  "frontal 0346"
  "frontal 0923"
  "side 0103"
  "side 0108"
  "side 0110"
  "side 0278"
  "side 0921"
)

if [[ -n "${SELECTED_TASK_RUNS}" ]]; then
  tasks=()
  IFS=, read -r -a selected_specs <<< "${SELECTED_TASK_RUNS}"
  for spec in "${selected_specs[@]}"; do
    IFS=: read -r scenario sequence run_index extra <<< "${spec}"
    if [[ -n "${extra:-}" || -z "${scenario}" || ! "${sequence}" =~ ^[0-9]{4}$ || ! "${run_index}" =~ ^[0-9]+$ ]]; then
      echo "Invalid SELECTED_TASK_RUNS item: ${spec}; expected category:NNNN:run_index" >&2
      exit 2
    fi
    tasks+=("${scenario} ${sequence} ${run_index}")
  done
else
  for idx in "${!tasks[@]}"; do
    tasks[idx]="${tasks[idx]} all"
  done
fi

check_paths() {
  if [[ ! -f "${NCAP_ROOT}/main.py" ]]; then
    echo "NeuroNCAP main.py not found: ${NCAP_ROOT}/main.py" >&2
    exit 1
  fi
  if [[ ! -f "${NEURAD_ROOT}/nerfstudio/scripts/closed_loop/main.py" ]]; then
    echo "NeuRAD closed-loop renderer not found: ${NEURAD_ROOT}/nerfstudio/scripts/closed_loop/main.py" >&2
    exit 1
  fi
  if [[ ! -d "${NUSCENES_ROOT}/v1.0-trainval" ]]; then
    echo "nuScenes v1.0-trainval metadata not found: ${NUSCENES_ROOT}/v1.0-trainval" >&2
    exit 1
  fi
  if [[ ! -d "${NEURAD_CHECKPOINT_ROOT}" ]]; then
    echo "NeuRAD checkpoint root not found: ${NEURAD_CHECKPOINT_ROOT}" >&2
    echo "Download first: cd ${NCAP_ROOT} && bash scripts/download/download_neurad_weights.sh" >&2
    exit 1
  fi

  mkdir -p "${NEURAD_ROOT}/data"
  if [[ ! -e "${NEURAD_ROOT}/data/nuscenes" ]]; then
    ln -s "${NUSCENES_ROOT}" "${NEURAD_ROOT}/data/nuscenes"
  fi
}

check_neurad_python() {
  (
    cd "${NEURAD_ROOT}"
    export PYTHONPATH="${NEURAD_ROOT}:${PYTHONPATH:-}"
    "${NEURAD_PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ["uvicorn", "fastapi", "jaxtyping", "viser", "tyro", "PIL", "torch", "nerfstudio"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "NeuRAD renderer Python is missing packages: "
        + ", ".join(missing)
        + "\nInstall them in the B-server environment, for example: pip install uvicorn fastapi"
    )
print("NeuRAD renderer Python OK:", sys.executable)
PY
  )
}

resolve_neurad_config() {
  local sequence="$1"
  local candidates=(
    "${NEURAD_CHECKPOINT_ROOT}/${sequence}/config.yml"
    "${NEURAD_CHECKPOINT_ROOT}/scene-${sequence}/config.yml"
    "${NEURAD_CHECKPOINT_ROOT}/${sequence}/config.yaml"
    "${NEURAD_CHECKPOINT_ROOT}/scene-${sequence}/config.yaml"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "No NeuRAD config found for sequence ${sequence}. Tried:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  return 1
}

check_model_servers() {
  for port in "${model_ports[@]}"; do
    "${PYTHON_BIN}" - <<PY
import httpx
base = "http://${MODEL_HOST}:${port}"
alive = httpx.get(base + "/alive", timeout=30.0)
print("${port} /alive", alive.status_code, alive.text)
alive.raise_for_status()
reset = httpx.post(base + "/reset", json={}, timeout=30.0)
print("${port} /reset", reset.status_code, reset.text)
reset.raise_for_status()
PY
  done
}

wait_renderer() {
  local port="$1"
  local renderer_pid="$2"
  local renderer_log="$3"
  "${PYTHON_BIN}" - <<PY
import time
import httpx
import os
from pathlib import Path

url = "http://${RENDERER_HOST}:${port}/alive"
renderer_pid = int("${renderer_pid}")
renderer_log = Path("${renderer_log}")
last_error = None
for _ in range(450):
    try:
        os.kill(renderer_pid, 0)
    except ProcessLookupError:
        tail = ""
        if renderer_log.exists():
            tail = "\n".join(renderer_log.read_text(errors="replace").splitlines()[-80:])
        raise SystemExit(f"Renderer process exited before ready: pid={renderer_pid}, port=${port}\n{tail}")
    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            print("renderer ready", url, response.text)
            raise SystemExit(0)
        last_error = RuntimeError(f"status={response.status_code}, body={response.text}")
    except Exception as exc:
        last_error = exc
    time.sleep(2)
raise SystemExit(f"Renderer did not become ready: {url}; last error: {last_error}")
PY
}

assert_renderer_port_free() {
  local port="$1"
  "${PYTHON_BIN}" - <<PY
import httpx

url = "http://${RENDERER_HOST}:${port}/alive"
try:
    response = httpx.get(url, timeout=1.0)
except Exception:
    raise SystemExit(0)
raise SystemExit(
    f"Renderer port already has a live service before launch: {url}; "
    "refusing to reuse it because it may be a previous scene. "
    "Stop the old local NeuRAD renderer on the NeuroNCAP machine, then rerun."
)
PY
}

wait_renderer_port_closed() {
  local port="$1"
  "${PYTHON_BIN}" - <<PY
import time
import httpx

url = "http://${RENDERER_HOST}:${port}/alive"
last_status = None
for _ in range(60):
    try:
        response = httpx.get(url, timeout=1.0)
        last_status = f"status={response.status_code}, body={response.text}"
    except Exception:
        raise SystemExit(0)
    time.sleep(1)
raise SystemExit(f"Renderer port did not close after cleanup: {url}; last response: {last_status}")
PY
}

run_task() {
  local scenario="$1"
  local sequence="$2"
  local run_index="$3"
  local model_port="$4"
  local renderer_port="$5"
  local renderer_gpu="$6"
  local output_dir="${RESULT_ROOT}/${scenario}-${sequence}"
  local run_suffix=""
  if [[ "${run_index}" != "all" ]]; then
    run_suffix="_run${run_index}"
  fi
  local log_file="${LOG_ROOT}/${scenario}_${sequence}${run_suffix}_model${model_port}_renderer${renderer_port}.log"
  local renderer_log="${LOG_ROOT}/${scenario}_${sequence}_renderer${renderer_port}.log"
  local neurad_config
  local renderer_pid

  neurad_config="$(resolve_neurad_config "${sequence}")"
  mkdir -p "${output_dir}"

  {
    echo "[$(date '+%F %T')] Running ${scenario}-${sequence}"
    if [[ "${run_index}" == "all" ]]; then
      echo "  runs:             0..$((RUNS - 1))"
    else
      echo "  run:              ${run_index}"
    fi
    echo "  output dir:       ${output_dir}"
    echo "  model:            ${MODEL_HOST}:${model_port}"
    echo "  renderer:         ${RENDERER_HOST}:${renderer_port}"
    echo "  renderer gpu:     ${renderer_gpu}"
    echo "  neurad config:    ${neurad_config}"
    echo "  renderer log:     ${renderer_log}"

    assert_renderer_port_free "${renderer_port}"

    (
      cd "${NEURAD_ROOT}"
      export PYTHONPATH="${NEURAD_ROOT}:${PYTHONPATH:-}"
      export CUDA_VISIBLE_DEVICES="${renderer_gpu}"
      args=(
        -u nerfstudio/scripts/closed_loop/main.py
        --host "${RENDERER_HOST}"
        --port "${renderer_port}"
        --load-config "${neurad_config}"
      )
      if [[ "${ADJUST_POSE}" == "1" ]]; then
        args+=(--adjust-pose)
      fi
      exec "${NEURAD_PYTHON_BIN}" "${args[@]}"
    ) >> "${renderer_log}" 2>&1 &
    renderer_pid="$!"

    cleanup() {
      if kill -0 "${renderer_pid}" >/dev/null 2>&1; then
        kill "${renderer_pid}" >/dev/null 2>&1 || true
        wait "${renderer_pid}" >/dev/null 2>&1 || true
      fi
      wait_renderer_port_closed "${renderer_port}" || true
    }
    trap cleanup RETURN

    wait_renderer "${renderer_port}" "${renderer_pid}" "${renderer_log}"

    "${PYTHON_BIN}" - <<PY
import httpx
url = "http://${MODEL_HOST}:${model_port}/reset"
response = httpx.post(url, json={}, timeout=30.0)
print("POST", url, "status:", response.status_code, "body:", response.text)
response.raise_for_status()
PY

    main_args=(
      main.py
      --engine.renderer.host "${RENDERER_HOST}"
      --engine.renderer.port "${renderer_port}"
      --engine.model.host "${MODEL_HOST}"
      --engine.model.port "${model_port}"
      --scenario-category "${scenario}"
      --engine.dataset.data-root "${NUSCENES_ROOT}"
      --engine.dataset.version v1.0-trainval
      --engine.dataset.sequence "${sequence}"
      --engine.logger.log-dir "${output_dir}"
      --runs "${RUNS}"
    )
    if [[ "${run_index}" != "all" ]]; then
      main_args+=(--run-indices "${run_index}")
    fi
    if [[ "${RESUME}" == "1" ]]; then
      main_args+=(--resume)
    fi
    "${PYTHON_BIN}" "${main_args[@]}"

    echo "[$(date '+%F %T')] Finished ${scenario}-${sequence}"
  } >> "${log_file}" 2>&1
}

echo "[$(date '+%F %T')] Starting full NeuroNCAP NeuRAD eval for UniDriveVLA iter_17580"
echo "  ncap root:          ${NCAP_ROOT}"
echo "  neurad root:        ${NEURAD_ROOT}"
echo "  neurad checkpoints: ${NEURAD_CHECKPOINT_ROOT}"
echo "  nuscenes root:      ${NUSCENES_ROOT}"
echo "  model host:         ${MODEL_HOST}"
echo "  model ports:        ${MODEL_PORTS}"
echo "  renderer base port: ${RENDERER_BASE_PORT}"
echo "  renderer gpus:      ${RENDERER_GPUS}"
echo "  runs:               ${RUNS}"
echo "  resume:             ${RESUME}"
echo "  start task index:   ${START_TASK_INDEX}"
echo "  scenarios:          ${#tasks[@]}"
if [[ "${SELECTED_TASK_RUNS}" != "" ]]; then
  echo "  selected task runs:  ${SELECTED_TASK_RUNS}"
  echo "  total runs:         $((${#tasks[@]} - START_TASK_INDEX))"
else
  echo "  total runs:         $(((${#tasks[@]} - START_TASK_INDEX) * RUNS))"
fi
echo "  result root:        ${RESULT_ROOT}"
echo "  log root:           ${LOG_ROOT}"
echo

check_paths
check_neurad_python
check_model_servers

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1, got: ${RESUME}" >&2
  exit 1
fi

parallel_count="${#model_ports[@]}"
if [[ "${#renderer_gpus[@]}" -lt "${parallel_count}" ]]; then
  parallel_count="${#renderer_gpus[@]}"
fi
if [[ "${parallel_count}" -lt 1 ]]; then
  echo "No model ports or renderer GPUs configured." >&2
  exit 1
fi

task_count="${#tasks[@]}"
if ! [[ "${START_TASK_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "START_TASK_INDEX must be a non-negative integer, got: ${START_TASK_INDEX}" >&2
  exit 1
fi
if [[ "${START_TASK_INDEX}" -gt "${task_count}" ]]; then
  echo "START_TASK_INDEX ${START_TASK_INDEX} is greater than task count ${task_count}." >&2
  exit 1
fi
task_idx="${START_TASK_INDEX}"

while [[ "${task_idx}" -lt "${task_count}" ]]; do
  pids=()
  batch_items=()
  for slot in $(seq 0 $((parallel_count - 1))); do
    if [[ "${task_idx}" -ge "${task_count}" ]]; then
      break
    fi
    read -r scenario sequence run_index <<< "${tasks[task_idx]}"
    model_port="${model_ports[slot]}"
    renderer_gpu="${renderer_gpus[slot]}"
    renderer_port="$((RENDERER_BASE_PORT + slot))"
    echo "Launching ${scenario}-${sequence}: model ${model_port}, renderer ${renderer_port}, gpu ${renderer_gpu}"
    run_task "${scenario}" "${sequence}" "${run_index}" "${model_port}" "${renderer_port}" "${renderer_gpu}" &
    pids+=("$!")
    batch_items+=("${scenario}-${sequence}-run${run_index}@model${model_port}/renderer${renderer_port}")
    task_idx=$((task_idx + 1))
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[idx]}"; then
      echo "Task failed: ${batch_items[idx]}" >&2
      exit 1
    fi
    echo "Task completed: ${batch_items[idx]}"
  done
done

echo
echo "[$(date '+%F %T')] All scenarios finished. Aggregating..."
"${PYTHON_BIN}" scripts/aggregate_results.py "${RESULT_ROOT}" 2>&1 | tee "${SUMMARY_LOG}"

echo
echo "[$(date '+%F %T')] Full NeuroNCAP NeuRAD eval finished"
echo "  result root: ${RESULT_ROOT}"
echo "  summary:     ${SUMMARY_LOG}"
