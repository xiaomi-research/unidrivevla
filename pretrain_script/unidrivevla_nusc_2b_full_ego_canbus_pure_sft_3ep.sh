#!/usr/bin/env bash
set -euo pipefail

# Pure VLM SFT on the full-canbus nuScenes QA jsonl with ordinary ms-swift
# transformers backend. This does not use Megatron-SWIFT, ARCotrainingHook,
# or any Stage2 det/map/ego/planning losses.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
SWIFT_ROOT=${SWIFT_ROOT:-""}

MODEL_PATH=${MODEL_PATH:-"/path/to/UniDriveVLA_Stage1_Nuscenes_2B"}
EGO_CANBUS_TRAIN_JSONL=${EGO_CANBUS_TRAIN_JSONL:-"/path/to/nuscenes_planning_qa_train_only_command_his_traj_vel_acc_can_bus.jsonl"}
EGO_CANBUS_VAL_JSONL=${EGO_CANBUS_VAL_JSONL:-"/path/to/nuscenes_planning_qa_val_only_command_his_traj_vel_acc_can_bus.jsonl"}
SAVE_DIR=${SAVE_DIR:-"${REPO_ROOT}/nuScenes/work_dirs2/unidrivevla_stage1_2b_full_ego_canbus_pure_ms_swift_3ep"}
LOG_FILE=${LOG_FILE:-"${SAVE_DIR}/logs/pure_ms_swift_full_ego_canbus_3ep.$(date +%m%d%H%M).log"}
LATEST_PATH_FILE=${LATEST_PATH_FILE:-"${SAVE_DIR}/latest_hf_model_path.txt"}
ROOT_IMAGE_DIR=${ROOT_IMAGE_DIR:-$(cd "${REPO_ROOT}/.." && pwd)}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}

ENV_ACTIVATE=${ENV_ACTIVATE:-""}
PYTHON_BIN=${PYTHON_BIN:-""}

if [[ -n "${ENV_ACTIVATE}" ]]; then
    if [[ ! -f "${ENV_ACTIVATE}" ]]; then
        echo "ENV_ACTIVATE does not exist: ${ENV_ACTIVATE}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${ENV_ACTIVATE}"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN=$(command -v python)
fi

if [[ -n "${SWIFT_ROOT}" ]]; then
    export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH:-}"
fi

if [[ "${PYTHON_BIN}" == */* ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
        exit 1
    fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "PYTHON_BIN was not found on PATH: ${PYTHON_BIN}" >&2
    exit 1
fi


if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing MODEL_PATH: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/model.safetensors" && ! -f "${MODEL_PATH}/model.safetensors.index.json" && ! -f "${MODEL_PATH}/pytorch_model.bin.index.json" ]]; then
    echo "Warning: MODEL_PATH may not be a complete HF checkpoint: ${MODEL_PATH}" >&2
fi

if [[ ! -f "${EGO_CANBUS_TRAIN_JSONL}" ]]; then
    echo "Missing EGO_CANBUS_TRAIN_JSONL: ${EGO_CANBUS_TRAIN_JSONL}" >&2
    exit 1
fi

for required_file in \
    "${REPO_ROOT}/pretrain_script/tokens.txt" \
    "${REPO_ROOT}/pretrain_script/system.txt"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required release file: ${required_file}" >&2
        exit 1
    fi
done

VAL_ARGS=()
if [[ -f "${EGO_CANBUS_VAL_JSONL}" ]]; then
    VAL_ARGS=(--val_dataset "${EGO_CANBUS_VAL_JSONL}")
fi

mkdir -p "${SAVE_DIR}/logs"

export ROOT_IMAGE_DIR
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}

NPROC_PER_NODE=${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
MASTER_PORT=${PURE_SFT_MASTER_PORT:-${MLP_WORKER_0_PORT:-29520}}

SWIFT_INFO=$("${PYTHON_BIN}" - <<'PYINFO'
import swift
print(f"{getattr(swift, '__version__', 'unknown')} ({getattr(swift, '__file__', 'unknown')})")
PYINFO
)

echo "Launching pure full-ego canbus SFT"
echo "  python: ${PYTHON_BIN}"
echo "  swift: ${SWIFT_INFO}"
echo "  model: ${MODEL_PATH}"
echo "  train jsonl: ${EGO_CANBUS_TRAIN_JSONL}"
echo "  val jsonl: ${EGO_CANBUS_VAL_JSONL}"
echo "  save dir: ${SAVE_DIR}"
echo "  log file: ${LOG_FILE}"

PURE_SFT_PACKING=${PURE_SFT_PACKING:-false}
PURE_SFT_PER_DEVICE_BATCH_SIZE=${PURE_SFT_PER_DEVICE_BATCH_SIZE:-1}
PURE_SFT_GRAD_ACCUM_STEPS=${PURE_SFT_GRAD_ACCUM_STEPS:-16}
PURE_SFT_SAVE_STEPS=${PURE_SFT_SAVE_STEPS:-50}
PURE_SFT_LOGGING_STEPS=${PURE_SFT_LOGGING_STEPS:-1}
PURE_SFT_MAX_PIXELS=${PURE_SFT_MAX_PIXELS:-522240}
PURE_SFT_IMAGE_MAX_TOKEN_NUM=${PURE_SFT_IMAGE_MAX_TOKEN_NUM:-506}
PURE_SFT_DEEPSPEED=${PURE_SFT_DEEPSPEED:-zero2}
PURE_SFT_DATASET_NUM_PROC=${PURE_SFT_DATASET_NUM_PROC:-16}
PURE_SFT_DATALOADER_NUM_WORKERS=${PURE_SFT_DATALOADER_NUM_WORKERS:-16}
PURE_SFT_USE_LOGITS_TO_KEEP=${PURE_SFT_USE_LOGITS_TO_KEEP:-true}
PURE_SFT_MAX_STEPS=${PURE_SFT_MAX_STEPS:-}

EXTRA_TRAIN_ARGS=()
if [[ -n "${PURE_SFT_USE_LOGITS_TO_KEEP}" ]]; then
    EXTRA_TRAIN_ARGS+=(--use_logits_to_keep "${PURE_SFT_USE_LOGITS_TO_KEEP}")
fi
if [[ -n "${PURE_SFT_MAX_STEPS}" ]]; then
    EXTRA_TRAIN_ARGS+=(--max_steps "${PURE_SFT_MAX_STEPS}")
fi

echo "  backend: ordinary ms-swift transformers"
echo "  deepspeed: ${PURE_SFT_DEEPSPEED}"
echo "  per-device batch: ${PURE_SFT_PER_DEVICE_BATCH_SIZE}"
echo "  grad accumulation: ${PURE_SFT_GRAD_ACCUM_STEPS}"
echo "  effective batch: $((NPROC_PER_NODE * NNODES * PURE_SFT_PER_DEVICE_BATCH_SIZE * PURE_SFT_GRAD_ACCUM_STEPS))"
if [[ -n "${PURE_SFT_USE_LOGITS_TO_KEEP}" ]]; then
    echo "  use logits to keep: ${PURE_SFT_USE_LOGITS_TO_KEEP}"
fi
if [[ -n "${PURE_SFT_MAX_STEPS}" ]]; then
    echo "  max steps: ${PURE_SFT_MAX_STEPS}"
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-14} \
SWIFT_PATCH_CONV3D=1 \
IMAGE_MAX_TOKEN_NUM="${PURE_SFT_IMAGE_MAX_TOKEN_NUM}" \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --module swift.cli.sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl \
    --train_type full \
    --dataset "${EGO_CANBUS_TRAIN_JSONL}#28130" \
    "${VAL_ARGS[@]}" \
    --load_from_cache_file "${LOAD_FROM_CACHE_FILE}" \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --max_pixels "${PURE_SFT_MAX_PIXELS}" \
    --packing "${PURE_SFT_PACKING}" \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size "${PURE_SFT_PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${PURE_SFT_GRAD_ACCUM_STEPS}" \
    --learning_rate 4e-5 \
    --warmup_ratio 0.05 \
    --save_safetensors true \
    --new_special_tokens "${REPO_ROOT}/pretrain_script/tokens.txt" \
    --system "${REPO_ROOT}/pretrain_script/system.txt" \
    --output_dir "${SAVE_DIR}" \
    --add_version false \
    --save_steps "${PURE_SFT_SAVE_STEPS}" \
    --save_total_limit 1 \
    --save_only_model true \
    --logging_steps "${PURE_SFT_LOGGING_STEPS}" \
    --eval_strategy no \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
    --max_length 8192 \
    --dataloader_num_workers "${PURE_SFT_DATALOADER_NUM_WORKERS}" \
    --dataset_num_proc "${PURE_SFT_DATASET_NUM_PROC}" \
    --deepspeed "${PURE_SFT_DEEPSPEED}" \
    "${EXTRA_TRAIN_ARGS[@]}" > "${LOG_FILE}" 2>&1

PURE_SFT_VLM_PATH=$(find "${SAVE_DIR}" -type f \( -name "model.safetensors" -o -name "model.safetensors.index.json" \) -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [[ -z "${PURE_SFT_VLM_PATH}" ]]; then
    echo "Pure SFT finished, but no HF safetensors checkpoint was found under ${SAVE_DIR}." >&2
    echo "Stage2 expects a directory with model.safetensors/model.safetensors.index.json and tokenizer files." >&2
    exit 1
fi

if [[ ! -f "${PURE_SFT_VLM_PATH}/tokenizer_config.json" && ! -f "${PURE_SFT_VLM_PATH}/tokenizer.json" ]]; then
    echo "Pure SFT checkpoint is missing tokenizer files: ${PURE_SFT_VLM_PATH}" >&2
    echo "Stage2 calls AutoProcessor.from_pretrained() on this directory." >&2
    exit 1
fi

if [[ ! -f "${PURE_SFT_VLM_PATH}/processor_config.json" && ! -f "${PURE_SFT_VLM_PATH}/preprocessor_config.json" ]]; then
    echo "Warning: pure SFT checkpoint has no explicit processor/preprocessor config: ${PURE_SFT_VLM_PATH}" >&2
    echo "If Stage2 AutoProcessor loading fails, copy processor files from MODEL_PATH or export this checkpoint to HF format." >&2
fi

echo "${PURE_SFT_VLM_PATH}" > "${LATEST_PATH_FILE}"
echo "Pure SFT HF checkpoint: ${PURE_SFT_VLM_PATH}"
echo "Wrote latest checkpoint path to: ${LATEST_PATH_FILE}"
