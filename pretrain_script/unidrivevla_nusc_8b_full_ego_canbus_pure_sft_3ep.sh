#!/usr/bin/env bash
set -euo pipefail

# Full-parameter VLM SFT on the nuScenes can-bus planning QA dataset.
# This is independent of Stage2 det/map/ego/planning training.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
SWIFT_ROOT=${SWIFT_ROOT:-/path/to/ms-swift-main}
MODEL_PATH=${MODEL_PATH:-/path/to/UniDriveVLA_Nusc_Large_Stage1}
EGO_CANBUS_TRAIN_JSONL=${EGO_CANBUS_TRAIN_JSONL:-/path/to/nuscenes_planning_qa_train_only_command_his_traj_vel_acc_can_bus.jsonl}
EGO_CANBUS_VAL_JSONL=${EGO_CANBUS_VAL_JSONL:-/path/to/nuscenes_planning_qa_val_only_command_his_traj_vel_acc_can_bus.jsonl}
SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/nuScenes/work_dirs2/unidrivevla_stage1_8b_full_ego_canbus_pure_ms_swift_3ep}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}/logs/pure_ms_swift_full_ego_canbus_8b_3ep.$(date +%m%d%H%M).log}
LATEST_PATH_FILE=${LATEST_PATH_FILE:-${SAVE_DIR}/latest_hf_model_path.txt}
ROOT_IMAGE_DIR=${ROOT_IMAGE_DIR:-$(dirname "${REPO_ROOT}")}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export PYTHONPATH=${SWIFT_ROOT}:${PYTHONPATH:-}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${PURE_SFT_MASTER_PORT:-29540}

PURE_SFT_PER_DEVICE_BATCH_SIZE=${PURE_SFT_PER_DEVICE_BATCH_SIZE:-1}
PURE_SFT_GRAD_ACCUM_STEPS=${PURE_SFT_GRAD_ACCUM_STEPS:-8}
PURE_SFT_GRADIENT_CHECKPOINTING=${PURE_SFT_GRADIENT_CHECKPOINTING:-true}
PURE_SFT_VIT_GRADIENT_CHECKPOINTING=${PURE_SFT_VIT_GRADIENT_CHECKPOINTING:-true}
PURE_SFT_SAVE_STEPS=${PURE_SFT_SAVE_STEPS:-50}
PURE_SFT_LOGGING_STEPS=${PURE_SFT_LOGGING_STEPS:-1}
PURE_SFT_MAX_PIXELS=${PURE_SFT_MAX_PIXELS:-522240}
PURE_SFT_IMAGE_MAX_TOKEN_NUM=${PURE_SFT_IMAGE_MAX_TOKEN_NUM:-506}
PURE_SFT_DEEPSPEED=${PURE_SFT_DEEPSPEED:-zero2}
PURE_SFT_ATTN_IMPL=${PURE_SFT_ATTN_IMPL:-sdpa}
PURE_SFT_DATASET_NUM_PROC=${PURE_SFT_DATASET_NUM_PROC:-16}
PURE_SFT_DATALOADER_NUM_WORKERS=${PURE_SFT_DATALOADER_NUM_WORKERS:-16}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}

for required_file in \
    "${MODEL_PATH}/config.json" \
    "${EGO_CANBUS_TRAIN_JSONL}" \
    "${REPO_ROOT}/pretrain_script/tokens.txt" \
    "${REPO_ROOT}/pretrain_script/system.txt"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -f "${MODEL_PATH}/model.safetensors" && ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
    echo "MODEL_PATH is not a complete HF safetensors checkpoint: ${MODEL_PATH}" >&2
    exit 1
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

if ! "${PYTHON_BIN}" -c 'import peft, swift' >/dev/null 2>&1; then
    echo "The selected Python environment must provide peft and ms-swift: ${PYTHON_BIN}" >&2
    exit 1
fi

VAL_ARGS=()
if [[ -f "${EGO_CANBUS_VAL_JSONL}" ]]; then
    VAL_ARGS=(--val_dataset "${EGO_CANBUS_VAL_JSONL}")
fi

mkdir -p "${SAVE_DIR}/logs"

export ROOT_IMAGE_DIR
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-torchvision}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}

echo "Launching Qwen3-VL-8B full can-bus SFT"
echo "  repository: ${REPO_ROOT}"
echo "  initialization: ${MODEL_PATH}"
echo "  train jsonl: ${EGO_CANBUS_TRAIN_JSONL}"
echo "  output: ${SAVE_DIR}"
echo "  schedule: 3 epochs on ${NPROC_PER_NODE} GPU(s)"
echo "  batch: per-device=${PURE_SFT_PER_DEVICE_BATCH_SIZE}, accumulation=${PURE_SFT_GRAD_ACCUM_STEPS}"
echo "  effective batch: $((NPROC_PER_NODE * NNODES * PURE_SFT_PER_DEVICE_BATCH_SIZE * PURE_SFT_GRAD_ACCUM_STEPS))"
echo "  gradient checkpointing: LLM=${PURE_SFT_GRADIENT_CHECKPOINTING}, ViT=${PURE_SFT_VIT_GRADIENT_CHECKPOINTING}"
echo "  attention implementation: ${PURE_SFT_ATTN_IMPL}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-14} \
SWIFT_PATCH_CONV3D=1 \
IMAGE_MAX_TOKEN_NUM=${PURE_SFT_IMAGE_MAX_TOKEN_NUM} \
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
    --attn_impl "${PURE_SFT_ATTN_IMPL}" \
    --max_pixels "${PURE_SFT_MAX_PIXELS}" \
    --packing false \
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
    --gradient_checkpointing "${PURE_SFT_GRADIENT_CHECKPOINTING}" \
    --vit_gradient_checkpointing "${PURE_SFT_VIT_GRADIENT_CHECKPOINTING}" \
    --max_length 8192 \
    --dataloader_num_workers "${PURE_SFT_DATALOADER_NUM_WORKERS}" \
    --dataset_num_proc "${PURE_SFT_DATASET_NUM_PROC}" \
    --deepspeed "${PURE_SFT_DEEPSPEED}" \
    --use_logits_to_keep true \
    2>&1 | tee "${LOG_FILE}"

PURE_SFT_VLM_PATH=$(find "${SAVE_DIR}" -type f \( -name model.safetensors -o -name model.safetensors.index.json \) -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [[ -z "${PURE_SFT_VLM_PATH}" ]]; then
    echo "SFT finished, but no HF safetensors checkpoint was found under ${SAVE_DIR}." >&2
    exit 1
fi

echo "${PURE_SFT_VLM_PATH}" > "${LATEST_PATH_FILE}"
echo "8B can-bus SFT checkpoint: ${PURE_SFT_VLM_PATH}"
echo "Wrote latest checkpoint path to: ${LATEST_PATH_FILE}"
