#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${NUSCENES_ROOT:-${SCRIPT_DIR}}
DATA_CWD=${NUSCENES_DATA_CWD:-${CODE_ROOT}}
cd "${DATA_CWD}"
export NUSCENES_ROOT="${CODE_ROOT}"

CFG=${CFG:-${CODE_ROOT}/projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage2.py}
EXP_NAME=${EXP_NAME:-unidrivevla_stage2_2b_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs4_accum4}
WORK_DIR=${WORK_DIR:-${CODE_ROOT}/work_dirs2/UniDriveVLA/${EXP_NAME}}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-${WORK_DIR}/final_eval}
LOG_DIR=${WORK_DIR}/logs
mkdir -p "${LOG_DIR}"

T=$(date +%m%d%H%M)
LOG_FILE=${LOG_DIR}/train.${T}.node${MLP_ROLE_INDEX:-0}.log
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NUM_GPUS=${NUM_GPUS:-8}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export UNIDRIVE_WASP_SVD_CACHE_DIR=${UNIDRIVE_WASP_SVD_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/unidrivevla/wasp_svd}
export UNIDRIVE_DISABLE_COMPILED_FLEX=${UNIDRIVE_DISABLE_COMPILED_FLEX:-1}
# Large-memory training nodes default to the faster visual forward/backward.
# Set this to 1 when activation memory is more important than iteration speed.
export UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING=${UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING:-0}
case "${UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|on) VISUAL_GC_CFG=True ;;
  0|false|no|off) VISUAL_GC_CFG=False ;;
  *) echo "ERROR: UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING must be 0 or 1." >&2; exit 2 ;;
esac

export NCCL_DEBUG=${NCCL_DEBUG:-ERROR}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}

export MMCV_ROOT=${MMCV_ROOT:-${CODE_ROOT}/../third_party/mmcv-1.7.2}
export MMDET3D_ROOT=${MMDET3D_ROOT:-${CODE_ROOT}/../third_party/mmdetection3d-1.0.0rc6}
export PYTHONPATH=${MMCV_ROOT}:${MMDET3D_ROOT}:${CODE_ROOT}:${PYTHONPATH:-}

export OCCWORLD_VAE_PATH=${OCCWORLD_VAE_PATH:-/path/to/occvae_latest.pth}
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${CODE_ROOT}/zero_configs/adam_zero1_bf16.json}
export DATA_INFOS_ROOT=${DATA_INFOS_ROOT:-${DATA_CWD}/data/infos}
export NUSCENES_DATA_ROOT=${NUSCENES_DATA_ROOT:-${DATA_CWD}/data/nuscenes}
export NUSCENES_ANCHOR_ROOT=${NUSCENES_ANCHOR_ROOT:-${DATA_CWD}/data/kmeans}
export VLM_PRETRAINED_PATH=${VLM_PRETRAINED_PATH:-/path/to/UniDriveVLA_Stage1_Nuscenes_2B}
export ENABLE_COC=${ENABLE_COC:-1}
export COC_JSONL_PATH=${COC_JSONL_PATH:-${CODE_ROOT}/data/nuscenes/alpamayo_coc_train_with_nav.jsonl}
export COC_PLANNING_LOSS_WEIGHT=${COC_PLANNING_LOSS_WEIGHT:-0.0}
export COC_CONTEXT_DROPOUT=${COC_CONTEXT_DROPOUT:-0.0}
export COC_LANGUAGE_LOSS_WEIGHT=${COC_LANGUAGE_LOSS_WEIGHT:-0.1}
export COC_LANGUAGE_TARGET_FORMAT=${COC_LANGUAGE_TARGET_FORMAT:-cot}
export ENABLE_COC_GENERATE=${ENABLE_COC_GENERATE:-0}
export DRIVING_PREFIX_MAX_LEN=${DRIVING_PREFIX_MAX_LEN:-3840}
export VLA_ATTN_IMPLEMENTATION=${VLA_ATTN_IMPLEMENTATION:-sdpa}

for anchor_file in kmeans_det_900.npy kmeans_map_100.npy kmeans_motion_6.npy; do
  if [[ ! -f "${NUSCENES_ANCHOR_ROOT}/${anchor_file}" ]]; then
    echo "Missing nuScenes anchor: ${NUSCENES_ANCHOR_ROOT}/${anchor_file}" >&2
    echo "Generate the external k-means assets or set NUSCENES_ANCHOR_ROOT." >&2
    exit 1
  fi
done

MASTER_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:-28691}}
EVAL_PORT=${EVAL_PORT:-28693}
NNODES=${MLP_WORKER_NUM:-1}
NPROC_PER_NODE=${MLP_WORKER_GPU:-${NUM_GPUS}}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
RUN_FINAL_EVAL=${RUN_FINAL_EVAL:-0}
RESUME_FROM=${RESUME_FROM:-}

normalize_bool_cfg() {
  local name="$1"
  local value="$2"
  case "${value,,}" in
    1|true|yes|on) printf 'True' ;;
    0|false|no|off) printf 'False' ;;
    *) echo "ERROR: ${name} must be 0 or 1, got ${value}." >&2; return 2 ;;
  esac
}

EVAL_WITH_DET_CFG=$(normalize_bool_cfg EVAL_WITH_DET "${EVAL_WITH_DET:-1}")
EVAL_WITH_TRACKING_CFG=$(normalize_bool_cfg EVAL_WITH_TRACKING "${EVAL_WITH_TRACKING:-1}")
EVAL_WITH_MAP_CFG=$(normalize_bool_cfg EVAL_WITH_MAP "${EVAL_WITH_MAP:-1}")
EVAL_WITH_MOTION_CFG=$(normalize_bool_cfg EVAL_WITH_MOTION "${EVAL_WITH_MOTION:-0}")
EVAL_WITH_PLANNING_CFG=$(normalize_bool_cfg EVAL_WITH_PLANNING "${EVAL_WITH_PLANNING:-1}")
EVAL_WITH_EGO_STATUS_CFG=$(normalize_bool_cfg EVAL_WITH_EGO_STATUS "${EVAL_WITH_EGO_STATUS:-1}")

find_latest_ds_ckpt_dir() {
  local root="$1"
  local latest_link="${root%/}/latest"
  local latest_target=""

  if [[ -L "${latest_link}" ]]; then
    latest_target=$(readlink -f "${latest_link}")
    if [[ -d "${latest_target}" ]]; then
      echo "${latest_target}"
      return
    fi
  elif [[ -f "${latest_link}" ]]; then
    local latest_tag
    latest_tag=$(cat "${latest_link}")
    latest_target="${root%/}/${latest_tag}"
    if [[ -d "${latest_target}" ]]; then
      echo "${latest_target}"
      return
    fi
  fi

  find "${root}" -maxdepth 1 -type d -name "iter_*" 2>/dev/null | sort -V | tail -1
}

echo "UniDriveVLA Stage 2 only: VLM & Planning & Det & Map & Ego with COC language"
echo "Config: ${CFG}"
echo "Work dir: ${WORK_DIR}"
echo "Log file: ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "NNODES: ${NNODES}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "NODE_RANK: ${NODE_RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "EVAL_PORT: ${EVAL_PORT}"
echo "RUN_FINAL_EVAL: ${RUN_FINAL_EVAL}"
echo "Final eval tasks: det=${EVAL_WITH_DET_CFG}, tracking=${EVAL_WITH_TRACKING_CFG}, map=${EVAL_WITH_MAP_CFG}, motion=${EVAL_WITH_MOTION_CFG}, planning=${EVAL_WITH_PLANNING_CFG}, ego_status=${EVAL_WITH_EGO_STATUS_CFG}"
echo "UNIDRIVE_DISABLE_COMPILED_FLEX: ${UNIDRIVE_DISABLE_COMPILED_FLEX}"
echo "UNIDRIVE_WASP_SVD_CACHE_DIR: ${UNIDRIVE_WASP_SVD_CACHE_DIR}"
echo "UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING: ${UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING} (${VISUAL_GC_CFG})"
echo "VLM_PRETRAINED_PATH: ${VLM_PRETRAINED_PATH}"
echo "ENABLE_COC: ${ENABLE_COC}"
echo "COC_JSONL_PATH: ${COC_JSONL_PATH}"
echo "COC_CONTEXT_DROPOUT: ${COC_CONTEXT_DROPOUT}"
echo "COC_PLANNING_LOSS_WEIGHT: ${COC_PLANNING_LOSS_WEIGHT}"
echo "COC_LANGUAGE_LOSS_WEIGHT: ${COC_LANGUAGE_LOSS_WEIGHT}"
echo "COC_LANGUAGE_TARGET_FORMAT: ${COC_LANGUAGE_TARGET_FORMAT}"
echo "ENABLE_COC_GENERATE: ${ENABLE_COC_GENERATE}"
echo "DRIVING_PREFIX_MAX_LEN: ${DRIVING_PREFIX_MAX_LEN}"
echo "VLA_ATTN_IMPLEMENTATION: ${VLA_ATTN_IMPLEMENTATION}"

for integer_var in NUM_GPUS NNODES NPROC_PER_NODE NODE_RANK; do
  if ! [[ "${!integer_var}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${integer_var} must be a non-negative integer, got ${!integer_var}." >&2
    exit 2
  fi
done
if (( NUM_GPUS < 1 || NNODES < 1 || NPROC_PER_NODE < 1 ||
      NNODES * NPROC_PER_NODE != NUM_GPUS || NODE_RANK >= NNODES )); then
  echo "ERROR: invalid distributed layout: NUM_GPUS=${NUM_GPUS}, NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, NODE_RANK=${NODE_RANK}." >&2
  exit 2
fi
if (( NNODES > 1 )) &&
   [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "::1" ]]; then
  echo "ERROR: multi-node training requires a node-0 address reachable from every node; got MASTER_ADDR=${MASTER_ADDR}." >&2
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python}
AVAILABLE_GPUS=$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")
echo "Detected CUDA devices: ${AVAILABLE_GPUS}"
if [[ "${AVAILABLE_GPUS}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE}, but only ${AVAILABLE_GPUS} local CUDA device(s) are visible."
  exit 2
fi

if [[ -z "${RESUME_FROM}" ]]; then
  RESUME_FROM=$(find_latest_ds_ckpt_dir "${WORK_DIR}")
fi

if [[ -n "${RESUME_FROM}" && ! -d "${RESUME_FROM}" ]]; then
  echo "ERROR: RESUME_FROM must be a DeepSpeed checkpoint directory such as iter_3516, got: ${RESUME_FROM}" >&2
  exit 3
fi

find_latest_mp_rank_ckpt() {
  local root="$1"
  find "${root}" -path "*/mp_rank_00_model_states.pt" -type f 2>/dev/null | sort -V | tail -1
}

TRAIN_EXTRA_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  echo "AutoResume: resuming from ${RESUME_FROM}"
  TRAIN_EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")
else
  echo "AutoResume: no checkpoint found under ${WORK_DIR}; starting from VLM_PRETRAINED_PATH."
fi

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${CODE_ROOT}/tools/train.py" \
  "${CFG}" \
  --launcher pytorch \
  --deterministic \
  --work-dir "${WORK_DIR}" \
  --no-validate \
  --cfg-options model.planning_head.attn_implementation="${VLA_ATTN_IMPLEMENTATION}" \
  model.planning_head.visual_gradient_checkpointing="${VISUAL_GC_CFG}" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  "$@"

STAGE2_CKPT=$(find_latest_mp_rank_ckpt "${WORK_DIR}")
if [ -z "${STAGE2_CKPT}" ]; then
  echo "ERROR: no Stage 2 checkpoint found under ${WORK_DIR}" >&2
  exit 1
fi

if [[ "${RUN_FINAL_EVAL}" != "1" ]]; then
  echo "Stage 2 training finished. Final eval skipped because RUN_FINAL_EVAL=${RUN_FINAL_EVAL}."
  echo "Latest checkpoint: ${STAGE2_CKPT}"
  exit 0
fi

mkdir -p "${EVAL_OUT_DIR}/logs"
EVAL_T=$(date +%Y%m%d_%H%M%S)
echo "Stage 2 training finished. Final eval checkpoint: ${STAGE2_CKPT}"
echo "Final eval out dir: ${EVAL_OUT_DIR}"
export ENABLE_COC_GENERATE=${ENABLE_COC_GENERATE_FOR_FINAL_EVAL:-1}
export COC_JSONL_PATH=${COC_VAL_JSONL_PATH:-${CODE_ROOT}/data/nuscenes/alpamayo_coc_val_with_nav.jsonl}
echo "Final eval ENABLE_COC_GENERATE: ${ENABLE_COC_GENERATE}"
echo "Final eval COC_JSONL_PATH: ${COC_JSONL_PATH}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${EVAL_PORT}" \
  "${CODE_ROOT}/tools/test.py" \
  "${CFG}" \
  "${STAGE2_CKPT}" \
  --launcher pytorch \
  --out "${EVAL_OUT_DIR}/results.pkl" \
  --eval bbox \
  --show-dir "${EVAL_OUT_DIR}" \
  --cfg-options \
    model.planning_head.attn_implementation="${VLA_ATTN_IMPLEMENTATION}" \
    model.planning_head.visual_gradient_checkpointing="${VISUAL_GC_CFG}" \
    model.planning_head.unified_decoder_cfg.with_instance_id="${EVAL_WITH_TRACKING_CFG}" \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=4 \
    evaluation.eval_mode.with_det="${EVAL_WITH_DET_CFG}" \
    evaluation.eval_mode.with_tracking="${EVAL_WITH_TRACKING_CFG}" \
    evaluation.eval_mode.with_map="${EVAL_WITH_MAP_CFG}" \
    evaluation.eval_mode.with_motion="${EVAL_WITH_MOTION_CFG}" \
    evaluation.eval_mode.with_planning="${EVAL_WITH_PLANNING_CFG}" \
    evaluation.eval_mode.with_ego_status="${EVAL_WITH_EGO_STATUS_CFG}" \
  2>&1 | tee "${EVAL_OUT_DIR}/logs/eval_stage2_coc_language.${EVAL_T}.node${NODE_RANK}.log"
