#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${NUSCENES_ROOT:-${SCRIPT_DIR}}
DATA_CWD=${NUSCENES_DATA_CWD:-${CODE_ROOT}}

usage() {
  cat <<'EOF'
Usage:
  bash train_stage2_large_8b_16gpu_2node.sh EXPERIMENT [train arguments]

Validated experiment:
  vision_beta0005_llm_beta0003  Vision WASP beta=0.005, LLM WASP beta=0.003

The launcher expects two nodes with eight visible GPUs per node. The cluster
runtime must provide MLP_ROLE_INDEX and MLP_WORKER_0_HOST on both nodes.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

EXPERIMENT=$1
shift

case "${EXPERIMENT}" in
  vision_beta0005_llm_beta0003)
    DEFAULT_CFG=${CODE_ROOT}/projects/configs/UniDriveVLA/unidrivevla_nusc_8b_stage2.py
    DEFAULT_EXP_NAME=unidrivevla_stage2_8b_nusc_large_stage1_coc_weight01_visual_wasp_beta0005_r128a256_top16_projector_lora_r128a256_llm_wasp_beta0003_r128a256_top16_2node16gpu_bs2_accum4_zero2_vgc
    VISION_WASP_BETA=0.005
    LLM_WASP_BETA=0.003
    DEFAULT_MASTER_PORT=28841
    DEFAULT_EVAL_PORT=28842
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unsupported Large experiment: ${EXPERIMENT}" >&2
    usage >&2
    exit 2
    ;;
esac

cd "${DATA_CWD}"
export NUSCENES_ROOT="${CODE_ROOT}"

export CFG=${CFG:-${DEFAULT_CFG}}
export EXP_NAME=${EXP_NAME:-${DEFAULT_EXP_NAME}}
export VLM_PRETRAINED_PATH=${VLM_PRETRAINED_PATH:-/path/to/UniDriveVLA_Nusc_Large_Stage1}
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${CODE_ROOT}/zero_configs/adam_zero2_bf16_bs2_accum4_16gpu.json}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING=${UNIDRIVE_VISUAL_GRADIENT_CHECKPOINTING:-1}

export ENABLE_COC=1
export COC_POLICY_PREFIX=0
export ENABLE_COC_LANGUAGE_OUTPUT=1
export COC_PLANNING_LOSS_WEIGHT=0.0
export COC_CONTEXT_DROPOUT=0.0
export COC_LANGUAGE_LOSS_WEIGHT=0.1
export COC_LANGUAGE_TARGET_FORMAT=cot
export ENABLE_COC_GENERATE=0
export COC_EVAL_SOURCE=pred
export COC_EVAL_INJECTION=answer

export NUM_GPUS=${NUM_GPUS:-16}
export MLP_WORKER_NUM=${MLP_WORKER_NUM:-2}
export MLP_WORKER_GPU=${MLP_WORKER_GPU:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:-${DEFAULT_MASTER_PORT}}}
export EVAL_PORT=${EVAL_PORT:-${DEFAULT_EVAL_PORT}}
export RUN_FINAL_EVAL=${RUN_FINAL_EVAL:-1}
export ENABLE_COC_GENERATE_FOR_FINAL_EVAL=${ENABLE_COC_GENERATE_FOR_FINAL_EVAL:-1}

echo "Initialization: UniDriveVLA_Nusc_Large_Stage1 (Qwen3-VL-8B)"
echo "Experiment key: ${EXPERIMENT}"
VISION_WASP_RANK=${VISION_WASP_RANK:-64}
VISION_WASP_ALPHA=${VISION_WASP_ALPHA:-128}
VISION_WASP_TOP_K=${VISION_WASP_TOP_K:-16}
LLM_WASP_RANK=${LLM_WASP_RANK:-64}
LLM_WASP_ALPHA=${LLM_WASP_ALPHA:-128}
LLM_WASP_TOP_K=${LLM_WASP_TOP_K:-16}
PROJECTOR_LORA_RANK=${PROJECTOR_LORA_RANK:-128}
PROJECTOR_LORA_ALPHA=${PROJECTOR_LORA_ALPHA:-256}
echo "Adapters: Vision WASP beta=${VISION_WASP_BETA}; Projector LoRA r=${PROJECTOR_LORA_RANK} alpha=${PROJECTOR_LORA_ALPHA}; LLM WASP beta=${LLM_WASP_BETA}"
echo "WASP: vision rank=${VISION_WASP_RANK} alpha=${VISION_WASP_ALPHA} top_k=${VISION_WASP_TOP_K}; language rank=${LLM_WASP_RANK} alpha=${LLM_WASP_ALPHA} top_k=${LLM_WASP_TOP_K}; expected modules: vision=108, language=252"
echo "CoC: language weight=0.1; predicted CoC answer injected at final eval"
echo "Memory: ZeRO-2 + visual gradient checkpointing + expandable segments"
echo "Batch: 2 samples/GPU x 16 GPUs x 4 accumulation = 128"
echo "Schedule: 20 epochs, 879 iter/epoch, 17580 total iter"
echo "Final eval: predicted CoC/ego; det, tracking, map, planning and ego enabled; motion disabled"

exec bash "${SCRIPT_DIR}/train_stage2_coc_language_8gpu_workdirs2.sh" "$@"
