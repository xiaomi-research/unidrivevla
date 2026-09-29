#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${NUSCENES_ROOT:-${SCRIPT_DIR}}
DATA_CWD=${NUSCENES_DATA_CWD:-${CODE_ROOT}}
cd "${DATA_CWD}"
export NUSCENES_ROOT="${CODE_ROOT}"

export CFG=${CFG:-${CODE_ROOT}/projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage2.py}
export EXP_NAME=${EXP_NAME:-unidrivevla_stage2_2b_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs4_accum4}

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

export NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_PORT=${MASTER_PORT:-28731}
export EVAL_PORT=${EVAL_PORT:-28732}
export RUN_FINAL_EVAL=${RUN_FINAL_EVAL:-1}
export ENABLE_COC_GENERATE_FOR_FINAL_EVAL=${ENABLE_COC_GENERATE_FOR_FINAL_EVAL:-1}

echo "Experiment: Vision WASP beta=0.01 + Projector LoRA + LLM WASP beta=0.003"
echo "PEFT: r=64 alpha=128 top_k=16; separate Vision/LLM WASP losses"
echo "CoC: language weight=0.1; predicted CoC answer injected at final eval"
echo "Batch: 4 samples/GPU x 8 GPUs x 4 accumulation = 128"
echo "Schedule: 20 epochs, 879 iter/epoch, save each epoch, keep 2"
echo "Final eval: predicted CoC/ego, det+tracking+map+planning+ego on, motion off"

exec bash "${SCRIPT_DIR}/train_stage2_coc_language_8gpu_workdirs2.sh" "$@"
