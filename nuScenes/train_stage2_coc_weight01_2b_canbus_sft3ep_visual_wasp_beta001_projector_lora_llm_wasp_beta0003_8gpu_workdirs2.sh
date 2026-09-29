#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${NUSCENES_ROOT:-${SCRIPT_DIR}}
DATA_CWD=${NUSCENES_DATA_CWD:-${CODE_ROOT}}
cd "${DATA_CWD}"
export NUSCENES_ROOT="${CODE_ROOT}"

export CFG=${CFG:-${CODE_ROOT}/projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage2_with_ego.py}
export EXP_NAME=${EXP_NAME:-unidrivevla_stage2_2b_canbus_sft3ep_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs4_accum4}
export VLM_PRETRAINED_PATH=${VLM_PRETRAINED_PATH:-/path/to/unidrivevla_stage1_2b_full_ego_canbus_pure_ms_swift_3ep/checkpoint-660}
PROMPT_ALIGN_PYTHON=${PROMPT_ALIGN_PYTHON:-python}
PROMPT_ALIGN_JSONL=${PROMPT_ALIGN_JSONL:-/path/to/nuscenes_planning_qa_train_only_command_his_traj_vel_acc_can_bus.jsonl}
PROMPT_ALIGN_VAD=${PROMPT_ALIGN_VAD:-${DATA_CWD}/data/infos/vad_nuscenes_infos_temporal_train.pkl}
PROMPT_ALIGN_CHECK_SAMPLES=${PROMPT_ALIGN_CHECK_SAMPLES:-28130}

if [[ ! -f "${VLM_PRETRAINED_PATH}/config.json" ]]; then
    echo "Missing HF config: ${VLM_PRETRAINED_PATH}/config.json" >&2
    exit 1
fi
if [[ ! -f "${VLM_PRETRAINED_PATH}/model.safetensors" && ! -f "${VLM_PRETRAINED_PATH}/model.safetensors.index.json" ]]; then
    echo "VLM_PRETRAINED_PATH is not a complete HF safetensors checkpoint: ${VLM_PRETRAINED_PATH}" >&2
    exit 1
fi
if [[ ! -f "${VLM_PRETRAINED_PATH}/tokenizer.json" && ! -f "${VLM_PRETRAINED_PATH}/tokenizer_config.json" ]]; then
    echo "VLM_PRETRAINED_PATH has no tokenizer files: ${VLM_PRETRAINED_PATH}" >&2
    exit 1
fi
for required_file in "${PROMPT_ALIGN_JSONL}" "${PROMPT_ALIGN_VAD}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing prompt-alignment input: ${required_file}" >&2
        exit 1
    fi
done
if [[ "${PROMPT_ALIGN_PYTHON}" == */* ]]; then
    if [[ ! -x "${PROMPT_ALIGN_PYTHON}" ]]; then
        echo "PROMPT_ALIGN_PYTHON is not executable: ${PROMPT_ALIGN_PYTHON}" >&2
        exit 1
    fi
elif ! command -v "${PROMPT_ALIGN_PYTHON}" >/dev/null 2>&1; then
    echo "PROMPT_ALIGN_PYTHON was not found on PATH: ${PROMPT_ALIGN_PYTHON}" >&2
    exit 1
fi

echo "Checking all ${PROMPT_ALIGN_CHECK_SAMPLES} canbus prompts before training"
"${PROMPT_ALIGN_PYTHON}" "${SCRIPT_DIR}/tools/check_canbus_prompt_alignment.py" \
    --jsonl "${PROMPT_ALIGN_JSONL}" \
    --vad-pkl "${PROMPT_ALIGN_VAD}" \
    --num-samples "${PROMPT_ALIGN_CHECK_SAMPLES}"
"${PROMPT_ALIGN_PYTHON}" "${SCRIPT_DIR}/tools/check_stage2_canbus_prompt_alignment.py" \
    --jsonl "${PROMPT_ALIGN_JSONL}" \
    --vad-pkl "${PROMPT_ALIGN_VAD}" \
    --num-samples "${PROMPT_ALIGN_CHECK_SAMPLES}"

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
export MASTER_PORT=${MASTER_PORT:-28931}
export EVAL_PORT=${EVAL_PORT:-28932}
export RUN_FINAL_EVAL=${RUN_FINAL_EVAL:-1}
export ENABLE_COC_GENERATE_FOR_FINAL_EVAL=${ENABLE_COC_GENERATE_FOR_FINAL_EVAL:-1}

echo "Initialization: 2B full can-bus SFT 3ep (${VLM_PRETRAINED_PATH})"
echo "Experiment: Vision WASP beta=0.01 + Projector LoRA + LLM WASP beta=0.003"
echo "PEFT: r=64 alpha=128 top_k=16; Vision WASP=96, Projector LoRA=8, LLM WASP=196"
echo "CoC: language weight=0.1; predicted CoC answer injected at final eval"
echo "Prompt: with-ego canbus context enabled; without-ego baseline unchanged"
echo "Batch: 4 samples/GPU x 8 GPUs x 4 accumulation = 128"
echo "Schedule: 20 epochs, 879 iter/epoch, save each epoch, keep 2"
echo "Final eval: predicted CoC/ego, det+tracking+map+planning+ego on, motion off"
echo "Auto-resume: latest checkpoint under the unique work directory"

exec bash "${SCRIPT_DIR}/train_stage2_coc_language_8gpu_workdirs2.sh" "$@"
