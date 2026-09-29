#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

# External artifacts are intentionally not part of the GitHub release.
: "${FLASH_MMCV_ROOT:?Set FLASH_MMCV_ROOT to the MMCV 1.7.2 checkout}"
: "${FLASH_MMDET3D_ROOT:?Set FLASH_MMDET3D_ROOT to the MMDetection3D 1.0.0rc6 checkout}"
: "${FLASH_DATA_ROOT:?Set FLASH_DATA_ROOT to the NuScenes data/nuscenes directory}"
: "${FLASH_DATA_INFOS_ROOT:?Set FLASH_DATA_INFOS_ROOT to the NuScenes info directory}"
: "${FLASH_ANCHOR_ROOT:?Set FLASH_ANCHOR_ROOT to the NuScenes kmeans anchor directory}"
: "${FLASH_VLM_PRETRAINED_PATH:?Set FLASH_VLM_PRETRAINED_PATH to the 2B Qwen base checkpoint}"
: "${FLASH_OCCWORLD_VAE_PATH:?Set FLASH_OCCWORLD_VAE_PATH to occvae_latest.pth}"

export UNIDRIVEVLA_DATA_INFOS_ROOT="${FLASH_DATA_INFOS_ROOT}"
export UNIDRIVEVLA_DATA_ROOT="${FLASH_DATA_ROOT}"
export UNIDRIVEVLA_ANCHOR_ROOT="${FLASH_ANCHOR_ROOT}"
export UNIDRIVEVLA_VLM_PRETRAINED_PATH="${FLASH_VLM_PRETRAINED_PATH}"
export UNIDRIVEVLA_OCCWORLD_VAE_PATH="${FLASH_OCCWORLD_VAE_PATH}"
export UNIDRIVEVLA_DEEPSPEED_CONFIG="${FLASH_DEEPSPEED_CONFIG:-${ROOT}/zero_configs/adam_zero1_bf16.json}"
export ENABLE_COC=0
export COC_PLANNING_LOSS_WEIGHT=0.0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset COC_JSONL_PATH

FLASH_CONFIG="${FLASH_CONFIG:-projects/configs/UniDriveVLA/UniDriveVLA-Flash/unidrivevla_stage1_2b_no_cotraining_keep25_adanorm_action_query.py}"
FLASH_EXP="${FLASH_EXP:-../work_dirs2/unidrivevla_main_fast_stage2_visionzip_keep25_adanorm_action_query_l2_20ep}"
FLASH_SAMPLES_PER_GPU="${FLASH_SAMPLES_PER_GPU:-4}"
FLASH_QWEN3_REPLACEMENT="${FLASH_QWEN3_REPLACEMENT:-${ROOT}/projects/configs/UniDriveVLA/UniDriveVLA-Flash/transformers_replace/models}"

if [[ ! -d "${FLASH_QWEN3_REPLACEMENT}/qwen3_vl" ]]; then
    echo "Missing Flash Qwen3-VL replacement: ${FLASH_QWEN3_REPLACEMENT}/qwen3_vl" >&2
    exit 1
fi
for anchor in kmeans_det_900.npy kmeans_map_100.npy kmeans_motion_6.npy; do
    if [[ ! -f "${FLASH_ANCHOR_ROOT}/${anchor}" ]]; then
        echo "Missing Flash anchor: ${FLASH_ANCHOR_ROOT}/${anchor}" >&2
        exit 1
    fi
done

python -c 'import peft' >/dev/null 2>&1 || {
    echo "The active environment must provide peft; install it before launching Flash training." >&2
    exit 1
}

TRANSFORMERS_DIR="$(python -c 'import os, transformers; print(os.path.dirname(transformers.__file__))')"
cp -a "${FLASH_QWEN3_REPLACEMENT}/qwen3_vl" "${TRANSFORMERS_DIR}/models/"
python -c 'import inspect; from transformers.models.qwen3_vl import modeling_qwen3_vl as m; model_src = inspect.getsource(m.Qwen3VLModel.forward); vision_src = inspect.getsource(m.Qwen3VLVisionModel.forward); assert "image_embeds, deepstack_image_embeds, _ = self.get_image_features" in model_src, "Qwen3-VL replacement check failed"; assert "return_visionzip" in vision_src, "Qwen3-VL VisionZip replacement check failed"; print("Qwen3-VL VisionZip replacement check passed")'

export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-29500}"
export FLASH_MMCV_ROOT FLASH_MMDET3D_ROOT

bash tools/flash_dist_train.sh \
    "${FLASH_CONFIG}" \
    8 \
    "${FLASH_EXP}" \
    --cfg-options \
    num_gpus=8 \
    total_batch_size=128 \
    batch_size="${FLASH_SAMPLES_PER_GPU}" \
    data.samples_per_gpu="${FLASH_SAMPLES_PER_GPU}" \
    num_iters_per_epoch=879 \
    num_epochs=20 \
    total_epochs=20 \
    runner.max_iters=17580 \
    unified_decoder_cfg.ego_refine_layer.status_dims=10 \
    model.test_use_gt_ego_status=False \
    model.planning_head.unified_decoder_cfg.ego_refine_layer.status_dims=10 \
    model.planning_head.action_training_mode=query_supervised \
    model.planning_head.action_query_cfg.num_queries=8 \
    model.planning_head.action_query_cfg.l1_loss_weight=0.0 \
    model.planning_head.action_query_cfg.l2_loss_weight=1.0 \
    evaluation.interval=17580 \
    evaluation.eval_mode.with_det=True \
    evaluation.eval_mode.with_tracking=True \
    evaluation.eval_mode.with_map=True \
    evaluation.eval_mode.with_motion=False \
    evaluation.eval_mode.with_planning=True \
    evaluation.eval_mode.with_ego_status=True
