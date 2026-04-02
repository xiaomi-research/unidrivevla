#!/bin/bash
# VLM Pretraining for UniDriveVLA-Large (8B) on nuScenes driving QA data
# Run from the qwenvl3/ directory: bash ../pretrain_script/unidrivevla_nusc_8b_stage1.sh

# ===== User Configuration =====
MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen3-VL-8B-Instruct"}
DATASET_ROOT=${DATASET_ROOT:-"/path/to/driving_datasets"}
SWIFT_ROOT=${SWIFT_ROOT:-"/path/to/ms-swift-main"}
SAVE_DIR=${SAVE_DIR:-"megatron_output/UniDriveVLA_Nusc_Large_Stage1"}
# ==============================

export FORCE_QWENVL_VIDEO_READER="torchvision"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

NPROC_PER_NODE=${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
MASTER_PORT=${MLP_WORKER_0_PORT:-29500}

DISTRIBUTED_ARGS=(
    --nproc_per_node ${NPROC_PER_NODE}
    --nnodes ${NNODES}
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
SWIFT_PATCH_CONV3D=1 \
IMAGE_MAX_TOKEN_NUM=506 \
VIDEO_MAX_TOKEN_NUM=368 \
FPS_MAX_FRAMES=16 \
torchrun "${DISTRIBUTED_ARGS[@]}" \
    swift/cli/_megatron/sft.py \
    --model ${MODEL_PATH} \
    --load_safetensors true \
    --save_safetensors true \
    --dataset \
        "${DATASET_ROOT}/nuscenes_traj_train.jsonl#28130" \              # NuScenes planning QA (same as RecogDrive)
        "${DATASET_ROOT}/dataset_drivelm.jsonl" \                       # DriveLM
        "${DATASET_ROOT}/dataset_nuinstruct.jsonl#57317" \              # NuInstruct
        "${DATASET_ROOT}/dataset_coda_lm.jsonl#20318" \                 # CODA-LM
        "${DATASET_ROOT}/dataset_drivegpt4.jsonl#26319" \               # DriveGPT4
        "${DATASET_ROOT}/dataset_drama.jsonl#16401" \                   # DRAMA
        "${DATASET_ROOT}/dataset_lingoqa.jsonl#26824" \                 # LingoQA
        "${DATASET_ROOT}/dataset_sutd.jsonl#9916" \                     # SUTD-TrafficQA
        "${DATASET_ROOT}/dataset_talk2car.jsonl#8079" \                 # Talk2Car
        "${DATASET_ROOT}/dataset_nuscenes_qa.jsonl#24988" \             # NuScenes-QA
        "${DATASET_ROOT}/dataset_omnidrive.jsonl#28009" \               # OmniDrive
        "${DATASET_ROOT}/dataset_senna.jsonl#27885" \                   # Senna
        "${DATASET_ROOT}/dataset_maplm.jsonl#10612" \                   # MapLM
        "${DATASET_ROOT}/dataset_finevision.jsonl#641439" \             # FineVision (general VQA)
    --load_from_cache_file true \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 4e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 3 \
    --new_special_tokens "${SWIFT_ROOT}/tokens.txt" \
    --system "${SWIFT_ROOT}/system.txt" \
    --save ${SAVE_DIR} \
    --save_interval 500 \
    --vit_gradient_checkpointing true \
    --max_length 16384 \
    --num_workers 32 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32 \
    --attention_backend flash > unidrivevla-pretrain-8b-nusc.txt 2>&1
