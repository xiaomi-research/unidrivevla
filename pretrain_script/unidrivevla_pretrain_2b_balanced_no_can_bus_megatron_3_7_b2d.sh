#!/bin/bash
# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \

GPUS=${IDP_N_GPU:-8}
NNODES=${IDP_N_NODES:-4}
NODE_RANK=${IDP_N_RANK:-0}
MASTER_ADDR=${IDP_MASTER_ADDR:-localhost}

export FORCE_QWENVL_VIDEO_READER="torchvision"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

DISTRIBUTED_ARGS=(
    --nproc_per_node $MLP_WORKER_GPU
    --nnodes $MLP_WORKER_NUM
    --node_rank $MLP_ROLE_INDEX
    --master_addr $MLP_WORKER_0_HOST
    --master_port $MLP_WORKER_0_PORT
)

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=8 \
SWIFT_PATCH_CONV3D=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=506 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
torchrun "${DISTRIBUTED_ARGS[@]}" \
    swift/cli/_megatron/sft.py \
    --model /high_perf_store3/world-model/zhuzhenxin/ckpts/Qwen3-VL-2B-Instruct/ \
    --load_safetensors true \
    --save_safetensors true \
    --dataset   '/high_perf_store3/world-model/yongkangli/ABCDEFG_NISHIDASHABI/A/B/UniDriveVLA/Bench2Drive/data/b2d_planning_qa_train_residual.jsonl' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/Orion_Data/train_converted_processed.jsonl' \
                '/high_perf_store3/world-model/yongkangli/B2D/Bench2DriveZoo-tcp-admlp/output_final_modified_finalview_processed.jsonl' \
                '/high_perf_store3/world-model/yongkangli/finevision_subset_cleaned.jsonl#1141184' \
    --load_from_cache_file true \
    --tensor_model_parallel_size 1 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 4e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 3 \
    --new_special_tokens '/high_perf_store3/world-model/yongkangli/ACL-2026/ms-swift-main/tokens.txt' \
    --system '/high_perf_store3/world-model/yongkangli/ACL-2026/ms-swift-main/system.txt' \
    --save megatron_output/Qwen3-VL-2B-Instruct-3-7-b2d \
    --save_interval 500 \
    --vit_gradient_checkpointing true \
    --max_length 16384 \
    --num_workers 32 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32 \
    --attention_backend flash > unidrivevla-pretrain-2b-3-7-b2d.txt 2>&1