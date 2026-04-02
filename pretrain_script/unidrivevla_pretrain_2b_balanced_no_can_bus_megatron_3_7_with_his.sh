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
    --dataset   '/high_perf_store3/world-model/yongkangli/UniDriveVLA/nuscenes_planning_qa_train_only_command_his_traj.jsonl#28130' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/NuInstruct/qa_dataset_scored_clean_fixed_updated_token_fixed.jsonl#57317' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/CODA-LM/output_coda_lm_updated_processed_clean.jsonl#20318' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/drivegpt4/output_scored_clean_fixed.jsonl#26319' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/Drama/drama_qa_scored_original_conversation_batched_retry_score_retry_clean_filter.jsonl#16401' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/LingoQA/qa_dataset_lingoqa_clean.jsonl#26824' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/sutd/qa_dataset_rewritten_explained_scored_video_clean.jsonl#9916' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/talk2car/talk2car_internvl_normalized_cleaned_updated.jsonl#8079' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/NuScenes-QA/qa_dataset_clean_updated_token_fixed.jsonl#24988' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/Omnidrive/qa_dataset_scored_omnidrive_clean_updated_token_fixed_no_system.jsonl#28009' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/Senna/senna_final_reordered_updated_token_fixed.jsonl#27885' \
                '/high_perf_store3/world-model/yongkangli/Dataset_vqa/maplm/qa_dataset_rewritten_explained_scored_maplm_new_clean_token_fixed.jsonl#10612' \
                '/high_perf_store3/world-model/yongkangli/finevision_subset_cleaned.jsonl#641439' \
    --val_dataset '/high_perf_store3/world-model/yongkangli/UniDriveVLA/nuscenes_planning_qa_val_only_command_his_traj.jsonl' \
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
    --save megatron_output/Qwen3-VL-2B-Instruct-3-7-with-his \
    --save_interval 500 \
    --vit_gradient_checkpointing true \
    --max_length 16384 \
    --num_workers 32 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32 \
    --attention_backend flash > unidrivevla-pretrain-2b-3-7-with-his.txt 2>&1