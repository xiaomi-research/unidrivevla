_base_ = ['./unified_decoder_stage1_exp128_baseline_final_vlm.py']

# ====================================================================
# Stage 1: Perception + AR + Planning (全部解冻)
# AR Cotraining with JSONL multi-view planning QA data
# ====================================================================

model = dict(
    planning_head=dict(
        # AR cotraining 配置
        # Note: enable_knowledge_insulation 控制 text-only AR (内部模式)
        #       外部 JSONL AR 由 ar_dataset_cfg.enabled=True 控制
        #       这里不启用 enable_knowledge_insulation，只使用外部 JSONL 数据
        ar_loss_weight=0.1,  # AR损失权重
        train_vlm=True,      # VLM解冻（必需，用于 AR forward）
    ),
)

# AR 数据集配置
ar_dataset_cfg = dict(
    enabled=True,
    jsonl_path='C:/Users/owl/Desktop/0224_UniDriveVLA/nuscenes_planning_qa_train_only_command_his_traj (1).jsonl',
    data_root='./data/nuscenes',
    samples_per_gpu=2,  # AR batch size per GPU (降低以节省显存)
    workers_per_gpu=2,
    max_length=2048,    # Maximum sequence length for AR
    fix_image_size=None,  # Optional: (width, height) for fixed image size
)

# 优化器配置 (VLM使用更小的学习率)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-7,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'planning_head.qwen3_vl_with_expert.qwen3_vl': dict(
                lr_mult=0.05,  # VLM学习率降为基础的5%（基于审查建议: 防止过拟合）
            ),
        }
    )
)

# 学习率配置（增加 warmup）
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,  # 从 500 增加到 1000（基于审查建议）
    warmup_ratio=0.001,
    min_lr_ratio=1e-3,
)

# 梯度裁剪（增强训练稳定性）
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0, norm_type=2),
)

# 训练参数
total_epochs = 30
checkpoint_config = dict(interval=3)
evaluation = dict(interval=3)

# 日志配置（增加 AR 相关指标）
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

# 工作目录
work_dir = './work_dirs/unified_decoder_stage1_ar_cotrain'

# 加载预训练权重（如果有）
# load_from = 'path/to/pretrained.pth'

# 恢复训练（如果中断）
# resume_from = 'work_dirs/unified_decoder_stage1_ar_cotrain/latest.pth'

# 其他设置
workflow = [('train', 1)]
gpu_ids = range(0, 32)  # 32 GPUs
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = False

# AR Cotraining 说明
# ===================
# 本配置启用 AR Cotraining，将规划 QA 数据作为辅助任务训练 VLM
# 预期效果:
#   - 提升指令理解能力（TURN LEFT/RIGHT/STRAIGHT）
#   - 增强 VLM 泛化能力（语言任务正则化）
#   - 改善历史轨迹嵌入的鲁棒性
#
# 监控指标:
#   - loss_ar: 加权后的 AR 损失
#   - loss_vlm_raw: 原始 VLM 语言建模损失
#   - ar_planning_ratio: AR loss / Planning loss 的比值（目标: 0.3-0.7）
#   - loss_planning: 规划损失（主任务）
#
# 调优建议:
#   - 如果 ar_planning_ratio > 1.0: 减小 ar_loss_weight
#   - 如果 ar_planning_ratio < 0.1: 增大 ar_loss_weight
#   - 如果训练不稳定: 增加 warmup_iters 或降低 VLM 学习率
