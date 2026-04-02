_base_ = ['./unified_decoder_stage1_ar_cotrain.py']

# ====================================================================
# Stage 2: 冻结 VLM，只训练 Motion + Planning
# 在 Stage1 的基础上，冻结 VLM backbone，专注于下游任务
# ====================================================================

model = dict(
    planning_head=dict(
        # 禁用 AR cotraining (VLM已冻结，无需语言任务)
        ar_enabled=False,
        ar_loss_weight=0.0,
        train_vlm=False,  # 冻结 VLM
    ),
)

# AR 数据集配置（禁用）
ar_dataset_cfg = dict(
    enabled=False,
)

# 加载 Stage1 checkpoint
load_from = 'work_dirs/unified_decoder_stage1_ar_cotrain/latest.pth'

# 优化器配置 (VLM已冻结，不在优化器中)
optimizer = dict(
    type='AdamW',
    lr=1e-4,  # Stage2可以用稍小的学习率
    weight_decay=1e-7,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            # VLM 已冻结，无需特殊学习率配置
        }
    )
)

# 学习率配置
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,  # Stage2 warmup 可以更短
    warmup_ratio=0.001,
    min_lr_ratio=1e-3,
)

# 梯度裁剪
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0, norm_type=2),
)

# 训练参数
total_epochs = 20  # Stage2 训练更少的 epochs
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2)

# 工作目录
work_dir = './work_dirs/unified_decoder_stage2_ar_cotrain'

# 其他设置
workflow = [('train', 1)]
gpu_ids = range(0, 32)  # 32 GPUs
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = False

# Stage 2 训练说明
# ===================
# 本阶段在 Stage1 的基础上，冻结 VLM backbone，专注于训练:
#   - Motion Decoder (运动预测)
#   - Planning Decoder (轨迹规划)
#
# 设计思路:
#   - Stage1: 通过 AR cotraining 让 VLM 学习视觉-语言对齐和指令理解
#   - Stage2: 冻结 VLM，避免过拟合到规划任务的视觉分布
#   - 结果: VLM 保持泛化能力，下游任务性能提升
#
# 监控指标:
#   - loss_planning: 规划损失（主要优化目标）
#   - loss_motion: 运动预测损失
#   - L2 error (1s/3s): 轨迹误差
#   - Collision rate: 碰撞率
#
# 注意事项:
#   - 确保 load_from 指向 Stage1 训练完成的 checkpoint
#   - 如果 Stage1 训练了 30 epochs，应该加载 epoch_30.pth 或 latest.pth
#   - VLM 参数会被冻结，不会在 Stage2 中更新
