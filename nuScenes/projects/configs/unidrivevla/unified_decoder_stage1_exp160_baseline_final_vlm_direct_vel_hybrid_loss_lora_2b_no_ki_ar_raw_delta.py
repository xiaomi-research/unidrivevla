_base_ = ['./unified_decoder_stage1_exp109_vlm.py']

# ====================================================================
# Exp2: Direct VLM Fusion (no feature fusion)
# 验证Stage1特征的必要性：直接用VLM输出，丢弃Stage1
# ====================================================================
loss_planning = dict(type='FlowPlanningLoss', use_min_snr_loss=True, min_snr_gamma=5.0, loss_weight=1.0, hybrid_loss_weight=0.5, detach_window_size=3)

model = dict(
    planning_head=dict(
        # VLM Fusion: Direct (直接使用VLM输出，不融合Stage1)
        vlm_fusion_cfg=dict(
            type='direct',
        ),
        # Feature Fusion: None (原始独立特征)
        feature_fusion_cfg=dict(
            type='none',
        ),
        loss_planning=loss_planning,
        lora_cfg = dict(
            r=64,                      
            lora_alpha=128,             
            target_modules=[           
                "q_proj", "k_proj", "v_proj", "o_proj",   # attention全覆盖
                "gate_proj", "up_proj", "down_proj",       # FFN全覆盖
            ],
            lora_dropout=0.0,            
            bias="none",        
        ),
        enable_traj_ar=True,
        traj_ar_loss_weight=0.1,
        traj_ar_target="raw_delta",
    ),
)
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
total_batch_size = 128
num_gpus = 32
batch_size = 4
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 30
total_epochs = 30
checkpoint_epoch_interval = 5


eval_mode = dict(
    with_det=False,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)

evaluation = dict(
    interval=num_iters_per_epoch * 5,
    eval_mode=eval_mode,
)
