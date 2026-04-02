_base_ = ['./unified_decoder_stage1_exp128_baseline_final_vlm_single_decoder.py']

# ====================================================================
# Exp167: Single Decoder + Direct VLM Fusion + LoRA (r=64) + vlm_lr=0.5
# 对应 exp166 (三expert baseline) 的 single decoder 版本
# 区别：QwenVL3ASingleDecoderPlanningHead，所有token共享同一个LLM
# ====================================================================
loss_planning = dict(type='FlowPlanningLoss', use_min_snr_loss=True, min_snr_gamma=5.0, loss_weight=1.0, hybrid_loss_weight=0.5, detach_window_size=3)

model = dict(
    planning_head=dict(
        vlm_fusion_cfg=dict(
            type='direct',
        ),
        feature_fusion_cfg=dict(
            type='none',
        ),
        loss_planning=loss_planning,
        lora_cfg=dict(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
        ),
    ),
)
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
total_batch_size = 64
num_gpus = 32
batch_size = 2
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 30
total_epochs = 30
checkpoint_epoch_interval = 5


deepspeed = True
deepspeed_config = '/high_perf_store3/world-model/yongkangli/UniDriveVLA/zero_configs/adam_zero1_bf16.json'

# EMAHook is incompatible with ZeRO Stage 3 (parameters are partitioned).
custom_hooks = []


vlm_lr_mult = 0.5

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=1e-07,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'planning_head.qwen3_vl_with_expert.qwen3_vl': dict(lr_mult=vlm_lr_mult, decay_mult=1.0),
        }
    )
)
eval_mode = dict(
    with_det=True,
    with_tracking=False,
    with_map=True,
    with_motion=False,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)

evaluation = dict(
    interval=num_iters_per_epoch * 1,
    eval_mode=eval_mode,
)
