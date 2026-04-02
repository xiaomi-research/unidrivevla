_base_ = ['./unified_decoder_stage1_exp109_vlm.py']

# ====================================================================
# Exp2: Direct VLM Fusion (no feature fusion)
# 验证Stage1特征的必要性：直接用VLM输出，丢弃Stage1
# ====================================================================

model = dict(
    planning_head=dict(
        # VLM Fusion: Direct (直接使用VLM输出，不融合Stage1)
        use_cma_flow=True,
        cma_flow_prior_path='data/kmeans/cma_flow_prior_K8_meter.pkl',
        cma_flow_loss_type='wta',
        cma_flow_top_k=8,
        vlm_fusion_cfg=dict(
            type='direct',
        ),
        # Feature Fusion: None (原始独立特征)
        feature_fusion_cfg=dict(
            type='none',
        ),
        collision_loss_weight=1.0,
        map_bound_loss_weight=1.0,
        map_bound_dis_thresh=1.0,
        map_dir_loss_weight=0.5,
        map_dir_dis_thresh=2.0,
    ),
)
eval_mode = dict(
    with_det=True,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)

