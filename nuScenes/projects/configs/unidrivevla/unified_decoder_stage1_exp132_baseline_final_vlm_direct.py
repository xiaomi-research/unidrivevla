_base_ = ['./unified_decoder_stage1_exp109_vlm.py']

# ====================================================================
# Exp2: Direct VLM Fusion (no feature fusion)
# 验证Stage1特征的必要性：直接用VLM输出，丢弃Stage1
# ====================================================================

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
    ),
)
