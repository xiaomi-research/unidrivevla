_base_ = ['./unified_decoder_stage1_exp109_vlm.py']

# ====================================================================
# Exp3: MLP VLM Fusion + VIT-FPN Feature Fusion
# 轻量级VLM融合 + Top-down语义传播
# ====================================================================

model = dict(
    planning_head=dict(
        # VLM Fusion: MLP (轻量级融合)
        vlm_fusion_cfg=dict(
            type='mlp',
        ),
        # Feature Fusion: FPN (VIT-FPN Top-down语义传播)
        feature_fusion_cfg=dict(
            type='fpn',
        ),
    ),
)
