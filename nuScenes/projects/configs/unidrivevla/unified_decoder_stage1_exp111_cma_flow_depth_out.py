_base_ = ["./unified_decoder_stage1_exp109_vlm.py"]

# ====================================================================
#  CMA-Flow Configuration: Conditional Multi-Anchor Flow Matching
#  Enables multi-modal trajectory prediction with K anchors per command
# ====================================================================

# Override planning_head configuration to enable CMA-Flow
model = dict(
    planning_head=dict(
        # ✅ Enable CMA-Flow
        use_cma_flow=True,
        with_depth_supervision=True,
        depth_supervision_source="output",
        depth_loss_weight=0.2,
        # ✅ CMA-Flow Prior Path (must be generated beforehand)
        # Run: python tools/build_cma_flow_prior.py to generate this file
        cma_flow_prior_path='data/kmeans/cma_flow_prior_K8_fixed.pkl',
        # ✅ Number of modes per command (must match prior file)
        # K=6 means 6 clusters per command (left, right, straight)
        # Total clusters: 3 commands × 6 modes = 18 modes
        num_clusters_per_cmd=8,

        # ✅ CMA-Flow specific parameters
        cma_flow_config=dict(
            # Trajectory classifier architecture
            traj_classifier_hidden_dim=512,  # Hidden dimension for classifier MLP

            # Inference mode selection
            cma_flow_top_k=1,  # How many top modes to use at inference
                              # 1 = select best mode only (fastest, recommended)
                              # 3 = run ODE for top-3 modes (slower but more robust)

            # Training parameters
            k_star_method='anchor_dist',  # Always use 'anchor_dist' (static physical distance)
                                          # DO NOT use 'flow_loss' (causes information leakage!)
        ),

        # ✅ FlexAttention for CMA-Flow (optional but recommended for speed)
        # Requires PyTorch >= 2.5.0
        attn_implementation='flex',  # or 'eager' if torch < 2.5

        # ✅ Update loss weight for trajectory classifier
        # CMA-Flow adds a classification loss to predict the best mode
        loss_traj_cls_weight=1.0,  # Weight for trajectory classification loss
    ),
)
