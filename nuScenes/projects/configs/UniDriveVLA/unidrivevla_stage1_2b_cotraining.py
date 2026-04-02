_base_ = ["../_base_/default_runtime.py"]

import os
# ===== User Configuration =====
vlm_pretrained_path = os.environ.get("VLM_PRETRAINED_PATH", "/path/to/Qwen3-VL-2B-Instruct")
occworld_vae_path = os.environ.get("OCCWORLD_VAE_PATH", "/path/to/ckpt/occvae_latest.pth")
deepspeed_config = os.environ.get("DEEPSPEED_CONFIG", "/path/to/zero_configs/adam_zero1_bf16.json")
driving_jsonl_root = os.environ.get("DRIVING_JSONL_ROOT", "/path/to/lyk_datasets_json")
data_infos_root = os.environ.get("DATA_INFOS_ROOT", "data/infos")
# ==============================

# ====================================================================
#  Stage 1: Unified perception decoder (det + map + ego, no motion)
#  Train perception queries first, motion loss disabled.
# ====================================================================

# Update-2023-06-12:
# [Enhance] Update some freezing args of UniAD
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
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
checkpoint_epoch_interval = 1

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

input_shape = (960, 544)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]

num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4 # history + current

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

# EgoInstanceBank feature map scale (actual feature map size after backbone)
ego_feature_map_scale = (544 // 16, 960 // 16)

# =================== Operation Order (HiP-AD style) ===================
# ✅ FIX: First layer has NO temporal attention (single frame decoder)
# Subsequent layers have temporal attention (temporal frame decoder)
single_frame_layer = [
    'concat',      # 1. Merge det/map/ego queries
    'gnn',         # 2. Self-attention (task-specific)
    'inter_gnn',   # 3. Interactive attention (ego ← det+map, Stage2 only)
    'norm',        # 4. Normalization
    'split',       # 5. Split queries back to det/map/ego
    'deformable',  # 6. Deformable feature aggregation (per task)
    'concat',      # 7. Merge again for FFN
    'ffn',         # 8. Feed-forward network
    'norm',        # 9. Normalization
    'split',       # 10. Split queries
    'refine',      # 11. Task-specific refinement (Layer 0 output)
]
temporal_frame_layer = [
    'concat',      # 1. Merge det/map/ego queries
    'temp_gnn',    # 2. Temporal cross-attention (current ← history)
    'gnn',         # 3. Self-attention (task-specific)
    'inter_gnn',   # 4. Interactive attention (ego ← det+map, Stage2 only)
    'norm',        # 5. Normalization
    'split',       # 6. Split queries back to det/map/ego
    'deformable',  # 7. Deformable feature aggregation (per task)
    'concat',      # 8. Merge again for FFN
    'ffn',         # 9. Feed-forward network
    'norm',        # 10. Normalization
    'split',       # 11. Split queries
    'refine',      # 12. Task-specific refinement (Layers 1-5 output)
]
# First layer: single frame (no temporal), Layers 2-6: temporal
# Total: 11 + 12*5 = 71 operations (matches HiP-AD exactly)
unified_decoder_operation_order = single_frame_layer * num_single_frame_decoder + \
                                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

# =================== Unified Decoder Configuration (Stage1: det + map + ego, no motion) ===================
unified_decoder_cfg = dict(
    type="UnifiedPerceptionDecoder",
    embed_dims=embed_dims,
    task_select=["det", "map", "ego"],  # Stage1: no motion
    query_select=["det", "map", "ego"],
    num_stage1_layers=3,
    num_stage2_layers=3,
    num_single_frame_decoder=1,
    cls_threshold_to_reg=0.05,
    decouple_attn=decouple_attn,
    use_vlm_in_stage2=True,
    # ---- Operation Order (HiP-AD style) ----
    operation_order=unified_decoder_operation_order,

    # ---- Instance Banks ----
    det_instance_bank=dict(
        type="InstanceBank",
        num_anchor=900,
        embed_dims=embed_dims,
        anchor="data/kmeans/kmeans_det_900.npy",
        anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
        num_temp_instances=600,
        confidence_decay=0.6,
        feat_grad=False,
    ),
    map_instance_bank=dict(
        type="InstanceBank",
        num_anchor=100,
        embed_dims=embed_dims,
        anchor="data/kmeans/kmeans_map_100.npy",
        anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
        num_temp_instances=0,
        confidence_decay=0.6,
        feat_grad=True,
    ),
    ego_instance_bank=dict(
        type="EgoInstanceBank",
        embed_dims=embed_dims,
        anchor_type='nus',
        num_temp_instances=1,
        feature_map_scale=ego_feature_map_scale,
    ),

    # ---- Anchor Encoders ----
    det_anchor_encoder=dict(
        type="SparseBox3DEncoder",
        vel_dims=3,
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
    ),
    map_anchor_encoder=dict(
        type="SparsePoint3DEncoder",
        embed_dims=embed_dims,
        num_sample=num_sample,
    ),

    # ---- GNN (SeparateAttention) ----
    graph_model=dict(
        type="SeparateAttention",
        query_select=["det", "map", "ego"],
        separate_list=[["det"], ["map"]],
        decouple_list=[True, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

    # ---- Temporal GNN (TemporalSeparateAttention) ----
    temp_graph_model=dict(
        type="TemporalSeparateAttention",
        query_select=["det", "map", "ego"],
        query_list=[["det"], ["map"], ["ego"]],
        key_list=[["det"], ["map"], ["det", "map"]],
        decouple_list=[True, False, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

    # ---- Inter GNN (InteractiveAttention, Stage2 only) ----
    inter_graph_model=dict(
        type="InteractiveAttention",
        query_select=["det", "map", "ego"],
        query_list=[["ego"]],
        key_list=[["det", "map"]],
        decouple_list=[False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

    # ---- Deformable per task ----
    det_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=use_deformable_func,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0], [-0.45, 0, 0],
                [0, 0.45, 0], [0, -0.45, 0],
                [0, 0, 0.45], [0, 0, -0.45],
            ],
        ),
    ),
    map_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=use_deformable_func,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparsePoint3DKeyPointsGenerator",
            embed_dims=embed_dims,
            num_sample=num_sample,
            num_learnable_pts=3,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        ),
    ),
    ego_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=use_deformable_func,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0], [-0.45, 0, 0],
                [0, 0.45, 0], [0, -0.45, 0],
                [0, 0, 0.45], [0, 0, -0.45],
            ],
        ),
    ),

    # ---- FFN + Norm ----
    ffn=dict(
        type="AsymmetricFFN",
        in_channels=embed_dims * 2,
        pre_norm=dict(type="LN"),
        embed_dims=embed_dims,
        feedforward_channels=embed_dims * 4,
        num_fcs=2,
        ffn_drop=drop_out,
        act_cfg=dict(type="ReLU", inplace=True),
    ),
    norm_layer=dict(type="LN", normalized_shape=embed_dims),

    # ---- Refine layers ----
    det_refine_layer=dict(
        type="SparseBox3DRefinementModule",
        embed_dims=embed_dims,
        num_cls=num_classes,
        refine_yaw=True,
        with_quality_estimation=with_quality_estimation,
    ),
    map_refine_layer=dict(
        type="SparsePoint3DRefinementModule",
        embed_dims=embed_dims,
        num_sample=num_sample,
        num_cls=num_map_classes,
    ),
    ego_refine_layer=dict(
        type="EgoStatusRefinementModule",
        embed_dims=embed_dims,
        status_dims=6,
    ),
    motion_refine_layer=dict(
        type="SparseMotionRefinementModule",
        embed_dims=embed_dims,
        fut_ts=fut_ts,
        fut_mode=fut_mode,
    ),

    # ---- Samplers ----
    det_sampler=dict(
        type="SparseBox3DTarget",
        num_dn_groups=0,
        num_temp_dn_groups=0,
        dn_noise_scale=[2.0] * 3 + [0.5] * 7,
        max_dn_gt=32,
        add_neg_dn=True,
        cls_weight=2.0,
        box_weight=0.25,
        reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
        cls_wise_reg_weights={
            9: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        },
    ),
    map_sampler=dict(
        type="SparsePoint3DTarget",
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
            ),
        ),
        num_cls=num_map_classes,
        num_sample=num_sample,
        roi_size=roi_size,
    ),
    motion_sampler=dict(type='SparseMotionTarget'),

    # ---- Decoders ----
    det_decoder=dict(type="SparseBox3DDecoder"),
    map_decoder=dict(type="SparsePoint3DDecoder"),

    # ---- Losses ----
    loss_det_cls=dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0,
    ),
    loss_det_reg=dict(
        type="SparseBox3DLoss",
        loss_box=dict(type="L1Loss", loss_weight=0.25),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
        loss_yawness=dict(type="GaussianFocalLoss"),
        cls_allow_reverse=[5],  # barrier
    ),
    loss_map_cls=dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
    ),
    loss_map_reg=dict(
        type="SparseLineLoss",
        loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
        num_sample=num_sample,
        roi_size=roi_size,
    ),
    loss_ego_status=dict(type="L1Loss", loss_weight=1.0),
    loss_motion_cls=dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=0.0,  # Stage1: motion loss disabled
    ),
    loss_motion_reg=dict(type="L1Loss", loss_weight=0.0),  # Stage1: disabled

    # ---- Reg weights for dimension truncation ----
    det_reg_weights=[2.0] * 3 + [1.0] * 7,  # 10-dim: [x,y,z, w,l,h, sin_yaw,cos_yaw, vx,vy]
    map_reg_weights=[1.0] * 40,

    # ---- Motion anchor ----
    motion_anchor="data/kmeans/kmeans_motion_6.npy",
)

vlm_lr_mult = 0.5

lora_cfg = dict(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
)

loss_planning = dict(type='FlowPlanningLoss', use_min_snr_loss=True, min_snr_gamma=5.0, loss_weight=1.0, hybrid_loss_weight=0.5, detach_window_size=3)

model = dict(
    type='UniDriveVLA',
    planning_head=dict(
        type='QwenVL3APlanningHead',
        pretrained_path=vlm_pretrained_path,
        action_dim=2,
        action_horizon=6,
        dtype='bfloat16',
        train_vlm=True,
        loss_planning=loss_planning,
        # Depth Supervision
        with_depth_supervision=False,
        depth_loss_weight=0.2,
        # OccWorld VAE
        occworld_vae_config=dict(
            type='VAERes3D',
            encoder_cfg=dict(
                type='Encoder2D',
                ch=64,
                out_ch=64,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50,),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=128,
                double_z=False,
            ),
            decoder_cfg=dict(
                type='Decoder3D',
                ch=64,
                out_ch=128,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50,),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=64,
                give_pre_end=False,
            ),
            num_classes=18,
            expansion=8,
            vqvae_cfg=None,
        ),
        occworld_vae_path=occworld_vae_path,
        feat_grad=True,
        feature_source="raw",  # or "raw"

        # Unified Decoder (Stage1: det + map + ego, no motion)
        unified_decoder_cfg=unified_decoder_cfg,

        # Legacy dual-head configs disabled
        det_vla_head_cfg=None,
        map_vla_head_cfg=None,
        lora_cfg=lora_cfg,
        driving_deepstack=True,
        vlm_fusion_cfg=dict(type='direct'),
        feature_fusion_cfg=dict(type='none'),
    ),
    task_loss_weight=dict(planning=1.0),
)

ar_dataset_cfg = dict(
    enabled=True,
    # 驾驶数据
    jsonl_paths=[
            os.path.join(driving_jsonl_root, 'qa_dataset_lingoqa_clean_packed2048.jsonl'), #26319
            os.path.join(driving_jsonl_root, 'drivelm_converted.with_prefix_packed2048.jsonl'), #4071
            os.path.join(driving_jsonl_root, 'senna_final_reordered_updated_token_fixed_packed2048.jsonl'), # 27885
        ],
    max_length=2048,
    # 通用 VQA 数据
    vqa_jsonl_paths=[os.path.join(driving_jsonl_root, 'finevision_subset_90k.jsonl')],
    vqa_max_length=2048,   # VQA 序列更短，单独控制
    # 共用
    samples_per_gpu=2,
    workers_per_gpu=2,
    max_pixels=100000,
)


dataset_type = "NuScenes3DDataset"
data_root = "data/nuscenes/"
anno_root = "data/infos/"
info_root = "data/infos/"
file_client_args = dict(backend="disk")

ann_file_train = info_root + "nuscenes_infos_train.pkl"
ann_file_val = info_root + "nuscenes_infos_val.pkl"
ann_file_test = info_root + "nuscenes_infos_val.pkl"

vad_ann_file_train = os.path.join(data_infos_root, "vad_nuscenes_infos_temporal_train.pkl")
vad_ann_file_val = os.path.join(data_infos_root, "vad_nuscenes_infos_temporal_val.pkl")
vad_ann_file_test = os.path.join(data_infos_root, "vad_nuscenes_infos_temporal_val.pkl")

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type='LoadOccWorldLabels', data_root='data/nuscenes', input_dataset='gts'),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_map_labels',
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
            "gt_occ_dense",
            "hist_traj",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            'ego_status',
            'gt_ego_fut_cmd',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            "hist_traj",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

eval_pipeline = [
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=True,
        normalize=False,
    ),
    dict(
        type='Collect',
        keys=[
            'vectors',
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'fut_boxes',
            "hist_traj",
        ],
        meta_keys=['token', 'timestamp']
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + 'nuscenes_infos_val.pkl',
    vad_ann_file=vad_ann_file_val,
    pipeline=eval_pipeline,
    test_mode=True,
)

data_aug_conf = {
    "resize_lim": (0.6, 0.6),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        vad_ann_file=vad_ann_file_train,
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        vad_ann_file=vad_ann_file_val,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        vad_ann_file=vad_ann_file_test,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
    shuffler_sampler=dict(type="GroupInBatchSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

deepspeed = True
deepspeed_config = deepspeed_config

gradient_checkpointing = dict(
    enabled=True,
    checkpoint_activations=True,
    checkpoint_attention=True,
)

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
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

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

log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, interval=1, warm_up=2000, priority='VERY_HIGH')
]

checkpoint_config = dict(deepspeed=deepspeed, interval=num_iters_per_epoch * checkpoint_epoch_interval, max_keep_ckpts=2)
load_from = None

find_unused_parameters = True
logger_name = 'mmdet'
