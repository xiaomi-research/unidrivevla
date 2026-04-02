import os
# ===== User Configuration =====
num_gpus = int(os.environ.get("NUM_GPUS", 64))
# ==============================

log_level = "INFO"
dist_params = dict(backend="nccl")

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
batch_size = 4
num_iters_per_epoch = int(234769 // (num_gpus * batch_size))
num_epochs = 5
total_epochs = num_epochs
checkpoint_epoch_interval = 1

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

import os
# ===== User Configuration =====
project_dir = os.environ.get("PROJECT_DIR", "/path/to/unidrivevla/Bench2Drive")
vlm_pretrained_path = os.environ.get("VLM_PRETRAINED_PATH", "/path/to/Qwen3-VL-2B-Instruct")
stage1_checkpoint = os.environ.get("STAGE1_CHECKPOINT", "/path/to/stage1/checkpoint.pt")
deepspeed_config = os.environ.get("DEEPSPEED_CONFIG", os.path.join(os.path.dirname(project_dir), "zero_configs/adam_zero1_bf16.json"))
# ==============================

load_from = stage1_checkpoint
resume_from = None
workflow = [("train", 1)]

input_shape = (960, 544)
num_cams = 6

det_class_names = ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"]
map_class_names = ["Broken", "Solid", "SolidSolid", "Center"]

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)
decouple_attn = True
decouple_attn_map = False
map_roi_size = (30, 60)
map_num_pts = 20
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1

fut_ts = 6
fut_mode = 6
ego_fut_ts = 6
ego_fut_cmd = 1
ego_fut_mode = 48

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
decouple_attn = True
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

temporal = True
temporal_det = True
temporal_map = True
temporal_ego = True
temporal_plan = True

task_select = ["det", "map", "ego", "motion"]
query_select = ["det", "map", "ego", "motion"]


anchor_paths = {
    "det": f"{project_dir}/data/kmeans/b2d_det_900.npy",
    "map": f"{project_dir}/data/kmeans/b2d_map_100.npy",
    "motion": f"{project_dir}/data/kmeans/b2d_motion_{fut_mode}.npy",
}

spat_path_6x8_2m = f"{project_dir}/data/kmeans/b2d_plan_spat_6x8_2m.npy"
spat_path_6x8_5m = f"{project_dir}/data/kmeans/b2d_plan_spat_6x8_5m.npy"

plan_anchor_refer = ("spat", "2m")
plan_anchor_types = [
    ("temp", "5hz"), ("spat", "2m"), ("temp", "2hz"), ("spat", "5m"),
    ("speed", "5hz", (0, 0.4)), ("speed", "5hz", (0.4, 3)), ("speed", "5hz", (3, 999)),
    ("speed", "2hz", (0, 0.4)), ("speed", "2hz", (0.4, 3)), ("speed", "2hz", (3, 999)),
]

use_min_snr_loss = True
min_snr_gamma = 5.0
min_snr_weight = 1.0

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

loss_planning = dict(
    type='FlowPlanningLoss',
    use_min_snr_loss=use_min_snr_loss,
    min_snr_gamma=min_snr_gamma,
    loss_weight=min_snr_weight,
    hybrid_loss_weight=0.5,
    detach_window_size=3,
    delta_mu=[-0.0222, 2.0249],
    delta_std=[0.6720, 1.8586],
)

ego_feature_map_scale = (544 // 16, 960 // 16)

single_frame_layer = [
    'concat', 'gnn', 'inter_gnn', 'norm', 'split',
    'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
]
temporal_frame_layer = [
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split',
    'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
]
unified_decoder_operation_order = single_frame_layer * num_single_frame_decoder + \
                                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

unified_decoder_cfg = dict(
    type='UnifiedPerceptionDecoder',
    embed_dims=embed_dims,
    task_select=task_select,
    query_select=query_select,
    num_stage1_layers=3,
    num_stage2_layers=3,
    num_single_frame_decoder=num_single_frame_decoder,
    cls_threshold_to_reg=0.05,
    decouple_attn=decouple_attn,
    operation_order=unified_decoder_operation_order,
    motion_anchor=anchor_paths["motion"],

    det_instance_bank=dict(
        type="InstanceBank",
        num_anchor=900,
        embed_dims=embed_dims,
        anchor=anchor_paths["det"],
        anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
        num_temp_instances=600 if temporal_det else -1,
        confidence_decay=0.6,
        feat_grad=False,
        class_names=det_class_names,
        zero_velocity_classes=["traffic_sign", "traffic_cone", "traffic_light"],
    ),

    map_instance_bank=dict(
        type="InstanceBank",
        num_anchor=100,
        embed_dims=embed_dims,
        anchor=anchor_paths["map"],
        anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
        num_temp_instances=0 if temporal_map else -1,
        confidence_decay=0.6,
        feat_grad=True,
    ),

    ego_instance_bank=dict(
        type="EgoInstanceBank",
        anchor_type="b2d",
        embed_dims=embed_dims,
        num_temp_instances=1 if temporal_ego else -1,
        feature_map_scale=ego_feature_map_scale,
    ),

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
        num_sample=map_num_pts,
    ),

    motion_anchor_encoder=dict(
        type="SparseBox3DEncoder",
        vel_dims=3,
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
    ),

    graph_model=dict(
        type="SeparateAttention",
        query_select=task_select,
        separate_list=[["det"], ["map"]],
        decouple_list=[True, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2, num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims, num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

    temp_graph_model=dict(
        type="TemporalSeparateAttention",
        query_select=task_select,
        query_list=[["det"], ["map"], ["ego"]],
        key_list=[["det"], ["map"], ["det", "map"]],
        decouple_list=[True, False, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2, num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims, num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims, num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ) if temporal else None,

    inter_graph_model=dict(
        type="InteractiveAttention",
        query_select=task_select,
        query_list=[["ego"]],
        key_list=[["det", "map"]],
        decouple_list=[False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims, num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

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
            fix_scale=[[0,0,0],[0.45,0,0],[-0.45,0,0],[0,0.45,0],[0,-0.45,0],[0,0,0.45],[0,0,-0.45]],
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
            num_sample=map_num_pts,
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
            num_learnable_pts=12,
            fix_scale=[[0.45, 0, 0]],
        ),
    ),

    det_refine_layer=dict(
        type="SparseBox3DRefinementModule",
        embed_dims=embed_dims,
        num_cls=num_det_classes,
        refine_yaw=True,
        with_quality_estimation=True,
    ),

    map_refine_layer=dict(
        type="SparsePoint3DRefinementModule",
        embed_dims=embed_dims,
        num_sample=map_num_pts,
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
            det_class_names.index("traffic_cone"): [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        },
    ),

    loss_det_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
    loss_det_reg=dict(
        type="SparseBox3DLoss",
        loss_box=dict(type="L1Loss", loss_weight=0.25),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
        loss_yawness=dict(type="GaussianFocalLoss"),
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
        num_sample=map_num_pts,
        roi_size=map_roi_size,
    ),

    loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
    loss_map_reg=dict(
        type="SparseLineLoss",
        loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
        num_sample=map_num_pts,
        roi_size=map_roi_size,
    ),

    loss_ego_status=dict(type="L1Loss", loss_weight=1.0),

    motion_sampler=dict(type='SparseMotionTarget'),
    loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
    loss_motion_reg=dict(type="L1Loss", loss_weight=0.2),

    det_decoder=dict(type="SparseBox3DDecoder"),
    map_decoder=dict(type="SparsePoint3DDecoder"),

    det_reg_weights=[2.0] * 3 + [1.0] * 7,
    map_reg_weights=[1.0] * (map_num_pts * 2),
)

model = dict(
    type='UniDriveVLA',
    planning_head=dict(
        type='QwenVL3APlanningHead',
        pretrained_path=vlm_pretrained_path,
        action_dim=2,
        action_horizon=6,
        dtype='bfloat16',
        vlm_variant='2b',
        train_vlm=False,
        lora_cfg=lora_cfg,
        driving_deepstack=True,
        loss_planning=loss_planning,
        unified_decoder_cfg=unified_decoder_cfg,
        with_depth_supervision=False,
        depth_loss_weight=0.2,
        feature_source='raw',
        feat_grad=True,
        attn_implementation='flex',
        vlm_fusion_cfg=dict(type='direct'),
        feature_fusion_cfg=dict(type='none'),
    ),
    task_loss_weight=dict(planning=1.0),
)

dataset_type = "Bench2DriveDataset"
data_root = "data/bench2drive"
anno_root = "data/infos/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="B2DLoadPointsFromFile"),
    dict(type="ResizeCropFlipImage"),
    dict(type="B2DMultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizePloyLine",
         roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=[
             "img", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
             "gt_ego_fut_cmd", "ego_status", "ego_status_mask",
             "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
             "gt_agent_fut_trajs", "gt_agent_fut_masks",
             "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
         ],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=[
             "img", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat", "image_wh", "ego_status",
             "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz", "gt_attr_labels",
         ],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token"],
    ),
]

inference_only_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "gt_ego_fut_cmd", "projection_mat", "timestamp", "image_wh", "ego_status"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token"],
    ),
]

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    det_classes=det_class_names,
    map_classes=map_class_names,
    modality=input_modality,
    plan_anchor_types=plan_anchor_types,
)

data_aug_conf = {
    "resize_lim": (0.60, 0.60),
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
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_train.pkl",
        map_file=anno_root + "b2d_map_infos.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_val.pkl",
        map_file=anno_root + "b2d_map_infos.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_val.pkl",
        map_file=anno_root + "b2d_map_infos.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)

checkpoint_epoch_interval = 5
deepspeed = True
deepspeed_config = deepspeed_config
checkpoint_config = dict(deepspeed=deepspeed, interval=num_iters_per_epoch * checkpoint_epoch_interval, max_keep_ckpts=2)

gradient_checkpointing = dict(
    enabled=True,
    checkpoint_activations=True,
    checkpoint_attention=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'planning_head.qwen3_vl_with_expert.qwen3_vl': dict(lr_mult=vlm_lr_mult, decay_mult=1.0),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))

runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.01,
    min_lr_ratio=0.05,
    by_epoch=False,
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
    interval=num_iters_per_epoch,
    jsonfile_prefix="val/",
    pipeline=test_pipeline,
    eval_mode=eval_mode,
)
