_base_ = ["../_base_/default_runtime.py"]

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
total_batch_size = 32
num_gpus = 8
batch_size = 4
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 30
total_epochs = 30
checkpoint_epoch_interval = 5

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
vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
group_id_list = [[0,1,2,3,4], [6,7], [8], [5,9]]
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
queue_length = 3  # each sequence contains `queue_length` frames.

### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 6
past_steps = 4
use_nonlinear_optimizer = True

## occflow setting
occ_n_future = 4
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])

### planning ###
planning_steps = 6
use_col_optim = True
# there exists multiple interpretations of the planning metric, where it differs between uniad and stp3/vad
# uniad: computed at a particular time (e.g., L2 distance between the predicted and ground truth future trajectory at time 3.0s)
# stp3: computed as the average up to a particular time (e.g., average L2 distance between the predicted and ground truth future trajectory up to 3.0s)
planning_evaluation_strategy = "uniad"  # uniad or stp3

### Occ args ###
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# Other settings
train_gt_iou_threshold=0.3

# VLM learning rate multiplier (relative to base learning rate)
# Set vlm_lr_mult = 0.1 to freeze VLM (effectively), or higher values to partially train
vlm_lr_mult = 0.1  # VLM backbone learning rate multiplier


# Min SNR loss configuration
# Enable min SNR loss for better training stability with diffusion-based planning
use_min_snr_loss = True  # Toggle to enable/disable min SNR loss
min_snr_gamma = 5.0      # Gamma parameter for min SNR loss (typically 1-10)
min_snr_weight = 1.0     # Weight for min SNR loss relative to flow loss

model = dict(
    type='UniDriveVLA',
    planning_head=dict(
        type='QwenVL3APlanningHead',
        pretrained_path='/high_perf_store3/world-model/yongkangli/ms-swift-main/megatron_output/Qwen3-VL-2B-Instruct-5-5-with-his/v0-20260209-024954/checkpoint-1500/',
        action_dim=2,
        action_horizon=6,
        dtype='bfloat16',
        train_vlm=True,  # Whether to train the VLM component
        use_min_snr_loss=use_min_snr_loss,
        min_snr_gamma=min_snr_gamma,
        min_snr_weight=min_snr_weight,
        # OccWorld VAE (moved from head __init__)
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
        occworld_vae_path='/high_perf_store3/world-model/yongkangli/UniDriveVLA/ckpt/occvae_latest.pth',
        # SparseDrive-aligned perception configs
        det_anchor_path='data/kmeans/kmeans_det_900.npy',
        map_anchor_path='data/kmeans/kmeans_map_100.npy',
        motion_anchor_path='data/kmeans/kmeans_motion_6.npy',
        plan_anchor_path='data/kmeans/kmeans_plan_6.npy',
        num_det_queries=900,
        num_propagated_queries=450,
        confidence_decay=0.6,
        max_time_interval=2.0,
        num_map_queries=100,
        num_motion_queries=6,
        num_plan_queries=6,
        det_vla_head_cfg=dict(
            num_cls=len(class_names),
            vel_dims=3,
            output_dim=11,
            refine_steps=6,
            with_quality_estimation=True,
            refine_yaw=True,
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_reg=dict(
                type='SparseBox3DLoss',
                loss_box=dict(type='L1Loss', loss_weight=0.25),
                loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True),
                loss_yawness=dict(type='GaussianFocalLoss'),
                cls_allow_reverse=[class_names.index('barrier')],
            ),
            sampler=dict(
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [1.0] * 3 + [1.0] * 2 + [0.5] * 2,
                cls_wise_reg_weights={
                    class_names.index('traffic_cone'): [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                },
                num_dn_groups=5,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
            ),
            decoder=dict(type='SparseBox3DDecoder', num_output=300, score_threshold=None, sorted=True),
        ),
        map_vla_head_cfg=dict(
            num_cls=len(map_class_names),
            num_sample=20,
            roi_size=(30, 60),
            refine_steps=6,
            score_threshold=None,
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
            assigner=dict(
                type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    cls_cost=dict(type='FocalLossCost', weight=1.0),
                    reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                ),
            ),
        ),
    ),
    task_loss_weight=dict(planning=1.0),
)



dataset_type = "NuScenes3DDataset"
data_root = "data/nuscenes/"
anno_root = "data/infos/" 
info_root = "data/infos/"
file_client_args = dict(backend="disk")

# Switch to SparseDrive-style nuScenes infos.
ann_file_train = info_root + "nuscenes_infos_train.pkl"
ann_file_val = info_root + "nuscenes_infos_val.pkl"
ann_file_test = info_root + "nuscenes_infos_val.pkl"

# VAD infos are not used by SparseDrive pipeline.
vad_ann_file_train = "/high_perf_store3/world-model/yongkangli/UniDriveVLA/data/infos/vad_nuscenes_infos_temporal_train.pkl"
vad_ann_file_val = "/high_perf_store3/world-model/yongkangli/UniDriveVLA/data/infos/vad_nuscenes_infos_temporal_val.pkl"
vad_ann_file_test = "/high_perf_store3/world-model/yongkangli/UniDriveVLA/data/infos/vad_nuscenes_infos_temporal_val.pkl"

# SparseDrive-style pipelines (det/map/motion/ego supervision).
# Note: this config keeps existing Occ/Flow/Plan keys out of the pipeline for now.
# Those will be re-introduced once det/map/motion queries are integrated.

roi_size = (30, 60)
num_sample = 20
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3

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
    # dict(
    #     type="MultiScaleDepthMapGenerator",
    #     downsample=strides[:num_depth_layers],
    # ),
    dict(type='LoadOccWorldLabels', data_root='data/nuscenes', input_dataset='gts'),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    #dict(type="NormalizeMultiviewImage", **img_norm_cfg),
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
            # "gt_depth",  # optional; generated by MultiScaleDepthMapGenerator (disabled above)
            # "focal",     # optional; filled by NuScenesSparse4DAdaptor from cam_intrinsic
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
    #dict(type="NormalizeMultiviewImage", **img_norm_cfg),
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
    workers_per_gpu=16,
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
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)


deepspeed = True
deepspeed_config = '/high_perf_store3/world-model/yongkangli/UniDriveVLA/zero_configs/adam_zero1_bf16.json'



# Gradient checkpointing for memory optimization
gradient_checkpointing = dict(
    enabled=True,
    checkpoint_activations=True,
    checkpoint_attention=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
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
# learning policy
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
    with_det=True,
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
# runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
checkpoint_config = dict(deepspeed=deepspeed, interval=num_iters_per_epoch * checkpoint_epoch_interval, max_keep_ckpts=2)
load_from = None

# custom_hooks = [
#     dict(
#         type='MEGVIIEMAHook',
#         init_updates=0,
#         priority='NORMAL',
#     ),
# ]

find_unused_parameters = True
logger_name = 'mmdet'