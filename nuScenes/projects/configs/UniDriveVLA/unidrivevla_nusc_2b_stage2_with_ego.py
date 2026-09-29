checkpoint_config = dict(interval=879, deepspeed=True, max_keep_ckpts=2)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs2/UniDriveVLA/unidrivevla_stage2_2b_canbus_sft3ep_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs4_accum4'
load_from = None
resume_from = None
workflow = [('train', 1)]
vlm_pretrained_path = '/path/to/nuScenes/work_dirs2/unidrivevla_stage1_2b_full_ego_canbus_pure_ms_swift_3ep/checkpoint-660'
occworld_vae_path = '/path/to/occvae_latest.pth'
deepspeed_config = 'zero_configs/adam_zero1_bf16.json'
data_infos_root = 'data/infos'
enable_coc = True
coc_jsonl_path = 'data/nuscenes/alpamayo_coc_train_with_nav.jsonl'
coc_planning_loss_weight = 0.0
coc_policy_prefix = False
coc_context_dropout = 0.0
coc_language_loss_weight = 0.1
coc_language_output = True
coc_language_target_format = 'cot'
coc_generate = False
coc_eval_source = 'pred'
coc_eval_injection = 'answer'
driving_prefix_max_len = 3840
num_gpus = 8
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
version = 'trainval'
length = dict(trainval=28130, mini=323)
total_batch_size = 128
batch_size = 4
num_iters_per_epoch = 879
num_epochs = 20
total_epochs = 20
checkpoint_epoch_interval = 1
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_shape = (960, 544)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_class_names = ['ped_crossing', 'divider', 'boundary']
num_classes = 10
num_map_classes = 3
roi_size = (30, 60)
num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = 4
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True
ego_feature_map_scale = (34, 60)
single_frame_layer = [
    'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable', 'concat',
    'ffn', 'norm', 'split', 'refine'
]
temporal_frame_layer = [
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine'
]
unified_decoder_operation_order = [
    'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable', 'concat',
    'ffn', 'norm', 'split', 'refine', 'concat', 'temp_gnn', 'gnn', 'inter_gnn',
    'norm', 'split', 'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine', 'concat', 'temp_gnn', 'gnn',
    'inter_gnn', 'norm', 'split', 'deformable', 'concat', 'ffn', 'norm',
    'split', 'refine', 'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm',
    'split', 'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine'
]
unified_decoder_cfg = dict(
    type='UnifiedPerceptionDecoder',
    embed_dims=256,
    task_select=['det', 'map', 'ego'],
    query_select=['det', 'map', 'ego'],
    num_stage1_layers=3,
    num_stage2_layers=3,
    num_single_frame_decoder=1,
    cls_threshold_to_reg=0.05,
    decouple_attn=True,
    use_vlm_in_stage2=True,
    operation_order=[
        'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable', 'concat',
        'ffn', 'norm', 'split', 'refine', 'concat', 'temp_gnn', 'gnn',
        'inter_gnn', 'norm', 'split', 'deformable', 'concat', 'ffn', 'norm',
        'split', 'refine', 'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm',
        'split', 'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
        'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split',
        'deformable', 'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
        'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
        'concat', 'ffn', 'norm', 'split', 'refine', 'concat', 'temp_gnn',
        'gnn', 'inter_gnn', 'norm', 'split', 'deformable', 'concat', 'ffn',
        'norm', 'split', 'refine'
    ],
    det_instance_bank=dict(
        type='InstanceBank',
        num_anchor=900,
        embed_dims=256,
        anchor='data/kmeans/kmeans_det_900.npy',
        anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
        num_temp_instances=600,
        confidence_decay=0.6,
        feat_grad=False),
    map_instance_bank=dict(
        type='InstanceBank',
        num_anchor=100,
        embed_dims=256,
        anchor='data/kmeans/kmeans_map_100.npy',
        anchor_handler=dict(type='SparsePoint3DKeyPointsGenerator'),
        num_temp_instances=0,
        confidence_decay=0.6,
        feat_grad=True),
    ego_instance_bank=dict(
        type='EgoInstanceBank',
        embed_dims=256,
        anchor_type='nus',
        num_temp_instances=1,
        feature_map_scale=(34, 60)),
    det_anchor_encoder=dict(
        type='SparseBox3DEncoder',
        vel_dims=3,
        embed_dims=[128, 32, 32, 64],
        mode='cat',
        output_fc=False,
        in_loops=1,
        out_loops=4),
    map_anchor_encoder=dict(
        type='SparsePoint3DEncoder', embed_dims=256, num_sample=20),
    graph_model=dict(
        type='SeparateAttention',
        query_select=['det', 'map', 'ego'],
        separate_list=[['det'], ['map']],
        decouple_list=[True, False],
        attn=[
            dict(
                type='MultiheadFlashAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1)
        ]),
    temp_graph_model=dict(
        type='TemporalSeparateAttention',
        query_select=['det', 'map', 'ego'],
        query_list=[['det'], ['map'], ['ego']],
        key_list=[['det'], ['map'], ['det', 'map']],
        decouple_list=[True, False, False],
        attn=[
            dict(
                type='MultiheadFlashAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1)
        ]),
    inter_graph_model=dict(
        type='InteractiveAttention',
        query_select=['det', 'map', 'ego'],
        query_list=[['ego']],
        key_list=[['det', 'map']],
        decouple_list=[False],
        attn=[
            dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1)
        ]),
    det_deformable=dict(
        type='DeformableFeatureAggregation',
        embed_dims=256,
        num_groups=8,
        num_levels=4,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=True,
        use_camera_embed=True,
        residual_mode='cat',
        kps_generator=dict(
            type='SparseBox3DKeyPointsGenerator',
            num_learnable_pts=6,
            fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0],
                       [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]])),
    map_deformable=dict(
        type='DeformableFeatureAggregation',
        embed_dims=256,
        num_groups=8,
        num_levels=4,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=True,
        use_camera_embed=True,
        residual_mode='cat',
        kps_generator=dict(
            type='SparsePoint3DKeyPointsGenerator',
            embed_dims=256,
            num_sample=20,
            num_learnable_pts=3,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023)),
    ego_deformable=dict(
        type='DeformableFeatureAggregation',
        embed_dims=256,
        num_groups=8,
        num_levels=4,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=True,
        use_camera_embed=True,
        residual_mode='cat',
        kps_generator=dict(
            type='SparseBox3DKeyPointsGenerator',
            num_learnable_pts=6,
            fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0],
                       [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]])),
    ffn=dict(
        type='AsymmetricFFN',
        in_channels=512,
        pre_norm=dict(type='LN'),
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.1,
        act_cfg=dict(type='ReLU', inplace=True)),
    norm_layer=dict(type='LN', normalized_shape=256),
    det_refine_layer=dict(
        type='SparseBox3DRefinementModule',
        embed_dims=256,
        num_cls=10,
        refine_yaw=True,
        with_quality_estimation=True),
    map_refine_layer=dict(
        type='SparsePoint3DRefinementModule',
        embed_dims=256,
        num_sample=20,
        num_cls=3),
    ego_refine_layer=dict(
        type='EgoStatusRefinementModule', embed_dims=256, status_dims=10),
    motion_refine_layer=dict(
        type='SparseMotionRefinementModule',
        embed_dims=256,
        fut_ts=12,
        fut_mode=6),
    det_sampler=dict(
        type='SparseBox3DTarget',
        num_dn_groups=0,
        num_temp_dn_groups=0,
        dn_noise_scale=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        max_dn_gt=32,
        add_neg_dn=True,
        cls_weight=2.0,
        box_weight=0.25,
        reg_weights=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        cls_wise_reg_weights=dict(
            {9: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]})),
    map_sampler=dict(
        type='SparsePoint3DTarget',
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(
                    type='LinesL1Cost', weight=10.0, beta=0.01,
                    permute=True))),
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60)),
    motion_sampler=dict(type='SparseMotionTarget'),
    det_decoder=dict(type='SparseBox3DDecoder'),
    map_decoder=dict(type='SparsePoint3DDecoder'),
    loss_det_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0),
    loss_det_reg=dict(
        type='SparseBox3DLoss',
        loss_box=dict(type='L1Loss', loss_weight=0.25),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True),
        loss_yawness=dict(type='GaussianFocalLoss'),
        cls_allow_reverse=[5]),
    loss_map_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    loss_map_reg=dict(
        type='SparseLineLoss',
        loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
        num_sample=20,
        roi_size=(30, 60)),
    loss_ego_status=dict(type='L1Loss', loss_weight=1.0),
    loss_motion_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=0.0),
    loss_motion_reg=dict(type='L1Loss', loss_weight=0.0),
    det_reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    map_reg_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ],
    motion_anchor='data/kmeans/kmeans_motion_6.npy')
vlm_lr_mult = 0.5
lora_cfg = dict(
    r=64,
    lora_alpha=128,
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
        'down_proj'
    ],
    lora_dropout=0.0,
    bias='none')
loss_planning = dict(
    type='FlowPlanningLoss',
    use_min_snr_loss=True,
    min_snr_gamma=5.0,
    loss_weight=1.0,
    hybrid_loss_weight=0.5,
    detach_window_size=3)
model = dict(
    type='UniDriveVLA',
    planning_head=dict(
        type='QwenVL3APlanningHead',
        pretrained_path=
        '/path/to/nuScenes/work_dirs2/unidrivevla_stage1_2b_full_ego_canbus_pure_ms_swift_3ep/checkpoint-660',
        action_dim=2,
        action_horizon=6,
        dtype='bfloat16',
        train_vlm=True,
        coc_jsonl_path=
        'data/nuscenes/alpamayo_coc_train_with_nav.jsonl',
        enable_coc=True,
        enable_coc_policy_prefix=False,
        enable_coc_context=False,
        coc_context_format='nav_cot',
        coc_context_dropout=0.0,
        coc_planning_loss_weight=0.0,
        enable_coc_language_output=True,
        coc_language_target_format='cot',
        coc_language_loss_weight=0.1,
        enable_coc_generate=False,
        coc_eval_source='pred',
        coc_eval_injection='answer',
        driving_prefix_max_len=3840,
        loss_planning=dict(
            type='FlowPlanningLoss',
            use_min_snr_loss=True,
            min_snr_gamma=5.0,
            loss_weight=1.0,
            hybrid_loss_weight=0.5,
            detach_window_size=3),
        with_depth_supervision=False,
        depth_loss_weight=0.2,
        occworld_vae_config=dict(
            type='VAERes3D',
            encoder_cfg=dict(
                type='Encoder2D',
                ch=64,
                out_ch=64,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50, ),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=128,
                double_z=False),
            decoder_cfg=dict(
                type='Decoder3D',
                ch=64,
                out_ch=128,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50, ),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=64,
                give_pre_end=False),
            num_classes=18,
            expansion=8,
            vqvae_cfg=None),
        occworld_vae_path=
        '/path/to/occvae_latest.pth',
        feat_grad=True,
        feature_source='raw',
        unified_decoder_cfg=dict(
            type='UnifiedPerceptionDecoder',
            embed_dims=256,
            task_select=['det', 'map', 'ego'],
            query_select=['det', 'map', 'ego'],
            num_stage1_layers=3,
            num_stage2_layers=3,
            num_single_frame_decoder=1,
            cls_threshold_to_reg=0.05,
            decouple_attn=True,
            use_vlm_in_stage2=True,
            operation_order=[
                'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine'
            ],
            det_instance_bank=dict(
                type='InstanceBank',
                num_anchor=900,
                embed_dims=256,
                anchor='data/kmeans/kmeans_det_900.npy',
                anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
                num_temp_instances=600,
                confidence_decay=0.6,
                feat_grad=False),
            map_instance_bank=dict(
                type='InstanceBank',
                num_anchor=100,
                embed_dims=256,
                anchor='data/kmeans/kmeans_map_100.npy',
                anchor_handler=dict(type='SparsePoint3DKeyPointsGenerator'),
                num_temp_instances=0,
                confidence_decay=0.6,
                feat_grad=True),
            ego_instance_bank=dict(
                type='EgoInstanceBank',
                embed_dims=256,
                anchor_type='nus',
                num_temp_instances=1,
                feature_map_scale=(34, 60)),
            det_anchor_encoder=dict(
                type='SparseBox3DEncoder',
                vel_dims=3,
                embed_dims=[128, 32, 32, 64],
                mode='cat',
                output_fc=False,
                in_loops=1,
                out_loops=4),
            map_anchor_encoder=dict(
                type='SparsePoint3DEncoder', embed_dims=256, num_sample=20),
            graph_model=dict(
                type='SeparateAttention',
                query_select=['det', 'map', 'ego'],
                separate_list=[['det'], ['map']],
                decouple_list=[True, False],
                attn=[
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=512,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1),
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1)
                ]),
            temp_graph_model=dict(
                type='TemporalSeparateAttention',
                query_select=['det', 'map', 'ego'],
                query_list=[['det'], ['map'], ['ego']],
                key_list=[['det'], ['map'], ['det', 'map']],
                decouple_list=[True, False, False],
                attn=[
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=512,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1),
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1),
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1)
                ]),
            inter_graph_model=dict(
                type='InteractiveAttention',
                query_select=['det', 'map', 'ego'],
                query_list=[['ego']],
                key_list=[['det', 'map']],
                decouple_list=[False],
                attn=[
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1)
                ]),
            det_deformable=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparseBox3DKeyPointsGenerator',
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                               [0, 0, -0.45]])),
            map_deformable=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparsePoint3DKeyPointsGenerator',
                    embed_dims=256,
                    num_sample=20,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023)),
            ego_deformable=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparseBox3DKeyPointsGenerator',
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                               [0, 0, -0.45]])),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=512,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            norm_layer=dict(type='LN', normalized_shape=256),
            det_refine_layer=dict(
                type='SparseBox3DRefinementModule',
                embed_dims=256,
                num_cls=10,
                refine_yaw=True,
                with_quality_estimation=True),
            map_refine_layer=dict(
                type='SparsePoint3DRefinementModule',
                embed_dims=256,
                num_sample=20,
                num_cls=3),
            ego_refine_layer=dict(
                type='EgoStatusRefinementModule',
                embed_dims=256,
                status_dims=10),
            motion_refine_layer=dict(
                type='SparseMotionRefinementModule',
                embed_dims=256,
                fut_ts=12,
                fut_mode=6),
            det_sampler=dict(
                type='SparseBox3DTarget',
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[
                    2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                cls_wise_reg_weights=dict(
                    {9: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]})),
            map_sampler=dict(
                type='SparsePoint3DTarget',
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(
                            type='LinesL1Cost',
                            weight=10.0,
                            beta=0.01,
                            permute=True))),
                num_cls=3,
                num_sample=20,
                roi_size=(30, 60)),
            motion_sampler=dict(type='SparseMotionTarget'),
            det_decoder=dict(type='SparseBox3DDecoder'),
            map_decoder=dict(type='SparsePoint3DDecoder'),
            loss_det_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_det_reg=dict(
                type='SparseBox3DLoss',
                loss_box=dict(type='L1Loss', loss_weight=0.25),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True),
                loss_yawness=dict(type='GaussianFocalLoss'),
                cls_allow_reverse=[5]),
            loss_map_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_map_reg=dict(
                type='SparseLineLoss',
                loss_line=dict(
                    type='LinesL1Loss', loss_weight=10.0, beta=0.01),
                num_sample=20,
                roi_size=(30, 60)),
            loss_ego_status=dict(type='L1Loss', loss_weight=1.0),
            loss_motion_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.0),
            loss_motion_reg=dict(type='L1Loss', loss_weight=0.0),
            det_reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            map_reg_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0
            ],
            motion_anchor='data/kmeans/kmeans_motion_6.npy'),
        det_vla_head_cfg=None,
        map_vla_head_cfg=None,
        lora_cfg=None,
        driving_deepstack=True,
        vlm_fusion_cfg=dict(type='direct'),
        feature_fusion_cfg=dict(type='none'),
        joint_transformer_checkpoint=False,
        visual_lora_cfg=dict(
            type='lora',
            r=64,
            lora_alpha=128,
            target_modules=
            '^(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$',
            lora_dropout=0.0,
            bias='none'),
        visual_wasp_cfg=dict(
            type='wasp',
            rank=64,
            alpha=128,
            top_k=16,
            beta=0.01,
            target_modules=['qkv', 'proj', 'linear_fc1', 'linear_fc2'],
            expected_num_layers=96,
            dropout=0.0,
            eps=1e-06),
        peft_cfg=dict(
            type='wasp',
            rank=64,
            alpha=128,
            top_k=16,
            beta=0.003,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
                'down_proj'
            ],
            expected_num_layers=196,
            dropout=0.0,
            eps=1e-06),
        visual_projector_wasp_cfg=None,
        include_ego_status_prompt=True,
        attn_implementation='sdpa',
        visual_gradient_checkpointing=False),
    task_loss_weight=dict(planning=1.0))
dataset_type = 'NuScenes3DDataset'
data_root = 'data/nuscenes/'
anno_root = 'data/infos/'
info_root = 'data/infos/'
file_client_args = dict(backend='disk')
ann_file_train = 'data/infos/nuscenes_infos_train.pkl'
ann_file_val = 'data/infos/nuscenes_infos_val.pkl'
ann_file_test = 'data/infos/nuscenes_infos_val.pkl'
vad_ann_file_train = 'data/infos/vad_nuscenes_infos_temporal_train.pkl'
vad_ann_file_val = 'data/infos/vad_nuscenes_infos_temporal_val.pkl'
vad_ann_file_test = 'data/infos/vad_nuscenes_infos_temporal_val.pkl'
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(
        type='LoadOccWorldLabels',
        data_root='data/nuscenes',
        input_dataset='gts'),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(type='BBoxRotation'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='InstanceNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='VectorizeMap',
        roi_size=(30, 60),
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'gt_bboxes_3d',
            'gt_labels_3d', 'gt_map_labels', 'gt_map_pts',
            'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs',
            'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status',
            'gt_ego_lcf_feat', 'gt_ego_his_trajs', 'gt_occ_dense', 'hist_traj'
        ],
        meta_keys=[
            'token', 'sample_idx', 'T_global', 'T_global_inv', 'timestamp',
            'instance_id'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status',
            'gt_ego_lcf_feat', 'gt_ego_his_trajs', 'gt_ego_fut_cmd',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'hist_traj'
        ],
        meta_keys=[
            'token', 'sample_idx', 'T_global', 'T_global_inv', 'timestamp'
        ])
]
eval_pipeline = [
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='InstanceNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(
        type='VectorizeMap', roi_size=(30, 60), simplify=True,
        normalize=False),
    dict(
        type='Collect',
        keys=[
            'vectors', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_agent_fut_trajs',
            'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks',
            'gt_ego_fut_cmd', 'fut_boxes', 'hist_traj'
        ],
        meta_keys=['token', 'timestamp'])
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
data_basic_config = dict(
    type='NuScenes3DDataset',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    map_classes=['ped_crossing', 'divider', 'boundary'],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    version='v1.0-trainval')
eval_config = dict(
    type='NuScenes3DDataset',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    map_classes=['ped_crossing', 'divider', 'boundary'],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    version='v1.0-trainval',
    ann_file='data/infos/nuscenes_infos_val.pkl',
    vad_ann_file=
    'data/infos/vad_nuscenes_infos_temporal_val.pkl',
    pipeline=[
        dict(
            type='CircleObjectRangeFilter',
            class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
        dict(
            type='InstanceNameFilter',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]),
        dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
        dict(
            type='VectorizeMap',
            roi_size=(30, 60),
            simplify=True,
            normalize=False),
        dict(
            type='Collect',
            keys=[
                'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs',
                'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'fut_boxes', 'hist_traj'
            ],
            meta_keys=['token', 'timestamp'])
    ],
    test_mode=True)
data_aug_conf = dict(
    resize_lim=(0.6, 0.6),
    final_dim=(544, 960),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[0, 0])
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='NuScenes3DDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/infos/nuscenes_infos_train.pkl',
        vad_ann_file=
        'data/infos/vad_nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='LoadOccWorldLabels',
                data_root='data/nuscenes',
                input_dataset='gts'),
            dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
            dict(type='BBoxRotation'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='CircleObjectRangeFilter',
                class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
            dict(
                type='InstanceNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='VectorizeMap',
                roi_size=(30, 60),
                simplify=False,
                normalize=False,
                sample_num=20,
                permute=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'gt_bboxes_3d', 'gt_labels_3d', 'gt_map_labels',
                    'gt_map_pts', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                    'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
                    'ego_status', 'gt_ego_lcf_feat', 'gt_ego_his_trajs',
                    'gt_occ_dense', 'hist_traj'
                ],
                meta_keys=[
                    'token', 'sample_idx', 'T_global', 'T_global_inv',
                    'timestamp', 'instance_id'
                ])
        ],
        test_mode=False,
        data_aug_conf=dict(
            resize_lim=(0.6, 0.6),
            final_dim=(544, 960),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True),
    val=dict(
        type='NuScenes3DDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/infos/nuscenes_infos_val.pkl',
        vad_ann_file=
        'data/infos/vad_nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'ego_status', 'gt_ego_lcf_feat', 'gt_ego_his_trajs',
                    'gt_ego_fut_cmd', 'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                    'hist_traj'
                ],
                meta_keys=[
                    'token', 'sample_idx', 'T_global', 'T_global_inv',
                    'timestamp'
                ])
        ],
        data_aug_conf=dict(
            resize_lim=(0.6, 0.6),
            final_dim=(544, 960),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        test_mode=True,
        eval_config=dict(
            type='NuScenes3DDataset',
            data_root='data/nuscenes/',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            map_classes=['ped_crossing', 'divider', 'boundary'],
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            version='v1.0-trainval',
            ann_file='data/infos/nuscenes_infos_val.pkl',
            vad_ann_file=
            'data/infos/vad_nuscenes_infos_temporal_val.pkl',
            pipeline=[
                dict(
                    type='CircleObjectRangeFilter',
                    class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
                dict(
                    type='InstanceNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
                dict(
                    type='VectorizeMap',
                    roi_size=(30, 60),
                    simplify=True,
                    normalize=False),
                dict(
                    type='Collect',
                    keys=[
                        'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                        'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                        'gt_ego_fut_cmd', 'fut_boxes', 'hist_traj'
                    ],
                    meta_keys=['token', 'timestamp'])
            ],
            test_mode=True)),
    test=dict(
        type='NuScenes3DDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/infos/nuscenes_infos_val.pkl',
        vad_ann_file=
        'data/infos/vad_nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'ego_status', 'gt_ego_lcf_feat', 'gt_ego_his_trajs',
                    'gt_ego_fut_cmd', 'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                    'hist_traj'
                ],
                meta_keys=[
                    'token', 'sample_idx', 'T_global', 'T_global_inv',
                    'timestamp'
                ])
        ],
        data_aug_conf=dict(
            resize_lim=(0.6, 0.6),
            final_dim=(544, 960),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        test_mode=True,
        eval_config=dict(
            type='NuScenes3DDataset',
            data_root='data/nuscenes/',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            map_classes=['ped_crossing', 'divider', 'boundary'],
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            version='v1.0-trainval',
            ann_file='data/infos/nuscenes_infos_val.pkl',
            vad_ann_file=
            'data/infos/vad_nuscenes_infos_temporal_val.pkl',
            pipeline=[
                dict(
                    type='CircleObjectRangeFilter',
                    class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
                dict(
                    type='InstanceNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
                dict(
                    type='VectorizeMap',
                    roi_size=(30, 60),
                    simplify=True,
                    normalize=False),
                dict(
                    type='Collect',
                    keys=[
                        'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                        'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                        'gt_ego_fut_cmd', 'fut_boxes', 'hist_traj'
                    ],
                    meta_keys=['token', 'timestamp'])
            ],
            test_mode=True)),
    shuffler_sampler=dict(type='GroupInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
deepspeed = True
gradient_checkpointing = dict(
    enabled=True, checkpoint_activations=True, checkpoint_attention=True)
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=1e-07,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict({
            'planning_head.qwen3_vl_with_expert.qwen3_vl':
            dict(lr_mult=0.5, decay_mult=1.0)
        })))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='IterBasedRunner', max_iters=17580)
eval_mode = dict(
    with_det=True,
    with_tracking=True,
    with_map=True,
    with_motion=False,
    with_planning=True,
    with_ego_status=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2)
evaluation = dict(
    interval=17580,
    eval_mode=dict(
        with_det=True,
        with_tracking=True,
        with_map=True,
        with_motion=False,
        with_planning=True,
        with_ego_status=True,
        tracking_threshold=0.2,
        motion_threshhold=0.2))
custom_hooks = [
    dict(type='WASPAccumulationHook', enabled=True, priority='VERY_HIGH')
]
find_unused_parameters = True
logger_name = 'mmdet'
visual_projector_target_modules = '^(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$'
visual_lora_cfg = dict(
    type='lora',
    r=64,
    lora_alpha=128,
    target_modules='^(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$',
    lora_dropout=0.0,
    bias='none')
visual_wasp_cfg = dict(
    type='wasp',
    rank=64,
    alpha=128,
    top_k=16,
    beta=0.01,
    target_modules=['qkv', 'proj', 'linear_fc1', 'linear_fc2'],
    expected_num_layers=96,
    dropout=0.0,
    eps=1e-06)
llm_wasp_cfg = dict(
    type='wasp',
    rank=64,
    alpha=128,
    top_k=16,
    beta=0.003,
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
        'down_proj'
    ],
    expected_num_layers=196,
    dropout=0.0,
    eps=1e-06)
projector_lora_cfg = dict(
    type='lora',
    r=64,
    lora_alpha=128,
    target_modules='^(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$',
    lora_dropout=0.0,
    bias='none')
vision_wasp_cfg = dict(
    type='wasp',
    rank=64,
    alpha=128,
    top_k=16,
    beta=0.01,
    target_modules=['qkv', 'proj', 'linear_fc1', 'linear_fc2'],
    expected_num_layers=96,
    dropout=0.0,
    eps=1e-06)
gpu_ids = range(0, 8)

# Keep the released recipe identical to the checkpoint recipe while making
# paths portable outside the original training machine.
import os as _os
_configured_root = _os.environ.get("NUSCENES_ROOT")
_release_root = _os.path.abspath(
    _configured_root if _configured_root else _os.getcwd()
)
_source_root = "/path/to/original/nuScenes"
_data_root = _os.path.abspath(
    _os.environ.get("NUSCENES_DATA_ROOT", _os.path.join(_release_root, "data/nuscenes"))
)
_info_root = _os.path.abspath(
    _os.environ.get("DATA_INFOS_ROOT", _os.path.join(_release_root, "data/infos"))
)
_anchor_root = _os.path.abspath(
    _os.environ.get("NUSCENES_ANCHOR_ROOT", _os.path.join(_release_root, "data/kmeans"))
)


def _rewrite_release_paths(value):
    if isinstance(value, str):
        rewritten = value.replace(_source_root, _release_root)
        for relative_root, external_root in (("data/nuscenes", _data_root), ("data/infos", _info_root), ("data/kmeans", _anchor_root)):
            release_prefix = _release_root.rstrip("/") + "/" + relative_root
            if rewritten == relative_root or rewritten.startswith(relative_root + "/"):
                rewritten = external_root + rewritten[len(relative_root):]
            elif rewritten == release_prefix or rewritten.startswith(release_prefix + "/"):
                rewritten = external_root + rewritten[len(release_prefix):]
        return rewritten
    if isinstance(value, dict):
        return {key: _rewrite_release_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_release_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_release_paths(item) for item in value)
    return value


for _name in (
    "model", "data", "data_basic_config", "eval_config",
    "train_pipeline", "test_pipeline", "eval_pipeline",
):
    if _name in globals():
        globals()[_name] = _rewrite_release_paths(globals()[_name])
vlm_pretrained_path = _os.environ.get(
    "VLM_PRETRAINED_PATH",
    "/path/to/unidrivevla_stage1_2b_full_ego_canbus_pure_ms_swift_3ep/checkpoint-660",
)
occworld_vae_path = _os.environ.get("OCCWORLD_VAE_PATH", "/path/to/occvae_latest.pth")
deepspeed_config = _os.environ.get(
    "DEEPSPEED_CONFIG", _release_root + "/zero_configs/adam_zero1_bf16.json"
)
data_infos_root = _info_root
data_root = _data_root.rstrip("/") + "/"
anno_root = _info_root.rstrip("/") + "/"
info_root = _info_root.rstrip("/") + "/"
ann_file_train = _os.path.join(_info_root, "nuscenes_infos_train.pkl")
ann_file_val = _os.path.join(_info_root, "nuscenes_infos_val.pkl")
ann_file_test = ann_file_val
vad_ann_file_train = _os.path.join(_info_root, "vad_nuscenes_infos_temporal_train.pkl")
vad_ann_file_val = _os.path.join(_info_root, "vad_nuscenes_infos_temporal_val.pkl")
vad_ann_file_test = vad_ann_file_val
motion_anchor_path = _os.environ.get(
    "MOTION_ANCHOR", _os.path.join(_anchor_root, "kmeans_motion_6.npy")
)
work_dir = _os.environ.get(
    "WORK_DIR",
    "work_dirs2/UniDriveVLA/unidrivevla_stage2_2b_canbus_sft3ep_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs4_accum4",
)
coc_jsonl_path = _os.environ.get(
    "COC_JSONL_PATH", _release_root + "/data/nuscenes/alpamayo_coc_train_with_nav.jsonl"
)
if "model" in globals() and "planning_head" in model:
    _planning_head = model["planning_head"]
    _planning_head["pretrained_path"] = vlm_pretrained_path
    _planning_head["occworld_vae_path"] = occworld_vae_path
    _planning_head["coc_jsonl_path"] = coc_jsonl_path
    _planning_head["unified_decoder_cfg"]["motion_anchor"] = motion_anchor_path

del _configured_root, _release_root, _source_root, _data_root
del _info_root, _anchor_root, _name, _planning_head
