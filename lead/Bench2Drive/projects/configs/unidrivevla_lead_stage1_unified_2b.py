# UniDriveVLA LEAD Stage1 Configuration
# Based on B2D Stage1. Replaces B2D infos/map data with LEAD official
# meta/bbox/rgb data converted by tools/generate_lead_stage1_infos.py.

_base_ = "./unidrivevla_b2d_stage1_unified_2b_no_cotraining.py"

import json
import math
import os

custom_imports = dict(
    imports=[
        "lead_stage1_plugin.lead_dataset",
        "lead_stage1_plugin.lead_loading",
        "lead_stage1_plugin.lead_ego_instance_bank",
    ],
    allow_failed_imports=False,
)

project_dir = os.environ.get(
    "PROJECT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)
vlm_pretrained_path = os.environ.get(
    "VLM_PRETRAINED_PATH",
    "/path/to/Qwen3-VL-2B-Instruct",
)
deepspeed_config = os.environ.get(
    "DEEPSPEED_CONFIG",
    os.path.join(project_dir, "zero_configs/adam_zero1_bf16.json"),
)
num_gpus = int(os.environ.get("NUM_GPUS", 64))

lead_attn_implementation = os.environ.get(
    "LEAD_STAGE1_ATTN_IMPLEMENTATION", "eager"
).strip().lower()
if lead_attn_implementation not in {"eager", "sdpa", "flex"}:
    raise ValueError(
        f"Unsupported LEAD_STAGE1_ATTN_IMPLEMENTATION={lead_attn_implementation!r}"
    )
batch_size = int(os.environ.get("SAMPLES_PER_GPU", 1))
num_epochs = int(os.environ.get("LEAD_STAGE1_EPOCHS", 3))
lead_stage1_lr = float(os.environ.get("LEAD_STAGE1_LR", "2e-4"))
lead_sequences_split_num = int(
    os.environ.get("LEAD_SEQUENCES_SPLIT_NUM", "2")
)
if lead_stage1_lr <= 0.0:
    raise ValueError(f"LEAD_STAGE1_LR must be positive, got {lead_stage1_lr}")
if lead_sequences_split_num <= 0:
    raise ValueError(
        "LEAD_SEQUENCES_SPLIT_NUM must be positive, got "
        f"{lead_sequences_split_num}"
    )
load_from = os.environ.get("LEAD_STAGE1_LOAD_FROM", "").strip() or None
resume_from = os.environ.get("LEAD_STAGE1_RESUME_FROM", "").strip() or None
lead_log_interval = int(os.environ.get("LEAD_STAGE1_LOG_INTERVAL", "10"))
if lead_log_interval <= 0:
    raise ValueError(
        f"LEAD_STAGE1_LOG_INTERVAL must be positive, got {lead_log_interval}"
    )
if load_from and resume_from:
    raise ValueError(
        "Set only one of LEAD_STAGE1_LOAD_FROM and LEAD_STAGE1_RESUME_FROM"
    )
lead_sample_stride = int(os.environ.get("LEAD_SAMPLE_STRIDE", 1))
lead_sample_offset = int(os.environ.get("LEAD_SAMPLE_OFFSET", 0))
lead_info_frame_stride = int(os.environ.get("LEAD_INFO_FRAME_STRIDE", "4"))
lead_info_frame_offset = int(os.environ.get("LEAD_INFO_FRAME_OFFSET", "0"))
lead_info_shard_cache_size = int(os.environ.get("LEAD_INFO_SHARD_CACHE_SIZE", "32"))
if lead_info_shard_cache_size <= 0:
    raise ValueError("LEAD_INFO_SHARD_CACHE_SIZE must be positive")
if lead_info_frame_stride not in (1, 4) or lead_info_frame_offset != 0:
    raise ValueError(
        "LEAD Stage1 supports native 4Hz infos (stride=1) or the legacy "
        "1Hz infos (stride=4), both with offset=0"
    )
lead_num_train_samples = int(os.environ.get("LEAD_NUM_TRAIN_SAMPLES", 245444))
lead_val_ratio = float(os.environ.get("LEAD_VAL_RATIO", 0.0))
lead_split_strategy = os.environ.get("LEAD_SPLIT_STRATEGY", "hash")
lead_sensor_perturbation = os.environ.get("LEAD_SENSOR_PERTURBATION", "1") == "1"
lead_sensor_perturbation_probability = float(
    os.environ.get("LEAD_SENSOR_PERTURBATION_PROB", "0.5")
)
if not 0.0 <= lead_sensor_perturbation_probability <= 1.0:
    raise ValueError(
        "LEAD_SENSOR_PERTURBATION_PROB must be in [0, 1], got "
        + repr(lead_sensor_perturbation_probability)
    )
lead_effective_sensor_perturbation_probability = (
    lead_sensor_perturbation_probability if lead_sensor_perturbation else 0.0
)
num_iters_per_epoch = max(
    1, math.ceil(lead_num_train_samples / max(1, num_gpus * batch_size))
)
checkpoint_interval = int(os.environ.get("LEAD_STAGE1_CKPT_INTERVAL", num_iters_per_epoch))
lead_max_iters_override = os.environ.get("LEAD_STAGE1_MAX_ITERS", "").strip()
runner_max_iters = (
    int(lead_max_iters_override)
    if lead_max_iters_override
    else num_iters_per_epoch * num_epochs
)
if runner_max_iters <= 0:
    raise ValueError(f"LEAD_STAGE1_MAX_ITERS must be positive, got {runner_max_iters}")
max_keep_ckpts = int(os.environ.get("LEAD_STAGE1_MAX_KEEP_CKPTS", 1))
total_epochs = num_epochs
lead_split_group = 1  # Preserve adjacent frames for the temporal cache.

# LEAD official CARLA RGB is three 384x384 cameras concatenated into a 1152x384 strip.
input_shape = (384, 384)
num_cams = 3
ego_feature_map_scale = (384 // 16, 384 // 16)
lead_plan_hz = 4
# Official LEAD RGB is saved at 4Hz. The legacy infos keep every fourth frame;
# full infos keep every saved frame.
lead_perception_hz = 4 // lead_info_frame_stride
lead_plan_horizon = 8
lead_plan_horizon_seconds = 2.0
lead_target_speed_classes = [0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0]
lead_route_points = 10
lead_trajectory_stats_path = os.environ.get(
    "LEAD_TRAJECTORY_STATS_PATH",
    os.path.join(project_dir, "data/lead/infos/lead_trajectory_stats.json"),
)
lead_expected_infos_version = int(
    os.environ.get("LEAD_EXPECTED_INFOS_VERSION", "14")
)
if not os.path.isfile(lead_trajectory_stats_path):
    raise FileNotFoundError(
        f"Missing LEAD trajectory statistics. Regenerate version-{lead_expected_infos_version} infos with "
        "tools/generate_lead_stage1_infos.py: " + lead_trajectory_stats_path
    )
with open(lead_trajectory_stats_path, "r", encoding="utf-8") as stats_file:
    lead_trajectory_stats = json.load(stats_file)
del stats_file
expected_trajectory_stats = {
    "stats_version": 7,
    "lead_stage1_infos_version": lead_expected_infos_version,
    "sensor_perturbation_version": 1,
    "detection_filter_version": 4,
    "detection_class_mapping_version": 2,
    "detection_z_range": [-4.0, 4.0],
    "detection_visibility_filter": "lead_b2d_visibility_v1",
    "detection_traffic_control_filter": "tfv6_affects_ego_non_green_v1",
    "detection_box_dimension_order": "length_width_height",
    "detection_box_center": "carla_bbox_center",
    "bbox_center_offset_version": 1,
    "sensor_perturbation_probability": lead_effective_sensor_perturbation_probability,
    "num_sensor_perturbation_samples": (
        lead_num_train_samples
        if lead_effective_sensor_perturbation_probability > 0.0
        else 0
    ),
    "perturbed_image_layout": "rgb_perturbated_strip",
    "frame_sample_stride": lead_info_frame_stride,
    "frame_sample_offset": lead_info_frame_offset,
    "route_quality_filter": os.environ.get("LEAD_ROUTE_QUALITY_FILTER", "tfv6_posttrain"),
    "route_skip_first": int(os.environ.get("LEAD_ROUTE_SKIP_FIRST", "8")),
    "route_skip_last": int(os.environ.get("LEAD_ROUTE_SKIP_LAST", "8")),
    "source_sim_hz": 20,
    "data_save_freq": 5,
    "jpeg_storage_quality": 30,
    "perception_sample_hz": float(lead_perception_hz),
    "planning_step_sim_frames": 5,
    "planning_freq_hz": 4.0,
    "planning_horizon_steps": lead_plan_horizon,
    "planning_horizon_seconds": lead_plan_horizon_seconds,
    "num_train_samples": lead_num_train_samples,
    "val_ratio": lead_val_ratio,
    "split_strategy": lead_split_strategy,
}
stats_mismatches = {
    key: (lead_trajectory_stats.get(key), expected)
    for key, expected in expected_trajectory_stats.items()
    if lead_trajectory_stats.get(key) != expected
}
if stats_mismatches:
    raise RuntimeError(
        f"LEAD trajectory statistics do not match the requested version-{lead_expected_infos_version} dataset: "
        + repr(stats_mismatches)
        + " in "
        + lead_trajectory_stats_path
    )
lead_delta_mu = list(lead_trajectory_stats["delta_mu"])
lead_delta_std = list(lead_trajectory_stats["delta_std"])
lead_jpeg_quality = int(lead_trajectory_stats["jpeg_storage_quality"])
if (
    len(lead_delta_mu) != 2
    or len(lead_delta_std) != 2
    or not all(math.isfinite(value) for value in lead_delta_mu + lead_delta_std)
    or not all(value > 0.0 for value in lead_delta_std)
):
    raise RuntimeError(
        "Invalid LEAD trajectory delta normalization statistics in "
        + lead_trajectory_stats_path
        + ": mu="
        + repr(lead_delta_mu)
        + ", std="
        + repr(lead_delta_std)
    )
lead_aux_route_loss_weight = float(os.environ.get("LEAD_AUX_ROUTE_LOSS_WEIGHT", "1.0"))
lead_aux_target_speed_loss_weight = float(
    os.environ.get("LEAD_AUX_TARGET_SPEED_LOSS_WEIGHT", "1.0")
)
lead_deterministic_waypoint_loss_weight = float(
    os.environ.get("LEAD_DETERMINISTIC_WAYPOINT_LOSS_WEIGHT", "1.0")
)
lead_planning_query_layout = os.environ.get(
    "LEAD_PLANNING_QUERY_LAYOUT", "action_expert"
).strip()
if lead_planning_query_layout != "action_expert":
    raise ValueError(
        "LEAD Stage1 requires LEAD_PLANNING_QUERY_LAYOUT=action_expert"
    )
lead_system_prompt = (
    "Generalist Autonomous Driving Agent. "
    "Role: You are an advanced multimodal driving model for CARLA Leaderboard 2.0. "
    "Context and coordinate system: the ego vehicle is at the origin. "
    "Trajectory coordinates use (forward, lateral): the first value is forward distance in meters, "
    "and the second value is lateral distance in meters. "
    "Inputs include three synchronized camera views, ego state, and a navigation command. "
    "Planning task: generate a safe, comfortable 2-second trajectory with 8 waypoints at 0.25-second intervals. "
    "Output shape is [8, 2], and each waypoint is (forward, lateral)."
)
lead_user_prompt_template = (
    "You are given 3 synchronized camera views from the ego vehicle. "
    "Predict the future ego trajectory based on visual perception from the 3 camera views, "
    "and the active navigation command [{nav_cmd}]. "
    "Output requirements: predict 8 future trajectory points as (forward, lateral), "
    "use output shape [8,2], and keep 2 decimal places."
)
lead_view_tokens = ["<FRONT_LEFT_VIEW>", "<FRONT_VIEW>", "<FRONT_RIGHT_VIEW>"]
anchor_paths = {
    "det": f"{project_dir}/data/kmeans/lead_det_900.npy",
    "map": f"{project_dir}/data/kmeans/lead_map_100.npy",
}
# The flow-matching planning head consumes gt_ego_fut_trajs_4hz directly and
# does not use a trajectory anchor file.
plan_anchor_refer = ("temp", "4hz")
plan_anchor_types = [("temp", "4hz")]

data_root = "data/lead"
anno_root = os.environ.get(
    "LEAD_INFO_DIR", os.path.join(project_dir, "data/lead/infos")
)
map_info_path = os.environ.get(
    "LEAD_MAP_INFO_PATH",
    os.path.join(project_dir, "data/lead/infos/lead_official_map_infos.pkl"),
)
dataset_type = "LeadDataset"

train_pipeline = [
    dict(type="LoadLeadMultiViewImageFromFiles", to_float32=True, num_cams=num_cams),
    dict(type="ResizeCropFlipImage"),
    dict(type="BBoxRotation"),
    dict(type="LeadTFV6ImageAugmentation", probability=0.2, seed=0),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * 9, z_range=(-4.0, 4.0)),
    dict(
        type="InstanceNameFilter",
        classes=["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"],
    ),
    dict(
        type="VectorizePloyLine",
        roi_size=(30, 60),
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True,
    ),
    dict(type="LeadSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "focal",
            "gt_ego_fut_cmd",
            "target_point_previous",
            "target_point",
            "target_point_next",
            "gt_route",
            "gt_route_masks",
            "gt_target_speed",
            "gt_brake",
            "ego_status",
            "ego_status_mask",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_map_labels",
            "gt_map_pts",
            "gt_agent_fut_trajs",
            "gt_agent_fut_masks",
            "gt_ego_fut_trajs_4hz",
            "gt_ego_fut_masks_4hz",
        ],
        meta_keys=[
            "T_global",
            "T_global_inv",
            "timestamp",
            "instance_id",
            "scene_token",
            "sensor_perturbation_applied",
            "sensor_perturbation_translation",
            "sensor_perturbation_rotation",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadLeadMultiViewImageFromFiles", to_float32=True, num_cams=num_cams),
    dict(type="ResizeCropFlipImage"),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * 9, z_range=(-4.0, 4.0)),
    dict(
        type="InstanceNameFilter",
        classes=["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"],
    ),
    dict(
        type="VectorizePloyLine",
        roi_size=(30, 60),
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True,
    ),
    dict(type="LeadSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "timestamp",
            "projection_mat",
            "image_wh",
            "ego_status",
            "ego_status_mask",
            "gt_map_labels",
            "gt_map_pts",
            "gt_ego_fut_cmd",
            "target_point_previous",
            "target_point",
            "target_point_next",
            "gt_ego_fut_trajs_4hz",
            "gt_ego_fut_masks_4hz",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token"],
    ),
]

inference_only_pipeline = [
    dict(type="LoadLeadMultiViewImageFromFiles", to_float32=True, num_cams=num_cams),
    dict(type="ResizeCropFlipImage"),
    dict(type="LeadSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_ego_fut_cmd",
            "target_point_previous",
            "target_point",
            "target_point_next",
            "projection_mat",
            "timestamp",
            "image_wh",
            "ego_status",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token"],
    ),
]

data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 384,
    "W": 384,
    "rand_flip": False,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=int(os.environ.get("WORKERS_PER_GPU", 2)),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        det_classes=["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"],
        map_classes=["Broken", "Solid", "SolidSolid", "Center"],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        plan_anchor_types=plan_anchor_types,
        perception_coord_mode="stored_sparse4d",
        sample_stride=lead_sample_stride,
        sample_offset=lead_sample_offset,
        expected_frame_sample_stride=lead_info_frame_stride,
        expected_frame_sample_offset=lead_info_frame_offset,
        expected_infos_version=lead_expected_infos_version,
        sharded_info_cache_size=lead_info_shard_cache_size,
        split_group=lead_split_group,
        sensor_perturbation=lead_sensor_perturbation,
        sensor_perturbation_probability=lead_effective_sensor_perturbation_probability,
        ann_file=os.path.join(anno_root, "lead_infos_train.pkl"),
        map_file=map_info_path,
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=lead_sequences_split_num,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        det_classes=["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"],
        map_classes=["Broken", "Solid", "SolidSolid", "Center"],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        plan_anchor_types=plan_anchor_types,
        perception_coord_mode="stored_sparse4d",
        sample_stride=lead_sample_stride,
        sample_offset=lead_sample_offset,
        expected_frame_sample_stride=lead_info_frame_stride,
        expected_frame_sample_offset=lead_info_frame_offset,
        expected_infos_version=lead_expected_infos_version,
        sharded_info_cache_size=lead_info_shard_cache_size,
        split_group=lead_split_group,
        ann_file=os.path.join(anno_root, "lead_infos_val.pkl"),
        sensor_perturbation=False,
        sensor_perturbation_probability=0.0,
        map_file=map_info_path,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        det_classes=["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"],
        map_classes=["Broken", "Solid", "SolidSolid", "Center"],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        plan_anchor_types=plan_anchor_types,
        perception_coord_mode="stored_sparse4d",
        sample_stride=lead_sample_stride,
        sample_offset=lead_sample_offset,
        expected_frame_sample_stride=lead_info_frame_stride,
        expected_frame_sample_offset=lead_info_frame_offset,
        expected_infos_version=lead_expected_infos_version,
        sharded_info_cache_size=lead_info_shard_cache_size,
        split_group=lead_split_group,
        ann_file=os.path.join(anno_root, "lead_infos_val.pkl"),
        map_file=map_info_path,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        sensor_perturbation=False,
        sensor_perturbation_probability=0.0,
        test_mode=True,
    ),
)

model = dict(
    planning_head=dict(
        pretrained_path=vlm_pretrained_path,
        action_horizon=lead_plan_horizon,
        action_delta_mu=lead_delta_mu,
        action_delta_std=lead_delta_std,
        system_prompt=lead_system_prompt,
        user_prompt_template=lead_user_prompt_template,
        strict_user_prompt_fields=["nav_cmd"],
        view_tokens=lead_view_tokens,
        aux_route_loss_weight=lead_aux_route_loss_weight,
        aux_target_speed_loss_weight=lead_aux_target_speed_loss_weight,
        target_speed_classes=lead_target_speed_classes,
        route_points_prediction=lead_route_points,
        planning_query_layout=lead_planning_query_layout,
        attn_implementation=lead_attn_implementation,
        planning_status_layout="tfv6_5token",
        use_deterministic_waypoint_queries=True,
        deterministic_waypoint_loss_weight=lead_deterministic_waypoint_loss_weight,
        inject_navigation_into_ego=False,
        velocity_normalization=25.0,
        prefix_max_len=None,
        target_points_normalization_constants=[200.0, 50.0],
        waypoint_hz=lead_plan_hz,
        loss_planning=dict(
            delta_mu=lead_delta_mu,
            delta_std=lead_delta_std,
        ),
        unified_decoder_cfg=dict(
            open_loop_hz=lead_perception_hz,
            close_loop_hz=20,
            det_instance_bank=dict(anchor=f"{project_dir}/data/kmeans/lead_det_900.npy"),
            map_instance_bank=dict(anchor=f"{project_dir}/data/kmeans/lead_map_100.npy"),
            ego_instance_bank=dict(
                type="LeadEgoInstanceBank",
                anchor_type="lead",
                ego_anchor=f"{project_dir}/data/kmeans/lead_ego_anchor.npy",
                feature_map_scale=ego_feature_map_scale,
            ),
            det_deformable=dict(num_cams=num_cams),
            map_deformable=dict(num_cams=num_cams),
            ego_deformable=dict(num_cams=num_cams),
        ),
    ),
)

# Keep the inherited display variables semantically distinct. Agent motion is
# disabled in Stage1 and its retained labels use the B2D 2Hz/6-step format;
# ego planning is the TFV6-aligned 4Hz/8-step target configured above.
fut_ts = 6
ego_fut_ts = lead_plan_horizon
loss_planning = model["planning_head"]["loss_planning"]
unified_decoder_cfg = model["planning_head"]["unified_decoder_cfg"]
data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    det_classes=["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"],
    map_classes=["Broken", "Solid", "SolidSolid", "Center"],
    modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
    plan_anchor_types=plan_anchor_types,
)

log_config = dict(
    interval=lead_log_interval,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)
checkpoint_config = dict(
    deepspeed=True,
    interval=checkpoint_interval,
    max_keep_ckpts=max_keep_ckpts,
)
runner = dict(type="IterBasedRunner", max_iters=runner_max_iters)
optimizer = dict(lr=lead_stage1_lr)
evaluation = dict(
    interval=num_iters_per_epoch,
    jsonfile_prefix="val_lead/",
    pipeline=test_pipeline,
    eval_mode=dict(
        with_det=False,
        with_tracking=False,
        with_map=False,
        with_motion=False,
        with_planning=True,
        tracking_threshold=0.2,
        motion_threshhold=0.2,
    ),
)
deepspeed = True
custom_hooks = [
    dict(type="DeepSpeedGradNormHook", priority="LOW"),
]
