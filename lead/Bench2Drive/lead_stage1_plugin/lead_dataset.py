import math
import copy
import bisect
import pickle
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from shapely.geometry import LineString
from torch.utils.data import Dataset


class _LeadShardedInfos:
    """Random-access view over a compact manifest and lazily loaded shards."""

    def __init__(self, manifest, manifest_path, cache_size=32):
        if manifest.get("lead_sharded_infos_version") != 1:
            raise RuntimeError("Unsupported LEAD sharded infos manifest")
        self.root = osp.dirname(osp.abspath(manifest_path))
        self.shards = list(manifest["shards"])
        self.num_samples = int(manifest["num_samples"])
        self.folder_names = list(manifest["folder_names"])
        self.folder_ids = np.asarray(manifest["folder_ids"], dtype=np.int32)
        self.frame_idxs = np.asarray(manifest["frame_idxs"], dtype=np.int32)
        self.first_info = manifest["first_info"]
        self.all_sensor_perturbation_fields = bool(
            manifest.get("all_sensor_perturbation_fields", False)
        )
        lengths = np.asarray([int(shard["length"]) for shard in self.shards], dtype=np.int64)
        self.shard_offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(lengths)]
        )
        if (
            int(self.shard_offsets[-1]) != self.num_samples
            or len(self.folder_ids) != self.num_samples
            or len(self.frame_idxs) != self.num_samples
        ):
            raise RuntimeError("LEAD sharded infos manifest has inconsistent lengths")
        self.cache_size = max(1, int(cache_size))
        self._shard_cache = OrderedDict()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.num_samples))]
        index = int(index)
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.shard_offsets, index) - 1
        shard = self._shard_cache.get(shard_index)
        if shard is None:
            shard_path = osp.join(self.root, self.shards[shard_index]["file"])
            shard = mmcv.load(shard_path)
            if len(shard) != int(self.shards[shard_index]["length"]):
                raise RuntimeError(f"LEAD infos shard length mismatch: {shard_path}")
            self._shard_cache[shard_index] = shard
            if len(self._shard_cache) > self.cache_size:
                self._shard_cache.popitem(last=False)
        else:
            self._shard_cache.move_to_end(shard_index)
        return shard[index - int(self.shard_offsets[shard_index])]

    def clear_cache(self):
        self._shard_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        return state


@DATASETS.register_module()
class LeadDataset(Dataset):
    """LEAD adapter for UniDriveVLA Stage1 training.

    The official LEAD data provides a 3-camera RGB strip, ego future positions,
    ego status, and object bboxes. Vector-map annotations are supplied through
    a B2D-format CARLA town map pkl generated for the LEAD towns.
    """

    def __init__(
        self,
        data_root,
        ann_file=None,
        pipeline=None,
        modality=None,
        data_aug_conf=None,
        map_file=None,
        test_mode=False,
        det_classes=None,
        map_classes=None,
        map_num_pts=20,
        with_connect_lane=True,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        plan_anchor_types=None,
        perception_coord_mode="stored_sparse4d",
        split_group=5,
        sample_stride=1,
        sample_offset=0,
        with_seq_flag=False,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
        sensor_perturbation=False,
        sensor_perturbation_probability=0.0,
        expected_frame_sample_stride=None,
        expected_frame_sample_offset=0,
        expected_infos_version=14,
        sharded_info_cache_size=32,
        agent_fut_ts=6,
        **kwargs,
    ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.map_file = map_file
        self.modality = modality or dict(use_camera=True)
        self.data_aug_conf = data_aug_conf
        self.test_mode = test_mode
        self.det_classes = det_classes or []
        self.map_classes = map_classes or []
        self.map_num_classes = len(self.map_classes)
        self.map_element_class = {
            map_class: map_label for map_label, map_class in enumerate(self.map_classes)
        }
        self.map_num_pts = map_num_pts
        self.with_connect_lane = with_connect_lane
        self.point_cloud_range = np.array(point_cloud_range)
        self.plan_anchor_types = [("temp", "4hz")] if plan_anchor_types is None else plan_anchor_types
        self.perception_coord_mode = perception_coord_mode
        self.split_group = split_group
        self.sample_stride = max(1, int(sample_stride))
        self.sample_offset = int(sample_offset) % self.sample_stride
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.sensor_perturbation_probability = float(sensor_perturbation_probability)
        if not 0.0 <= self.sensor_perturbation_probability <= 1.0:
            raise ValueError(
                "sensor_perturbation_probability must be in [0, 1], "
                f"got {self.sensor_perturbation_probability}"
            )
        self.sensor_perturbation = (
            bool(sensor_perturbation)
            and not self.test_mode
            and self.sensor_perturbation_probability > 0.0
        )
        self.expected_frame_sample_stride = (
            None
            if expected_frame_sample_stride is None
            else int(expected_frame_sample_stride)
        )
        self.expected_frame_sample_offset = int(expected_frame_sample_offset)
        self.expected_infos_version = int(expected_infos_version)
        self.sharded_info_cache_size = max(1, int(sharded_info_cache_size))
        self.agent_fut_ts = int(agent_fut_ts)
        if not 1 <= self.agent_fut_ts <= 64:
            raise ValueError(
                f"agent_fut_ts must be in [1, 64], got {self.agent_fut_ts}"
            )

        self.data_infos = self.load_annotations(ann_file)
        self.map_infos = {}
        if self.map_file is not None:
            with open(self.map_file, "rb") as f:
                self.map_infos = pickle.load(f)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if self.split_group > 1:
            self._split_data_infos()
        if with_seq_flag:
            self._set_sequence_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        data_infos = mmcv.load(ann_file)
        if (
            isinstance(data_infos, dict)
            and data_infos.get("lead_sharded_infos_version") is not None
        ):
            data_infos = _LeadShardedInfos(
                data_infos, ann_file, cache_size=self.sharded_info_cache_size
            )
        if self.perception_coord_mode == "stored_sparse4d" and data_infos:
            first = (
                data_infos.first_info
                if isinstance(data_infos, _LeadShardedInfos)
                else data_infos[0]
            )
            perturbed_required = {
                "sensor_perturbation_version",
                "perturbation_translation",
                "perturbation_rotation",
                "perturbed_image_path",
                "lidar2global_perturbed",
                "gt_boxes_perturbed",
                "gt_agent_fut_trajs_perturbed",
                "ego_status_perturbed",
                "ego_status_mask_perturbed",
                "target_point_previous_perturbed",
                "target_point_perturbed",
                "target_point_next_perturbed",
                "gt_ego_fut_trajs_4hz_perturbed",
                "gt_ego_fut_masks_4hz_perturbed",
                "gt_route_perturbed",
                "gt_route_masks_perturbed",
                "gt_ego_fut_trajs_2hz_perturbed",
                "gt_ego_fut_masks_2hz_perturbed",
            }
            if self.expected_infos_version >= 15:
                perturbed_required.update(
                    {
                        "gt_ego_fut_trajs_5hz_perturbed",
                        "gt_ego_fut_masks_5hz_perturbed",
                    }
                )
            ok = (
                first.get("lead_stage1_infos_version") == self.expected_infos_version
                and first.get("detection_filter_version") == 4
                and first.get("detection_class_mapping_version") == 2
                and first.get("detection_z_range") == [-4.0, 4.0]
                and first.get("detection_visibility_filter") == "lead_b2d_visibility_v1"
                and first.get("detection_traffic_control_filter")
                == "tfv6_affects_ego_non_green_v1"
                and first.get("detection_box_dimension_order") == "length_width_height"
                and first.get("detection_box_center") == "carla_bbox_center"
                and first.get("bbox_center_offset_version") == 1
                and first.get("perception_coord_system") == "sparse4d_lateral_forward"
                and first.get("planning_coord_system") == "lead_forward_lateral"
                and first.get("planning_freq_hz") == 4.0
                and first.get("planning_horizon_steps") == 8
                and "gt_ego_fut_trajs_4hz" in first
                and "gt_ego_fut_masks_4hz" in first
                and (
                    self.expected_infos_version < 15
                    or ("gt_ego_fut_trajs_5hz" in first and "gt_ego_fut_masks_5hz" in first)
                )
                and "gt_route" in first
                and "gt_route_masks" in first
                and "gt_target_speed" in first
                and "gt_brake" in first
                and "ego_status" in first
                and "ego_status_mask" in first
                and "target_point_previous" in first
                and "target_point" in first
                and "target_point_next" in first
                and "data_save_freq" in first
                and "source_sim_hz" in first
                and (
                    getattr(self, "expected_frame_sample_stride", None) is None
                    or first.get("frame_sample_stride")
                    == getattr(self, "expected_frame_sample_stride", None)
                )
                and first.get("frame_sample_offset")
                == getattr(self, "expected_frame_sample_offset", 0)
                and first.get("route_quality_filter") == "tfv6_posttrain"
                and first.get("route_skip_first") == 8
                and first.get("route_skip_last") == 8
                and (
                    not self.sensor_perturbation
                    or (
                        first.get("sensor_perturbation_version") == 1
                        and (
                            data_infos.all_sensor_perturbation_fields
                            if isinstance(data_infos, _LeadShardedInfos)
                            else all(perturbed_required.issubset(info) for info in data_infos)
                        )
                    )
                )
            )
            if not ok:
                raise RuntimeError(
                    "LEAD infos are missing the regenerated coordinate/time/detection/perturbation markers. "
                    "Regenerate lead_infos_*.pkl with tools/generate_lead_stage1_infos.py, "
                    "or set perception_coord_mode='lead_ego' only for legacy debug checks."
                )
        if self.sample_stride > 1:
            if isinstance(data_infos, _LeadShardedInfos):
                raise RuntimeError(
                    "Apply frame sampling while generating sharded LEAD infos; "
                    "runtime sample_stride > 1 is not supported for a manifest."
                )
            data_infos = [
                info
                for info in data_infos
                if int(info.get("frame_idx", 0)) % self.sample_stride == self.sample_offset
            ]
        if isinstance(data_infos, _LeadShardedInfos):
            data_infos.clear_cache()
        else:
            for info in data_infos:
                info.setdefault("timestamp", info.get("frame_idx", 0) / 20.0)
        return data_infos

    def _split_data_infos(self):
        if isinstance(self.data_infos, _LeadShardedInfos):
            raise RuntimeError(
                "split_group > 1 is not supported with sharded LEAD infos"
            )
        group_infos = []
        group_length = []
        for i in range(self.split_group):
            group_info = self.data_infos[i::self.split_group]
            group_infos.extend(group_info)
            group_length.append(len(group_info))
        self.data_infos = group_infos
        self.group_length = np.array(group_length)
        self.group_cumsum = np.array([0] + group_length[:-1]).cumsum()

    def _set_sequence_group_flag(self):
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos), dtype=np.int64)
            return
        if isinstance(self.data_infos, _LeadShardedInfos):
            folder_ids = self.data_infos.folder_ids
            self.flag = np.zeros(len(folder_ids), dtype=np.int64)
            if len(folder_ids) > 1:
                self.flag[1:] = np.cumsum(folder_ids[1:] != folder_ids[:-1])
        else:
            res = []
            curr_sequence = 0
            for idx in range(len(self.data_infos)):
                if idx != 0 and self.data_infos[idx]["folder"] != self.data_infos[idx - 1]["folder"]:
                    curr_sequence += 1
                res.append(curr_sequence)
            self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.arange(len(self.data_infos), dtype=np.int64)
                return
            bin_counts = np.bincount(self.flag)
            new_flags = []
            curr_new_flag = 0
            for curr_flag in range(len(bin_counts)):
                curr_sequence_length = np.array(
                    list(
                        range(
                            0,
                            bin_counts[curr_flag],
                            math.ceil(bin_counts[curr_flag] / self.sequences_split_num),
                        )
                    )
                    + [bin_counts[curr_flag]]
                )
                for sub_seq_idx in curr_sequence_length[1:] - curr_sequence_length[:-1]:
                    for _ in range(sub_seq_idx):
                        new_flags.append(curr_new_flag)
                    curr_new_flag += 1
            self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        h, w = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fh, fw = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(w * resize), int(h * resize))
            new_w, new_h = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * new_h) - fh
            crop_w = int(np.random.uniform(0, max(0, new_w - fw)))
            crop = (crop_w, crop_h, crop_w + fw, crop_h + fh)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fh / h, fw / w)
            resize_dims = (int(w * resize), int(h * resize))
            new_w, new_h = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * new_h) - fh
            crop_w = int(max(0, new_w - fw) / 2)
            crop = (crop_w, crop_h, crop_w + fw, crop_h + fh)
            flip = False
            rotate = 0
            rotate_3d = 0
        augmentation = dict(
            resize=resize,
            resize_dims=resize_dims,
            crop=crop,
            flip=flip,
            rotate=rotate,
            rotate_3d=rotate_3d,
        )
        return augmentation

    def _sample_sensor_perturbation(self):
        """Sample independently per dataset item, matching TFV6 post-training."""
        return bool(
            self.sensor_perturbation
            and np.random.random() < self.sensor_perturbation_probability
        )

    @staticmethod
    def invert_pose(pose):
        inv_pose = np.eye(4, dtype=np.float32)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, 3] = -inv_pose[:3, :3] @ pose[:3, 3]
        return inv_pose

    @staticmethod
    def _yaw_to_matrix(yaw):
        c = math.cos(yaw)
        s = math.sin(yaw)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return mat

    @staticmethod
    def _camera_to_ego_matrix(yaw, pos):
        """CARLA ego: x forward, y right/lateral, z up.

        Pinhole camera: x right, y down, z forward.
        """
        c = math.cos(yaw)
        s = math.sin(yaw)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.array(
            [
                [-s, 0.0, c],
                [c, 0.0, s],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        mat[:3, 3] = pos
        return mat

    @staticmethod
    def _camera_intrinsic(width, height, fov):
        focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = width / 2.0
        intrinsic[1, 2] = height / 2.0
        return intrinsic

    @staticmethod
    def _sparse4d_to_lead_ego_matrix():
        """Sparse4D local frame: x lateral/right, y forward, z up."""
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return mat

    @staticmethod
    def _lead_ego_boxes_to_sparse4d(boxes):
        """Change the coordinate basis while preserving intrinsic L/W/H."""
        boxes = np.array(boxes, dtype=np.float32, copy=True).reshape(-1, 9)
        if len(boxes) == 0:
            return boxes
        boxes[:, [0, 1]] = boxes[:, [1, 0]]
        boxes[:, 6] = (np.pi / 2.0) - boxes[:, 6]
        boxes[:, [7, 8]] = boxes[:, [8, 7]]
        return boxes

    @staticmethod
    def _lead_ego_trajs_to_sparse4d(trajs):
        trajs = np.array(trajs, dtype=np.float32, copy=True)
        if trajs.size == 0:
            return trajs
        trajs[..., [0, 1]] = trajs[..., [1, 0]]
        return trajs

    def _camera_mats(self, info):
        sensor_info = info.get("sensor_information") or {}
        camera_calib = sensor_info.get("camera_calibration") or {}
        ego2img_rts, lidar2img_rts, lidar2cam_rts, cam_intrinsics = [], [], [], []
        sparse4d_to_ego = self._sparse4d_to_lead_ego_matrix()

        for cam_id in sorted(camera_calib.keys(), key=lambda x: int(x)):
            calib = camera_calib[cam_id]
            width = float(calib.get("width", 384))
            height = float(calib.get("cropped_height", calib.get("height", 384)))
            fov = float(calib.get("fov", 60))
            pos = np.array(calib.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
            yaw_deg = float(calib.get("rot", [0.0, 0.0, 0.0])[2])

            cam2ego = self._camera_to_ego_matrix(math.radians(yaw_deg), pos)
            ego2cam = self.invert_pose(cam2ego)
            intrinsic = self._camera_intrinsic(width, height, fov)
            ego2img = intrinsic @ ego2cam
            lidar2img = ego2img @ sparse4d_to_ego
            lidar2cam = ego2cam @ sparse4d_to_ego

            ego2img_rts.append(ego2img.astype(np.float32))
            lidar2img_rts.append(lidar2img.astype(np.float32))
            lidar2cam_rts.append(lidar2cam.T.astype(np.float32))
            cam_intrinsics.append(intrinsic.astype(np.float32))

        if not ego2img_rts:
            intrinsic = self._camera_intrinsic(384, 384, 60)
            for _ in range(3):
                cam2ego = self._camera_to_ego_matrix(0.0, np.array([0.35, 0.0, 2.25], dtype=np.float32))
                ego2cam = self.invert_pose(cam2ego)
                ego2img = intrinsic @ ego2cam
                lidar2img = ego2img @ sparse4d_to_ego
                lidar2cam = ego2cam @ sparse4d_to_ego
                ego2img_rts.append(ego2img.copy())
                lidar2img_rts.append(lidar2img.astype(np.float32))
                lidar2cam_rts.append(lidar2cam.T.astype(np.float32))
                cam_intrinsics.append(intrinsic.copy())

        return ego2img_rts, lidar2img_rts, lidar2cam_rts, cam_intrinsics

    @staticmethod
    def _select_sensor_frame(info, use_sensor_perturbation):
        selected = dict(info)
        selected["sensor_perturbation_applied"] = bool(use_sensor_perturbation)
        if not use_sensor_perturbation:
            selected["sensor_perturbation_translation"] = 0.0
            selected["sensor_perturbation_rotation"] = 0.0
            return selected

        field_mapping = {
            "image_path": "perturbed_image_path",
            "lidar2global": "lidar2global_perturbed",
            "gt_boxes": "gt_boxes_perturbed",
            "gt_agent_fut_trajs": "gt_agent_fut_trajs_perturbed",
            "ego_status": "ego_status_perturbed",
            "ego_status_mask": "ego_status_mask_perturbed",
            "target_point_previous": "target_point_previous_perturbed",
            "target_point": "target_point_perturbed",
            "target_point_next": "target_point_next_perturbed",
            "gt_ego_fut_trajs_4hz": "gt_ego_fut_trajs_4hz_perturbed",
            "gt_ego_fut_masks_4hz": "gt_ego_fut_masks_4hz_perturbed",
            "gt_route": "gt_route_perturbed",
            "gt_route_masks": "gt_route_masks_perturbed",
            "gt_ego_fut_trajs_2hz": "gt_ego_fut_trajs_2hz_perturbed",
            "gt_ego_fut_masks_2hz": "gt_ego_fut_masks_2hz_perturbed",
        }
        if "gt_ego_fut_trajs_5hz_perturbed" in info:
            field_mapping.update(
                gt_ego_fut_trajs_5hz="gt_ego_fut_trajs_5hz_perturbed",
                gt_ego_fut_masks_5hz="gt_ego_fut_masks_5hz_perturbed",
            )
        missing = [source for source in field_mapping.values() if source not in info]
        if missing:
            raise RuntimeError(
                f"Missing synchronized LEAD sensor perturbation fields {missing} "
                f"for {info.get('folder')} frame {info.get('frame_idx')}"
            )
        for target, source in field_mapping.items():
            selected[target] = info[source]
        selected["sensor_perturbation_translation"] = float(
            info.get("perturbation_translation", 0.0)
        )
        selected["sensor_perturbation_rotation"] = float(
            info.get("perturbation_rotation", 0.0)
        )
        return selected

    def get_data_info(self, index, sensor_perturbation=False):
        info = self._select_sensor_frame(
            self.data_infos[index], bool(sensor_perturbation)
        )
        image_path = osp.join(self.data_root, info["image_path"])

        lidar2global = np.array(info.get("lidar2global", info.get("ego_matrix", np.eye(4))), dtype=np.float32)
        ego2img_rts, lidar2img_rts, lidar2cam_rts, cam_intrinsics = self._camera_mats(info)

        input_dict = dict(
            folder=info["folder"],
            scene_token=info["folder"],
            frame_idx=info["frame_idx"],
            timestamp=info.get("timestamp", info["frame_idx"] / 20.0),
            img_filename=[image_path],
            ego2img=ego2img_rts,
            lidar2img=lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            lidar2global=lidar2global,
            cam_intrinsic=cam_intrinsics,
            sensor_perturbation_applied=info["sensor_perturbation_applied"],
            sensor_perturbation_translation=info["sensor_perturbation_translation"],
            sensor_perturbation_rotation=info["sensor_perturbation_rotation"],
        )
        input_dict.update(self.get_ann_info(index, info=info))
        return input_dict

    @staticmethod
    def _spatial_offsets_from_temporal(fut_trajs, fut_masks, sample_distance, sample_points=6):
        fut_trajs = np.asarray(fut_trajs, dtype=np.float32).reshape(-1, 2)
        fut_masks = np.asarray(fut_masks, dtype=np.float32).reshape(-1)
        valid_len = int(min(len(fut_trajs), len(fut_masks)))
        valid_offsets = fut_trajs[:valid_len][fut_masks[:valid_len] > 0]

        spatial_offsets = np.zeros((sample_points, 2), dtype=np.float32)
        spatial_masks = np.zeros((sample_points,), dtype=np.float32)
        if len(valid_offsets) == 0:
            return spatial_offsets, spatial_masks

        positions = np.cumsum(valid_offsets, axis=0)
        points = np.concatenate([np.zeros((1, 2), dtype=np.float32), positions], axis=0)
        segments = points[1:] - points[:-1]
        seg_lens = np.linalg.norm(segments, axis=1)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)]).astype(np.float32)
        total_len = float(cum_lens[-1])

        prev_position = np.zeros((2,), dtype=np.float32)
        for sample_idx in range(sample_points):
            target_dist = float((sample_idx + 1) * sample_distance)
            if target_dist > total_len or total_len <= 1e-6:
                break
            seg_idx = int(np.searchsorted(cum_lens, target_dist, side="right") - 1)
            seg_idx = min(max(seg_idx, 0), len(segments) - 1)
            seg_len = float(seg_lens[seg_idx])
            if seg_len <= 1e-6:
                position = points[seg_idx + 1]
            else:
                ratio = (target_dist - float(cum_lens[seg_idx])) / seg_len
                position = points[seg_idx] + ratio * segments[seg_idx]
            spatial_offsets[sample_idx] = position - prev_position
            spatial_masks[sample_idx] = 1.0
            prev_position = position.astype(np.float32)

        return spatial_offsets, spatial_masks

    def _next_spatial_frame_index(self, index):
        if self.split_group <= 1 or isinstance(self.data_infos, _LeadShardedInfos):
            return index + 1

        group_diffs = index - self.group_cumsum
        group_diffs[group_diffs < 0] = len(self.data_infos)
        group_idx = int(np.argmin(group_diffs))
        group_diff = int(group_diffs[group_idx])
        if group_idx >= self.split_group - 1:
            group_idx = 0
            group_diff += 1
        else:
            group_idx += 1
        return int(self.group_cumsum[group_idx] + group_diff)

    def _future_ego_positions_for_spatial(
        self,
        index,
        current_info,
        max_distance,
    ):
        current_lidar2global = np.asarray(
            current_info.get("lidar2global", np.eye(4)), dtype=np.float64
        )
        if current_lidar2global.shape != (4, 4):
            raise ValueError(
                "Expected current lidar2global to have shape (4, 4), got "
                f"{current_lidar2global.shape}"
            )
        global2current = np.linalg.inv(current_lidar2global)
        current_folder = current_info.get("folder")
        positions = []
        future_index = int(index)

        while True:
            future_index = self._next_spatial_frame_index(future_index)
            if future_index < 0 or future_index >= len(self.data_infos):
                break
            future_info = self.data_infos[future_index]
            if future_info.get("folder") != current_folder:
                break

            future_lidar2global = np.asarray(
                future_info.get("lidar2global", np.eye(4)), dtype=np.float64
            )
            if future_lidar2global.shape != (4, 4):
                break
            future_in_current = global2current @ future_lidar2global
            # Sparse4D stores (lateral, forward); planning uses (forward, lateral).
            position = future_in_current[[1, 0], 3]
            if not np.isfinite(position).all():
                break
            positions.append(position.astype(np.float32))
            if np.linalg.norm(position) >= float(max_distance):
                break

        return np.asarray(positions, dtype=np.float32).reshape(-1, 2)

    @staticmethod
    def _spatial_offsets_from_future_positions(
        future_positions,
        sample_distance,
        sample_points=6,
    ):
        future_positions = np.asarray(
            future_positions, dtype=np.float64
        ).reshape(-1, 2)
        spatial_offsets = np.zeros((sample_points, 2), dtype=np.float32)
        spatial_masks = np.zeros((sample_points,), dtype=np.float32)
        if len(future_positions) == 0:
            return spatial_offsets, spatial_masks

        points = np.concatenate(
            [np.zeros((1, 2), dtype=np.float64), future_positions],
            axis=0,
        )
        previous_sample = np.zeros(2, dtype=np.float64)
        search_start = 0

        for sample_idx in range(sample_points):
            target_distance = float((sample_idx + 1) * sample_distance)
            sampled_position = None
            sampled_segment = None

            for segment_idx in range(search_start, len(points) - 1):
                start = points[segment_idx]
                delta = points[segment_idx + 1] - start
                a = float(np.dot(delta, delta))
                if a <= 1e-12:
                    continue
                b = 2.0 * float(np.dot(start, delta))
                c = float(np.dot(start, start) - target_distance ** 2)
                discriminant = b * b - 4.0 * a * c
                if discriminant < -1e-9:
                    continue
                root = math.sqrt(max(discriminant, 0.0))
                candidates = sorted(
                    value
                    for value in (
                        (-b - root) / (2.0 * a),
                        (-b + root) / (2.0 * a),
                    )
                    if -1e-7 <= value <= 1.0 + 1e-7
                )
                if candidates:
                    ratio = min(max(candidates[0], 0.0), 1.0)
                    sampled_position = start + ratio * delta
                    sampled_segment = segment_idx
                    break

            if sampled_position is None:
                break
            spatial_offsets[sample_idx] = (
                sampled_position - previous_sample
            ).astype(np.float32)
            spatial_masks[sample_idx] = 1.0
            previous_sample = sampled_position
            search_start = int(sampled_segment)

        return spatial_offsets, spatial_masks

    def get_spatial_plan_info(self, index, current_info):
        plan_results = {}
        spatial_anchors = [
            anchor for anchor in self.plan_anchor_types if anchor[0] == "spat"
        ]
        if not spatial_anchors:
            return plan_results

        sample_distances = []
        for plan_anchor in spatial_anchors:
            if plan_anchor[1] not in {
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m"
            }:
                raise NotImplementedError(
                    f"Unsupported LEAD spatial planning anchor: {plan_anchor}"
                )
            sample_distances.append(float(plan_anchor[1][:-1]))

        future_positions = self._future_ego_positions_for_spatial(
            index,
            current_info,
            max_distance=max(sample_distances) * 6.0,
        )
        for plan_anchor, sample_distance in zip(
            spatial_anchors, sample_distances
        ):
            spatial_offsets, spatial_masks = (
                self._spatial_offsets_from_future_positions(
                    future_positions,
                    sample_distance=sample_distance,
                    sample_points=6,
                )
            )
            plan_results[
                f"gt_ego_spat_trajs_{plan_anchor[1]}"
            ] = spatial_offsets
            plan_results[
                f"gt_ego_spat_masks_{plan_anchor[1]}"
            ] = spatial_masks
        return plan_results

    def get_ann_info(self, index, info=None):
        if info is None:
            info = self.data_infos[index]
        gt_bboxes_3d = np.array(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 9)
        gt_names = np.array(info.get("gt_names", []))
        gt_labels_3d = []
        for name in gt_names:
            gt_labels_3d.append(self.det_classes.index(name) if name in self.det_classes else -1)
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)
        instance_inds = np.array(info.get("gt_ids", []), dtype=np.int32)

        num_agents = len(gt_bboxes_3d)
        gt_agent_fut_trajs = np.asarray(
            info.get(
                "gt_agent_fut_trajs",
                np.zeros((num_agents, self.agent_fut_ts, 2)),
            ),
            dtype=np.float32,
        )
        gt_agent_fut_masks = np.asarray(
            info.get(
                "gt_agent_fut_masks",
                np.zeros((num_agents, self.agent_fut_ts)),
            ),
            dtype=np.float32,
        )
        if gt_agent_fut_trajs.size % max(num_agents * 2, 1) != 0:
            raise ValueError("Invalid LEAD agent trajectory tensor size")
        if gt_agent_fut_masks.size % max(num_agents, 1) != 0:
            raise ValueError("Invalid LEAD agent trajectory mask size")
        stored_traj_horizon = (
            gt_agent_fut_trajs.size // (num_agents * 2)
            if num_agents
            else self.agent_fut_ts
        )
        stored_mask_horizon = (
            gt_agent_fut_masks.size // num_agents
            if num_agents
            else self.agent_fut_ts
        )
        gt_agent_fut_trajs = gt_agent_fut_trajs.reshape(
            num_agents, stored_traj_horizon, 2
        )
        gt_agent_fut_masks = gt_agent_fut_masks.reshape(
            num_agents, stored_mask_horizon
        )
        if stored_traj_horizon != stored_mask_horizon:
            raise ValueError(
                "LEAD agent trajectory and mask horizons differ: "
                f"{stored_traj_horizon} != {stored_mask_horizon}"
            )
        if stored_traj_horizon < self.agent_fut_ts:
            raise ValueError(
                f"LEAD info stores {stored_traj_horizon} agent motion steps, "
                f"but dataset requests {self.agent_fut_ts}"
            )
        gt_agent_fut_trajs = gt_agent_fut_trajs[:, : self.agent_fut_ts]
        gt_agent_fut_masks = gt_agent_fut_masks[:, : self.agent_fut_ts]
        if self.perception_coord_mode == "lead_ego":
            gt_bboxes_3d = self._lead_ego_boxes_to_sparse4d(gt_bboxes_3d)
            gt_agent_fut_trajs = self._lead_ego_trajs_to_sparse4d(gt_agent_fut_trajs)
        elif self.perception_coord_mode != "stored_sparse4d":
            raise ValueError(f"Unsupported LEAD perception_coord_mode: {self.perception_coord_mode}")

        ego_status = np.array(info["ego_status"], dtype=np.float32)
        ego_status_mask = np.array(info["ego_status_mask"], dtype=np.float32)
        fut_trajs = np.array(info.get("gt_ego_fut_trajs_4hz", np.zeros((8, 2))), dtype=np.float32).reshape(8, 2)
        fut_masks = np.array(info.get("gt_ego_fut_masks_4hz", np.zeros(8)), dtype=np.float32).reshape(8)
        fut_trajs_5hz = np.array(info.get("gt_ego_fut_trajs_5hz", np.zeros((6, 2))), dtype=np.float32).reshape(6, 2)
        fut_masks_5hz = np.array(info.get("gt_ego_fut_masks_5hz", np.zeros(6)), dtype=np.float32).reshape(6)
        fut_trajs_2hz = np.array(info.get("gt_ego_fut_trajs_2hz", np.zeros((6, 2))), dtype=np.float32).reshape(6, 2)
        fut_masks_2hz = np.array(info.get("gt_ego_fut_masks_2hz", np.zeros(6)), dtype=np.float32).reshape(6)
        command = np.array(info.get("gt_ego_fut_cmd", np.array([0, 0, 0, 1, 0, 0])), dtype=np.float32)
        target_point_previous = np.array(
            info.get("target_point_previous", np.zeros(2)), dtype=np.float32
        ).reshape(2)
        target_point = np.array(
            info.get("target_point", np.zeros(2)), dtype=np.float32
        ).reshape(2)
        target_point_next = np.array(
            info.get("target_point_next", np.zeros(2)), dtype=np.float32
        ).reshape(2)
        gt_route = np.array(info.get("gt_route", np.zeros((10, 2))), dtype=np.float32).reshape(10, 2)
        gt_route_masks = np.array(info.get("gt_route_masks", np.zeros(10)), dtype=np.float32).reshape(10)
        gt_target_speed = np.array(info["gt_target_speed"], dtype=np.float32).reshape(1)
        gt_brake = np.array(info["gt_brake"], dtype=np.float32).reshape(1)

        ann_info = dict(
            gt_names=gt_names,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            instance_inds=instance_inds,
            map_geoms=self.get_map_info(index, info=info),
            map_infos={},
            gt_agent_fut_trajs=gt_agent_fut_trajs,
            gt_agent_fut_masks=gt_agent_fut_masks,
            gt_ego_fut_trajs=fut_trajs,
            gt_ego_fut_masks=fut_masks,
            gt_ego_fut_trajs_4hz=fut_trajs,
            gt_ego_fut_masks_4hz=fut_masks,
            gt_ego_fut_trajs_5hz=fut_trajs_5hz,
            gt_ego_fut_masks_5hz=fut_masks_5hz,
            gt_ego_fut_trajs_2hz=fut_trajs_2hz,
            gt_ego_fut_masks_2hz=fut_masks_2hz,
            gt_ego_fut_cmd=command,
            target_point_previous=target_point_previous,
            target_point=target_point,
            target_point_next=target_point_next,
            gt_route=gt_route,
            gt_route_masks=gt_route_masks,
            gt_target_speed=gt_target_speed,
            gt_brake=gt_brake,
            ego_status=ego_status,
            ego_status_mask=ego_status_mask,
        )
        ann_info.update(self.get_spatial_plan_info(index, info))
        return ann_info

    def geom2anno(self, map_geoms):
        annos = {}
        for label, geom_list in map_geoms.items():
            annos[label] = []
            for geom in geom_list:
                annos[label].append(np.array(geom.xy).T)
        return annos

    def connect_lanes(self, all_line_list, all_line_id_list, all_target_id):
        index = 0
        iter_time = 0
        stop_loop = True
        while True:
            if index >= len(all_line_list):
                iter_time += 1
                assert iter_time < 1000, "iter time over 1000"
                if stop_loop:
                    break
                index = 0
                stop_loop = True

            line_list = copy.deepcopy(all_line_list[index])
            line_id_list = copy.deepcopy(all_line_id_list[index])
            target_id = copy.deepcopy(all_target_id[index])
            has_merged = False

            for target_index, target_line_id_list in enumerate(all_line_id_list):
                if target_index == index:
                    continue
                if target_id[0] in [x[0] for x in line_id_list]:
                    continue
                if target_id == target_line_id_list[0]:
                    target_line_list = all_line_list[target_index]
                    if np.linalg.norm(target_line_list[0][0] - line_list[-1][-1]) < 0.1:
                        line_list.extend(target_line_list)
                        line_id_list.extend(target_line_id_list)
                        all_line_list[target_index] = line_list
                        all_line_id_list[target_index] = line_id_list
                        has_merged = True
                elif target_id in target_line_id_list:
                    sub_target_index = target_line_id_list.index(target_id)
                    target_line_list = all_line_list[target_index]
                    target_target_id = all_target_id[target_index]
                    if np.linalg.norm(target_line_list[sub_target_index][0] - line_list[-1][-1]) < 0.1:
                        line_list.extend(copy.deepcopy(target_line_list[sub_target_index:]))
                        line_id_list.extend(copy.deepcopy(target_line_id_list[sub_target_index:]))
                        if line_id_list != target_line_id_list:
                            all_line_list.append(line_list)
                            all_line_id_list.append(line_id_list)
                            all_target_id.append(target_target_id)
                            has_merged = True

            if has_merged:
                stop_loop = False
                all_line_list.pop(index)
                all_line_id_list.pop(index)
                all_target_id.pop(index)
            else:
                index += 1

        i = 0
        while i < len(all_line_list):
            j = i + 1
            while j < len(all_line_list):
                if (
                    len(all_line_id_list[i]) == len(all_line_id_list[j])
                    and (np.array(all_line_id_list[i]) == np.array(all_line_id_list[j])).all()
                ):
                    lane_points1 = np.concatenate(all_line_list[i], axis=0)
                    lane_points2 = np.concatenate(all_line_list[j], axis=0)
                    if len(lane_points1) == len(lane_points2) and (lane_points1 == lane_points2).all():
                        all_line_list.pop(j)
                        all_line_id_list.pop(j)
                        all_target_id.pop(j)
                    else:
                        j += 1
                else:
                    j += 1
            i += 1
        return all_line_list, all_line_id_list, all_target_id

    def get_map_info(self, index, info=None):
        if info is None:
            info = self.data_infos[index]
        town_name = info.get("town_name")
        if not town_name or town_name not in self.map_infos:
            return {}

        polylines = []
        gt_labels = []
        map_info = self.map_infos[town_name]
        lane_types = map_info["lane_types"]
        lane_points = map_info["lane_points"]
        lane_sample_points = map_info["lane_sample_points"]
        trigger_volumes_points = map_info["trigger_volumes_points"]
        trigger_volumes_types = map_info["trigger_volumes_types"]

        world2lidar = np.linalg.inv(np.array(info.get("lidar2global", np.eye(4)), dtype=np.float32))
        ego_xy = np.linalg.inv(world2lidar)[0:2, 3]
        max_distance = 50

        if self.with_connect_lane and "lane_ids" in map_info and "lane_topos" in map_info:
            lane_ids = map_info["lane_ids"]
            lane_topos = map_info["lane_topos"]
            for lane_type in self.map_element_class.keys():
                all_lane_list = []
                all_lane_id_list = []
                all_target_id = []
                for idx in range(len(lane_sample_points)):
                    single_sample_points = lane_sample_points[idx]
                    distance = np.linalg.norm((single_sample_points[:, 0:2] - ego_xy), axis=-1)
                    if np.min(distance) < max_distance and lane_type == lane_types[idx]:
                        if lane_type == "Center":
                            all_lane_list.append([copy.deepcopy(lane_points[idx])])
                        else:
                            for target_id in lane_topos[idx]:
                                all_lane_list.append([copy.deepcopy(lane_points[idx])])
                                all_lane_id_list.append([copy.deepcopy(lane_ids[idx])])
                                all_target_id.append(copy.deepcopy(target_id))

                if len(all_lane_list) and lane_type != "Center":
                    all_lane_list, all_lane_id_list, all_target_id = self.connect_lanes(
                        all_lane_list, all_lane_id_list, all_target_id
                    )

                for idx in range(len(all_lane_list)):
                    points = np.concatenate(all_lane_list[idx])
                    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
                    points_in_lidar = (world2lidar @ points.T).T
                    mask = (
                        (points_in_lidar[:, 0] > self.point_cloud_range[0])
                        & (points_in_lidar[:, 0] < self.point_cloud_range[3])
                        & (points_in_lidar[:, 1] > self.point_cloud_range[1])
                        & (points_in_lidar[:, 1] < self.point_cloud_range[4])
                    )
                    change_points = np.diff(mask.astype(int))
                    starts = np.where(change_points == 1)[0] + 1
                    ends = np.where(change_points == -1)[0] + 1
                    if mask[0]:
                        starts = np.concatenate([[0], starts])
                    if mask[-1]:
                        ends = np.append(ends, len(mask))
                    for start, end in zip(starts, ends):
                        points_in_lidar_range = points_in_lidar[start:end, 0:2]
                        if len(points_in_lidar_range) > 1:
                            gt_labels.append(self.map_element_class[lane_type])
                            polylines.append(LineString(points_in_lidar_range))
        else:
            for idx in range(len(lane_sample_points)):
                single_sample_points = lane_sample_points[idx]
                distance = np.linalg.norm((single_sample_points[:, 0:2] - ego_xy), axis=-1)
                if np.min(distance) >= max_distance:
                    continue
                lane_type = lane_types[idx]
                if lane_type not in self.map_element_class:
                    continue
                points = lane_points[idx]
                points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
                points_in_lidar = (world2lidar @ points.T).T
                mask = (
                    (points_in_lidar[:, 0] > self.point_cloud_range[0])
                    & (points_in_lidar[:, 0] < self.point_cloud_range[3])
                    & (points_in_lidar[:, 1] > self.point_cloud_range[1])
                    & (points_in_lidar[:, 1] < self.point_cloud_range[4])
                )
                points_in_lidar_range = points_in_lidar[mask, 0:2]
                if len(points_in_lidar_range) > 1:
                    polylines.append(LineString(points_in_lidar_range))
                    gt_labels.append(self.map_element_class[lane_type])

        for idx in range(len(trigger_volumes_points)):
            trigger_type = trigger_volumes_types[idx]
            if trigger_type not in self.map_element_class:
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
            points_in_lidar = (world2lidar @ points.T).T
            mask = (
                (points_in_lidar[:, 0] > self.point_cloud_range[0])
                & (points_in_lidar[:, 0] < self.point_cloud_range[3])
                & (points_in_lidar[:, 1] > self.point_cloud_range[1])
                & (points_in_lidar[:, 1] < self.point_cloud_range[4])
            )
            points_in_lidar_range = points_in_lidar[mask, 0:2]
            if mask.all() and len(points_in_lidar_range) > 1:
                polylines.append(
                    LineString(np.concatenate((points_in_lidar_range, points_in_lidar_range[0:1]), axis=0))
                )
                gt_labels.append(self.map_element_class[trigger_type])

        map_geoms = {}
        for label, polyline in zip(gt_labels, polylines):
            map_geoms.setdefault(label, []).append(polyline)
        return map_geoms

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        use_sensor_perturbation = self._sample_sensor_perturbation()
        data = self.get_data_info(idx, sensor_perturbation=use_sensor_perturbation)
        data["aug_config"] = aug_config
        return self.pipeline(data)
