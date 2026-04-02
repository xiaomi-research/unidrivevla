import os
import copy
import math
import mmcv
import torch
import shapely
import warnings
import tempfile
import prettytable
import numpy as np
import json, pickle
import os.path as osp

from mmcv.utils import print_log
from mmcv.fileio.io import load
from mmcv.utils import track_iter_progress
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset
from numpy.polynomial import Polynomial
from shapely.geometry import LineString
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from .bench2drive_eval_utils import (calc_ap, calc_tp, center_distance, accumulate, EvalBoxes,
                                     DetectionMetrics, DetectionBox, DetectionMetricDataList)

NameMapping = {
    # =================vehicle=================
    # bicycle
    "vehicle.bh.crossbike": "bicycle",
    "vehicle.diamondback.century": "bicycle",
    "vehicle.gazelle.omafiets": "bicycle",
    # car
    "vehicle.audi.etron": "car",
    "vehicle.chevrolet.impala": "car",
    "vehicle.dodge.charger_2020": "car",
    "vehicle.dodge.charger_police": "car",
    "vehicle.dodge.charger_police_2020": "car",
    "vehicle.lincoln.mkz_2017": "car",
    "vehicle.lincoln.mkz_2020": "car",
    "vehicle.mini.cooper_s_2021": "car",
    "vehicle.mercedes.coupe_2020": "car",
    "vehicle.ford.mustang": "car",
    "vehicle.nissan.patrol_2021": "car",
    "vehicle.audi.tt": "car",
    "vehicle.audi.etron": "car",
    "vehicle.ford.crown": "car",
    "vehicle.ford.mustang": "car",
    "vehicle.tesla.model3": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": "car",
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "car",
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": "truck",
    # =========================================

    # =================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": "traffic_sign",
    "traffic.speed_limit.40": "traffic_sign",
    "traffic.speed_limit.50": "traffic_sign",
    "traffic.speed_limit.60": "traffic_sign",
    "traffic.speed_limit.90": "traffic_sign",
    "traffic.speed_limit.120": "traffic_sign",

    "traffic.stop": "traffic_sign",
    "traffic.yield": "traffic_sign",
    "traffic.traffic_light": "traffic_light",
    # =========================================

    # ===================Construction===========
    "static.prop.warningconstruction": "traffic_cone",
    "static.prop.warningaccident": "traffic_cone",
    "static.prop.trafficwarning": "traffic_cone",
    "static.prop.constructioncone": "traffic_cone",

    # =================pedestrian==============
    "walker.pedestrian.0001": "pedestrian",
    "walker.pedestrian.0003": "pedestrian",
    "walker.pedestrian.0004": "pedestrian",
    "walker.pedestrian.0005": "pedestrian",
    "walker.pedestrian.0007": "pedestrian",
    "walker.pedestrian.0010": "pedestrian",
    "walker.pedestrian.0013": "pedestrian",
    "walker.pedestrian.0014": "pedestrian",
    "walker.pedestrian.0015": "pedestrian",
    "walker.pedestrian.0016": "pedestrian",
    "walker.pedestrian.0017": "pedestrian",
    "walker.pedestrian.0018": "pedestrian",
    "walker.pedestrian.0019": "pedestrian",
    "walker.pedestrian.0020": "pedestrian",
    "walker.pedestrian.0021": "pedestrian",
    "walker.pedestrian.0022": "pedestrian",
    "walker.pedestrian.0025": "pedestrian",
    "walker.pedestrian.0027": "pedestrian",
    "walker.pedestrian.0030": "pedestrian",
    "walker.pedestrian.0031": "pedestrian",
    "walker.pedestrian.0032": "pedestrian",
    "walker.pedestrian.0034": "pedestrian",
    "walker.pedestrian.0035": "pedestrian",
    "walker.pedestrian.0041": "pedestrian",
    "walker.pedestrian.0042": "pedestrian",
    "walker.pedestrian.0046": "pedestrian",
    "walker.pedestrian.0047": "pedestrian",

    # ==========================================
    "static.prop.dirtdebris01": "others",
    "static.prop.dirtdebris02": "others",
}

EvalConfig = {
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "mean_ap_weight": 5,
    "class_names": ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian"],
    "tp_metrics": ["trans_err", "scale_err", "orient_err", "vel_err"],
    "err_name_maping": {"trans_err": "mATE", "scale_err": "mASE", "orient_err": "mAOE", "vel_err": "mAVE", "attr_err": "mAAE"},
    "class_range": {
        "car": (50, 50),
        "van": (50, 50),
        "truck": (50, 50),
        "bicycle": (40, 40),
        "traffic_sign": (30, 30),
        "traffic_cone": (30, 30),
        "traffic_light": (30, 30),
        "pedestrian": (40, 40)
    }
}

@DATASETS.register_module()
class Bench2DriveDataset(Dataset):
    def __init__(
            self,
            data_root,
            ann_file=None,
            work_dir=None,
            pipeline=None,
            modality=None,
            map_file=None,
            eval_cfg=None,
            test_mode=False,
            map_classes=None,
            name_mapping=None,
            data_aug_conf=None,
            split_group=5,
            with_seq_flag=False,
            sequences_split_num=2,
            keep_consistent_seq_aug=True,
            point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            # det
            remap_box=True,
            det_classes=None,
            with_velocity=True,
            filter_empty_gt=True,
            align_static_yaw=False,
            # map
            map_num_pts=20,
            with_connect_lane=True,
            # traj
            sample_rate=1,
            past_frames=2,
            future_frames=6,
            spatial_points=6,
            plan_anchor_types=None,
            # info
            with_next_target_point=False,
    ):
        super().__init__()
        self.ann_file = ann_file
        self.map_file = map_file
        self.modality = modality
        self.data_root = data_root
        self.test_mode = test_mode
        self.data_aug_conf = data_aug_conf
        self.point_cloud_range = np.array(point_cloud_range)
        # det
        self.remap_box = remap_box
        self.det_classes = det_classes
        self.with_velocity = with_velocity
        self.filter_empty_gt = filter_empty_gt
        self.align_static_yaw = align_static_yaw
        # map
        self.map_classes = map_classes
        self.map_num_classes = len(map_classes)
        self.map_element_class = {map_class: map_label for map_label, map_class in enumerate(self.map_classes)}
        self.map_num_pts = map_num_pts
        self.map_ann_file = './data/infos/b2d_map_anno.pkl'  # for map evaluation
        self.with_connect_lane = with_connect_lane
        # traj
        self.sample_rate = sample_rate
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.spatial_points = spatial_points
        self.plan_anchor_types = plan_anchor_types
        # info
        self.with_next_target_point = with_next_target_point
        self.eval_cfg = eval_cfg if eval_cfg is not None else EvalConfig
        self.name_mapping = name_mapping if name_mapping is not None else NameMapping

        self.data_infos = self.load_annotations(self.ann_file)

        with open(self.map_file, 'rb') as f:
            self.map_infos = pickle.load(f)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.split_group = split_group
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug

        if self.split_group > 0:
            self._split_data_infos()

        if with_seq_flag:
            self._set_sequence_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def _split_data_infos(self):
        print("using splits group: {} !!!!!!!".format(self.split_group))
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
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return

        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]["folder"] != self.data_infos[idx-1]["folder"]:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                            curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num)
                self.flag = np.array(new_flags, dtype=np.int64)

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def geom2anno(self, map_geoms):
        annos = {}
        for label, geom_list in map_geoms.items():
            annos[label] = []
            for geom in geom_list:
                anno = np.array(geom.xy).T
                annos[label].append(anno)
        return annos

    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms

    def command2hot(self, command, max_dim=6):
        # "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim, dtype=np.float32)
        cmd_one_hot[command] = 1
        return cmd_one_hot

    def connect_lanes(self, all_line_list, all_line_id_list, all_target_id):
        index = 0
        iter_time = 0 # for debug
        stop_loop = True
        while True:
            if index >= len(all_line_list):
                iter_time += 1
                assert iter_time < 1000, "iter time over 1000"
                if stop_loop:
                    break
                else:
                    index = 0
                    stop_loop = True

            line_list = copy.deepcopy(all_line_list[index])
            line_id_list = copy.deepcopy(all_line_id_list[index])
            target_id = copy.deepcopy(all_target_id[index])

            has_merged = False
            for target_index, target_line_id_list in enumerate(all_line_id_list):
                if target_index == index:
                    continue

                elif target_id[0] in [x[0] for x in line_id_list]:
                    continue

                elif target_id == target_line_id_list[0]:
                    target_line_list = all_line_list[target_index]

                    if np.linalg.norm(target_line_list[0][0] - line_list[-1][-1]) < 0.1:
                        line_list.extend(target_line_list)
                        line_id_list.extend(target_line_id_list)
                        all_line_list[target_index] = line_list
                        all_line_id_list[target_index] = line_id_list
                        has_merged = True

                elif (target_id != target_line_id_list[0]) and (target_id in target_line_id_list):
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
                if ((len(all_line_id_list[i]) == len(all_line_id_list[j])) and
                        ((np.array(all_line_id_list[i]) == np.array(all_line_id_list[j])).all())):
                    lane_points1 = np.concatenate(all_line_list[i], axis=0)
                    lane_points2 = np.concatenate(all_line_list[j], axis=0)
                    if (len(lane_points1) == len(lane_points2) and (lane_points1==lane_points2).all()):
                        all_line_list.pop(j)
                        all_line_id_list.pop(j)
                        all_target_id.pop(j)
                    else:
                        j+=1
                else:
                    j+=1
            i+=1

        return all_line_list, all_line_id_list, all_target_id

    def get_ego_trajs(self, idx, past_frames, future_frames, sample_rate):
        adj_idx_list = range(idx-past_frames*sample_rate, idx+(future_frames+1)*sample_rate, sample_rate)
        cur_frame = self.data_infos[idx]
        full_adj_track = np.zeros((past_frames+future_frames+1, 2))
        full_adj_adj_mask = np.zeros(past_frames+future_frames+1)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']

        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if 0 <= adj_idx and adj_idx < len(self.data_infos):
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame['folder']:
                    break
                world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
                adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
                xy = adj2cur_lidar[0:2, 3]
                full_adj_track[j, 0:2] = xy
                full_adj_adj_mask[j] = 1

        offset_track = full_adj_track[1:] - full_adj_track[:-1]

        for j in range(past_frames-1, -1, -1):
            if full_adj_adj_mask[j] == 0:
                offset_track[j] = offset_track[j+1]

        for j in range(past_frames, past_frames + future_frames, 1):
            if full_adj_adj_mask[j+1] == 0 :
                offset_track[j] = 0

        ego_his_trajs = offset_track[:past_frames].copy()
        ego_fut_trajs = offset_track[past_frames:].copy()
        ego_fut_masks = full_adj_adj_mask[-future_frames:].copy()

        command = self.command2hot(cur_frame['command_near'])

        return ego_his_trajs, ego_fut_trajs, ego_fut_masks, command

    def get_ego_temporal_trajs(self, idx, future_frames, interval=1):
        adj_idx_list = [idx]

        adj_idx = idx
        for _ in range(future_frames*interval):
            # find real next frame
            if self.split_group > 0:
                tmp_adj_idx = adj_idx
                group_diffs = tmp_adj_idx - self.group_cumsum
                group_diffs[group_diffs<0] = len(self.data_infos)
                group_idx = np.argmin(group_diffs)
                group_diff = group_diffs[group_idx]
                if group_idx >= self.split_group - 1:
                    group_idx = 0
                    group_diff += 1
                else:
                    group_idx += 1
                adj_idx = self.group_cumsum[group_idx] + group_diff

            else:
                adj_idx += 1
            adj_idx_list.append(adj_idx)

        adj_idx_list = adj_idx_list[::interval]

        cur_frame = self.data_infos[idx]
        full_adj_track = np.zeros((future_frames+1, 2))
        full_adj_adj_mask = np.zeros(future_frames+1)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']

        # make sure the carla clip already init
        past_idx = adj_idx_list[0] - 2
        past_frame = self.data_infos[past_idx]
        if past_frame['folder'] == cur_frame['folder']:
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if 0 <= adj_idx and adj_idx < len(self.data_infos):
                    adj_frame = self.data_infos[adj_idx]
                    if adj_frame['folder'] != cur_frame['folder']:
                        break
                    world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
                    adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
                    xy = adj2cur_lidar[0:2, 3]
                    full_adj_track[j, 0:2] = xy
                    full_adj_adj_mask[j] = 1

        offset_track = full_adj_track[1:] - full_adj_track[:-1]

        for j in range(future_frames):
            if full_adj_adj_mask[j+1] == 0 :
                offset_track[j] = 0

        ego_fut_trajs = offset_track.copy()
        ego_fut_masks = full_adj_adj_mask[-future_frames:].copy()

        return ego_fut_trajs, ego_fut_masks
        
    def get_ego_spatial_trajs(self, idx, sample_points, sample_strategy, with_fitting=False):
        if sample_strategy["mode"] == 'LID':
            start_point = sample_strategy["start_distance"]
            end_point = sample_strategy["end_distance"]
            index = np.arange(0, sample_points, step=1)
            index_1 = index + 1
            bin_size = (end_point - start_point) / (sample_points * (1 + sample_points))
            sample_distances = start_point + bin_size * index * index_1
        elif sample_strategy["mode"] == "uniform":
            sample_distance = sample_strategy["sample_distance"]
            sample_distances = [i * sample_distance for i in range(1, sample_points + 1)]
        else:
            raise NotImplementedError

        cur_frame = self.data_infos[idx]
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']

        adj_idx = idx
        adj_positions = []
        while True:
            # find real next frame
            if self.split_group > 0:
                tmp_adj_idx = adj_idx
                group_diffs = tmp_adj_idx - self.group_cumsum
                group_diffs[group_diffs<0] = len(self.data_infos)
                group_idx = np.argmin(group_diffs)
                group_diff = group_diffs[group_idx]
                if group_idx >= self.split_group - 1:
                    group_idx = 0
                    group_diff += 1
                else:
                    group_idx += 1
                adj_idx = self.group_cumsum[group_idx] + group_diff

            else:
                adj_idx += 1

            if 0 <= adj_idx and adj_idx < len(self.data_infos):
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame['folder']:
                    break
                world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
                adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
                xy = adj2cur_lidar[0:2, 3]
                adj_positions.append(xy)
            else:
                break

        adj_positions = np.array(adj_positions)

        spatial_mask = np.array([0 for _ in range(sample_points)], np.float32)
        spatial_trajs = np.array([(-1, -1) for _ in range(sample_points)], np.float32)
        spatial_offset = np.array([(-1, -1) for _ in range(sample_points)], np.float32)

        if with_fitting and len(adj_positions) > 1:
            x = adj_positions[:, 1]
            y = adj_positions[:, 0]

            y_errs = []
            p_list = []
            for degree in range(1, 4):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    p = Polynomial.fit(x, y, degree)
                y_err = np.linalg.norm(p(x) - y)
                y_errs.append(y_err)
                p_list.append(p)
            p = p_list[np.argmin(y_errs)]

            x_fit = np.linspace(np.min(x), np.max(x), len(x) * 10)
            y_fit = p(x_fit)

            if abs(y_fit[0]-adj_positions[0][0])<0.1:
                adj_positions = np.stack([y_fit, x_fit], axis=1)

        adj_index = -1
        if len(adj_positions) > 0:
            for sample_idx, sample_distance in enumerate(sample_distances):
                pre_diff = sample_distance if sample_idx == 0 else sample_distance - sample_distances[sample_idx - 1]
                adj_diff = np.abs(sample_distance - np.linalg.norm(adj_positions, axis=1))
                min_index = np.argmin(adj_diff)
                if min_index > adj_index and adj_diff[min_index] < pre_diff * 0.25:
                    spatial_trajs[sample_idx] = adj_positions[min_index]
                    spatial_mask[sample_idx] = 1
                    adj_index = min_index

            for i in range(sample_points):
                if spatial_mask[i]:
                    spatial_offset[i] = spatial_trajs[i] if i == 0 else spatial_trajs[i] - spatial_trajs[i - 1]
                    if np.linalg.norm(spatial_offset[i]) < 0.1:
                        spatial_offset[i] = (-1, -1)
                        spatial_mask[i] = 0

        return spatial_offset, spatial_mask

    def get_agent_trajs(self, idx, future_frames, sample_rate):
        fut_idx_list = range(idx, idx+(future_frames + 1)*sample_rate, sample_rate)

        cur_info = self.data_infos[idx]

        cur_ids = cur_info['gt_ids']
        cur_boxes = cur_info['gt_boxes']
        world2lidar = cur_info['sensors']['LIDAR_TOP']['world2lidar']

        future_track = np.zeros((len(cur_boxes), future_frames+1, 2))
        future_mask = np.zeros((len(cur_boxes), future_frames+1))

        for i, (cur_id, cur_box) in enumerate(zip(cur_ids, cur_boxes)):
            for j, fut_idx in enumerate(fut_idx_list):
                if  0 <= fut_idx and fut_idx < len(self.data_infos):
                    adj_info = self.data_infos[fut_idx]
                    adj_ids = adj_info['gt_ids']

                    if adj_info['folder'] != cur_info['folder']:
                        break

                    if len(np.where(adj_ids == cur_id)[0]) == 0:
                        continue

                    adj_id = np.where(adj_ids == cur_id)[0][0]
                    adj2lidar = world2lidar @ adj_info['npc2world'][adj_id]
                    adj_xy = adj2lidar[0:2, 3]

                    future_track[i, j, :] = adj_xy
                    future_mask[i, j] = 1

        future_track_offset = future_track[:, 1:, :] - future_track[:, :-1, :]
        future_mask_offset = future_mask[:, 1:]

        # filter abnormal track
        agent_velocity = np.linalg.norm(future_track_offset / (10/self.split_group), axis=2)
        agent_velocity = np.concatenate([np.zeros_like(agent_velocity[:, :1]), agent_velocity], 1)
        agent_accelerate = abs(agent_velocity[:, 1:] - agent_velocity[:, :-1] / (10/self.split_group))

        for agent_idx, agent_acc in enumerate(agent_accelerate):
            abnorm_idx = np.where(agent_acc > 5)[0]
            if len(abnorm_idx) > 0:
                abnorm_idx = np.min(abnorm_idx)
                future_track_offset[agent_idx, abnorm_idx:, :] = -1
                future_mask_offset[agent_idx, abnorm_idx:] = 0

        return future_track_offset, future_mask_offset

    def get_box_attr_labels(self, idx, frames):
        adj_idx_list = range(idx, idx+(frames+1)*self.sample_rate, self.sample_rate)
        cur_frame = self.data_infos[idx]
        cur_box_names = cur_frame['gt_names']
        cur_boxes = cur_frame['gt_boxes'].copy()
        box_ids = cur_frame['gt_ids']
        future_track = np.zeros((len(box_ids), frames+1, 2))
        future_mask = np.zeros((len(box_ids), frames+1))
        future_yaw = np.zeros((len(box_ids), frames+1))
        gt_fut_goal = np.zeros((len(box_ids), 1))
        agent_lcf_feat = np.zeros((len(box_ids), 9))
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            agent_lcf_feat[i,0:2] = cur_boxes[i,0:2]
            agent_lcf_feat[i,2] = cur_boxes[i,6]
            agent_lcf_feat[i,3:5] = cur_boxes[i,7:]
            agent_lcf_feat[i,5:8] = cur_boxes[i,3:6]
            cur_box_name = cur_box_names[i]
            if cur_box_name in self.det_classes:
                agent_lcf_feat[i, 8] = self.det_classes.index(cur_box_name)
            else:
                agent_lcf_feat[i, 8] = -1

            box_id = box_ids[i]
            cur_box2lidar = world2lidar_lidar_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2, 3]
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx < 0 or adj_idx >= len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids'] == box_id)[0])==0:
                    continue
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]
                future_track[i, j,:] = adj_xy
                future_mask[i, j] = 1
                future_yaw[i, j] = np.arctan2(adj_box2lidar[1,0], adj_box2lidar[0,0])

            coord_diff = future_track[i, -1] - future_track[i, 0]
            if coord_diff.max() < 1.0: # static
                gt_fut_goal[i] = 9
            else:
                box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

        future_track_offset = future_track[:, 1:,:] - future_track[:, :-1, :]
        future_mask_offset = future_mask[:, 1:]
        future_track_offset[future_mask_offset==0] = 0
        future_yaw_offset = future_yaw[:, 1:] - future_yaw[:, :-1]
        mask1 = np.where(future_yaw_offset>np.pi)
        mask2 = np.where(future_yaw_offset<-np.pi)
        future_yaw_offset[mask1] -= np.pi*2
        future_yaw_offset[mask2] += np.pi*2

        attr_labels = np.concatenate(
            [future_track_offset.reshape(-1, frames * 2), future_mask_offset,
             gt_fut_goal, agent_lcf_feat, future_yaw_offset], axis=-1).astype(np.float32)
        return attr_labels.copy()

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def load_annotations(self, ann_file):
        data_infos = load(ann_file)

        for info in data_infos:
            info['timestamp'] = info['frame_idx'] / 10

        if self.name_mapping is not None:
            for info in data_infos:
                for i in range(len(info['gt_names'])):
                    if info['gt_names'][i] in self.name_mapping.keys():
                        info['gt_names'][i] = self.name_mapping[info['gt_names'][i]]

        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            timestamp=info['frame_idx'] / 10,
            pts_filename=osp.join(self.data_root, info['folder'], 'lidar/{:05}.laz'.format(info['frame_idx'])),
        )

        if self.modality['use_camera']:
            image_paths = []
            ego2img_rts = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            lidar2global = self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths.append(osp.join(self.data_root, cam_info['data_path']))
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2cam = self.invert_pose(cam2ego)
                ego2img = intrinsic_pad @ ego2cam
                lidar2cam = ego2cam @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam

                ego2img_rts.append(ego2img)
                lidar2img_rts.append(lidar2img)
                lidar2cam_rts.append(lidar2cam.T)
                cam_intrinsics.append(intrinsic_pad)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    ego2img=ego2img_rts,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    lidar2global=lidar2global,
                    cam_intrinsic=cam_intrinsics,
                )
            )

        annos = self.get_ann_info(index)
        input_dict.update(annos)

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        anns_results = dict()

        # det
        mask = (info['num_points'] != 0)
        gt_inds = info['gt_ids'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.det_classes:
                gt_labels_3d.append(self.det_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:, 0:7]

        if self.remap_box:
            if self.align_static_yaw:
                for i, name in enumerate(gt_names_3d):
                    if name in ["traffic_sign", "traffic_cone", "traffic_light"]:
                        gt_bboxes_3d[i, 6] = - (gt_bboxes_3d[i, 6] + np.pi / 2) - np.pi / 2
                    else:
                        temp = gt_bboxes_3d[i, 3]
                        gt_bboxes_3d[i, 3] = gt_bboxes_3d[i, 4]
                        gt_bboxes_3d[i, 4] = temp
                        gt_bboxes_3d[i, 6] = - (gt_bboxes_3d[i, 6] + np.pi / 2)
            else:
                temp = copy.deepcopy(gt_bboxes_3d[:, 3])
                gt_bboxes_3d[:, 3] = gt_bboxes_3d[:, 4]
                gt_bboxes_3d[:, 4] = temp
                gt_bboxes_3d[:, 6] = - (gt_bboxes_3d[:, 6] + np.pi/2)

        instance_inds = np.array(gt_inds, dtype=np.int32)

        gt_attr_labels = self.get_box_attr_labels(index, self.future_frames)
        gt_attr_labels = gt_attr_labels[mask]

        det_results = dict(
            gt_names=gt_names_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_attr_labels=gt_attr_labels,
            instance_inds=instance_inds,
        )

        # map
        map_geoms = self.get_map_info(index)
        map_infos = self.geom2anno(map_geoms)
        map_results = dict(
            map_geoms = map_geoms,
            map_infos = map_infos,
        )

        # motion
        gt_fut_trajs, gt_fut_masks = self.get_agent_trajs(index, self.future_frames, self.sample_rate)
        motion_results = dict()
        motion_results['gt_agent_fut_trajs'] = gt_fut_trajs.astype(np.float32)[mask]
        motion_results['gt_agent_fut_masks'] = gt_fut_masks.astype(np.float32)[mask]

        # planning
        plan_results = self.get_plan_info(index)

        # ego status
        ego_status = np.zeros(6)
        ego_status[0] = info['ego_vel'][0]               # speed
        ego_status[1:3] = info['ego_accel'][:2]          # acceleration
        ego_status[3:5] = info['ego_rotation_rate'][:2]  # angular_velocity
        ego_status[5] = info['steer']
        anns_results['ego_status'] = np.array(ego_status, np.float32)

        limit_vel = 20  # filter abnormal value
        limit_accel = limit_vel / (0.1 * self.split_group)
        ego_status_mask = np.ones(6)
        if info['ego_vel'][0] > limit_vel: ego_status_mask[0] = 0.0
        if np.linalg.norm(info['ego_accel'][:2]) > limit_accel: ego_status_mask[1:3] = 0.0
        anns_results['ego_status_mask'] = np.array(ego_status_mask, np.float32)

        # command
        command = self.command2hot(info['command_near'])
        anns_results['gt_ego_fut_cmd'] = command

        # target point
        theta_to_lidar = -(info['ego_yaw'] - np.pi / 2)
        rotation_matrix = np.array([[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)],
                                    [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])

        command_far_xy = info['command_far_xy'] - info['ego_translation'][:2]
        command_near_xy = info['command_near_xy'] - info['ego_translation'][:2]
        target_point_far = np.array(rotation_matrix @ command_far_xy, dtype=np.float32)
        target_point_near = np.array(rotation_matrix @ command_near_xy, dtype=np.float32)
        anns_results['target_point'] = target_point_far
        anns_results['target_point_far'] = target_point_far
        anns_results['target_point_near'] = target_point_near

        if self.with_next_target_point:
            index_next = index + 1
            while True:
                if index_next > len(self.data_infos) - 1:
                    info_next = info
                    break
                if info['folder'] != self.data_infos[index_next]['folder']:
                    info_next = self.data_infos[index_next-1]
                    break
                if (info['command_far_xy'] == self.data_infos[index_next]['command_far_xy']).all():
                    index_next += 1
                else:
                    info_next = self.data_infos[index_next]
                    break

            command_far_xy_next = info_next['command_far_xy'] - info['ego_translation'][:2]
            command_near_xy_next = info_next['command_near_xy'] - info['ego_translation'][:2]
            target_point_next_far = np.array(rotation_matrix @ command_far_xy_next, dtype=np.float32)
            target_point_next_near = np.array(rotation_matrix @ command_near_xy_next, dtype=np.float32)
            anns_results['target_point_next'] = target_point_next_far
            anns_results['target_point_next_far'] = target_point_next_far
            anns_results['target_point_next_near'] = target_point_next_near

        # update results
        anns_results.update(det_results)
        anns_results.update(map_results)
        anns_results.update(motion_results)
        anns_results.update(plan_results)

        return anns_results

    def get_map_info(self, index):
        polylines = []
        gt_labels = []

        ann_info = self.data_infos[index]
        map_info = self.map_infos[ann_info['town_name']]

        lane_types = map_info['lane_types']
        lane_points = map_info['lane_points']
        lane_sample_points = map_info['lane_sample_points']

        trigger_volumes_points = map_info['trigger_volumes_points']
        trigger_volumes_types = map_info['trigger_volumes_types']
        trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']

        world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2, 3]
        max_distance = 50

        if self.with_connect_lane:
            lane_ids = map_info['lane_ids']
            lane_topos = map_info['lane_topos']

            # Center, Solid, SolidSolid, Broken, BrokenSolid, SolidBroken, NONE
            for lane_type in self.map_element_class.keys():
                all_lane_list = []
                all_lane_id_list = []
                all_lane_target_id = []

                for idx in range(len(lane_sample_points)):
                    single_sample_points = lane_sample_points[idx]
                    distance = np.linalg.norm((single_sample_points[:, 0:2] - ego_xy), axis=-1)
                    if np.min(distance) < max_distance and lane_type==lane_types[idx]:
                        if lane_type == "Center":
                            all_lane_list.append([copy.deepcopy(lane_points[idx])])
                        else:
                            for target_id in lane_topos[idx]:
                                all_lane_list.append([copy.deepcopy(lane_points[idx])])
                                all_lane_id_list.append([copy.deepcopy(lane_ids[idx])])
                                all_lane_target_id.append(copy.deepcopy(target_id))

                if len(all_lane_list) and lane_type!= "Center":
                    all_lane_list, all_lane_id_list, all_lane_target_id = self.connect_lanes(
                        all_lane_list, all_lane_id_list, all_lane_target_id)

                for idx in range(len(all_lane_list)):
                    points = np.concatenate(all_lane_list[idx])
                    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
                    points_in_lidar = (world2lidar @ points.T).T
                    mask = ((points_in_lidar[:,0] > self.point_cloud_range[0]) &
                            (points_in_lidar[:,0] < self.point_cloud_range[3]) &
                            (points_in_lidar[:,1] > self.point_cloud_range[1]) &
                            (points_in_lidar[:,1] < self.point_cloud_range[4]))

                    change_points = np.diff(mask.astype(int))
                    starts = np.where(change_points == 1)[0] + 1
                    ends = np.where(change_points == -1)[0] + 1
                    if mask[0]:
                        starts = np.concatenate([[0], starts])
                    if mask[-1]:
                        ends = np.append(ends, len(mask))
                    assert len(starts) == len(ends)
                    for start, end in zip(starts, ends):
                        points_in_lidar_range = points_in_lidar[start:end, 0:2]
                        if len(points_in_lidar_range) > 1:
                            gt_label = self.map_element_class[lane_type]
                            gt_labels.append(gt_label)
                            polylines.append(LineString(points_in_lidar_range))
        else:
            chosed_idx = []
            # Center, Solid, SolidSolid, Broken, BrokenSolid, SolidBroken, NONE
            for idx in range(len(lane_sample_points)):
                single_sample_points = lane_sample_points[idx]
                distance = np.linalg.norm((single_sample_points[:, 0:2] - ego_xy), axis=-1)
                if np.min(distance) < max_distance:
                    chosed_idx.append(idx)

            for idx in chosed_idx:
                if not lane_types[idx] in self.map_element_class.keys():
                    continue
                points = lane_points[idx]
                points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
                points_in_lidar = (world2lidar @ points.T).T
                mask = ((points_in_lidar[:,0] > self.point_cloud_range[0]) &
                        (points_in_lidar[:,0] < self.point_cloud_range[3]) &
                        (points_in_lidar[:,1] > self.point_cloud_range[1]) &
                        (points_in_lidar[:,1] < self.point_cloud_range[4]))
                points_in_lidar_range = points_in_lidar[mask, 0:2]
                if len(points_in_lidar_range) > 1:
                    polylines.append(LineString(points_in_lidar_range))
                    gt_label = self.map_element_class[lane_types[idx]]
                    gt_labels.append(gt_label)

        # TrafficLight, StopSign
        for idx in range(len(trigger_volumes_points)):
            if not trigger_volumes_types[idx] in self.map_element_class.keys():
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
            points_in_lidar = (world2lidar @ points.T).T
            mask = ((points_in_lidar[:,0] > self.point_cloud_range[0]) &
                    (points_in_lidar[:,0] < self.point_cloud_range[3]) &
                    (points_in_lidar[:,1] > self.point_cloud_range[1]) &
                    (points_in_lidar[:,1] < self.point_cloud_range[4]))
            points_in_lidar_range = points_in_lidar[mask, 0:2]
            if mask.all():
                polylines.append(LineString(np.concatenate((points_in_lidar_range, points_in_lidar_range[0:1]), axis=0)))
                gt_label = self.map_element_class[trigger_volumes_types[idx]]
                gt_labels.append(gt_label)

        map_geoms = {}
        for label, polyline in zip(gt_labels, polylines):
            if label not in map_geoms:
                map_geoms[label] = []
            map_geoms[label].append(polyline)

        def filter_redundancy(lane_list):
            i = 0
            while i < len(lane_list):
                pop_i = False
                if polyline.length < 2.0:
                    lane_list.pop(i)
                    break
                j = i + 1
                while j < len(lane_list):
                    if lane_list[i].equals_exact(lane_list[j], 1e-3):
                        if lane_list[i].length < lane_list[j].length:
                            lane_list.pop(j)
                        else:
                            pop_i = True
                            lane_list.pop(i)
                            break
                    elif lane_list[i].intersects(lane_list[j]):
                        interaction = lane_list[i].intersection(lane_list[j])
                        union_length = lane_list[i].length + lane_list[j].length - interaction.length
                        ratio = interaction.length / union_length
                        if ratio > 0.75:
                            if lane_list[i].length < lane_list[j].length:
                                lane_list.pop(j)
                            else:
                                pop_i = True
                                lane_list.pop(i)
                                break
                        else:
                            j += 1
                    else:
                        j+=1
                if not pop_i:
                    i+=1
            return lane_list

        for label in map_geoms:
            map_geoms[label] = filter_redundancy(map_geoms[label])

        return map_geoms

    def get_plan_info(self, index):
        plan_results = dict()
        # temporal waypoint
        for plan_anchor in self.plan_anchor_types:
            if plan_anchor[0] != "temp": continue
            frequency = plan_anchor[1]
            interval = int(10 // float(frequency.split("hz")[0]))
            ego_stemporal_trajs, ego_temporal_masks = self.get_ego_temporal_trajs(index, self.future_frames, interval)
            plan_results['gt_ego_fut_trajs_{}'.format(frequency)] = ego_stemporal_trajs
            plan_results['gt_ego_fut_masks_{}'.format(frequency)] = ego_temporal_masks

            # add 2hz as default gt
            if plan_anchor[1] == '2hz':
                plan_results['gt_ego_fut_trajs'] = ego_stemporal_trajs
                plan_results['gt_ego_fut_masks'] = ego_temporal_masks

        # spatial waypoint
        for plan_anchor in self.plan_anchor_types:
            if plan_anchor[0] != "spat": continue

            if plan_anchor[1] in ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m"]:
                sample_distance = float(plan_anchor[1].split("m")[0])
                spatial_strategy = dict(mode="uniform", sample_distance=sample_distance)

            elif "lid" in plan_anchor[1]:  # e.g. lid_1m_20m
                distance_area = plan_anchor[1].split("lid_")[1]
                start_point = float(distance_area.split("_")[0].split("m")[0])
                end_point = float(distance_area.split("_")[1].split("m")[0])
                spatial_strategy = dict(mode="LID", start_distance=start_point, end_distance=end_point)

            else:
                raise NotImplementedError

            ego_spatial_trajs, ego_spatial_masks = self.get_ego_spatial_trajs(index, self.spatial_points, spatial_strategy)
            plan_results['gt_ego_spat_trajs_{}'.format(plan_anchor[1])] = ego_spatial_trajs
            plan_results['gt_ego_spat_masks_{}'.format(plan_anchor[1])] = ego_spatial_masks

        return plan_results

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]
            gt_boxes = sample_data['gt_boxes']
            for j in range(gt_boxes.shape[0]):
                # filter gt class
                class_name = sample_data['gt_names'][j]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                # filter gt range
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j, 0]) > range_x or abs(gt_boxes[j, 1]) > range_y:
                    continue
                sample_boxes.append(
                    DetectionBox(
                        sample_token=sample_data['folder'] + '_' + str(sample_data['frame_idx']),
                        translation=gt_boxes[j, 0:3],
                        size=gt_boxes[j, 3:6],
                        rotation=list(Quaternion(axis=[0, 0, 1], radians=-gt_boxes[j, 6] - np.pi/2)),
                        velocity=gt_boxes[j, 7:9],
                        num_pts=int(sample_data['num_points'][j]),
                        detection_name=class_name,
                        detection_score=-1.0,
                        attribute_name=class_name
                    )
                )
            all_annotations.add_boxes(sample_data['folder'] + '_' + str(sample_data['frame_idx']), sample_boxes)
        return all_annotations
    
    def load_motion_gt(self):
        from .evaluation.motion.motion_utils import MotionBox
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]
            gt_boxes = sample_data['gt_boxes']
            gt_fut_trajs, gt_fut_masks = self.get_agent_trajs(i, self.future_frames, self.sample_rate)
            for j in range(gt_boxes.shape[0]):
                class_name = sample_data['gt_names'][j]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j, 0]) > range_x or abs(gt_boxes[j, 1]) > range_y:
                    continue
                gt_fut_traj = gt_fut_trajs[j]
                gt_fut_traj = np.cumsum(gt_fut_traj, axis=0)
                gt_fut_traj += gt_boxes[j,0:2]
                mask = gt_fut_masks[j].astype(np.bool_)
                gt_fut_traj = gt_fut_traj[mask]
                sample_boxes.append(
                    MotionBox(
                        sample_token=sample_data['folder']+'_'+str(sample_data['frame_idx']),
                        translation=gt_boxes[j,0:3],
                        size=gt_boxes[j,3:6],
                        rotation=list(Quaternion(axis=[0, 0, 1], radians=-gt_boxes[j,6]-np.pi/2)),
                        velocity=gt_boxes[j,7:9],
                        num_pts=int(sample_data['num_points'][j]),
                        detection_name=class_name,
                        detection_score=-1.0,
                        attribute_name=class_name,
                        traj=gt_fut_traj
                    ))
            all_annotations.add_boxes(sample_data['folder']+'_'+str(sample_data['frame_idx']), sample_boxes)
        return all_annotations

    def _format_bbox(self, results, jsonfile_prefix=None, score_thresh=0.2):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        map_pred_annos = {}
        det_mapped_class_names = self.det_classes
        map_mapped_class_names = self.map_classes
        plan_annos = {}
        print('Start to convert detection format...')
        for sample_id, det in enumerate(track_iter_progress(results)):
            # format bbox
            if 'boxes_3d' in det:
                annos = []
                box3d = det['boxes_3d']
                scores = det['scores_3d']
                labels = det['labels_3d']

                if self.remap_box:
                    temp = copy.deepcopy(box3d[:, 3])
                    box3d[:, 3] = box3d[:, 4]
                    box3d[:, 4] = temp
                    # box3d[:, 6] = - (box3d[:, 6] + np.pi / 2)

                if hasattr(box3d, "gravity_center"):
                    box_gravity_center = box3d.gravity_center.numpy()
                    box_dims = box3d.dims.numpy()
                    nus_box_dims = box_dims[:, [1, 0, 2]]
                    box_yaw = box3d.yaw.numpy()
                else:
                    box3d = box3d.numpy()
                    box_gravity_center = box3d[..., :3].copy()
                    box_dims = box3d[..., 3:6].copy()
                    box_yaw = box3d[..., 6].copy()

                sample_token = self.data_infos[sample_id]['folder'] + '_' + str(self.data_infos[sample_id]['frame_idx'])
                for i in range(len(box3d)):

                    if scores[i] < score_thresh:
                        continue
                    quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                    velocity = [box3d[i, 7], box3d[i, 8]]
                    name = det_mapped_class_names[labels[i]]
                    nusc_anno = dict(
                        sample_token=sample_token,
                        translation=box_gravity_center[i].tolist(),
                        size=box_dims[i].tolist(),
                        rotation=quat,
                        velocity=velocity,
                        detection_name=name,
                        detection_score=scores[i].item(),
                        attribute_name=name)
                    nusc_anno.update(
                        dict(
                            trajs=det['trajs_3d'][i].numpy(),
                        )
                    )
                    annos.append(nusc_anno)
                nusc_annos[sample_token] = annos

            # format map
            if 'vectors' in det:
                map_pred_anno = {}
                vecs = output_to_vecs(det)
                sample_token = self.data_infos[sample_id]['folder'] + '_' + str(self.data_infos[sample_id]['frame_idx'])
                map_pred_anno['sample_token'] = sample_token
                pred_vec_list = []
                for i, vec in enumerate(vecs):
                    name = map_mapped_class_names[vec['label']]
                    anno = dict(
                        # sample_token=sample_token,
                        pts=vec['pts'],
                        pts_num=len(vec['pts']),
                        cls_name=name,
                        type=vec['label'],
                        confidence_level=vec['score'])
                    pred_vec_list.append(anno)


                map_pred_anno['vectors'] = pred_vec_list
                map_pred_annos[sample_token] = map_pred_anno

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
            'map_results': map_pred_annos,
            'plan_results': plan_annos
        }

        return nusc_submissions

    def _format_map(self):
        gt_annos = []
        print('Start to convert gt map format...')
        if not os.path.exists(self.map_ann_file):
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.map_classes
            for sample_id in range(dataset_length):
                gt_anno = dict()

                sample_token = self.data_infos[sample_id]['folder'] + '_' + str(self.data_infos[sample_id]['frame_idx'])
                gt_anno['sample_token'] = sample_token

                map_geoms = self.get_map_info(sample_id)

                gt_vecs = []
                gt_labels = []
                for key, val in map_geoms.items():
                    gt_vecs.extend(val)
                    gt_labels.extend([key for _ in range(len(val))])

                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors'] = gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()

            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(gt_annos, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        if isinstance(results, dict):
            results = results['bbox_results']
        assert isinstance(results, list)

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == 'metric_results':
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})

        # format map annotations
        if not os.path.exists(self.map_ann_file):
            self._format_map()

        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         map_metric='chamfer',
                         result_name='pts_bbox'):

        detail = dict()

        if isinstance(result_path, str):
            with open(result_path, 'r') as f:
                result_data = json.load(f)
        else:
            result_data = result_path

        # det
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']

        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:
            # Compute APs.
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        # print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print("\n")
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err']))

        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix, self.eval_cfg['err_name_maping'][k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics_summary['mean_ap']
        print("\n")

        # map
        from .evaluation.map.mean_ap import eval_map
        from .evaluation.map.mean_ap import format_res_gt_by_classes

        print('Formating results & gts by classes')
        map_results = result_data['map_results']
        map_annotations = load(self.map_ann_file)

        cls_gens, cls_gts = format_res_gt_by_classes(self.map_ann_file,
                                                     map_results,
                                                     map_annotations,
                                                     cls_names=self.map_classes,
                                                     num_pred_pts_per_instance=self.map_num_pts,
                                                     eval_use_same_gt_sample_num_flag=True,
                                                     pc_range=self.point_cloud_range)
        map_metrics = map_metric if isinstance(map_metric, list) else [map_metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in map_metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        for metric in map_metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)
            if metric == 'chamfer':
                thresholds = [0.5, 1.0, 1.5]
            elif metric == 'iou':
                thresholds = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.map_num_classes))
            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                map_results,
                                map_annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.map_classes,
                                logger=logger,
                                num_pred_pts_per_instance=self.map_num_pts,
                                pc_range=self.point_cloud_range,
                                metric=metric)
                for j in range(self.map_num_classes):
                    cls_aps[i, j] = cls_ap[j]['ap']
            for i, name in enumerate(self.map_classes):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric, name)] = cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()
            for i, name in enumerate(self.map_classes):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]
        print("\n")
        return detail

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):

        from .evaluation.motion.motion_eval import MotionEval
        from .evaluation.motion.motion_utils import MotionBox

        gt_boxes = self.load_motion_gt()

        if isinstance(results, str):
            with open(results, 'r') as f:
                result_data = json.load(f)
        else:
            result_data = results

        pred_boxes = EvalBoxes.deserialize(result_data['results'], MotionBox)

        nusc_eval = MotionEval(
            gt_boxes,
            pred_boxes,
            config=copy.deepcopy(self.eval_cfg),
            verbose=False)
        metrics, _ = nusc_eval.evaluate()
        
        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)
        return metrics
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 map_metric='chamfer',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['img_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 eval_mode=None):
        print('\n')

        if 'plan_L2_1s' in results[0]['metric_results']:
            print('-------------- Planning --------------')
            metric_dict = None
            num_valid = 0
            for res in results:
                metric_results = res['metric_results']
                if metric_results['fut_valid_flag']:
                    num_valid += 1
                else:
                    continue
                if metric_dict is None:
                    metric_dict = copy.deepcopy(metric_results)
                else:
                    for k in metric_results.keys():
                        metric_dict[k] += res['metric_results'][k]
            for k in metric_dict:
                metric_dict[k] = metric_dict[k] / num_valid

            L2_err = []
            Col_rate = []
            for k in metric_dict:
                if "plan_L2" in k:
                    L2_err.append(metric_dict[k])
                    print("{}: {:.4f}".format(k, metric_dict[k]))
                if "plan_obj_box_col" in k:
                    Col_rate.append(metric_dict[k] * 100)
                    print("{}: {:.4f} % ".format(k, metric_dict[k] * 100))

        with_det = eval_mode.get('with_det', True) if eval_mode else True
        with_map = eval_mode.get('with_map', True) if eval_mode else True
        with_motion = eval_mode.get('with_motion', True) if eval_mode else True

        results_dict = dict()
        if with_det or with_map or with_motion:
            print('-------------- perception --------------')
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    if with_det or with_map:
                        ret_dict = self._evaluate_single(result_files[name], metric=metric, map_metric=map_metric)
                        results_dict.update(ret_dict)
                    if with_motion:
                        motion_ret_dict = self._evaluate_single_motion(result_files[name], jsonfile_prefix, logger=logger)
                        results_dict.update(motion_ret_dict)
            elif isinstance(result_files, str):
                if with_det or with_map:
                    results_dict.update(self._evaluate_single(result_files, metric=metric, map_metric=map_metric))
                if with_motion:
                    results_dict.update(self._evaluate_single_motion(result_files, jsonfile_prefix, logger=logger))

            if tmp_dir is not None:
                tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

def output_to_vecs(detection):
    pts = detection['vectors']
    scores = detection['scores']
    labels = detection['labels']

    vec_list = []
    for i in range(len(pts)):
        vec = dict(
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list