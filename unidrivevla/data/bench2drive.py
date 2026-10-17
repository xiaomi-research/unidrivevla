"""Bench2Drive from the official infos pkl (tools/data_converter/bench2drive_converter.py) in ego or LiDAR axes.

Same frame as the nuScenes source: the current LiDAR frame turned onto the ego axes. The official dataset reorders the
pkl with split_group=5, so a 0.5 s step is five list positions of one clip (the converter drops frames without boxes,
so this is not frame_idx + 5). Ego futures are get_ego_temporal_trajs, including the same-folder gate two reordered
slots back; agent futures are get_agent_trajs with its acceleration cut. Boxes get remap_box (length, width, yaw of
the heading). The map is get_map_info.

    b2d:
      info_dir: .../data/infos          # b2d_infos_{train,val}.pkl, b2d_map_infos.pkl
      data_root: .../data/bench2drive   # the pkl's image paths are relative to it
      map_types: [Broken, Solid, SolidSolid, Center]
"""

import copy
import math
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from shapely.geometry import LineString
from torch.utils.data import Dataset

from unidrivevla.data.registry import register
from unidrivevla.data.sample import camera, transform_boxes, transform_xy, yaw_of

CAMERAS = {"CAM_FRONT": "front", "CAM_FRONT_LEFT": "front_left", "CAM_FRONT_RIGHT": "front_right",
           "CAM_BACK": "back", "CAM_BACK_LEFT": "back_left", "CAM_BACK_RIGHT": "back_right"}
COMMANDS = ["left", "right", "straight", "lane_follow", "change_lane_left", "change_lane_right"]


def command_name(command: int) -> str:
    """command2hot: negatives are lane follow; the one-hot write wraps, so 0 lands on the last command."""
    if command < 0:
        command = 4
    return COMMANDS[command - 1]


STEP_FRAMES, HISTORY, FUTURE = 5, 3, 6  # split_group, and the 2 Hz interval at 10 Hz
LANE_RADIUS_M = 50.0
LIDAR_RANGE = (-15.0, -30.0, 15.0, 30.0)  # x_min, y_min, x_max, y_max: Bench2DriveDataset.point_cloud_range
B2D_DET_CLASSES = ("car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light",
                   "pedestrian", "others")

NAME_MAPPING = {
    **dict.fromkeys(["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"], "bicycle"),
    **dict.fromkeys([
        "vehicle.audi.etron", "vehicle.chevrolet.impala", "vehicle.dodge.charger_2020", "vehicle.dodge.charger_police",
        "vehicle.dodge.charger_police_2020", "vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020",
        "vehicle.mini.cooper_s_2021", "vehicle.mercedes.coupe_2020", "vehicle.ford.mustang", "vehicle.nissan.patrol_2021",
        "vehicle.audi.tt", "vehicle.ford.crown", "vehicle.tesla.model3",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked"], "car"),
    # the official mapping lists the parked VolkswagenT2 as car and then van; the later entry wins
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    "vehicle.carlamotors.firetruck": "truck",
    **dict.fromkeys(["traffic.speed_limit.30", "traffic.speed_limit.40", "traffic.speed_limit.50",
                     "traffic.speed_limit.60", "traffic.speed_limit.90", "traffic.speed_limit.120", "traffic.stop",
                     "traffic.yield"], "traffic_sign"),
    "traffic.traffic_light": "traffic_light",
    **dict.fromkeys(["static.prop.warningconstruction", "static.prop.warningaccident", "static.prop.trafficwarning",
                     "static.prop.constructioncone"], "traffic_cone"),
    **dict.fromkeys([f"walker.pedestrian.{i:04d}" for i in (1, 3, 4, 5, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                                            25, 27, 30, 31, 32, 34, 35, 41, 42, 46, 47)], "pedestrian"),
    "static.prop.dirtdebris01": "others",
    "static.prop.dirtdebris02": "others",
}


def pose_in(frame: np.ndarray) -> np.ndarray:
    return np.array([frame[0, 3], frame[1, 3], yaw_of(frame)], np.float32)


def status_mask(info) -> np.ndarray:
    """Bench2DriveDataset's ego_status_mask on the profile's status_layout (vel_x, acc_x, acc_y, rot_x, rot_y, steer),
    the ego status loss's weight: no speed above 20 m/s, no acceleration above 40 m/s^2."""
    mask = np.ones(6, np.float32)
    if info["ego_vel"][0] > 20:
        mask[0] = 0.0
    if np.linalg.norm(np.asarray(info["ego_accel"], np.float64)[:2]) > 40:
        mask[1:3] = 0.0
    return mask


def remap_boxes(raw: np.ndarray):
    """Converter boxes (x, y, z, w, l, h, yaw of the old mmdet3d LiDAR box, vx, vy) -> canonical boxes, velocity."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 9)
    boxes = np.concatenate([raw[:, :3], raw[:, [4]], raw[:, [3]], raw[:, [5]], -(raw[:, [6]] + math.pi / 2)], axis=1)
    return boxes, raw[:, 7:9]


def connect_lanes(all_line_list, all_line_id_list, all_target_id):
    """Bench2DriveDataset.connect_lanes: joins lane pieces along their topology targets."""
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
            same_ids = (len(all_line_id_list[i]) == len(all_line_id_list[j]) and
                        (np.array(all_line_id_list[i]) == np.array(all_line_id_list[j])).all())
            if same_ids:
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
    return all_line_list


def filter_redundancy(lane_list, stub_length):
    """Bench2DriveDataset.get_map_info.filter_redundancy. stub_length is the length of the last polyline the
    function built: the official check reads that name, not the line it is about to drop."""
    i = 0
    while i < len(lane_list):
        pop_i = False
        if stub_length < 2.0:
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
                shared = lane_list[i].intersection(lane_list[j]).length
                union = lane_list[i].length + lane_list[j].length - shared
                if shared / union > 0.75:
                    if lane_list[i].length < lane_list[j].length:
                        lane_list.pop(j)
                    else:
                        pop_i = True
                        lane_list.pop(i)
                        break
                else:
                    j += 1
            else:
                j += 1
        if not pop_i:
            i += 1
    return lane_list


def in_range(points: np.ndarray) -> np.ndarray:
    x_min, y_min, x_max, y_max = LIDAR_RANGE
    return (points[:, 0] > x_min) & (points[:, 0] < x_max) & (points[:, 1] > y_min) & (points[:, 1] < y_max)


def to_lidar(points: np.ndarray, world2lidar: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1) @ world2lidar.T)[:, :2]


class Bench2DriveSource(Dataset):
    def __init__(self, info_path: str, map_path: str, data_root: str, map_types, bev_filter: bool = False,
                 det_classes=B2D_DET_CLASSES, frame: str = "ego"):
        if frame not in ("ego", "lidar"):
            raise ValueError(f"Bench2Drive frame must be 'ego' or 'lidar', got {frame!r}")
        self.frame = frame
        with open(info_path, "rb") as f:
            self.infos = pickle.load(f)
        with open(map_path, "rb") as f:
            self.maps = pickle.load(f)
        self.data_root, self.map_types = data_root, list(map_types)
        self.bev_filter = bev_filter  # the val/test pipeline's BEVObjectRangeFilter; training uses the 55 m circle
        self.det_classes = list(det_classes)
        self.index = {(info["folder"], int(info["frame_idx"])): i for i, info in enumerate(self.infos)}
        self.tokens = [f"{info['folder']}/{int(info['frame_idx']):05d}" for info in self.infos]
        self.by_token = {token: i for i, token in enumerate(self.tokens)}
        # frame_idx / 10 counts seconds within a clip; clips start 10^4 s apart so no two of them look consecutive
        self.clip_start = {folder: 1e4 * k for k, folder in enumerate(sorted({info["folder"] for info in self.infos}))}
        groups = [list(range(i, len(self.infos), STEP_FRAMES)) for i in range(STEP_FRAMES)]
        self.ordered = [i for group in groups for i in group]  # Bench2DriveDataset._split_data_infos
        lengths = [len(group) for group in groups]
        self.group_cumsum = np.cumsum(np.array([0, *lengths[:-1]], dtype=np.int64))
        self.reordered_of = np.empty(len(self.infos), np.int64)
        for reordered, original in enumerate(self.ordered):
            self.reordered_of[original] = reordered

    def __len__(self):
        return len(self.infos)

    def sequences(self):
        """Same-folder runs of the split_group reorder: GroupInBatchSampler's groups, before it cuts each in two."""
        runs, run, previous = [], [], None
        for original in self.ordered:
            folder = self.infos[original]["folder"]
            if run and folder != previous:
                runs.append(run)
                run = []
            run.append(original)
            previous = folder
        if run:
            runs.append(run)
        return runs

    def neighbour(self, index: int, step: int) -> Optional[dict]:
        """The frame `step` 0.5 s slots away in list order, inside the clip."""
        other = index + STEP_FRAMES * step
        if other < 0 or other >= len(self.infos) or self.infos[other]["folder"] != self.infos[index]["folder"]:
            return None
        return self.infos[other]

    def _next_reordered(self, adj_idx: int) -> int:
        """get_ego_temporal_trajs' one-frame step across the split_group blocks."""
        group_diffs = adj_idx - self.group_cumsum
        group_diffs = np.where(group_diffs < 0, len(self.infos), group_diffs)
        group_idx = int(np.argmin(group_diffs))
        group_diff = int(group_diffs[group_idx])
        if group_idx >= STEP_FRAMES - 1:
            group_idx = 0
            group_diff += 1
        else:
            group_idx += 1
        return int(self.group_cumsum[group_idx] + group_diff)

    def _ego_future(self, index: int, frame: Optional[str] = None):
        """get_ego_temporal_trajs: selected-frame offsets, mask, and the original index of each step.

        An index past the reordered list is skipped, not a stop, so a later step can still be filled and the
        offset across the hole is measured from the zero that was left there."""
        info = self.infos[index]
        adj = int(self.reordered_of[index])
        seq = [adj]
        for _ in range(FUTURE * STEP_FRAMES):
            adj = self._next_reordered(adj)
            seq.append(adj)
        seq = seq[::STEP_FRAMES]
        track = np.zeros((len(seq), 2), np.float64)
        seen = np.zeros(len(seq), np.float64)
        origins = [None] * len(seq)
        if self.infos[int(self.ordered[seq[0] - 2])]["folder"] == info["folder"]:
            for j, reordered in enumerate(seq):
                if not 0 <= reordered < len(self.infos):
                    continue
                original = int(self.ordered[reordered])
                other = self.infos[original]
                if other["folder"] != info["folder"]:
                    break
                track[j] = self.relative_pose(info, other, frame)[:2]
                seen[j] = 1
                origins[j] = original
        offsets = track[1:] - track[:-1]
        for j in range(FUTURE):
            if seen[j + 1] == 0:
                offsets[j] = 0
        return offsets, seen[1:].astype(np.float32), origins[1:]

    @staticmethod
    def axes(info) -> np.ndarray:
        """The actual LiDAR -> ego rotation, retained for canonical metrics in either training frame."""
        out = np.eye(4)
        out[:3, :3] = np.asarray(info["sensors"]["LIDAR_TOP"]["lidar2ego"])[:3, :3]
        return out

    def working_axes(self, info, frame: Optional[str] = None) -> np.ndarray:
        """Map raw LiDAR coordinates into the selected head frame; native mode keeps raw axes."""
        frame = self.frame if frame is None else frame
        if frame == "ego":
            return self.axes(info)
        if frame == "lidar":
            return np.eye(4)
        raise ValueError(f"Bench2Drive frame must be 'ego' or 'lidar', got {frame!r}")

    def to_current(self, info, other, frame: Optional[str] = None) -> np.ndarray:
        """`other`'s LiDAR frame -> the selected head frame of `info`."""
        return self.working_axes(info, frame) @ info["sensors"]["LIDAR_TOP"]["world2lidar"] @ np.linalg.inv(
            other["sensors"]["LIDAR_TOP"]["world2lidar"])

    def relative_pose(self, info, other, frame: Optional[str] = None) -> np.ndarray:
        return pose_in(self.to_current(info, other, frame) @ np.linalg.inv(self.working_axes(other, frame)))

    def features(self, index: int) -> Dict:
        info = self.infos[index]
        axes = self.working_axes(info)
        ego2lidar = np.linalg.inv(np.asarray(info["sensors"]["LIDAR_TOP"]["lidar2ego"]))
        cameras = []
        for name, view in CAMERAS.items():
            cam = info["sensors"][name]
            path = os.path.join(self.data_root, cam["data_path"])
            with Image.open(path) as image:
                size = image.size
            cameras.append(camera(view, path, cam["intrinsic"], axes @ ego2lidar @ np.asarray(cam["cam2ego"]), size))
        history, earliest = [], np.zeros(3, np.float32)
        for step in range(-1, -HISTORY - 1, -1):  # a clip's first frames repeat the earliest pose they have
            other = self.neighbour(index, step)
            earliest = self.relative_pose(info, other) if other is not None else earliest
            history.insert(0, earliest)
        return {
            "cameras": cameras,
            "history_trajectory": np.stack(history + [np.zeros(3, np.float32)]),
            "ego_velocity": np.asarray(info["ego_vel"], np.float32),
            "ego_acceleration": np.asarray(info["ego_accel"], np.float32),
            "ego_angular_velocity": np.asarray(info["ego_rotation_rate"], np.float32),
            "ego_steer": np.float32(info["steer"]),
            "command": command_name(int(info["command_near"])),
            "timestamp": self.clip_start[info["folder"]] + int(info["frame_idx"]) / 10,
            "ego2global": np.linalg.inv(np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"])) @ np.linalg.inv(axes),
            "coordinate_frame": self.frame,
            "native_to_ego": self.axes(info) if self.frame == "lidar" else np.eye(4),
        }

    def trajectory(self, index: int, frame: Optional[str] = None):
        """Absolute poses whose steps are get_ego_temporal_trajs' offsets."""
        info = self.infos[index]
        offsets, mask, origins = self._ego_future(index, frame)
        poses = np.zeros((FUTURE, 3), np.float32)
        poses[:, :2] = np.cumsum(offsets, axis=0)
        held = 0.0
        for step, original in enumerate(origins):
            if original is None:
                poses[step, 2] = held
                continue
            held = float(self.relative_pose(info, self.infos[original], frame)[2])
            poses[step, 2] = held
        return poses, mask

    def agent_future(self, index: int, ids: np.ndarray, centers: np.ndarray):
        """get_agent_trajs' offsets (reordered-list steps, then the acceleration cut), as positions from the box centres."""
        info = self.infos[index]
        ids = np.asarray(ids)
        n = len(ids)
        future = np.zeros((n, FUTURE, 2), np.float32)
        mask = np.zeros((n, FUTURE), np.float32)
        if n == 0:
            return future, mask
        start = int(self.reordered_of[index])
        track = np.zeros((n, FUTURE + 1, 2), np.float64)
        seen = np.zeros((n, FUTURE + 1), np.float64)
        world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"])
        for j, reordered in enumerate(range(start, start + FUTURE + 1)):
            if not 0 <= reordered < len(self.infos):
                continue
            other = self.infos[int(self.ordered[reordered])]
            if other["folder"] != info["folder"]:
                break
            where = {}
            for k, box_id in enumerate(other["gt_ids"]):
                where.setdefault(int(box_id), k)
            for row, box_id in enumerate(ids):
                k = where.get(int(box_id))
                if k is None:
                    continue
                track[row, j] = (world2lidar @ other["npc2world"][k])[:2, 3]
                seen[row, j] = 1
        offsets = track[:, 1:] - track[:, :-1]
        mask = seen[:, 1:].astype(np.float32)
        step_s = 10.0 / STEP_FRAMES
        velocity = np.linalg.norm(offsets / step_s, axis=2)
        velocity = np.concatenate([np.zeros((n, 1)), velocity], axis=1)
        accelerate = np.abs(velocity[:, 1:] - velocity[:, :-1] / step_s)
        for row, acc in enumerate(accelerate):
            bad = np.where(acc > 5)[0]
            if len(bad):
                cut = int(bad.min())
                offsets[row, cut:] = -1
                mask[row, cut:] = 0
        rotated = offsets @ self.working_axes(info)[:2, :2].T
        # cumsum keeps a masked hole (track left at 0, or the acceleration cut's -1) inside the position, so the
        # adapter's step from the last observed point absorbs it and the supervised cumulative path matches
        future = np.asarray(centers, np.float64)[:, None, :] + np.cumsum(rotated, axis=1)
        return future.astype(np.float32), mask

    def map_polylines(self, info, map_types=None) -> List:
        town = self.maps[info["town_name"]]
        world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"])
        ego_xy = np.linalg.inv(world2lidar)[:2, 3]
        near = [i for i, pts in enumerate(town["lane_sample_points"])
                if np.min(np.linalg.norm(np.asarray(pts)[:, :2] - ego_xy, axis=-1)) < LANE_RADIUS_M]
        map_types = self.map_types if map_types is None else map_types
        lines: Dict[str, List[LineString]] = {name: [] for name in map_types}
        for lane_type in map_types:
            chains, ids, targets = [], [], []
            for i in (i for i in near if town["lane_types"][i] == lane_type):
                if lane_type == "Center":
                    chains.append([copy.deepcopy(town["lane_points"][i])])
                    continue
                for target in town["lane_topos"][i]:
                    chains.append([copy.deepcopy(town["lane_points"][i])])
                    ids.append([copy.deepcopy(town["lane_ids"][i])])
                    targets.append(copy.deepcopy(target))
            if chains and lane_type != "Center":
                chains = connect_lanes(chains, ids, targets)
            for chain in chains:
                points = to_lidar(np.concatenate(chain), world2lidar)
                inside = np.concatenate([[False], in_range(points), [False]]).astype(int)
                starts, ends = np.where(np.diff(inside) == 1)[0], np.where(np.diff(inside) == -1)[0]
                lines[lane_type] += [LineString(points[s:e]) for s, e in zip(starts, ends) if e - s > 1]
        for points, kind in zip(town["trigger_volumes_points"], town["trigger_volumes_types"]):
            points = to_lidar(points, world2lidar)
            if kind in lines and in_range(points).all():
                lines[kind].append(LineString(np.concatenate([points, points[:1]])))
        flat = [line for group in lines.values() for line in group]
        stub = flat[-1].length if flat else 0.0
        for kind in lines:
            lines[kind] = filter_redundancy(lines[kind], stub)
        axes = self.working_axes(info)
        return [(kind, transform_xy(np.asarray(line.coords), axes))
                for kind, group in lines.items() for line in group]

    def targets(self, index: int) -> Dict:
        info = self.infos[index]
        keep = np.asarray(info["num_points"]) != 0
        if self.bev_filter and keep.any():
            xy = np.asarray(info["gt_boxes"])[:, :2]
            x0, y0, x1, y1 = LIDAR_RANGE
            keep = keep & (xy[:, 0] > x0) & (xy[:, 1] > y0) & (xy[:, 0] < x1) & (xy[:, 1] < y1)
        boxes, velocity = transform_boxes(*remap_boxes(np.asarray(info["gt_boxes"])[keep]), self.working_axes(info))
        future, future_mask = self.agent_future(index, np.asarray(info["gt_ids"])[keep], boxes[:, :2])
        trajectory, mask = self.trajectory(index)
        return {
            "trajectory": trajectory,
            "trajectory_mask": mask,
            "boxes": boxes,
            "box_velocity": velocity,
            "box_names": [NAME_MAPPING.get(str(n), str(n)) for n in np.asarray(info["gt_names"])[keep]],
            "box_ids": [str(i) for i in np.asarray(info["gt_ids"])[keep]],
            "agent_future": future,
            "agent_future_mask": future_mask,
            "map": self.map_polylines(info),
            "ego_status_mask": status_mask(info),
        }

    def __getitem__(self, index: int):
        return self.features(index), self.targets(index), self.tokens[index]

    def eval_info(self, token: str) -> Dict:
        index = self.by_token[token]
        info = self.infos[index]
        # Main's online metrics are expressed on ego axes even if the head trains on raw LiDAR axes.
        trajectory, mask = self.trajectory(index, frame="ego")
        _offsets, _mask, frames = self._ego_future(index, frame="ego")
        boxes = []
        for original in frames:
            if original is None:
                boxes.append(np.zeros((0, 5), np.float32))
                continue
            other = self.infos[original]
            keep = np.asarray(other["num_points"]) != 0
            moved, _ = transform_boxes(*remap_boxes(np.asarray(other["gt_boxes"])[keep]),
                                       self.to_current(info, other, frame="ego"))
            boxes.append(moved[:, [0, 1, 3, 4, 6]])
        return {"trajectory": trajectory, "mask": mask, "boxes": boxes,
                "axes": self.axes(info)[:2, :2].astype(np.float32), **self._planning_targets(index)}

    def _planning_targets(self, index: int) -> Dict:
        """Main test-pipeline actors and gt_attr_labels for the online STP-3 planning metric."""
        info = self.infos[index]
        raw_boxes = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 9)
        names = [NAME_MAPPING.get(str(name), str(name)) for name in np.asarray(info["gt_names"])]
        labels = np.array([self.det_classes.index(name) if name in self.det_classes else -1 for name in names])
        x0, y0, x1, y1 = LIDAR_RANGE
        keep = ((np.asarray(info["num_points"]) != 0) & (labels >= 0)
                & (raw_boxes[:, 0] > x0) & (raw_boxes[:, 1] > y0)
                & (raw_boxes[:, 0] < x1) & (raw_boxes[:, 1] < y1))
        attrs = self._planning_attributes(index, labels)
        boxes = raw_boxes.copy()
        boxes[:, [3, 4]] = boxes[:, [4, 3]]  # Bench2DriveDataset.get_ann_info(remap_box=True)
        boxes[:, 6] = -(boxes[:, 6] + np.pi / 2)
        return {"planning_boxes": boxes[keep].astype(np.float32),
                "planning_attr": attrs[keep].astype(np.float32)}

    def _planning_attributes(self, index: int, labels: np.ndarray) -> np.ndarray:
        """Port of Bench2DriveDataset.get_box_attr_labels, in its reordered split_group sequence."""
        info = self.infos[index]
        boxes = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 9)
        ids = np.asarray(info["gt_ids"])
        n = len(ids)
        track = np.zeros((n, FUTURE + 1, 2), np.float64)
        seen = np.zeros((n, FUTURE + 1), np.float64)
        yaw = np.zeros((n, FUTURE + 1), np.float64)
        world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], np.float64)
        start = int(self.reordered_of[index])
        for step in range(FUTURE + 1):
            reordered = start + step
            if reordered >= len(self.infos):
                break
            other = self.infos[int(self.ordered[reordered])]
            if other["folder"] != info["folder"]:
                break
            positions = {}
            for row, box_id in enumerate(np.asarray(other["gt_ids"])):
                positions.setdefault(int(box_id), []).append(row)
            for row, box_id in enumerate(ids):
                matches = positions.get(int(box_id), [])
                if not matches:
                    continue
                if len(matches) != 1:
                    raise AssertionError(f"duplicate Bench2Drive track id {box_id} in {other['folder']}")
                matrix = world2lidar @ np.asarray(other["npc2world"][matches[0]], np.float64)
                track[row, step] = matrix[:2, 3]
                yaw[row, step] = np.arctan2(matrix[1, 0], matrix[0, 0])
                seen[row, step] = 1
        offsets = track[:, 1:] - track[:, :-1]
        yaw_offsets = yaw[:, 1:] - yaw[:, :-1]
        yaw_offsets[yaw_offsets > np.pi] -= 2 * np.pi
        yaw_offsets[yaw_offsets < -np.pi] += 2 * np.pi
        goals = np.zeros((n, 1), np.float64)
        local = np.zeros((n, 9), np.float64)
        local[:, :2] = boxes[:, :2]
        local[:, 2] = boxes[:, 6]
        local[:, 3:5] = boxes[:, 7:9]
        local[:, 5:8] = boxes[:, 3:6]
        local[:, 8] = labels
        for row in range(n):
            delta = track[row, -1] - track[row, 0]
            goals[row, 0] = 9 if delta.max() < 1.0 else (np.arctan2(delta[1], delta[0]) + np.pi) // (np.pi / 4)
        return np.concatenate([offsets.reshape(n, FUTURE * 2), seen[:, 1:], goals, local, yaw_offsets], axis=-1).astype(np.float32)


@register("b2d")
def bench2drive_loader(cfg, agent, split: str):
    source = cfg.b2d
    name = {"train": "train", "val": "val", "test": "val"}[split]
    profile = getattr(agent, "profile", None)
    frame = getattr(profile, "frame", None) or OmegaConf.select(cfg, "profile.frame") or "ego"
    return Bench2DriveSource(os.path.join(source.info_dir, f"b2d_infos_{name}.pkl"),
                             os.path.join(source.info_dir, "b2d_map_infos.pkl"), source.data_root,
                             OmegaConf.select(source, "map_types") or ["Broken", "Solid", "SolidSolid", "Center"],
                             bev_filter=split != "train", det_classes=profile.det_classes if profile else B2D_DET_CLASSES,
                             frame=frame)
