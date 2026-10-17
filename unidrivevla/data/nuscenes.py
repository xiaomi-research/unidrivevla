"""nuScenes from the official infos pkl (tools/data_converter/nuscenes_converter.py) in ego or raw LiDAR axes.

The frame is the current LiDAR frame turned onto the ego axes (x forward, y left): its origin is the LiDAR, the point
whose trajectory the ST-P3 / UniAD planning metrics measure, so their numbers carry over (a rotation keeps L2 and the
collision check). Boxes, motion, map and cameras, which the pkl holds in the LiDAR frame, go through that rotation; ego
poses come from the neighbouring samples of the scene.

    nuscenes:
      info_dir: .../data/infos          # nuscenes_infos_{train,val}.pkl
      path_root: .../Code/nuScenes      # directory the converter ran in (the pkl's image paths are relative to it)
      data_root: .../data/nuscenes      # for v1.0-*/scene.json, only with occ_root
      version: v1.0-trainval
      occ_root: null                    # OccWorld / Occ3D gts/{scene name}/{token}/labels.npz
"""

import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from unidrivevla.data.registry import register
from unidrivevla.data.sample import camera, transform_xy, yaw_of

CAMERAS = {"CAM_FRONT": "front", "CAM_FRONT_LEFT": "front_left", "CAM_FRONT_RIGHT": "front_right",
           "CAM_BACK": "back", "CAM_BACK_LEFT": "back_left", "CAM_BACK_RIGHT": "back_right"}
COMMANDS = ("right", "left", "straight")  # order of the converter's gt_ego_fut_cmd one-hot
MAP_TYPES = ("ped_crossing", "divider", "boundary")  # map_annos keys 0, 1, 2
HISTORY, FUTURE = 3, 6
LIDAR_BOX_RADIUS = 55.0  # CircleObjectRangeFilter, xy radius in the LiDAR frame (the ego-axis roll changes it)
MAP_SAMPLES = 20  # VectorizeMap's sample_num: each extractor line resampled whole in the LiDAR frame


def map_line(line) -> np.ndarray:
    """VectorizeMap.interp_fixed_num: MAP_SAMPLES points evenly spaced along the polyline."""
    from shapely.geometry import LineString

    line = LineString(np.asarray(line, np.float64)[:, :2])
    return np.array([line.interpolate(d).coords[0] for d in np.linspace(0, line.length, MAP_SAMPLES)])


def rigid(rotation, translation) -> np.ndarray:
    out = np.eye(4)
    rotation = np.asarray(rotation, dtype=np.float64)
    out[:3, :3] = Quaternion(rotation).rotation_matrix if rotation.shape == (4,) else rotation
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def pose_in(frame: np.ndarray) -> np.ndarray:
    return np.array([frame[0, 3], frame[1, 3], yaw_of(frame)], np.float32)


def boxes_on_axes(boxes, velocity, rotation):
    """LiDAR (N, 7) boxes onto the ego axes. Yaw is the heading vector rotated by `rotation`, not yaw plus its z-yaw:
    lidar2ego's mounting roll makes those differ."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 7).copy()
    velocity = np.nan_to_num(np.asarray(velocity, np.float64).reshape(-1, 2))
    if len(boxes) == 0:
        return boxes.astype(np.float32), velocity.astype(np.float32)
    rotation = np.asarray(rotation, np.float64)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    centers = np.concatenate([boxes[:, :3], np.ones((len(boxes), 1))], axis=1) @ transform.T
    boxes[:, :3] = centers[:, :3]
    heading = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))], axis=1) @ rotation.T
    boxes[:, 6] = np.arctan2(heading[:, 1], heading[:, 0])
    velocity = velocity @ rotation[:2, :2].T
    return boxes.astype(np.float32), velocity.astype(np.float32)


class NuScenesSource(Dataset):
    def __init__(self, info_path: str, path_root: str = "", data_root: str = "", version: str = "v1.0-trainval",
                 occ_root: Optional[str] = None, frame: str = "ego"):
        if frame not in ("ego", "lidar"):
            raise ValueError(f"nuScenes frame must be 'ego' or 'lidar', got {frame!r}")
        self.frame = frame
        with open(info_path, "rb") as f:
            self.infos = sorted(pickle.load(f)["infos"], key=lambda e: e["timestamp"])
        self.tokens = [info["token"] for info in self.infos]
        self.path_root = path_root
        scenes = defaultdict(list)
        for index, info in enumerate(self.infos):
            scenes[info["scene_token"]].append(index)
        self.scenes = list(scenes.values())
        self.position = {i: (scene, k) for scene in self.scenes for k, i in enumerate(scene)}
        self.occ_root, self.scene_names = occ_root, {}
        if occ_root:
            with open(os.path.join(data_root, version, "scene.json")) as f:
                self.scene_names = {s["token"]: s["name"] for s in json.load(f)}
        self.by_token = {token: i for i, token in enumerate(self.tokens)}

    def __len__(self):
        return len(self.infos)

    def sequences(self):
        return self.scenes

    def neighbour(self, index: int, step: int) -> Optional[dict]:
        scene, k = self.position[index]
        return self.infos[scene[k + step]] if 0 <= k + step < len(scene) else None

    @staticmethod
    def axes(info) -> np.ndarray:
        """The actual LiDAR -> ego rotation, kept for canonical metrics in either training frame."""
        out = np.eye(4)
        out[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        return out

    def working_axes(self, info, frame: Optional[str] = None) -> np.ndarray:
        """Map raw LiDAR coordinates into the selected head frame; native mode keeps raw axes."""
        frame = self.frame if frame is None else frame
        if frame == "ego":
            return self.axes(info)
        if frame == "lidar":
            return np.eye(4)
        raise ValueError(f"nuScenes frame must be 'ego' or 'lidar', got {frame!r}")

    @staticmethod
    def lidar2global(info) -> np.ndarray:
        return rigid(info["ego2global_rotation"], info["ego2global_translation"]) @ rigid(
            info["lidar2ego_rotation"], info["lidar2ego_translation"])

    def relative_pose(self, info, other, frame: Optional[str] = None) -> np.ndarray:
        """Pose of `other`'s LiDAR frame in the selected frame of `info`.

        Main's waypoints are LiDAR-frame xy of the LiDAR origin, z dropped. Ego mode rotates that plane with
        lidar2ego's xy block; rotating the 3D point first would mix LiDAR z through the mounting roll.
        """
        to_current = np.linalg.inv(self.lidar2global(info)) @ self.lidar2global(other)
        axes = self.working_axes(info, frame)
        other_axes = self.working_axes(other, frame)
        rotation = axes[:3, :3]
        xy = rotation[:2, :2] @ to_current[:2, 3]
        yaw = yaw_of(axes @ to_current @ np.linalg.inv(other_axes))
        return np.array([xy[0], xy[1], yaw], np.float32)

    def path(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(self.path_root, path))

    def features(self, index: int) -> Dict:
        info = self.infos[index]
        axes = self.working_axes(info)
        cameras = []
        for name, view in CAMERAS.items():
            cam = info["cams"][name]
            path = self.path(cam["data_path"])
            with Image.open(path) as image:
                size = image.size
            sensor2lidar = rigid(cam["sensor2lidar_rotation"], cam["sensor2lidar_translation"])
            cameras.append(camera(view, path, cam["cam_intrinsic"], axes @ sensor2lidar, size))
        status = np.asarray(info["ego_status"], np.float32)  # CAN bus: acc (3), rotation rate (3), velocity (3), steer
        return {
            "cameras": cameras,
            "history_trajectory": self.history(index),
            "ego_acceleration": status[0:3],
            "ego_angular_velocity": status[3:6],
            "ego_velocity": status[6:9],
            "ego_steer": status[9],
            "command": COMMANDS[int(np.argmax(info["gt_ego_fut_cmd"]))],
            "timestamp": info["timestamp"] / 1e6,
            "ego2global": self.lidar2global(info) @ np.linalg.inv(axes),
            "coordinate_frame": self.frame,
            "native_to_ego": self.axes(info) if self.frame == "lidar" else np.eye(4),
        }

    def pose_at(self, index: int, step: int, frame: Optional[str] = None):
        """Pose of the sample `step` keyframes away in the selected frame; None past the scene ends."""
        if step == 0:
            return np.zeros(3, np.float32)
        other = self.neighbour(index, step)
        return None if other is None else self.relative_pose(self.infos[index], other, frame)

    def history(self, index: int, frame: Optional[str] = None) -> np.ndarray:
        """(4, 3) selected-frame poses at t-1.5, -1.0, -0.5, 0 s.

        vad_nuscenes_converter stores four lidar-frame offsets (his_ts=4, five positions) and
        LoadAnnotations3D_E2E integrates them and subtracts the last, so the oldest position cancels and the result
        is the three previous poses plus now. A missing past sample repeats the next step's displacement (the next
        frame's, on the first sample of a scene), which is an affine combination and holds in this frame.
        """
        pos, delta, step = [None] * 5, [None] * 5, 0
        for i in range(4, -1, -1):
            local = self.pose_at(index, step, frame) if step is not None else None
            if local is not None:
                nxt = self.pose_at(index, step + 1, frame)
                pos[i] = local
                delta[i] = nxt - local if nxt is not None else np.zeros(3, np.float32)
                step = step - 1 if self.neighbour(index, step - 1) is not None else None
            else:
                pos[i] = pos[i + 1] - delta[i + 1]
                delta[i] = delta[i + 1]
        return np.stack(pos[1:]).astype(np.float32)

    def trajectory(self, index: int, frame: Optional[str] = None):
        info = self.infos[index]
        poses, mask = np.zeros((FUTURE, 3), np.float32), np.zeros(FUTURE, np.float32)
        last = np.zeros(3, np.float32)
        for step in range(1, FUTURE + 1):
            other = self.neighbour(index, step)
            if other is None:
                # the converter copies the last pose, so the stored offset is 0 where the mask is 0
                poses[step - 1] = last
                continue
            last = self.relative_pose(info, other, frame)
            poses[step - 1], mask[step - 1] = last, 1
        return poses, mask

    def targets(self, index: int) -> Dict:
        info = self.infos[index]
        axes = self.working_axes(info)
        lidar_boxes = np.asarray(info["gt_boxes"], np.float64)
        if lidar_boxes.size == 0:
            lidar_boxes = lidar_boxes.reshape(0, 7)
        keep = (np.asarray(info["num_lidar_pts"]) > 0) & (np.hypot(lidar_boxes[:, 0], lidar_boxes[:, 1]) <= LIDAR_BOX_RADIUS)
        lidar_boxes = lidar_boxes[keep]
        velocity = np.nan_to_num(np.asarray(info["gt_velocity"], np.float64)[keep])
        boxes, velocity = boxes_on_axes(lidar_boxes, velocity, axes[:3, :3])
        offsets = np.asarray(info["gt_agent_fut_trajs"], np.float64)[keep]
        # same z as the box, so the adapter's per-step delta is the lidar offset rotated in the plane
        # (a z of 0 would leave the mounting roll inside the first step)
        points = np.concatenate([
            lidar_boxes[:, None, :2] + np.cumsum(offsets, axis=1),
            np.broadcast_to(lidar_boxes[:, None, 2:3], offsets.shape[:2] + (1,)),
        ], axis=-1)
        future = (points.reshape(-1, 3) @ axes[:3, :3].T)[:, :2].astype(np.float32)
        trajectory, mask = self.trajectory(index)
        out = {
            "trajectory": trajectory,
            "trajectory_mask": mask,
            "boxes": boxes,
            "box_velocity": velocity,
            "box_names": [str(n) for n in np.asarray(info["gt_names"])[keep]],
            "box_ids": [str(i) for i in np.asarray(info["instance_inds"])[keep]],
            "agent_future": future.reshape(offsets.shape),
            "agent_future_mask": np.asarray(info["gt_agent_fut_masks"], np.float32)[keep],
            "map": [(MAP_TYPES[int(label)], transform_xy(map_line(line), axes))
                    for label, lines in info["map_annos"].items() for line in lines],
        }
        if self.occ_root:
            # LoadOccWorldLabels leaves gt_occ_dense unset (None) when the npz is absent; don't point at a missing file
            path = os.path.join(self.occ_root, self.scene_names[info["scene_token"]], info["token"], "labels.npz")
            if os.path.isfile(path):
                out["occ_path"] = path
                out["occ_format"] = "occ3d"
        return out

    def __getitem__(self, index: int):
        return self.features(index), self.targets(index), self.tokens[index]

    def eval_info(self, token: str) -> Dict:
        """Future ego poses and both training-frame boxes and raw main-metric LiDAR boxes."""
        index = self.by_token[token]
        info = self.infos[index]
        # Main's online metrics are expressed on ego axes even if the head trains on raw LiDAR axes.
        trajectory, mask = self.trajectory(index, frame="ego")
        boxes, strict_boxes = [], []
        for step in range(1, FUTURE + 1):
            other = self.neighbour(index, step)
            if other is None:
                boxes.append(np.zeros((0, 5), np.float32))
                strict_boxes.append(np.zeros((0, 7), np.float32))
                continue
            keep = np.asarray(other["num_lidar_pts"]) > 0
            raw = np.asarray(other["gt_boxes"], np.float64)[keep]
            if len(raw) == 0:
                boxes.append(np.zeros((0, 5), np.float32))
                strict_boxes.append(np.zeros((0, 7), np.float32))
                continue
            # Active main planning_eval.py passes get_ann_info()['fut_boxes'] to its strict metric:
            # raw LiDAR [x,y,z,w,l,h,yaw], with no canonical-axis rotation or box remap.
            into = np.linalg.inv(self.lidar2global(info)) @ self.lidar2global(other)
            center = raw[:, :3] @ into[:3, :3].T + into[:3, 3]
            yaw_xy = np.stack([np.cos(raw[:, 6]), np.sin(raw[:, 6])], axis=1) @ into[:2, :2].T
            lidar = np.column_stack([center, raw[:, 3:6], np.arctan2(yaw_xy[:, 1], yaw_xy[:, 0])])
            strict_boxes.append(lidar.astype(np.float32))
            moved, _ = boxes_on_axes(lidar, np.zeros((len(lidar), 2)), self.axes(info)[:3, :3])
            boxes.append(moved[:, [0, 1, 3, 4, 6]])
        return {"trajectory": trajectory, "mask": mask, "boxes": boxes, "strict_boxes": strict_boxes,
                "axes": self.axes(info)[:2, :2].astype(np.float32)}


@register("nuscenes")
def nuscenes_loader(cfg, agent, split: str):
    source = cfg.nuscenes
    name = {"train": "train", "val": "val", "test": "val"}[split]
    profile = getattr(agent, "profile", None)
    frame = getattr(profile, "frame", None) or OmegaConf.select(cfg, "profile.frame") or "ego"
    return NuScenesSource(os.path.join(source.info_dir, f"nuscenes_infos_{name}.pkl"),
                          path_root=OmegaConf.select(source, "path_root") or "",
                          data_root=OmegaConf.select(source, "data_root") or "",
                          version=OmegaConf.select(source, "version") or "v1.0-trainval",
                          occ_root=OmegaConf.select(source, "occ_root"), frame=frame)
