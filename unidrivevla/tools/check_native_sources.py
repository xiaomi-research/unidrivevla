"""CPU fixture oracle for native LiDAR data and the legacy canonical source mode."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pyquaternion import Quaternion

from unidrivevla.adapters.unidrivevla import Profile
from unidrivevla.data.bench2drive import Bench2DriveSource, remap_boxes
from unidrivevla.data.nuscenes import NuScenesSource, boxes_on_axes, rigid
from unidrivevla.tools.fit_anchors import _native_map_centres, _native_motion, map_segments
from unidrivevla.data.sample import transform_boxes, transform_xy


def close(a, b, label, atol=2e-5):
    if not np.allclose(a, b, atol=atol, rtol=0):
        raise AssertionError(f"{label} differs by {np.max(np.abs(np.asarray(a) - np.asarray(b)))}")


def rotation(angle):
    return Quaternion(axis=[0, 0, 1], angle=angle).elements


def make_nuscenes(tmp: Path):
    Image.new("RGB", (8, 6), (20, 30, 40)).save(tmp / "image.png")
    infos = []
    camera_names = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    for i in range(2):
        cams = {}
        for camera_i, name in enumerate(camera_names):
            cam_rotation = rotation(0.03 * (camera_i + 1))
            cams[name] = {"data_path": "image.png", "cam_intrinsic": np.eye(3),
                          "sensor2lidar_rotation": cam_rotation,
                          "sensor2lidar_translation": np.array([0.1, -0.2, 1.2])}
        infos.append({
            "token": f"nusc-{i}", "scene_token": "scene-a", "timestamp": 1_000_000 * (i + 1),
            "lidar2ego_rotation": rotation(0.21), "lidar2ego_translation": np.array([1.0, 0.0, 1.8]),
            "ego2global_rotation": rotation(0.4 + 0.03 * i),
            "ego2global_translation": np.array([3.0 + 1.2 * i, -2.0 + 0.1 * i, 0.0]),
            "cams": cams, "ego_status": np.arange(10, dtype=np.float32),
            "gt_ego_fut_cmd": np.array([0.0, 0.0, 1.0]),
            "gt_boxes": np.array([[3.0, -1.0, -0.4, 4.2, 1.8, 1.5, 0.35]], np.float32),
            "gt_velocity": np.array([[1.0, -0.5]], np.float32), "num_lidar_pts": np.array([4]),
            "gt_names": np.array(["car"]), "instance_inds": np.array([17]),
            "gt_agent_fut_trajs": np.tile(np.array([[[0.2, -0.1]]], np.float32), (1, 12, 1)),
            "gt_agent_fut_masks": np.ones((1, 12), np.float32),
            "map_annos": {0: [np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 3.5]], np.float32)]},
        })
    with open(tmp / "nusc.pkl", "wb") as f:
        pickle.dump({"infos": infos}, f)
    return (NuScenesSource(str(tmp / "nusc.pkl"), path_root=str(tmp), frame="lidar"),
            NuScenesSource(str(tmp / "nusc.pkl"), path_root=str(tmp), frame="ego"))


def make_b2d(tmp: Path):
    Image.new("RGB", (8, 6), (20, 30, 40)).save(tmp / "image.png")
    infos = []
    cam_names = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    lidar2ego = rigid(rotation(0.17), [0.4, -0.2, 1.2])
    for i in range(40):
        lidar2world = rigid(rotation(0.0), [0.4 * i, 0.1 * i, 0.0])
        world2lidar = np.linalg.inv(lidar2world)
        sensors = {"LIDAR_TOP": {"lidar2ego": lidar2ego, "world2lidar": world2lidar}}
        for camera_i, name in enumerate(cam_names):
            sensors[name] = {"data_path": "image.png", "intrinsic": np.eye(3),
                             "cam2ego": rigid(rotation(0.05 * camera_i), [0.2, 0.0, 1.1])}
        actor_world = rigid(rotation(0.0), [0.4 * i + 4.0, 0.1 * i + 1.0, 0.0])
        infos.append({
            "folder": "Town01/clip-a", "frame_idx": i, "town_name": "Town01", "sensors": sensors,
            "ego_vel": np.array([8.0, 0.0, 0.0]), "ego_accel": np.array([0.1, -0.2, 0.0]),
            "ego_rotation_rate": np.array([0.0, 0.0, 0.01]), "steer": np.float32(0.0),
            "command_near": 0, "gt_boxes": np.array([[4.0, 1.0, 0.5, 1.8, 4.0, 1.5, 0.2, 0.5, 0.0]], np.float32),
            "gt_ids": np.array([4]), "gt_names": np.array(["car"]), "num_points": np.array([4]),
            "npc2world": np.array([actor_world]),
        })
    map_infos = {"Town01": {
        "lane_sample_points": [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])],
        "lane_types": ["Center"],
        "lane_points": [np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [4.0, 1.0, 0.0]])],
        "lane_topos": [[]], "lane_ids": [0],
        "trigger_volumes_points": [], "trigger_volumes_types": [],
    }}
    with open(tmp / "b2d.pkl", "wb") as f:
        pickle.dump(infos, f)
    with open(tmp / "maps.pkl", "wb") as f:
        pickle.dump(map_infos, f)
    args = (str(tmp / "b2d.pkl"), str(tmp / "maps.pkl"), str(tmp), ["Broken", "Solid", "SolidSolid", "Center"])
    return (Bench2DriveSource(*args, frame="lidar"), Bench2DriveSource(*args, frame="ego"))


def main():
    profile_dir = Path(__file__).resolve().parents[1] / "config" / "profile"
    native_ns = OmegaConf.load(profile_dir / "nuscenes.yaml")
    native_b2d = OmegaConf.load(profile_dir / "b2d.yaml")
    ego_ns = OmegaConf.load(profile_dir / "nuscenes_ego.yaml")
    ego_b2d = OmegaConf.load(profile_dir / "b2d_ego.yaml")
    assert native_ns.frame == native_b2d.frame == "lidar"
    assert ego_ns.frame == ego_b2d.frame == "ego"
    assert native_ns.prefix_max_len == 3740 and native_b2d.prefix_max_len == 3670
    assert native_ns.official_attention_mask and native_b2d.official_attention_mask
    assert ego_ns.prefix_max_len == 0 and ego_b2d.prefix_max_len == 0
    assert not ego_ns.official_attention_mask and not ego_b2d.official_attention_mask
    assert native_ns.delta_mean == [0.0233, 2.2707] and native_ns.delta_std == [0.3427, 1.8668]
    assert native_b2d.delta_mean == [-0.0222, 2.0249] and native_b2d.delta_std == [0.672, 1.8586]
    assert native_ns.map_roi == native_b2d.map_roi == [30.0, 60.0]
    assert native_ns.ego_box == [0.0, 0.5, -1.06, 4.08, 1.73, 1.56, 0.0]
    assert native_b2d.ego_box == [0.0, 0.5, -1.095, 4.89, 1.84, 1.49, 0.0]

    with tempfile.TemporaryDirectory(prefix="unidrive-native-check-") as directory:
        tmp = Path(directory)
        nusc_native, nusc_ego = make_nuscenes(tmp)
        feat_n, target_n, _ = nusc_native[0]
        feat_e, target_e, _ = nusc_ego[0]
        axes_n = nusc_native.axes(nusc_native.infos[0])
        assert feat_n["coordinate_frame"] == "lidar" and feat_e["coordinate_frame"] == "ego"
        close(feat_n["native_to_ego"], axes_n, "nuScenes native-to-ego rotation metadata")
        close(feat_e["native_to_ego"], np.eye(4), "nuScenes ego-frame metadata")
        close(feat_n["history_trajectory"], nusc_native.history(0, frame="lidar"), "nuScenes native history")
        close(feat_e["history_trajectory"][:, :2], transform_xy(feat_n["history_trajectory"][:, :2], axes_n),
              "nuScenes legacy history xy")
        close(feat_e["history_trajectory"][:, 2], feat_n["history_trajectory"][:, 2], "nuScenes legacy history yaw")
        close(feat_n["ego_velocity"], nusc_native.infos[0]["ego_status"][6:9], "nuScenes main ego status")
        close(feat_n["cameras"][0]["sensor2ego"], rigid(
            nusc_native.infos[0]["cams"]["CAM_FRONT"]["sensor2lidar_rotation"],
            nusc_native.infos[0]["cams"]["CAM_FRONT"]["sensor2lidar_translation"]), "nuScenes raw camera calibration")
        close(feat_e["cameras"][0]["sensor2ego"], axes_n @ feat_n["cameras"][0]["sensor2ego"],
              "nuScenes legacy camera calibration")
        close(target_n["boxes"], nusc_native.infos[0]["gt_boxes"], "nuScenes raw main boxes")
        close(target_n["box_velocity"], nusc_native.infos[0]["gt_velocity"], "nuScenes raw main velocity")
        close(target_n["trajectory"], nusc_native.trajectory(0, frame="lidar")[0], "nuScenes native future")
        expected_future = nusc_native.infos[0]["gt_boxes"][:, None, :2] + np.cumsum(
            nusc_native.infos[0]["gt_agent_fut_trajs"], axis=1)
        close(target_n["agent_future"], expected_future, "nuScenes raw main motion labels")
        expected_boxes, expected_velocity = boxes_on_axes(nusc_native.infos[0]["gt_boxes"],
                                                           nusc_native.infos[0]["gt_velocity"], axes_n[:3, :3])
        close(target_e["boxes"], expected_boxes, "nuScenes legacy boxes")
        close(target_e["box_velocity"], expected_velocity, "nuScenes legacy velocity")
        close(target_e["map"][0][1], transform_xy(target_n["map"][0][1], axes_n), "nuScenes legacy map")
        close(_native_map_centres(nusc_native, 0, False)[0],
              np.asarray(nusc_native.infos[0]["map_annos"][0][0]).mean(axis=0), "nuScenes raw map anchor centre")
        motion_n = _native_motion(nusc_native, Profile.from_config(OmegaConf.load(profile_dir / "nuscenes.yaml")), False)
        c, s = np.cos(0.35), np.sin(0.35)
        expected_motion = np.stack([c * np.arange(1, 13) * 0.2 + s * np.arange(1, 13) * -0.1,
                                    -s * np.arange(1, 13) * 0.2 + c * np.arange(1, 13) * -0.1], axis=1).reshape(-1)
        close(motion_n[0][0], expected_motion, "nuScenes native motion-anchor track")
        eval_n, eval_e = nusc_native.eval_info("nusc-0"), nusc_ego.eval_info("nusc-0")
        close(eval_n["trajectory"], eval_e["trajectory"], "nuScenes canonical metric trajectory")
        close(eval_n["axes"], axes_n[:2, :2], "nuScenes metric axes")

        b2d_native, b2d_ego = make_b2d(tmp)
        feat_n, target_n, token = b2d_native[10]
        feat_e, target_e, _ = b2d_ego[10]
        info = b2d_native.infos[10]
        axes_b = b2d_native.axes(info)
        raw_boxes, raw_velocity = remap_boxes(np.asarray(info["gt_boxes"]))
        assert feat_n["coordinate_frame"] == "lidar" and feat_e["coordinate_frame"] == "ego"
        close(feat_n["native_to_ego"], axes_b, "B2D native-to-ego rotation metadata")
        close(feat_e["native_to_ego"], np.eye(4), "B2D ego-frame metadata")
        close(feat_n["history_trajectory"], b2d_native.features(10)["history_trajectory"], "B2D native history")
        close(feat_e["history_trajectory"][:, :2], transform_xy(feat_n["history_trajectory"][:, :2], axes_b),
              "B2D legacy history xy")
        close(feat_e["history_trajectory"][:, 2], feat_n["history_trajectory"][:, 2], "B2D legacy history yaw")
        close(feat_n["ego2global"], np.linalg.inv(np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"])),
              "B2D native pose")
        close(feat_n["ego_velocity"], info["ego_vel"], "B2D main ego status")
        expected_camera = np.linalg.inv(np.asarray(info["sensors"]["LIDAR_TOP"]["lidar2ego"])) @ \
            np.asarray(info["sensors"]["CAM_FRONT"]["cam2ego"])
        close(feat_n["cameras"][0]["sensor2ego"], expected_camera, "B2D raw camera calibration")
        close(feat_e["cameras"][0]["sensor2ego"], axes_b @ expected_camera, "B2D legacy camera calibration")
        close(target_n["boxes"], raw_boxes, "B2D raw main boxes")
        close(target_n["box_velocity"], raw_velocity, "B2D raw main velocity")
        close(target_n["trajectory"], b2d_native.trajectory(10, frame="lidar")[0], "B2D native future")
        expected_boxes, expected_velocity = transform_boxes(raw_boxes, raw_velocity, axes_b)
        close(target_e["boxes"], expected_boxes, "B2D legacy boxes")
        close(target_e["box_velocity"], expected_velocity, "B2D legacy velocity")
        close(target_e["trajectory"][:, :2], transform_xy(target_n["trajectory"][:, :2], axes_b),
              "B2D legacy trajectory xy")
        close(target_e["trajectory"][:, 2], target_n["trajectory"][:, 2], "B2D legacy trajectory yaw")
        map_centres_b = _native_map_centres(b2d_native, 10, True)
        raw_lane = np.asarray(b2d_native.maps["Town01"]["lane_points"][0], np.float64)
        world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], np.float64)
        raw_lane = (np.concatenate([raw_lane, np.ones((len(raw_lane), 1))], axis=1) @ world2lidar.T)[:, :2]
        inside = ((raw_lane[:, 0] > -15) & (raw_lane[:, 0] < 15)
                  & (raw_lane[:, 1] > -30) & (raw_lane[:, 1] < 30))
        close(map_centres_b[0], raw_lane[inside].mean(axis=0), "B2D raw map anchor centre")
        motion_b = _native_motion(b2d_native, Profile.from_config(OmegaConf.load(profile_dir / "b2d.yaml")), True)
        # Main kmeans_motion samples raw indices at 5-frame intervals (0.5 s): the fixture actor moves
        # (2.0, 0.5) m in the current LiDAR frame each step, then lidar2agent applies the raw 0.2 rad box yaw.
        step = np.arange(1, 7, dtype=np.float64)
        c, s = np.cos(0.2), np.sin(0.2)
        expected_motion_b = np.stack([step * (2.0 * c + 0.5 * s),
                                      step * (-2.0 * s + 0.5 * c)], axis=1).reshape(-1)
        close(motion_b[0][0], expected_motion_b, "B2D native motion-anchor track")
        assert token.endswith("00010")
        eval_n, eval_e = b2d_native.eval_info(token), b2d_ego.eval_info(token)
        close(eval_n["trajectory"], eval_e["trajectory"], "B2D canonical metric trajectory")
        close(eval_n["axes"], axes_b[:2, :2], "B2D metric axes")
        close(eval_n["boxes"][0], eval_e["boxes"][0], "B2D canonical metric actors")

    centre = np.array([[10.0, 20.0]])
    nusc_map_anchor = map_segments(centre, 20, b2d=False, native=True)
    close(nusc_map_anchor[0, :, 0], np.full(20, 10.0), "native nuScenes map anchor direction")
    close(nusc_map_anchor[0, :, 1], 20 + np.linspace(-4, 4, 20), "native nuScenes map anchor span")
    print("PASS native LiDAR profiles, source labels/calibration, canonical metric metadata, and native anchor direction")


if __name__ == "__main__":
    main()
