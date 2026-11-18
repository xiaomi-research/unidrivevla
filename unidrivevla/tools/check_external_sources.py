"""Small CPU checks for the SDK-free PAI/LEAD mappings and cache boundary."""

import math
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unidrivevla.adapters.unidrivevla import Profile, prepare_sample  # noqa: E402
from unidrivevla.data.external import ExternalCacheDataset, ExternalCacheWriter  # noqa: E402
from unidrivevla.data.lead import (HORIZON, LEAD_CAMERA_VIEWS, TARGET_INTERVAL_TICKS,
                                   TARGET_INTERVAL_US, exact_label_origin, lead_inference_features,
                                   lead_sample)  # noqa: E402
from unidrivevla.data.physicalai import CAMERA_FEATURES, ftheta_remap, physicalai_sample, physicalai_loader, rectify_ftheta  # noqa: E402


class FThetaFixture:
    """Small nonlinear camera model following the official FTheta ray2pixel API."""

    width, height = 32, 24
    principal_point = np.array([15.5, 11.5])

    def ray2pixel(self, ray):
        ray = np.asarray(ray, dtype=np.float64)
        theta = np.arccos(np.clip(ray[..., 2:] / np.linalg.norm(ray, axis=-1, keepdims=True), -1, 1))
        norm_xy = np.linalg.norm(ray[..., :2], axis=-1, keepdims=True)
        radius = 12.0 * theta + 0.8 * theta ** 3  # nonlinear forward polynomial
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.principal_point + radius * ray[..., :2] / norm_xy


class FakePose:
    def __init__(self, xyz, yaw=0.0):
        self.array = np.array([xyz[0], xyz[1], xyz[2], math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
                              dtype=np.float64)
        c, s = math.cos(yaw), math.sin(yaw)
        self.rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


class MotionFixture:
    def __call__(self, timestamps_us):
        times = np.asarray(timestamps_us, dtype=np.float64).reshape(-1) / 1e6
        yaw = np.zeros_like(times)
        rotations = np.zeros((len(times), 3, 3), dtype=np.float64)
        rotations[:, 0, 0] = rotations[:, 1, 1] = np.cos(yaw)
        rotations[:, 1, 0], rotations[:, 0, 1] = np.sin(yaw), -np.sin(yaw)
        rotations[:, 2, 2] = 1.0
        return SimpleNamespace(pose=SimpleNamespace(translation=np.column_stack((10 + times, times * 0, times * 0)),
                                                     rotation=SimpleNamespace(as_matrix=lambda: rotations)))


class LeadCameraMetadata:
    def __init__(self, fx, fy, cx, cy, pose):
        self.k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.camera_to_imu_se3 = pose

    def project_to_image(self, points):
        points = np.asarray(points, dtype=np.float64)
        projected = points @ self.k.T
        pixels = projected[:, :2] / projected[:, 2:3]
        visible = (points[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < 32) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 24)
        return pixels, visible, points[:, 2]


def check_pai_mapping():
    model = FThetaFixture()
    image = np.zeros((model.height, model.width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(model.width, dtype=np.uint8)[None, :]
    rectified, intrinsic, valid = rectify_ftheta(image, model, 45.0, (16, 12))
    assert intrinsic.shape == (3, 3) and rectified.shape == (12, 16, 3)
    assert valid.dtype == np.bool_ and valid.mean() > 0.98
    # FTheta's nonlinear forward projection must differ from the linear pinhole pixel map at an off-axis ray.
    _, map_x, _, _ = ftheta_remap(model, (16, 12), 45.0)
    assert not np.isclose(map_x[2, -1], 31.0)
    _, center_x, center_y, center_valid = ftheta_remap(model, (15, 11), 45.0)
    assert center_valid[5, 7] and np.allclose([center_x[5, 7], center_y[5, 7]], model.principal_point)

    camera_rows = []
    views = ("front", "front_tele", "front_left", "front_right", "back_left", "back_right", "back")
    assert [row[2] for row in CAMERA_FEATURES] == list(views)
    assert next(row[3] for row in CAMERA_FEATURES if row[2] == "front_tele") == 18.0
    for i, view in enumerate(views):
        camera_rows.append((view, rectified.copy(), intrinsic.copy(), np.eye(4), valid.copy(), 2_000_000 + i))
    features, targets, assets = physicalai_sample("clip-a", 2_000_000, MotionFixture(), camera_rows)
    assert features["command"] == "unknown"
    assert features["history_trajectory"].shape == (4, 3)
    assert np.allclose(targets["trajectory"][:, 0], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], atol=1e-4)
    assert targets["task_valid"] == {"planning": True, "det": False, "map": False, "motion": False,
                                     "occupancy": False, "geometry": False}

    def nonfinite_motion(timestamps):
        state = MotionFixture()(timestamps)
        state.pose.translation[0, 0] = np.inf
        return state

    _must_fail(lambda: physicalai_sample("clip-a", 2_000_000, nonfinite_motion, camera_rows),
               "egomotion contains non-finite pose values")
    return features, targets, assets


def check_lead_mapping():
    cameras = []
    for index in range(6):
        image = np.full((24, 32, 3), index * 10, dtype=np.uint8)
        cameras.append(SimpleNamespace(image=image,
            metadata=LeadCameraMetadata(20, 20, 15.5, 11.5, FakePose([index * 0.1, 0, 1])),
            timestamp=SimpleNamespace(time_us=5_000_000)))
    anchor = np.eye(4)
    yaw = math.pi / 2
    anchor[:2, :2] = [[0, -1], [1, 0]]
    anchor[:2, 3] = [10, 20]
    past_positions = np.zeros((31, 2), dtype=np.float64)
    past_positions[:, 0] = -np.arange(31) * 0.05
    exact_anchor = FakePose([10.25, 20.5, 0], yaw)
    scene = SimpleNamespace(cameras=cameras, past_ego_positions=past_positions,
                             past_ego_yaws=np.zeros(31), ego_state=SimpleNamespace(imu_se3=exact_anchor),
                             driving_meta={"localized_ego_state_se3": anchor})
    future = {step * TARGET_INTERVAL_TICKS:
              SimpleNamespace(imu_se3=FakePose([10.25, 20.5 + step * 0.25, 0], yaw))
              for step in range(1, HORIZON + 1)}
    features, targets, assets = lead_sample(scene, future, "log-1", label_origin_pose=exact_label_origin(scene))
    assert [cam["view"] for cam in features["cameras"]] == ["front_left", "front", "front_right",
                                                             "back_right", "back", "back_left"]
    assert np.allclose(features["history_trajectory"][:, 0], [-1.5, -1.0, -0.5, 0.0])
    assert np.allclose(targets["trajectory"][:, 0], np.arange(1, HORIZON + 1) * 0.25)
    assert np.allclose(targets["trajectory"][:, 1], 0)
    assert np.array_equal(features["ego2global"], anchor)  # model input keeps observed localization, not exact labels
    assert not np.array_equal(features["ego2global"][:3, 3], exact_anchor.array[:3])
    assert features["stream_id"] == "lead:log-1"
    assert len(assets) == 6
    online_scene = SimpleNamespace(cameras=cameras, past_ego_positions=past_positions,
                                   past_ego_yaws=np.zeros(31), ego_state=None,
                                   driving_meta=scene.driving_meta)
    online = lead_inference_features(online_scene, ego2global=anchor, timestamp_seconds=5.0)
    assert list(online["cameras"]) == list(LEAD_CAMERA_VIEWS)
    assert "trajectory" not in online and "targets" not in online
    assert np.array_equal(online["history_trajectory"], features["history_trajectory"])
    assert np.allclose(online["ego_velocity"], features["ego_velocity"], equal_nan=True)
    for index, (view, camera_feature) in enumerate(zip(LEAD_CAMERA_VIEWS, features["cameras"])):
        assert np.array_equal(online["cameras"][view]["image"], cameras[index].image)
        assert np.allclose(online["cameras"][view]["intrinsic"], camera_feature["intrinsic"])
        assert np.allclose(online["cameras"][view]["sensor2ego"], camera_feature["sensor2ego"])

    # A filtered current pose shifts history into its frame without consulting the exact label pose.
    filtered = anchor.copy()
    filtered[:2, 3] += [0.2, -0.1]
    filtered_yaw = yaw + 0.05
    filtered[:2, :2] = [[math.cos(filtered_yaw), -math.sin(filtered_yaw)],
                        [math.sin(filtered_yaw), math.cos(filtered_yaw)]]
    filtered_online = lead_inference_features(online_scene, ego2global=filtered, timestamp_seconds=5.0)
    expected_last_xy = filtered[:2, :2].T @ (anchor[:2, 3] - filtered[:2, 3])
    assert np.allclose(filtered_online["history_trajectory"][-1, :2], expected_last_xy)
    expected_last_yaw = (yaw - filtered_yaw + np.pi) % (2 * np.pi) - np.pi
    assert np.isclose(filtered_online["history_trajectory"][-1, 2], expected_last_yaw)

    # Only current localized state, past localization and exact future poses enter the mapping.
    changed = dict(scene.driving_meta)
    changed["steer"] = 1e6  # the future expert action is not an input
    scene.driving_meta = changed
    again, _, _ = lead_sample(scene, future, "log-1", label_origin_pose=exact_label_origin(scene))
    assert np.array_equal(features["history_trajectory"], again["history_trajectory"])

    missing = dict(future)
    missing[TARGET_INTERVAL_TICKS] = None
    _, masked, _ = lead_sample(scene, missing, "log-1", label_origin_pose=exact_label_origin(scene))
    assert masked["trajectory_mask"][0] == 0 and np.array_equal(masked["trajectory"][0], np.zeros(3, np.float32))

    stopped = {step * TARGET_INTERVAL_TICKS: SimpleNamespace(imu_se3=FakePose([10.25, 20.5, 0], yaw))
               for step in range(1, HORIZON + 1)}
    _, stopped_targets, _ = lead_sample(scene, stopped, "log-1", label_origin_pose=exact_label_origin(scene))
    assert np.array_equal(stopped_targets["trajectory"], np.zeros((HORIZON, 3), np.float32))

    invalid_future = dict(future)
    invalid_future[TARGET_INTERVAL_TICKS] = SimpleNamespace(imu_se3=FakePose([float("nan"), 20.5, 0], yaw))
    _must_fail(lambda: lead_sample(scene, invalid_future, "log-1", label_origin_pose=exact_label_origin(scene)),
               "invalid py123d PoseSE3")
    old_api = SimpleNamespace()
    _must_fail(lambda: exact_label_origin(old_api), "scene_data.ego_state.imu_se3 is required")
    invalid_current = SimpleNamespace(ego_state=SimpleNamespace(imu_se3=FakePose([10, float("inf"), 0], yaw)))
    _must_fail(lambda: lead_sample(scene, future, "log-1", label_origin_pose=exact_label_origin(invalid_current)),
               "invalid py123d PoseSE3")
    bad_localizer = dict(scene.driving_meta)
    bad_localizer["localized_ego_state_se3"] = anchor.copy()
    bad_localizer["localized_ego_state_se3"][0, 3] = np.nan
    scene.driving_meta = bad_localizer
    _must_fail(lambda: lead_sample(scene, future, "log-1", label_origin_pose=exact_label_origin(scene)),
               "finite rigid 4x4 pose")
    return features, targets, assets


def _must_fail(fn, fragment):
    try:
        fn()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        assert fragment in str(exc), (fragment, str(exc))
    else:
        raise AssertionError(f"expected failure containing {fragment!r}")


def check_cache_and_adapter(source, rows):
    features, targets, assets = rows
    interval_us = TARGET_INTERVAL_US if source == "lead" else 500_000
    horizon = len(targets["trajectory"])
    with tempfile.TemporaryDirectory() as temp:
        features2, targets2 = copy.deepcopy(features), copy.deepcopy(targets)
        features2["timestamp"] += interval_us / 1e6
        assets2 = {}
        digest = "second"
        for cam in features2["cameras"]:
            for key in ("path", "valid_mask_path"):
                if key in cam:
                    old = cam[key]
                    new = old.replace(old.split("/")[-2], digest)
                    assets2[new[7:]] = assets[old[7:]]
                    cam[key] = new
        with ExternalCacheWriter(temp, source, "train", sample_interval_us=interval_us,
                                 target_horizon=horizon) as writer:
            writer.write_sample("one", features, targets, assets)
            writer.write_sample("two", features2, targets2, assets2)
        dataset = ExternalCacheDataset(temp, source, "train")
        dataset.validate_sampling(interval_us, horizon)
        loaded_features, loaded_targets, token = dataset[0]
        assert token == "one"
        assert loaded_targets["task_valid"]["planning"] is True
        eval_info = dataset.eval_info(token)
        assert np.array_equal(eval_info["trajectory"], loaded_targets["trajectory"])
        assert np.array_equal(eval_info["mask"], loaded_targets["trajectory_mask"])
        assert eval_info["sample_interval_us"] == interval_us
        assert dataset.sequences() == [[0, 1]]
        profile = Profile(name=source, cameras=[cam["view"] for cam in features["cameras"]],
                          view_tokens=[f"<VIEW_{i}>" for i in range(len(features["cameras"]))],
                          horizon=horizon, pose_interval=interval_us / 1e6, commands=["unknown"],
                          command_prompts=["UNKNOWN"], status_layout=["vel_x", "vel_y"],
                          det_classes=["placeholder"], box_range=[-60, -60, 60, 60],
                          map_classes=["divider"], map_roi=[60, 30], ego_box=[1, 0, 0, 4, 2, 1, 0],
                          anchors_dir="", delta_mean=[0, 0], delta_std=[1, 1], image_hw=[12, 16],
                          prompt_style="ego", use_occ_tokens=False)
        model_features, model_targets = prepare_sample(loaded_features, loaded_targets, profile)
        assert tuple(model_features["img"].shape) == (len(features["cameras"]), 3, 12, 16)
        assert model_targets["gt_ego_fut_masks"].sum().item() == float(np.asarray(targets["trajectory_mask"]).sum())
        assert model_targets["gt_task_valid"]["planning"] is True


def main():
    pai = check_pai_mapping()
    lead = check_lead_mapping()
    check_cache_and_adapter("physicalai", pai)
    check_cache_and_adapter("lead", lead)
    # v1 lead caches encode six 0.5 s targets; the new profile must reject, not reinterpret, them.
    with tempfile.TemporaryDirectory() as temp:
        features, targets, assets = copy.deepcopy(lead)
        old_targets = copy.deepcopy(targets)
        old_targets["trajectory"] = old_targets["trajectory"][:6]
        old_targets["trajectory_mask"] = old_targets["trajectory_mask"][:6]
        with ExternalCacheWriter(temp, "lead", "train", sample_interval_us=500_000,
                                 target_horizon=6) as writer:
            writer.write_sample("old-lead", features, old_targets, assets)
        metadata_path = Path(temp) / "splits" / "train" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = {key: value for key, value in metadata.items()
                    if key not in ("sample_interval_us", "target_horizon")}
        metadata["schema"] = "unidrivevla.external.v1"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        old_dataset = ExternalCacheDataset(temp, "lead", "train")
        assert (old_dataset.sample_interval_us, old_dataset.target_horizon) == (500_000, 6)
        _must_fail(lambda: old_dataset.validate_sampling(TARGET_INTERVAL_US, HORIZON),
                   "reconvert this split instead of reinterpreting its labels")
    # A valid six-camera legacy cache must fail clearly under the seven-camera PAI profile.
    with tempfile.TemporaryDirectory() as temp:
        features, targets, assets = copy.deepcopy(pai)
        features["cameras"] = [row for row in features["cameras"] if row["view"] != "front_tele"]
        assets = {name: payload for name, payload in assets.items() if "/front_tele" not in name}
        with ExternalCacheWriter(temp, "physicalai", "train") as writer:
            writer.write_sample("legacy-six-camera", features, targets, assets)
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"physicalai": {"cache_root": temp}})
        _must_fail(lambda: physicalai_loader(cfg, None, "train"), "reconvert it with the current seven-camera profile")
        legacy_profile = SimpleNamespace(cameras=[row[2] for row in CAMERA_FEATURES if row[2] != "front_tele"])
        legacy_dataset = physicalai_loader(cfg, SimpleNamespace(profile=legacy_profile), "train")
        assert len(legacy_dataset) == 1  # saved six-camera checkpoint config remains usable
    # Mask-zero rows may be absent, while a masked-in non-finite target cannot enter the cache.
    with tempfile.TemporaryDirectory() as temp:
        features, targets, assets = copy.deepcopy(lead)
        targets["trajectory"][0, 0] = np.nan
        targets["trajectory_mask"][0] = 0
        with ExternalCacheWriter(temp, "lead", "train", sample_interval_us=TARGET_INTERVAL_US,
                                 target_horizon=HORIZON) as writer:
            writer.write_sample("missing-first-future", features, targets, assets)
    with tempfile.TemporaryDirectory() as temp:
        features, targets, assets = copy.deepcopy(lead)
        targets["trajectory"][0, 0] = np.nan
        with ExternalCacheWriter(temp, "lead", "train", sample_interval_us=TARGET_INTERVAL_US,
                                 target_horizon=HORIZON) as writer:
            _must_fail(lambda: writer.write_sample("nan-supervised-future", features, targets, assets),
                       "supervised trajectory coordinates must be finite")
    with tempfile.TemporaryDirectory() as temp:
        features, targets, assets = copy.deepcopy(pai)
        features["cameras"][0]["sensor2ego"][0, 3] = np.inf
        with ExternalCacheWriter(temp, "physicalai", "train") as writer:
            _must_fail(lambda: writer.write_sample("bad-calibration", features, targets, assets),
                       "camera sensor2ego must be finite")
    _must_fail(lambda: ftheta_remap(FThetaFixtureWithInvalidCalibration(), (16, 12), 45.0),
               "FTheta calibration")
    assert "physical_ai_av" not in sys.modules and "lead" not in sys.modules
    print("external PAI 7-camera/finite-calibration and LEAD exact-anchor/masked-target cache checks passed")


class FThetaFixtureWithInvalidCalibration(FThetaFixture):
    angle_to_pixeldist_poly = np.array([1.0, np.nan])


if __name__ == "__main__":
    main()
