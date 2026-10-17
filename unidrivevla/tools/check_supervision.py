"""CPU regression for unknown versus labelled-empty task targets."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.data.registry import collate  # noqa: E402
from unidrivevla.data.sample import check_sample  # noqa: E402
from unidrivevla.data.navsim import SampleTargetBuilder, WithTokens  # noqa: E402
from unidrivevla.adapters.unidrivevla import Profile, prepare_sample  # noqa: E402
from unidrivevla.models.unidrivevla.dense_heads.unified_perception_decoder import UnifiedPerceptionDecoder  # noqa: E402


class EmptySceneSampler:
    def sample(self, cls, pred, labels, boxes):
        assert len(labels) == cls.shape[0] == len(boxes)
        return torch.zeros_like(cls), pred.new_zeros((*pred.shape[:2], 2)), pred.new_zeros((*pred.shape[:2], 2))


class ClassificationLoss(nn.Module):
    def forward(self, pred, target, avg_factor):
        return F.binary_cross_entropy_with_logits(pred, target.float(), reduction="sum") / avg_factor


class RegressionLoss(nn.Module):
    def forward(self, pred, target, prefix, suffix, **kwargs):
        zero = pred.sum() * 0.0
        return {f"{prefix}_box{suffix}": zero, f"{prefix}_cns{suffix}": zero,
                f"{prefix}_yns{suffix}": zero}


def decoder_for_check():
    decoder = UnifiedPerceptionDecoder.__new__(UnifiedPerceptionDecoder)
    nn.Module.__init__(decoder)
    decoder.task_select = ["det"]
    decoder.loss_det_cls = ClassificationLoss()
    decoder.loss_det_reg = RegressionLoss()
    decoder.det_sampler = EmptySceneSampler()
    decoder.det_reg_weights = None
    decoder.cls_threshold_to_reg = 0.0
    return decoder


def outputs(logits):
    det = {"det_instance_feature": logits, "det_classifications": [logits],
           "det_predictions": [logits.expand(-1, -1, 2)], "det_qualities": [None],
           "map_classifications": [], "map_predictions": [], "map_qualities": [], "ego_status_list": []}
    return det, {key: (value if not isinstance(value, list) else list(value)) for key, value in det.items()}


class CachedPair:
    def __init__(self, targets):
        self.targets = targets

    def __getitem__(self, _index):
        return {}, self.targets


def navsim_validity_checks():
    ann = SimpleNamespace(boxes=np.empty((0, 7), np.float32), velocity_3d=np.empty((0, 3), np.float32),
                          names=[], track_tokens=[])
    frames = [SimpleNamespace(annotations=ann, ego_status=SimpleNamespace(ego_pose=np.zeros(3))) for _ in range(3)]
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=1, num_future_frames=2),
        frames=frames, map_api=object(),
        get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(
            poses=np.zeros((num_trajectory_frames, 3), np.float32)),
    )
    source_module = sys.modules[SampleTargetBuilder.__module__]
    original_map = source_module.map_polylines
    source_module.map_polylines = lambda _scene, _frame: []
    try:
        built = SampleTargetBuilder().compute_targets(scene)
    finally:
        source_module.map_polylines = original_map
    expected = {"planning": True, "det": True, "map": True, "motion": True,
                "occupancy": False, "geometry": True}
    assert built["task_valid"] == expected
    # NAVSIM 1.1/2.0 caches produced before task_valid stored the same actual target fields, but no validity map.
    legacy_cache = {key: value for key, value in built.items() if key != "task_valid"}
    assert set(legacy_cache) == {"trajectory", "trajectory_mask", "boxes", "box_velocity", "box_names",
                                 "box_ids", "agent_future", "agent_future_mask", "map"}
    loaded = WithTokens(CachedPair(legacy_cache), ["legacy-cache"])[0][1]
    assert loaded["task_valid"] == expected
    assert check_sample({"cameras": [], "history_trajectory": np.zeros((4, 3)), "ego_velocity": np.zeros(3),
                         "ego_acceleration": np.zeros(3), "ego_angular_velocity": np.zeros(3),
                         "ego_steer": 0.0, "command": "straight"}, loaded) == []

    planning_only = {"trajectory": np.zeros((2, 3), np.float32), "trajectory_mask": np.ones(2, np.float32)}
    unknown = WithTokens(CachedPair(planning_only), ["partial-cache"])[0][1]
    assert unknown["task_valid"] == {"planning": True, "det": False, "map": False, "motion": False,
                                      "occupancy": False, "geometry": False}
    assert check_sample({"cameras": [], "history_trajectory": np.zeros((4, 3)), "ego_velocity": np.zeros(3),
                         "ego_acceleration": np.zeros(3), "ego_angular_velocity": np.zeros(3),
                         "ego_steer": 0.0, "command": "straight"}, unknown) == []

    malformed = {**legacy_cache, "boxes": np.zeros((1, 6), np.float32), "task_valid": expected}
    try:
        WithTokens(CachedPair(malformed), ["bad-cache"])[0]
    except ValueError as exc:
        assert "boxes" in str(exc)
    else:
        raise AssertionError("NAVSIM cache accepted malformed present detection annotations")


def run():
    navsim_validity_checks()
    rows = [
        {"planning": True, "det": False, "map": False, "motion": False, "occupancy": False, "geometry": False},
        {"planning": True, "det": True, "map": False, "motion": False, "occupancy": False, "geometry": False},
    ]
    _, batch_targets, _ = collate([({}, {"gt_task_valid": row}, None) for row in rows])
    assert batch_targets["gt_task_valid"] == rows

    logits = torch.tensor([[[2.0]], [[2.0]]], requires_grad=True)
    stage1, stage2 = outputs(logits)
    data = {"gt_task_valid": rows,
            "gt_bboxes_3d": [torch.empty(0, 7), torch.empty(0, 7)],
            "gt_labels_3d": [torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)]}
    losses = decoder_for_check().loss(stage1, stage2, data)
    mixed_loss = sum(value for key, value in losses.items() if key.startswith("det_loss_cls_"))
    mixed_loss.backward()
    assert logits.grad[0].abs().sum() == 0  # unknown labels must not become negatives
    assert logits.grad[1].abs().sum() > 0   # labelled-empty scene is valid background

    def one_scene_loss(task_valid):
        value = torch.tensor([[[2.0]]], requires_grad=True)
        first, second = outputs(value)
        sample_data = {"gt_bboxes_3d": [torch.empty(0, 7)],
                       "gt_labels_3d": [torch.empty(0, dtype=torch.long)]}
        if task_valid is not None:
            sample_data["gt_task_valid"] = [task_valid]
        return decoder_for_check().loss(first, second, sample_data)

    known_empty = {**rows[1]}
    baseline = one_scene_loss(known_empty)
    assert torch.allclose(mixed_loss.detach(), sum(v for k, v in baseline.items() if k.startswith("det_loss_cls_")))
    old_path = one_scene_loss(None)
    all_valid = one_scene_loss({"planning": True, "det": True, "map": True, "motion": True,
                                "occupancy": True, "geometry": True})
    assert old_path.keys() == all_valid.keys()
    assert all(torch.equal(old_path[key], all_valid[key]) for key in old_path)

    invalid = {**known_empty, "det": False}
    invalid_logits = torch.tensor([[[2.0]]], requires_grad=True)
    invalid_stage1, invalid_stage2 = outputs(invalid_logits)
    invalid_losses = decoder_for_check().loss(
        invalid_stage1, invalid_stage2,
        {"gt_task_valid": [invalid], "gt_bboxes_3d": [torch.empty(0, 7)],
         "gt_labels_3d": [torch.empty(0, dtype=torch.long)]})
    assert invalid_losses.keys() == baseline.keys()
    assert all(value.item() == 0 for value in invalid_losses.values())
    sum(invalid_losses.values()).backward()
    assert invalid_logits.grad is not None and invalid_logits.grad.abs().sum() == 0

    occ_batch = collate([({}, {"gt_occ_dense": torch.zeros(1), "gt_task_valid": rows[1]}, None),
                         ({}, {"gt_occ_dense": None, "gt_task_valid": rows[0]}, None)])[1]
    assert isinstance(occ_batch["gt_occ_dense"], list) and occ_batch["gt_occ_dense"][1] is None

    profile = Profile(name="check", cameras=["front"], view_tokens=["front"], horizon=2,
                      commands=["straight"], command_prompts=["drive straight"], status_layout=["speed"],
                      det_classes=["car"], box_range=[-10, -10, 10, 10], map_classes=["lane"],
                      map_roi=[20, 20], ego_box=[0, 0, 0, 4, 2, 1.5, 0], anchors_dir="",
                      delta_mean=[0, 0], delta_std=[1, 1], image_hw=[4, 4])
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = str(Path(temp_dir) / "camera.png")
        Image.new("RGB", (4, 4)).save(image_path)
        features = {
            "cameras": [{"view": "front", "path": image_path, "intrinsic": np.eye(3),
                         "sensor2ego": np.eye(4), "size": (4, 4)}],
            "history_trajectory": np.zeros((4, 3), np.float32), "ego_velocity": np.zeros(3),
            "ego_acceleration": np.zeros(3), "ego_angular_velocity": np.zeros(3), "ego_steer": 0.0,
            "command": "straight",
        }
        planning_only = {"trajectory": np.array([[1, 0, 0]], np.float32), "trajectory_mask": np.ones(1),
                         "task_valid": {name: False for name in ("planning", "det", "map", "motion", "occupancy", "geometry")}}
        _, prepared_targets = prepare_sample(features, planning_only, profile)
        assert check_sample(features, planning_only) == []
        assert prepared_targets["gt_ego_fut_masks"].tolist() == [1.0, 0.0]
        assert prepared_targets["box_mask"].sum() == prepared_targets["map_mask"].sum() == 0
        assert prepared_targets["gt_task_valid"]["planning"] is False

        labelled_empty = {"trajectory": planning_only["trajectory"], "trajectory_mask": np.ones(1),
                          "boxes": np.zeros((0, 7), np.float32), "box_velocity": np.zeros((0, 2), np.float32),
                          "box_names": [], "map": [],
                          "task_valid": {"planning": True, "det": True, "map": True, "motion": False,
                                         "occupancy": False, "geometry": False}}
        _, empty_targets = prepare_sample(features, labelled_empty, profile)
        assert empty_targets["gt_task_valid"]["det"] is True and empty_targets["gt_task_valid"]["map"] is True
        assert empty_targets["box_mask"].sum() == empty_targets["map_mask"].sum() == 0
        legacy_empty = {key: value for key, value in labelled_empty.items() if key != "task_valid"}
        legacy_empty.update(agent_future=np.zeros((0, 2, 2), np.float32),
                            agent_future_mask=np.zeros((0, 2), np.float32))
        assert check_sample(features, legacy_empty) == []
        _, legacy_targets = prepare_sample(features, legacy_empty, profile)
        assert not legacy_targets["gt_task_valid"]["occupancy"]
        assert check_sample(features, {**legacy_empty, "occ_path": "labels.npz"}) == []
        explicitly_missing = {**legacy_empty, "task_valid": {name: True for name in
                              ("planning", "det", "map", "motion", "occupancy", "geometry")}}
        assert "occupancy supervision is valid but occ_path is missing" in check_sample(features, explicitly_missing)
    print("supervision validity: NAVSIM 1.1/2.0 new/legacy labels, partial targets, unknown masking, and empty labels OK")


if __name__ == "__main__":
    run()
