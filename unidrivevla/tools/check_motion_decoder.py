"""CPU oracle for UniDriveVLA's nuScenes agent-motion decoder versus the pinned main implementation."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from unidrivevla.models.compat import Registry
from unidrivevla.models.models.detection3d.decoder import SparseBox3DDecoder
from unidrivevla.models.models.motion.decoder import SparseMotionDecoder


MAIN_PLUGIN = ROOT / "nuScenes" / "projects" / "mmdet3d_plugin"
MAIN_BOX3D = MAIN_PLUGIN / "core" / "box3d.py"
MAIN_DETECTION_DECODER = MAIN_PLUGIN / "models" / "detection3d" / "decoder.py"
MAIN_MOTION_DECODER = MAIN_PLUGIN / "models" / "motion" / "decoder.py"


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def load_main_decoders():
    """Load main's real box constants, detection base, and motion decoder; stub only external registry/util imports."""
    for path in (MAIN_BOX3D, MAIN_DETECTION_DECODER, MAIN_MOTION_DECODER):
        if not path.is_file():
            raise FileNotFoundError(f"pinned main source is missing: {path}")

    names = (
        "mmdet", "mmdet.core", "mmdet.core.bbox", "mmdet.core.bbox.builder",
        "projects", "projects.mmdet3d_plugin", "projects.mmdet3d_plugin.core",
        "projects.mmdet3d_plugin.core.box3d", "projects.mmdet3d_plugin.models",
        "projects.mmdet3d_plugin.models.detection3d",
        "projects.mmdet3d_plugin.models.detection3d.decoder",
        "projects.mmdet3d_plugin.datasets", "projects.mmdet3d_plugin.datasets.utils",
    )
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        _package(name)

    registry = Registry("pinned-main-oracle")
    sys.modules["mmdet.core.bbox.builder"].BBOX_CODERS = registry
    sys.modules["projects.mmdet3d_plugin.datasets.utils"].box3d_to_corners = lambda boxes: np.zeros(
        (len(boxes), 8, 3), dtype=np.float32
    )

    try:
        box3d_spec = importlib.util.spec_from_file_location("projects.mmdet3d_plugin.core.box3d", MAIN_BOX3D)
        box3d = importlib.util.module_from_spec(box3d_spec)
        sys.modules[box3d_spec.name] = box3d
        box3d_spec.loader.exec_module(box3d)

        detection_spec = importlib.util.spec_from_file_location(
            "projects.mmdet3d_plugin.models.detection3d.decoder", MAIN_DETECTION_DECODER
        )
        detection = importlib.util.module_from_spec(detection_spec)
        sys.modules[detection_spec.name] = detection
        detection_spec.loader.exec_module(detection)

        motion_spec = importlib.util.spec_from_file_location("pinned_main_motion_decoder_oracle", MAIN_MOTION_DECODER)
        motion = importlib.util.module_from_spec(motion_spec)
        motion_spec.loader.exec_module(motion)
        return motion.SparseMotionDecoder, detection.SparseBox3DDecoder
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _fixture(seed=731):
    generator = torch.Generator().manual_seed(seed)
    batch, anchors, classes, modes, steps, box_dims = 2, 5, 3, 4, 6, 10
    cls = [torch.randn(batch, anchors, classes, generator=generator) for _ in range(2)]
    box = [torch.randn(batch, anchors, box_dims, generator=generator) for _ in range(2)]
    # Keep exponentiated dimensions in a realistic finite range and yaw encodings nonzero.
    for layer in box:
        layer[..., 3:6] = layer[..., 3:6].clamp(-1.0, 1.0)
        layer[..., 6] = torch.sin(layer[..., 6])
        layer[..., 7] = torch.cos(layer[..., 7])
    quality = [torch.randn(batch, anchors, 2, generator=generator) for _ in range(2)]
    motion = {
        "classification": [torch.randn(batch, anchors, modes, generator=generator) for _ in range(2)],
        "prediction": [torch.randn(batch, anchors, modes, steps, 2, generator=generator) for _ in range(2)],
    }
    instance_ids = torch.arange(batch * anchors).reshape(batch, anchors)
    return cls, box, quality, motion, instance_ids


def _assert_same(main, native, *, context):
    assert len(main) == len(native), context
    for sample_idx, (expected, actual) in enumerate(zip(main, native)):
        assert set(expected) == {"trajs_3d", "trajs_score"}, (context, sample_idx, expected.keys())
        assert set(actual) == set(expected), (context, sample_idx, actual.keys())
        assert expected["trajs_3d"].device.type == actual["trajs_3d"].device.type == "cpu"
        torch.testing.assert_close(actual["trajs_3d"], expected["trajs_3d"], rtol=0, atol=0,
                                   msg=f"{context}: sample {sample_idx} trajectory mismatch")
        torch.testing.assert_close(actual["trajs_score"], expected["trajs_score"], rtol=0, atol=0,
                                   msg=f"{context}: sample {sample_idx} mode score mismatch")
        assert actual["trajs_3d"].shape[:2] == actual["trajs_score"].shape
        assert actual["trajs_3d"].shape[-1] == 2
        assert torch.isfinite(actual["trajs_3d"]).all()


def _configured_decoder(decoder_cls, config):
    """Production config only selects the decoder type; set inherited knobs on this small oracle fixture."""
    decoder = decoder_cls()
    for name, value in config.items():
        setattr(decoder, name, value)
    return decoder


def check_decoder_parity(MainSparseMotionDecoder):
    cls, boxes, quality, motion, instance_ids = _fixture()
    cases = (
        ("class-topk-no-quality", dict(num_output=8), None, None, -1),
        ("quality-sort-and-threshold", dict(num_output=12, score_threshold=0.38), quality, None, -1),
        ("instance-id-squeeze", dict(num_output=5, sorted=False), quality, instance_ids, -1),
        ("selected-layer-with-null-quality", dict(num_output=7), [quality[0], None], None, -1),
        ("threshold-empty", dict(num_output=9, score_threshold=1.1), None, None, -1),
        ("first-output-layer", dict(num_output=6), quality, None, 0),
    )
    for name, kwargs, qualities, ids, output_idx in cases:
        args = dict(cls_scores=cls, box_preds=boxes, instance_id=ids, quality=qualities,
                    motion_output=motion, output_idx=output_idx)
        expected = _configured_decoder(MainSparseMotionDecoder, kwargs).decode(**args)
        actual = _configured_decoder(SparseMotionDecoder, kwargs).decode(**args)
        _assert_same(expected, actual, context=name)


def check_post_process_and_no_motion_stage(MainSparseMotionDecoder, MainSparseBox3DDecoder):
    from unidrivevla.models.unidrivevla.dense_heads.unified_perception_decoder import UnifiedPerceptionDecoder

    cls, boxes, quality, motion, _ = _fixture(seed=991)
    stage2 = {
        "det_classifications": cls,
        "det_predictions": boxes,
        "det_qualities": quality,
        "motion_cls_list": motion["classification"],
        "motion_reg_list": motion["prediction"],
        "map_classifications": [],
        "map_predictions": [],
    }

    class CountingMotionDecoder:
        def __init__(self, decoder):
            self.decoder = decoder
            self.calls = 0

        def decode(self, *args, **kwargs):
            self.calls += 1
            return self.decoder.decode(*args, **kwargs)

    def decoder_with_motion(motion_decoder):
        decoder = UnifiedPerceptionDecoder.__new__(UnifiedPerceptionDecoder)
        nn.Module.__init__(decoder)
        decoder.with_instance_id = True
        decoder.det_decoder = _configured_decoder(SparseBox3DDecoder, {"num_output": 5})
        decoder.map_decoder = None
        decoder.motion_decoder = motion_decoder
        decoder.det_instance_bank = types.SimpleNamespace(
            get_instance_id=lambda *_: torch.arange(cls[0].shape[0] * cls[0].shape[1]).reshape(
                cls[0].shape[0], cls[0].shape[1]
            )
        )
        return decoder

    no_motion, _ = decoder_with_motion(None).post_process(stage2)
    assert no_motion is not None
    assert all("trajs_3d" not in item and "trajs_score" not in item for item in no_motion), \
        "stage 1 / disabled motion decode must not synthesize per-actor outputs"

    counted_motion_decoder = CountingMotionDecoder(_configured_decoder(SparseMotionDecoder, {"num_output": 5}))
    with_motion, _ = decoder_with_motion(counted_motion_decoder).post_process(stage2)
    assert counted_motion_decoder.calls == 1, "post_process must run the motion decoder exactly once"
    assert with_motion is not None
    assert all("trajs_3d" in item and "trajs_score" in item for item in with_motion)
    instance_ids = torch.arange(cls[0].shape[0] * cls[0].shape[1]).reshape(cls[0].shape[0], cls[0].shape[1])
    expected_motion = _configured_decoder(MainSparseMotionDecoder, {"num_output": 5}).decode(
        cls, boxes, instance_id=instance_ids, quality=quality, motion_output=motion
    )
    expected_detections = _configured_decoder(MainSparseBox3DDecoder, {"num_output": 5}).decode(
        cls, boxes, instance_id=instance_ids, quality=quality
    )
    for i, item in enumerate(with_motion):
        assert item["trajs_3d"].shape[0] == item["boxes_3d"].shape[0]
        assert item["trajs_score"].shape[0] == item["scores_3d"].shape[0]
        assert item["cls_scores"].shape == item["scores_3d"].shape
        assert item["instance_ids"].shape == item["scores_3d"].shape
        for key in ("boxes_3d", "scores_3d", "labels_3d", "cls_scores", "instance_ids"):
            torch.testing.assert_close(item[key], expected_detections[i][key], rtol=0, atol=0,
                                       msg=f"detection output changed while adding motion key {key}")
        expected = expected_motion[i]
        torch.testing.assert_close(item["trajs_3d"], expected["trajs_3d"], rtol=0, atol=0)
        torch.testing.assert_close(item["trajs_score"], expected["trajs_score"], rtol=0, atol=0)


def check_profile_gate_and_checkpoint_neutrality():
    from omegaconf import OmegaConf

    from unidrivevla.adapters.unidrivevla import Profile, head_kwargs

    config_root = ROOT / "unidrivevla" / "config" / "profile"
    nuscenes_profile = Profile.from_config(OmegaConf.load(config_root / "nuscenes.yaml"))
    ego_profile = Profile.from_config(OmegaConf.load(config_root / "nuscenes_ego.yaml"))
    b2d_profile = Profile.from_config(OmegaConf.load(config_root / "b2d.yaml"))

    def decoder_cfg(profile, with_motion):
        return head_kwargs(profile, "unused-vlm", None, False, with_motion)["unified_decoder_cfg"]

    assert "motion_decoder" not in decoder_cfg(nuscenes_profile, with_motion=False)
    assert decoder_cfg(nuscenes_profile, with_motion=True)["motion_decoder"] == {"type": "SparseMotionDecoder"}
    assert "motion_decoder" not in decoder_cfg(ego_profile, with_motion=True)
    assert "motion_decoder" not in decoder_cfg(b2d_profile, with_motion=True)

    # The decoder is an ordinary, parameter-free callable; attaching it to the nn.Module does not add checkpoint keys.
    native = SparseMotionDecoder()
    assert not isinstance(native, nn.Module)
    holder = nn.Module()
    holder.motion_decoder = native
    assert holder.state_dict() == {}
    assert not list(native.__dict__.values()) or all(not torch.is_tensor(v) for v in native.__dict__.values())


def check_stage3_motion_encoder_and_ego_anchor():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from unidrivevla.adapters.unidrivevla import Profile, head_kwargs
    from unidrivevla.models.compat import Linear, PLUGIN_LAYERS, POSITIONAL_ENCODING, build_from_cfg
    from unidrivevla.models.models.attention import gen_sineembed_for_position
    from unidrivevla.models.models.blocks import linear_relu_ln
    import unidrivevla.models.models.detection3d.detection3d_blocks  # noqa: F401
    import unidrivevla.models.models.ego.instance_bank  # noqa: F401

    config_dir = ROOT / "unidrivevla" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve()),
                               job_name="check_nusc_stage3_motion_encoder"):
        stage3 = compose(config_name="train", overrides=["data=nuscenes", "stage=nusc_2b_stage3"])
    profile = Profile.from_config(OmegaConf.to_container(stage3.profile, resolve=True))
    assert profile.motion_anchor_encoder == "MotionAnchorSineEncoder"
    decoder = head_kwargs(profile, "unused-vlm", None, train_vlm=False, with_motion=True)["unified_decoder_cfg"]
    encoder_cfg = decoder["motion_anchor_encoder"]
    assert encoder_cfg == {"type": "MotionAnchorSineEncoder", "embed_dims": 256}

    torch.manual_seed(241)
    encoder = build_from_cfg(encoder_cfg, POSITIONAL_ENCODING)
    torch.manual_seed(241)
    reference = nn.Sequential(*linear_relu_ln(256, 1, 1), Linear(256, 256))
    expected_keys = {"encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias",
                     "encoder.3.weight", "encoder.3.bias"}
    assert set(encoder.state_dict()) == expected_keys
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 132096
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(encoder.state_dict()[f"encoder.{name}"], value, rtol=0, atol=0)

    positions = torch.tensor([[[0.1, 0.25], [-0.5, 0.75], [1.0, -0.25]],
                              [[0.0, 0.0], [0.4, -0.8], [-0.2, 0.6]]])
    sine_embed = gen_sineembed_for_position(positions)
    assert sine_embed.shape == (2, 3, 256)
    torch.testing.assert_close(encoder(sine_embed), reference(sine_embed), rtol=0, atol=0)

    def build_ego_bank(profile):
        bank_cfg = head_kwargs(profile, "unused-vlm", None, train_vlm=False,
                               with_motion=False)["unified_decoder_cfg"]["ego_instance_bank"]
        return build_from_cfg(bank_cfg, PLUGIN_LAYERS)

    config_root = ROOT / "unidrivevla" / "config" / "profile"
    old_native = Profile.from_config(OmegaConf.to_container(OmegaConf.load(config_root / "nuscenes.yaml"),
                                                             resolve=True))
    assert old_native.motion_anchor_encoder is None  # metadata predating the Stage3 field keeps the legacy encoder
    old_native.ego_box[-1] = 0.0
    assert head_kwargs(old_native, "unused-vlm", None, train_vlm=False,
                       with_motion=True)["unified_decoder_cfg"]["motion_anchor_encoder"]["type"] \
        == "SparseBox3DEncoder"
    old_native_bank = build_ego_bank(old_native)
    old_saved_state = old_native_bank.state_dict()

    native_bank = build_ego_bank(profile)
    expected_anchor = torch.tensor([[0.0, 0.5, -1.06, np.log(4.08), np.log(1.73), np.log(1.56),
                                     1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(native_bank.anchor.detach(), expected_anchor, rtol=0, atol=1e-6)
    assert native_bank.state_dict()["anchor"].shape == old_saved_state["anchor"].shape == (1, 11)
    native_bank.load_state_dict(old_saved_state, strict=True)
    torch.testing.assert_close(native_bank.anchor[0, 6:8], torch.tensor([0.0, 1.0]), rtol=0, atol=0)

    legacy = Profile.from_config(OmegaConf.to_container(OmegaConf.load(config_root / "nuscenes_ego.yaml"),
                                                         resolve=True))
    b2d = Profile.from_config(OmegaConf.to_container(OmegaConf.load(config_root / "b2d.yaml"), resolve=True))
    for unchanged in (legacy, b2d):
        bank = build_ego_bank(unchanged)
        torch.testing.assert_close(bank.anchor[0, 6:8], torch.tensor([0.0, 1.0]), rtol=0, atol=0)
        assert head_kwargs(unchanged, "unused-vlm", None, train_vlm=False,
                           with_motion=True)["unified_decoder_cfg"]["motion_anchor_encoder"]["type"] \
            == "SparseBox3DEncoder"

    lead_cfg = OmegaConf.to_container(OmegaConf.load(config_root / "lead.yaml"), resolve=False)
    lead_cfg["delta_mean"], lead_cfg["delta_std"] = [0.0, 0.0], [1.0, 1.0]
    lead = Profile.from_config(lead_cfg)
    lead_decoder = head_kwargs(lead, "unused-vlm", None, train_vlm=False,
                               with_motion=True)["unified_decoder_cfg"]
    assert lead.ego_box[-1] == 0.0
    if lead.prompt_style == "lead_pr10":
        assert Path(lead_decoder["ego_instance_bank"]["ego_anchor"]).name == "lead_ego_anchor.npy"
    else:
        torch.testing.assert_close(build_ego_bank(lead).anchor[0, 6:8], torch.tensor([0.0, 1.0]), rtol=0, atol=0)
    assert lead_decoder["motion_anchor_encoder"]["type"] == "SparseBox3DEncoder"


def check_policy_inference_contract():
    from types import SimpleNamespace

    from unidrivevla.adapters.unidrivevla import Profile, UniDriveVLAPolicy

    policy = UniDriveVLAPolicy.__new__(UniDriveVLAPolicy)
    nn.Module.__init__(policy)
    policy.profile = Profile(name="nuscenes", cameras=[], view_tokens=[], horizon=2, commands=[],
                             command_prompts=[], status_layout=[], det_classes=[], box_range=[0, 0, 0, 0],
                             map_classes=[], map_roi=[0, 0], ego_box=[0] * 7, anchors_dir="unused",
                             delta_mean=[0, 0], delta_std=[1, 1], frame="lidar")
    det = [{"boxes_3d": torch.tensor([[1.0, 2.0, 0, 4, 2, 1, 0]]),
            "trajs_3d": torch.ones(1, 2, 3, 2), "trajs_score": torch.ones(1, 2)}]
    policy.head = SimpleNamespace(forward_test=lambda **_: {"traj": torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]),
                                                            "det": det, "map": None})
    policy.eval()
    matrix = torch.tensor([[[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])
    result = policy({"coordinate_frame": "lidar", "native_to_ego": matrix})
    assert set(result) == {"pred_traj", "pred_traj_head", "det", "det_coordinate_frame"}
    assert result["det"] is det
    assert result["det_coordinate_frame"] == "lidar"
    torch.testing.assert_close(result["pred_traj"][..., :2], torch.tensor([[[0.0, 1.0], [-1.0, 1.0]]]))
    torch.testing.assert_close(result["pred_traj_head"][..., :2], torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]))

    policy.head = SimpleNamespace(forward_test=lambda **_: {"traj": torch.zeros(1, 2, 2), "det": [
        {"boxes_3d": torch.zeros(1, 7)}
    ]})
    without_motion = policy({"coordinate_frame": "lidar", "native_to_ego": matrix})
    assert set(without_motion) == {"pred_traj", "pred_traj_head"}, \
        "detection-only profiles must not change the policy's planning output"


def main():
    torch.set_num_threads(1)
    MainSparseMotionDecoder, MainSparseBox3DDecoder = load_main_decoders()
    check_decoder_parity(MainSparseMotionDecoder)
    check_post_process_and_no_motion_stage(MainSparseMotionDecoder, MainSparseBox3DDecoder)
    check_profile_gate_and_checkpoint_neutrality()
    check_stage3_motion_encoder_and_ego_anchor()
    check_policy_inference_contract()
    print("PASS main SparseMotionDecoder CPU oracle: top-k, quality/order, threshold masks, IDs, layer selection")
    print("PASS stage gating, no-motion output, state-dict neutrality, and policy native-frame contract")
    print("PASS Stage3 sine encoder construction/initialization and nuScenes native ego-anchor orientation")


if __name__ == "__main__":
    main()
