"""python unidrivevla/tools/check_official_heads.py [--nuscenes DIR] [--b2d DIR]

The planning head's prompt and the inputs that do not need the VLM, against the official nuScenes and Bench2Drive
heads. Imports those heads with the mmcv / mmdet registries stubbed (no checkpoint, no GPU). A fixed batch is built
twice: the official method sees LiDAR-frame history and a command one-hot, ours sees the same history on the ego
axes and the command index the adapter stores. The tokenized prompts must match with official_command_text on (the
official `command.view(-1)` read of the one-hot); by default each sample's text must be its own command. Also the status features, and the profile fields head_kwargs turns into
occupancy queries, the history token, delta normalization, keypoint source and the log-size clamp.

Exits 1 on a mismatch. NUSCENES_OFFICIAL and B2D_OFFICIAL (the official repositories' roots) override the defaults.
"""

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from unidrivevla.adapters.unidrivevla import Profile, head_kwargs  # noqa: E402
from unidrivevla.models.unidrivevla.dense_heads.qwenvl3_vla_planning_head import (  # noqa: E402
    QwenVL3APlanningHead,
)

FAILURES = []


def check(ok, message):
    print(("ok   " if ok else "FAIL ") + message, flush=True)
    if not ok:
        FAILURES.append(message)


class _Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        return [ord(ch) for ch in text] or [0]

    def convert_tokens_to_ids(self, token):
        return 1

    def add_tokens(self, tokens):
        return 0


def _registry():
    class Registry:
        def register_module(self, cls=None, force=False, name=None):
            def deco(c):
                return c
            return deco(cls) if cls is not None else deco

    return Registry()


def _ensure(name, package=False):
    if name not in sys.modules:
        module = types.ModuleType(name)
        if package:
            module.__path__ = []
            module.__package__ = name
        sys.modules[name] = module
    return sys.modules[name]


def install_stubs():
    # The nuScenes head imports deepspeed at module level; its import pulls Triton, which wants a GPU.
    deepspeed = _ensure("deepspeed")
    deepspeed.zero = types.SimpleNamespace(Init=lambda *args, **kwargs: None)
    models = _ensure("mmdet.models", package=True)
    models.HEADS = _registry()
    builder = _ensure("mmdet.models.builder")
    builder.build_head = lambda *args, **kwargs: None
    builder.build_loss = lambda *args, **kwargs: None
    losses = _ensure("mmdet.models.losses")
    losses.accuracy = lambda *args, **kwargs: None
    if importlib.util.find_spec("timm") is None:
        layers = _ensure("timm.models.layers")
        layers.Mlp = object
    if importlib.util.find_spec("einops") is None:
        _ensure("einops").rearrange = lambda *args, **kwargs: None
    for name in (
        "projects", "projects.mmdet3d_plugin", "projects.mmdet3d_plugin.losses",
        "projects.mmdet3d_plugin.models", "projects.mmdet3d_plugin.models.detection3d",
        "projects.mmdet3d_plugin.models.map", "projects.mmdet3d_plugin.ops",
        "projects.mmdet3d_plugin.core",
    ):
        _ensure(name, package=True)
    classes = {
        "projects.mmdet3d_plugin.losses.collision_loss": ["CollisionLoss"],
        "projects.mmdet3d_plugin.losses.plan_map_loss": ["GTMapBoundLoss", "GTMapDirectionLoss"],
        "projects.mmdet3d_plugin.models.detection3d.target": ["SparseBox3DTarget"],
        "projects.mmdet3d_plugin.models.detection3d.losses": ["SparseBox3DLoss"],
        "projects.mmdet3d_plugin.models.detection3d.detection3d_blocks": ["SparseBox3DEncoder"],
        "projects.mmdet3d_plugin.models.map.target": ["SparsePoint3DTarget"],
        "projects.mmdet3d_plugin.models.map.loss": ["SparseLineLoss"],
        "projects.mmdet3d_plugin.models.map.map_blocks": ["SparsePoint3DEncoder"],
        "projects.mmdet3d_plugin.models.map.decoder": ["SparsePoint3DDecoder"],
        "projects.mmdet3d_plugin.models.attention": ["gen_sineembed_for_position"],
        "projects.mmdet3d_plugin.ops": ["feature_maps_format"],
        "projects.mmdet3d_plugin.core.box3d": [],
    }
    for name, attrs in classes.items():
        module = _ensure(name, package=False)
        for attr in attrs:
            if not hasattr(module, attr):
                setattr(module, attr, type(attr, (), {}))


def load_py(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official_head(root: Path, kind: str):
    install_stubs()
    if kind == "nuscenes":
        folder = root / "projects" / "mmdet3d_plugin" / "unidrivevla" / "dense_heads"
        base = "off_ns_heads"
    else:
        folder = root / "projects" / "mmdet3d_plugin" / "models" / "vla"
        base = "off_b2d_vla"
    package = _ensure(base, package=True)
    package.__path__ = [str(folder)]
    package.__package__ = base
    load_py(f"{base}.constants", folder / "constants.py")
    load_py(f"{base}.utils", folder / "utils.py")
    for sibling, attrs in (
        ("unidrivevla_vlm_qwenvl3", ["Qwen3VLWithExpertModel"]),
        ("flex_attention_opt", ["build_blockmask_unidrive"]),
        ("modules", ["OccLatentDecoder", "DenseDepthNet"]),
        ("unified_perception_decoder", ["UnifiedPerceptionDecoder"]),
    ):
        module = _ensure(f"{base}.{sibling}")
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
    return load_py(f"{base}.qwenvl3_vla_planning_head", folder / "qwenvl3_vla_planning_head.py")


def _fake(head_cls, prompts, num_views, style, frame):
    system, user = prompts
    fake = head_cls.__new__(head_cls)
    fake._num_views = num_views
    fake.view_token_str_list = [
        "<FRONT_VIEW>", "<FRONT_LEFT_VIEW>", "<FRONT_RIGHT_VIEW>",
        "<BACK_LEFT_VIEW>", "<BACK_RIGHT_VIEW>", "<BACK_VIEW>",
    ][:num_views]
    fake.view_token_ids = torch.zeros(num_views, dtype=torch.long)
    fake.num_commands = len(fake_prompts_commands(style))
    fake.command_prompts = fake_prompts_commands(style)
    fake.system_prompt = system
    fake.user_prompt_template = user
    fake.prompt_style = style
    fake.coordinate_frame = frame
    fake.traj_ar_target = "waypoint"
    tokenizer = _Tokenizer()
    fake.qwen3_vl_with_expert = types.SimpleNamespace(processor=types.SimpleNamespace(tokenizer=tokenizer))
    return fake, tokenizer


def fake_prompts_commands(style):
    if style == "b2d":
        return ["TURN LEFT", "TURN RIGHT", "GO STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"]
    return ["TURN RIGHT ", "TURN LEFT  ", "GO STRAIGHT"]


def ego_from_lidar(hist):
    """LiDAR (x right, y forward) -> ego (x forward, y left)."""
    return torch.stack((hist[..., 1], -hist[..., 0]), dim=-1)


def one_hot(index, width):
    out = torch.zeros(index.shape[0], width)
    out.scatter_(1, index.view(-1, 1).long(), 1)
    return out


def compare_prompts(official_cls, our_cls, style, frame):
    from unidrivevla.models.unidrivevla.dense_heads.constants import select_prompts

    batch = 2
    width = 6 if style == "b2d" else 3
    index = torch.tensor([2, 0])  # straight, then the first command
    hist = torch.tensor([[[1.25, -3.5], [0.0, 10.0], [-2.0, 0.5], [4.0, 1.0]],
                         [[0.5, 2.0], [-1.0, -1.0], [3.0, 0.0], [0.0, 0.0]]])
    status = torch.tensor([[12.5, -0.4, 1.25, 0.0, 0.1, 0.2], [3.0, 0.5, -0.25, 0.0, 0.0, 0.0]])
    image = torch.zeros(batch, 6, 3, 8, 8)
    prompts = select_prompts(style, 6)
    ours, our_tok = _fake(our_cls, prompts, 6, style, frame)
    ours.official_command_text = True
    official, off_tok = _fake(official_cls, prompts, 6, style, frame)
    if style == "b2d":
        official._build_driving_batch(image, one_hot(index, width), status)
        ours._build_driving_batch(image, index, status, hist_traj=ego_from_lidar(hist))
    else:
        official._build_driving_batch(image, one_hot(index, width), status, hist)
        ours._build_driving_batch(image, index, status, hist_traj=ego_from_lidar(hist))
    check(our_tok.seen == off_tok.seen, f"{style} prompt strings ({len(our_tok.seen)} encode calls)")
    if style == "b2d":
        user = [text for text in our_tok.seen if "Ego speed" in text]
        check(len(user) == batch and all("PT_HIST" not in text for text in user), f"{style} prompt has no history")
        check("12.50" in user[0] and "-0.40" in user[0] and "1.25" in user[0], f"{style} speed and acceleration text")
    else:
        user = [text for text in our_tok.seen if "PT_HIST" in text]
        check(len(user) == batch and "(+001.25, -003.50)" in user[0], f"{style} history is the LiDAR-frame print")
        check("TURN RIGHT " in user[1] or "TURN LEFT" in user[0], f"{style} command text came from the one-hot flatten")
    fixed, fixed_tok = _fake(our_cls, prompts, 6, style, frame)
    fixed.official_command_text = False  # explicit ablation: each sample's own command, index or one-hot
    for command in (index, one_hot(index, width)):
        fixed_tok.seen.clear()
        fixed._build_driving_batch(image, command, status, hist_traj=ego_from_lidar(hist))
        user = [text for text in fixed_tok.seen if ("Ego speed" if style == "b2d" else "PT_HIST") in text]
        own = [fixed.command_prompts[int(i)] for i in index]
        check(len(user) == batch and all(text in u for text, u in zip(own, user)),
              f"{style} ablation command text is each sample's own ({'one-hot' if command.dim() == 2 else 'index'})")


def compare_status():
    fake = QwenVL3APlanningHead.__new__(QwenVL3APlanningHead)
    batch = types.SimpleNamespace(command=torch.tensor([1, 0]), ego_status=torch.arange(12, dtype=torch.float32).view(2, 6))
    for width in (3, 6):
        fake.num_commands = width
        fake.ego_status_dim = 6
        out = QwenVL3APlanningHead._maybe_get_status_features(fake, batch, None, use_gt=True, dtype=torch.float32)
        expect = torch.zeros(2, width + 6)
        expect[0, 1] = 1
        expect[1, 0] = 1
        expect[:, width:] = batch.ego_status
        check(torch.equal(out, expect), f"status features, {width} commands")


def compare_profiles():
    root = PACKAGE / "config" / "profile"
    nus = Profile.from_config(OmegaConf.load(root / "nuscenes.yaml"))
    b2d = Profile.from_config(OmegaConf.load(root / "b2d.yaml"))
    nus_kw = head_kwargs(nus, "vlm", None, False, False)
    b2d_kw = head_kwargs(b2d, "vlm", None, False, True)
    check(nus_kw["prompt_style"] == "nuscenes" and nus_kw["use_hist_traj"] and nus_kw["use_occ_tokens"],
          "nuScenes profile keeps history, occupancy and the nuScenes prompt")
    check(not b2d_kw["use_hist_traj"] and not b2d_kw["use_occ_tokens"] and b2d_kw["prompt_style"] == "b2d",
          "Bench2Drive profile drops history and occupancy queries")
    check(nus_kw["loss_planning"]["delta_mu"] == [0.0233, 2.2707]
          and nus_kw["loss_planning"]["delta_std"] == [0.3427, 1.8668],
          "nuScenes preserves main's native LiDAR loss normalization")
    check(b2d_kw["loss_planning"]["delta_mu"] == [-0.0222, 2.0249]
          and b2d_kw["loss_planning"]["delta_std"] == [0.672, 1.8586],
          "Bench2Drive preserves main's native LiDAR loss normalization")
    check(nus_kw["coordinate_frame"] == b2d_kw["coordinate_frame"] == "lidar",
          "main profiles keep native LiDAR inputs through the learned head")
    check(nus_kw["prefix_max_len"] == 3740 and b2d_kw["prefix_max_len"] == 3670
          and nus_kw["official_attention_mask"] and b2d_kw["official_attention_mask"],
          "main prefix caps and padding-context visibility are preserved")
    check(b2d_kw["inference_attn_impl"] == "eager" and not b2d_kw["cast_inference_mask"],
          "Bench2Drive inference attention is eager and the mask stays fp32")
    check(b2d_kw["summed_planning_loss"] and b2d_kw["official_command_text"]
          and nus_kw["official_command_text"], "the official profiles preserve main's loss sum and command read")
    check(b2d_kw["detach_plan_geometry"] and b2d_kw["processor_use_fast"] is False,
          "Bench2Drive geometry detach, slow processor")
    det = b2d_kw["unified_decoder_cfg"]["det_deformable"]
    check(det.get("keypoint_from_anchor") is True and det["kps_generator"]["size_clamp"] == [-5.0, 5.0],
          "Bench2Drive keypoints use the anchor embed and the wider size clamp")
    nus_det = nus_kw["unified_decoder_cfg"]["det_deformable"]
    check("keypoint_from_anchor" not in nus_det and "size_clamp" not in nus_det["kps_generator"],
          "nuScenes deformable config is unchanged")
    check(b2d_kw["unified_decoder_cfg"]["map_sampler"]["assigner"]["cost"].get("nan_to_num") is True
          and "nan_to_num" not in nus_kw["unified_decoder_cfg"]["map_sampler"]["assigner"]["cost"],
          "map-cost nan_to_num is Bench2Drive only")


def _production_ar_loss(weight):
    class FakeQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.language_model = types.SimpleNamespace(config=types.SimpleNamespace())
            self.visual = types.SimpleNamespace(config=types.SimpleNamespace())

        def forward(self, **kwargs):
            assert kwargs["input_ids"].shape == kwargs["labels"].shape
            return types.SimpleNamespace(loss=3.0 * self.weight + 1.25)

    class FakeRope:
        def get_rope_index(self, *, input_ids, **kwargs):
            positions = torch.zeros((3, *input_ids.shape), dtype=torch.long, device=input_ids.device)
            return positions, None

    head = QwenVL3APlanningHead.__new__(QwenVL3APlanningHead)
    nn.Module.__init__(head)
    qwen = FakeQwen()
    head.ar_loss_weight = float(weight)
    head.qwen3_vl_with_expert = types.SimpleNamespace(
        qwen3_vl=qwen,
        processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0)),
        vlm_base=types.SimpleNamespace(model=FakeRope()),
    )
    batch = {
        "ar_input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "ar_labels": torch.tensor([[-100, 2, 0], [-100, 4, 5]], dtype=torch.long),
    }
    result = QwenVL3APlanningHead.forward_ar_batch(head, batch)
    raw_expected = 3.0 * qwen.weight.detach() + 1.25
    scaled_expected = float(weight) * raw_expected
    check(torch.allclose(result["loss_vlm_raw"], raw_expected), "forward_ar_batch reports the unscaled raw AR loss")
    check(torch.allclose(result["loss_ar"], scaled_expected), f"forward_ar_batch applies production weight {weight:g}")
    result["loss_ar"].backward()
    grad_expected = torch.tensor(float(weight) * 3.0)
    check(torch.allclose(qwen.weight.grad, grad_expected), f"forward_ar_batch gradient is scaled by {weight:g}")
    return result["loss_ar"].detach(), qwen.weight.grad.detach()


def compare_ar_loss_contract(main_nuscenes, main_b2d):
    main_weights = {
        "nuscenes": inspect.signature(main_nuscenes.QwenVL3APlanningHead.__init__)
        .parameters["ar_loss_weight"].default,
        "b2d": inspect.signature(main_b2d.QwenVL3APlanningHead.__init__)
        .parameters["ar_loss_weight"].default,
    }
    check(main_weights == {"nuscenes": 0.1, "b2d": 1.0},
          "loaded main head signatures declare nuScenes=0.1 and B2D=1.0")

    profiles_root = PACKAGE / "config" / "profile"
    paths = {name: profiles_root / f"{name}.yaml" for name in
             ("nuscenes", "nuscenes_ego", "b2d", "b2d_ego")}
    profiles = {name: Profile.from_config(OmegaConf.load(path)) for name, path in paths.items()}
    expected = {"nuscenes": main_weights["nuscenes"], "nuscenes_ego": 0.1,
                "b2d": main_weights["b2d"], "b2d_ego": 0.1}
    kwargs = {
        name: head_kwargs(profile, "vlm", None, True, False)
        for name, profile in profiles.items()
    }
    for name, expected_weight in expected.items():
        check(profiles[name].ar_loss_weight == expected_weight
              and kwargs[name]["ar_loss_weight"] == expected_weight,
              f"{name} profile passes ar_loss_weight={expected_weight:g} into the planning head")

    legacy = OmegaConf.to_container(OmegaConf.load(paths["b2d_ego"]), resolve=True)
    legacy.pop("ar_loss_weight")
    check(Profile.from_config(legacy).ar_loss_weight == 0.1,
          "older profile metadata without ar_loss_weight keeps the 0.1 recipe")

    _, b2d_grad = _production_ar_loss(kwargs["b2d"]["ar_loss_weight"])
    _, nus_grad = _production_ar_loss(kwargs["nuscenes"]["ar_loss_weight"])
    check(torch.allclose(b2d_grad / nus_grad, torch.tensor(10.0)),
          "native B2D VQA/AR gradient matches main's 10x nuScenes weighting")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes", default=None)
    parser.add_argument("--b2d", default=None)
    args = parser.parse_args()
    code = PACKAGE.parent
    nus_root = Path(args.nuscenes or __import__("os").environ.get("NUSCENES_OFFICIAL", code / "nuScenes"))
    b2d_root = Path(args.b2d or __import__("os").environ.get("B2D_OFFICIAL", code / "Bench2Drive"))
    compare_profiles()
    compare_status()
    ns_mod = load_official_head(nus_root, "nuscenes")
    b2d_mod = load_official_head(b2d_root, "b2d")
    compare_ar_loss_contract(ns_mod, b2d_mod)
    from unidrivevla.models.unidrivevla.dense_heads.constants import (
        B2D_USER_PROMPT_TEMPLATE, NUSCENES_SYSTEM_PROMPT, NUSCENES_USER_PROMPT_TEMPLATE,
    )
    check(ns_mod.NUSCENES_SYSTEM_PROMPT == NUSCENES_SYSTEM_PROMPT == b2d_mod.B2D_SYSTEM_PROMPT,
          "system prompt matches both official files")
    check(ns_mod.NUSCENES_USER_PROMPT_TEMPLATE == NUSCENES_USER_PROMPT_TEMPLATE, "nuScenes user prompt")
    check(b2d_mod.B2D_USER_PROMPT_TEMPLATE == B2D_USER_PROMPT_TEMPLATE, "Bench2Drive user prompt")
    compare_prompts(ns_mod.QwenVL3APlanningHead, QwenVL3APlanningHead, "nuscenes", "ego")
    compare_prompts(b2d_mod.QwenVL3APlanningHead, QwenVL3APlanningHead, "b2d", "ego")
    if FAILURES:
        print(f"{len(FAILURES)} failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
