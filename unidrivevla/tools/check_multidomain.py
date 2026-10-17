"""CPU contract check for the opt-in multi-profile model and its source-homogeneous sampler."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig

from unidrivevla.adapters.multidomain import MultiProfilePolicy, _embedding_migration
from unidrivevla.adapters.unidrivevla import Profile
from unidrivevla.data.multidomain import (HomogeneousProfileBatchSampler, MultiProfileDataset,
                                           ProfileEvalBatchSampler)
from unidrivevla.data import registry
from unidrivevla.data.registry import LOADERS
import unidrivevla.data.multidomain  # registers the Hydra loader


class Tokenizer:
    def __init__(self):
        self.vocab = {"base": 0}

    def get_vocab(self):
        return dict(self.vocab)

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __len__(self):
        return len(self.vocab)


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()


class TinyCore(nn.Module):
    def __init__(self, view_tokens):
        super().__init__()
        self.qwen3_vl = nn.Linear(2, 2)
        self.qwen3_perception_expert = nn.Linear(2, 2)
        self.qwen3_action_expert = nn.Linear(2, 2)
        self.processor = Processor()
        self.view_token_base_vocab_size = 1
        self.processor.tokenizer.add_tokens(view_tokens)
        configs = [PretrainedConfig() for _ in range(5)]
        for config, implementation in zip(configs, (None, "eager", "sdpa", None, "flash_attention_2")):
            config._attn_implementation = implementation
        self.qwen3_vl.config = configs[0]
        self.qwen3_vl.language_model = SimpleNamespace(config=configs[1])
        self.qwen3_vl.visual = SimpleNamespace(config=configs[2])
        self.qwen3_perception_expert.config = configs[3]
        self.qwen3_action_expert.config = configs[4]


class TinyHead(nn.Module):
    train_vlm = True

    def __init__(self, profile, runtime, with_motion=False):
        super().__init__()
        self.profile_name = profile.name
        self.horizon = profile.horizon
        self.private = nn.Parameter(torch.tensor(1.0))
        self.with_motion = bool(with_motion)
        self.motion_proj_up = nn.Linear(2, 3) if with_motion else None
        self.motion_proj_down = nn.Linear(3, 2) if with_motion else None
        self.unified_decoder = nn.Module()
        if with_motion:
            self.unified_decoder.motion_anchor = nn.Parameter(torch.ones(2))
            self.unified_decoder.motion_instance_bank = nn.Linear(2, 2)
            self.unified_decoder.motion_anchor_encoder = nn.Linear(2, 2)
            self.unified_decoder.motion_deformable = nn.Linear(2, 2)
            self.unified_decoder.motion_refine = nn.Linear(2, 2)
            self.unified_decoder.motion_sampler = nn.Linear(2, 2)
            self.unified_decoder.motion_decoder = nn.Linear(2, 2)
            self.unified_decoder.loss_motion_cls = nn.Linear(2, 2)
            self.unified_decoder.loss_motion_reg = nn.Linear(2, 2)
        object.__setattr__(self, "runtime", runtime)
        self.ar_calls = 0

    def forward_ar_batch(self, _batch):
        self.ar_calls += 1
        shared = self.runtime.core.qwen3_vl.weight.sum()
        return {"loss_ar": shared * self.private, "loss_vlm_raw": shared.detach()}

    def forward_train(self, ar_batch=None, **_kwargs):
        loss = self.runtime.core.qwen3_vl.weight.sum() * self.private
        losses = {"loss_plan": loss}
        if ar_batch is not None:
            losses["loss_ar"] = self.forward_ar_batch(ar_batch)["loss_ar"]
        return {"losses": losses}

    def forward_test(self, **_kwargs):
        return {"traj": torch.zeros(1, self.horizon, 2)}


class Samples:
    def __init__(self, start, size):
        self.start, self.size = start, size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"value": self.start + index}, {}, f"token-{self.start + index}"


class SequencedSamples(Samples):
    def sequences(self):
        return [list(range(index, min(index + 2, self.size))) for index in range(0, self.size, 2)]


def composed_config():
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).parents[1] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve()),
                               job_name="check_multidomain"):
        cfg = compose(config_name="train", overrides=["agent=multidomain", "data=multidomain",
                                                        "train_test_split=navtrain"])
    assert cfg.data_loader == "multidomain"
    assert set(cfg.profiles) == {"nuscenes", "b2d", "navsim"}
    assert cfg.agent._target_.endswith("MultiProfilePolicy")
    assert cfg.agent.exact_resume is False
    return cfg


def profile(name, tokens, horizon, cast_stream_dtype=False, processor_use_fast=None):
    return Profile(name=name, cameras=["front"] * len(tokens), view_tokens=tokens, horizon=horizon,
                   commands=["go"], command_prompts=["GO"], status_layout=[], det_classes=["car"],
                   box_range=[-10, -10, 10, 10], map_classes=["line"], map_roi=[20, 20],
                   ego_box=[0, 0, 0, 4, 2, 1, 0], anchors_dir="unused", delta_mean=[0, 0], delta_std=[1, 1],
                   cast_stream_dtype=cast_stream_dtype, processor_use_fast=processor_use_fast)


def policy(checkpoint=None, initial_profile=None, with_motion=False, checkpoint_with_motion=None, exact_resume=False,
           checkpoint_frame=None):
    profiles = {"nuscenes": profile("nuscenes", ["<A>", "<B>"], 6),
                "navsim": profile("navsim", ["<B>", "<C>"], 8, cast_stream_dtype=True,
                                  processor_use_fast=False)}
    core_builds = []

    def build_core(**kwargs):
        core_builds.append(tuple(kwargs["view_tokens"]))
        return TinyCore(kwargs["view_tokens"])

    model = MultiProfilePolicy(
        profiles, vlm_path="unused", train_vlm=True, with_motion=with_motion,
        checkpoint_path=checkpoint, initial_profile=initial_profile,
        checkpoint_with_motion=checkpoint_with_motion,
        checkpoint_frame=checkpoint_frame,
        exact_resume=exact_resume,
        _shared_core_factory=build_core,
        _processor_factory=lambda _path, use_fast: SimpleNamespace(tokenizer=Tokenizer(), use_fast=use_fast),
        _head_factory=lambda profile, runtime: TinyHead(profile, runtime, with_motion),
    )
    assert len(core_builds) == 1
    return model


def main():
    model = policy().train()
    try:
        model.initialize(strict=True)
    except ValueError as exc:
        assert "checkpoint_path" in str(exc)
    else:
        raise AssertionError("strict inference initialization must require a checkpoint")
    runtime = model._profile_runtime["navsim"]
    attention_configs = [model.shared_core.qwen3_vl.config,
                         model.shared_core.qwen3_vl.language_model.config,
                         model.shared_core.qwen3_vl.visual.config,
                         model.shared_core.qwen3_perception_expert.config,
                         model.shared_core.qwen3_action_expert.config]
    saved_attention = [cfg._attn_implementation for cfg in attention_configs]
    with runtime.inference_config():
        assert all(cfg._attn_implementation == runtime._inference_attn_impl for cfg in attention_configs)
    assert [cfg._attn_implementation for cfg in attention_configs] == saved_attention
    assert len({id(head.runtime.core) for head in model.heads.values()}) == 1
    assert list(model.heads) == ["navsim", "nuscenes"]
    assert len({id(head.runtime.processor) for head in model.heads.values()}) == 2
    assert model.heads["nuscenes"].runtime.cast_stream_dtype is False
    assert model.heads["navsim"].runtime.cast_stream_dtype is True
    assert model.heads["navsim"].runtime.processor.use_fast is False
    keys = list(model.state_dict())
    assert sum(key.startswith("shared_core.qwen3_vl.") for key in keys) == 2
    assert not any("runtime" in key for key in keys)
    navsim_view = model.policy_for_source("navsim").to("cpu").eval()
    assert navsim_view.trajectory_sampling.num_poses == 8
    assert len(list(navsim_view.parameters())) == len(list(model.parameters()))
    assert navsim_view({"image": torch.zeros(1)})["pred_traj"].shape == (1, 8, 3)
    runtime_state = navsim_view.training_runtime_state_dict()
    assert runtime_state == {"modules": {}, "samplers": {}}
    navsim_view.load_training_runtime_state_dict(runtime_state, device="cpu")
    exact_policy = policy(exact_resume=True)
    multi_runtime = exact_policy.training_runtime_state_dict()
    assert set(multi_runtime) == {"profiles"}
    assert set(multi_runtime["profiles"]) == set(exact_policy.profiles)
    exact_policy.load_training_runtime_state_dict(multi_runtime, device="cpu")
    for invalid_state, expected in (
        ({"profiles": {"nuscenes": multi_runtime["profiles"]["nuscenes"]}}, "do not match"),
        ({"profiles": {**multi_runtime["profiles"], "extra": {}}}, "do not match"),
    ):
        try:
            exact_policy.load_training_runtime_state_dict(invalid_state, device="cpu")
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("multidomain runtime restore accepted an incomplete or unknown profile")
    try:
        exact_policy.load_training_runtime_state_dict(multi_runtime, device="not-a-device")
    except ValueError as exc:
        assert "invalid runtime-state restore device" in str(exc)
    else:
        raise AssertionError("multidomain runtime restore accepted an invalid device")
    try:
        navsim_view.initialize(strict=True)
    except ValueError as exc:
        assert "checkpoint_path" in str(exc)
    else:
        raise AssertionError("strict profile-view initialization must require a checkpoint")
    model.train()

    # The single-profile adapter sends combined VQA/planning into forward_train exactly once.
    targets = {
        "gt_bboxes_3d": torch.zeros(1, 1, 7), "gt_labels_3d": torch.zeros(1, 1, dtype=torch.long),
        "gt_agent_fut_trajs": torch.zeros(1, 1, 2, 2), "gt_agent_fut_masks": torch.ones(1, 1, 2),
        "gt_map_pts": torch.zeros(1, 1, 2), "gt_map_labels": torch.zeros(1, 1, dtype=torch.long),
        "box_mask": torch.ones(1, 1, dtype=torch.bool), "map_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    out = model({"profile": ["nuscenes"], "ar_batch": {}}, targets)
    assert model.heads["nuscenes"].ar_calls == 1
    out.loss.backward()
    assert model.shared_core.qwen3_vl.weight.grad is not None
    assert model.heads["nuscenes"].private.grad is not None
    assert model.heads["navsim"].private.grad is None
    try:
        model({"task": "vqa", "profile": ["nuscenes", "navsim"], "ar_batch": {}})
    except ValueError as exc:
        assert "homogeneous" in str(exc)
    else:
        raise AssertionError("mixed-domain batches must be rejected")

    all_none = registry.collate([
        ({"profile": "nuscenes"}, {"optional": None}, "source=nuscenes::a"),
        ({"profile": "nuscenes"}, {"optional": None}, "source=nuscenes::b"),
    ])[1]["optional"]
    assert all_none is None
    mixed_optional = registry.collate([
        ({"profile": "nuscenes"}, {"optional": torch.ones(1)}, "source=nuscenes::a"),
        ({"profile": "nuscenes"}, {"optional": None}, "source=nuscenes::b"),
    ])[1]["optional"]
    assert isinstance(mixed_optional, list) and mixed_optional[0] is not None and mixed_optional[1] is None

    optimizer = model.get_optimizers()
    optimized = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimized) == len(set(optimized))

    dataset = MultiProfileDataset([Samples(0, 7), Samples(100, 9)], ["nuscenes", "navsim"], [1.0, 3.0])
    item, _, token = dataset[0]
    assert item["profile"] == "nuscenes" and token.startswith("source=nuscenes::")
    samplers = [HomogeneousProfileBatchSampler(dataset, 2, rank, 2, seed=17, sequence_split=0)
                for rank in range(2)]
    batches = [list(sampler) for sampler in samplers]
    assert len(batches[0]) == len(batches[1])
    for left, right in zip(*batches):
        domains = {dataset.source_names[dataset.source_of(index)] for index in left + right}
        assert len(domains) == 1

    eval_batches = [list(ProfileEvalBatchSampler(dataset, 2, rank, 2)) for rank in range(2)]
    eval_indices = [index for rank_batches in eval_batches for batch in rank_batches for index in batch]
    assert sorted(eval_indices) == list(range(len(dataset))) and len(set(eval_indices)) == len(dataset)
    for rank_batches in eval_batches:
        for batch in rank_batches:
            assert len({dataset.source_names[dataset.source_of(index)] for index in batch}) == 1

    sequenced = MultiProfileDataset([SequencedSamples(0, 8), SequencedSamples(10, 8)],
                                    ["nuscenes", "navsim"], [1, 1])
    seq_batches = [list(HomogeneousProfileBatchSampler(sequenced, 1, rank, 2, seed=4, sequence_split=2))
                   for rank in range(2)]
    for left, right in zip(*seq_batches):
        indices = [item[0] if isinstance(item, tuple) else item for item in left + right]
        assert len({sequenced.source_names[sequenced.source_of(index)] for index in indices}) == 1

    # The new checkpoint owns one shared core and all profile heads; restore is exact.
    with tempfile.TemporaryDirectory() as tmp:
        def frame_metadata(model, with_motion):
            return {"_target_": "unidrivevla.adapters.multidomain.MultiProfilePolicy",
                    "profiles": {name: {"name": profile.name, "frame": profile.frame}
                                 for name, profile in model.profiles.items()},
                    "with_motion": bool(with_motion)}

        legacy_frames = {name: source.frame for name, source in model.profiles.items()}
        checkpoint = Path(tmp) / "multi.ckpt"
        stage1_state = model.state_dict()
        torch.save({"state_dict": stage1_state, "agent_config": frame_metadata(model, False)}, checkpoint)
        restored = policy(checkpoint=checkpoint).eval()
        restored.initialize()
        restored.initialize(strict=True)
        for key, value in stage1_state.items():
            assert torch.equal(value, restored.state_dict()[key]), key

        stage2 = policy(checkpoint=checkpoint, with_motion=True)
        fresh_motion = {key: value.clone() for key, value in stage2.state_dict().items()
                        if stage2._motion_only_key(key, stage2.profiles)}
        assert fresh_motion and all(key not in stage1_state for key in fresh_motion)
        stage2.initialize()
        stage2_state = stage2.state_dict()
        assert all(torch.equal(value, stage2_state[key]) for key, value in stage1_state.items()
                   if key not in ("_multidomain_format_version", "_multidomain_with_motion"))
        assert all(torch.equal(value, stage2_state[key]) for key, value in fresh_motion.items())
        assert bool(stage2_state["_multidomain_with_motion"].item())
        try:
            policy(checkpoint=checkpoint, with_motion=True).initialize(strict=True)
        except ValueError as exc:
            assert "strict inference" in str(exc)
        else:
            raise AssertionError("strict inference must reject fresh motion-module expansion")

        # Complete legacy multidomain checkpoints infer stage from their motion tensors.
        legacy_multi = dict(stage1_state)
        legacy_multi.pop("_multidomain_with_motion")
        legacy_multi["_multidomain_format_version"] = torch.tensor(1, dtype=torch.int32)
        legacy_multi_path = Path(tmp) / "legacy-multidomain.ckpt"
        torch.save({"state_dict": legacy_multi}, legacy_multi_path)
        legacy_stage2 = policy(checkpoint=legacy_multi_path, with_motion=True, checkpoint_frame=legacy_frames)
        legacy_stage2.initialize()
        assert all(torch.equal(value, legacy_stage2.state_dict()[key]) for key, value in stage1_state.items()
                   if key not in ("_multidomain_format_version", "_multidomain_with_motion"))
        configured_legacy = Path(tmp) / "configured-legacy-multidomain.ckpt"
        torch.save({"state_dict": legacy_multi, "agent_config": {"with_motion": False}}, configured_legacy)
        policy(checkpoint=configured_legacy, with_motion=True, checkpoint_frame=legacy_frames).initialize()

        # A source motion tensor proves stage 2; omitting one can never become a fresh-weights exception.
        full_stage2 = policy(with_motion=True)
        stage2_full_state = full_stage2.state_dict()
        stage2_missing_motion = dict(stage2_full_state)
        motion_key = next(key for key in stage2_full_state if full_stage2._motion_only_key(key, full_stage2.profiles))
        stage2_missing_motion.pop(motion_key)
        stage2_missing_path = Path(tmp) / "stage2-missing-motion.ckpt"
        torch.save({"state_dict": stage2_missing_motion, "agent_config": frame_metadata(full_stage2, True)},
                   stage2_missing_path)
        try:
            policy(checkpoint=stage2_missing_path, with_motion=True).initialize()
        except KeyError as exc:
            assert motion_key in str(exc)
        else:
            raise AssertionError("same-stage checkpoint accepted a missing motion tensor")

        # Missing shared/private weights and shape changes are never covered by the motion exception.
        for bad_name in ("shared_core.qwen3_vl.weight", "heads.nuscenes.private"):
            broken = dict(stage1_state)
            broken.pop(bad_name)
            bad_path = Path(tmp) / f"missing-{bad_name.replace('.', '-')}.ckpt"
            torch.save({"state_dict": broken, "agent_config": frame_metadata(model, False)}, bad_path)
            try:
                policy(checkpoint=bad_path, with_motion=True).initialize()
            except KeyError as exc:
                assert "unrelated or partial missing" in str(exc)
            else:
                raise AssertionError(f"motion transition accepted missing common tensor {bad_name}")
        wrong_shape = dict(stage1_state)
        wrong_shape["shared_core.qwen3_vl.weight"] = torch.zeros(3, 2)
        wrong_shape_path = Path(tmp) / "wrong-shape.ckpt"
        torch.save({"state_dict": wrong_shape, "agent_config": frame_metadata(model, False)}, wrong_shape_path)
        try:
            policy(checkpoint=wrong_shape_path, with_motion=True).initialize()
        except RuntimeError as exc:
            assert "has shape" in str(exc)
        else:
            raise AssertionError("motion transition accepted a wrong-shaped common tensor")

        # Legacy single-profile loading requires explicit selection and maps its core once.
        single = {"head.qwen3_vl_with_expert." + key.removeprefix("shared_core."): value
                  for key, value in stage1_state.items() if key.startswith("shared_core.")}
        single.update({"head." + key.removeprefix("heads.nuscenes."): value
                       for key, value in stage1_state.items() if key.startswith("heads.nuscenes.")})
        legacy = Path(tmp) / "single.ckpt"
        torch.save({"state_dict": single,
                    "agent_config": {"profile": {"name": "nuscenes", "frame": model.profiles["nuscenes"].frame},
                                     "with_motion": False}}, legacy)
        selected = policy(checkpoint=legacy, initial_profile="nuscenes")
        selected.initialize()
        assert torch.equal(selected.shared_core.qwen3_vl.weight, stage1_state["shared_core.qwen3_vl.weight"])
        assert torch.equal(selected.heads["nuscenes"].private, stage1_state["heads.nuscenes.private"])
        selected_stage2 = policy(checkpoint=legacy, initial_profile="nuscenes", with_motion=True)
        selected_stage2.initialize()
        assert torch.equal(selected_stage2.heads["nuscenes"].private, stage1_state["heads.nuscenes.private"])
        assert all(torch.equal(value, selected_stage2.state_dict()[key])
                   for key, value in stage1_state.items() if key.startswith("shared_core."))
        assert all(torch.equal(value, selected_stage2.heads["nuscenes"].state_dict()[key.removeprefix("heads.nuscenes.")])
                   for key, value in stage1_state.items() if key.startswith("heads.nuscenes.")
                   and not selected_stage2._motion_only_key(key, selected_stage2.profiles))
        missing_selected_head = dict(single)
        missing_selected_head.pop("head.private")
        missing_selected_path = Path(tmp) / "single-missing-head.ckpt"
        torch.save({"state_dict": missing_selected_head,
                    "agent_config": {"profile": {"name": "nuscenes", "frame": model.profiles["nuscenes"].frame},
                                     "with_motion": False}}, missing_selected_path)
        try:
            policy(checkpoint=missing_selected_path, initial_profile="nuscenes").initialize()
        except KeyError as exc:
            assert "selected head" in str(exc)
        else:
            raise AssertionError("single-profile migration accepted a missing selected-head tensor")
        try:
            policy(checkpoint=legacy).initialize()
        except ValueError as exc:
            assert "initial_profile" in str(exc)
        else:
            raise AssertionError("legacy single-profile imports must name their source profile")

    source = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    target = torch.full((5, 2), 9.0)
    migrated = _embedding_migration(source, target, 2, {"<A>": 2, "<B>": 3}, {"<A>": 3, "<B>": 2}, "embed")
    assert torch.equal(migrated[:2], source[:2])
    assert torch.equal(migrated[2], source[3]) and torch.equal(migrated[3], source[2])
    assert torch.equal(migrated[4], target[4])

    # Compose the real Hydra configs, then exercise the registered loader with tiny source factories.
    cfg = composed_config()
    names = ("nuscenes", "b2d", "navsim")
    previous = {key: LOADERS.get(key) for key in names}
    previous_transform = registry.build_worker_transform
    received_augments = {}
    for source_id, name in enumerate(names):
        LOADERS[name] = lambda source_cfg, agent, _split, source_id=source_id, name=name: (
            Samples(source_id * 10, 4) if agent.profile.name == name and source_cfg.profile.name == name else None)

    class WorkerAgent:
        def policy_for_source(self, name):
            return SimpleNamespace(name=name)

    def fake_worker_transform(agent_view, augment, train):
        received_augments[agent_view.name] = (augment, train)
        return lambda features, targets: ({**features, "worker_profile": agent_view.name}, targets)

    try:
        registry.build_worker_transform = fake_worker_transform
        built = registry.build_split(cfg, WorkerAgent(), "train")
        nusc, _, _ = built[0]
        b2d, _, _ = built[4]
        navsim, _, _ = built[8]
        assert nusc["profile"] == "nuscenes" and nusc["worker_profile"] == "nuscenes"
        assert b2d["profile"] == "b2d" and b2d["worker_profile"] == "b2d"
        assert navsim["profile"] == "navsim" and navsim["worker_profile"] == "navsim"
        assert received_augments["nuscenes"][0].geometry.flip_p == 0.5
        assert received_augments["b2d"][0].photometric is True
        assert received_augments["navsim"][0] is None
    finally:
        registry.build_worker_transform = previous_transform
        for key, loader in previous.items():
            if loader is None:
                LOADERS.pop(key, None)
            else:
                LOADERS[key] = loader


if __name__ == "__main__":
    main()
    print("multidomain CPU checks passed")
