"""CPU check for the DDP communication-dtype contract and its two-rank reference."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "unidrivevla" / "config"
sys.path.insert(0, str(ROOT))


class MixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fp32_weight = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
        self.bf16_weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))

    def forward(self, fp32_scale, bf16_scale):
        return SimpleNamespace(loss=(self.fp32_weight * fp32_scale).sum() +
                               (self.bf16_weight * bf16_scale).sum())


class CastPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable_bf16 = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))
        self.frozen_fp32 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)
        self.frozen_bf16 = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.bfloat16), requires_grad=False)
        self.register_buffer("float_buffer", torch.tensor([3.0], dtype=torch.float32))


def _collective_reference(local_gradient, mode, world_size):
    if mode == "native":
        reference = local_gradient.detach().clone()
        dist.all_reduce(reference)
        return reference.div_(world_size)
    if mode == "fp32":
        reference = local_gradient.detach().float().div_(world_size)
        dist.all_reduce(reference)
        return reference.to(dtype=local_gradient.dtype)
    if mode == "bf16":
        reference = local_gradient.detach().to(dtype=torch.bfloat16)
        dist.all_reduce(reference)
        reference.div_(world_size)
        return reference.to(dtype=local_gradient.dtype)
    raise AssertionError(f"unknown reference mode: {mode}")


def _resolved_mode(requested, precision):
    if requested != "auto":
        return requested
    # This is the old registration condition in parallelize(): master not in (None, fp32).
    legacy_hook_was_enabled = precision == "bf16"
    return "fp32" if legacy_hook_was_enabled else "native"


def _run_case(rank, mode, precision):
    from unidrivevla.distributed.parallelize import parallelize
    from unidrivevla.train.args import TrainArgs

    policy = MixedPolicy()
    args = TrainArgs(strategy="ddp", precision=precision, compile=False,
                     ddp_reduce_dtype=mode, ddp_bucket_cap_mb=1)
    ctx = SimpleNamespace(device=torch.device("cpu"), world_size=2, local_rank=rank, is_main=(rank == 0))
    wrapped = parallelize(policy, args, ctx)

    # Unequal rank-local gradients. The fp32 value is deliberately between bf16 values:
    # bf16 communication rounds before SUM, so its result must differ from fp32 averaging.
    local_fp32 = torch.tensor([1.00390625 if rank == 0 else 1.0078125], dtype=torch.float32)
    local_bf16 = torch.tensor([0.5 if rank == 0 else 2.5], dtype=torch.bfloat16)
    loss = wrapped(local_fp32, local_bf16).loss
    loss.backward()

    effective = _resolved_mode(mode, precision)
    for name, parameter, local in (
        ("fp32_weight", policy.fp32_weight, local_fp32),
        ("bf16_weight", policy.bf16_weight, local_bf16),
    ):
        gradient = parameter.grad
        assert gradient is not None, f"{name}: missing gradient"
        assert gradient.dtype == parameter.dtype, f"{name}: {gradient.dtype} != {parameter.dtype}"
        assert gradient.shape == parameter.shape, f"{name}: {gradient.shape} != {parameter.shape}"
        expected = _collective_reference(local.to(dtype=parameter.dtype), effective, world_size=2)
        try:
            torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        except AssertionError as exc:
            raise AssertionError(f"{mode}/{precision}/{name}: {exc}") from exc

    result = torch.cat((policy.fp32_weight.grad.float(), policy.bf16_weight.grad.float()))
    outputs = [torch.empty_like(result) for _ in range(2)]
    dist.all_gather(outputs, result)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    if mode == "bf16" and precision == "native":
        assert result[0].item() != _collective_reference(local_fp32, "fp32", 2)[0].item(), (
            "the bf16 rounding probe must differ from fp32 reduction"
        )
    del wrapped


def _run_cast_case(rank, precision, ddp_cast_frozen_parameters):
    from unidrivevla.distributed.parallelize import parallelize
    from unidrivevla.train.args import TrainArgs

    policy = CastPolicy()
    parameter_ids = {name: id(parameter) for name, parameter in policy.named_parameters()}
    state_keys = tuple(policy.state_dict())
    buffer_before = policy.float_buffer.detach().clone()
    buffer_id = id(policy.float_buffer)
    args = TrainArgs(strategy="ddp", precision=precision,
                     ddp_cast_frozen_parameters=ddp_cast_frozen_parameters, compile=False)
    ctx = SimpleNamespace(device=torch.device("cpu"), world_size=2, local_rank=rank, is_main=(rank == 0))
    wrapped = parallelize(policy, args, ctx)

    expected_trainable = torch.bfloat16 if precision == "bf16" else torch.float32
    expected_frozen_fp32 = torch.bfloat16 if precision == "bf16" and ddp_cast_frozen_parameters else torch.float32
    expected_frozen_bf16 = torch.float32 if precision == "fp32" and ddp_cast_frozen_parameters else torch.bfloat16
    assert policy.trainable_bf16.dtype == expected_trainable
    assert policy.frozen_fp32.dtype == expected_frozen_fp32
    assert policy.frozen_bf16.dtype == expected_frozen_bf16
    assert policy.float_buffer.dtype == torch.float32
    torch.testing.assert_close(policy.float_buffer, buffer_before, rtol=0, atol=0)
    assert id(policy.float_buffer) == buffer_id
    assert {name: id(parameter) for name, parameter in policy.named_parameters()} == parameter_ids
    assert tuple(policy.state_dict()) == state_keys
    del wrapped


def _distributed_check(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), CUDA_VISIBLE_DEVICES="")
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        for mode, precision in (
            ("native", "native"),
            ("fp32", "native"),
            ("bf16", "native"),
            ("auto", "native"),  # old master=None path: no comm hook
            ("auto", "bf16"),    # old bf16 master path: fp32 comm hook
        ):
            _run_case(rank, mode, precision)
        for precision in ("bf16", "fp32"):
            for cast_frozen in (False, True):
                _run_cast_case(rank, precision, cast_frozen)
        dist.barrier()
        if rank == 0:
            print("OK: two-rank Gloo DDP modes and parameter casting; mixed gradients, frozen parameters and buffers checked")
    finally:
        dist.destroy_process_group()


def _mock_future_check():
    import importlib

    parallelize_module = importlib.import_module("unidrivevla.distributed.parallelize")

    class Future:
        def __init__(self, value):
            self._value = [value]

        def value(self):
            return self._value

        def then(self, callback):
            return Future(callback(self))

    class Work:
        def __init__(self, value):
            self.value = value

        def get_future(self):
            return Future(self.value)

    class Bucket:
        def __init__(self, value):
            self._value = value

        def buffer(self):
            return self._value

    original_world_size = parallelize_module.dist.get_world_size
    original_all_reduce = parallelize_module.dist.all_reduce
    peer = torch.tensor([1.0078125], dtype=torch.float32)
    seen = []

    def fake_all_reduce(value, group=None, async_op=False):
        assert async_op
        seen.append(value.dtype)
        other_rank = peer.to(dtype=value.dtype)
        if value.dtype == torch.float32:
            # fp32_average_hook scales each rank's input before SUM; bf16_average_hook scales afterward.
            other_rank = other_rank.div(2)
        value.add_(other_rank)
        return Work(value)

    parallelize_module.dist.get_world_size = lambda group=None: 2
    parallelize_module.dist.all_reduce = fake_all_reduce
    try:
        fp32_buffer = torch.tensor([1.00390625], dtype=torch.float32)
        fp32_result = parallelize_module.fp32_average_hook(None, Bucket(fp32_buffer))
        torch.testing.assert_close(fp32_result.value()[0], torch.tensor([1.005859375]), rtol=0, atol=0)

        bf16_buffer = torch.tensor([1.00390625], dtype=torch.float32)
        bf16_result = parallelize_module.bf16_average_hook(None, Bucket(bf16_buffer))
        torch.testing.assert_close(bf16_result.value()[0], torch.tensor([1.0]), rtol=0, atol=0)
        assert seen == [torch.float32, torch.bfloat16], seen
    finally:
        parallelize_module.dist.get_world_size = original_world_size
        parallelize_module.dist.all_reduce = original_all_reduce


def _config_check():
    from omegaconf import OmegaConf
    from unidrivevla.train.args import TrainArgs
    from unidrivevla.distributed.parallelize import _effective_ddp_reduce_dtype

    for value in ("auto", "native", "fp32", "bf16"):
        TrainArgs(ddp_reduce_dtype=value).validate(1)
    for value in ("", "fp16", "int8", None):
        try:
            TrainArgs(ddp_reduce_dtype=value).validate(1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid ddp_reduce_dtype accepted: {value!r}")
    for value in (0, 1, None, "false"):
        try:
            TrainArgs(ddp_cast_frozen_parameters=value).validate(1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-boolean ddp_cast_frozen_parameters accepted: {value!r}")
    try:
        TrainArgs(strategy="ddp", precision="native", ddp_cast_frozen_parameters=True).validate(2)
    except ValueError:
        pass
    else:
        raise AssertionError("native precision must reject ddp_cast_frozen_parameters=true")

    assert _effective_ddp_reduce_dtype("auto", legacy_hook_was_enabled=False) == "native"
    assert _effective_ddp_reduce_dtype("auto", legacy_hook_was_enabled=True) == "fp32"
    assert _effective_ddp_reduce_dtype("native", legacy_hook_was_enabled=True) == "native"
    assert _effective_ddp_reduce_dtype("bf16", legacy_hook_was_enabled=False) == "bf16"

    for dataset in ("nuScenes", "Bench2Drive"):
        config = json.loads((ROOT / dataset / "zero_configs" / "adam_zero1_bf16.json").read_text())
        assert config["communication_data_type"] == "bf16"
        assert config["loss_scale"] == 0
        print(f"Main reference {dataset}: communication_data_type={config['communication_data_type']}, "
              f"loss_scale={config['loss_scale']} (config reference only; no DeepSpeed runtime/equivalence claim)")
    _profile_selection_check(TrainArgs, OmegaConf)
    print("OK: argument validation and legacy auto-selection conditions")


def _compose_train(overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_ROOT.resolve()), version_base=None,
                               job_name="check_gradient_reduction"):
        return compose(config_name="train", overrides=overrides)


def _assert_training_recipe(cfg, expected, description, TrainArgs):
    args = TrainArgs.from_cfg(cfg)
    args.validate(world_size=8)
    actual = (args.precision, args.ddp_reduce_dtype, args.ddp_cast_frozen_parameters)
    assert actual == expected, f"{description}: expected {expected}, got {actual}"
    print(f"Config {description}: precision={actual[0]}, reduction={actual[1]}, "
          f"cast_frozen={actual[2]}")


def _profile_selection_check(TrainArgs, OmegaConf):
    native = ("bf16", "bf16", True)
    legacy = ("native", "auto", False)
    cases = (
        (["data=nuscenes"], native, "native nuScenes"),
        (["data=b2d"], native, "native Bench2Drive"),
        (["data=nuscenes", "profile=nuscenes_ego"], legacy, "legacy nuScenes ego"),
        (["data=b2d", "profile=b2d_ego"], legacy, "legacy B2D ego"),
        (["data=navsim", "train_test_split=navtrain"], legacy, "NAVSIM"),
        (["agent=multidomain", "data=multidomain", "train_test_split=navtrain"],
         legacy, "multi-profile default"),
    )
    for overrides, expected, description in cases:
        _assert_training_recipe(_compose_train(overrides), expected, description, TrainArgs)

    # A resumed config can carry an older agent.profile snapshot while the selected top-level profile
    # comes from the newer native preset. TrainArgs must follow agent.profile and its legacy fallbacks.
    native_cfg = _compose_train(["data=nuscenes"])
    native_profile = OmegaConf.to_container(native_cfg.profile, resolve=False)
    old_profile = dict(native_profile)
    for field in ("train_precision", "ddp_reduce_dtype", "ddp_cast_frozen_parameters"):
        old_profile.pop(field, None)
    top_profile_raw = OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / "profile" / "nuscenes.yaml"),
                                             resolve=False)
    old_train_template = OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / "train.yaml").train,
                                                resolve=False)
    resumed_old_metadata = OmegaConf.create({
        "profile": top_profile_raw,
        "agent": {"profile": old_profile},
        "train": old_train_template,
    })
    assert resumed_old_metadata.profile.train_precision == "bf16"
    assert "train_precision" not in resumed_old_metadata.agent.profile
    _assert_training_recipe(resumed_old_metadata, legacy, "old agent.profile over native top-level", TrainArgs)

    # An explicit command-line precision override remains valid for the legacy profile, whose frozen cast stays off.
    _assert_training_recipe(
        _compose_train(["data=nuscenes", "profile=nuscenes_ego", "train.precision=bf16"]),
        ("bf16", "auto", False), "CLI precision override on legacy profile", TrainArgs,
    )


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _config_check()
    _mock_future_check()
    print("OK: asynchronous future-hook oracle, including bf16 rounding difference")
    with tempfile.TemporaryDirectory(prefix="unidrivevla-ddp-reduce-") as directory:
        mp.spawn(_distributed_check, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    main()
