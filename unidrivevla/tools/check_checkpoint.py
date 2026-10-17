"""python unidrivevla/tools/check_checkpoint.py: CPU round trips for stage-2 frozen weights and resume state."""

import socket
import sys
import tempfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.adapters import initialize_for_inference  # noqa: E402
from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy  # noqa: E402
from unidrivevla.distributed.checkpoint import DONE_FILE, ResumeCheckpointer, export_model_state, training_state  # noqa: E402
from unidrivevla.train.trainer import Trainer, _compute_parameter_dtypes  # noqa: E402
from unidrivevla.train.optim import MasterAdamW  # noqa: E402


class TinyPolicy(nn.Module):
    initialize = UniDriveVLAPolicy.initialize

    def __init__(self, checkpoint_path=None, stage2=False):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.vlm = nn.Linear(2, 2, bias=False)
        self.lora = nn.Linear(2, 2, bias=False)
        self.plan = nn.Linear(2, 1, bias=False)
        if stage2:
            self.motion = nn.Linear(2, 1)
            self.vlm.requires_grad_(False)
            self.lora.requires_grad_(False)


class ComputeDtypePolicy(nn.Module):
    initialize = UniDriveVLAPolicy.initialize

    def __init__(self, checkpoint_path=None, dtype=torch.float32, frozen_dtype=None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.primary = nn.Parameter(torch.tensor([1.25, -0.75], dtype=dtype))
        self.tied_alias = self.primary
        self.frozen = nn.Parameter(torch.tensor([0.5, 2.0], dtype=frozen_dtype or dtype), requires_grad=False)
        self.frozen_integer = nn.Parameter(torch.tensor(3, dtype=torch.int64), requires_grad=False)
        self.register_buffer("float_buffer", torch.tensor([3.0], dtype=torch.float32))
        self.register_buffer("integer_buffer", torch.tensor([7], dtype=torch.int64))

    def forward(self):
        values = torch.tensor([0.25, -1.5], dtype=self.primary.dtype)
        return (self.primary * values).sum() + (self.frozen * values).sum() + self.float_buffer[0]


class BranchPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2)
        self.heads = nn.ModuleDict({"nuscenes": nn.Linear(2, 2), "navsim": nn.Linear(2, 2)})

    def forward(self, profile, value):
        return (self.shared(value) + self.heads[profile](value)).square().sum()


def optimizer_states_by_name(model, optimizer):
    return {name: {key: value.detach().clone() if torch.is_tensor(value) else value
                   for key, value in optimizer.state[parameter].items()}
            for name, parameter in model.named_parameters() if optimizer.state.get(parameter)}


def tree_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def branch_step(model, optimizer, profile):
    optimizer.zero_grad(set_to_none=True)
    value = torch.tensor([[0.25, -0.75], [1.5, 0.5]], dtype=torch.float32)
    model(profile, value).backward()
    optimizer.step()


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"ok   {message}")


def start_cpu_process_group():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


class Context:
    is_main = True
    rank = 0
    world_size = 1

    def barrier(self):
        pass

    def gloo_group(self):
        if not hasattr(self, "group"):
            self.group = dist.new_group(backend="gloo")
        return self.group


def check_sparse_optimizer_resume(root, optimizer_name):
    def make_optimizer(model):
        if optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
        return MasterAdamW(model.parameters(), lr=0.002, weight_decay=0.01)

    torch.manual_seed(31)
    initial = BranchPolicy().state_dict()
    uninterrupted, interrupted = BranchPolicy(), BranchPolicy()
    uninterrupted.load_state_dict(initial)
    interrupted.load_state_dict(initial)
    continuous_optimizer, saved_optimizer = make_optimizer(uninterrupted), make_optimizer(interrupted)
    branch_step(uninterrupted, continuous_optimizer, "nuscenes")
    branch_step(interrupted, saved_optimizer, "nuscenes")

    # MasterAdamW can create zero-valued state for no-gradient parameters during step();
    # remove those untouched-profile placeholders to model a genuinely cold branch.
    for model, optimizer in ((uninterrupted, continuous_optimizer), (interrupted, saved_optimizer)):
        for name, parameter in model.named_parameters():
            if name.startswith("heads.navsim."):
                optimizer.state.pop(parameter, None)
    saved_states = optimizer_states_by_name(interrupted, saved_optimizer)
    check(saved_states and all(name.startswith(("shared.", "heads.nuscenes.")) for name in saved_states),
          f"{optimizer_name} state remains sparse before the navsim branch is used")

    checkpointer = ResumeCheckpointer(root / f"sparse-{optimizer_name}", Context(), async_save=False)
    checkpointer.save(1, interrupted, saved_optimizer, {"epoch": 0, "global_step": 1, "best_val": 0.0})
    checkpoint = checkpointer.latest()
    metadata = FileSystemReader(str(checkpoint)).read_metadata()
    saved_optimizer_keys = {key for key in metadata.state_dict_metadata if key.startswith("optim.state.")}
    check(not any(key.startswith("optim.state.heads.navsim.") for key in saved_optimizer_keys),
          f"{optimizer_name} DCP metadata omits cold navsim slots")

    resumed = BranchPolicy()
    resumed.load_state_dict(initial)
    resumed_optimizer = make_optimizer(resumed)
    checkpointer.load(checkpoint, resumed, resumed_optimizer)
    restored_states = optimizer_states_by_name(resumed, resumed_optimizer)
    check(tree_equal(saved_states, restored_states), f"{optimizer_name} restores only source optimizer slots")
    check(not any(name.startswith("heads.navsim.") for name in restored_states),
          f"{optimizer_name} leaves the unused profile optimizer state uninitialized")
    check(not any(name.startswith("heads.navsim.") and parameter in resumed_optimizer.state
                  for name, parameter in resumed.named_parameters()),
          f"{optimizer_name} does not persist empty cold-profile placeholders")

    branch_step(uninterrupted, continuous_optimizer, "navsim")
    branch_step(resumed, resumed_optimizer, "navsim")
    check(tree_equal(uninterrupted.state_dict(), resumed.state_dict()),
          f"{optimizer_name} first-use branch weights match uninterrupted training bitwise")
    check(tree_equal(optimizer_states_by_name(uninterrupted, continuous_optimizer),
                     optimizer_states_by_name(resumed, resumed_optimizer)),
          f"{optimizer_name} first-use optimizer state matches uninterrupted training bitwise")
    if optimizer_name == "master_adamw":
        states = optimizer_states_by_name(resumed, resumed_optimizer)
        check(states["shared.weight"]["step"].item() == 2
              and states["heads.nuscenes.weight"]["step"].item() == 1
              and states["heads.navsim.weight"]["step"].item() == 1,
              "MasterAdamW preserves actual shared/profile step counts across restore")


def check_inference_compute_dtypes(root):
    from types import SimpleNamespace

    source = ComputeDtypePolicy(dtype=torch.bfloat16)
    with torch.no_grad():
        source.primary.add_(torch.tensor([0.125, -0.25], dtype=torch.bfloat16))
        source.frozen.copy_(torch.tensor([-1.25, 0.375], dtype=torch.bfloat16))
    checkpoint = root / "bf16-compute.ckpt"
    trainer = Trainer.__new__(Trainer)
    trainer.agent = source
    trainer.args = SimpleNamespace(strategy="ddp", precision="native")
    trainer.write_ckpt(source.state_dict(), checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check(saved["compute_parameter_dtypes"] == {"primary": "bfloat16", "frozen": "bfloat16"},
          "training checkpoint records actual BF16 compute dtypes once per tied parameter")

    restored = ComputeDtypePolicy(checkpoint_path=str(checkpoint))
    parameter_identity, tied_identity = id(restored.primary), id(restored.tied_alias)
    initialize_for_inference(restored)
    check(restored.primary.dtype == torch.bfloat16 and restored.frozen.dtype == torch.bfloat16,
          "strict inference initialization restores trainable and frozen BF16 parameter dtypes")
    check(torch.equal(source(), restored()), "BF16 training and inference outputs match bitwise after round trip")
    check(id(restored.primary) == parameter_identity == id(restored.tied_alias) == tied_identity,
          "dtype restoration preserves Parameter identity and tied aliases")
    check(restored.float_buffer.dtype == torch.float32 and torch.equal(restored.float_buffer, source.float_buffer)
          and restored.integer_buffer.dtype == torch.int64,
          "inference dtype restoration leaves floating and integer buffers unchanged")

    legacy = root / "legacy-no-compute-dtypes.ckpt"
    torch.save({"state_dict": source.state_dict()}, legacy)
    legacy_target = ComputeDtypePolicy(checkpoint_path=str(legacy))
    initialize_for_inference(legacy_target)
    check(legacy_target.primary.dtype == torch.float32 and legacy_target.frozen.dtype == torch.float32,
          "legacy checkpoints without dtype metadata retain historical inference behavior")

    invalid = root / "invalid-compute-dtype.ckpt"
    torch.save({"state_dict": source.state_dict(), "compute_parameter_dtypes": {
        "primary": "bfloat16", "frozen": "arbitrary"}}, invalid)
    invalid_target = ComputeDtypePolicy(checkpoint_path=str(invalid))
    try:
        initialize_for_inference(invalid_target)
    except ValueError as exc:
        check("unsupported" in str(exc), "inference rejects non-allowlisted dtype names")
    else:
        raise AssertionError("inference accepted an arbitrary dtype name")
    check(invalid_target.primary.dtype == torch.float32,
          "invalid dtype maps are rejected before any parameter is cast")

    incomplete = root / "incomplete-compute-dtypes.ckpt"
    torch.save({"state_dict": source.state_dict(), "compute_parameter_dtypes": {"primary": "bfloat16"}},
               incomplete)
    try:
        initialize_for_inference(ComputeDtypePolicy(checkpoint_path=str(incomplete)))
    except ValueError as exc:
        check("does not match" in str(exc), "inference rejects partial dtype maps")
    else:
        raise AssertionError("inference accepted a partial dtype map")

    fsdp_source = ComputeDtypePolicy(dtype=torch.float32, frozen_dtype=torch.float64)
    fsdp_args = SimpleNamespace(strategy="fsdp", precision="bf16", replicate_frozen=["frozen"])
    integer_before = fsdp_source.frozen_integer.detach().clone()
    fsdp_dtypes = _compute_parameter_dtypes(fsdp_source, fsdp_args)
    check(fsdp_dtypes == {"primary": "bfloat16", "frozen": "float64"},
          "FSDP dtype metadata uses execution precision for managed params and source dtype for replicated floats")
    check(fsdp_source.frozen_integer.dtype == torch.int64
          and torch.equal(fsdp_source.frozen_integer, integer_before)
          and "frozen_integer" not in fsdp_dtypes,
          "replicated nonfloating scalar parameters stay outside the compute dtype map")

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

    wrapped = ComputeDtypePolicy(dtype=torch.bfloat16)
    wrapped.primary_layer = checkpoint_wrapper(nn.Linear(2, 2, dtype=torch.float32),
                                                checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    wrapped_dtypes = _compute_parameter_dtypes(wrapped, SimpleNamespace(strategy="ddp", precision="native"))
    check("primary_layer.weight" in wrapped_dtypes
          and all("_checkpoint_wrapped_module" not in name for name in wrapped_dtypes),
          "dtype metadata canonicalizes actual activation-checkpoint wrapper parameter names")

    legacy_trainer = Trainer.__new__(Trainer)
    legacy_trainer.cfg = None
    legacy_payload_path = root / "legacy-trainer-payload.ckpt"
    legacy_trainer.write_ckpt({"sentinel": torch.tensor(1)}, legacy_payload_path)
    legacy_payload = torch.load(legacy_payload_path, map_location="cpu", weights_only=True)
    check(set(legacy_payload) == {"state_dict"},
          "Trainer.__new__ without agent or args keeps the historical checkpoint payload")


def main():
    torch.manual_seed(7)
    check(not torch.cuda.is_initialized(), "all model and checkpoint work starts on CPU")
    start_cpu_process_group()
    try:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            check_sparse_optimizer_resume(root, "sgd")
            check_sparse_optimizer_resume(root, "master_adamw")
            check_inference_compute_dtypes(root)
            stage1 = TinyPolicy()
            with torch.no_grad():
                stage1.vlm.weight.fill_(1.25)
                stage1.lora.weight.fill_(-0.75)
                stage1.plan.weight.fill_(0.5)
            stage1_path = root / "stage1.ckpt"
            torch.save({"state_dict": export_model_state(stage1)}, stage1_path)

            stage2 = TinyPolicy(checkpoint_path=str(stage1_path), stage2=True)
            stage2.initialize()
            check(torch.equal(stage2.vlm.weight, stage1.vlm.weight), "stage 2 initializes stage-1 VLM weights")
            check(torch.equal(stage2.lora.weight, stage1.lora.weight), "stage 2 initializes stage-1 LoRA weights")
            check(not stage2.vlm.weight.requires_grad and not stage2.lora.weight.requires_grad,
                  "stage-1 VLM and LoRA are frozen in stage 2")

            with torch.no_grad():
                stage2.plan.weight.add_(2.0)
                stage2.motion.weight.fill_(3.0)
            stage2_path = root / "stage2.ckpt"
            torch.save({"state_dict": export_model_state(stage2)}, stage2_path)
            restored = TinyPolicy(checkpoint_path=str(stage2_path), stage2=True)
            restored.initialize()
            check(all(torch.equal(value, restored.state_dict()[name])
                      for name, value in stage2.state_dict().items()),
                  "actual policy initialize restores all stage-2 weights, including frozen VLM and LoRA")
            restored.initialize(strict=True)
            try:
                TinyPolicy(stage2=True).initialize(strict=True)
            except ValueError:
                pass
            else:
                raise AssertionError("strict inference must not run random weights without a checkpoint")
            for omitted in ("vlm.weight", "lora.weight", "plan.weight", "motion.weight"):
                incomplete = root / f"missing-{omitted.replace('.', '-')}.ckpt"
                torch.save({"state_dict": {name: value for name, value in stage2.state_dict().items()
                                           if name != omitted}}, incomplete)
                try:
                    TinyPolicy(checkpoint_path=str(incomplete), stage2=True).initialize()
                except KeyError as exc:
                    check(omitted in str(exc), f"training initialization rejects missing {omitted}")
                else:
                    raise AssertionError(f"training initialization silently accepted missing {omitted}")

            bad_motion_meta = root / "bad-motion-meta.ckpt"
            torch.save({"state_dict": {name: value for name, value in stage2.state_dict().items()
                                       if not name.startswith("motion.")},
                        "agent_config": {"with_motion": True}}, bad_motion_meta)
            try:
                TinyPolicy(checkpoint_path=str(bad_motion_meta), stage2=True).initialize()
            except ValueError as exc:
                check("motion metadata" in str(exc), "checkpoint motion metadata cannot bless a truncated stage-2 state")
            else:
                raise AssertionError("metadata made a checkpoint with every motion tensor removed look complete")

            optimizer = torch.optim.AdamW((p for p in stage2.parameters() if p.requires_grad), lr=1e-3)
            loss = sum(p.square().sum() for p in stage2.parameters() if p.requires_grad)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            saved = training_state(stage2, optimizer)
            check(set(saved["model"]) == set(stage2.state_dict()),
                  "resume model state includes frozen and trainable tensors")

            checkpointer = ResumeCheckpointer(root / "resume", Context(), async_save=False)
            checkpointer.save(1, stage2, optimizer, {"epoch": 1, "global_step": 1, "best_val": 0.0})
            resume_path = checkpointer.latest()
            saved_extra = torch.load(resume_path / DONE_FILE, map_location="cpu", weights_only=False)
            check(saved_extra["model_state_complete"], "new resume metadata marks the full model state")
            resumed = TinyPolicy(stage2=True)
            resumed_optimizer = torch.optim.AdamW(
                (p for p in resumed.parameters() if p.requires_grad), lr=1e-3)
            checkpointer.load(checkpointer.latest(), resumed, resumed_optimizer)
            check(all(torch.equal(value, resumed.state_dict()[name])
                      for name, value in stage2.state_dict().items()),
                  "DCP resume restores learned frozen weights and trainable weights")

            async_checkpointer = ResumeCheckpointer(root / "async_resume", Context(), async_save=True)
            async_checkpointer.save(2, stage2, optimizer, {"epoch": 2, "global_step": 2, "best_val": 0.0})
            check(async_checkpointer.latest() is None, "async checkpoint stays incomplete until wait finalizes metadata")
            async_checkpointer.wait()
            async_restored = TinyPolicy(stage2=True)
            async_optimizer = torch.optim.AdamW(
                (p for p in async_restored.parameters() if p.requires_grad), lr=1e-3)
            async_checkpointer.load(async_checkpointer.latest(), async_restored, async_optimizer)
            check(all(torch.equal(value, async_restored.state_dict()[name])
                      for name, value in stage2.state_dict().items()),
                  "async DCP round trip completes data before publishing its marker")

            failed_future_checkpointer = ResumeCheckpointer(root / "failed_async", Context(), async_save=True)
            failed_future = Future()
            failed_future.set_exception(CheckpointException("injected async data failure", {}))
            with patch.object(dcp, "async_save", return_value=failed_future):
                failed_future_checkpointer.save(3, stage2, optimizer,
                                                {"epoch": 3, "global_step": 3, "best_val": 0.0})
            try:
                failed_future_checkpointer.wait()
            except RuntimeError as exc:
                check(isinstance(exc.__cause__, CheckpointException), "async DCP CheckpointException propagates from wait")
            else:
                check(False, "async DCP CheckpointException propagates from wait")
            check(failed_future_checkpointer.latest() is None, "failed async data write never gets a completion marker")

            failed_start_checkpointer = ResumeCheckpointer(root / "failed_async_start", Context(), async_save=True)
            with patch.object(dcp, "async_save",
                              side_effect=CheckpointException("injected async start failure", {})):
                failed_start_checkpointer.save(5, stage2, optimizer,
                                               {"epoch": 5, "global_step": 5, "best_val": 0.0})
            try:
                failed_start_checkpointer.wait()
            except RuntimeError as exc:
                check(isinstance(exc.__cause__, CheckpointException), "async start CheckpointException propagates")
            else:
                check(False, "async start CheckpointException propagates")
            check(failed_start_checkpointer.latest() is None,
                  "async start CheckpointException never gets a completion marker")

            failed_sync_checkpointer = ResumeCheckpointer(root / "failed_sync", Context(), async_save=False)
            with patch.object(dcp, "save", side_effect=CheckpointException("injected sync write failure", {})):
                try:
                    failed_sync_checkpointer.save(6, stage2, optimizer,
                                                  {"epoch": 6, "global_step": 6, "best_val": 0.0})
                except RuntimeError as exc:
                    check(isinstance(exc.__cause__, CheckpointException), "sync DCP CheckpointException propagates")
                else:
                    check(False, "sync DCP CheckpointException propagates")
            check(failed_sync_checkpointer.latest() is None,
                  "sync DCP CheckpointException never gets a completion marker")

            failed_marker_checkpointer = ResumeCheckpointer(root / "failed_marker", Context(), async_save=True)
            failed_marker_checkpointer.save(4, stage2, optimizer,
                                            {"epoch": 4, "global_step": 4, "best_val": 0.0})
            def fail_marker(_path, _extra):
                raise OSError("injected completion marker failure")
            failed_marker_checkpointer._write_done = fail_marker
            try:
                failed_marker_checkpointer.wait()
            except RuntimeError as exc:
                check("injected completion marker failure" in str(exc), "marker write errors propagate from wait")
            else:
                check(False, "marker write errors propagate from wait")
            check(failed_marker_checkpointer.latest() is None,
                  "failed marker write never makes an incomplete checkpoint visible")

            model_sd, optim_sd = get_state_dict(
                stage2, optimizer, options=StateDictOptions(ignore_frozen_params=True))
            legacy_path = root / "legacy_resume"
            dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(legacy_path),
                     process_group=Context().gloo_group())
            torch.save({"epoch": 1, "global_step": 1, "best_val": 0.0, "ema": False},
                       legacy_path / DONE_FILE)
            legacy_target = TinyPolicy(checkpoint_path=str(stage1_path), stage2=True)
            legacy_target.initialize()
            legacy_optimizer = torch.optim.AdamW(
                (p for p in legacy_target.parameters() if p.requires_grad), lr=1e-3)
            checkpointer.load(legacy_path, legacy_target, legacy_optimizer)
            check(torch.equal(legacy_target.vlm.weight, stage1.vlm.weight)
                  and torch.equal(legacy_target.lora.weight, stage1.lora.weight),
                  "pre-fix DCP resume keeps frozen VLM and LoRA from its stage-1 initialization")

            corrupt_path = root / "corrupt_marked_resume"
            dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(corrupt_path),
                     process_group=Context().gloo_group())
            torch.save({"epoch": 1, "global_step": 1, "best_val": 0.0, "ema": False,
                        "model_state_complete": True}, corrupt_path / DONE_FILE)
            corrupt_target = TinyPolicy(stage2=True)
            corrupt_optimizer = torch.optim.AdamW(
                (p for p in corrupt_target.parameters() if p.requires_grad), lr=1e-3)
            try:
                checkpointer.load(corrupt_path, corrupt_target, corrupt_optimizer)
            except CheckpointException:
                check(True, "marked complete resume with missing frozen weights fails without fallback")
            else:
                check(False, "marked complete resume with missing frozen weights fails without fallback")
    finally:
        dist.destroy_process_group()
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}")
        sys.exit(1)
