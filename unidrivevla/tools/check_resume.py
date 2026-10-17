"""CPU check: resume replays sequence/VQA position, RNG, temporal state, and partial gradients."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import socket
import sys
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.data.registry import make_loader  # noqa: E402
from unidrivevla.data.multidomain import MultiProfileDataset  # noqa: E402
from unidrivevla.data.tasks import TaskLoader  # noqa: E402
from unidrivevla.distributed import DistributedContext, ResumeCheckpointer  # noqa: E402
from unidrivevla.train.trainer import Trainer  # noqa: E402


class SequenceDataset(Dataset):
    sequence_sources = [0, 1, 0, 1, 0]
    source_weights = [1.0, 1.5]

    def __len__(self):
        return 10

    def sequences(self):
        return [list(range(i * 2, i * 2 + 2)) for i in range(5)]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        value = float(index) + random.random() + float(np.random.random()) + torch.rand(()).item()
        source = self.sequence_sources[index // 2]
        return ({"x": torch.tensor([value]), "stream_id": f"seq={index // 2}"},
                {"y": torch.tensor([float(index % 3) / 10])}, f"source={source}:{index}")


class ProfileSequenceDataset(Dataset):
    """Small canonical source with consecutive frames and all planning fields expected by the real adapter."""

    def __init__(self, name, size=16):
        self.name, self.size = name, size

    def __len__(self):
        return self.size

    def sequences(self):
        return [list(range(start, min(start + 2, self.size))) for start in range(0, self.size, 2)]

    def __getitem__(self, index):
        sequence = index // 2
        value = float(index) + random.random() + float(np.random.random()) + torch.rand(()).item()
        features = {"x": torch.tensor([value]), "stream_id": f"{self.name}:sequence={sequence}"}
        targets = {
            "y": torch.tensor([float(index % 5) / 7]),
            "gt_bboxes_3d": torch.zeros(1, 7), "gt_labels_3d": torch.zeros(1, dtype=torch.long),
            "gt_agent_fut_trajs": torch.zeros(1, 2, 2), "gt_agent_fut_masks": torch.ones(1, 2),
            "gt_map_pts": torch.zeros(1, 2, 2), "gt_map_labels": torch.zeros(1, dtype=torch.long),
            "box_mask": torch.ones(1, dtype=torch.bool), "map_mask": torch.ones(1, dtype=torch.bool),
        }
        return features, targets, f"{self.name}:sequence={sequence}:frame={index % 2}"


class QADataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"x": torch.tensor([float(index) + random.random() + torch.rand(()).item()]),
                "stream_id": f"vqa:{index}"}


class IndexedQADataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"x": torch.tensor([float(index)])}


class BudgetAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.received = None

    def get_optimizers(self, total_steps=0, steps_per_epoch=0, micro_steps=0, grad_accum=1):
        self.received = (total_steps, steps_per_epoch, micro_steps, grad_accum)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "micro"}}


def check_micro_schedule_budget():
    trainer = Trainer.__new__(Trainer)
    trainer.agent = BudgetAgent()
    trainer.args = SimpleNamespace(max_epochs=5, grad_accum=4, max_steps=3)
    trainer.train_loader = list(range(8))
    trainer.build_optimizer()
    assert trainer.agent.received == (3, 2, 12, 4), trainer.agent.received
    assert trainer.lr_interval == "micro"


class PlanningEpochs:
    dataset = list(range(5))
    sampler = None
    batch_sampler = None

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for index in self.dataset:
            yield {"x": torch.tensor([float(index)])}, {}, None


class TinyAgent(nn.Module):
    def __init__(self, history=None):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.dropout = nn.Dropout(0.25)
        self.temporal_cache = torch.zeros(())
        self.history = history if history is not None else []

    def forward(self, features, targets=None, tokens=None):
        task = features.get("task", "planning")
        x = features["ar_batch"]["x"].float() if task == "vqa" else features["x"].float()
        if "ar_batch" in features and task != "vqa":
            x = x + features["ar_batch"]["x"].float().mean()
        self.history.append((task, features.get("stream_id"), x.detach().clone()))
        self.temporal_cache = (self.temporal_cache * 0.5 + x.mean()).detach()
        prediction = self.linear(self.dropout(x)).squeeze(-1) + self.temporal_cache
        target = torch.zeros_like(prediction) if task == "vqa" else targets["y"].squeeze(-1)
        return SimpleNamespace(loss=(prediction - target).square().mean())

    def training_runtime_state_dict(self):
        return {"temporal_cache": self.temporal_cache}

    def load_training_runtime_state_dict(self, state, device=None):
        self.temporal_cache = state["temporal_cache"].to(device or "cpu")


class HeadRuntimeCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.zeros(()), persistent=False)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long), persistent=False)

    def update(self, value):
        self.value = (self.value * 0.5 + value.detach().mean()).detach()
        self.updates += 1
        return self.value

    def training_runtime_state_dict(self):
        return {"value": self.value.detach().cpu().clone(), "updates": int(self.updates.item())}

    def load_training_runtime_state_dict(self, state, device=None):
        self.value = state["value"].to(device or "cpu")
        self.updates.fill_(int(state["updates"]))


class ResumeProfileHead(nn.Module):
    """Tiny head used through the actual MultiProfilePolicy dispatch and single-profile loss adapter."""

    train_vlm = True

    def __init__(self, profile, runtime, history=None):
        super().__init__()
        self.profile_name = profile.name
        self.private = nn.Parameter(torch.ones(1))
        self.qwen3_vl = nn.Linear(1, 1)
        self.dropout = nn.Dropout(0.25)
        self.runtime_cache = HeadRuntimeCache()
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "history", history if history is not None else [])

    def _inputs(self, values):
        x = values["x"].float()
        stream_id = values.get("stream_id")
        if isinstance(stream_id, (list, tuple)):
            stream_id = tuple(stream_id)
        else:
            stream_id = (stream_id,)
        return x, stream_id

    def forward_ar_batch(self, batch):
        x, stream_id = self._inputs(batch)
        cache = self.runtime_cache.update(x)
        shared = self.runtime.core.qwen3_vl.weight.sum()
        private = self.qwen3_vl.weight.sum() + self.private.sum()
        loss = (x.mean() + cache + shared * 0.01 + private * 0.01).square()
        self.history.append(("vqa", self.profile_name, stream_id, x.detach().clone()))
        return {"loss_ar": loss, "loss_vlm_raw": loss.detach()}

    def forward_train(self, x, y, stream_id, ar_batch=None, **_kwargs):
        x, stream_id = self._inputs({"x": x, "stream_id": stream_id})
        dropped = self.dropout(x).mean()
        cache = self.runtime_cache.update(x)
        shared = self.runtime.core.qwen3_vl.weight.sum()
        private = self.qwen3_vl.weight.sum() + self.private.sum()
        prediction = dropped + cache + shared * 0.01 + private * 0.01
        losses = {"loss_plan": (prediction - y.float().mean()).square()}
        ar_values = None
        if ar_batch is not None:
            ar_values, _ = self._inputs(ar_batch)
            losses["loss_ar"] = self.forward_ar_batch(ar_batch)["loss_ar"]
        self.history.append(("planning", self.profile_name, stream_id, x.detach().clone(),
                             None if ar_values is None else ar_values.detach().clone()))
        return {"losses": losses}


def multidomain_policy(history=None):
    from unidrivevla.adapters.multidomain import MultiProfilePolicy
    from unidrivevla.tools.check_multidomain import TinyCore, Tokenizer, profile

    profiles = {
        "nuscenes": profile("nuscenes", ["<NUSC>"], 2),
        "navsim": profile("navsim", ["<NAVSIM>"], 2),
    }
    model = MultiProfilePolicy(
        profiles, vlm_path="unused", train_vlm=True, exact_resume=True,
        _shared_core_factory=lambda **kwargs: TinyCore(kwargs["view_tokens"]),
        _processor_factory=lambda _path, use_fast: SimpleNamespace(tokenizer=Tokenizer(), use_fast=use_fast),
        _head_factory=lambda selected, runtime: ResumeProfileHead(selected, runtime, history),
    )
    object.__setattr__(model, "history", history if history is not None else [])
    return model


class InterruptingCheckpointer(ResumeCheckpointer):
    def __init__(self, *args, interrupt_when, **kwargs):
        super().__init__(*args, **kwargs)
        self.interrupt_when = interrupt_when

    def save(self, step, model, optimizer, extra, ema=None, rank_state=None, position=None):
        super().save(step, model, optimizer, extra, ema, rank_state, position)
        if self.interrupt_when(extra):
            raise RuntimeError("intentional process stop after a completed checkpoint")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_trainer(ctx, root, initial, seed, mode, save_every, resume=False, interrupt_when=None, history=None,
                 rank=0, world=1, wrap_ddp=False, policy_factory=TinyAgent, planning_factory=None,
                 batch_size=2, grad_accum=3, micro_schedule=False):
    policy = policy_factory(history)
    policy.load_state_dict(initial)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.002, momentum=0.8)
    planning = (planning_factory(rank, world, seed) if planning_factory else make_loader(
        SequenceDataset(), {"batch_size": batch_size, "num_workers": 0, "drop_last": True}, True,
        rank=rank, world=world, sequence_split=1, seed=seed, exact_resume=True))
    vqa = DataLoader(QADataset(), batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    loader = TaskLoader(planning, vqa, mode, exact_resume=True, seed=seed, rank=rank, world=world)
    cfg = OmegaConf.create({"output_dir": str(root), "train": {
        "resume": resume, "exact_resume": True, "max_epochs": 2, "grad_accum": grad_accum,
        "save_every_n_steps": save_every, "async_save": False,
    }})
    trainer = Trainer.__new__(Trainer)
    trainer.cfg, trainer.args, trainer.ctx = cfg, SimpleNamespace(
        exact_resume=True, strategy="ddp", max_epochs=2, grad_accum=grad_accum, max_steps=0,
        grad_clip=0.0, log_every=10000, save_every_n_steps=save_every,
    ), ctx
    trainer.agent = policy
    trainer.model = policy
    if wrap_ddp:
        from unidrivevla.distributed import parallelize
        from unidrivevla.train.args import TrainArgs

        trainer.model = parallelize(policy, TrainArgs(strategy="ddp", precision="native", compile=False,
            ddp_find_unused_parameters=policy_factory is multidomain_policy), ctx)
    if micro_schedule:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1.0 + 0.02 * step))
        interval = "micro"
    else:
        scheduler, interval = None, "step"
    trainer.optimizer, trainer.lr_scheduler, trainer.lr_interval, trainer.ema = optimizer, scheduler, interval, None
    trainer.clip_params = list(policy.parameters())
    trainer.train_loader = loader
    trainer.start_epoch = trainer.resume_batch = trainer.epoch = trainer.global_step = trainer.micro = 0
    trainer.batch_in_epoch = 0
    trainer.best_val = float("inf")
    trainer.global_batch = batch_size * world
    trainer._resume_rank_state = None
    checkpointer_args = (root / "checkpoints", ctx)
    trainer.checkpointer = (InterruptingCheckpointer(*checkpointer_args, async_save=False,
        interrupt_when=interrupt_when) if interrupt_when else ResumeCheckpointer(*checkpointer_args, async_save=False))
    trainer.end_epoch = MethodType(lambda self, epoch, _loss: self.save_resume(epoch, epoch_complete=True), trainer)
    trainer.log_train = MethodType(lambda self, *_args: None, trainer)
    if resume:
        trainer.resume()
    return trainer


def run_to_completion(trainer):
    trainer.train_loop()
    trainer.checkpointer.wait()


def assert_same(expected, actual, label):
    histories_match = tree_equal(expected.agent.history, actual.agent.history)
    if not histories_match:
        for index, (left, right) in enumerate(zip(expected.agent.history, actual.agent.history)):
            if not tree_equal(left, right):
                print("first input mismatch", index, left, right)
                break
        else:
            print("history lengths", len(expected.agent.history), len(actual.agent.history))
        raise AssertionError(f"{label}: input samples or task order differ after resume")
    for name, value in expected.agent.state_dict().items():
        try:
            torch.testing.assert_close(value, actual.agent.state_dict()[name], rtol=0, atol=0, msg=label)
        except AssertionError:
            print("parameter mismatch", name, value, actual.agent.state_dict()[name])
            raise
    if callable(getattr(expected.agent, "training_runtime_state_dict", None)):
        assert tree_equal(expected.agent.training_runtime_state_dict(), actual.agent.training_runtime_state_dict()), label
    elif hasattr(expected.agent, "temporal_cache"):
        torch.testing.assert_close(expected.agent.temporal_cache, actual.agent.temporal_cache, rtol=0, atol=0)
    assert expected.global_step == actual.global_step and expected.micro == actual.micro, label
    assert tree_equal(expected.lr_scheduler.state_dict() if expected.lr_scheduler else None,
                      actual.lr_scheduler.state_dict() if actual.lr_scheduler else None), label
    expected_opt, actual_opt = expected.optimizer.state_dict(), actual.optimizer.state_dict()
    if not tree_equal(expected_opt, actual_opt):
        names = [name for name, _ in expected.agent.named_parameters()]
        ids = expected_opt["param_groups"][0]["params"]
        id_names = dict(zip(ids, names))
        print("optimizer group equal:", tree_equal(expected_opt["param_groups"], actual_opt["param_groups"]))
        print("optimizer state keys:", sorted(expected_opt["state"]), sorted(actual_opt["state"]))
        for param_id in sorted(set(expected_opt["state"]) | set(actual_opt["state"])):
            left, right = expected_opt["state"].get(param_id), actual_opt["state"].get(param_id)
            if not tree_equal(left, right):
                print("optimizer state mismatch", id_names.get(param_id, param_id), left, right)
    assert tree_equal(expected_opt, actual_opt), label
    expected_grads = {name: parameter.grad for name, parameter in expected.agent.named_parameters()}
    actual_grads = {name: parameter.grad for name, parameter in actual.agent.named_parameters()}
    assert tree_equal(expected_grads, actual_grads), f"{label}: accumulated gradients differ"


def tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and
                all(tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(tree_equal(a, b) for a, b in zip(left, right)))
    return left == right


def run_scenario(ctx, root, seed, mode, interrupt_when, save_every, rank=0, world=1, wrap_ddp=False,
                 policy_factory=TinyAgent, planning_factory=None, batch_size=2, grad_accum=3,
                 micro_schedule=False):
    seed_all(seed)
    initial_agent = policy_factory(None)
    initial = {name: value.clone() for name, value in initial_agent.state_dict().items()}

    rng_seed = seed + 10 + rank * 1000
    seed_all(rng_seed)
    full_history = []
    interrupted_history = []
    uninterrupted = make_trainer(ctx, root / "full", initial, seed, mode, save_every, history=full_history,
                                 rank=rank, world=world, wrap_ddp=wrap_ddp, policy_factory=policy_factory,
                                 planning_factory=planning_factory, batch_size=batch_size, grad_accum=grad_accum,
                                 micro_schedule=micro_schedule)
    seed_all(rng_seed)
    run_to_completion(uninterrupted)

    seed_all(rng_seed)
    first = make_trainer(ctx, root / "interrupted", initial, seed, mode, save_every,
                         interrupt_when=interrupt_when, history=interrupted_history,
                         rank=rank, world=world, wrap_ddp=wrap_ddp, policy_factory=policy_factory,
                         planning_factory=planning_factory, batch_size=batch_size, grad_accum=grad_accum,
                         micro_schedule=micro_schedule)
    seed_all(rng_seed)
    try:
        first.train_loop()
    except RuntimeError as exc:
        assert str(exc) == "intentional process stop after a completed checkpoint"
    else:
        raise AssertionError("the configured checkpoint interruption did not happen")
    first.checkpointer.wait()

    seed_all(rng_seed + 999)
    resumed = make_trainer(ctx, root / "interrupted", initial, seed, mode, save_every, resume=True,
                           history=interrupted_history, rank=rank, world=world, wrap_ddp=wrap_ddp,
                           policy_factory=policy_factory, planning_factory=planning_factory, batch_size=batch_size,
                           grad_accum=grad_accum, micro_schedule=micro_schedule)
    run_to_completion(resumed)
    assert_same(uninterrupted, resumed, f"{mode} interrupted training matches uninterrupted training")
    if policy_factory is multidomain_policy:
        planning = [item for item in uninterrupted.agent.history if item[0] == "planning"]
        assert {item[1] for item in planning} == {"nuscenes", "navsim"}
        assert all(item[2] and all("sequence=" in value for value in item[2]) for item in planning)
        assert any(item[0] == "vqa" for item in uninterrupted.agent.history)
    if micro_schedule:
        assert first.micro % first.args.grad_accum != 0
        assert first.lr_scheduler.last_epoch == first.micro
        assert resumed.lr_scheduler.last_epoch == resumed.micro
    return resumed


def distributed_worker(rank, rendezvous, root):
    os.environ.update(CUDA_VISIBLE_DEVICES="", RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    ctx = DistributedContext()
    try:
        run_scenario(ctx, Path(root) / "ddp", 31, "alternate",
                     lambda extra: extra["global_step"] == 1 and not extra["epoch_complete"], 1,
                     rank=rank, world=2, wrap_ddp=True)
        run_scenario(ctx, Path(root) / "ddp-multidomain", 37, "combined",
                     lambda extra: extra["global_step"] == 1 and not extra["epoch_complete"], 1,
                     rank=rank, world=2, wrap_ddp=True, policy_factory=multidomain_policy,
                     planning_factory=multidomain_planning_loader(1), batch_size=1, grad_accum=1)
        ctx.barrier()
        if rank == 0:
            print("ok   two-rank DDP resumes per-rank RNG, two profile heads, source batches, and VQA")
    finally:
        ctx.close()


def check_worker_guards():
    try:
        make_loader(QADataset(), {"batch_size": 2, "num_workers": 1}, True, 0, 1, exact_resume=True)
    except ValueError as exc:
        assert "num_workers=0" in str(exc)
    else:
        raise AssertionError("exact resume must reject planning workers")
    mixed = MultiProfileDataset([ProfileSequenceDataset("nuscenes"), ProfileSequenceDataset("navsim")],
                                ["nuscenes", "navsim"], [1.0, 1.0])
    augmented, _, _ = mixed[(0, {"flip": True})]
    assert augmented["profile"] == "nuscenes" and augmented["augment"] == {"flip": True}
    try:
        make_loader(mixed, {"batch_size": 2, "num_workers": 1}, True, 0, 1,
                    sequence_split=1, exact_resume=True)
    except ValueError as exc:
        assert "num_workers=0" in str(exc)
    else:
        raise AssertionError("multidomain exact resume must reject planning workers")
    planning = make_loader(QADataset(), {"batch_size": 2, "num_workers": 0}, True, 0, 1)
    vqa = DataLoader(QADataset(), batch_size=2, num_workers=1, drop_last=True)
    try:
        TaskLoader(planning, vqa, exact_resume=True)
    except ValueError as exc:
        assert "vqa.num_workers=0" in str(exc)
    else:
        raise AssertionError("exact resume must reject VQA workers")


def check_profile_source_of_truth():
    from unidrivevla.data import registry
    from unidrivevla.data.multidomain import MultiProfileDataset, _profiles
    from unidrivevla.data.registry import LOADERS

    model = multidomain_policy()
    actual_profiles = model.profiles
    cfg_profiles = {name: {**profile.__dict__, "frame": "lidar"}
                    for name, profile in actual_profiles.items()}
    cfg = OmegaConf.create({"data_loader": "multidomain", "profiles": cfg_profiles, "multidomain": {
        "sources": [{"name": name, "loader": name} for name in actual_profiles],
    }})
    assert _profiles(cfg, SimpleNamespace(profiles=actual_profiles)) is actual_profiles
    assert _profiles(cfg, SimpleNamespace())["nuscenes"].frame == "lidar"
    looked_up = _profiles(cfg, SimpleNamespace(profile_for_source=lambda name: actual_profiles[name]))
    assert all(looked_up[name] is profile for name, profile in actual_profiles.items())

    originals = {name: LOADERS[name] for name in actual_profiles}
    received = {}

    def source_loader(source_cfg, source_agent, _split):
        name = source_agent.profile.name
        received[name] = (source_cfg.profile.frame, source_agent.profile.frame)
        return ProfileSequenceDataset(name, size=2)

    LOADERS.update({name: source_loader for name in actual_profiles})
    try:
        split = registry.build_split(cfg, model, "train")
        assert isinstance(split.dataset, MultiProfileDataset)
    finally:
        LOADERS.update(originals)
    assert received == {name: (profile.frame, profile.frame) for name, profile in actual_profiles.items()}


def check_continuous_vqa_cursor():
    from torch.utils.data import DistributedSampler

    seed = 41

    def collect(exact_resume):
        dataset = IndexedQADataset()
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, drop_last=True, seed=seed + 1)
        vqa = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0, drop_last=True)
        loader = TaskLoader(PlanningEpochs(), vqa, exact_resume=exact_resume, seed=seed, rank=0, world=1)
        observed = []
        for _ in range(2):
            for features, _, _ in loader:
                observed.extend(features["ar_batch"]["x"].flatten().tolist())
        return observed, loader.state_dict()

    default_order, default_state = collect(False)
    exact_order, exact_state = collect(True)
    assert default_order == exact_order
    assert default_state["vqa_batches_seen"] == exact_state["vqa_batches_seen"] == 10
    assert default_state["vqa_cycle"] == exact_state["vqa_cycle"]
    assert default_state["epoch_vqa_batches_seen"] == exact_state["epoch_vqa_batches_seen"] == 5


def multidomain_planning_loader(batch_size):
    def build(rank, world, seed):
        dataset = MultiProfileDataset(
            [ProfileSequenceDataset("nuscenes"), ProfileSequenceDataset("navsim")],
            ["nuscenes", "navsim"], [1.0, 1.3])
        return make_loader(dataset, {"batch_size": batch_size, "num_workers": 0, "drop_last": True}, True,
                           rank=rank, world=world, sequence_split=1, seed=seed, exact_resume=True)

    return build


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    ctx = DistributedContext()
    try:
        check_micro_schedule_budget()
        print("ok   Trainer caps micro-step schedule to max_steps times grad_accum")
        check_worker_guards()
        print("ok   exact resume rejects planning, multidomain, and VQA worker prefetch")
        check_profile_source_of_truth()
        print("ok   inference uses checkpoint-restored profile metadata, with config fallback for mocks")
        check_continuous_vqa_cursor()
        print("ok   exact and default VQA cursors match across planning-epoch boundaries")
        with tempfile.TemporaryDirectory(prefix="uni-resume-") as temp:
            root = Path(temp)
            run_scenario(ctx, root / "alternate", 17, "alternate",
                         lambda extra: extra["global_step"] == 1 and not extra["epoch_complete"], 1)
            print("ok   alternate VQA resume replays the saved mid-epoch task position and RNG")
            run_scenario(ctx, root / "combined", 23, "combined",
                         lambda extra: extra["epoch"] == 1 and extra["epoch_complete"], 0)
            print("ok   epoch-boundary resume restores the partial accumulation gradient and temporal state")
            run_scenario(ctx, root / "micro-lr", 29, "combined",
                         lambda extra: extra["epoch"] == 1 and extra["epoch_complete"], 0,
                         micro_schedule=True)
            print("ok   actual Trainer resumes a micro-step LR scheduler after an epoch-ending nonsync micro")
            run_scenario(ctx, root / "multidomain-mid", 43, "combined",
                         lambda extra: extra["global_step"] == 1 and not extra["epoch_complete"], 1,
                         policy_factory=multidomain_policy,
                         planning_factory=multidomain_planning_loader(2), batch_size=2, grad_accum=1)
            print("ok   multidomain mid-epoch resume matches source sequence, task, VQA, cache, gradient, and optimizer state")
            run_scenario(ctx, root / "multidomain-epoch", 47, "combined",
                         lambda extra: extra["epoch"] == 1 and extra["epoch_complete"], 0,
                         policy_factory=multidomain_policy,
                         planning_factory=multidomain_planning_loader(2), batch_size=2)
            print("ok   multidomain epoch-boundary resume restores combined VQA, profile caches, and partial gradients")
        with tempfile.TemporaryDirectory(prefix="uni-resume-ddp-") as temp:
            rendezvous = str(Path(temp) / "rendezvous")
            mp.spawn(distributed_worker, args=(rendezvous, temp), nprocs=2, join=True)
    finally:
        ctx.close()
    print("OK")


if __name__ == "__main__":
    main()
