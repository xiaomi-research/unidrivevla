"""CPU checks for stream isolation and eval-pass temporal-state isolation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.adapters import predict
from unidrivevla.models.models.instance_bank import InstanceBank
from unidrivevla.models.models.ego.instance_bank import EgoInstanceBank
from unidrivevla.train.trainer import Trainer


def metadata(streams):
    return {"timestamp": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "img_metas": [{"T_global": np.eye(4), "T_global_inv": np.eye(4),
                           **({"stream_id": value} if value is not None else {})} for value in streams]}


class TemporalQueue(nn.Module):
    """A temporal module with an in-place reset and nested mutable runtime state."""

    def __init__(self):
        super().__init__()
        self.counter = torch.tensor([0.0])
        self.nested = {"history": [torch.tensor([0.0]), {"labels": ["seed"]}]}

    def reset(self):
        self.counter.zero_()
        self.nested["history"][0].zero_()
        self.nested["history"][1]["labels"].clear()

    def training_runtime_state_dict(self):
        return {"counter": self.counter, "nested": self.nested}

    def load_training_runtime_state_dict(self, state, device=None):
        self.counter = state["counter"]
        self.nested = state["nested"]


class TinyHead(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = float(bias)
        self.det_bank = InstanceBank(2, 4, np.zeros((2, 11), np.float32))
        self.ego_bank = EgoInstanceBank(4, feature_map_scale=(4, 4), with_instance_feat=True)
        self.queue = TemporalQueue()

    def reset_temporal(self):
        for module in self.modules():
            if module is not self and hasattr(module, "cached_anchor") and callable(getattr(module, "reset", None)):
                module.reset()

    def training_runtime_state_dict(self):
        return {"det": self.det_bank.training_runtime_state_dict(),
                "ego": self.ego_bank.training_runtime_state_dict(),
                "queue": self.queue.training_runtime_state_dict()}

    def load_training_runtime_state_dict(self, state, device=None):
        self.det_bank.load_training_runtime_state_dict(state["det"], device=device)
        self.ego_bank.load_training_runtime_state_dict(state["ego"], device=device)
        self.queue.load_training_runtime_state_dict(state["queue"], device=device)

    def step(self):
        self.queue.counter.add_(1)
        self.queue.nested["history"][0].add_(1)
        return self.queue.counter.item() + self.bias


class TinyMultiProfilePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({"nuscenes": TinyHead(0), "b2d": TinyHead(0)})
        self.raise_on_forward = False

    @property
    def head(self):
        return self.heads["nuscenes"]

    def training_runtime_state_dict(self):
        return {"profiles": {name: head.training_runtime_state_dict()
                             for name, head in self.heads.items()}}

    def load_training_runtime_state_dict(self, state, device=None):
        for name, value in state["profiles"].items():
            self.heads[name].load_training_runtime_state_dict(value, device=device)

    def forward(self, features, targets=None, tokens=None):
        if self.raise_on_forward:
            raise RuntimeError("intentional prediction failure")
        name = features["profile"]
        score = self.heads[name].step()
        batch = len(tokens) if tokens is not None else 1
        trajectory = torch.full((batch, 2, 3), score, dtype=torch.float32)
        return {"pred_traj": trajectory, "score": torch.tensor(score)}

    def compute_loss(self, features, targets, predictions):
        return predictions["score"]


def seed_live_state(policy):
    for index, name in enumerate(policy.heads):
        head = policy.heads[name]
        base = 10.0 + index * 10.0
        head.det_bank.cached_anchor = torch.tensor([[base]])
        head.det_bank.cached_feature = torch.tensor([[base + 1]])
        head.det_bank.mask = torch.tensor([True])
        head.det_bank.metas = {"timestamp": torch.tensor([base + 2]),
                               "img_metas": [{"stream_id": f"live-{name}"}]}
        head.ego_bank.cached_anchor = torch.tensor([[base + 3]])
        head.ego_bank.cached_feature = torch.tensor([[base + 4]])
        head.ego_bank.metas = {"timestamp": torch.tensor([base + 5]),
                               "img_metas": [{"stream_id": f"live-ego-{name}"}]}
        head.queue.counter = torch.tensor([base + 6])
        head.queue.nested = {"history": [torch.tensor([base + 7]), {"labels": [name, "kept"]}]}


def assert_live_state(policy):
    for index, name in enumerate(policy.heads):
        head = policy.heads[name]
        base = 10.0 + index * 10.0
        torch.testing.assert_close(head.det_bank.cached_anchor, torch.tensor([[base]]))
        torch.testing.assert_close(head.det_bank.cached_feature, torch.tensor([[base + 1]]))
        assert head.det_bank.metas["img_metas"][0]["stream_id"] == f"live-{name}"
        torch.testing.assert_close(head.det_bank.metas["timestamp"], torch.tensor([base + 2]))
        torch.testing.assert_close(head.ego_bank.cached_anchor, torch.tensor([[base + 3]]))
        torch.testing.assert_close(head.ego_bank.cached_feature, torch.tensor([[base + 4]]))
        assert head.ego_bank.metas["img_metas"][0]["stream_id"] == f"live-ego-{name}"
        torch.testing.assert_close(head.queue.counter, torch.tensor([base + 6]))
        torch.testing.assert_close(head.queue.nested["history"][0], torch.tensor([base + 7]))
        assert head.queue.nested["history"][1]["labels"] == [name, "kept"]


def assert_modes(policy, expected):
    actual = {name: module.training for name, module in policy.named_modules()}
    assert actual == expected, (actual, expected)


def make_loader(profiles):
    return [({"profile": profile}, None, [f"{profile}-{index}"])
            for index, profile in enumerate(profiles)]


def trainer_for(policy, loader):
    trainer = object.__new__(Trainer)
    trainer.model = policy
    trainer.agent = policy
    trainer.val_loader = loader
    trainer.ctx = SimpleNamespace(world_size=1, device=torch.device("cpu"),
                                  all_reduce_sum=lambda value: value)
    trainer.ema = None
    return trainer


def prediction_context():
    return SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"),
                           gloo_group=lambda: None)


def local_gather(obj, object_gather_list=None, dst=0, group=None):
    object_gather_list[0] = obj


def check_bank_streams():
    banks = [InstanceBank(2, 4, np.zeros((2, 11), np.float32)),
             EgoInstanceBank(4, feature_map_scale=(4, 4), with_instance_feat=True)]
    for bank in banks:
        for old, new, reset in ((["a", "a"], ["a", "b"], True),
                                (["a", "a"], [None, None], True),
                                (["a", "a"], ["a", "a"], False),
                                ([None, None], [None, None], False)):
            bank.cached_anchor = bank.anchor.detach().unsqueeze(0).repeat(2, 1, 1)
            bank.cached_feature = torch.zeros(2, bank.num_anchor, 4)
            bank.metas = metadata(old)
            args = {"feature_maps": []} if isinstance(bank, EgoInstanceBank) else {}
            output = bank.get(2, metadata(new), **args)
            assert (output[2] is None) == reset, type(bank).__name__
            if not reset:
                assert bank.mask.tolist() == [True, True]


def check_validate_and_predict_isolation():
    policy = TinyMultiProfilePolicy()
    seed_live_state(policy)
    policy.train()
    policy.heads["b2d"].eval()
    policy.heads["nuscenes"].queue.eval()
    modes = {name: module.training for name, module in policy.named_modules()}
    loader = make_loader(["nuscenes", "b2d", "nuscenes"])
    trainer = trainer_for(policy, loader)

    first_loss = trainer.validate()
    assert abs(first_loss - 4 / 3) < 1e-6
    assert_live_state(policy)
    assert_modes(policy, modes)
    second_loss = trainer.validate()
    assert second_loss == first_loss
    assert_live_state(policy)
    assert_modes(policy, modes)

    original_gather = dist.gather_object
    dist.gather_object = local_gather
    try:
        first = predict(policy, loader, prediction_context())
        assert [float(first[f"{name}-{index}"][0, 0])
                for index, name in enumerate(["nuscenes", "b2d", "nuscenes"])] == [1.0, 1.0, 2.0]
        assert_live_state(policy)
        assert_modes(policy, modes)
        second = predict(policy, loader, prediction_context())
        assert first.keys() == second.keys()
        for token in first:
            np.testing.assert_array_equal(first[token], second[token])
        assert_live_state(policy)
        assert_modes(policy, modes)

        policy.raise_on_forward = True
        try:
            predict(policy, loader, prediction_context())
        except RuntimeError as exc:
            assert "intentional prediction failure" in str(exc)
        else:
            raise AssertionError("prediction failure was swallowed")
        assert_live_state(policy)
        assert_modes(policy, modes)
        try:
            trainer.validate()
        except RuntimeError as exc:
            assert "intentional prediction failure" in str(exc)
        else:
            raise AssertionError("validation failure was swallowed")
        assert_live_state(policy)
        assert_modes(policy, modes)
    finally:
        dist.gather_object = original_gather

    print("OK: Trainer.validate and predict isolate all profile caches, preserve nested live state and mixed modes, "
          "repeat deterministically, and restore after prediction errors")


def main():
    check_bank_streams()
    check_validate_and_predict_isolation()
    print("OK: instance and ego caches isolate source changes; single-source behavior is preserved")


if __name__ == "__main__":
    main()
