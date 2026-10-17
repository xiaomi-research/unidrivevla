"""CPU check of FSDP evaluation schedule signatures; this does not run FSDP forwards."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FSDPModule

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.distributed.parallelize import (_eval_forward_signature, _require_matching_eval_schedules,
                                               check_eval_loader)


class Loader:
    def __init__(self, batch_count, schedule=None):
        self.batch_count = batch_count
        self.batch_sampler = type("Sampler", (), {"fsdp_eval_schedule": schedule})()

    def __len__(self):
        return self.batch_count


class GuardOnlyFSDP(FSDPModule, torch.nn.Module):
    """Exercise only the collective guard; no parameters are sharded and no forward is run."""


def rank_check(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        ctx = SimpleNamespace(world_size=2, gloo_group=lambda: dist.group.WORLD)
        # FSDPModule.__new__ deliberately constructs the original class; fully_shard assigns the mixed class.
        model = torch.nn.Module()
        model.__class__ = GuardOnlyFSDP
        assert isinstance(model, FSDPModule)
        check_eval_loader(model, Loader(2), ctx)
        for loader in (Loader(2 + rank), Loader(5, ((0, "a", 2 + rank), (1, "b", 3 - rank)))):
            try:
                check_eval_loader(model, loader, ctx)
            except RuntimeError:
                pass
            else:
                raise AssertionError("all ranks must reject unequal schedules before forward")
    finally:
        dist.destroy_process_group()


def main():
    assert _eval_forward_signature(Loader(4)) == ("batch_count", 4)
    domain_order = ((0, "nuscenes", 2), (1, "b2d", 3))
    same_order = ((0, "nuscenes", 2), (1, "b2d", 3))
    _require_matching_eval_schedules([_eval_forward_signature(Loader(5, domain_order)),
                                      _eval_forward_signature(Loader(5, same_order))])
    for schedule in (
        ((0, "nuscenes", 3), (1, "b2d", 2)),  # same total, different per-source counts
        ((0, "b2d", 2), (1, "nuscenes", 3)),  # same counts, different domain order
    ):
        try:
            _require_matching_eval_schedules([_eval_forward_signature(Loader(5, domain_order)),
                                              _eval_forward_signature(Loader(5, schedule))])
        except RuntimeError as exc:
            assert "Use train.strategy=ddp" in str(exc)
        else:
            raise AssertionError("unequal FSDP evaluation schedules must be rejected")
    try:
        _require_matching_eval_schedules([("batch_count", 4), ("batch_count", 5)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("unequal single-source forward counts must be rejected")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(rank_check, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)
    print("OK: equal FSDP eval signatures pass; unequal counts/order reject before forward")


if __name__ == "__main__":
    main()
