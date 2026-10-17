"""CPU check: combined batches preserve the summed objective; alternate batches keep a rank-independent order."""

import sys
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.data.tasks import TaskLoader


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.plan = torch.nn.Parameter(torch.ones(1))
        self.vqa = torch.nn.Parameter(torch.ones(1))

    def forward(self, features, targets=None, tokens=None):
        if features.get("task") == "vqa":
            loss = self.vqa.sum() * features["ar_batch"]["x"] * 0.25
        else:
            loss = self.plan.sum() * features["x"]
            if "ar_batch" in features:
                loss = loss + self.vqa.sum() * features["ar_batch"]["x"] * 0.25
        return SimpleNamespace(loss=loss)

    def get_optimizers(self, total_steps=0, steps_per_epoch=0):
        self.expected_steps = total_steps
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0), "interval": "step"}}


class Loader:
    def __init__(self, items):
        self.items, self.dataset = items, items
        self.epochs = []
        self.sampler = SimpleNamespace(set_epoch=self.epochs.append)
        self.batch_sampler = None

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def distributed_check(rank, rendezvous):
    from unidrivevla.distributed import DistributedContext, parallelize
    from unidrivevla.train.args import TrainArgs
    from unidrivevla.train.trainer import Trainer

    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), CUDA_VISIBLE_DEVICES="")
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    ctx = DistributedContext()
    try:
        for mode, expected_steps in (("combined", 3), ("alternate", 6)):
            planning = Loader([({"x": torch.tensor(float(i * (rank + 1)))}, {}, [str(i)]) for i in (1, 2, 3)])
            vqa = Loader([{"x": torch.tensor(float(i * (rank + 1)))} for i in (4, 5)])
            trainer = Trainer.__new__(Trainer)
            trainer.args = TrainArgs(strategy="ddp", precision="native", compile=False,
                                     ddp_find_unused_parameters=True, grad_accum=2, max_epochs=2, log_every=100)
            trainer.ctx = ctx
            trainer.agent = TinyPolicy()
            trainer.model = parallelize(trainer.agent, trainer.args, ctx)
            trainer.train_loader = TaskLoader(planning, vqa, mode)
            trainer.start_epoch = trainer.epoch = trainer.global_step = trainer.micro = 0
            trainer.resume_batch = trainer.batch_in_epoch = 0
            trainer._resume_rank_state = None
            trainer.ema = None
            trainer.end_epoch = lambda epoch, loss: None
            trainer.build_optimizer()
            trainer.train_loop()
            assert trainer.global_step == trainer.agent.expected_steps == expected_steps
            assert trainer.lr_scheduler.last_epoch == expected_steps
            values = torch.cat([trainer.agent.plan.detach(), trainer.agent.vqa.detach()])
            torch.testing.assert_close(values, torch.tensor([0.91, 0.949375]), rtol=1e-6, atol=1e-6)
            copies = [torch.zeros_like(values) for _ in range(2)]
            dist.all_gather(copies, values)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
        if rank == 0:
            print("OK: real two-rank CPU DDP trainer, summed gradients, accumulation and scheduler counts")
    finally:
        ctx.close()


def main():
    from omegaconf import OmegaConf
    from unidrivevla.data.registry import evaluation_params
    from unidrivevla.train.args import TrainArgs

    for invalid in ({"max_epochs": 0}, {"max_steps": -1}, {"val_every_n_epochs": 0}):
        try:
            TrainArgs(**invalid).validate(1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid training duration must fail: {invalid}")

    cfg = OmegaConf.create({"dataloader": {"params": {"batch_size": 4, "num_workers": 0}}})
    assert evaluation_params(cfg)["batch_size"] == 1
    assert cfg.dataloader.params.batch_size == 4
    planning = Loader([({"x": torch.tensor(float(i))}, {}, [str(i)]) for i in (1, 2, 3)])
    vqa = Loader([{"x": torch.tensor(4.0)}, {"x": torch.tensor(5.0)}])
    batches = TaskLoader(planning, vqa)
    weight = torch.tensor(2.0, requires_grad=True)
    for features, _, _ in batches:
        (weight * features["x"] + 0.25 * weight * features["ar_batch"]["x"]).backward()
    assert weight.grad.item() == 1 + 2 + 3 + 0.25 * (4 + 5 + 4)
    assert len(batches) == 3 and vqa.epochs == [0, 1]
    order = []
    for rank in (0, 1):
        source = Loader([{"rank": rank}])
        stream = TaskLoader(planning, source, "alternate")
        tasks = [features.get("task", "planning") for features, _, _ in stream]
        assert len(stream) == 6
        order.append(tasks)
    assert order[0] == order[1] == ["planning", "vqa"] * 3
    try:
        TaskLoader(planning, Loader([]))
    except ValueError:
        pass
    else:
        raise AssertionError("empty VQA loader must be rejected")
    print("OK: separate eval batch, summed gradients, VQA cycling and identical task order on both ranks")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(distributed_check, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    main()
