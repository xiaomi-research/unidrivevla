"""Compose planning and VQA loaders without mixing their collation or target formats."""

import torch


class TaskLoader:
    def __init__(self, planning, vqa, mode="combined", exact_resume=False, seed=0, rank=0, world=1):
        if mode not in ("combined", "alternate"):
            raise ValueError(f"vqa.mode must be combined or alternate, got {mode!r}")
        if len(vqa) == 0:
            raise ValueError("VQA loader has no complete batch; reduce its batch_size or provide more samples")
        self.planning, self.vqa, self.mode = planning, vqa, mode
        if exact_resume:
            if vqa.num_workers != 0:
                raise ValueError("train.exact_resume requires vqa.num_workers=0 so prefetched worker batches "
                                 "cannot be lost or replayed")
            from torch.utils.data import DataLoader, DistributedSampler

            sampler = DistributedSampler(vqa.dataset, num_replicas=world, rank=rank, shuffle=True,
                                         drop_last=vqa.drop_last, seed=seed + 1)
            self.vqa = DataLoader(
                vqa.dataset, batch_size=vqa.batch_size, sampler=sampler, num_workers=0,
                pin_memory=vqa.pin_memory, drop_last=vqa.drop_last, collate_fn=vqa.collate_fn,
                generator=torch.Generator().manual_seed(seed + rank + 1),
            )
        self.dataset = planning.dataset
        self.sampler = planning.sampler
        self.batch_sampler = planning.batch_sampler
        self.vqa_cycle = 0
        self.vqa_cycle_batches_seen = 0
        self._vqa_iterator = None
        self.epoch_vqa_cycle = 0
        self.epoch_vqa_cycle_batches_seen = 0
        self.epoch_vqa_batches_seen = 0
        self.planning_batches_seen = 0
        self.vqa_batches_seen = 0
        self.output_batches_seen = 0

    def __len__(self):
        return len(self.planning) * (2 if self.mode == "alternate" else 1)

    def next_vqa(self):
        if self._vqa_iterator is None:
            sampler = getattr(self.vqa, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.vqa_cycle)
            self._vqa_iterator = iter(self.vqa)
        try:
            batch = next(self._vqa_iterator)
            self.vqa_batches_seen += 1
            self.vqa_cycle_batches_seen += 1
            return batch
        except StopIteration:
            self.vqa_cycle += 1
            self.vqa_cycle_batches_seen = 0
            self._vqa_iterator = None
            return self.next_vqa()

    def state_dict(self):
        sampler = self.vqa.sampler
        return {
            "cursor_version": 2,
            "mode": self.mode,
            "vqa_cycle": self.vqa_cycle,
            "vqa_cycle_batches_seen": self.vqa_cycle_batches_seen,
            "vqa_batches_seen": self.vqa_batches_seen,
            "epoch_vqa_cycle": self.epoch_vqa_cycle,
            "epoch_vqa_cycle_batches_seen": self.epoch_vqa_cycle_batches_seen,
            "epoch_vqa_batches_seen": self.epoch_vqa_batches_seen,
            "planning_batches_seen": self.planning_batches_seen,
            "output_batches_seen": self.output_batches_seen,
            "vqa_sampler_epoch": getattr(sampler, "epoch", None),
        }

    def restore_state(self, state, epoch_complete=False):
        if state.get("cursor_version") != 2 or state.get("mode") != self.mode:
            raise RuntimeError("exact resume requires a compatible continuous VQA cursor")
        cycle = int(state["vqa_cycle"] if epoch_complete else state["epoch_vqa_cycle"])
        cycle_batches = int(state["vqa_cycle_batches_seen"] if epoch_complete
                            else state["epoch_vqa_cycle_batches_seen"])
        total_batches = int(state["vqa_batches_seen"] if epoch_complete else state["epoch_vqa_batches_seen"])
        self.vqa_cycle = cycle
        self.vqa_cycle_batches_seen = 0
        self.vqa_batches_seen = total_batches - cycle_batches
        self._vqa_iterator = None
        for _ in range(cycle_batches):
            self.next_vqa()
        if self.vqa_cycle != cycle:
            raise RuntimeError(f"exact resume VQA cycle mismatch: expected {cycle}, got {self.vqa_cycle}")
        self.epoch_vqa_cycle = self.vqa_cycle
        self.epoch_vqa_cycle_batches_seen = self.vqa_cycle_batches_seen
        self.epoch_vqa_batches_seen = self.vqa_batches_seen
        self.planning_batches_seen = self.output_batches_seen = 0

    def __iter__(self):
        # Every rank uses the same planning batch count and this same task order.
        self.planning_batches_seen = self.output_batches_seen = 0
        self.epoch_vqa_cycle = self.vqa_cycle
        self.epoch_vqa_cycle_batches_seen = self.vqa_cycle_batches_seen
        self.epoch_vqa_batches_seen = self.vqa_batches_seen
        for features, targets, tokens in self.planning:
            self.planning_batches_seen += 1
            if self.mode == "combined":
                # main's AR hook: one planning + one AR batch, one summed loss and optimizer update.
                batch = {**features, "ar_batch": self.next_vqa()}
                self.output_batches_seen += 1
                yield batch, targets, tokens
            else:
                # Explicit schedule ablation: separate micro-iterations change optimizer/accumulation semantics.
                self.output_batches_seen += 1
                yield features, targets, tokens
                self.output_batches_seen += 1
                yield {"task": "vqa", "ar_batch": self.next_vqa()}, {}, None
