"""
AR Cotraining Hook for UniDriveVLA

Integrates AR (Autoregressive) cotraining into MMDetection training loop
using mmcv's Hook mechanism.

Reference: Plan document Section 3.1
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ARCotrainingHook(Hook):
    """
    AR Cotraining Hook for UniDriveVLA

    This hook manages the AR dataloader and prepares AR batches before each
    training iteration. The AR batch is stored as a model attribute to be
    accessed by train_step().

    Each GPU independently loads its own non-overlapping shard via
    DistributedSampler, eliminating the broadcast bottleneck and restoring
    the full effective batch size (samples_per_gpu * num_gpus).

    Args:
        ar_dataset_cfg: Configuration dict for AR dataset
            - enabled (bool): Whether AR cotraining is enabled
            - jsonl_path (str): Path to JSONL file
            - data_root (str): Root directory for nuScenes data
            - samples_per_gpu (int): Batch size per GPU
            - workers_per_gpu (int): Number of workers per GPU
    """

    def __init__(self, ar_dataset_cfg):
        super().__init__()
        self.ar_dataset_cfg = ar_dataset_cfg
        self.ar_dataloader = None
        self.ar_iter = None

    def before_run(self, runner):
        """
        Build AR dataloader before training starts.

        Supports two modes:
        1. Single dataset: jsonl_paths / jsonl_path  →  one ARPlanningQADataset
        2. Mixed dataset:  driving_jsonl_paths + vqa_jsonl_paths  →  InterleaveDataset
           Interleave ratio is 1:1 by default (one driving sample per one VQA sample).
        """
        if not self.ar_dataset_cfg.get('enabled', False):
            runner.logger.info("[AR Cotraining] Disabled")
            return

        try:
            from ..datasets.ar_planning_qa_dataset import ARPlanningQADataset, ar_collate_fn

            model = runner.model
            if hasattr(model, 'module'):
                planning_head = model.module.planning_head
            else:
                planning_head = model.planning_head

            if hasattr(planning_head, 'qwen3_vl_with_expert'):
                pass  # existence check only
            else:
                raise AttributeError("Cannot find qwen3_vl_with_expert in planning_head")

            processor = planning_head.qwen3_vl_with_expert.processor
            tokenizer = processor.tokenizer

            cfg = self.ar_dataset_cfg
            common_kwargs = dict(
                processor=processor,
                tokenizer=tokenizer,
                max_pixels=cfg.get('max_pixels', None),
                fix_image_size=cfg.get('fix_image_size', None),
            )

            vqa_paths = cfg.get('vqa_jsonl_paths') or cfg.get('vqa_jsonl_path')
            driving_paths = cfg.get('jsonl_paths') or cfg.get('jsonl_path')

            if vqa_paths is not None and driving_paths is not None:
                # ── Mixed mode: interleave driving + VQA 1:1 ──────────────
                driving_dataset = ARPlanningQADataset(
                    jsonl_path=driving_paths,
                    max_length=cfg.get('max_length', 2048),
                    **common_kwargs,
                )
                vqa_dataset = ARPlanningQADataset(
                    jsonl_path=vqa_paths,
                    max_length=cfg.get('vqa_max_length', cfg.get('max_length', 1024)),
                    **common_kwargs,
                )
                # Repeat the smaller dataset to match the larger, then interleave
                n_driving = len(driving_dataset)
                n_vqa = len(vqa_dataset)
                if n_driving >= n_vqa:
                    from torch.utils.data import Subset
                    vqa_indices = (list(range(n_vqa)) * (n_driving // n_vqa + 1))[:n_driving]
                    vqa_dataset = Subset(vqa_dataset, vqa_indices)
                else:
                    from torch.utils.data import Subset
                    driving_indices = (list(range(n_driving)) * (n_vqa // n_driving + 1))[:n_vqa]
                    driving_dataset = Subset(driving_dataset, driving_indices)

                # Interleave: [driving_0, vqa_0, driving_1, vqa_1, ...]
                n = len(driving_dataset)
                interleaved_indices = []
                for i in range(n):
                    interleaved_indices.append(('driving', i))
                    interleaved_indices.append(('vqa', i))

                class InterleavedDataset(torch.utils.data.Dataset):
                    def __init__(self, ds_a, ds_b, indices):
                        self.ds_a = ds_a
                        self.ds_b = ds_b
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        tag, i = self.indices[idx]
                        return self.ds_a[i] if tag == 'driving' else self.ds_b[i]

                ar_dataset = InterleavedDataset(driving_dataset, vqa_dataset, interleaved_indices)
                runner.logger.info(
                    f"[AR Cotraining] Mixed mode: {n_driving} driving + {n_vqa} VQA "
                    f"→ {len(ar_dataset)} interleaved samples"
                )
            else:
                # ── Single dataset mode ───────────────────────────────────
                if driving_paths is None:
                    raise ValueError("ar_dataset_cfg must contain jsonl_paths or jsonl_path")
                ar_dataset = ARPlanningQADataset(
                    jsonl_path=driving_paths,
                    max_length=cfg.get('max_length', 2048),
                    **common_kwargs,
                )
                runner.logger.info(
                    f"[AR Cotraining] Single mode: {len(ar_dataset)} samples"
                )

            # ── DistributedSampler: each GPU loads its own non-overlapping shard ──
            # This restores the full effective batch size (samples_per_gpu * num_gpus)
            # and eliminates the broadcast bottleneck from the old rank-0-only logic.
            sampler = None
            if dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(ar_dataset, shuffle=True, drop_last=True)

            self.ar_dataloader = DataLoader(
                ar_dataset,
                batch_size=cfg['samples_per_gpu'],
                sampler=sampler,
                shuffle=(sampler is None),  # shuffle only when no DistributedSampler
                num_workers=cfg.get('workers_per_gpu', 2),
                collate_fn=lambda batch: ar_collate_fn(batch, tokenizer),
                pin_memory=True,
                drop_last=True,
            )

            runner.logger.info(
                f"[AR Cotraining] Dataloader ready, batch_size={cfg['samples_per_gpu']}, "
                f"distributed={'yes' if sampler is not None else 'no'}"
            )

        except Exception as e:
            runner.logger.error(f"[AR Cotraining] Failed to build AR dataloader: {e}")
            raise

    def before_epoch(self, runner):
        """Reset AR iterator and update DistributedSampler epoch at the start of each epoch."""
        if self.ar_dataloader is None:
            return

        # Must call set_epoch so DistributedSampler reshuffles differently each epoch
        sampler = self.ar_dataloader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(runner.epoch)

        self.ar_iter = iter(self.ar_dataloader)
        runner.logger.info(
            f"[AR Cotraining] Reset AR iterator for epoch {runner.epoch + 1}"
        )

    def before_train_iter(self, runner):
        """
        Prepare AR batch before each training iteration.

        Each GPU independently calls next() on its own dataloader shard.
        No broadcast needed.
        """
        if self.ar_iter is None:
            self._set_ar_batch(runner, None)
            return

        try:
            ar_batch = next(self.ar_iter)
        except StopIteration:
            # AR dataloader exhausted mid-epoch; restart from the current shard
            self.ar_iter = iter(self.ar_dataloader)
            ar_batch = next(self.ar_iter)
            runner.logger.info("[AR Cotraining] AR dataloader exhausted, restarting")

        ar_batch = self._move_to_cuda(ar_batch)
        self._set_ar_batch(runner, ar_batch)

    def _move_to_cuda(self, obj):
        """Recursively move all tensors in nested structures to CUDA."""
        if obj is None:
            return None
        elif isinstance(obj, torch.Tensor):
            return obj.cuda(non_blocking=True)
        elif isinstance(obj, dict):
            return {k: self._move_to_cuda(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_cuda(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._move_to_cuda(item) for item in obj)
        else:
            return obj

    def _set_ar_batch(self, runner, ar_batch):
        """Store AR batch as model attribute for access in train_step()."""
        model = runner.model
        if hasattr(model, 'module'):  # DDP/DeepSpeed wrapper
            model.module._current_ar_batch = ar_batch
        else:
            model._current_ar_batch = ar_batch

    def after_train_epoch(self, runner):
        """Log AR cotraining statistics after each epoch."""
        if self.ar_dataloader is None:
            return
        runner.logger.info(
            f"[AR Cotraining] Completed epoch {runner.epoch + 1} "
            f"with AR cotraining enabled"
        )
