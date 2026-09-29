"""Export a merged HuggingFace VLM checkpoint after PEFT training."""

import os.path as osp
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


def _get_planning_head(runner):
    model = runner.model
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "planning_head"):
        return model.planning_head
    if hasattr(model, "pts_bbox_head"):
        return model.pts_bbox_head
    return None


def _get_export_dir(runner, planning_head):
    configured = getattr(planning_head, "lora_merge_save_dir", None)
    if configured:
        return configured
    return osp.join(runner.work_dir, "merged_hf", f"iter_{runner.iter + 1}")


@HOOKS.register_module()
class LoRAMergeHook(Hook):
    """
    Registered automatically by mmdet_train.py when `lora_cfg` is set in the
    planning_head config.

    after_run: merges language/visual LoRA and visual WASP, then saves the
    resulting Qwen3-VL model and processor on rank 0.
    """

    def _lora_enabled(self, runner) -> bool:
        ph = _get_planning_head(runner)
        if ph is None:
            return False
        return getattr(getattr(ph, "qwen3_vl_with_expert", None), "_lora_enabled", False)

    def after_run(self, runner):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if not self._lora_enabled(runner):
            if rank == 0:
                runner.logger.info(
                    "[LoRAMergeHook] LoRA is not enabled on this run — skipping in-memory merge.")
            return

        ph = _get_planning_head(runner)
        if ph is None or (
            not hasattr(ph, "merge_and_save_lora")
            and not hasattr(ph, "merge_lora")
        ):
            if rank == 0:
                runner.logger.warning(
                    "[LoRAMergeHook] Could not locate a PEFT merge method. Skipping.")
            return

        # DeepSpeed ZeRO-3: consolidate sharded params on all ranks first
        _is_deepspeed = False
        try:
            import deepspeed
            _is_deepspeed = isinstance(runner.model, deepspeed.DeepSpeedEngine)
        except ImportError:
            pass

        if _is_deepspeed and hasattr(runner.model, "consolidate_16bit_params"):
            runner.model.consolidate_16bit_params()
            if is_dist:
                dist.barrier()
        elif _is_deepspeed and rank == 0:
            runner.logger.info(
                "[LoRAMergeHook] DeepSpeed engine does not expose "
                "consolidate_16bit_params(); skipping explicit consolidation."
            )

        if rank == 0:
            if hasattr(ph, "merge_and_save_lora"):
                export_dir = _get_export_dir(runner, ph)
                runner.logger.info(
                    "[LoRAMergeHook] Exporting merged HuggingFace VLM checkpoint to %s ...",
                    export_dir,
                )
                ph.merge_and_save_lora(export_dir)
                runner.logger.info(
                    "[LoRAMergeHook] Merged HuggingFace VLM export complete."
                )
            else:
                runner.logger.info("[LoRAMergeHook] Falling back to in-memory PEFT merge.")
                ph.merge_lora()

        if is_dist:
            dist.barrier()
