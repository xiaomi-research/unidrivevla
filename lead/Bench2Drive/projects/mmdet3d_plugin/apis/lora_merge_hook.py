"""
LoRA Merge Hook for UniDriveVLA (Bench2Drive)

after_run (rank-0, synchronous):
  At the end of training, merges the LoRA adapter into the live model weights
  so that any subsequent mmcv checkpoint hook saves merged weights.
  With DeepSpeed ZeRO-3, consolidates sharded params on all ranks first.
"""

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


def _merge_lora(planning_head):
    if hasattr(planning_head, "merge_lora"):
        planning_head.merge_lora()
        return "planning_head.merge_lora"
    if hasattr(planning_head, "merge_and_save_lora"):
        planning_head.merge_and_save_lora()
        return "planning_head.merge_and_save_lora"
    vlm = getattr(planning_head, "qwen3_vl_with_expert", None)
    if hasattr(vlm, "merge_lora"):
        vlm.merge_lora()
        return "planning_head.qwen3_vl_with_expert.merge_lora"
    return None


@HOOKS.register_module()
class LoRAMergeHook(Hook):
    """
    Registered automatically by mmdet_train.py when `lora_cfg` is set in the
    planning_head config.

    after_run: merges the LoRA adapter into the live model weights.
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
        if ph is None:
            if rank == 0:
                runner.logger.warning(
                    "[LoRAMergeHook] Could not locate planning head. Skipping.")
            return

        # DeepSpeed ZeRO-3: consolidate sharded params on all ranks first
        _is_deepspeed = False
        try:
            import deepspeed
            _is_deepspeed = isinstance(runner.model, deepspeed.DeepSpeedEngine)
        except ImportError:
            pass

        zero_stage = (
            runner.model.zero_optimization_stage()
            if _is_deepspeed and hasattr(runner.model, "zero_optimization_stage")
            else 0
        )
        if zero_stage == 3:
            if not hasattr(runner.model, "consolidate_16bit_params"):
                if rank == 0:
                    runner.logger.warning(
                        "[LoRAMergeHook] ZeRO-3 engine cannot consolidate parameters; "
                        "skipping in-memory merge."
                    )
                return
            runner.model.consolidate_16bit_params()
            if is_dist:
                dist.barrier()

        if rank == 0:
            runner.logger.info("[LoRAMergeHook] Merging LoRA adapter into live model weights...")
            merge_method = _merge_lora(ph)
            if merge_method is None:
                runner.logger.warning(
                    "[LoRAMergeHook] Could not locate a LoRA merge method. Skipping.")
            else:
                runner.logger.info(
                    f"[LoRAMergeHook] In-memory LoRA merge complete via {merge_method}.")

        if is_dist:
            dist.barrier()
