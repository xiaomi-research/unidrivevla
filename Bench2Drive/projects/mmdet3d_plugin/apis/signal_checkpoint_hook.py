"""
SignalCheckpointHook for UniDriveVLA

Catches SIGTERM (sent by the scheduler when a preemptible job is about to be
killed) and immediately saves a DeepSpeed checkpoint so training can be resumed
from the latest state after the job is rescheduled.

Usage: automatically registered by mmdet_train.py when deepspeed is enabled.
"""

import os
import re
import signal
import threading
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


def _latest_saved_iter(work_dir):
    """
    Return the iter number of the most recent DeepSpeed checkpoint already on
    disk, or 0 if none exists.  Reads the 'latest' symlink/file first, then
    falls back to scanning iter_* directories.
    """
    latest_path = os.path.join(work_dir, 'latest')
    if os.path.islink(latest_path):
        tag = os.path.basename(os.readlink(latest_path))
    elif os.path.isfile(latest_path):
        with open(latest_path) as f:
            tag = f.read().strip()
    else:
        tag = None

    if tag:
        m = re.fullmatch(r'iter_(\d+)', tag)
        if m:
            return int(m.group(1))

    # Fallback: scan all iter_* dirs
    best = 0
    if os.path.isdir(work_dir):
        for name in os.listdir(work_dir):
            m = re.fullmatch(r'iter_(\d+)', name)
            if m and os.path.isdir(os.path.join(work_dir, name)):
                best = max(best, int(m.group(1)))
    return best


@HOOKS.register_module()
class SignalCheckpointHook(Hook):
    """
    Saves a checkpoint on SIGTERM (preemption signal from scheduler).

    The hook installs a SIGTERM handler in before_run().  When the signal fires,
    a flag is set and the next after_train_iter() call saves the checkpoint and
    exits cleanly so DeepSpeed can write consistent shard files from all ranks.

    Safety: only saves when the current iter is strictly greater than the last
    checkpoint already on disk, preventing a freshly-started job from
    overwriting a good checkpoint with iter_1.
    """

    def before_run(self, runner):
        self._save_flag = threading.Event()
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            runner.logger.info(
                "[SignalCheckpointHook] SIGTERM received — "
                "will save checkpoint at end of current iteration."
            )
            self._save_flag.set()

        signal.signal(signal.SIGTERM, _handler)
        runner.logger.info(
            "[SignalCheckpointHook] SIGTERM handler installed."
        )

    def after_train_iter(self, runner):
        if not self._save_flag.is_set():
            return

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        current_iter = runner.iter + 1  # iter is 0-based before increment

        # Guard: don't overwrite a better existing checkpoint
        existing_iter = _latest_saved_iter(runner.work_dir)
        if current_iter <= existing_iter:
            if rank == 0:
                runner.logger.warning(
                    f"[SignalCheckpointHook] SIGTERM at iter {current_iter}, "
                    f"but latest checkpoint is iter {existing_iter} — "
                    f"skipping save to avoid overwrite. Exiting."
                )
            if is_dist:
                dist.barrier()
            raise SystemExit(0)

        tag = f"iter_{current_iter}"

        # Detect DeepSpeed engine
        _is_deepspeed = False
        try:
            import deepspeed
            _is_deepspeed = isinstance(runner.model, deepspeed.DeepSpeedEngine)
        except ImportError:
            pass

        if _is_deepspeed:
            if rank == 0:
                runner.logger.info(
                    f"[SignalCheckpointHook] Saving DeepSpeed checkpoint: {tag}"
                )
            # Use runner.save_deepspeed_checkpoint so that the 'latest' symlink
            # is updated on rank-0 (bare engine.save_checkpoint does not do this)
            runner.save_deepspeed_checkpoint(
                runner.work_dir,
                filename_tmpl=tag,  # saves to work_dir/iter_N/
                meta=None,
            )
            if rank == 0:
                runner.logger.info(
                    f"[SignalCheckpointHook] DeepSpeed checkpoint saved: "
                    f"{runner.work_dir}/{tag}"
                )
        else:
            # Standard mmcv checkpoint (rank-0 only)
            if rank == 0:
                runner.logger.info(
                    f"[SignalCheckpointHook] Saving mmcv checkpoint: {tag}"
                )
                runner.save_checkpoint(runner.work_dir, filename_tmpl=tag + ".pth")

        if is_dist:
            dist.barrier()

        # Re-raise as SystemExit so the training loop terminates cleanly
        runner.logger.info(
            "[SignalCheckpointHook] Checkpoint saved. Exiting."
        )
        raise SystemExit(0)

    def after_run(self, runner):
        # Restore original SIGTERM handler
        signal.signal(signal.SIGTERM, self._original_sigterm)
