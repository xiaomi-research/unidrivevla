#!/usr/bin/env python3
"""
NaN Debug Hook for MMDetection Training

This hook integrates the NaN debugger into the mmcv runner training loop.
It checks for NaN values in losses, gradients, and parameters at each training step.
"""

import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class NaNDebugHook(Hook):
    """
    Hook to check for NaN values during training.

    Args:
        check_losses (bool): Check loss values before backward pass. Default: True
        check_gradients (bool): Check gradients after backward pass. Default: True
        check_parameters (bool): Check parameters after optimizer step. Default: False
        check_params_interval (int): Interval for parameter checking. Default: 100
        verbose (bool): Print detailed gradient statistics. Default: False
        abort_on_nan (bool): Abort training if NaN is detected. Default: True
    """

    def __init__(
        self,
        check_losses=True,
        check_gradients=True,
        check_parameters=False,
        check_params_interval=100,
        verbose=False,
        abort_on_nan=True,
    ):
        self.check_losses = check_losses
        self.check_gradients = check_gradients
        self.check_parameters = check_parameters
        self.check_params_interval = check_params_interval
        self.verbose = verbose
        self.abort_on_nan = abort_on_nan
        self.nan_debugger = None

    def before_run(self, runner):
        """Initialize NaN debugger from runner config"""
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model

        # Try to get debugger from config first
        if hasattr(runner, '_cfg') and hasattr(runner._cfg, 'nan_debugger'):
            self.nan_debugger = runner._cfg.nan_debugger
            runner.logger.info("[NaNDebugHook] Using NaN debugger from config")
        else:
            # Fallback: create new debugger
            from tools.nan_debugger import NaNDebugger
            self.nan_debugger = NaNDebugger(model, enabled=True)
            self.nan_debugger.register_hooks()
            runner.logger.info("[NaNDebugHook] Created new NaN debugger")

    def after_train_iter(self, runner):
        """Check for NaN after each training iteration"""
        if self.nan_debugger is None or not self.nan_debugger.enabled:
            return

        # Update debugger step counter
        self.nan_debugger.step = runner.iter

        # Check losses (stored in runner.outputs)
        if self.check_losses and hasattr(runner, 'outputs'):
            outputs = runner.outputs
            if isinstance(outputs, dict) and 'losses' in outputs:
                self.nan_debugger.check_losses(outputs['losses'])
            elif isinstance(outputs, dict) and 'loss' in outputs:
                # Some models might use 'loss' instead of 'losses'
                losses = {'total_loss': outputs['loss']}
                self.nan_debugger.check_losses(losses)

        # Check gradients
        if self.check_gradients:
            self.nan_debugger.check_gradients(verbose=self.verbose)

        # Check parameters periodically
        if self.check_parameters and (runner.iter % self.check_params_interval == 0):
            self.nan_debugger.check_parameters()

        # Abort if NaN detected
        if self.abort_on_nan and self.nan_debugger.nan_detected:
            runner.logger.error(
                f"[NaNDebugHook] NaN detected at iteration {runner.iter}. "
                f"Aborting training. Check logs above for details."
            )
            self.nan_debugger.abort_if_nan()
