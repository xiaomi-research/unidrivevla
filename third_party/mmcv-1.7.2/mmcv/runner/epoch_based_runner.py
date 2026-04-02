# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, no_type_check

import torch
from torch.utils.data import DataLoader
import os
import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint, save_deepspeed_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
    
    def save_deepspeed_checkpoint(self,
                                  out_dir: str,
                                  filename_tmpl: str = 'epoch_{}.pth',
                                  meta: Optional[Dict] = None,
                                  save_optimizer: bool = True,
                                  create_symlink: bool = True) -> None:
        """Save checkpoint to file for DeepSpeed.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'epoch_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
        
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        
        # DeepSpeed saves checkpoints as directories
        os.makedirs(filepath, exist_ok=True)
        
        # client_state in DeepSpeed engine.save_checkpoint corresponds to our meta
        save_deepspeed_checkpoint(
            self.model, 
            filepath, 
            tag=None, 
            client_state=meta, 
            save_latest=False
        )

        # Create symlink to the latest checkpoint directory
        if torch.distributed.get_rank() == 0:
            if create_symlink:
                dst_file = osp.join(out_dir, 'latest')
                if platform.system() != 'Windows':
                    mmcv.symlink(filename, dst_file)
                else:
                    shutil.copy(filepath, dst_file)
    
    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default',
               tag = None) -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to 'default'.
            tag (str, optional): Tag for DeepSpeed checkpoint.
        """
        deepspeed_enabled = isinstance(self.model, mmcv.parallel.MMDeepSpeedEngine)
        
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            if deepspeed_enabled:
                # DeepSpeed loads from a directory. 
                # Note: load_deepspeed_checkpoint must be implemented/mixin 
                # or we assume standard deepspeed loading logic here.
                # Assuming BaseRunner or the user's environment provides load_deepspeed_checkpoint
                # If not available, standard logic is:
                # load_path, client_state = self.model.load_checkpoint(checkpoint, tag=tag)
                # checkpoint = client_state
                checkpoint = self.load_deepspeed_checkpoint(checkpoint, tag)
            else:
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        if deepspeed_enabled:
            # DeepSpeed handles optimizer states internally
            # The returned 'checkpoint' from load_deepspeed_checkpoint is essentially the 'meta' dict
            self._epoch = checkpoint['epoch']
            self._iter = checkpoint['iter']
            self._inner_iter = 0 # Epoch based runner usually resets inner_iter on resume
        else:
            self._epoch = checkpoint['meta']['epoch']
            self._iter = checkpoint['meta']['iter']
            self._inner_iter = checkpoint['meta']['iter']
            if 'optimizer' in checkpoint and resume_optimizer:
                if isinstance(self.optimizer, Optimizer):
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                elif isinstance(self.optimizer, dict):
                    for k in self.optimizer.keys():
                        self.optimizer[k].load_state_dict(
                            checkpoint['optimizer'][k])
                else:
                    raise TypeError(
                        'Optimizer should be dict or torch.optim.Optimizer '
                        f'but got {type(self.optimizer)}')

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)
