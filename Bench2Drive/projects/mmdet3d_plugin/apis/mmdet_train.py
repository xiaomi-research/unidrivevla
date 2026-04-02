# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings
import os
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import (
    CustomDistEvalHook,
)
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from .util_distribution import build_dp, build_ddp, build_ZeROddp
import json
from torch.optim import AdamW

def find_latest_checkpoint(work_dir, extension="pth"):
    if not os.path.exists(work_dir): return None
    latest_link = os.path.join(work_dir, 'latest')
    if os.path.islink(latest_link):
        return os.readlink(latest_link)
    return None


def find_latest_deepspeed_checkpoint(work_dir):
    """
    Find the latest DeepSpeed checkpoint directory under work_dir.

    DeepSpeed saves checkpoints as:
        work_dir/iter_<N>/global_step<M>/mp_rank_00_model_states.pt

    The 'latest' symlink/file in work_dir points to the most recent iter dir.
    Falls back to scanning iter_* dirs by iter number if 'latest' is absent.

    Returns the absolute path to the latest iter_* directory, or None.
    """
    if not os.path.isdir(work_dir):
        return None

    # 1. Prefer the 'latest' symlink/dir written by DeepSpeed
    latest_path = os.path.join(work_dir, 'latest')
    if os.path.islink(latest_path):
        target = os.readlink(latest_path)
        # target may be relative (e.g. "iter_4585") or absolute
        if not os.path.isabs(target):
            target = os.path.join(work_dir, target)
        if os.path.isdir(target):
            return target
    elif os.path.isfile(latest_path):
        # Some DeepSpeed versions write the tag name as plain text
        with open(latest_path) as f:
            tag = f.read().strip()
        candidate = os.path.join(work_dir, tag)
        if os.path.isdir(candidate):
            return candidate

    # 2. Fall back: scan iter_* directories and pick the one with the largest number
    import re
    iter_dirs = []
    for name in os.listdir(work_dir):
        m = re.fullmatch(r'iter_(\d+)', name)
        if m and os.path.isdir(os.path.join(work_dir, name)):
            iter_dirs.append((int(m.group(1)), os.path.join(work_dir, name)))
    if iter_dirs:
        iter_dirs.sort(key=lambda x: x[0])
        return iter_dirs[-1][1]

    return None


def build_optimizer_manual(model, cfg):
    """
    通用手动优化器构建函数：支持 'AdamW' 和 'Muon'。
    同时处理参数去重 (Weight Tying) 和 VLM 学习率倍率。
    """
    # 1. 获取优化器类型和基础配置
    optim_type = cfg.optimizer.get('type', 'AdamW')
    
    # 基础参数
    base_lr = cfg.optimizer.get('lr', 1e-4)
    base_wd = cfg.optimizer.get('weight_decay', 0.0)
    betas = cfg.optimizer.get('betas', (0.9, 0.999))
    eps = cfg.optimizer.get('eps', 1e-8)

    # VLM 降权配置
    paramwise_cfg = cfg.optimizer.get('paramwise_cfg', {})
    custom_keys = paramwise_cfg.get('custom_keys', {})

    # 获取模型实体
    if hasattr(model, 'module'):
        real_model = model.module
    else:
        real_model = model

    # 准备参数遍历
    params = []
    seen_params = set() # 用于去重
    
    print(f"[Optimizer Builder] Mode: {optim_type}")
    
    # ================= 分支 A: 构建 Muon 优化器 =================
    if optim_type == 'Muon':
        # 读取 Muon 特有配置
        muon_lr = base_lr  # Config 里的 lr 作为 Muon 的主 lr (e.g. 0.02)
        muon_momentum = cfg.optimizer.get('momentum', 0.95)
        
        # 读取 Aux AdamW 配置
        adamw_cfg = cfg.optimizer.get('adamw_cfg', {})
        aux_lr = adamw_cfg.get('lr', 1e-4)
        aux_wd = adamw_cfg.get('weight_decay', 0.01)
        aux_betas = adamw_cfg.get('betas', (0.9, 0.999))
        aux_eps = adamw_cfg.get('eps', 1e-8)

        print(f" -> Muon LR: {muon_lr}, Momentum: {muon_momentum}")
        print(f" -> Aux AdamW LR: {aux_lr}, WD: {aux_wd}")

        for name, param in real_model.named_parameters():
            if not param.requires_grad: continue
            if param in seen_params: continue
            seen_params.add(param)

            # 计算倍率
            lr_mult = 1.0
            decay_mult = 1.0
            for key, mults in custom_keys.items():
                if key in name:
                    lr_mult = mults.get('lr_mult', 1.0)
                    decay_mult = mults.get('decay_mult', 1.0)
                    break
            
            # 判断是否使用 Muon (>=2D 且不是 Embedding)
            is_muon_candidate = (param.ndim >= 2) and ("embed" not in name)
            
            if is_muon_candidate:
                params.append({
                    "params": [param],
                    "lr": muon_lr * lr_mult,
                    "momentum": muon_momentum,
                    "weight_decay": base_wd * decay_mult, # Muon 这里的 wd 通常为 0
                    "use_muon": True,   # <--- Muon 标志
                    "name": name
                })
            else:
                params.append({
                    "params": [param],
                    "lr": aux_lr * lr_mult,
                    "betas": aux_betas,
                    "eps": aux_eps,
                    "weight_decay": aux_wd * decay_mult,
                    "use_muon": False,  # <--- AdamW 标志
                    "name": name
                })
        
        # 实例化 MuonWithAuxAdam
        optimizer = MuonWithAuxAdam(params)

    # ================= 分支 B: 构建标准 AdamW 优化器 =================
    else:
        # 这是标准的 AdamW 逻辑，但也加上了手动去重和 paramwise_cfg 支持
        print(f" -> Base LR: {base_lr}, WD: {base_wd}, Betas: {betas}")
        
        for name, param in real_model.named_parameters():
            if not param.requires_grad: continue
            if param in seen_params: continue
            seen_params.add(param)

            # 计算倍率
            lr_mult = 1.0
            decay_mult = 1.0
            for key, mults in custom_keys.items():
                if key in name:
                    lr_mult = mults.get('lr_mult', 1.0)
                    decay_mult = mults.get('decay_mult', 1.0)
                    break
            
            params.append({
                "params": [param],
                "lr": base_lr * lr_mult,
                "weight_decay": base_wd * decay_mult,
                "name": name
            })
            
        # 实例化 PyTorch AdamW
        optimizer = AdamW(params, lr=base_lr, betas=betas, eps=eps)

    print(f"[Optimizer Builder] Total parameter groups: {len(params)} (Deduplicated)")
    return optimizer

def custom_train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # assert len(dataset)==1s
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    if "runner" in cfg:
        runner_type = cfg.runner["type"]
    else:
        runner_type = "EpochBasedRunner"
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            nonshuffler_sampler=dict(
                type="DistributedSampler"
            ),  # dict(type='DistributedSampler'),
            runner_type=runner_type,
        )
        for ds in dataset
    ]

    deepspeed_enabled = cfg.get('deepspeed', False)
    optimizer = None

    if deepspeed_enabled:
        # 1. DeepSpeed 要求先构建 Optimizer
        print("[DeepSpeed] Building optimizer manually to handle shared weights...")
        
                # 2. 手动读取 JSON 配置文件为字典
        ds_cfg_dict = None
        if hasattr(cfg, 'deepspeed_config'):
            # 如果已经是字典（有些流程可能已经处理过）
            if isinstance(cfg.deepspeed_config, dict):
                ds_cfg_dict = cfg.deepspeed_config
            # 如果是路径字符串
            elif isinstance(cfg.deepspeed_config, str):
                print(f"Loading DeepSpeed config from: {cfg.deepspeed_config}")
                try:
                    with open(cfg.deepspeed_config, 'r') as f:
                        ds_cfg_dict = json.load(f)
                except Exception as e:
                    print(f"Error loading DeepSpeed config json: {e}")
                    raise e
        
        if ds_cfg_dict is None:
            raise ValueError("Failed to load deepspeed_config. Please check your config file.")

        # 1. 检查是否开启了 CPU Offload
        is_offload = False
        if ds_cfg_dict:
             offload_conf = ds_cfg_dict.get('zero_optimization', {}).get('offload_optimizer', {})
             if offload_conf.get('device') == 'cpu':
                 is_offload = True

        # 2. 构建 PyTorch 原生 AdamW (利用上面的函数进行去重)
        # 注意：这里我们先用 build_optimizer_manual 拿到分好组的 params
        # 但返回的是一个 torch.optim.AdamW 对象
        temp_optimizer = build_optimizer_manual(model, cfg)

        if is_offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            print("[DeepSpeed] Using DeepSpeedCPUAdam for Offload.")
            # 提取上面手动分好的参数组，传给 DeepSpeed 优化器
            optimizer = DeepSpeedCPUAdam(
                temp_optimizer.param_groups, 
                lr=cfg.optimizer.lr,
                betas=cfg.optimizer.betas,
                eps=cfg.optimizer.get('eps', 1e-8),
                weight_decay=cfg.optimizer.weight_decay
            )
            # 释放临时对象
            del temp_optimizer
        else:
            # 如果没有 Offload，直接用刚才构建好的 AdamW 即可
            optimizer = temp_optimizer


        # 3. 处理 Checkpoint (保持不变)
        if cfg.load_from:
            logger.info(f"Loading checkpoint from {cfg.load_from} for DeepSpeed...")
            checkpoint = torch.load(cfg.load_from, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            new_state_dict = new_state_dict['module']
            model.load_state_dict(new_state_dict, strict=False)

        # 4. 构建 ZeRO DDP 【关键修改：传入 config_dict】
        model, optimizer, _, _ = build_ZeROddp(
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            args=cfg,
            config_dict=ds_cfg_dict  
        )
        
        model.device_ids = [int(os.environ['LOCAL_RANK'])]

    else:
        # ================== 原有逻辑：标准 DDP/DP ==================
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


        # 标准模式下，DDP 之后构建 Optimizer
        optimizer = build_optimizer(model, cfg.optimizer)

    if "runner" not in cfg:
        cfg.runner = {
            "type": "EpochBasedRunner",
            "max_epochs": cfg.total_epochs,
        }
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )
    else:
        if "total_epochs" in cfg:
            pass
            #assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    
    if deepspeed_enabled:
        assert isinstance(optimizer_config, OptimizerHook), "deepspeed must use OptimizerHook"

    # AR Cotraining Hook Registration
    if hasattr(cfg, 'ar_dataset_cfg') and cfg.ar_dataset_cfg.get('enabled', False):
        from .ar_cotrain_hook import ARCotrainingHook
        ar_hook = ARCotrainingHook(cfg.ar_dataset_cfg)
        runner.register_hook(ar_hook, priority='NORMAL')
        logger.info(f"[AR Cotraining] Registered ARCotrainingHook")

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )

    # LoRA Merge Hook: auto-register when lora_cfg is set in the planning head.
    # Runs after_run on rank-0: merges LoRA adapter into base weights and saves
    # a single merged HuggingFace checkpoint (no separate adapter files).
    _planning_head_cfg = cfg.get('model', {}).get('planning_head', {})
    if _planning_head_cfg.get('lora_cfg') is not None:
        from .lora_merge_hook import LoRAMergeHook
        lora_hook = LoRAMergeHook()
        runner.register_hook(lora_hook, priority='VERY_LOW')
        logger.info("[LoRA] Registered LoRAMergeHook → will merge adapter into LLM before checkpoint save")

    # register profiler hook
    # trace_config = dict(type='tb_trace', dir_name='work_dir')
    # profiler_config = dict(on_trace_ready=trace_config)
    # runner.register_profiler_hook(profiler_config)

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline
            )
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_cfg["jsonfile_prefix"] = osp.join(
            "val",
            cfg.work_dir,
            time.ctime().replace(" ", "_").replace(":", "_"),
        )
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # ── Auto-resume: fill cfg.resume_from if not explicitly set ──────────
    if not cfg.resume_from:
        if deepspeed_enabled:
            latest_ckpt = find_latest_deepspeed_checkpoint(cfg.work_dir)
            if latest_ckpt:
                cfg.resume_from = latest_ckpt
                logger.info(f"[AutoResume] Found latest DeepSpeed checkpoint: {latest_ckpt}")
        else:
            latest_ckpt = find_latest_checkpoint(cfg.work_dir)
            if latest_ckpt:
                cfg.resume_from = latest_ckpt
                logger.info(f"[AutoResume] Found latest checkpoint: {latest_ckpt}")

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        if deepspeed_enabled:
            # mmcv save_deepspeed_checkpoint layout:
            #   work_dir/iter_N/              ← filepath (passed as save_dir to DS)
            #     global_step<M>/             ← DS tag (auto-set to global_step when tag=None)
            #       mp_rank_00_model_states.pt
            # So load needs: load_dir=work_dir/iter_N, tag=global_step<M>
            #
            # cfg.resume_from may be:
            #   (a) work_dir/iter_N  — most common (from latest symlink or --resume-from)
            #   (b) work_dir         — fallback
            resume_path = os.path.abspath(cfg.resume_from)
            import re as _re

            def _find_global_step_tag(iter_dir):
                """Find the global_step* subdir inside an iter_N directory."""
                candidates = sorted(
                    [d for d in os.listdir(iter_dir)
                     if d.startswith('global_step') and os.path.isdir(os.path.join(iter_dir, d))],
                    key=lambda x: int(x.replace('global_step', '') or 0)
                )
                return candidates[-1] if candidates else None

            if _re.search(r'iter_\d+$', resume_path):
                # Case (a): resume_path = work_dir/iter_N
                ds_load_dir = resume_path
                ds_tag = _find_global_step_tag(resume_path)
            else:
                # Case (b): resume_path = work_dir — find latest iter_N subdir
                ds_load_dir = resume_path
                ds_tag = None
            logger.info(f"[Resume] DeepSpeed load_dir={ds_load_dir}, tag={ds_tag}")
            runner.resume(ds_load_dir, tag=ds_tag)
        else:
            runner.resume(cfg.resume_from)

    elif cfg.load_from and not deepspeed_enabled:
        # DeepSpeed 的 load_from 在上面已经手动处理过了，这里跳过
        runner.load_checkpoint(cfg.load_from)
    # =====================================================

    runner.run(data_loaders, cfg.workflow)