# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version as pkg_version
import torch.distributed as dist
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, MMDeepSpeedEngine
from deepspeed.pipe import PipelineModule
from deepspeed.git_version_info import version, git_hash, git_branch
from deepspeed.utils import log_dist, OnDevice
from deepspeed.runtime.config import DeepSpeedConfig
import deepspeed
dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}

ZeROddp_factory = {'cuda': MMDeepSpeedEngine}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel module; if device is mlu,
    return a MLUDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        :class:`nn.Module`: parallelized module.
    """
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel module;
    if device is mlu, return a MLUDistributedDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: parallelized module.

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu'], 'Only available for cuda or mlu devices.'
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return ddp_factory[device](model, *args, **kwargs)


def build_ZeROddp(model, optimizer=None, model_parameters=None, device='cuda', args=None, config_dict=None):
    """Build DeepSpeed ZeRO DDP module (Correct World Size Version)."""
    
    assert device == 'cuda', "zero only support cuda device for now."
    assert model is not None, "deepspeed.initialize requires a model"

    model = model.cuda()
    
    # 1. 准备配置字典
    ds_config = config_dict
    if ds_config is None and args is not None:
        if hasattr(args, 'deepspeed_config'):
            ds_config = args.deepspeed_config
        elif isinstance(args, dict) and 'deepspeed_config' in args:
            ds_config = args['deepspeed_config']
            
    if ds_config is None:
        raise ValueError("DeepSpeed config dict is None!")

    # ================= 步骤 1: 确保 DeepSpeed 后端初始化 =================
    # 这一步非常重要，确保 deepspeed.comm 能读到 world_size=8
    if dist.is_initialized():
        deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
    else:
        # 如果连 torch dist 都没初始化，尝试补救 (通常不会走到这，因为你有 --launcher pytorch)
        deepspeed.init_distributed(dist_backend='nccl')

    # ================= 步骤 2: 计算正确的 Batch Size (128) =================
    
    # 获取真实 World Size (现在应该是 8 了)
    world_size = dist.get_world_size()
    
    # 获取 Micro Batch (8)
    micro_batch = 8
    if args is not None:
         micro_batch = int(getattr(args.data, 'samples_per_gpu', 8))
    
    # 获取 Accumulation (2)
    grad_accum = 2
    if 'gradient_accumulation_steps' in ds_config and ds_config['gradient_accumulation_steps'] != 'auto':
        grad_accum = int(ds_config['gradient_accumulation_steps'])

    # 计算全局 Batch Size: 8 * 2 * 8 = 128
    global_batch = micro_batch * grad_accum * world_size
    
    print(f"[DeepSpeed Init] Detected WorldSize={world_size}. Configured Batch={global_batch} (Micro={micro_batch}, Acc={grad_accum})")

    # ================= 步骤 3: 填入正确的数值 =================
    # 我们不再传 16 了，直接传 128
    ds_config['train_micro_batch_size_per_gpu'] = micro_batch
    ds_config['gradient_accumulation_steps'] = grad_accum
    ds_config['train_batch_size'] = global_batch

    # ================= 步骤 4: 创建 Config 对象 =================
    # 此时: train_batch (128) == micro (8) * acc (2) * world (8)
    # 等式成立，DeepSpeedConfig 初始化通过！
    ds_config_obj = DeepSpeedConfig(ds_config)

    # ================= 步骤 5: 初始化引擎 =================
    ZeRO_engine = ZeROddp_factory[device](
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        args=None,
        config=ds_config,           
        config_class=ds_config_obj  
    )

    return tuple([ZeRO_engine, ZeRO_engine.optimizer, ZeRO_engine.training_dataloader, ZeRO_engine.lr_scheduler])

def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) == 1 else 'cpu'
