# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple
from deepspeed.runtime.engine import DeepSpeedEngine
from mmcv.parallel import MODULE_WRAPPERS
import torch
from .scatter_gather import ScatterInputs, scatter_kwargs


@MODULE_WRAPPERS.register_module()
class MMDeepSpeedEngine(DeepSpeedEngine):
    """A prototype for Deepspeed Engine with BF16 Support."""

    def to_kwargs(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_id: int) -> Tuple[tuple, tuple]:
        return scatter_kwargs(inputs, kwargs, [device_id], dim=0)

    def scatter(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_ids: List[int]) -> Tuple[tuple, tuple]:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)
    
    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        
        if inputs and len(inputs) > 0 and len(inputs[0]) > 0:
            kwargs[0].update(inputs[0][0])
            
        # ================= 核心修复：开启 BFloat16 Autocast =================
        # 即使 DeepSpeed 已经把模型转为 BF16，我们仍需 Autocast 来管理
        # Float32 输入/Loss 和 BF16 模型之间的梯度转换，
        # 特别是针对 Custom Ops (如 BEVFormer 中的 Attention)。
        
        # 检查是否开启了 bf16 (可以通过 DeepSpeed config 判断，这里简单起见直接根据环境尝试)
        use_bf16 = self.bfloat16_enabled() if hasattr(self, 'bfloat16_enabled') else False
        
        if use_bf16:
            # 强制输入转换为 bf16 往往能解决更多潜在的 Op 类型检查问题
            # (可选，视情况而定，autocast 通常足够，但显式转换更稳)
            if 'img' in kwargs[0] and isinstance(kwargs[0]['img'], torch.Tensor):
                if kwargs[0]['img'].dtype == torch.float32:
                    kwargs[0]['img'] = kwargs[0]['img'].to(dtype=torch.bfloat16)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                losses = super().forward(**kwargs[0])
        else:
            # FP16 或 FP32 模式
            losses = super().forward(**kwargs[0])
        # ===================================================================
        
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(kwargs[0]['img_metas']))
        return outputs
    
    def forward(self, *inputs: Any, **kwargs: Any):
        # Eval mode
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        
        # Eval 时也建议开启 autocast 以防万一，虽然不涉及 backward
        use_bf16 = self.bfloat16_enabled() if hasattr(self, 'bfloat16_enabled') else False
        
        if use_bf16:
             if 'img' in kwargs[0] and isinstance(kwargs[0]['img'], torch.Tensor):
                if kwargs[0]['img'].dtype == torch.float32:
                    kwargs[0]['img'] = kwargs[0]['img'].to(dtype=torch.bfloat16)
                    
             with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                losses = super().forward(**kwargs[0])
        else:
             losses = super().forward(**kwargs[0])
             
        return losses