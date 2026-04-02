# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple
from deepspeed.inference.engine  import InferenceEngine
from mmcv.parallel import MODULE_WRAPPERS

from .scatter_gather import ScatterInputs, scatter_kwargs


@MODULE_WRAPPERS.register_module()
class MMDeepSpeedInferenceEngine(InferenceEngine):
    """A prototype for Deepspeed Enginer.

    MMDeepSpeedEngine is a protytpe to support mmcv and mmengine to use Deepspeed

    - It implement two APIs ``train_step()`` and ``val_step()``.
    """

    def to_kwargs(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_id: int) -> Tuple[tuple, tuple]:
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=0)

    def scatter(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_ids: List[int]) -> Tuple[tuple, tuple]:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    
    def forward(self, *inputs: Any, **kwargs: Any):
        # Eval mode will use this method.
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        input_img = kwargs[0].pop('img')
        input_img = self._cast_inputs_half(input_img)
        losses = super().forward(input_img, **kwargs[0])
        return losses
    
    def _cast_inputs_half(self, inputs):
        if isinstance(inputs, (list, tuple)):
            new_inputs = []
            for v in inputs:
                new_inputs.append(self._cast_inputs_half(v))
            return inputs.__class__(new_inputs)
        elif isinstance(inputs, dict):
            new_inputs = {}
            for k, v in inputs.items():
                new_inputs[k] = self._cast_inputs_half(v)
            return new_inputs
        elif hasattr(inputs, 'half'):
            return inputs.half()
        else:
            return inputs