import functools
import math
from collections.abc import Callable

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from lead.training.config_training import TrainingConfig


@beartype
def normalize_imagenet(
    x: jt.Float[torch.Tensor, "B 3 H W"],
) -> jt.Float[torch.Tensor, "B 3 H W"]:
    """Normalize input images according to ImageNet standards.
    Args:
        x: Input images batch.

    Returns:
        Normalized images batch.
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


def _fp32_forward_wrapper(original_forward):
    """Wrap a normalization layer's forward method to force FP32 operations."""

    def forward(self, x):
        # Convert input to FP32, apply normalization, return in original dtype
        input_dtype = x.dtype
        x_fp32 = x.float()
        out_fp32 = original_forward(x_fp32)
        return out_fp32.to(input_dtype)

    return forward


def patch_norm_fp32(module: torch.nn.Module) -> torch.nn.Module:
    """Patch normalization layers to use FP32 operations while preserving module structure.

    Args:
        module: The module to patch.

    Returns:
        The patched module with FP32 normalization operations.
    """
    for child in module.modules():
        if isinstance(
            child,
            (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm),
        ):
            # Ensure parameters are in FP32
            child.float()
            # Patch the forward method to handle input/output dtype conversion
            child.forward = _fp32_forward_wrapper(child.forward).__get__(
                child,
                type(child),
            )
    return module


def force_fp32(apply_to: tuple[str, ...] | None = None):
    """
    Decorator to force a function to run in fp32 precision.

    Args:
        apply_to: Tuple of argument names to convert to fp32. If None, converts all tensor arguments.

    Returns:
        Decorated function that runs in fp32 precision.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Disable autocast to prevent fp16 operations
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # Convert specified arguments to fp32
                if apply_to is not None:
                    # Get function argument names
                    import inspect

                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())

                    # Convert positional arguments
                    new_args = list(args)
                    for i, arg in enumerate(args):
                        if i < len(param_names) and param_names[i] in apply_to:
                            if isinstance(arg, torch.Tensor):
                                new_args[i] = arg.float()

                    # Convert keyword arguments
                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if key in apply_to and isinstance(value, torch.Tensor):
                            new_kwargs[key] = value.float()
                        else:
                            new_kwargs[key] = value

                    return func(*new_args, **new_kwargs)
                else:
                    # Convert all tensor arguments to fp32
                    new_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            new_args.append(arg.float())
                        else:
                            new_args.append(arg)

                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor):
                            new_kwargs[key] = value.float()
                        else:
                            new_kwargs[key] = value

                    return func(*new_args, **new_kwargs)

        return wrapper

    return decorator


@beartype
def gen_sineembed_for_position(
    pos_tensor: jt.Float[torch.Tensor, "B 2"],
    hidden_dim: int = 64,
):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR
    Args:
        pos_tensor: Last dimension is (x, y). Values are expected to be in range [0, 1].
        hidden_dim: Dimension of the output positional embedding. Must be even.
    Returns:
        Positional embedding with shape (B, hidden_dim)
    """
    assert 0 <= pos_tensor.min() and pos_tensor.max() <= 1, (
        "pos_tensor values should be in range [0, 1]"
    )
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=-1,
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
        dim=-1,
    ).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


@beartype
def unit_normalize_bev_points(
    points: jt.Float[npt.NDArray | torch.Tensor, "... 2"],
    config: TrainingConfig,
) -> jt.Float[npt.NDArray | torch.Tensor, "... 2"]:
    """Unit normalize BEV points to range [0, 1].

    Args:
        points: BEV points in meters.
        config: Training configuration containing BEV parameters.
    Returns:
        Normalized BEV points of shape in range [0, 1].
    """
    min_x, max_x, min_y, max_y = (
        config.min_x_meter,
        config.max_x_meter,
        config.min_y_meter,
        config.max_y_meter,
    )
    if isinstance(points, torch.Tensor):
        points = points.clone()
    else:
        points = points.copy()
    points[..., 0] = (points[..., 0] - min_x) / (max_x - min_x)
    points[..., 1] = (points[..., 1] - min_y) / (max_y - min_y)
    return points


@beartype
def bev_grid_sample(
    bev: jt.Float[torch.Tensor, "B D H W"],
    ref_points: jt.Float[torch.Tensor, "B N 2"],  # absolute coords (x, y)
    config: TrainingConfig,
) -> jt.Float[torch.Tensor, "B N D"]:
    """
    Deterministic bilinear sampling of BEV features at given reference points.

    Args:
        bev: BEV feature map in ego space.
        ref_points: Absolute coordinates in ego space.
        config: object with min_x, max_x, min_y, max_y attributes

    Returns:
        sampled: interpolated BEV features at given points (B, N, D)
    """
    B, D, H, W = bev.shape
    N = ref_points.shape[1]

    x = ref_points[..., 0]
    y = ref_points[..., 1]

    # Normalize to [-1, 1]
    u = 2 * (y - config.min_y_meter) / (config.max_y_meter - config.min_y_meter) - 1
    v = 2 * (x - config.min_x_meter) / (config.max_x_meter - config.min_x_meter) - 1

    grid = torch.stack([u, v], dim=-1)  # (B, N, 2)
    grid = grid.view(B, N, 1, 2)  # (B, N, 1, 2)

    sampled = F.grid_sample(
        bev,
        grid,
        mode="bilinear",
        align_corners=True,
    )  # (B, D, N, 1)

    return sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, D)


def class2angle(
    angle_cls: torch.Tensor,
    angle_res: torch.Tensor,
    config: TrainingConfig,
    limit_period: bool = True,
) -> torch.Tensor:
    """Convert discrete angle class and residual back to continuous angle.

    Inverse function to angle2class for decoding predicted angle values.

    Args:
        angle_cls: Discrete angle class tensor to decode.
        angle_res: Angle residual tensor to decode.
        config: Training configuration containing num_dir_bins.
        limit_period: Whether to limit angle to [-π, π] range.

    Returns:
        Decoded continuous angle tensor.
    """
    angle_per_class = 2 * np.pi / float(config.num_dir_bins)
    angle_center = angle_cls.float() * angle_per_class
    angle = angle_center + angle_res
    if limit_period:
        angle[angle > np.pi] -= 2 * np.pi
        return angle
