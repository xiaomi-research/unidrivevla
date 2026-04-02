import torch
import math

def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    pad_masks = pad_masks.bool()
    att_masks_int = att_masks.long()

    cumsum = torch.cumsum(att_masks_int, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]

    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d & pad_2d

def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device
) -> torch.Tensor:
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def permute_metas_per_camera_fields(img_metas, permute_indices, target_sensor_order):
    """
    Permute per-camera fields in img_metas according to permute_indices.

    Args:
        img_metas: List of metadata dicts, or a single dict
        permute_indices: List of indices to reorder cameras (e.g., [0, 2, 1, 4, 5, 3])
        target_sensor_order: Target camera order (e.g., ["CAM_FRONT", "CAM_FRONT_RIGHT", ...])

    Returns:
        Permuted img_metas with same structure as input

    Example:
        >>> img_metas = [{"cams": {"CAM_FRONT": ..., "CAM_BACK": ..., ...}}]
        >>> permute_indices = [0, 2, 1, 4, 5, 3]
        >>> target_sensor_order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", ...]
        >>> permuted = permute_metas_per_camera_fields(img_metas, permute_indices, target_sensor_order)
    """
    if not isinstance(img_metas, list):
        return img_metas

    out = []
    for m in img_metas:
        if not isinstance(m, dict):
            out.append(m)
            continue

        m2 = dict(m)

        # Permute any list fields with 6 elements (assumed to be per-camera)
        for k, v in list(m2.items()):
            if isinstance(v, list) and len(v) == 6:
                m2[k] = [v[i] for i in permute_indices]

        # Special handling for 'cams' dict
        cams = m2.get("cams", None)
        if isinstance(cams, dict) and len(cams) == 6:
            ordered_keys = list(cams.keys())
            if all(k in cams for k in target_sensor_order):
                ordered_keys = target_sensor_order
            m2["cams"] = {k: cams[k] for k in [ordered_keys[i] for i in permute_indices]}

        out.append(m2)

    return out
