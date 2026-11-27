"""Check the production yawness, motion weighting and occupancy gradient paths on CPU."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.models.compat import FocalLoss, GaussianFocalLoss, L1Loss
from unidrivevla.models.models.detection3d.losses import SparseBox3DLoss
from unidrivevla.models.models.motion.target import SparseMotionTarget
from unidrivevla.models.unidrivevla.dense_heads.modules import OccLatentDecoder
from unidrivevla.models.unidrivevla.dense_heads.unified_perception_decoder import UnifiedPerceptionDecoder


def check_yawness():
    detector_loss = SparseBox3DLoss(
        dict(type="L1Loss"), dict(type="CrossEntropyLoss", use_sigmoid=True), dict(type="GaussianFocalLoss")
    )
    boxes = torch.zeros(3, 10)
    boxes[:, 7] = 1
    targets = boxes.clone()
    targets[0, 7] = -1
    quality = torch.tensor([[0., -4.], [0., 0.], [0., 4.]], requires_grad=True)
    actual = detector_loss(boxes, targets, quality=quality, avg_factor=3)["loss_yns"]
    # MMDetection 2.28.2 gaussian_focal_loss consumes the sigmoid probabilities from SparseBox3DLoss.
    p = quality[:, 1].sigmoid()
    target = torch.tensor([0., 1., 1.])
    reference = (-(p + 1e-12).log() * (1 - p).square() * (target == 1)
                 - (1 - p + 1e-12).log() * p.square() * (1 - target).pow(4)).sum() / (3 + torch.finfo(p.dtype).eps)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, quality)[0], torch.autograd.grad(reference, quality)[0])
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        endpoints = torch.tensor([0., 1.], dtype=dtype)
        loss = GaussianFocalLoss()(endpoints, endpoints.float(), avg_factor=0)
        assert loss.isfinite() and loss.item() == 0
        saturated = torch.tensor([[0., 30.], [0., -30.], [0., -30.]], dtype=dtype, requires_grad=True)
        loss = detector_loss(boxes, targets, quality=saturated, avg_factor=3)["loss_yns"]
        assert loss.isfinite() and torch.autograd.grad(loss, saturated)[0].isfinite().all()


def check_motion(batch, queries):
    modes, steps = 6, 12
    cls = torch.randn(batch, queries, modes, requires_grad=True)
    reg = torch.randn(batch, queries, modes, steps, 2, requires_grad=True)
    indices = [(torch.tensor([0, queries - 1]), torch.tensor([0, 1])) for _ in range(batch)]
    trajectories = [torch.randn(2, steps, 2) for _ in range(batch)]
    masks = [torch.ones(2, steps) for _ in range(batch)]
    sampler = SparseMotionTarget()
    model = SimpleNamespace(det_sampler=SimpleNamespace(indices=indices), motion_sampler=sampler,
                            loss_motion_cls=FocalLoss(loss_weight=.2), loss_motion_reg=L1Loss(loss_weight=.2))
    actual = UnifiedPerceptionDecoder._compute_motion_loss(model, [cls], [reg], trajectories, masks)
    labels, weights, best, gt, w, count = sampler.sample(reg, trajectories, masks, {"indices": indices})
    y = F.one_hot(labels, modes).float()
    p = cls.sigmoid()
    error_prob = (1 - p) * y + p * (1 - y)
    focal = F.binary_cross_entropy_with_logits(cls, y, reduction="none") * error_prob.square() * (.25*y + .75*(1-y))
    expected = .2 * ((focal * weights[..., None]).sum()
                     + ((best.cumsum(-2) - gt.cumsum(-2)).abs() * w[..., None]).sum()) / (
                         count.item() + torch.finfo(torch.float32).eps)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual, (cls, reg), retain_graph=True)
    for got, oracle in zip(grads, torch.autograd.grad(expected, (cls, reg))):
        torch.testing.assert_close(got, oracle)
    assert torch.count_nonzero(grads[0][~weights]) == 0
    assert torch.count_nonzero(grads[0][weights]) > 0


class TinyVAE(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.decoder = nn.Conv3d(4, 3, kernel_size=1).to(dtype)

    def forward_decoder(self, z, shapes, input_shape):
        assert z.shape == (1, 4, 1, 25, 25) and input_shape == (1, 1, 200, 200, 16)
        return self.decoder(z).permute(0, 2, 3, 4, 1).unsqueeze(-2)


def check_occupancy(vae_dtype):
    vae = TinyVAE(vae_dtype)
    with patch("unidrivevla.models.unidrivevla.dense_heads.modules.build_head", return_value=vae):
        decoder = OccLatentDecoder(8, {"decoder_cfg": {"z_channels": 4}})
    decoder.feat_proj.bfloat16()
    inputs = torch.randn(1, 625, 8, requires_grad=True)
    logits = decoder(inputs)
    loss = F.cross_entropy(logits.float(), torch.randint(3, (1, 1, 25, 25)))
    loss.backward()
    assert loss.isfinite() and inputs.grad.isfinite().all() and inputs.grad.abs().sum() > 0
    assert decoder.feat_proj.weight.grad.dtype == torch.bfloat16 and decoder.feat_proj.weight.grad.isfinite().all()
    assert all(p.grad is None for p in vae.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    check_yawness()
    check_motion(1, 6)  # A square matrix used to silently weight modes instead of queries.
    check_motion(4, 900)
    for dtype in (torch.float32, torch.bfloat16):
        check_occupancy(dtype)
    print("OK: detector yawness, per-query motion loss and gradients, BF16 occupancy with FP32/BF16 VAE")
