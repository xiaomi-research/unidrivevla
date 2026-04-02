from __future__ import annotations

import os
import sys

import torch

# Ensure repo root is on sys.path when running this file directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projects.mmdet3d_plugin.models.instance_bank import InstanceBank  # noqa: E402


def test_instance_bank_cache_and_get_mask_cpu():
    torch.manual_seed(0)

    bank = InstanceBank(
        num_anchor=10,
        embed_dims=16,
        anchor=torch.zeros(10, 11),
        anchor_handler=None,
        num_temp_instances=4,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=False,
        feat_grad=False,
        max_time_interval=2.0,
    )

    B = 2
    feats = torch.randn(B, 10, 16)
    anchors = torch.randn(B, 10, 11)
    cls_scores = torch.randn(B, 10, 3)

    metas0 = {"timestamp": torch.tensor([0.0, 0.0]), "img_metas": [{}, {}]}
    bank.cache(instance_feature=feats, anchor=anchors, confidence=cls_scores, metas=metas0)

    assert bank.cached_feature is not None
    assert bank.cached_anchor is not None
    assert bank.cached_feature.shape == (B, 4, 16)
    assert bank.cached_anchor.shape == (B, 4, 11)

    # Within max_time_interval => mask True
    metas1 = {"timestamp": torch.tensor([1.0, 1.5]), "img_metas": [{}, {}]}
    _, _, _, _, _ = bank.get(batch_size=B, metas=metas1)
    assert bank.mask is not None
    assert bank.mask.dtype == torch.bool
    assert bool(bank.mask[0].item()) is True
    assert bool(bank.mask[1].item()) is True

    # Beyond max_time_interval => mask False
    metas2 = {"timestamp": torch.tensor([5.0, 5.0]), "img_metas": [{}, {}]}
    _, _, _, _, _ = bank.get(batch_size=B, metas=metas2)
    assert bool(bank.mask[0].item()) is False
    assert bool(bank.mask[1].item()) is False
