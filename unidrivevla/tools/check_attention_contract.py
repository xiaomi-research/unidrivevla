"""Check SDPA/frozen-prefix masks against the original main Flex segment predicate on CPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from unidrivevla.models.unidrivevla.dense_heads.qwenvl3_vla_planning_head import (
    QwenVL3APlanningHead,
)
from unidrivevla.models.unidrivevla.dense_heads.utils import make_att_2d_masks


def main_flex_predicate(prefix_len, perception_len, suffix_len, prompt_only_len, device):
    """Direct tensor translation of main's `_unidrive_mask_mod` (which receives no pad masks)."""
    p = int(prefix_len)
    pp = p + int(perception_len)
    length = pp + int(suffix_len)
    q = torch.arange(length, device=device)[:, None]
    k = torch.arange(length, device=device)[None, :]
    answer = (k >= prompt_only_len) & (k < p) if prompt_only_len >= 0 else torch.zeros_like(k, dtype=torch.bool)
    prefix = (q < p) & (k <= q)
    perception = (q >= p) & (q < pp) & (k < pp) & ~answer
    suffix = (q >= pp) & ~answer
    return (prefix | perception | suffix).unsqueeze(0)


def attention_and_grad(mask, q0, k0, v0):
    q, k, v = (tensor.detach().clone().requires_grad_() for tensor in (q0, k0, v0))
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, :, :])
    grads = torch.autograd.grad(output.square().sum(), (q, k, v))
    return output.detach(), tuple(grad.detach() for grad in grads)


def main():
    # A short sequence stands in for a prefix cut at its fixed cap; check both sides of that boundary.
    batch_size, fixed_cap, perception_len, suffix_len = 2, 6, 3, 2
    prefix_pad = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    perception_pad = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    suffix_pad = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    prefix_att = torch.tensor([[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]])
    perception_att = torch.ones(batch_size, perception_len, dtype=torch.long)
    suffix_att = torch.ones(batch_size, suffix_len, dtype=torch.long)

    head = QwenVL3APlanningHead.__new__(QwenVL3APlanningHead)
    head.official_attention_mask = True
    actual = head._training_attention_masks(
        prefix_pad, prefix_att, perception_pad, perception_att, suffix_pad, suffix_att,
    )
    expected = main_flex_predicate(fixed_cap, perception_len, suffix_len, -1, prefix_pad.device).expand(
        batch_size, -1, -1,
    )
    assert actual.dtype == torch.bool and actual.shape == (batch_size, 11, 11)
    assert torch.equal(actual, expected), "official mask differs from main's Flex segment predicate"

    # Prefix position cap-1 remains a causal prefix query; the next row starts bidirectional perception.
    assert actual[0, fixed_cap - 1, fixed_cap - 1]
    assert not actual[0, fixed_cap - 1, fixed_cap]
    assert actual[0, fixed_cap, fixed_cap + perception_len - 1]
    assert not actual[0, fixed_cap, fixed_cap + perception_len]
    # PAD keys and PAD queries stay visible/active, as they do in the Flex mask.
    assert actual[0, fixed_cap, fixed_cap - 1]
    assert actual[0, fixed_cap + perception_len + 1].all()

    prefix_only = head._official_flex_attention_mask(
        fixed_cap, 0, 0, batch_size, prefix_pad.device,
    )
    assert torch.equal(
        prefix_only,
        main_flex_predicate(fixed_cap, 0, 0, -1, prefix_pad.device).expand(batch_size, -1, -1),
    )
    assert prefix_only[0, fixed_cap - 1, fixed_cap - 1]  # last capped PAD state is a causal KV
    assert head._frozen_prefix_attention_mask(prefix_pad, prefix_att, torch.float32) is None

    # The old canonical path is still the existing key/query-pad-masked helper.
    head.official_attention_mask = False
    canonical = head._training_attention_masks(
        prefix_pad, prefix_att, perception_pad, perception_att, suffix_pad, suffix_att,
    )
    legacy = make_att_2d_masks(
        torch.cat([prefix_pad, perception_pad, suffix_pad], dim=1),
        torch.cat([prefix_att, perception_att, suffix_att], dim=1),
    )
    assert torch.equal(canonical, legacy), "default masked attention behavior changed"
    assert not canonical[0, fixed_cap, fixed_cap - 1]
    assert not canonical[0, fixed_cap + perception_len, fixed_cap + perception_len + 1]
    old_prefix = head._frozen_prefix_attention_mask(prefix_pad, prefix_att, torch.float32)
    expected_old_prefix = head._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad, prefix_att)).to(torch.float32)
    assert torch.equal(old_prefix, expected_old_prefix), "default frozen-prefix mask changed"

    torch.manual_seed(19)
    q = torch.randn(batch_size, 2, 11, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expected_out, expected_grads = attention_and_grad(expected, q, k, v)
    actual_out, actual_grads = attention_and_grad(actual, q, k, v)
    assert torch.equal(actual_out, expected_out)
    assert all(torch.equal(actual_grad, expected_grad) for actual_grad, expected_grad in zip(actual_grads, expected_grads))

    canonical_out, _ = attention_and_grad(canonical, q, k, v)
    assert (actual_out - canonical_out).abs().max().item() > 1e-4, "PAD visibility no longer affects attention"

    # The original Flex predicate excludes trajectory-answer keys from later segments.
    answer_mask = head._official_flex_attention_mask(
        fixed_cap, perception_len, suffix_len, batch_size, prefix_pad.device, prompt_only_len=4,
    )
    answer_expected = main_flex_predicate(fixed_cap, perception_len, suffix_len, 4, prefix_pad.device).expand(
        batch_size, -1, -1,
    )
    assert torch.equal(answer_mask, answer_expected)
    assert not answer_mask[0, fixed_cap, 4:fixed_cap].any()
    print("official Flex segment mask, PAD visibility, cap boundary, and SDPA gradients passed")


if __name__ == "__main__":
    main()
