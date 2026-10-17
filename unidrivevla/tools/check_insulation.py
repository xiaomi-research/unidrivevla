"""Compare the opt-in B2D insulation CE and hidden-state gradient with main's expression."""

import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from unidrivevla.models.unidrivevla.dense_heads.qwenvl3_vla_planning_head import (
    QwenVL3APlanningHead,
)


def main():
    head = QwenVL3APlanningHead.__new__(QwenVL3APlanningHead)
    nn.Module.__init__(head)
    head.enable_knowledge_insulation = True
    head.prompt_style = "b2d"
    head.ar_loss_weight = 0.25

    batch_size, seq_len, text_len, hidden_size, vocab_size = 2, 9, 5, 4, 13
    torch.manual_seed(71)
    prefix_out = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    lm_head.requires_grad_(False)
    head.qwen3_vl_with_expert = types.SimpleNamespace(
        qwen3_vl=types.SimpleNamespace(lm_head=lm_head)
    )
    batch = types.SimpleNamespace(
        tokenized_prompt=torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 0, 0]]),
        tokenized_prompt_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
    )

    actual = head._knowledge_insulation_prompt_loss(prefix_out, batch)

    # Keep the official B2D main expression verbatim as the numerical oracle.
    lm_logits = lm_head(prefix_out)
    text_logits = lm_logits[:, -text_len:, :]
    text_labels = batch.tokenized_prompt.clone()
    text_labels.masked_fill_(~batch.tokenized_prompt_mask.bool(), -100)
    shift_logits = text_logits[..., :-1, :].contiguous()
    shift_labels = text_labels[..., 1:].contiguous()
    expected = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    assert torch.allclose(actual, expected, atol=0, rtol=0), (actual, expected)
    actual_grad = torch.autograd.grad(actual, prefix_out, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prefix_out)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=0, rtol=0)
    assert lm_head.weight.grad is None

    weighted = head.ar_loss_weight * actual
    main_total = head.ar_loss_weight * expected
    assert torch.allclose(weighted, main_total, atol=0, rtol=0)

    head.enable_knowledge_insulation = False
    assert head._knowledge_insulation_prompt_loss(prefix_out, batch) is None
    head.enable_knowledge_insulation = True
    head.prompt_style = "nuscenes"
    assert head._knowledge_insulation_prompt_loss(prefix_out, batch) is None
    print("B2D insulation prompt CE, mask, weighting, gradient, and opt-in guards passed")


if __name__ == "__main__":
    main()
