# Copyright 2026 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import torch
from packaging.version import Version

if Version(torch.__version__) >= Version("2.5.0"):
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )
else:
    create_block_mask = None
    flex_attention = None


@torch.compile(mode="max-autotune-no-cudagraphs")
def compiled_flex_attention_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    scale: float,
):
    return flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True
    )


def _unidrive_mask_mod(
    prefix_len: int,
    perception_len: int,
    suffix_len: int,
    prompt_only_len: int = -1,
):
    p = int(prefix_len)
    pp = p + int(perception_len)
    l = p + int(perception_len) + int(suffix_len)
    pol = int(prompt_only_len) if prompt_only_len >= 0 else p

    def mask_mod(b, h, q_idx, kv_idx):
        in_range = (q_idx < l) & (kv_idx < l)

        q_is_prefix = q_idx < p
        q_is_occ = (q_idx >= p) & (q_idx < pp)
        q_is_suffix = q_idx >= pp

        kv_is_answer = (kv_idx >= pol) & (kv_idx < p)

        mask_prefix = q_is_prefix & (kv_idx <= q_idx)

        mask_occ = q_is_occ & (kv_idx < pp) & ~kv_is_answer

        mask_suffix = q_is_suffix & ~kv_is_answer

        return in_range & (mask_prefix | mask_occ | mask_suffix)

    return mask_mod


def build_blockmask_unidrive(
    *,
    bsz: int,
    hq: int,
    prefix_len: int,
    perception_len: int,
    suffix_len: int,
    device: torch.device,
    block_size: int = 128,
    compile_blockmask: bool = True,
    prompt_only_len: int = -1,
):
    if create_block_mask is None:
        raise RuntimeError("FlexAttention requires torch >= 2.5.0")

    l = int(prefix_len) + int(perception_len) + int(suffix_len)

    q_len_rounded = int(math.ceil(l / block_size) * block_size)
    kv_len_rounded = q_len_rounded

    mask_mod = _unidrive_mask_mod(
        prefix_len, perception_len, suffix_len,
        prompt_only_len=prompt_only_len,
    )

    block_mask = create_block_mask(
        mask_mod=mask_mod,
        B=int(bsz),
        H=int(hq),
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=int(block_size),
        device=device,
        _compile=compile_blockmask,
    )
    return block_mask, q_len_rounded


def flex_attention_forward_optimized(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_mask,
    scaling: float = None,
    q_len_rounded: int = None,
) -> torch.Tensor:
    bsz, hq, l, d = query_states.shape

    if q_len_rounded is not None and q_len_rounded != l:
        pad = q_len_rounded - l
        query_states = torch.nn.functional.pad(query_states, (0, 0, 0, pad))
        key_states = torch.nn.functional.pad(key_states, (0, 0, 0, pad))
        value_states = torch.nn.functional.pad(value_states, (0, 0, 0, pad))

    out = compiled_flex_attention_wrapper(
        query_states,
        key_states,
        value_states,
        block_mask,
        scaling
    )

    out = out[:, :, :l, :]
    out = out.transpose(1, 2).contiguous().reshape(bsz, l, hq * d)
    return out
