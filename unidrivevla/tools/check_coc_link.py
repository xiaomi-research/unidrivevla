"""CPU contract check for CoC labels, greedy generation, and planner-visible prefix KV."""

import json
import os
import sys
import tempfile
import types

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from unidrivevla.models.unidrivevla.dense_heads.canbus_prompt import (  # noqa: E402
    COC_MAX_NEW_TOKENS,
    build_driving_user_prompt,
    coc_language_target,
    format_ego_status,
    load_coc_by_token,
)
from unidrivevla.models.unidrivevla.dense_heads.qwenvl3_vla_planning_head import (  # noqa: E402
    QwenVL3APlanningHead,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    special = {"<|im_start|>": 1, "<|im_end|>": 2, "\n": 3}

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        if text in self.special:
            return [self.special[text]]
        return [100 + ord(char) for char in text]


def make_head(tokenizer=None, *, path="", table=None):
    tokenizer = tokenizer or FakeTokenizer()
    head = QwenVL3APlanningHead.__new__(QwenVL3APlanningHead)
    nn.Module.__init__(head)
    head._num_views = 1
    head.action_horizon = 6
    head.pose_interval = 0.5
    head.view_token_str_list = ["<FRONT_VIEW>"]
    head.view_token_ids = torch.tensor([42], dtype=torch.long)
    head.command_prompts = ["TURN RIGHT", "TURN LEFT", "GO STRAIGHT"]
    head.num_commands = len(head.command_prompts)
    head.system_prompt = "You are a driving assistant."
    head.user_prompt_template = "history {hist_traj_str}; command [{nav_cmd}]"
    head.prompt_style = "nuscenes"
    head.coordinate_frame = "ego"
    head.official_command_text = False
    head.coc_language_prompt = True
    head.canbus_sft_prompt = False
    head.coc_jsonl_path = path
    head.coc_by_token = table
    head.enable_traj_ar = False
    head.qwen3_vl_with_expert = types.SimpleNamespace(
        processor=types.SimpleNamespace(tokenizer=tokenizer)
    )
    return head, tokenizer


def driving_inputs():
    images = torch.zeros(2, 1, 3, 4, 4)
    commands = torch.tensor([0, 2])
    history = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])
    return images, commands, history


class FakeCache:
    def __init__(self, seq_len, batch_size):
        self.seq_len = seq_len
        self.history = [[] for _ in range(batch_size)]
        self.visual_prefill = True


class FakeEmbedding:
    def __call__(self, token_ids):
        return token_ids.unsqueeze(-1).to(torch.float32)


class FakeLMHead:
    weight = torch.empty(1, dtype=torch.float32)

    def __call__(self, hidden):
        next_ids = hidden[..., 0].long().clamp(0, 32)
        logits = torch.full((*next_ids.shape, 33), -10.0, device=hidden.device)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 10.0)
        return logits


class FakeQwen:
    def __init__(self):
        self.lm_head = FakeLMHead()
        self.language_model = types.SimpleNamespace(config=types.SimpleNamespace())
        self.embedding = FakeEmbedding()

    def get_input_embeddings(self):
        return self.embedding


class FakeCore:
    def __init__(self, cache, prefix_out=None):
        self.qwen3_vl = FakeQwen()
        self.processor = types.SimpleNamespace(tokenizer=FakeTokenizer())
        self.cache = cache
        self.prefix_out = prefix_out
        self.calls = []
        self.prefill_calls = []
        self.image_encode_calls = 1  # the caller already did one visual prefill

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache,
                deepstack_visual_embeds=None, visual_pos_masks=None):
        if past_key_values is None:
            assert use_cache and self.prefix_out is not None
            assert attention_mask.shape[-1] == inputs_embeds[0].shape[1]
            self.prefill_calls.append((deepstack_visual_embeds, visual_pos_masks))
            return [self.prefix_out, None, None], self.cache, {}
        assert use_cache and past_key_values is self.cache
        assert attention_mask.shape[-1] == self.cache.seq_len + inputs_embeds[0].shape[1]
        current_ids = inputs_embeds[0][:, 0, 0].long().tolist()
        allowed = attention_mask[:, 0, 0, -1].eq(0).tolist()
        self.calls.append({
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            "current_ids": current_ids,
            "allowed": allowed,
            "cache_len": self.cache.seq_len,
            "visual_prefill": self.cache.visual_prefill,
        })
        next_ids = []
        for row, (token_id, visible) in enumerate(zip(current_ids, allowed)):
            if visible:
                self.cache.history[row].append(token_id)
            history = self.cache.history[row]
            if row == 0:
                next_id = 2 if history else 10
            elif len(history) == 0:
                next_id = 11
            elif len(history) == 1:
                next_id = 12
            elif len(history) == 2:
                next_id = 13
            else:
                next_id = 2
            next_ids.append(next_id)
        self.cache.seq_len += 1
        hidden = torch.tensor(next_ids, dtype=torch.float32).view(-1, 1, 1)
        return [hidden, None, None], self.cache, {}


class EndlessCore(FakeCore):
    def forward(self, **kwargs):
        outputs, cache, middle = super().forward(**kwargs)
        outputs[0].fill_(10.0)  # never emits EOS
        return outputs, cache, middle


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "coc.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"sample_token": "abc", "cot": "the car yields"}) + "\n"
                + json.dumps({"token": "def", "cot": "a pedestrian crosses"}) + "\n"
                + json.dumps({"sample_token": "blank", "cot": "   "}) + "\n"
            )
        table = load_coc_by_token(path)
        assert coc_language_target(table["abc"]) == "the car yields"
        assert coc_language_target(table.get("missing")) == ""
        assert coc_language_target(table["blank"]) == ""

        head, tokenizer = make_head(path=path)
        head.train_vlm = False  # frozen-VLM planning still uses valid teacher-forced CoC context
        images, commands, history = driving_inputs()
        train_batch = head._build_driving_batch(
            images, commands, hist_traj=history, sample_tokens=["abc", "def"], include_coc_targets=True,
        )
        assert train_batch.coc_labels is not None and train_batch.coc_answer_mask is not None
        expected = tokenizer.encode("the car yields") + [tokenizer.special["<|im_end|>"], tokenizer.special["\n"]]
        assert train_batch.coc_labels[0, :len(expected)].tolist() == expected
        assert int(train_batch.coc_answer_mask[0].sum()) == len(expected)
        assert train_batch.coc_labels[0, len(expected):].eq(-100).all()
        assert train_batch.coc_answer_lens == train_batch.coc_answer_mask.sum(dim=1).tolist()
        # Different prompt/answer lengths must still give each sample the same weight.
        train_batch.prompt_lens = [2, 5]
        head.ar_loss_weight = 0.1
        lm_head = nn.Linear(4, 512, bias=False)
        head.qwen3_vl_with_expert.qwen3_vl = types.SimpleNamespace(lm_head=lm_head)
        prefix = torch.randn(2, COC_MAX_NEW_TOKENS + 5, 4, requires_grad=True)
        reference = []
        for index, start in enumerate(train_batch.prompt_lens):
            offsets = train_batch.coc_answer_mask[index].nonzero(as_tuple=True)[0]
            reference.append(torch.nn.functional.cross_entropy(lm_head(prefix[index, start - 1 + offsets]),
                                                               train_batch.coc_labels[index, offsets]))
        expected_loss = torch.stack(reference).mean() * head.ar_loss_weight
        actual_loss = head._coc_language_loss(train_batch, prefix)
        torch.testing.assert_close(actual_loss, expected_loss)
        params = (prefix, lm_head.weight)
        for actual, expected_grad in zip(torch.autograd.grad(actual_loss, params),
                                         torch.autograd.grad(expected_loss, params)):
            torch.testing.assert_close(actual, expected_grad)
        assert any("chain-of-causation reasoning" in text for text in tokenizer.seen)
        training_prompts = [text for text in tokenizer.seen if text.startswith("As an autonomous driving system")]
        assert len(training_prompts) == 2
        assert all("6 future waypoints over 3.0 seconds" in text for text in training_prompts)
        assert head._training_coc_targets(["abc", "def"], 2) == ["the car yields", "a pedestrian crosses"]

        mixed_head, mixed_tokenizer = make_head(path=path)
        mixed_head.ar_loss_weight = 0.1
        mixed_batch = mixed_head._build_driving_batch(
            images, commands, hist_traj=history, sample_tokens=["abc", "blank"], include_coc_targets=True,
        )
        assert mixed_batch.images.shape[0] == 2 and mixed_batch.tokenized_prompt_mask.bool().any(dim=1).all()
        assert mixed_batch.coc_labels is not None and mixed_batch.coc_answer_mask is not None
        assert mixed_batch.coc_answer_mask[0].any() and not mixed_batch.coc_answer_mask[1].any()
        assert mixed_batch.coc_labels[1].eq(-100).all()
        mixed_prompts = [text for text in mixed_tokenizer.seen
                         if text.startswith(("As an autonomous driving system, ", "history "))]
        assert len(mixed_prompts) == 2
        assert "first generate chain-of-causation reasoning" in mixed_prompts[0]
        assert "first generate chain-of-causation reasoning" not in mixed_prompts[1]
        assert mixed_prompts[1].startswith("history [PT_HIST, ")
        assert mixed_prompts[1].endswith("command [GO STRAIGHT]")

        mixed_head.qwen3_vl_with_expert = types.SimpleNamespace(
            qwen3_vl=types.SimpleNamespace(lm_head=nn.Linear(4, 512, bias=False))
        )
        mixed_batch.prompt_lens = [1, 1]
        prefix_out = torch.randn(2, COC_MAX_NEW_TOKENS + 1, 4, requires_grad=True)
        mixed_loss = mixed_head._coc_language_loss(mixed_batch, prefix_out)
        valid_offsets = mixed_batch.coc_answer_mask[0].bool().nonzero(as_tuple=False).flatten()
        expected_loss = 0.1 * torch.nn.functional.cross_entropy(
            mixed_head.qwen3_vl_with_expert.qwen3_vl.lm_head(prefix_out[0, valid_offsets]),
            mixed_batch.coc_labels[0, valid_offsets],
        )
        assert torch.allclose(mixed_loss, expected_loss)
        mixed_loss.backward()
        assert prefix_out.grad is not None and prefix_out.grad[1].eq(0).all()
        assert prefix_out.grad[0].abs().sum() > 0

        empty_head, _ = make_head(path=path)
        empty_head.ar_loss_weight = 0.1
        empty_batch = empty_head._build_driving_batch(
            images[:1], commands[:1], hist_traj=history[:1], sample_tokens=["blank"],
            include_coc_targets=True,
        )
        assert empty_batch.images.shape[0] == 1 and empty_batch.coc_labels is None
        empty_head.qwen3_vl_with_expert = types.SimpleNamespace(
            qwen3_vl=types.SimpleNamespace(lm_head=nn.Linear(4, 512, bias=False))
        )
        empty_prefix_out = torch.randn(1, 2, 4, requires_grad=True)
        empty_loss = empty_head._coc_language_loss(empty_batch, empty_prefix_out)
        assert empty_loss.item() == 0.0 and empty_loss.requires_grad
        empty_loss.backward()
        assert empty_prefix_out.grad is not None and empty_prefix_out.grad.eq(0).all()

        from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy, UNPAD
        from unidrivevla.models.unidrivevla.dense_heads.canbus_prompt import coc_record_for_token

        mixture_tokens = ["source=0:nuscenes::abc", "source=1:nuscenes::def"]
        assert coc_record_for_token(table, mixture_tokens[0]) == table["abc"]
        assert coc_record_for_token(table, "abc") == table["abc"]
        assert coc_record_for_token(table, "source=nuscenes::abc") is None
        assert coc_record_for_token(table, "source=0:nuscenes::source=1:nuscenes::abc") is None
        assert coc_record_for_token(table, None) is None
        payload = {}
        for key, mask_key in UNPAD:
            payload[key] = torch.zeros(2, 1, 1)
            payload[mask_key] = torch.ones(2, 1, dtype=torch.bool)
        received = []

        def train_with_coc(**kwargs):
            received.extend(head._training_coc_targets(kwargs["sample_tokens"], 2))
            return {"losses": {"loss_coc": torch.tensor(0.0)}}

        policy = types.SimpleNamespace(training=True, head=types.SimpleNamespace(forward_train=train_with_coc))
        UniDriveVLAPolicy.forward(policy, payload, tokens=mixture_tokens)
        assert received == ["the car yields", "a pedestrian crosses"]
        assert mixture_tokens == ["source=0:nuscenes::abc", "source=1:nuscenes::def"]
        head.coc_by_token["source=1:nuscenes::abc"] = {"cot": "a different source stops"}
        assert head._training_coc_targets(["source=0:nuscenes::abc", "source=1:nuscenes::abc"], 2) == [
            "the car yields", "a different source stops",
        ]

        long_head, long_tokenizer = make_head(path=path, table={"long": {"cot": "x" * 200}})
        long_batch = long_head._build_driving_batch(
            images[:1], commands[:1], hist_traj=history[:1], sample_tokens=["long"],
            include_coc_targets=True,
        )
        assert long_batch.coc_labels.shape == long_batch.coc_answer_mask.shape == (1, COC_MAX_NEW_TOKENS)
        assert long_batch.coc_labels[0, -2:].tolist() == [
            long_tokenizer.special["<|im_end|>"], long_tokenizer.special["\n"],
        ]
        assert int(long_batch.coc_answer_mask[0].sum()) == COC_MAX_NEW_TOKENS
        assert long_batch.coc_labels[0, :COC_MAX_NEW_TOKENS - 2].tolist() == long_tokenizer.encode(
            "x" * (COC_MAX_NEW_TOKENS - 2)
        )

        missing_path_head, _ = make_head(path="")
        try:
            missing_path_head._training_coc_targets(["abc"], 1)
        except RuntimeError as exc:
            assert "coc_jsonl_path" in str(exc)
        else:
            raise AssertionError("CoC training silently accepted an empty annotation path")

        bad_path_head, _ = make_head(path=os.path.join(directory, "absent.jsonl"))
        try:
            bad_path_head._training_coc_targets(["abc"], 1)
        except OSError:
            pass
        else:
            raise AssertionError("CoC training silently accepted a nonexistent annotation file")

        malformed_path = os.path.join(directory, "malformed.jsonl")
        with open(malformed_path, "w", encoding="utf-8") as handle:
            handle.write('{"sample_token":\n')
        malformed_head, _ = make_head(path=malformed_path)
        try:
            malformed_head._training_coc_targets(["abc"], 1)
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("CoC training silently accepted malformed annotation JSON")

        missing_label_head, _ = make_head(path=path, table={"abc": {"cot": "only one label"}})
        try:
            missing_label_head._training_coc_targets(["abc", "missing"], 2)
        except RuntimeError as exc:
            assert "batch row 1" in str(exc)
        else:
            raise AssertionError("CoC training silently accepted a missing sample label")

        class ForbiddenTable:
            def get(self, *_args, **_kwargs):
                raise AssertionError("inference attempted a GT CoC lookup")

        infer_head, infer_tokenizer = make_head(path="/unavailable/train-only.jsonl", table=ForbiddenTable())
        infer_batch = infer_head._build_driving_batch(
            images, commands, hist_traj=history, sample_tokens=["abc", "def"], include_coc_targets=False,
        )
        assert infer_batch.coc_answer_ids is None
        inference_prompts = [text for text in infer_tokenizer.seen if text.startswith("As an autonomous driving system")]
        assert inference_prompts == training_prompts

        # B2D CoC opts into the same history-bearing prompt for training and inference.
        b2d_head, b2d_tokenizer = make_head(path=path)
        b2d_head.prompt_style = "b2d"
        b2d_train = b2d_head._build_driving_batch(
            images[:1], commands[:1], hist_traj=history[:1], sample_tokens=["abc"], include_coc_targets=True,
        )
        b2d_train_prompts = [
            text for text in b2d_tokenizer.seen if text.startswith("As an autonomous driving system")
        ]
        b2d_head._build_driving_batch(
            images[:1], commands[:1], hist_traj=history[:1], include_coc_targets=False,
        )
        b2d_infer_prompts = [
            text for text in b2d_tokenizer.seen if text.startswith("As an autonomous driving system")
        ][len(b2d_train_prompts):]
        assert b2d_train.coc_labels is not None
        assert b2d_train_prompts == b2d_infer_prompts and len(b2d_train_prompts) == 1
        assert "PT_HIST" in b2d_train_prompts[0]
        assert "6 future waypoints over 3.0 seconds" in b2d_train_prompts[0]
        assert "speed" not in b2d_train_prompts[0].lower()

        # LEAD's 8 x 0.25 s recipe describes its actual 2-second horizon.
        lead_head, lead_tokenizer = make_head(path=path)
        lead_head.prompt_style = "ego"
        lead_head.action_horizon = 8
        lead_head.pose_interval = 0.25
        lead_head._build_driving_batch(
            images[:1], commands[:1], hist_traj=history[:1], sample_tokens=["abc"], include_coc_targets=True,
        )
        lead_train_prompts = [
            text for text in lead_tokenizer.seen if text.startswith("As an autonomous driving system")
        ]
        lead_head._build_driving_batch(
            images[:1], commands[:1], hist_traj=history[:1], include_coc_targets=False,
        )
        lead_infer_prompts = [
            text for text in lead_tokenizer.seen if text.startswith("As an autonomous driving system")
        ][len(lead_train_prompts):]
        assert lead_train_prompts == lead_infer_prompts and len(lead_train_prompts) == 1
        assert "8 future waypoints over 2.0 seconds" in lead_train_prompts[0]

        try:
            format_ego_status(
                gt_ego_lcf_feat=None, gt_ego_his_trajs=None, ego_status=[0.0] * 6, hist_traj=None,
            )
        except ValueError as exc:
            assert "at least 10 values" in str(exc)
        else:
            raise AssertionError("short status was silently padded with fabricated CanBus values")

        from unidrivevla.adapters.unidrivevla import head_kwargs

        b2d_profile = types.SimpleNamespace(
            name="b2d", status_layout=["vel_x", "acc_x", "acc_y", "rot_x", "rot_y", "steer"]
        )
        try:
            head_kwargs(b2d_profile, "unused", None, False, False, canbus_sft_prompt=True)
        except ValueError as exc:
            assert "requires all 10 ego-status fields" in str(exc)
        else:
            raise AssertionError("B2D CanBus prompt accepted an incomplete ego-status profile")

    # The prefix already contains image/prompt KV. Decode only text tokens into that same cache.
    head, _ = make_head()
    head.cast_inference_mask = True
    head._inference_attn_impl = "sdpa"
    prefix_len = 5
    prefix_masks = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    prefix_ids = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 4, 0]], dtype=torch.long)
    positions = torch.tensor([
        [[0, 1, 1, 1, 1], [0, 1, 2, 3, 1]],
        [[0, 1, 1, 1, 1], [0, 1, 2, 3, 1]],
        [[0, 1, 1, 1, 1], [0, 1, 2, 3, 1]],
    ], dtype=torch.long)
    prefix_out = torch.zeros(2, prefix_len, 1)
    prefix_out[0, 1, 0] = 10  # sample 0: token, EOS
    prefix_out[1, 3, 0] = 11  # sample 1: token, token, token, EOS
    batch = types.SimpleNamespace(prompt_lens=[2, 4], coc_answer_ids=None)
    cache = FakeCache(prefix_len, 2)
    core = FakeCore(cache, prefix_out)
    head.qwen3_vl_with_expert = core
    deepstack = [torch.ones(2, 1, 1)]
    visual_mask = torch.tensor([[True, False, False, False, False], [True, False, False, False, False]])
    (prefix_out, _, _), prefilled_cache, _ = core.forward(
        attention_mask=torch.zeros(2, 1, prefix_len, prefix_len),
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[torch.zeros(2, prefix_len, 1), None, None],
        use_cache=True,
        deepstack_visual_embeds=deepstack,
        visual_pos_masks=visual_mask,
    )
    assert prefilled_cache is cache and len(core.prefill_calls) == 1
    assert core.prefill_calls[0][0] is deepstack
    assert torch.equal(core.prefill_calls[0][1], visual_mask)
    new_ids, new_masks, new_positions, updated_cache = head._generate_coc_tokens(
        batch, prefix_out, prefilled_cache, prefix_masks, positions,
    )
    assert updated_cache is cache and cache.visual_prefill
    assert core.image_encode_calls == 1  # generation did not re-encode the cameras
    assert new_ids.tolist() == [[10, 2, 3, 0, 0], [11, 12, 13, 2, 3]]
    assert new_masks.tolist() == [[True, True, True, False, False], [True, True, True, True, True]]
    assert new_ids.shape[1] <= COC_MAX_NEW_TOKENS
    assert cache.seq_len == prefix_len + new_ids.shape[1]  # cache stops at the longest EOS/newline answer
    assert [call["position_ids"][0, :, 0].tolist() for call in core.calls] == [
        [2, 4], [3, 5], [4, 6], [5, 7], [6, 8],
    ]
    first_allowed = core.calls[0]["attention_mask"][:, 0, 0].eq(0)
    assert first_allowed.tolist() == [[True, True, False, False, False, True],
                                      [True, True, True, True, False, True]]
    assert core.calls[0]["cache_len"] == prefix_len
    assert core.calls[0]["visual_prefill"]

    # Planner append must decode into its own, fresh image/prompt cache.
    append_cache = FakeCache(prefix_len, 2)
    append_core = FakeCore(append_cache, prefix_out)
    head.qwen3_vl_with_expert = append_core
    (_, _, _), append_prefilled_cache, _ = append_core.forward(
        attention_mask=torch.zeros(2, 1, prefix_len, prefix_len),
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[torch.zeros(2, prefix_len, 1), None, None],
        use_cache=True,
    )
    max_pos = torch.tensor([[1], [3]], dtype=torch.long)
    planned_ids, planned_masks, planned_cache, planned_max = head._append_coc_for_planning(
        batch, prefix_out, append_prefilled_cache, prefix_ids, prefix_masks, positions, max_pos,
    )
    assert planned_cache is append_cache and planned_ids.shape == planned_masks.shape == (2, prefix_len + 5)
    assert torch.equal(planned_ids[:, -5:], new_ids) and torch.equal(planned_masks[:, -5:], new_masks)
    planner_keys = planned_masks[:, None, :].expand(2, 1, planned_masks.shape[1])
    planner_attention = head._inference_mask(planner_keys, prefix_out.dtype)[:, 0, 0].eq(0)
    assert planner_attention[0, prefix_len:prefix_len + 3].all()
    assert not planner_attention[0, prefix_len + 3:].any()
    assert planner_attention[1, prefix_len:].all()
    assert planned_max.tolist() == [[4], [8]]

    # When EOS occupies the final permitted slot, keep it and omit only its newline delimiter.
    edge_head, _ = make_head()
    edge_head.cast_inference_mask = True
    edge_head._inference_attn_impl = "sdpa"
    edge_core = FakeCore(FakeCache(2, 1))
    edge_core.prefix_out = torch.zeros(1, 2, 1)
    edge_core.prefix_out[0, 1, 0] = 10  # one content token, then EOS
    edge_head.qwen3_vl_with_expert = edge_core
    edge_core.forward(
        attention_mask=torch.zeros(1, 1, 2, 2), position_ids=positions[:, :1, :2],
        past_key_values=None, inputs_embeds=[torch.zeros(1, 2, 1), None, None], use_cache=True,
    )
    edge_ids, edge_mask, _, _ = edge_head._generate_coc_tokens(
        types.SimpleNamespace(prompt_lens=[2]), edge_core.prefix_out, edge_core.cache,
        torch.ones(1, 2, dtype=torch.bool), positions[:, :1, :2], max_new_tokens=2,
    )
    assert edge_ids.tolist() == [[10, 2]] and edge_mask.all()

    # A model that never emits EOS is still bounded by the hard 128-token release ceiling.
    one_head, _ = make_head()
    one_head.cast_inference_mask = True
    one_head._inference_attn_impl = "sdpa"
    endless = EndlessCore(FakeCache(2, 1))
    one_head.qwen3_vl_with_expert = endless
    one_prefix = torch.zeros(1, 2, 1)
    one_prefix[0, 1, 0] = 10
    one_masks = torch.tensor([[True, True]])
    one_positions = torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]])
    endless.prefix_out = one_prefix
    endless.forward(
        attention_mask=torch.zeros(1, 1, 2, 2), position_ids=one_positions,
        past_key_values=None, inputs_embeds=[torch.zeros(1, 2, 1), None, None], use_cache=True,
    )
    one_batch = types.SimpleNamespace(prompt_lens=[2])
    ids, mask, _, _ = one_head._generate_coc_tokens(
        one_batch, one_prefix, endless.cache, one_masks, one_positions, max_new_tokens=999,
    )
    assert ids.shape == mask.shape == (1, COC_MAX_NEW_TOKENS)
    assert mask.all()
    print("CoC labels/truncation, train/inference isolation, greedy EOS/limit, KV positions, and planner visibility passed")


if __name__ == "__main__":
    main()
