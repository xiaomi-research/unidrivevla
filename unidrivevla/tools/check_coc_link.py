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
from unidrivevla.models.unidrivevla.dense_heads.utils import make_att_2d_masks  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 4
    special = {"<|im_start|>": 1, "<|im_end|>": 2, "\n": 3, "<|eos|>": 4}
    decoded = {10: "keep", 11: "turn", 12: "left", 13: "safely", 14: "yield"}
    round_trip = {
        "keep lane": [20, 21],
        "turn left safely": [22, 23, 24],
        "yield": [25],
    }

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        if text in self.special:
            return [self.special[text]]
        if text in self.round_trip:
            return list(self.round_trip[text])
        return [100 + ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "  " + " ".join(self.decoded[token] for token in token_ids if token in self.decoded) + "  "


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
        self.language_model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="eager"),
            layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(q_proj=nn.Linear(1, 1, bias=False)))],
        )
        self.embedding = FakeEmbedding()
        self.config = types.SimpleNamespace(
            image_token_id=8, vision_start_token_id=5, vision_end_token_id=6,
        )
        self.visual = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="eager"))

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
        self.perception_calls = 0
        self.image_encode_calls = 1  # the caller already did one visual prefill
        self.qwen3_perception_expert = types.SimpleNamespace(config=types.SimpleNamespace())
        self.qwen3_action_expert = types.SimpleNamespace(config=types.SimpleNamespace())

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache,
                deepstack_visual_embeds=None, visual_pos_masks=None):
        if past_key_values is None:
            assert use_cache and self.prefix_out is not None
            assert attention_mask.shape[-1] == inputs_embeds[0].shape[1]
            self.prefill_calls.append((
                deepstack_visual_embeds, visual_pos_masks, attention_mask.detach().clone(),
                position_ids.detach().clone(), past_key_values, tuple(inputs_embeds[0].shape),
            ))
            return [self.prefix_out, None, None], self.cache, {}
        assert use_cache and past_key_values is self.cache
        if inputs_embeds[0] is None:
            self.perception_calls += 1
            return [None, None, None], self.cache, {}
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
                next_id = 4 if history else 10
            elif row == 1 and len(history) == 0:
                next_id = 11
            elif row == 1 and len(history) == 1:
                next_id = 12
            elif row == 1 and len(history) == 2:
                next_id = 13
            elif row == 2 and len(history) == 0:
                next_id = 14
            elif row == 2:
                next_id = 0
            elif row == 1:
                next_id = 2
            else:
                next_id = 4
            next_ids.append(next_id)
        self.cache.seq_len += 1
        hidden = torch.tensor(next_ids, dtype=torch.float32).view(-1, 1, 1)
        return [hidden, None, None], self.cache, {}


class EndlessCore(FakeCore):
    def forward(self, **kwargs):
        outputs, cache, middle = super().forward(**kwargs)
        outputs[0].fill_(10.0)  # never emits EOS
        return outputs, cache, middle


class FakePrefixCore:
    def __init__(self, tokenizer):
        self.qwen3_vl = FakeQwen()
        self.processor = types.SimpleNamespace(tokenizer=tokenizer)

    def embed_image_tensor(self, images):
        batch_size, views = images.shape[:2]
        lens = [3] * (batch_size * views)
        features = torch.ones(sum(lens), 1)
        grids = [torch.tensor([1, 1, 3])] * len(lens)
        return features, lens, grids, None, None


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
        assert mixed_batch.coc_answer_ids[1].eq(mixed_tokenizer.pad_token_id).all()
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
        assert empty_batch.images.shape[0] == 1 and empty_batch.coc_labels is not None
        assert empty_batch.coc_answer_ids.shape == empty_batch.coc_answer_mask.shape == (1, COC_MAX_NEW_TOKENS)
        assert empty_batch.coc_answer_lens == [0]
        assert not empty_batch.coc_answer_mask.any() and empty_batch.coc_labels.eq(-100).all()
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

        boundary_texts = {f"len{length}": {"cot": "x" * length} for length in (126, 127, 128)}
        long_head, long_tokenizer = make_head(path=path, table=boundary_texts)
        long_batch = long_head._build_driving_batch(
            torch.zeros(3, 1, 3, 4, 4), torch.tensor([0, 1, 2]),
            hist_traj=history[:1].expand(3, -1, -1), sample_tokens=list(boundary_texts),
            include_coc_targets=True,
        )
        assert long_batch.coc_labels.shape == long_batch.coc_answer_mask.shape == (3, COC_MAX_NEW_TOKENS)
        assert long_batch.coc_answer_lens == [128, 128, 128]
        content_id = 100 + ord("x")
        assert long_batch.coc_labels[0, -3:].tolist() == [
            content_id, long_tokenizer.special["<|im_end|>"], long_tokenizer.special["\n"],
        ]  # 126 content + both terminators fit exactly
        assert long_batch.coc_labels[1, -2:].tolist() == [
            content_id, long_tokenizer.special["<|im_end|>"],
        ]  # appending both terminators first then capping at 128 drops only newline
        assert long_batch.coc_labels[2].eq(content_id).all()  # 128 content tokens leave no room for terminators
        assert long_batch.coc_answer_mask.eq(1).all()

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

    # Generation stops independently on EOS, im_end, and PAD; each row decodes only content tokens.
    head, tokenizer = make_head(path="/unavailable/train-only.jsonl", table=None)
    head.cast_inference_mask = True
    head._inference_attn_impl = "sdpa"
    prefix_len = 5
    prefix_masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
    ], dtype=torch.bool)
    positions = torch.tensor([
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 0], [0, 1, 2, 0, 0]],
        [[0, 1, 9, 9, 4], [0, 1, 8, 0, 0], [0, 1, 7, 0, 0]],
        [[0, 1, 9, 9, 4], [0, 1, 8, 0, 0], [0, 1, 7, 0, 0]],
    ], dtype=torch.long)
    prefix_out = torch.zeros(3, prefix_len, 1)
    prefix_out[0, 4, 0] = 10  # content then the distinct tokenizer EOS id
    prefix_out[1, 3, 0] = 11  # three content tokens then im_end
    prefix_out[2, 2, 0] = 14  # content then PAD
    batch = types.SimpleNamespace(prompt_lens=[5, 4, 3], coc_answer_ids=None)
    cache = FakeCache(prefix_len, 3)
    core = FakeCore(cache, prefix_out)
    core.processor.tokenizer = tokenizer
    head.qwen3_vl_with_expert = core
    deepstack = [torch.ones(3, 1, 1)]
    visual_mask = torch.tensor([
        [True, False, False, False, False],
        [True, False, False, False, False],
        [True, False, False, False, False],
    ])
    (prefix_out, _, _), prefilled_cache, _ = core.forward(
        attention_mask=torch.zeros(3, 1, prefix_len, prefix_len),
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[torch.zeros(3, prefix_len, 1), None, None],
        use_cache=True,
        deepstack_visual_embeds=deepstack,
        visual_pos_masks=visual_mask,
    )
    assert prefilled_cache is cache and len(core.prefill_calls) == 1
    assert core.prefill_calls[0][0] is deepstack
    assert torch.equal(core.prefill_calls[0][1], visual_mask)
    decoded_texts = head._generate_coc_tokens(
        batch, prefix_out, prefilled_cache, prefix_masks, positions,
    )
    assert decoded_texts == ["keep", "turn left safely", "yield"]
    assert core.qwen3_vl.language_model.config._attn_implementation == "eager"
    assert core.image_encode_calls == 1  # generation did not re-encode the cameras
    assert cache.seq_len == prefix_len + 3
    assert len(core.calls) == 3  # the EOS/PAD rows stop while the third row continues
    assert [call["position_ids"][:, :, 0].T.tolist() for call in core.calls] == [
        [[10, 10, 10], [9, 9, 9], [8, 8, 8]],
        [[11, 11, 11], [10, 10, 10], [9, 9, 9]],
        [[12, 12, 12], [11, 11, 11], [10, 10, 10]],
    ]  # use the maximum MRoPE axis, not the last text position
    first_allowed = core.calls[0]["attention_mask"][:, 0, 0].eq(0)
    assert first_allowed.tolist() == [
        [True, True, True, True, True, True],
        [True, True, True, True, False, True],
        [True, True, True, False, False, True],
    ]
    assert core.calls[0]["cache_len"] == prefix_len and core.calls[0]["visual_prefill"]

    # Rebuild a fresh assistant-answer prefix from decoded text; keep each row's command and image.
    rebuild_head, rebuild_tokenizer = make_head(path="/unavailable/train-only.jsonl", table=None)
    rebuild_images = torch.zeros(3, 1, 3, 4, 4)
    rebuild_commands = torch.tensor([0, 1, 2])
    rebuild_history = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]).expand(3, -1, -1)
    rebuild_batch = rebuild_head._build_driving_batch(
        rebuild_images, rebuild_commands, hist_traj=rebuild_history, include_coc_targets=False,
    )
    rebuild_tokenizer.seen.clear()
    planned_batch = rebuild_head._rebuild_planning_batch_with_coc(
        rebuild_batch, ["  keep lane  ", " turn left safely ", " yield "], hist_traj=rebuild_history,
    )
    assert planned_batch.images.shape[0] == 3
    assert torch.equal(planned_batch.images, rebuild_images)
    assert planned_batch.command.tolist() == [0, 1, 2]
    assert planned_batch.coc_answer_ids.shape == (3, COC_MAX_NEW_TOKENS)
    assert planned_batch.coc_answer_ids[:, :5].tolist() == [
        [20, 21, 2, 3, 0], [22, 23, 24, 2, 3], [25, 2, 3, 0, 0],
    ]
    assert planned_batch.coc_answer_lens == [4, 5, 3]
    planned_prompts = [
        text for text in rebuild_tokenizer.seen if text.startswith("As an autonomous driving system")
    ]
    assert len(planned_prompts) == 3
    assert all("chain-of-causation reasoning" in text for text in planned_prompts)
    assert [next(command for command in rebuild_head.command_prompts if command in prompt)
            for prompt in planned_prompts] == ["TURN RIGHT", "TURN LEFT", "GO STRAIGHT"]

    # Padded multi-row answers stay attached to each sample's own prompt and command slot.
    rebuild_head.prefix_max_len = 1000
    rebuild_head.action_in_proj = nn.Linear(1, 1, bias=False)
    rebuild_head.qwen3_vl_with_expert = FakePrefixCore(rebuild_tokenizer)
    padded_prefix = rebuild_head.embed_prefix(planned_batch)
    assert padded_prefix[4].shape == (3, 1000 + COC_MAX_NEW_TOKENS)
    assert len(set(planned_batch.prompt_lens)) > 1
    for row, answer_len in enumerate(planned_batch.coc_answer_lens):
        start = planned_batch.prompt_lens[row]
        assert padded_prefix[4][row, start:start + answer_len].tolist() == \
            planned_batch.coc_answer_ids[row, :answer_len].tolist()
        assert padded_prefix[1][row, start:start + answer_len].all()

    # Blank decoded text remains the ordinary prompt, with no phantom assistant answer.
    empty_batch = rebuild_head._build_driving_batch(
        rebuild_images[:1], rebuild_commands[:1], hist_traj=rebuild_history[:1], include_coc_targets=False,
    )
    rebuild_tokenizer.seen.clear()
    empty_planned = rebuild_head._rebuild_planning_batch_with_coc(
        empty_batch, ["   "], hist_traj=rebuild_history[:1],
    )
    assert empty_planned.coc_answer_ids.shape == empty_planned.coc_answer_mask.shape == (1, COC_MAX_NEW_TOKENS)
    assert empty_planned.coc_answer_lens == [0]
    assert not empty_planned.coc_answer_mask.any()
    assert not any("chain-of-causation reasoning" in text for text in rebuild_tokenizer.seen)

    # The prefix cap keeps all image tokens and the newest user-prompt tokens.
    prefix_head, prefix_tokenizer = make_head()
    prefix_head.prefix_max_len = 10
    prefix_head.action_in_proj = nn.Linear(1, 1, bias=False)
    prefix_head.qwen3_vl_with_expert = FakePrefixCore(prefix_tokenizer)
    long_images = torch.zeros(1, 1, 3, 4, 4)
    long_command = torch.tensor([0])
    long_history = torch.zeros(1, 2, 2)
    prefix_batch = prefix_head._build_driving_batch(
        long_images, long_command, hist_traj=long_history, include_coc_targets=False,
    )
    prefix_result = prefix_head.embed_prefix(prefix_batch)
    vision_prefix = [42, 3, 5, 8, 8, 8, 6, 3]
    prompt_ids = prefix_batch.tokenized_prompt[0][prefix_batch.tokenized_prompt_mask[0].bool()].tolist()
    assert prefix_result[4].shape == (1, 10)
    assert prefix_result[4][0, :8].tolist() == vision_prefix
    assert prefix_result[4][0, 8:].tolist() == prompt_ids[-2:]
    assert prefix_batch.prompt_lens == [10]
    assert int((prefix_result[4] == 8).sum()) == 3

    with tempfile.TemporaryDirectory() as answer_prefix_dir:
        answer_prefix_path = os.path.join(answer_prefix_dir, "coc.jsonl")
        with open(answer_prefix_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"sample_token": "label", "cot": "keep lane"}) + "\n")
        answer_prefix_head, answer_prefix_tokenizer = make_head(path=answer_prefix_path)
        answer_prefix_head.prefix_max_len = 10
        answer_prefix_head.action_in_proj = nn.Linear(1, 1, bias=False)
        answer_prefix_head.qwen3_vl_with_expert = FakePrefixCore(answer_prefix_tokenizer)
        answer_prefix_batch = answer_prefix_head._build_driving_batch(
            long_images, long_command, hist_traj=long_history, sample_tokens=["label"], include_coc_targets=True,
        )
        answer_prefix_result = answer_prefix_head.embed_prefix(answer_prefix_batch)
        assert answer_prefix_result[4].shape == (1, 10 + COC_MAX_NEW_TOKENS)
        assert answer_prefix_result[4][0, :8].tolist() == vision_prefix
        assert answer_prefix_result[4][0, 8:10].tolist() == prompt_ids[-2:]
        assert answer_prefix_result[4][0, 10:14].tolist() == [20, 21, 2, 3]
        assert answer_prefix_batch.prompt_lens == [10]

    empty_prefix_head, empty_prefix_tokenizer = make_head()
    empty_prefix_head.prefix_max_len = 10
    empty_prefix_head.action_in_proj = nn.Linear(1, 1, bias=False)
    empty_prefix_head.qwen3_vl_with_expert = FakePrefixCore(empty_prefix_tokenizer)
    empty_prefix_batch = empty_prefix_head._build_driving_batch(
        long_images, long_command, hist_traj=long_history, include_coc_targets=True,
        coc_override_texts=[""],
    )
    empty_prefix_result = empty_prefix_head.embed_prefix(empty_prefix_batch)
    assert empty_prefix_result[4].shape == (1, 10 + COC_MAX_NEW_TOKENS)
    assert empty_prefix_batch.coc_answer_lens == [0]
    assert empty_prefix_batch.coc_answer_ids[0].eq(empty_prefix_tokenizer.pad_token_id).all()
    assert not empty_prefix_result[1][0, 10:].any()

    # Exercise the public inference path: generation uses the first prefill; planning gets a second, fresh one.
    forward_head, forward_tokenizer = make_head(path="/unavailable/train-only.jsonl", table=None)
    forward_head.action_horizon = 6
    forward_head.action_dim = 2
    forward_head.ego_status_dim = 6
    forward_head.num_sample_steps = 1
    forward_head.hf_attn_impl = "sdpa"
    forward_head._inference_attn_impl = "sdpa"
    forward_head.cast_inference_mask = True
    forward_head.coc_language_prompt = True
    forward_head.driving_deepstack = False
    forward_head.feature_source = "raw"
    forward_head.feat_grad = False
    forward_head.vlm_hidden_size = 1

    class DummyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward_stage1(self, *_args, **_kwargs):
            return {}

    forward_head.unified_decoder = DummyDecoder()
    forward_core = FakeCore(FakeCache(2, 1))
    forward_core.processor.tokenizer = forward_tokenizer
    forward_head.qwen3_vl_with_expert = forward_core
    prefix_builds = []

    def fake_embed_prefix(active_batch):
        if active_batch.coc_answer_ids is None:
            ids = torch.tensor([[100, 101]], dtype=torch.long)
            forward_core.prefix_out = torch.zeros(1, 2, 1)
            forward_core.prefix_out[0, 1, 0] = 10
        else:
            answer_len = int(active_batch.coc_answer_mask[0].sum())
            answer = active_batch.coc_answer_ids[0, :answer_len]
            ids = torch.cat((torch.tensor([100, 101]), answer)).view(1, -1)
            forward_core.prefix_out = torch.zeros(1, ids.shape[1], 1)
        active_batch.prompt_lens = [2]
        length = ids.shape[1]
        forward_core.cache = FakeCache(length, 1)
        prefix_builds.append({
            "answer": None if active_batch.coc_answer_ids is None else active_batch.coc_answer_ids[0].tolist(),
            "length": length,
            "ids": ids.clone(),
        })
        pad = torch.ones((1, length), dtype=torch.bool)
        marks = torch.ones((1, length), dtype=torch.long)
        return torch.zeros(1, length, 1), pad, marks, [], ids, None, torch.zeros(1, 1, 1), length

    forward_head.embed_prefix = fake_embed_prefix
    forward_head.get_position_ids = lambda ids, _grids, _pad: (
        torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, ids.shape[0], -1), None,
    )
    forward_head.project_and_reshape_features = lambda *_args: []
    forward_head.prepare_for_deformable_aggregation = lambda values: values
    forward_head.embed_perception = lambda b, device, _dtype, _stage1: (
        torch.empty((b, 0, 1), device=device),
        torch.empty((b, 0), dtype=torch.bool, device=device),
        torch.empty((b, 0), dtype=torch.bool, device=device),
        {"det": 0, "map": 0, "occ": 0, "ego": 0, "motion": 0},
    )
    forward_head.reset_temporal = lambda: None
    forward_head._denoise_step = lambda _batch, _mask, _cache, x, *_args, **_kwargs: torch.zeros_like(x)
    forward_head.denorm_delta = lambda value: value
    result = forward_head.forward_test(
        img=torch.zeros(1, 1, 3, 4, 4),
        gt_ego_fut_cmd=torch.tensor([0]),
        hist_traj=torch.zeros(1, 2, 2),
        img_metas=[{}],
        noise=torch.zeros(1, 6, 2),
        num_steps=1,
    )
    assert result["traj"].shape == (1, 6, 2)
    assert len(prefix_builds) == len(forward_core.prefill_calls) == 2
    assert len(forward_core.calls) == 1  # one generated content step, then EOS
    assert prefix_builds[0]["answer"] is None and prefix_builds[1]["answer"] is not None
    assert prefix_builds[1]["answer"][:6] == [207, 201, 201, 212, 2, 3]
    assert all(call[4] is None for call in forward_core.prefill_calls)  # both prefills start without a KV cache
    assert forward_core.prefill_calls[1][5] == (1, prefix_builds[1]["length"], 1)
    assert forward_core.perception_calls == 1
    final_len = prefix_builds[1]["length"]
    final_pad = torch.ones((1, final_len), dtype=torch.bool)
    final_marks = torch.ones((1, final_len), dtype=torch.long)
    expected_prefill_mask = forward_head._inference_mask(
        make_att_2d_masks(final_pad, final_marks), torch.float32,
    )
    assert torch.equal(forward_core.prefill_calls[1][2], expected_prefill_mask)
    expected_positions = torch.arange(final_len).view(1, 1, -1).expand(3, 1, -1)
    assert torch.equal(forward_core.prefill_calls[1][3], expected_positions)

    # A model that emits EOS at the last allowed slot still returns only content text.
    edge_head, edge_tokenizer = make_head()
    edge_head.cast_inference_mask = True
    edge_head._inference_attn_impl = "sdpa"
    edge_core = FakeCore(FakeCache(2, 1))
    edge_core.processor.tokenizer = edge_tokenizer
    edge_core.prefix_out = torch.zeros(1, 2, 1)
    edge_core.prefix_out[0, 1, 0] = 10
    edge_head.qwen3_vl_with_expert = edge_core
    edge_core.forward(
        attention_mask=torch.zeros(1, 1, 2, 2),
        position_ids=torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]]),
        past_key_values=None, inputs_embeds=[torch.zeros(1, 2, 1), None, None], use_cache=True,
    )
    edge_text, = edge_head._generate_coc_tokens(
        types.SimpleNamespace(prompt_lens=[2]), edge_core.prefix_out, edge_core.cache,
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]]),
        max_new_tokens=2,
    )
    assert edge_text == "keep"

    # A model that never emits EOS is still bounded by the hard 128-token release ceiling.
    one_head, _ = make_head()
    one_head.cast_inference_mask = True
    one_head._inference_attn_impl = "sdpa"
    endless = EndlessCore(FakeCache(2, 1))
    endless.processor.tokenizer = one_head.qwen3_vl_with_expert.processor.tokenizer
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
    endless_text, = one_head._generate_coc_tokens(
        one_batch, one_prefix, endless.cache, one_masks, one_positions, max_new_tokens=999,
    )
    assert len(endless_text.split()) == COC_MAX_NEW_TOKENS
    print("CoC label boundaries, empty rows, EOS/PAD stopping, decode/re-encode, prompt crop, and RoPE positions passed")


if __name__ == "__main__":
    main()
