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

import copy
import math
from typing import Literal, Optional, Dict, Any, List
import glob
import os

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.auto import CONFIG_MAPPING

def _pr10_rmsnorm(self, hidden_states):
    # PR10 multiplies the fp32 norm by the weight, then casts. Stock transformers casts first.
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight * hidden_states).to(input_dtype)

modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = _pr10_rmsnorm
from transformers import AutoConfig, AutoProcessor

from qwen_vl_utils import smart_resize
from PIL import Image

from .flex_attention_opt import flex_attention_forward_optimized

NUSCENES_VIEW_TOKENS = [
    "<FRONT_VIEW>",
    "<FRONT_LEFT_VIEW>",
    "<FRONT_RIGHT_VIEW>",
    "<BACK_LEFT_VIEW>",
    "<BACK_RIGHT_VIEW>",
    "<BACK_VIEW>",
]

def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _unwrap_lm(language_model):
    base = getattr(language_model, "base_model", None)
    if base is not None and hasattr(base, "model"):
        return base.model
    return language_model


def cache_kv(cache):
    """Per-layer (key, value) of a transformers KV cache."""
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)


def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    qwen3_vl,
    qwen3_perception_expert,
    qwen3_action_expert,
    attn_implementation: str = "eager",
    q_len_rounded: int = None,
    deepstack_visual_embeds=None,
    visual_pos_masks=None,
    prefix_kv=None,
):
    """One layer of the streams in inputs_embeds (None: absent). prefix_kv, the prefix's cached (key, value) of this
    layer (rotary applied), goes in front of the keys when the prefix stream itself is absent."""
    base_lm = _unwrap_lm(qwen3_vl.language_model)
    models = [base_lm, qwen3_perception_expert, qwen3_action_expert]

    query_states = []
    key_states = []
    value_states = []

    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is None:
            continue
        layer = models[i].layers[layer_idx]

        hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )

    cos, sin = base_lm.rotary_emb(dummy_tensor, position_ids)

    query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    if prefix_kv is not None:
        key_states = torch.cat([prefix_kv[0].to(key_states.dtype), key_states], dim=2)
        value_states = torch.cat([prefix_kv[1].to(value_states.dtype), value_states], dim=2)

    batch_size = query_states.shape[0]
    head_dim = base_lm.layers[layer_idx].self_attn.head_dim
    num_heads = base_lm.layers[layer_idx].self_attn.config.num_attention_heads

    scaling = base_lm.layers[layer_idx].self_attn.scaling

    if attn_implementation == "flex":
        att_output = flex_attention_forward_optimized(
            query_states,
            key_states,
            value_states,
            block_mask=attention_mask,
            scaling=scaling,
            q_len_rounded=q_len_rounded
        )

    elif attn_implementation == "sdpa":
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                raise ValueError(
                    "SDPA backend expects an additive attention_mask (float) like OpenPI (0 for allow, -inf for block)."
                )
            if attention_mask.dim() != 4:
                raise ValueError(
                    f"SDPA backend expects 4D attention_mask (B,1|H,Q,K); got {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape[-2] != query_states.shape[2] or attention_mask.shape[-1] != key_states.shape[2]:
                raise ValueError(
                    "SDPA attention_mask shape mismatch: "
                    f"mask(Q,K)=({attention_mask.shape[-2]},{attention_mask.shape[-1]}) "
                    f"but Q,K=({query_states.shape[2]},{key_states.shape[2]})."
                )
            attention_mask = attention_mask.to(query_states.dtype)  # SDPA wants the bias in the query dtype

        num_kv_heads = base_lm.layers[layer_idx].self_attn.config.num_key_value_heads
        n_rep = num_heads // num_kv_heads

        if n_rep * num_kv_heads != num_heads:
            raise ValueError(f"Invalid GQA config: num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}")

        if n_rep > 1:
            key_states = repeat_kv(key_states, n_rep)
            value_states = repeat_kv(value_states, n_rep)

        sdpa_kwargs = {
            "dropout_p": 0.0,
            "is_causal": False,
        }
        try:
            att_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                scale=scaling,
                **sdpa_kwargs,
            )
        except TypeError:
            att_output = F.scaled_dot_product_attention(
                query_states * scaling,
                key_states,
                value_states,
                attn_mask=attention_mask,
                **sdpa_kwargs,
            )

        att_output = att_output.transpose(1, 2).contiguous()
        att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

    elif attn_implementation == "eager":
        att_output, _ = modeling_qwen3_vl.eager_attention_forward(
            base_lm.layers[layer_idx].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
        att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

    else:
        raise ValueError(f"Unknown attn_implementation: {attn_implementation}")

    outputs_embeds = []
    start_pos = 0

    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is None:
            outputs_embeds.append(None)
            continue
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]

        curr_att_out = att_output[:, start_pos:end_pos]
        if curr_att_out.dtype != layer.self_attn.o_proj.weight.dtype:
            curr_att_out = curr_att_out.to(layer.self_attn.o_proj.weight.dtype)

        out_emb = layer.self_attn.o_proj(curr_att_out)

        out_emb = out_emb + hidden_states
        after_first_residual = out_emb.clone()

        out_emb = layer.post_attention_layernorm(out_emb)

        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)

        out_emb = layer.mlp(out_emb)

        out_emb = out_emb + after_first_residual

        outputs_embeds.append(out_emb)
        start_pos = end_pos

    if (deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and outputs_embeds[0] is not None
            and layer_idx < len(deepstack_visual_embeds)):
        vlm_out = outputs_embeds[0]
        ds_feat = deepstack_visual_embeds[layer_idx]
        ds_feat = ds_feat.to(device=vlm_out.device, dtype=vlm_out.dtype)
        vlm_out = vlm_out.clone()
        vlm_out[visual_pos_masks] = vlm_out[visual_pos_masks] + ds_feat
        outputs_embeds[0] = vlm_out

    return outputs_embeds




class Qwen3VLWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        perception_expert_config,
        action_expert_config,
        pretrained_path,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        train_vlm: bool = False,
        lora_cfg: Optional[Dict[str, Any]] = None,
        peft_cfg: Optional[Dict[str, Any]] = None,
        visual_lora_cfg: Optional[Dict[str, Any]] = None,
        visual_wasp_cfg: Optional[Dict[str, Any]] = None,
        visual_projector_wasp_cfg: Optional[Dict[str, Any]] = None,
        view_tokens: Optional[List[str]] = None,
        processor_use_fast: Optional[bool] = None,
    ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["qwen3_vl"]()
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads
        vlm_config_hf.text_config.max_position_embeddings = 262144
        vlm_config_hf.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        }
        is_8b = (vlm_config.hidden_size == 4096)
        if is_8b:
            vlm_config_hf.text_config.tie_word_embeddings = False
            vlm_config_hf.tie_word_embeddings = False
            vlm_config_hf.vision_config.deepstack_visual_indexes = [8, 16, 24]
            vlm_config_hf.vision_config.depth = 27
            vlm_config_hf.vision_config.hidden_size = 1152
            vlm_config_hf.vision_config.intermediate_size = 4304
            vlm_config_hf.vision_config.out_hidden_size = 4096
        else:
            vlm_config_hf.text_config.tie_word_embeddings = True
            vlm_config_hf.tie_word_embeddings = True
            vlm_config_hf.vision_config.deepstack_visual_indexes = [5, 11, 17]
            vlm_config_hf.vision_config.depth = 24
            vlm_config_hf.vision_config.hidden_size = 1024
            vlm_config_hf.vision_config.intermediate_size = 4096
            vlm_config_hf.vision_config.out_hidden_size = 2048

        self.qwen3_vl = Qwen3VLForConditionalGeneration(config=vlm_config_hf)
        self.pretrained_path = pretrained_path

        safetensor_files = sorted(
            glob.glob(os.path.join(pretrained_path, "*.safetensors"))
        )

        state_dict = {}

        for file in safetensor_files:
            sd = load_file(file, device="cpu")

            for k, v in sd.items():
                if "action_preprocessor.normalizer" in k:
                    continue

                new_key = k

                if new_key.startswith("model.layers."):
                    new_key = new_key.replace(
                        "model.layers.",
                        "model.language_model.layers.",
                        1
                    )

                elif new_key.startswith("model.embed_tokens."):
                    new_key = new_key.replace(
                        "model.embed_tokens.",
                        "model.language_model.embed_tokens.",
                        1
                    )

                elif new_key.startswith("model.norm."):
                    new_key = new_key.replace(
                        "model.norm.",
                        "model.language_model.norm.",
                        1
                    )

                elif new_key.startswith("visual."):
                    new_key = "model.visual." + new_key[len("visual.") :]

                state_dict[new_key] = v

        self.qwen3_vl.load_state_dict(state_dict, strict=False)

        perception_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        perception_expert_config_hf.hidden_size = perception_expert_config.hidden_size
        perception_expert_config_hf.intermediate_size = perception_expert_config.intermediate_size
        perception_expert_config_hf.num_attention_heads = perception_expert_config.num_attention_heads
        perception_expert_config_hf.num_hidden_layers = vlm_config.num_hidden_layers
        perception_expert_config_hf.num_key_value_heads = perception_expert_config.num_key_value_heads
        perception_expert_config_hf.max_position_embeddings = self.qwen3_vl.config.text_config.max_position_embeddings
        perception_expert_config_hf.rope_scaling = self.qwen3_vl.config.text_config.rope_scaling
        self.qwen3_perception_expert = Qwen3VLTextModel(config=perception_expert_config_hf)
        self.qwen3_perception_expert.embed_tokens = None

        action_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        action_expert_config_hf.head_dim=action_expert_config.head_dim
        action_expert_config_hf.hidden_size=action_expert_config.hidden_size
        action_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        action_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        action_expert_config_hf.num_hidden_layers=vlm_config.num_hidden_layers
        action_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        action_expert_config_hf.max_position_embeddings = self.qwen3_vl.config.text_config.max_position_embeddings
        action_expert_config_hf.rope_scaling = self.qwen3_vl.config.text_config.rope_scaling
        self.qwen3_action_expert = Qwen3VLTextModel(config=action_expert_config_hf)

        self.qwen3_action_expert.embed_tokens = None

        if lora_cfg is not None and peft_cfg is not None:
            raise ValueError("Use only one of lora_cfg or peft_cfg, not both.")
        self._lora_enabled = False
        self._language_lora_enabled = False
        self._language_wasp_enabled = False
        self._visual_lora_enabled = False
        self._visual_wasp_enabled = False
        self._visual_projector_wasp_enabled = False
        self._language_adapter_param_ids = set()
        self._visual_lora_param_ids = set()
        self._visual_wasp_param_ids = set()
        self._visual_projector_wasp_param_ids = set()
        active_language_peft_cfg = peft_cfg
        if lora_cfg is not None:
            active_language_peft_cfg = dict(lora_cfg)
            active_language_peft_cfg.setdefault("type", "lora")
        if active_language_peft_cfg is not None:
            language_peft_options = dict(active_language_peft_cfg)
            language_peft_type = str(language_peft_options.pop("type", "lora")).lower()
            if language_peft_type == "lora":
                try:
                    from peft import LoraConfig, get_peft_model
                except ImportError as exc:
                    raise ImportError("peft is required for LoRA training. Install it with: pip install peft") from exc
                self.qwen3_vl.model.language_model = get_peft_model(
                    self.qwen3_vl.model.language_model, LoraConfig(**language_peft_options))
                self.qwen3_vl.model.language_model.print_trainable_parameters()
                self._language_lora_enabled = True
            elif language_peft_type == "wasp":
                from .wasp import apply_wasp
                replaced = apply_wasp(self.qwen3_vl.model.language_model, language_peft_options)
                expected_layers = language_peft_options.get("expected_num_layers")
                if expected_layers is not None and replaced != int(expected_layers):
                    raise RuntimeError(f"Language WASP target count mismatch: expected {expected_layers}, replaced {replaced}.")
                self._language_wasp_enabled = True
            else:
                raise ValueError(f"peft_cfg only supports type='lora' or type='wasp', got {language_peft_type!r}.")
            self._lora_enabled = True
            self._language_adapter_param_ids = {id(param) for param in self.qwen3_vl.model.language_model.parameters() if param.requires_grad}
            if not self._language_adapter_param_ids:
                raise RuntimeError("Language PEFT created no trainable parameters; check target_modules.")
        if visual_lora_cfg is not None:
            visual_lora_options = dict(visual_lora_cfg)
            if str(visual_lora_options.pop("type", "lora")).lower() != "lora":
                raise ValueError("visual_lora_cfg only supports type='lora'")
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError("peft is required for visual LoRA training. Install it with: pip install peft") from exc
            visual = getattr(self.qwen3_vl.model, "visual", None)
            if visual is None:
                raise AttributeError("Qwen3-VL model has no visual module for visual LoRA.")
            self.qwen3_vl.model.visual = get_peft_model(visual, LoraConfig(**visual_lora_options))
            self._visual_lora_param_ids = {id(param) for param in self.qwen3_vl.model.visual.parameters() if param.requires_grad}
            if not self._visual_lora_param_ids:
                raise RuntimeError("visual_lora_cfg created no trainable parameters; check target_modules.")
            self._lora_enabled = True
            self._visual_lora_enabled = True
        if visual_wasp_cfg is not None:
            visual_wasp_options = dict(visual_wasp_cfg)
            if str(visual_wasp_options.pop("type", "wasp")).lower() != "wasp":
                raise ValueError("visual_wasp_cfg only supports type='wasp'")
            visual_core = _unwrap_lm(self.qwen3_vl.model.visual)
            visual_blocks = getattr(visual_core, "blocks", None)
            if visual_blocks is None:
                raise AttributeError("Qwen3-VL visual module has no encoder blocks for WASP.")
            from .wasp import apply_wasp
            replaced = apply_wasp(visual_blocks, visual_wasp_options)
            expected_layers = visual_wasp_options.get("expected_num_layers")
            if expected_layers is not None and replaced != int(expected_layers):
                raise RuntimeError(f"Visual WASP target count mismatch: expected {expected_layers}, replaced {replaced}.")
            self._visual_wasp_param_ids = {id(param) for param in visual_blocks.parameters() if param.requires_grad}
            self._lora_enabled = True
            self._visual_wasp_enabled = True
        if visual_projector_wasp_cfg is not None:
            projector_wasp_options = dict(visual_projector_wasp_cfg)
            if str(projector_wasp_options.pop("type", "wasp")).lower() != "wasp":
                raise ValueError("visual_projector_wasp_cfg only supports type='wasp'")
            visual_core = _unwrap_lm(self.qwen3_vl.model.visual)
            projector_modules = []
            merger = getattr(visual_core, "merger", None)
            if merger is not None:
                projector_modules.append(merger)
            deepstack = getattr(visual_core, "deepstack_merger_list", None)
            if deepstack is not None:
                projector_modules.append(deepstack)
            from .wasp import apply_wasp
            replaced = sum(apply_wasp(module, projector_wasp_options) for module in projector_modules)
            expected_layers = projector_wasp_options.get("expected_num_layers")
            if expected_layers is not None and replaced != int(expected_layers):
                raise RuntimeError(f"Visual projector WASP target count mismatch: expected {expected_layers}, replaced {replaced}.")
            self._visual_projector_wasp_enabled = True
            self._lora_enabled = True
        self._use_mixed_visual_peft = visual_lora_cfg is not None or visual_wasp_cfg is not None or visual_projector_wasp_cfg is not None

        def apply_trainability_policy():
            if not train_vlm:
                self.qwen3_vl.requires_grad_(False)
                return
            if not self._use_mixed_visual_peft and peft_cfg is None:
                return
            self.qwen3_vl.requires_grad_(False)
            for param in self.qwen3_vl.model.language_model.parameters():
                if id(param) in self._language_adapter_param_ids:
                    param.requires_grad_(True)
            visual_adapter_ids = self._visual_lora_param_ids | self._visual_wasp_param_ids | self._visual_projector_wasp_param_ids
            for param in self.qwen3_vl.model.visual.parameters():
                if id(param) in visual_adapter_ids:
                    param.requires_grad_(True)

        self._apply_trainability_policy = apply_trainability_policy
        apply_trainability_policy()

        _vlm_config = self.qwen3_vl.config
        assert _vlm_config.text_config.num_hidden_layers == self.qwen3_perception_expert.config.num_hidden_layers == self.qwen3_action_expert.config.num_hidden_layers

        processor_kwargs = {} if processor_use_fast is None else {"use_fast": processor_use_fast}
        self.processor = AutoProcessor.from_pretrained(pretrained_path, **processor_kwargs)
        self.cast_stream_dtype = False
        tokenizer = self.processor.tokenizer

        existing_vocab = tokenizer.get_vocab()
        self.view_token_base_vocab_size = len(tokenizer)
        # Added before anything reads the embedding table; adding tokens later would index past it.
        tokens_to_add = [t for t in dict.fromkeys(view_tokens or NUSCENES_VIEW_TOKENS) if t not in existing_vocab]
        if len(tokens_to_add) > 0:
            tokenizer.add_tokens(tokens_to_add)
            self.qwen3_vl.resize_token_embeddings(len(tokenizer))
        self._apply_trainability_policy()

        if hasattr(self.qwen3_action_expert, "_init_weights"):
            self.qwen3_action_expert.apply(self.qwen3_action_expert._init_weights)

        self.to_bfloat16_for_selected_params(precision)

        self._vla_attn_impl = "flex"



    def visual_wasp_regularization(self):
        if not self._visual_wasp_enabled:
            return None
        from .wasp import compute_wasp_regularization
        visual_core = _unwrap_lm(self.qwen3_vl.model.visual)
        blocks = getattr(visual_core, "blocks", None)
        return None if blocks is None else compute_wasp_regularization(blocks)

    @property
    def vlm_base(self):
        return self.qwen3_vl

    def merge_lora(self) -> None:
        if not self._lora_enabled:
            return
        if self._language_lora_enabled:
            self.qwen3_vl.model.language_model = self.qwen3_vl.model.language_model.merge_and_unload()
        elif self._language_wasp_enabled:
            from .wasp import merge_wasp
            merge_wasp(self.qwen3_vl.model.language_model)
        if self._visual_lora_enabled:
            self.qwen3_vl.model.visual = self.qwen3_vl.model.visual.merge_and_unload()
        if self._visual_wasp_enabled or self._visual_projector_wasp_enabled:
            from .wasp import merge_wasp
            merge_wasp(self.qwen3_vl.model.visual)
        self._lora_enabled = False

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def embed_image(self, image_paths: list[list[str]], chunk_size: int = 6, processor=None):
        processor = processor or self.processor
        flat_image_paths = [path for sample in image_paths for path in sample]

        pil_images = []
        for path in flat_image_paths:
            img = Image.open(path)
            img = to_rgb(img)
            w, h = img.size
            img = img
            pil_images.append(img)

        all_embs = []
        all_grids = []

        for i in range(0, len(pil_images), chunk_size):
            chunk = pil_images[i : i + chunk_size]

            inputs = processor.image_processor(
                images=chunk, return_tensors="pt", do_resize=False
            )
            pix = inputs["pixel_values"].to(self.qwen3_vl.device, self.qwen3_vl.dtype)
            grid = inputs["image_grid_thw"].to(self.qwen3_vl.device)

            embs_list, _ = self.qwen3_vl.visual(
                hidden_states=pix, grid_thw=grid
            )

            all_embs.append(embs_list)
            all_grids.append(grid)

            del pix, grid

        image_features = torch.cat(all_embs, dim=0)
        all_grids = torch.cat(all_grids, dim=0)

        merge_size = self.qwen3_vl.visual.spatial_merge_size
        feature_lens = (all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        return image_features, feature_lens.tolist(), all_grids

    def pixel_patches(self, images_tensor: torch.Tensor, processor=None):
        """(B, N, 3, H, W) BGR 0-255 images -> what the image processor returns with do_resize=False (rescale,
        normalize, still frame repeated over the temporal patch, patches in merge order), on the images' device."""
        proc = (processor or self.processor).image_processor
        p, t, m = proc.patch_size, proc.temporal_patch_size, proc.merge_size
        x = images_tensor.flatten(0, 1)[:, [2, 1, 0]].clamp(0, 255).to(torch.uint8).float()
        # rescale folded into mean / std, in the fast processor's order: bit-identical pixel values
        mean = (torch.tensor(proc.image_mean) * (1.0 / proc.rescale_factor)).to(x.device).view(1, -1, 1, 1)
        std = (torch.tensor(proc.image_std) * (1.0 / proc.rescale_factor)).to(x.device).view(1, -1, 1, 1)
        x = (x - mean) / std
        n, c, h, w = x.shape
        if h % (p * m) or w % (p * m):
            raise ValueError(f"image size {h}x{w} is not a multiple of patch_size * merge_size = {p * m}")
        gh, gw = h // p, w // p
        x = x.unsqueeze(1).expand(n, t, c, h, w).reshape(n, t, c, gh // m, m, p, gw // m, m, p)
        pixel_values = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(n * gh * gw, c * t * p * p)
        grid = torch.tensor([[1, gh, gw]], device=x.device, dtype=torch.long).repeat(n, 1)
        return pixel_values, grid

    def embed_image_tensor(self, images_tensor: torch.Tensor, chunk_size: int = 6, processor=None):
        pixel_values, grids = self.pixel_patches(images_tensor, processor=processor)
        rows = int(grids[0, 1] * grids[0, 2])

        visual = self.qwen3_vl.visual
        # The official patched Qwen3-VL (qwenvl3/transformers_replace) returns the pre-merger hidden states after
        # the deepstack blocks and after the last block as a third output; stock transformers returns two, and forward
        # hooks on those blocks take the same tensors.
        taps = list(visual.deepstack_visual_indexes) + [len(visual.blocks) - 1]
        all_embs, all_grids, all_deepstack, all_raw = [], [], [], []

        for i in range(0, grids.shape[0], chunk_size):
            pix = pixel_values[i * rows:(i + chunk_size) * rows].to(self.qwen3_vl.dtype)
            grid = grids[i:i + chunk_size]

            raw = []
            hooks = [visual.blocks[k].register_forward_hook(lambda _module, _args, out: raw.append(out)) for k in taps]
            try:
                visual_out = visual(hidden_states=pix, grid_thw=grid)
            finally:
                for hook in hooks:
                    hook.remove()
            if len(visual_out) > 2:
                raw = list(visual_out[2])

            all_embs.append(visual_out[0])
            all_grids.append(grid)
            all_deepstack.append(visual_out[1] or [])
            all_raw.append(raw)

        image_features = torch.cat(all_embs, dim=0)
        all_grids = torch.cat(all_grids, dim=0)

        merge_size = self.qwen3_vl.visual.spatial_merge_size
        feature_lens = (all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        deepstack_features = [torch.cat(level, dim=0) for level in zip(*all_deepstack)]
        raw_features = [torch.cat(level, dim=0) for level in zip(*all_raw)]

        return image_features, feature_lens.tolist(), all_grids, deepstack_features, raw_features

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        return_middle_layers: Optional[list[int]] = None,
        q_len_rounded: int | None = None,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
        cast_stream_dtype: bool | None = None,
        attn_implementation: str | None = None,
    ):
        cast_stream_dtype = self.cast_stream_dtype if cast_stream_dtype is None else cast_stream_dtype
        middle_layer_outputs: dict[int, torch.Tensor] = {}
        if return_middle_layers is not None:
            return_middle_layers = sorted(set(return_middle_layers))

        if len(inputs_embeds) == 3 and inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_embed = inputs_embeds[0]
            if cast_stream_dtype:
                prefix_embed = prefix_embed.to(self.qwen3_vl.language_model.dtype)
            prefix_output = self.qwen3_vl.language_model.forward(
                inputs_embeds=prefix_embed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                deepstack_visual_embeds=deepstack_visual_embeds,
                visual_pos_masks=visual_pos_masks,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            middle_output = None
            suffix_output = None

        elif len(inputs_embeds) == 3 and inputs_embeds[0] is None and inputs_embeds[2] is None:
            middle_embed = inputs_embeds[1]
            if cast_stream_dtype:
                middle_embed = middle_embed.to(self.qwen3_perception_expert.dtype)
            middle_output = self.qwen3_perception_expert.forward(
                inputs_embeds=middle_embed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = middle_output.past_key_values
            middle_output = middle_output.last_hidden_state
            prefix_output = None
            suffix_output = None

        elif len(inputs_embeds) == 3 and inputs_embeds[0] is None and inputs_embeds[1] is None:
            suffix_embed = inputs_embeds[2]
            if cast_stream_dtype:
                suffix_embed = suffix_embed.to(self.qwen3_action_expert.dtype)
            suffix_output = self.qwen3_action_expert.forward(
                inputs_embeds=suffix_embed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            middle_output = None
            prefix_past_key_values = None

        else:
            models = [_unwrap_lm(self.qwen3_vl.language_model), self.qwen3_perception_expert, self.qwen3_action_expert]
            num_layers = self.qwen3_vl.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                hasattr(self.qwen3_action_expert, "gradient_checkpointing")
                and self.qwen3_action_expert.gradient_checkpointing
                and self.training
            )

            attn_implementation = attn_implementation or getattr(
                self, "_vla_attn_impl", getattr(self.qwen3_vl.config, "_attn_implementation", "eager"))
            layer_kv = cache_kv(past_key_values) if past_key_values is not None else None

            for layer_idx in range(num_layers):
                prefix_kv = layer_kv[layer_idx] if layer_kv is not None else None
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        self.qwen3_vl,
                        self.qwen3_perception_expert,
                        self.qwen3_action_expert,
                        attn_implementation,
                        q_len_rounded,
                        deepstack_visual_embeds,
                        visual_pos_masks,
                        prefix_kv,
                        use_reentrant=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        self.qwen3_vl,
                        self.qwen3_perception_expert,
                        self.qwen3_action_expert,
                        attn_implementation=attn_implementation,
                        q_len_rounded=q_len_rounded,
                        deepstack_visual_embeds=deepstack_visual_embeds,
                        visual_pos_masks=visual_pos_masks,
                        prefix_kv=prefix_kv,
                    )

                if return_middle_layers is not None and layer_idx in return_middle_layers:
                    middle_layer_outputs[layer_idx] = inputs_embeds[1]

            def compute_final_norms(inputs_embeds):
                return [None if h is None else models[i].norm(h) for i, h in enumerate(inputs_embeds)]

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, use_reentrant=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            prefix_output = outputs_embeds[0]
            middle_output = outputs_embeds[1]
            suffix_output = outputs_embeds[2]
            prefix_past_key_values = None

            if return_middle_layers is not None and len(middle_layer_outputs) > 0:
                for k, v in list(middle_layer_outputs.items()):
                    middle_layer_outputs[k] = models[1].norm(v)

        return [prefix_output, middle_output, suffix_output], prefix_past_key_values, middle_layer_outputs
