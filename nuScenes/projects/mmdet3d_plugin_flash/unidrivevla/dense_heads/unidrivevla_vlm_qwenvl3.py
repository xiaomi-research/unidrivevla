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
from typing import Literal, Optional, Dict, Any
import glob
import os

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
import deepspeed

from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.auto import CONFIG_MAPPING
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
    if hasattr(language_model, "base_model"):
        return language_model.base_model.model
    return language_model


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
):
    base_lm = _unwrap_lm(qwen3_vl.language_model)
    models = [base_lm, qwen3_perception_expert, qwen3_action_expert]

    query_states = []
    key_states = []
    value_states = []
    segment_lengths = [hidden_states.shape[1] for hidden_states in inputs_embeds]
    active_segments = []

    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states.shape[1] == 0:
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
        active_segments.append(i)

    if not query_states:
        return [hidden_states for hidden_states in inputs_embeds]

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
    active_output_by_segment = {}
    start_pos = 0

    for i in active_segments:
        hidden_states = inputs_embeds[i]
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + segment_lengths[i]

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

        active_output_by_segment[i] = out_emb
        start_pos = end_pos

    for i, hidden_states in enumerate(inputs_embeds):
        if segment_lengths[i] == 0:
            outputs_embeds.append(hidden_states)
        else:
            outputs_embeds.append(active_output_by_segment[i])

    if (deepstack_visual_embeds is not None
            and visual_pos_masks is not None
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
        image_token_compression_cfg: Optional[Dict[str, Any]] = None,
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

        self._lora_enabled = False
        if lora_cfg is not None:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                raise ImportError(
                    "peft is required for LoRA training. "
                    "Install it with: pip install peft"
                )
            lora_config = LoraConfig(**lora_cfg)
            self.qwen3_vl.model.language_model = get_peft_model(
                self.qwen3_vl.model.language_model, lora_config
            )
            self.qwen3_vl.model.language_model.print_trainable_parameters()
            self._lora_enabled = True

        if not train_vlm:
            for p in self.qwen3_vl.parameters():
                p.requires_grad = False

        _vlm_config = self.qwen3_vl.config
        assert _vlm_config.text_config.num_hidden_layers == self.qwen3_perception_expert.config.num_hidden_layers == self.qwen3_action_expert.config.num_hidden_layers

        self.processor = AutoProcessor.from_pretrained(pretrained_path)
        tokenizer = self.processor.tokenizer

        existing_vocab = tokenizer.get_vocab()
        tokens_to_add = [t for t in NUSCENES_VIEW_TOKENS if t not in existing_vocab]
        if len(tokens_to_add) > 0:
            tokenizer.add_tokens(tokens_to_add)
            self.qwen3_vl.resize_token_embeddings(len(tokenizer))

        if hasattr(self.qwen3_action_expert, "_init_weights"):
            self.qwen3_action_expert.apply(self.qwen3_action_expert._init_weights)

        self.to_bfloat16_for_selected_params(precision)

        self._vla_attn_impl = "flex"
        self.image_token_compression_cfg = copy.deepcopy(image_token_compression_cfg) or {}



    @property
    def vlm_base(self):
        return self.qwen3_vl

    def merge_lora(self) -> None:
        if not self._lora_enabled:
            return

        merged_lm = self.qwen3_vl.model.language_model.merge_and_unload()
        self.qwen3_vl.model.language_model = merged_lm
        self._lora_enabled = False

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def embed_image(self, image_paths: list[list[str]], chunk_size: int = 6):
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

            inputs = self.processor.image_processor(
                images=chunk, return_tensors="pt", do_resize=False
            )
            pix = inputs["pixel_values"].to(self.qwen3_vl.device, self.qwen3_vl.dtype)
            grid = inputs["image_grid_thw"].to(self.qwen3_vl.device)

            visual_outputs = self.qwen3_vl.visual(
                hidden_states=pix, grid_thw=grid
            )
            embs_list = visual_outputs[0]

            all_embs.append(embs_list)
            all_grids.append(grid)

            del pix, grid

        image_features = torch.cat(all_embs, dim=0)
        all_grids = torch.cat(all_grids, dim=0)

        merge_size = self.qwen3_vl.visual.spatial_merge_size
        feature_lens = (all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        return image_features, feature_lens.tolist(), all_grids

    def _image_token_compression_enabled(self) -> bool:
        cfg = self.image_token_compression_cfg
        return bool(cfg) and bool(cfg.get("enabled", True)) and cfg.get("method", "visionzip") != "none"

    def _get_visionzip_counts(self, tokens_per_image: int, image_index: Optional[int] = None):
        cfg = self.image_token_compression_cfg
        view_index = None if image_index is None else int(image_index) % int(cfg.get("num_views", 6))
        view_dominant_ratios = cfg.get("view_dominant_ratios", None)
        view_contextual_ratios = cfg.get("view_contextual_ratios", None)
        if view_index is not None and (view_dominant_ratios is not None or view_contextual_ratios is not None):
            if view_dominant_ratios is None or view_contextual_ratios is None:
                raise ValueError("VisionZip per-view compression requires both view_dominant_ratios and view_contextual_ratios")
            if len(view_dominant_ratios) <= view_index or len(view_contextual_ratios) <= view_index:
                raise ValueError(
                    f"VisionZip per-view ratios must cover view index {view_index}; "
                    f"got dominant={view_dominant_ratios}, contextual={view_contextual_ratios}"
                )
            dominant_count = int(round(tokens_per_image * float(view_dominant_ratios[view_index])))
            contextual_count = int(round(tokens_per_image * float(view_contextual_ratios[view_index])))
        elif "dominant_num" in cfg or "contextual_num" in cfg:
            dominant_count = int(cfg.get("dominant_num", 0))
            contextual_count = int(cfg.get("contextual_num", 0))
        else:
            dominant_ratio = float(cfg.get("dominant_ratio", 0.65))
            contextual_ratio = float(cfg.get("contextual_ratio", 0.05))
            dominant_count = int(round(tokens_per_image * dominant_ratio))
            contextual_count = int(round(tokens_per_image * contextual_ratio))

        if dominant_count < 0 or contextual_count < 0:
            raise ValueError(
                f"VisionZip token counts must be non-negative, got dominant={dominant_count}, "
                f"contextual={contextual_count}"
            )
        keep_count = dominant_count + contextual_count
        if keep_count <= 0 or keep_count > tokens_per_image:
            raise ValueError(
                f"VisionZip keeps {keep_count} tokens from {tokens_per_image}; check dominant/contextual settings"
            )
        dominant_count = min(dominant_count, keep_count)
        contextual_count = keep_count - dominant_count
        return dominant_count, contextual_count, keep_count

    @staticmethod
    def _visionzip_assign_contextual_tokens(
        source_metric: torch.Tensor,
        anchor_metric: torch.Tensor,
    ):
        if source_metric.numel() == 0 or anchor_metric.numel() == 0:
            empty = torch.empty((0,), device=source_metric.device, dtype=torch.long)
            return empty, torch.zeros((anchor_metric.shape[0],), device=anchor_metric.device, dtype=torch.float32)
        source_metric = F.normalize(source_metric.float(), dim=-1)
        anchor_metric = F.normalize(anchor_metric.float(), dim=-1)
        assign = torch.matmul(source_metric, anchor_metric.t()).argmax(dim=-1)
        counts = torch.zeros((anchor_metric.shape[0],), device=anchor_metric.device, dtype=torch.float32)
        counts.index_add_(0, assign, torch.ones((source_metric.shape[0],), device=source_metric.device, dtype=torch.float32))
        return assign, counts

    @staticmethod
    def _visionzip_merge_contextual_tokens(
        source_tokens: torch.Tensor,
        anchor_tokens: torch.Tensor,
        assign: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        if source_tokens.numel() == 0 or anchor_tokens.numel() == 0:
            return anchor_tokens
        aggregated = torch.zeros_like(anchor_tokens.float())
        aggregated.index_add_(0, assign, source_tokens.float())
        return anchor_tokens.float() + aggregated / counts[:, None].clamp_min(1.0)

    def _compress_image_tokens_visionzip(
        self,
        image_features: torch.Tensor,
        feature_lens,
        all_grids: torch.Tensor,
        deepstack_features=None,
        raw_features=None,
        visionzip_info=None,
    ):
        if raw_features is not None:
            raise ValueError("VisionZip image token compression must not be applied to perception raw_features")
        if not bool(self.image_token_compression_cfg.get("per_view", True)):
            raise ValueError("VisionZip image token compression currently requires per_view=True")
        if not bool(self.image_token_compression_cfg.get("preserve_position_ids", True)):
            raise ValueError("VisionZip image token compression currently requires preserve_position_ids=True")
        score_mode = self.image_token_compression_cfg.get("score", "visionzip_attn")
        if score_mode != "visionzip_attn":
            raise ValueError(
                "VisionZip compression in this path uses official visual attention scores; "
                f"unsupported score={score_mode!r}. Use score='visionzip_attn' or remove the field."
            )

        merge_size = int(self.qwen3_vl.visual.spatial_merge_size)
        num_images = int(all_grids.shape[0])
        raw_h = int(all_grids[0, 1].item())
        raw_w = int(all_grids[0, 2].item())
        if all_grids[:, 1:].ne(all_grids[0, 1:]).any():
            raise ValueError(
                "VisionZip batch compression currently requires fixed image grids across all "
                f"views/samples, got image_grid_thw={all_grids.detach().cpu().tolist()}"
            )
        if raw_h % merge_size != 0 or raw_w % merge_size != 0:
            raise ValueError(
                f"image grid ({raw_h}, {raw_w}) must be divisible by spatial_merge_size={merge_size}"
            )
        in_h = raw_h // merge_size
        in_w = raw_w // merge_size
        tokens_per_image = in_h * in_w

        if image_features.shape[0] != num_images * tokens_per_image:
            raise ValueError(
                f"VisionZip expects {num_images * tokens_per_image} image tokens, got {image_features.shape[0]}"
            )
        image_2d = image_features.view(num_images, tokens_per_image, -1)
        deepstack_2d = None
        if deepstack_features is not None:
            deepstack_2d = [feat.view(num_images, tokens_per_image, -1) for feat in deepstack_features]

        visionzip_score_2d = None
        visionzip_key_2d = None
        if visionzip_info is not None:
            if visionzip_info.get("attn_scores") is not None:
                visionzip_score_2d = visionzip_info["attn_scores"].view(num_images, tokens_per_image)
            if visionzip_info.get("key_metrics") is not None:
                visionzip_key_2d = visionzip_info["key_metrics"].view(num_images, tokens_per_image, -1)

        compressed_images = []
        compressed_deepstack = [[] for _ in range(len(deepstack_2d or []))]
        selected_position_ids = []
        compressed_feature_lens = []

        for img_idx in range(num_images):
            dominant_count, contextual_count, keep_count = self._get_visionzip_counts(tokens_per_image, img_idx)
            compressed_feature_lens.append(keep_count)
            img_tokens = image_2d[img_idx]
            if visionzip_score_2d is not None:
                score = visionzip_score_2d[img_idx].float()
            else:
                score = img_tokens.float().norm(dim=-1)

            dominant_idx = (
                torch.topk(score, k=dominant_count, largest=True).indices
                if dominant_count > 0
                else torch.empty((0,), device=img_tokens.device, dtype=torch.long)
            )
            selected_mask = torch.zeros((tokens_per_image,), device=img_tokens.device, dtype=torch.bool)
            selected_mask[dominant_idx] = True
            remain_idx = torch.nonzero(~selected_mask, as_tuple=False).flatten()

            contextual_idx = torch.empty((0,), device=img_tokens.device, dtype=torch.long)
            if contextual_count > 0:
                if remain_idx.numel() < contextual_count:
                    raise ValueError("VisionZip contextual_count exceeds available non-dominant tokens")
                step = max(int(remain_idx.numel() // contextual_count), 1)
                contextual_pos = torch.arange(
                    0, step * contextual_count, step, device=img_tokens.device, dtype=torch.long
                ).clamp_max(remain_idx.numel() - 1)
                contextual_idx = remain_idx.index_select(0, contextual_pos)
                selected_mask[contextual_idx] = True

            selected_idx = torch.sort(torch.cat([dominant_idx, contextual_idx], dim=0)).values
            selected_tokens = img_tokens.index_select(0, selected_idx).clone()
            source_idx = torch.empty((0,), device=img_tokens.device, dtype=torch.long)
            contextual_pos = torch.empty((0,), device=img_tokens.device, dtype=torch.long)
            contextual_assign = torch.empty((0,), device=img_tokens.device, dtype=torch.long)
            contextual_counts = torch.empty((0,), device=img_tokens.device, dtype=torch.float32)

            if contextual_count > 0 and remain_idx.numel() > contextual_count:
                source_idx = torch.nonzero(~selected_mask, as_tuple=False).flatten()
                contextual_pos = torch.searchsorted(selected_idx, contextual_idx)
                if visionzip_key_2d is not None:
                    metric = visionzip_key_2d[img_idx]
                else:
                    metric = img_tokens
                contextual_assign, contextual_counts = self._visionzip_assign_contextual_tokens(
                    metric.index_select(0, source_idx),
                    metric.index_select(0, contextual_idx),
                )
                contextual_tokens = self._visionzip_merge_contextual_tokens(
                    img_tokens.index_select(0, source_idx),
                    img_tokens.index_select(0, contextual_idx),
                    contextual_assign,
                    contextual_counts,
                )
                selected_tokens[contextual_pos] = contextual_tokens.to(dtype=selected_tokens.dtype)

            compressed_images.append(selected_tokens)

            if deepstack_2d is not None:
                for ds_idx, ds_img_tokens in enumerate(feat[img_idx] for feat in deepstack_2d):
                    selected_ds_tokens = ds_img_tokens.index_select(0, selected_idx).clone()
                    if contextual_count > 0 and remain_idx.numel() > contextual_count:
                        contextual_ds_tokens = self._visionzip_merge_contextual_tokens(
                            ds_img_tokens.index_select(0, source_idx),
                            ds_img_tokens.index_select(0, contextual_idx),
                            contextual_assign,
                            contextual_counts,
                        )
                        selected_ds_tokens[contextual_pos] = contextual_ds_tokens.to(dtype=selected_ds_tokens.dtype)
                    compressed_deepstack[ds_idx].append(selected_ds_tokens)

            rows = torch.div(selected_idx, in_w, rounding_mode="floor")
            cols = selected_idx % in_w
            t = torch.zeros_like(rows)
            selected_position_ids.append(
                {
                    "position_ids": torch.stack([t, rows, cols], dim=0),
                    "grid_hw": (in_h, in_w),
                }
            )

        image_features = torch.cat(compressed_images, dim=0)
        if deepstack_features is not None:
            deepstack_features = [torch.cat(feat_list, dim=0) for feat_list in compressed_deepstack]

        self._last_sparse_image_position_ids = selected_position_ids
        return image_features, compressed_feature_lens, all_grids, deepstack_features, raw_features

    def _compress_image_tokens(
        self,
        image_features: torch.Tensor,
        feature_lens,
        all_grids: torch.Tensor,
        deepstack_features=None,
        raw_features=None,
        visionzip_info=None,
    ):
        if not self._image_token_compression_enabled():
            self._last_sparse_image_position_ids = None
            return image_features, feature_lens, all_grids, deepstack_features, raw_features
        method = self.image_token_compression_cfg.get("method", "visionzip")
        if method != "visionzip":
            raise ValueError(f"Unsupported image token compression method: {method}")
        return self._compress_image_tokens_visionzip(
            image_features=image_features,
            feature_lens=feature_lens,
            all_grids=all_grids,
            deepstack_features=deepstack_features,
            raw_features=raw_features,
            visionzip_info=visionzip_info,
        )

    def _gpu_image_processor_no_resize(self, flat_imgs: torch.Tensor):
        cfg = self.image_token_compression_cfg or {}
        if not bool(cfg.get("use_gpu_image_processor", False)):
            return None
        if flat_imgs.dim() != 4 or flat_imgs.shape[1] != 3:
            return None

        image_processor = self.processor.image_processor
        patch_size = int(getattr(image_processor, "patch_size", 16))
        temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 2))
        merge_size = int(getattr(image_processor, "merge_size", 2))
        height = int(flat_imgs.shape[-2])
        width = int(flat_imgs.shape[-1])
        if height % (patch_size * merge_size) != 0 or width % (patch_size * merge_size) != 0:
            return None

        do_rescale = bool(getattr(image_processor, "do_rescale", True))
        rescale_factor = float(getattr(image_processor, "rescale_factor", 1.0 / 255.0))
        do_normalize = bool(getattr(image_processor, "do_normalize", True))
        image_mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        image_std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

        patches = flat_imgs.to(device=self.qwen3_vl.device, dtype=torch.float32)
        if do_rescale:
            patches = patches * rescale_factor
        if do_normalize:
            mean = patches.new_tensor(image_mean).view(1, 3, 1, 1)
            std = patches.new_tensor(image_std).view(1, 3, 1, 1)
            patches = (patches - mean) / std

        patches = patches.unsqueeze(1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(1, temporal_patch_size - patches.shape[1] % temporal_patch_size, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)

        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = patches.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        pixel_values = pixel_values.reshape(-1, channel * temporal_patch_size * patch_size * patch_size)
        pixel_values = pixel_values.to(dtype=self.qwen3_vl.dtype)
        image_grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * batch_size,
            dtype=torch.long,
            device=self.qwen3_vl.device,
        )
        return pixel_values, image_grid_thw

    def embed_image_tensor(self, images_tensor: torch.Tensor, chunk_size: int = 6):
        imgs = images_tensor

        if imgs.shape[2] == 3:
            imgs = imgs[:, :, [2, 1, 0], :, :]

        imgs = torch.clamp(imgs, 0, 255).byte()

        B, N, C, H, W = imgs.shape

        flat_imgs = imgs.reshape(-1, C, H, W)
        use_tensor_image_processor = bool(
            self.image_token_compression_cfg.get("use_tensor_image_processor", False)
        )
        use_gpu_image_processor = bool(
            self.image_token_compression_cfg.get("use_gpu_image_processor", False)
        )
        gpu_processor_inputs = self._gpu_image_processor_no_resize(flat_imgs) if use_gpu_image_processor else None
        if gpu_processor_inputs is not None:
            flat_processor_inputs = None
        elif use_tensor_image_processor:
            flat_processor_inputs = [img for img in flat_imgs.cpu()]
        else:
            flat_processor_inputs = []
            for i in range(flat_imgs.shape[0]):
                img_np = flat_imgs[i].permute(1, 2, 0).cpu().numpy()
                flat_processor_inputs.append(Image.fromarray(img_np))

        all_embs = []
        all_grids = []
        all_deepstack_features = [[] for _ in range(len(self.qwen3_vl.config.vision_config.deepstack_visual_indexes))]
        all_visionzip_attn_scores = []
        all_visionzip_key_metrics = []
        use_visionzip = (
            self._image_token_compression_enabled()
            and self.image_token_compression_cfg.get("method", "visionzip") == "visionzip"
        )

        num_images = int(flat_imgs.shape[0])
        for i in range(0, num_images, chunk_size):
            if gpu_processor_inputs is not None:
                pix_all, grid_all = gpu_processor_inputs
                start_token = i * (pix_all.shape[0] // num_images)
                end_token = min(i + chunk_size, num_images) * (pix_all.shape[0] // num_images)
                pix = pix_all[start_token:end_token]
                grid = grid_all[i : i + chunk_size]
            else:
                chunk = flat_processor_inputs[i : i + chunk_size]
                processor_kwargs = {
                    "images": chunk,
                    "return_tensors": "pt",
                    "do_resize": False,
                }
                if use_tensor_image_processor:
                    processor_kwargs["input_data_format"] = "channels_first"
                inputs = self.processor.image_processor(**processor_kwargs)

                pix = inputs["pixel_values"].to(self.qwen3_vl.device, self.qwen3_vl.dtype)
                grid = inputs["image_grid_thw"].to(self.qwen3_vl.device)

            visual_outputs = self.qwen3_vl.visual(
                hidden_states=pix, grid_thw=grid, return_visionzip=use_visionzip
            )
            if use_visionzip and len(visual_outputs) == 4:
                embs_list, deepstack_embs, raw_feature_list, visionzip_info = visual_outputs
                if visionzip_info is not None:
                    all_visionzip_attn_scores.append(visionzip_info["attn_scores"])
                    all_visionzip_key_metrics.append(visionzip_info["key_metrics"])
            else:
                embs_list, deepstack_embs, raw_feature_list = visual_outputs

            all_embs.append(embs_list)
            all_grids.append(grid)

            for ds_idx, ds_feat in enumerate(deepstack_embs):
                all_deepstack_features[ds_idx].append(ds_feat)

            if i == 0:
                all_raw_features = [[] for _ in range(len(raw_feature_list))]

            for raw_idx, raw_feat in enumerate(raw_feature_list):
                all_raw_features[raw_idx].append(raw_feat)

            del pix, grid

        image_features = torch.cat(all_embs, dim=0)
        all_grids = torch.cat(all_grids, dim=0)
        perception_grids = all_grids.clone()

        merge_size = self.qwen3_vl.visual.spatial_merge_size
        feature_lens = (all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        deepstack_features = [torch.cat(ds_list, dim=0) for ds_list in all_deepstack_features]

        raw_features = [torch.cat(raw_list, dim=0) for raw_list in all_raw_features]
        perception_deepstack_features = deepstack_features
        perception_raw_features = raw_features

        visionzip_info = None
        if use_visionzip and all_visionzip_attn_scores and all_visionzip_key_metrics:
            visionzip_info = {
                "attn_scores": torch.cat(all_visionzip_attn_scores, dim=0),
                "key_metrics": torch.cat(all_visionzip_key_metrics, dim=0),
            }

        image_features, feature_lens, all_grids, deepstack_features, _ = self._compress_image_tokens(
            image_features=image_features,
            feature_lens=feature_lens.tolist(),
            all_grids=all_grids,
            deepstack_features=deepstack_features,
            raw_features=None,
            visionzip_info=visionzip_info,
        )
        self._last_image_token_count = sum(int(length) for length in feature_lens)
        self._last_perception_image_grids = perception_grids
        self._last_perception_deepstack_features = perception_deepstack_features
        self._last_perception_raw_features = perception_raw_features

        return image_features, feature_lens, all_grids, deepstack_features, raw_features

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
        action_expert_override=None,
    ):
        middle_layer_outputs: dict[int, torch.Tensor] = {}
        if return_middle_layers is not None:
            return_middle_layers = sorted(set(return_middle_layers))
        action_expert = action_expert_override or self.qwen3_action_expert

        if len(inputs_embeds) == 3 and inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_output = self.qwen3_vl.language_model.forward(
                inputs_embeds=inputs_embeds[0],
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
            middle_output = self.qwen3_perception_expert.forward(
                inputs_embeds=inputs_embeds[1],
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
            suffix_output = action_expert.forward(
                inputs_embeds=inputs_embeds[2],
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
            models = [_unwrap_lm(self.qwen3_vl.language_model), self.qwen3_perception_expert, action_expert]
            num_layers = self.qwen3_vl.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                hasattr(action_expert, "gradient_checkpointing")
                and action_expert.gradient_checkpointing
                and self.training
            )

            attn_implementation = getattr(self, '_vla_attn_impl', getattr(self.qwen3_vl.config, "_attn_implementation", "eager"))

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        self.qwen3_vl,
                        self.qwen3_perception_expert,
                        action_expert,
                        attn_implementation,
                        q_len_rounded,
                        deepstack_visual_embeds,
                        visual_pos_masks,
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
                        action_expert,
                        attn_implementation=attn_implementation,
                        q_len_rounded=q_len_rounded,
                        deepstack_visual_embeds=deepstack_visual_embeds,
                        visual_pos_masks=visual_pos_masks,
                    )

                if return_middle_layers is not None and layer_idx in return_middle_layers:
                    middle_layer_outputs[layer_idx] = inputs_embeds[1]

            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

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
