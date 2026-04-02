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

from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
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


class Qwen3VLSingleDecoderModel(nn.Module):
    """
    Single Decoder baseline: Use only Qwen3-VL language model as unified decoder.

    Unlike the three-expert architecture (VLM + Perception Expert + Action Expert),
    this model uses a single Transformer decoder (Qwen3-VL LLM) to process:
    - LLM tokens (text prompts + image features)
    - Perception tokens (det/map/occ/ego queries)
    - Action tokens (trajectory diffusion queries)

    All tokens are concatenated and fed into the same decoder.
    """

    def __init__(
        self,
        vlm_config,
        pretrained_path,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        train_vlm: bool = False,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Build Qwen3-VL config
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
        # Detect model size from VLM hidden_size to set vision_config and tie_word_embeddings
        # 2B: hidden=2048, tie_word_embeddings=True,  vision depth=24, hidden=1024, inter=4096,  out=2048, deepstack=[5,11,17]
        # 8B: hidden=4096, tie_word_embeddings=False, vision depth=27, hidden=1152, inter=4304,  out=4096, deepstack=[8,16,24]
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

        # Create Qwen3-VL model
        self.qwen3_vl = Qwen3VLForConditionalGeneration(config=vlm_config_hf)

        # Load pretrained weights
        safetensor_files = sorted(
            glob.glob(os.path.join(pretrained_path, "*.safetensors"))
        )

        state_dict = {}
        for file in safetensor_files:
            sd = load_file(file, device="cpu")

            for k, v in sd.items():
                if "action_preprocessor.normalizer" in k:
                    print(f"[SingleDecoder] Filter load model weight {k}")
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

        # VLM freezing logic
        if not train_vlm:
            print(f"[SingleDecoder] Freezing VLM parameters (train_vlm={train_vlm})")
            for p in self.qwen3_vl.parameters():
                p.requires_grad = False
        else:
            print(f"[SingleDecoder] Training VLM parameters (train_vlm={train_vlm})")

        # ── LoRA fine-tuning of the VLM ───────────────────────────────────────
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
            print(f"[LoRA] Applied LoRA to LLM only: r={lora_cfg.get('r')}, "
                  f"alpha={lora_cfg.get('lora_alpha')}, "
                  f"target_modules={lora_cfg.get('target_modules')}")

        # Load processor and add custom tokens
        self.processor = AutoProcessor.from_pretrained(pretrained_path)
        tokenizer = self.processor.tokenizer

        existing_vocab = tokenizer.get_vocab()
        tokens_to_add = [t for t in NUSCENES_VIEW_TOKENS if t not in existing_vocab]
        if len(tokens_to_add) > 0:
            tokenizer.add_tokens(tokens_to_add)
            self.qwen3_vl.resize_token_embeddings(len(tokenizer))

        self.to_bfloat16_for_selected_params(precision)

    @property
    def vlm_base(self):
        """Return the underlying Qwen3VLForConditionalGeneration (always qwen3_vl directly)."""
        return self.qwen3_vl

    def merge_lora(self) -> None:
        """Merge LoRA adapter weights into the base LLM in-place."""
        if not self._lora_enabled:
            print("[LoRA] merge_lora called but LoRA is not enabled — skipping.")
            return

        print("[LoRA] Merging LoRA adapter into LLM weights...")
        merged_lm = self.qwen3_vl.model.language_model.merge_and_unload()
        self.qwen3_vl.model.language_model = merged_lm
        self._lora_enabled = False
        print("[LoRA] Merge complete.")

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def embed_image(self, image_paths: list[list[str]], chunk_size: int = 6):
        """
        Embed images. Returns features list and grids.
        """
        flat_image_paths = [path for sample in image_paths for path in sample]

        pil_images = []
        for path in flat_image_paths:
            img = Image.open(path)
            img = to_rgb(img)
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

    def embed_image_tensor(self, images_tensor: torch.Tensor, chunk_size: int = 6):
        """
        Embed images from tensor (B, N, 3, H, W).
        """
        imgs = images_tensor

        # Convert BGR to RGB
        if imgs.shape[2] == 3:
            imgs = imgs[:, :, [2, 1, 0], :, :]

        imgs = torch.clamp(imgs, 0, 255).byte()

        B, N, C, H, W = imgs.shape
        flat_imgs = imgs.view(-1, C, H, W)

        pil_images = []
        for i in range(flat_imgs.shape[0]):
            img_np = flat_imgs[i].permute(1, 2, 0).cpu().numpy()
            pil_images.append(Image.fromarray(img_np))

        all_embs = []
        all_grids = []
        all_deepstack_features = [[] for _ in range(len(self.qwen3_vl.config.vision_config.deepstack_visual_indexes))]

        for i in range(0, len(pil_images), chunk_size):
            chunk = pil_images[i : i + chunk_size]

            chunk_resized = []
            for img in chunk:
                chunk_resized.append(img)

            inputs = self.processor.image_processor(
                images=chunk_resized, return_tensors="pt", do_resize=False
            )

            pix = inputs["pixel_values"].to(self.qwen3_vl.device, self.qwen3_vl.dtype)
            grid = inputs["image_grid_thw"].to(self.qwen3_vl.device)

            embs_list, deepstack_embs, raw_feature_list = self.qwen3_vl.visual(
                hidden_states=pix, grid_thw=grid
            )

            all_embs.append(embs_list)
            all_grids.append(grid)

            # Collect deepstack features
            for ds_idx, ds_feat in enumerate(deepstack_embs):
                all_deepstack_features[ds_idx].append(ds_feat)

            # Collect raw features
            if i == 0:
                all_raw_features = [[] for _ in range(len(raw_feature_list))]

            for raw_idx, raw_feat in enumerate(raw_feature_list):
                all_raw_features[raw_idx].append(raw_feat)

            del pix, grid

        image_features = torch.cat(all_embs, dim=0)
        del all_embs
        all_grids = torch.cat(all_grids, dim=0)

        merge_size = self.qwen3_vl.visual.spatial_merge_size
        feature_lens = (all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        deepstack_features = [torch.cat(ds_list, dim=0) for ds_list in all_deepstack_features]
        del all_deepstack_features
        raw_features = [torch.cat(raw_list, dim=0) for raw_list in all_raw_features]
        del all_raw_features

        return image_features, feature_lens.tolist(), all_grids, deepstack_features, raw_features

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        return_middle_layers: Optional[list[int]] = None,
        q_len_rounded: int | None = None,
    ):
        """
        Single decoder forward pass.

        Args:
            inputs_embeds: Concatenated embeddings [prefix, perception, action]
            attention_mask: Attention mask for all tokens
            position_ids: Position IDs for RoPE

        Returns:
            output: Hidden states from language model
            past_key_values: KV cache
            middle_layer_outputs: Intermediate layer outputs (if requested)
        """
        middle_layer_outputs: dict[int, torch.Tensor] = {}

        # Single decoder path: process all tokens together
        output = self.qwen3_vl.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Extract middle layers if requested
        if return_middle_layers is not None:
            # Note: This is a simplified version. Full implementation would need
            # to hook into intermediate layers during forward pass
            pass

        return output.last_hidden_state, output.past_key_values, middle_layer_outputs
