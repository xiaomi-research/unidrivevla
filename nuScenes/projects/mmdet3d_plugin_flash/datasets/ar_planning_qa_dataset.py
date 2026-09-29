"""
AR Planning QA Dataset for UniDriveVLA

Loads JSONL files containing planning QA data with multi-view images,
commands, and historical trajectories for AR (Autoregressive) cotraining.

Data format:
{
    "id": 0,
    "image": [cam_front, cam_fl, cam_fr, cam_bl, cam_br, cam_back],
    "conversations": [
        {"from": "human", "value": "<FRONT_VIEW>\n<image>\n..."},
        {"from": "gpt", "value": "[PT, (-0.27, 2.07), ...]"}
    ]
}

Reference: starVLA/dataloader/vlm_datasets.py
"""

import json
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


IGNORE_INDEX = -100


class ARPlanningQADataset(Dataset):
    """
    AR Planning QA Dataset for language modeling cotraining

    Args:
        jsonl_path: Path to JSONL file
        data_root: Root directory for nuScenes data (e.g., './data/nuscenes')
        processor: Qwen2.5-VL processor
        tokenizer: Qwen2.5-VL tokenizer
        max_length: Maximum sequence length (default: 2048)
        fix_image_size: Optional fixed image size (width, height)
    """

    def __init__(
        self,
        jsonl_path,
        processor,
        tokenizer,
        max_length: int = 2048,
        fix_image_size: Tuple[int, int] = None,
        max_pixels: int = None,
        data_root: str = None,  # unused, kept for backward compat
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fix_image_size = fix_image_size
        self.max_pixels = max_pixels

        # Support single str or list of paths
        if isinstance(jsonl_path, str):
            jsonl_path = [jsonl_path]

        # Load JSONL data from all paths
        self.data = []
        for p in jsonl_path:
            print(f"[ARPlanningQADataset] Loading JSONL from {p}")
            with open(p, 'r', encoding='utf-8') as f:
                self.data += [json.loads(line) for line in f]

        print(f"[ARPlanningQADataset] Loaded {len(self.data)} samples total")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # image field: str (single), List[str] (multi), or absent (text-only)
        raw_image = item.get('image', None)
        if raw_image is None:
            image_paths = []
        elif isinstance(raw_image, str):
            image_paths = [raw_image]
        else:
            image_paths = list(raw_image)

        pixel_values, image_grid_thw = self._load_images(image_paths)
        conversations = item['conversations']
        tokenized = self._tokenize_conversations(conversations, pixel_values, image_grid_thw)

        # If truncation dropped some images, keep only the images that survived.
        num_imgs = tokenized['num_images']
        if num_imgs < len(pixel_values):
            pixel_values   = pixel_values[:num_imgs]
            image_grid_thw = image_grid_thw[:num_imgs] if image_grid_thw is not None else None
            if image_grid_thw is not None and len(image_grid_thw) == 0:
                image_grid_thw = None

        return {
            'ar_input_ids': tokenized['input_ids'],
            'ar_labels': tokenized['labels'],
            'ar_pixel_values': pixel_values,      # List[Tensor], length = num_images
            'ar_image_grid_thw': image_grid_thw,  # Tensor [num_images, 3], or None
            'ar_traj_start_pos': tokenized['traj_start_pos'],
        }

    def _load_images(self, image_paths: List[str]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Load images from absolute paths. Supports any number of images (0, 1, 6, ...).

        Returns:
            pixel_values: List[Tensor], one per image
            image_grid_thw: Tensor [num_images, 3], or None if no images
        """
        if not image_paths:
            return [], None

        processor = self.processor
        pixel_values = []
        grid_thw_list = []

        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image {path}: {e}")

            # Process image with image_processor
            if self.fix_image_size is not None:
                img = img.resize(self.fix_image_size, resample=Image.BICUBIC)
                do_resize = False
            else:
                do_resize = True
                # smart_resize rejects aspect ratio >= 200; clamp before passing
                _w, _h = img.size
                if _w > 0 and _h > 0:
                    _ratio = _h / _w
                    _max_ratio = 199
                    if _ratio > _max_ratio:
                        img = img.resize((_w, int(_w * _max_ratio)), resample=Image.BICUBIC)
                    elif _ratio < 1.0 / _max_ratio:
                        img = img.resize((int(_h * _max_ratio), _h), resample=Image.BICUBIC)

            proc_kwargs = dict(images=[img], return_tensors="pt", do_resize=do_resize)
            if self.max_pixels is not None:
                proc_kwargs["max_pixels"] = self.max_pixels

            # CRITICAL: Must pass as list [img], not single PIL Image
            visual_processed = processor.image_processor(**proc_kwargs)
            # pixel_values shape: [total_patches, patch_dim] (all patches for this image)
            # DO NOT index [0] — that would take only the first patch (1D row)
            # image_processor merges all patches into a single 2D tensor
            pixel_values.append(visual_processed["pixel_values"])  # [num_patches, patch_dim]
            grid_thw_list.append(visual_processed["image_grid_thw"][0])  # [3]

        image_grid_thw = torch.stack(grid_thw_list)  # [num_images, 3]

        return pixel_values, image_grid_thw

    def _tokenize_conversations(
        self,
        conversations: List[Dict],
        pixel_values: List[torch.Tensor],
        image_grid_thw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize conversations with Qwen2.5-VL format.

        Strategy: tokenize each turn individually to get exact token boundaries,
        then concatenate and mask labels for non-assistant turns.
        """
        tokenizer = self.tokenizer

        # Role mapping
        roles = {"human": "user", "gpt": "assistant"}

        # System message
        system_message = "You are a helpful assistant."

        # Calculate vision token counts for each image
        # grid_thw[t,h,w] → total patches = t*h*w, divide by merge_size^2
        merge_size = getattr(self.processor, 'merge_size', 2)
        if image_grid_thw is not None:
            vision_token_counts = [
                int((thw[0] * thw[1] * thw[2]).item() // (merge_size ** 2))
                for thw in image_grid_thw
            ]
        else:
            vision_token_counts = []

        # Build messages list (with vision tokens substituted)
        messages = [{"role": "system", "content": system_message}]
        vision_idx = 0
        for conv in conversations:
            role = roles.get(conv.get("from", conv.get("role")), "user")
            content = conv.get("value", conv.get("content"))

            # Replace <image> placeholders with actual Qwen-VL vision tokens
            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    if vision_idx < len(vision_token_counts):
                        num_tokens = vision_token_counts[vision_idx]
                        vision_tokens = (
                            "<|vision_start|>"
                            + "<|image_pad|>" * num_tokens
                            + "<|vision_end|>"
                        )
                        new_parts.append(vision_tokens)
                        vision_idx += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)

            messages.append({"role": role, "content": content})

        # Tokenize the full conversation at once to get correct token sequence
        full_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )
        full_ids = torch.tensor(full_ids, dtype=torch.long)

        # Build labels by re-tokenizing each turn to find assistant token spans
        # Start with all IGNORE_INDEX, then unmask assistant turns
        labels = torch.full_like(full_ids, IGNORE_INDEX)

        # Re-tokenize prefix up to each turn boundary to locate assistant spans
        # We identify assistant turn content by finding where each assistant reply starts/ends
        prefix_messages = [{"role": "system", "content": system_message}]
        vision_idx = 0
        for conv in conversations:
            role = roles.get(conv.get("from", conv.get("role")), "user")
            content = conv.get("value", conv.get("content"))

            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    if vision_idx < len(vision_token_counts):
                        num_tokens = vision_token_counts[vision_idx]
                        vision_tokens = (
                            "<|vision_start|>"
                            + "<|image_pad|>" * num_tokens
                            + "<|vision_end|>"
                        )
                        new_parts.append(vision_tokens)
                        vision_idx += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)

            if role == "assistant":
                # Tokenize prefix (everything before this turn) to find start position
                prefix_ids = tokenizer.apply_chat_template(
                    prefix_messages,
                    add_generation_prompt=True,  # adds <|im_start|>assistant\n
                    tokenize=True,
                )
                start = len(prefix_ids)

                # Tokenize prefix + this assistant turn to find end position
                prefix_messages.append({"role": role, "content": content})
                full_prefix_ids = tokenizer.apply_chat_template(
                    prefix_messages,
                    add_generation_prompt=False,
                    tokenize=True,
                )
                end = len(full_prefix_ids)

                # Unmask assistant content tokens (exclude the trailing <|im_end|>\n)
                if start < end and end <= len(labels):
                    labels[start:end] = full_ids[start:end]
            else:
                prefix_messages.append({"role": role, "content": content})

        # Truncate if exceeds max_length
        num_complete_images = len(vision_token_counts)  # default: all images kept
        if full_ids.shape[0] > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]

            # Recount complete images after truncation so that image_grid_thw
            # stays consistent with the truncated input_ids.
            # A complete image has both <|vision_start|> and <|vision_end|>.
            # If the last image was partially truncated (vision_start present but
            # vision_end missing), also strip those dangling tokens from input_ids.
            if vision_token_counts:
                vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
                vision_end_id   = tokenizer.convert_tokens_to_ids('<|vision_end|>')

                num_complete_images = int((full_ids == vision_end_id).sum().item())

                starts = (full_ids == vision_start_id).nonzero(as_tuple=False).view(-1)
                if len(starts) > num_complete_images:
                    # Dangling vision_start — cut before it
                    cut = int(starts[num_complete_images].item())
                    full_ids = full_ids[:cut]
                    labels   = labels[:cut]

        # traj_start_pos: position of first unmasked label token (assistant reply start)
        # Used by forward_ar_batch to build suffix-blind attention mask for traj QA
        unmasked = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
        traj_start_pos = int(unmasked[0].item()) if len(unmasked) > 0 else len(full_ids)

        return {
            'input_ids': full_ids,
            'labels': labels,
            'traj_start_pos': traj_start_pos,
            'num_images': num_complete_images,
        }


def ar_collate_fn(batch: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for AR Planning QA Dataset

    Handles variable-length sequences and List[List[Tensor]] pixel_values

    Args:
        batch: List of samples from ARPlanningQADataset
        tokenizer: Qwen2.5-VL tokenizer

    Returns:
        dict with keys:
            - ar_input_ids: [B, max_L] padded
            - ar_labels: [B, max_L] padded with -100
            - ar_pixel_values: List[List[Tensor]], length B
            - ar_image_grid_thw: List[Tensor|None], length B  (variable num_images per sample)
            - ar_traj_start_pos: [B]
    """
    input_ids_padded = pad_sequence(
        [item['ar_input_ids'] for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels_padded = pad_sequence(
        [item['ar_labels'] for item in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )

    # pixel_values: List[List[Tensor]] — outer=batch, inner=images per sample
    pixel_values_list = [item['ar_pixel_values'] for item in batch]

    # image_grid_thw: keep as List (each sample may have different num_images)
    # driving sample: Tensor[6,3], VQA sample: Tensor[1,3] or None
    image_grid_thw_list = [item['ar_image_grid_thw'] for item in batch]

    traj_start_pos = torch.tensor(
        [item['ar_traj_start_pos'] for item in batch], dtype=torch.long
    )

    return {
        'ar_input_ids': input_ids_padded,
        'ar_labels': labels_padded,
        'ar_pixel_values': pixel_values_list,        # List[List[Tensor]], length B
        'ar_image_grid_thw': image_grid_thw_list,    # List[Tensor|None], length B
        'ar_traj_start_pos': traj_start_pos,
    }
