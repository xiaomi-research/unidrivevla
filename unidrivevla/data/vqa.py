"""AR VQA data path ported from the main UniDriveVLA dataset."""

import gzip
import json
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler

IGNORE_INDEX = -100


class ARPlanningQADataset(Dataset):
    """JSONL conversations tokenized with the Qwen processor used by the policy."""

    def __init__(self, jsonl_path, processor, tokenizer, max_length=2048, fix_image_size=None, max_pixels=None):
        self.processor, self.tokenizer = processor, tokenizer
        self.max_length, self.fix_image_size, self.max_pixels = int(max_length), fix_image_size, max_pixels
        if self.max_length < 1:
            raise ValueError("VQA max_length must be positive")
        paths = [jsonl_path] if isinstance(jsonl_path, str) else list(jsonl_path)
        self.data = []
        for path in paths:
            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as file:
                self.data.extend(json.loads(line) for line in file if line.strip())
        if not self.data:
            raise ValueError(f"VQA JSONL contains no samples: {paths}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_image = item.get("image")
        image_paths = [] if raw_image is None else [raw_image] if isinstance(raw_image, str) else list(raw_image)
        pixel_values, image_grid_thw = self._load_images(image_paths)
        tokenized = self._tokenize_conversations(item["conversations"], image_grid_thw)
        num_images = tokenized["num_images"]
        if num_images < len(pixel_values):
            pixel_values = pixel_values[:num_images]
            image_grid_thw = image_grid_thw[:num_images] if image_grid_thw is not None else None
            if image_grid_thw is not None and len(image_grid_thw) == 0:
                image_grid_thw = None
        return {
            "ar_input_ids": tokenized["input_ids"],
            "ar_labels": tokenized["labels"],
            "ar_pixel_values": pixel_values,
            "ar_image_grid_thw": image_grid_thw,
            "ar_traj_start_pos": tokenized["traj_start_pos"],
        }

    def _load_images(self, image_paths: List[str]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        if not image_paths:
            return [], None
        pixel_values, grids = [], []
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Failed to load VQA image {path}: {exc}") from exc
            if self.fix_image_size is not None:
                image = image.resize(tuple(self.fix_image_size), resample=Image.BICUBIC)
                do_resize = False
            else:
                do_resize = True
                width, height = image.size
                if width > 0 and height > 0:
                    ratio = height / width
                    if ratio > 199:
                        image = image.resize((width, int(width * 199)), resample=Image.BICUBIC)
                    elif ratio < 1.0 / 199:
                        image = image.resize((int(height * 199), height), resample=Image.BICUBIC)
            kwargs = {"images": [image], "return_tensors": "pt", "do_resize": do_resize}
            if self.max_pixels is not None:
                kwargs["max_pixels"] = self.max_pixels
            processed = self.processor.image_processor(**kwargs)
            pixel_values.append(processed["pixel_values"])
            grids.append(processed["image_grid_thw"][0])
        return pixel_values, torch.stack(grids)

    def _tokenize_conversations(self, conversations: List[Dict], image_grid_thw):
        tokenizer = self.tokenizer
        roles = {"human": "user", "gpt": "assistant"}
        system_message = "You are a helpful assistant."
        merge_size = getattr(self.processor, "merge_size", 2)
        vision_token_counts = [] if image_grid_thw is None else [
            int((grid[0] * grid[1] * grid[2]).item() // merge_size**2) for grid in image_grid_thw
        ]

        def messages_with_images(turns):
            messages, vision_idx = [{"role": "system", "content": system_message}], 0
            for turn in turns:
                role = roles.get(turn.get("from", turn.get("role")), "user")
                content = turn.get("value", turn.get("content"))
                if "<image>" in content:
                    parts, new_parts = content.split("<image>"), []
                    for part in parts[:-1]:
                        new_parts.append(part)
                        if vision_idx < len(vision_token_counts):
                            count = vision_token_counts[vision_idx]
                            new_parts.append("<|vision_start|>" + "<|image_pad|>" * count + "<|vision_end|>")
                            vision_idx += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                messages.append({"role": role, "content": content})
            return messages

        messages = messages_with_images(conversations)
        full_ids = torch.tensor(tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True),
                                dtype=torch.long)
        labels = torch.full_like(full_ids, IGNORE_INDEX)
        prefix_messages = [{"role": "system", "content": system_message}]
        image_index = 0
        for turn in conversations:
            role = roles.get(turn.get("from", turn.get("role")), "user")
            content = turn.get("value", turn.get("content"))
            if "<image>" in content:
                parts, new_parts = content.split("<image>"), []
                for part in parts[:-1]:
                    new_parts.append(part)
                    if image_index < len(vision_token_counts):
                        count = vision_token_counts[image_index]
                        new_parts.append("<|vision_start|>" + "<|image_pad|>" * count + "<|vision_end|>")
                        image_index += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)
            if role == "assistant":
                start = len(tokenizer.apply_chat_template(prefix_messages, add_generation_prompt=True, tokenize=True))
                prefix_messages.append({"role": role, "content": content})
                end = len(tokenizer.apply_chat_template(prefix_messages, add_generation_prompt=False, tokenize=True))
                if start < end and end <= len(labels):
                    labels[start:end] = full_ids[start:end]
            else:
                prefix_messages.append({"role": role, "content": content})

        num_complete_images = len(vision_token_counts)
        if len(full_ids) > self.max_length:
            full_ids, labels = full_ids[:self.max_length], labels[:self.max_length]
            if vision_token_counts:
                start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
                end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
                num_complete_images = int((full_ids == end_id).sum().item())
                starts = (full_ids == start_id).nonzero(as_tuple=False).view(-1)
                if len(starts) > num_complete_images:
                    cut = int(starts[num_complete_images].item())
                    full_ids, labels = full_ids[:cut], labels[:cut]
        unmasked = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
        return {
            "input_ids": full_ids,
            "labels": labels,
            "traj_start_pos": int(unmasked[0].item()) if len(unmasked) else len(full_ids),
            "num_images": num_complete_images,
        }


def ar_collate_fn(batch, tokenizer):
    return {
        "ar_input_ids": pad_sequence([item["ar_input_ids"] for item in batch], batch_first=True,
                                      padding_value=tokenizer.pad_token_id),
        "ar_labels": pad_sequence([item["ar_labels"] for item in batch], batch_first=True,
                                   padding_value=IGNORE_INDEX),
        "ar_pixel_values": [item["ar_pixel_values"] for item in batch],
        "ar_image_grid_thw": [item["ar_image_grid_thw"] for item in batch],
        "ar_traj_start_pos": torch.tensor([item["ar_traj_start_pos"] for item in batch], dtype=torch.long),
    }


def build_vqa_loader(cfg, policy, rank: int, world: int):
    """Build the optional CPU-side VQA DataLoader from the Hydra root config or its vqa node."""
    vqa = cfg.get("vqa", cfg) if hasattr(cfg, "get") else cfg
    if not bool(vqa.get("enabled", False)):
        return None
    mode = str(vqa.get("mode", "combined"))
    if mode not in ("combined", "alternate"):
        raise ValueError(f"vqa.mode must be 'combined' or 'alternate', got {mode!r}")
    paths = vqa.get("jsonl_paths") or []
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("vqa.enabled=true requires non-empty vqa.jsonl_paths")
    batch_size, workers = int(vqa.get("batch_size", 2)), int(vqa.get("num_workers", 2))
    if batch_size < 1 or workers < 0:
        raise ValueError("vqa.batch_size must be positive and vqa.num_workers non-negative")
    rank, world = int(rank), int(world)
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"invalid distributed rank/world: rank={rank}, world={world}")

    head = policy.head
    if not bool(getattr(head, "train_vlm", False)):
        raise ValueError("VQA training requires agent.head.train_vlm=true")
    vlm = getattr(getattr(head, "qwen3_vl_with_expert", None), "qwen3_vl", None)
    trainable_vlm = (any(param.requires_grad for param in vlm.parameters()) if vlm is not None else
                     any(param.requires_grad for name, param in head.named_parameters() if "qwen3_vl" in name))
    if not trainable_vlm:
        raise ValueError("VQA training requires at least one trainable Qwen VLM parameter")
    processor = head.qwen3_vl_with_expert.processor
    dataset = ARPlanningQADataset(
        paths, processor, processor.tokenizer, max_length=vqa.get("max_length", 2048),
        fix_image_size=vqa.get("fix_image_size"), max_pixels=vqa.get("max_pixels", 100000),
    )
    sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
               if world > 1 else None)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=workers, pin_memory=True, drop_last=True,
        collate_fn=lambda batch: ar_collate_fn(batch, processor.tokenizer),
    )
