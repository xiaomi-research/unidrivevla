"""CPU parity check for the main AR VQA tokenizer, collator, and policy routes."""

import argparse
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy
from unidrivevla.data.vqa import ARPlanningQADataset, ar_collate_fn, build_vqa_loader


class FakeTokenizer:
    pad_token_id = 0
    tokens = {"<|im_start|>": 1, "<|im_sep|>": 2, "<|im_end|>": 3,
              "<|vision_start|>": 4, "<|image_pad|>": 5, "<|vision_end|>": 6}
    token_pattern = re.compile(
        r"(<\|im_start\|>|<\|im_sep\|>|<\|im_end\|>|<\|vision_start\|>|<\|image_pad\|>|<\|vision_end\|>)"
    )

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def encode_text(self, text):
        ids = []
        for part in self.token_pattern.split(text):
            if part:
                ids.extend([self.tokens[part]] if part in self.tokens else [100 + ord(char) for char in part])
        return ids

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = []
        role_ids = {"system": 10, "user": 11, "assistant": 12}
        for message in messages:
            ids.extend((1, role_ids[message["role"]], 2))
            ids.extend(self.encode_text(message["content"]))
            ids.append(3)
        if add_generation_prompt:
            ids.extend((1, role_ids["assistant"], 2))
        return ids


class FakeProcessor:
    merge_size = 2

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def image_processor(self, images, return_tensors, do_resize, **kwargs):
        image = images[0]
        width, height = image.size
        rgb = torch.tensor(image.getpixel((0, 0)), dtype=torch.float32)
        pixel_values = rgb.repeat(width * height, 1)
        return {"pixel_values": pixel_values, "image_grid_thw": torch.tensor([[1, height, width]])}


class FakeHead(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.train_vlm = True
        self.qwen3_vl_with_expert = nn.Module()
        self.qwen3_vl_with_expert.qwen3_vl = nn.Linear(1, 1, bias=False)
        self.qwen3_vl_with_expert.processor = processor
        self.plan_weight = nn.Parameter(torch.tensor(2.0))
        self.planning_batch = None

    def forward_ar_batch(self, ar_batch):
        raw = self.qwen3_vl_with_expert.qwen3_vl.weight.sum()
        return {"loss_ar": raw * 0.25, "loss_vlm_raw": raw.detach()}

    def forward_train(self, **batch):
        self.planning_batch = batch
        ar = self.forward_ar_batch(batch["ar_batch"])
        return {"losses": {"loss_planning": self.plan_weight, **ar}}


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b), (a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b, (a, b)


def main_dataset_class(path=None):
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "Bench2Drive/projects/mmdet3d_plugin/datasets/ar_planning_qa_dataset.py"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"main AR dataset source not found: {path}")
    spec = importlib.util.spec_from_file_location("main_ar_planning_qa_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ARPlanningQADataset, module.ar_collate_fn


def check_dataset(main_dataset_path=None):
    main_dataset, main_collate = main_dataset_class(main_dataset_path)
    processor = FakeProcessor()
    tokenizer = processor.tokenizer
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        images = [root / "red.png", root / "green.png", root / "blue.png"]
        Image.new("RGB", (2, 2), (255, 0, 0)).save(images[0])
        Image.new("RGB", (2, 2), (0, 255, 0)).save(images[1])
        Image.new("RGB", (4, 2), (0, 0, 255)).save(images[2])
        rows = [
            {"conversations": [{"from": "human", "value": "Question?"}, {"from": "gpt", "value": "Answer."}]},
            {"image": str(images[0]), "conversations": [{"from": "human", "value": "<image> What color?"},
                                                            {"from": "gpt", "value": "Red."}]},
            {"image": [str(images[1]), str(images[2])], "conversations": [
                {"from": "human", "value": "Views: <image> then <image>. Compare."},
                {"from": "gpt", "value": "Both visible."},
            ]},
        ]
        path = root / "vqa.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        malformed = root / "malformed.jsonl"
        malformed.write_text("{not-json}\n", encoding="utf-8")
        try:
            ARPlanningQADataset([str(malformed)], processor, tokenizer)
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("malformed JSONL was accepted")
        for max_length in (512, 30):
            main = main_dataset([str(path)], processor, tokenizer, max_length=max_length)
            current = ARPlanningQADataset([str(path)], processor, tokenizer, max_length=max_length)
            main_items = [main[i] for i in range(len(main))]
            current_items = [current[i] for i in range(len(current))]
            for expected, actual in zip(main_items, current_items):
                equal(expected, actual)
            equal(main_collate(main_items, tokenizer), ar_collate_fn(current_items, tokenizer))
        text_only = ARPlanningQADataset([str(path)], processor, tokenizer)[0]
        prefix_len = len(tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "Question?"}], add_generation_prompt=True, tokenize=True))
        assert text_only["ar_traj_start_pos"] == prefix_len
        assert text_only["ar_labels"][:prefix_len].eq(-100).all()
        assert torch.equal(text_only["ar_labels"][prefix_len:], text_only["ar_input_ids"][prefix_len:])
        multi = ARPlanningQADataset([str(path)], processor, tokenizer)[2]
        assert multi["ar_image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
        assert multi["ar_pixel_values"][0][0].tolist() == [0.0, 255.0, 0.0]
        assert multi["ar_pixel_values"][1][0].tolist() == [0.0, 0.0, 255.0]
        loader = build_vqa_loader({"vqa": {"enabled": False}}, None, 0, 1)
        assert loader is None
        head = FakeHead(processor)
        policy = SimpleNamespace(head=head)
        loader = build_vqa_loader({"vqa": {"enabled": True, "mode": "combined", "jsonl_paths": [str(path)],
                                              "batch_size": 2, "num_workers": 0, "max_length": 512}},
                                  policy, 0, 1)
        batch = next(iter(loader))
        assert batch["ar_input_ids"].device.type == "cpu"
        assert batch["ar_labels"].eq(-100).any()
        for frozen_head in (FakeHeadWithoutTrainableVLM(processor), FakeHeadFrozenVLM(processor)):
            try:
                build_vqa_loader({"vqa": {"enabled": True, "jsonl_paths": [str(path)]}},
                                 SimpleNamespace(head=frozen_head), 0, 1)
            except ValueError as exc:
                assert "train_vlm" in str(exc) or "trainable" in str(exc)
            else:
                raise AssertionError("frozen VLM was accepted for VQA training")


class FakeHeadWithoutTrainableVLM(FakeHead):
    def __init__(self, processor):
        super().__init__(processor)
        self.train_vlm = False


class FakeHeadFrozenVLM(FakeHead):
    def __init__(self, processor):
        super().__init__(processor)
        self.qwen3_vl_with_expert.qwen3_vl.requires_grad_(False)


def check_policy_routes():
    head = FakeHead(FakeProcessor())
    policy = object.__new__(UniDriveVLAPolicy)
    nn.Module.__init__(policy)
    policy.head = head
    policy.train()
    vqa_out = policy({"task": "vqa", "ar_batch": {"source": "vqa"}})
    vqa_out.loss.backward()
    vlm_param = head.qwen3_vl_with_expert.qwen3_vl.weight
    assert torch.allclose(vlm_param.grad, torch.full_like(vlm_param, 0.25))
    assert not vqa_out.loss_vlm_raw.requires_grad and not vqa_out.loss_ar.requires_grad

    for frozen_head in (FakeHeadWithoutTrainableVLM(FakeProcessor()), FakeHeadFrozenVLM(FakeProcessor())):
        frozen_policy = object.__new__(UniDriveVLAPolicy)
        nn.Module.__init__(frozen_policy)
        frozen_policy.head = frozen_head
        frozen_policy.train()
        try:
            frozen_policy({"task": "vqa", "ar_batch": {"source": "vqa"}})
        except ValueError as exc:
            assert "train_vlm" in str(exc) or "trainable" in str(exc)
        else:
            raise AssertionError("frozen VLM was accepted for VQA training")

    head.zero_grad(set_to_none=True)
    ar_batch = {"source": "combined"}
    features = {"task": "planning", "ar_batch": ar_batch}
    targets = {
        "gt_bboxes_3d": torch.zeros(1, 2, 9), "gt_labels_3d": torch.zeros(1, 2, dtype=torch.long),
        "box_mask": torch.tensor([[True, False]]),
        "gt_agent_fut_trajs": torch.zeros(1, 2, 3, 2), "gt_agent_fut_masks": torch.zeros(1, 2, 3),
        "gt_map_pts": torch.zeros(1, 2, 2, 3, 2), "gt_map_labels": torch.zeros(1, 2, dtype=torch.long),
        "map_mask": torch.tensor([[True, False]]),
    }
    combined_out = policy(features, targets)
    assert head.planning_batch["ar_batch"] is ar_batch
    assert isinstance(head.planning_batch["gt_bboxes_3d"], list)
    vlm_weight = head.qwen3_vl_with_expert.qwen3_vl.weight.sum()
    assert torch.allclose(combined_out.loss, head.plan_weight + vlm_weight * 0.25 + vlm_weight.detach())
    combined_out.loss.backward()
    assert torch.allclose(vlm_param.grad, torch.full_like(vlm_param, 0.25))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-ar-dataset", default=os.environ.get("UNIDRIVE_MAIN_AR_DATASET"),
                        help="path to main's ar_planning_qa_dataset.py")
    args = parser.parse_args()
    check_dataset(args.main_ar_dataset)
    check_policy_routes()
    print("VQA main-path parity and policy routes passed on CPU")
