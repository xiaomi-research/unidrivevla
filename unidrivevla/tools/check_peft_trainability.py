"""CPU check: the real Qwen wrapper registers and trains visual WASP parameters correctly."""

import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.adapters.unidrivevla import UniDriveVLAPolicy  # noqa: E402
from unidrivevla.models.unidrivevla.dense_heads import unidrivevla_vlm_qwenvl3 as qwen_wrapper  # noqa: E402
from unidrivevla.models.unidrivevla.dense_heads.wasp import WASPLinear  # noqa: E402


class TinyConfig:
    def __init__(self):
        self.text_config = SimpleNamespace()
        self.vision_config = SimpleNamespace()
        self.tie_word_embeddings = True


class TinyDims:
    hidden_size = 4
    intermediate_size = 8
    num_attention_heads = 2
    head_dim = 2
    num_hidden_layers = 2
    num_key_value_heads = 2


class TinyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyTarget()])
        self.merger = TinyTarget()
        self.deepstack_merger_list = nn.ModuleList([TinyTarget(), TinyTarget()])


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_proj = nn.Linear(4, 4)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyLanguage()
        self.visual = TinyVisual()


class TinyConditional(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TinyBackbone()


class TinyTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(8, 4)


class TinyTokenizer:
    def __init__(self):
        self.vocab = {token: index for index, token in enumerate(qwen_wrapper.NUSCENES_VIEW_TOKENS)}

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return len(self.vocab)


class TinyProcessor:
    @staticmethod
    def from_pretrained(_path, **_kwargs):
        return SimpleNamespace(tokenizer=TinyTokenizer())


def make_model(path, train_vlm=True):
    from unidrivevla.models.unidrivevla.dense_heads.unidrivevla_vlm_qwenvl3 import Qwen3VLWithExpertModel

    visual_wasp = {"rank": 2, "top_k": 1, "alpha": 2.0, "target_modules": ["proj"],
                   "expected_num_layers": 1}
    projector_wasp = {"rank": 2, "top_k": 1, "alpha": 2.0, "target_modules": ["proj"],
                      "expected_num_layers": 3}
    config_mapping = {"qwen3_vl": TinyConfig, "qwen3_vl_text": TinyConfig}
    with patch.object(qwen_wrapper, "CONFIG_MAPPING", config_mapping), \
         patch.object(qwen_wrapper, "Qwen3VLForConditionalGeneration", TinyConditional), \
         patch.object(qwen_wrapper, "Qwen3VLTextModel", TinyTextModel), \
         patch.object(qwen_wrapper, "AutoProcessor", TinyProcessor):
        return Qwen3VLWithExpertModel(
            TinyDims(), TinyDims(), TinyDims(), str(path), precision="float32", train_vlm=train_vlm,
            visual_wasp_cfg=visual_wasp, visual_projector_wasp_cfg=projector_wasp,
            view_tokens=qwen_wrapper.NUSCENES_VIEW_TOKENS,
        )


def adapter_layers(module):
    return [child for child in module.modules() if isinstance(child, WASPLinear)]


def trainable_adapter_ids(layers):
    return {id(parameter) for layer in layers for parameter in (layer.lora_A, layer.lora_B)}


def assert_equal_state(expected, actual):
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        other = actual[name]
        if torch.is_tensor(value):
            assert torch.is_tensor(other) and torch.equal(value, other), name
        elif isinstance(value, dict):
            assert isinstance(other, dict), name
            assert_equal_state(value, other)
        else:
            assert value == other, name


def main():
    torch.manual_seed(41)
    with tempfile.TemporaryDirectory(prefix="unidrive-peft-") as temp:
        model = make_model(Path(temp))
        visual = model.qwen3_vl.model.visual
        encoder_layers = adapter_layers(visual.blocks)
        projector_layers = adapter_layers(visual.merger) + adapter_layers(visual.deepstack_merger_list)
        assert len(encoder_layers) == 1
        assert len(projector_layers) == 3, f"merger/deepstack WASP layers: {len(projector_layers)}"

        encoder_ids = trainable_adapter_ids(encoder_layers)
        projector_ids = trainable_adapter_ids(projector_layers)
        assert model._visual_wasp_param_ids == encoder_ids
        assert model._visual_projector_wasp_param_ids == projector_ids
        assert model._visual_projector_wasp_param_ids, "projector adapters must be registered before freeze policy"
        all_adapter_ids = encoder_ids | projector_ids
        qwen_trainable = {id(parameter) for parameter in model.qwen3_vl.parameters() if parameter.requires_grad}
        assert qwen_trainable == all_adapter_ids, "only visual adapter factors should train in the tiny mixed-PEFT fixture"
        assert all(not parameter.requires_grad for parameter in model.qwen3_vl.model.language_model.parameters())
        for module in (visual.blocks, visual.merger, visual.deepstack_merger_list):
            for name, parameter in module.named_parameters():
                is_adapter = name.endswith("lora_A") or name.endswith("lora_B")
                assert parameter.requires_grad == is_adapter, (name, parameter.requires_grad)

        policy_module = nn.Module()
        policy_module.head = nn.Module()
        policy_module.head.qwen3_vl_with_expert = model
        optimizer = UniDriveVLAPolicy.get_optimizers(SimpleNamespace(
            named_parameters=policy_module.named_parameters, lr=1e-3, weight_decay=0.0, vlm_lr_mult=0.5,
            warmup_steps=1, warmup_start=1 / 3, min_lr_ratio=1e-3,
        ))
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert all_adapter_ids <= optimizer_ids, "the native policy optimizer must include both visual adapter groups"
        assert not ({id(parameter) for parameter in model.qwen3_vl.parameters()
                     if not parameter.requires_grad} & optimizer_ids), "frozen VLM base weights entered the optimizer"

        with torch.no_grad():
            for layer in encoder_layers + projector_layers:
                layer.lora_B.fill_(0.1)
        image_features = torch.tensor([[0.2, -0.6, 0.9, 0.3], [-0.8, 0.4, 0.5, 1.0]])
        outputs = [layer(image_features) for layer in encoder_layers + projector_layers]
        sum(output.square().sum() for output in outputs).backward()
        adapters = [parameter for layer in encoder_layers + projector_layers
                    for parameter in (layer.lora_A, layer.lora_B)]
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapters)
        assert all(parameter.grad.abs().sum() > 0 for parameter in adapters), "adapter gradients must be nonzero"
        optimizer.step()

        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        torch.manual_seed(41)
        restored = make_model(Path(temp))
        restored.load_state_dict(state, strict=True)
        restored._apply_trainability_policy()
        assert_equal_state(state, restored.state_dict())
        restored_visual = restored.qwen3_vl.model.visual
        restored_projector = trainable_adapter_ids(
            adapter_layers(restored_visual.merger) + adapter_layers(restored_visual.deepstack_merger_list))
        assert restored._visual_projector_wasp_param_ids == restored_projector
        restored_policy = nn.Module()
        restored_policy.head = nn.Module()
        restored_policy.head.qwen3_vl_with_expert = restored
        restored_optimizer = UniDriveVLAPolicy.get_optimizers(SimpleNamespace(
            named_parameters=restored_policy.named_parameters, lr=1e-3, weight_decay=0.0, vlm_lr_mult=0.5,
            warmup_steps=1, warmup_start=1 / 3, min_lr_ratio=1e-3,
        ))
        restored_optimizer.load_state_dict(optimizer_state)
        assert_equal_state(optimizer_state["state"], restored_optimizer.state_dict()["state"])

        frozen = make_model(Path(temp), train_vlm=False)
        assert all(not parameter.requires_grad for parameter in frozen.qwen3_vl.parameters()), \
            "train_vlm=false must freeze encoder and all WASP factors"
        print("ok   real constructor registers visual encoder, merger, and deepstack WASP adapters")
        print("ok   trainability policy freezes VLM base and keeps only requested visual adapters trainable")
        print("ok   native optimizer includes adapters; adapter gradients are finite and nonzero")
        print("ok   checkpoint model/optimizer state restores with trainability IDs reconstructed")
        print("ok   train_vlm=false freezes the whole Qwen3-VL subtree")
        print("OK")


if __name__ == "__main__":
    main()
