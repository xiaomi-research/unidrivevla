"""python unidrivevla/tools/check_pixels.py [VLM_PATH]: the GPU pixel_patches equals the image processor's pixel_values and
image_grid_thw (do_resize=False) bit for bit on random 544 x 960 BGR images. VLM_PATH defaults to VLM_PRETRAINED_PATH.
Exits 1 on a mismatch."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.models.unidrivevla.dense_heads.unidrivevla_vlm_qwenvl3 import Qwen3VLWithExpertModel  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "VLM_PRETRAINED_PATH", "/data/yongkangli/ckpts/Qwen3-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained(path)
bgr = torch.randint(0, 256, (2, 3, 3, 544, 960), dtype=torch.uint8)
device = "cuda" if torch.cuda.is_available() else "cpu"
pixels, grid = Qwen3VLWithExpertModel.pixel_patches(SimpleNamespace(processor=processor), bgr.to(device))
rgb = [Image.fromarray(np.ascontiguousarray(im.permute(1, 2, 0).numpy())) for im in bgr.flatten(0, 1)[:, [2, 1, 0]]]
ref = processor.image_processor(images=rgb, return_tensors="pt", do_resize=False)
diff = (pixels.cpu() - ref["pixel_values"].float()).abs().max().item()
same_grid = torch.equal(grid.cpu(), ref["image_grid_thw"])
bf16_equal = (pixels.cpu().bfloat16() == ref["pixel_values"].bfloat16()).float().mean().item()
print(type(processor.image_processor).__name__, tuple(pixels.shape), tuple(ref["pixel_values"].shape),
      f"max abs diff {diff:.2e}, grids equal {same_grid}, bf16 identical {100 * bf16_equal:.4f}%")
sys.exit(0 if same_grid and diff == 0 else 1)
