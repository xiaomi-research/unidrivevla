"""python unidrivevla/tools/check_deformable.py: the deformable aggregation CUDA op (unidrivevla/models/ops/setup.py)
matches the grid_sample path in outputs and gradients on a det-sized DeformableFeatureAggregation, then times both in
bf16. Exits 1 on a mismatch or when the op is not built."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import unidrivevla.models.models.detection3d.detection3d_blocks  # noqa: E402,F401  (registers the keypoint generator)
from unidrivevla.adapters.unidrivevla import deformable, keypoints_box  # noqa: E402
from unidrivevla.models.compat import ATTENTION, build_from_cfg  # noqa: E402
from unidrivevla.models.models import blocks  # noqa: E402
from unidrivevla.models.ops import feature_maps_format  # noqa: E402

MIN_COSINE, MAX_REL_DIFF = 0.99999, 1e-4
if blocks.DAF is None:
    sys.exit("deformable_aggregation_ext is not built: cd unidrivevla/models/ops && python setup.py build_ext --inplace")
torch.manual_seed(0)
dev, bs, cams, anchors, hw = "cuda", 2, 6, 900, (17, 30)
offsets = torch.randn(cams, 2, device=dev)
# the samplers differ only for points within half a pixel outside the image; keep every point strictly inside
blocks.DeformableFeatureAggregation.project_points = staticmethod(
    lambda key_points, projection_mat, image_wh=None:
        torch.sigmoid(key_points[:, None, ..., :2] / 20 + offsets[:, None, None].to(key_points.dtype)))
module = build_from_cfg(deformable(cams, keypoints_box()), ATTENTION).to(dev).eval()
inputs = [torch.randn(bs, anchors, 256), torch.randn(bs, anchors, 11), torch.randn(bs, anchors, 256)] + \
    [torch.randn(bs, cams, 256, *hw) for _ in range(module.num_levels)]
inputs = [x.to(dev) for x in inputs]
projection = torch.randn(bs, cams, 4, 4, device=dev)
upstream = torch.randn(bs, anchors, 512, device=dev)


def run(use_func, dtype):
    module.use_deformable_func = use_func
    module.to(dtype).zero_grad(set_to_none=True)
    feat, anchor, embed, *maps = [x.detach().to(dtype, copy=True).requires_grad_() for x in inputs]
    out = module(feat, anchor, embed, feature_maps_format(maps) if use_func else maps, {"projection_mat": projection.to(dtype)})
    (out.float() * upstream).sum().backward()
    return [out, feat.grad, anchor.grad, *[m.grad for m in maps]] + [p.grad for p in module.parameters() if p.grad is not None]


ok = True
for ref, got in zip(run(False, torch.float32), run(True, torch.float32)):
    cosine = torch.nn.functional.cosine_similarity(ref.flatten().double(), got.flatten().double(), dim=0).item()
    rel = ((ref - got).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    ok &= cosine >= MIN_COSINE and rel <= MAX_REL_DIFF
    print(f"{str(tuple(ref.shape)):24s} cosine {cosine:.7f} max rel diff {rel:.2e}")
for use_func in (False, True):
    for i in range(13):
        if i == 3:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base, start = torch.cuda.memory_allocated(), time.perf_counter()
        run(use_func, torch.bfloat16)
    torch.cuda.synchronize()
    print(f"{'cuda op' if use_func else 'grid_sample':12s} bf16 fwd+bwd {(time.perf_counter() - start) / 10 * 1e3:6.2f} ms, "
          f"peak +{(torch.cuda.max_memory_allocated() - base) / 2 ** 20:.0f} MiB")
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
