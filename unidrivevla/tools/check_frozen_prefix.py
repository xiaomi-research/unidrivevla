"""torchrun --nproc_per_node=1 unidrivevla/tools/check_frozen_prefix.py data=nuscenes [hydra overrides]

One training batch, same noise, runtime state reset before each pass: the unmodified joint pass against
agent.cache_frozen_prefix, with the VLM
frozen (agent.train_vlm=false, forced: the cached prefix is stage 2's, a trained VLM runs the joint pass). They must
agree to bf16 kernel noise (the prefix attention runs another kernel); prints forward / backward time and peak memory
of the joint pass as built and of the frozen-prefix pass. Exits 1 on a disagreement.
"""

import sys
import time
import uuid
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from unidrivevla.data import DevicePrefetcher  # noqa: E402
from unidrivevla.distributed import parallelize  # noqa: E402
from unidrivevla.train import Trainer  # noqa: E402

MIN_GRAD_COSINE, MAX_LOSS_REL_DIFF = 0.998, 1e-3

if not torch.cuda.is_available():
    raise RuntimeError("check_frozen_prefix requires an available CUDA device")

overrides = sys.argv[1:]
if not any(arg.lstrip("+").startswith("experiment_name=") for arg in overrides):
    overrides = overrides + [f"experiment_name=check_frozen_prefix_{uuid.uuid4().hex[:12]}"]

with initialize_config_dir(config_dir=str(PACKAGE / "config"), version_base=None):
    cfg = compose("train", overrides=overrides + [
        "train.skip_validation=true", "evaluator.every_n_epochs=0",
        "evaluator.on_end=false", "agent.train_vlm=false"])
trainer = Trainer(cfg)
trainer.build_data()
trainer.agent.initialize()
model = parallelize(trainer.agent, trainer.args, trainer.ctx)
model.train()
head = trainer.agent.head
features, targets, tokens = next(iter(DevicePrefetcher(trainer.train_loader, trainer.ctx.device)))
params = [(n, p) for n, p in trainer.agent.named_parameters() if p.requires_grad]
initial_runtime = trainer.agent.training_runtime_state_dict()
cached_passes = 0
original_frozen_forward = head._forward_frozen_prefix


def observed_frozen_forward(*args, **kwargs):
    global cached_passes
    cached_passes += 1
    return original_frozen_forward(*args, **kwargs)


head._forward_frozen_prefix = observed_frozen_forward


def run(frozen_prefix: bool):
    head.cache_frozen_prefix = frozen_prefix
    # Both passes use the unmodified official-equivalent mask and start without temporal/DN history.
    trainer.agent.load_training_runtime_state_dict(initial_runtime, device=trainer.ctx.device)
    trainer.agent.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out = model(features, targets, tokens)
    torch.cuda.synchronize()
    mid = time.perf_counter()
    out.loss.backward()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return float(out.loss), (mid - start) * 1e3, (end - mid) * 1e3, torch.cuda.max_memory_allocated() / 2 ** 30


run(False)  # warm-up
joint = run(False)
assert cached_passes == 0, "joint reference unexpectedly used the frozen-prefix cache"
reference = {n: p.grad.detach().to("cpu", copy=True) for n, p in params if p.grad is not None}
run(True)  # warm-up
frozen = run(True)
assert cached_passes == 2, "frozen-prefix optimization was not exercised in both cached passes"
active = {n for n, p in params if p.grad is not None}
if active != set(reference):
    raise AssertionError(f"gradient paths changed: missing={sorted(set(reference) - active)}, "
                         f"extra={sorted(active - set(reference))}")
dot = na = nb = 0.0
for n, p in params:
    if p.grad is not None and n in reference:
        g, r = p.grad.detach().float(), reference[n].to(p.grad.device).float()
        dot, na, nb = dot + (g * r).sum().item(), na + (r * r).sum().item(), nb + (g * g).sum().item()
cosine = dot / max((na * nb) ** 0.5, 1e-30)
rel = abs(joint[0] - frozen[0]) / max(abs(joint[0]), 1e-6)
for label, r in (("joint pass", joint), ("frozen prefix", frozen)):
    print(f"{label:14s} forward {r[1]:7.0f} ms  backward {r[2]:7.0f} ms  peak {r[3]:.2f} GiB  loss {r[0]:.4f}")
ok = cosine >= MIN_GRAD_COSINE and rel <= MAX_LOSS_REL_DIFF
print(f"unmodified joint vs frozen prefix: loss {joint[0]:.4f} vs {frozen[0]:.4f} (rel {rel:.1e}), grad cosine {cosine:.6f}"
      f" -> {'OK' if ok else 'MISMATCH'}")
trainer.ctx.close()
sys.exit(0 if ok else 1)
