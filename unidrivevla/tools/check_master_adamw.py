"""python unidrivevla/tools/check_master_adamw.py: MasterAdamW on bf16 parameters (next to fp32 ones) tracks fp32 AdamW
where bf16 AdamW in place drifts, and its fp32 state survives the DCP state-dict round trip of a resume bit for bit.
Runs on a GPU when there is one (the fused kernel), else on the CPU (foreach); exits 1 on a mismatch."""

import copy
import sys
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.train.optim import MasterAdamW  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
HP = dict(lr=2e-4, weight_decay=1e-4)
REF = dict(HP, **({"fused": True} if dev == "cuda" else {"foreach": True}))  # the kernel MasterAdamW runs on its masters
ref = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.LayerNorm(256), torch.nn.Linear(8, 8)).to(dev)  # [2]: no grad
with torch.no_grad():
    for p in ref.parameters():
        p.copy_(p.bfloat16())
mixed = copy.deepcopy(ref)
mixed[0].bfloat16()  # the LayerNorm stays fp32
mixed[2].bfloat16()
plain = copy.deepcopy(ref).bfloat16()
optimizers = torch.optim.AdamW(ref.parameters(), **REF), MasterAdamW(mixed.parameters(), **HP), \
    torch.optim.AdamW(plain.parameters(), **REF)


def step(models, opts, n):
    for _ in range(n):
        grads = [torch.randn_like(p).bfloat16() for p in ref[:2].parameters()]  # what every path gets from bf16 compute
        for model, opt in zip(models, opts):
            for p, g in zip(model[:2].parameters(), grads):
                p.grad = g.to(p.dtype)
            opt.step()
            opt.zero_grad(set_to_none=True)


step((ref, mixed, plain), optimizers, 300)
opt = optimizers[1]
params = list(mixed.parameters())
masters = [opt.state[p].get("master", p) for p in params]
master_err = max((m - r).abs().max().item() for m, r in zip(masters, ref.parameters()))
plain_err = max((p.float() - r).abs().max().item() for p, r in zip(plain.parameters(), ref.parameters()))
rounded = all(torch.equal(p, m.to(p.dtype)) for p, m in zip(params, masters))
complete = all("step" in opt.state[p] for p in params)  # DCP's resume expects state for every parameter
print(f"300 steps: master vs fp32 AdamW {master_err:.2e}, bf16 AdamW in place vs fp32 AdamW {plain_err:.2e}, "
      f"parameters are the rounded masters {rounded}, state for gradient-less parameters {complete}")
ok = master_err < 1e-6 < plain_err and rounded and complete

restored = copy.deepcopy(mixed)
restored_opt = MasterAdamW(restored.parameters(), **HP)
model_sd, optim_sd = get_state_dict(mixed, opt)  # live tensors; a resume reads fresh ones from disk
set_state_dict(restored, restored_opt, model_state_dict=copy.deepcopy(model_sd), optim_state_dict=copy.deepcopy(optim_sd))
kept = all(
    restored_opt.state[q][k].dtype == torch.float32 and torch.equal(restored_opt.state[q][k], opt.state[p][k])
    for p, q in zip(params, restored.parameters()) for k in ("master", "exp_avg", "exp_avg_sq") if k in opt.state[p])
torch.manual_seed(1)
step((mixed,), (opt,), 5)
torch.manual_seed(1)
step((restored,), (restored_opt,), 5)
resumed = all(torch.equal(p, q) for p, q in zip(mixed.parameters(), restored.parameters()))
print(f"state dict round trip keeps the fp32 state {kept}, resumed steps identical {resumed}")
ok &= kept and resumed
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
