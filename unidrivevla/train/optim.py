"""AdamW over fp32 master weights for parameters that compute in bf16 (DeepSpeed's bf16 optimizer without ZeRO).

The optimizer state holds an fp32 copy of every low-precision parameter and fp32 moments; a step updates the copy and
rounds it into the parameter. In place, bf16 AdamW rounds away every update below half a bf16 step, 2^-9 of the
weight: all of an lr 2e-4 step on norm weights near 1, and every weight decay term. Gradients stay in the parameter
dtype. fp32 parameters (and FSDP2's fp32 shards) are updated as by torch.optim.AdamW.
"""

from itertools import chain

import torch
from torch.optim.adamw import adamw

FP32_STATE = ("master", "exp_avg", "exp_avg_sq")
CHUNK_NUMEL = 1 << 27  # elements per fused update: 512 MiB of fp32 gradient copies, few kernel launches


class MasterAdamW(torch.optim.AdamW):
    fp32_master = True

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        if any(group["amsgrad"] for group in self.param_groups):
            raise ValueError("MasterAdamW does not implement amsgrad")
        # Optimizer.load_state_dict casts floating state to the parameter dtype; put the saved fp32 values back
        self._held = []
        self.register_load_state_dict_pre_hook(MasterAdamW._hold_fp32)
        self.register_load_state_dict_post_hook(MasterAdamW._restore_fp32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # state for parameters without a gradient too: a resume initializes every parameter's state (DCP's
            # _init_optim_state) and then needs each of its keys in the checkpoint
            for p in group["params"]:
                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)  # the fused kernel's layout
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                if p.dtype != torch.float32 and "master" not in state:
                    state["master"] = p.detach().to(torch.float32, copy=True)
                if state["step"].device != p.device:  # loaded from a state saved with host-side steps
                    state["step"] = state["step"].to(p.device)
            params = [p for p in group["params"] if p.grad is not None]
            beta1, beta2 = group["betas"]
            chunks, numel = [[]], 0
            for p in params:
                if numel >= CHUNK_NUMEL:
                    chunks.append([])
                    numel = 0
                chunks[-1].append(p)
                numel += p.numel()
            for chunk in filter(None, chunks):
                states = [self.state[p] for p in chunk]
                masters = [state.get("master", p) for p, state in zip(chunk, states)]
                fused = masters[0].is_cuda  # one kernel for the whole update; foreach elsewhere
                adamw(masters, [p.grad.float() for p in chunk], [s["exp_avg"] for s in states],
                      [s["exp_avg_sq"] for s in states], [], [s["step"] for s in states], foreach=not fused,
                      fused=fused, amsgrad=False, beta1=beta1, beta2=beta2, lr=group["lr"],
                      weight_decay=group["weight_decay"], eps=group["eps"], maximize=group["maximize"])
                low = [(p, m) for p, m in zip(chunk, masters) if m is not p]
                if low:
                    torch._foreach_copy_([p for p, _ in low], [m for _, m in low])
        return loss

    @staticmethod
    def _hold_fp32(optimizer, state_dict):
        saved = chain.from_iterable(group["params"] for group in state_dict["param_groups"])
        optimizer._held = [{k: v for k, v in state_dict["state"].get(i, {}).items() if k in FP32_STATE} for i in saved]

    @staticmethod
    def _restore_fp32(optimizer):
        current = chain.from_iterable(group["params"] for group in optimizer.param_groups)
        for p, held in zip(current, optimizer._held):
            optimizer.state[p].update({k: v.to(device=p.device, dtype=torch.float32) for k, v in held.items()})
        optimizer._held = []
