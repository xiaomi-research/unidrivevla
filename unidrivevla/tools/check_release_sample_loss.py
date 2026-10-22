"""One fixed sample through the PR 10 loss and the merged loss.

Planning loss is FlowPlanningLoss (the release stage-2 call: weight 1, hybrid 0.5,
detach window 3, nuScenes LiDAR delta stats). WASP is the unweighted ratio times the
release visual beta 0.01. Both implementations are loaded from their own source.
"""
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def load_source(text, filename):
    namespace = {"__name__": filename}
    exec(compile(text, filename, "exec"), namespace)
    return namespace


def flow_class(text, import_line):
    stub = (
        "class LOSSES:\n"
        "    @staticmethod\n"
        "    def register_module():\n"
        "        return lambda cls: cls\n"
    )
    return load_source(text.replace(import_line, stub), "flow_planning_loss.py")["FlowPlanningLoss"]


def main():
    torch.manual_seed(0)
    pred = torch.tensor([[[0.20, -0.10], [0.05, 0.40], [-0.30, 0.15],
                           [0.10, 0.05], [0.00, -0.20], [0.25, 0.10]]])
    target = torch.tensor([[[0.00, 0.10], [0.20, 0.30], [-0.10, 0.00],
                             [0.15, -0.05], [0.05, 0.25], [0.10, 0.00]]])
    pred_x0 = torch.tensor([[[0.10, 0.20], [-0.20, 0.50], [0.30, -0.10],
                              [0.00, 0.40], [0.15, 0.10], [-0.05, 0.30]]])
    gt_x0 = torch.tensor([[[0.00, 0.25], [-0.10, 0.45], [0.20, 0.05],
                            [0.10, 0.35], [0.05, 0.20], [0.00, 0.15]]])
    time = torch.tensor([0.30])
    mask = torch.ones(1, 6)
    kwargs = dict(use_min_snr_loss=True, min_snr_gamma=5.0, loss_weight=1.0,
                  hybrid_loss_weight=0.5, detach_window_size=3,
                  delta_mu=[0.0233, 2.2707], delta_std=[0.3427, 1.8668])

    ours_src = (ROOT / "models/losses/flow_planning_loss.py").read_text(encoding="utf-8")
    pr_src = subprocess.check_output(
        ["git", "show", "pr-10:nuScenes/projects/mmdet3d_plugin/losses/flow_planning_loss.py"],
        cwd=REPO, encoding="utf-8")
    ours_loss = flow_class(ours_src, "from unidrivevla.models.compat import LOSSES")(**kwargs)
    pr_loss = flow_class(pr_src, "from mmdet.models import LOSSES")(**kwargs)
    ours_plan = ours_loss(pred, target, time, mask, pred_x0, gt_x0)
    pr_plan = pr_loss(pred, target, time, mask, pred_x0, gt_x0)

    weight = torch.tensor([
        [0.4, -0.2, 0.1, 0.0, 0.3, -0.1],
        [-0.3, 0.5, 0.2, -0.4, 0.1, 0.2],
        [0.2, 0.1, -0.5, 0.3, 0.0, 0.4],
        [0.0, -0.2, 0.2, 0.5, -0.3, 0.1],
    ])
    layer = torch.nn.Linear(6, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    pr_wasp_src = subprocess.check_output(
        ["git", "show", "pr-10:nuScenes/projects/mmdet3d_plugin/unidrivevla/dense_heads/wasp.py"],
        cwd=REPO, encoding="utf-8").replace("import fcntl\n", "fcntl = None\n")
    our_wasp_src = (ROOT / "models/unidrivevla/dense_heads/wasp.py").read_text(encoding="utf-8")
    pr_wasp = load_source(pr_wasp_src, "pr_wasp.py")["WASPLinear"]
    our_wasp = load_source(our_wasp_src, "our_wasp.py")["WASPLinear"]
    A = torch.tensor([[0.2, -0.1, 0.0, 0.3, 0.1, -0.2],
                      [0.0, 0.4, -0.3, 0.1, 0.2, 0.0],
                      [-0.2, 0.1, 0.2, 0.0, -0.1, 0.3],
                      [0.1, 0.0, -0.2, 0.2, 0.3, -0.1]])
    B = torch.tensor([[0.3, 0.0, -0.2, 0.1],
                      [-0.1, 0.2, 0.0, 0.4],
                      [0.2, -0.3, 0.1, 0.0],
                      [0.0, 0.1, 0.2, -0.2]])
    beta = 0.01

    def run(cls):
        mod = cls(layer, rank=4, top_k=2, alpha=8.0, dropout=0.0, eps=1e-6)
        with torch.no_grad():
            mod.lora_A.copy_(A)
            mod.lora_B.copy_(B)
        return mod.wasp_regularization()

    pr_raw = run(pr_wasp)
    our_raw = run(our_wasp)
    pr_weighted = beta * pr_raw
    our_weighted = beta * our_raw

    print("sample pred", pred.tolist())
    print("sample target", target.tolist())
    print("sample pred_x0", pred_x0.tolist())
    print("sample gt_x0", gt_x0.tolist())
    print("sample time", time.tolist(), "mask", mask.tolist())
    print(f"planning loss  pr10={float(pr_plan):.8f}  merged={float(ours_plan):.8f}  diff={float(ours_plan - pr_plan):.3e}")
    print("wasp weight", weight.tolist())
    print("wasp A", A.tolist())
    print("wasp B", B.tolist())
    print(f"wasp raw       pr10={float(pr_raw):.8f}  merged={float(our_raw):.8f}  diff={float(our_raw - pr_raw):.3e}")
    print(f"wasp * 0.01    pr10={float(pr_weighted):.8f}  merged={float(our_weighted):.8f}  diff={float(our_weighted - pr_weighted):.3e}")
    if not torch.allclose(ours_plan, pr_plan) or not torch.allclose(our_raw, pr_raw):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
