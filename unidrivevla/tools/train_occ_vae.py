"""Train VAERes3D the way DOME trains Occ-VAE.

DOME (config/train_occvae.py) uses AdamW 1e-3, grad clip 35, and
10 * cross-entropy + Lovasz + 5e-5 * KL. This script keeps that loss and
the VAERes3D layout the planning head loads, on NAVSIM grids of
200x200x16 with 9 classes (free + the 7 detection classes + background).

OpenScene stores occupancy as a sparse (N, 2) array: flat voxel index and
class id. Empty voxels are absent and class 16 is free. The flat index
matches OpenScene's meshgrid order, x fastest, then y, then z.
"""

import argparse
import glob
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import unidrivevla.models.unidrivevla.vae  # noqa: E402,F401
from unidrivevla.models.compat import build_head  # noqa: E402

OCC_SHAPE = (200, 200, 16)
NUM_CLASSES = 9
CE_WEIGHT = 10.0
LOVASZ_WEIGHT = 1.0
KL_WEIGHT = 5e-5
# OpenScene ids from CustomNuDataset.evaluate_occ_iou. 16 is the empty fill.
OPENSCENE_TO_OURS = {
    16: 0,  # free
    10: 1,  # background
    0: 2,  # vehicle
    5: 3,  # bicycle
    7: 4,  # pedestrian
    8: 5,  # traffic_cone
    9: 6,  # barrier
    4: 7,  # czone_sign
    6: 8,  # generic_object
}


def build_vae():
    # Same widths as unidrivevla_config.head_kwargs, 9 classes instead of DOME's 18.
    return build_head(dict(
        type="VAERes3D",
        encoder_cfg=dict(
            type="Encoder2D", ch=64, out_ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
            attn_resolutions=(50,), dropout=0.0, resamp_with_conv=True, in_channels=128,
            resolution=200, z_channels=128, double_z=False,
        ),
        decoder_cfg=dict(
            type="Decoder3D", ch=64, out_ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
            attn_resolutions=(50,), dropout=0.0, resamp_with_conv=True, in_channels=128,
            resolution=200, z_channels=64, give_pre_end=False,
        ),
        num_classes=NUM_CLASSES,
        expansion=8,
        vqvae_cfg=None,
    ))


def recon_loss(logits, labels):
    return F.cross_entropy(logits.permute(0, 5, 1, 2, 3, 4), labels)


def kl_loss(mu, sigma):
    # sample_z returns std = exp(logvar / 2). DOME's KL is the standard one on logvar.
    logvar = 2.0 * sigma.clamp_min(1e-6).log()
    mu = mu.permute(0, 2, 1, 3, 4).flatten(0, 1)
    logvar = logvar.permute(0, 2, 1, 3, 4).flatten(0, 1)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3)).mean()


def _lovasz_grad(gt_sorted):
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_loss(logits, labels):
    prob = logits.flatten(0, 1).permute(0, 4, 1, 2, 3).softmax(dim=1)
    labels = labels.flatten(0, 1)
    c = prob.shape[1]
    prob = prob.permute(0, 2, 3, 4, 1).reshape(-1, c)
    labels = labels.reshape(-1)
    losses = []
    for cls in range(c):
        fg = (labels == cls).float()
        if fg.sum() == 0:
            continue
        errors = (fg - prob[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def vae_loss(out, labels):
    ce = recon_loss(out["logits"], labels)
    lovasz = lovasz_loss(out["logits"], labels)
    kl = kl_loss(out["z_mu"], out["z_sigma"])
    total = CE_WEIGHT * ce + LOVASZ_WEIGHT * lovasz + KL_WEIGHT * kl
    return total, {"ce": float(ce), "lovasz": float(lovasz), "kl": float(kl)}


def lr_at(step, total, warmup, base=1e-3, min_lr=1e-6):
    if step < warmup:
        return min_lr + (base - min_lr) * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def densify(sparse):
    """OpenScene (N, 2) index/class -> (200, 200, 16) with ids 0..8."""
    xdim, ydim, zdim = OCC_SHAPE
    grid = np.zeros((xdim, ydim, zdim), np.int64)
    idx = np.asarray(sparse[:, 0], np.int64)
    cls = np.asarray(sparse[:, 1], np.int64)
    mapped = np.ones(cls.shape, np.int64)
    for src, dst in OPENSCENE_TO_OURS.items():
        mapped[cls == src] = dst
    ok = (idx >= 0) & (idx < xdim * ydim * zdim)
    idx, mapped = idx[ok], mapped[ok]
    x = idx % xdim
    y = (idx // xdim) % ydim
    z = idx // (xdim * ydim)
    grid[x, y, z] = mapped
    return grid


def load_grid(path):
    raw = np.load(path)
    if getattr(raw, "files", None) is not None:
        raw = raw["semantics"] if "semantics" in raw.files else raw[raw.files[0]]
    raw = np.asarray(raw)
    if raw.ndim == 2 and raw.shape[1] == 2:
        grid = densify(raw)
    elif raw.shape == OCC_SHAPE:
        grid = raw.astype(np.int64)
        if grid.max() >= NUM_CLASSES:
            grid = densify(np.stack([np.arange(grid.size), grid.reshape(-1)], axis=1))
    else:
        raise ValueError(f"{path} has shape {raw.shape}")
    if grid.min() < 0 or grid.max() >= NUM_CLASSES:
        raise ValueError(f"{path} class id outside 0..{NUM_CLASSES - 1}")
    return grid


def resolve_occ(data_root, rel):
    rel = str(rel).replace("\\", "/")
    cands = [
        rel,
        os.path.join(data_root, rel),
        os.path.join(os.path.dirname(data_root.rstrip("/")), rel),
    ]
    if rel.startswith("dataset/"):
        cands.append(os.path.join(data_root, rel[len("dataset/"):]))
    for cand in cands:
        if cand and os.path.isfile(cand):
            return cand
    return None


class NavsimOcc(Dataset):
    def __init__(self, log_dir, data_root, limit=0):
        files = []
        missing = 0
        for pkl in sorted(glob.glob(os.path.join(log_dir, "*.pkl"))):
            with open(pkl, "rb") as handle:
                frames = pickle.load(handle)
            for frame in frames:
                rel = frame.get("occ_gt_final_path") or ""
                if not rel:
                    continue
                path = resolve_occ(data_root, rel)
                if path is None:
                    missing += 1
                    continue
                files.append(path)
                if limit and len(files) >= limit:
                    break
            if limit and len(files) >= limit:
                break
        if not files:
            raise FileNotFoundError(
                f"navsim logs in {log_dir} point at occupancy under {data_root}, but 0 files exist ({missing} missing)"
            )
        self.files = files
        print(f"occ files {len(files)} missing {missing}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.from_numpy(load_grid(self.files[index])[None])


def self_check():
    sparse = np.array([[0, 0], [200, 5], [200 * 200, 16]], np.int64)
    grid = densify(sparse)
    assert grid.shape == OCC_SHAPE
    assert grid[0, 0, 0] == 2 and grid[0, 1, 0] == 3 and grid[0, 0, 1] == 0
    assert int(grid.sum()) == 5
    logits = torch.randn(2, 1, 4, 4, 2, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (2, 1, 4, 4, 2))
    mu = torch.zeros(2, 4, 1, 2, 2)
    sigma = torch.ones(2, 4, 1, 2, 2)
    total, parts = vae_loss({"logits": logits, "z_mu": mu, "z_sigma": sigma}, labels)
    assert total.ndim == 0 and math.isfinite(float(total)), parts
    print("loss_ok", float(total), parts, "grid", tuple(grid.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occ-root", default=os.environ.get("NAVSIM_OCC_ROOT", ""))
    parser.add_argument("--out", default=os.environ.get("OCCWORLD_VAE_PATH", "occvae_navsim.pth"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    data = OccGrids(args.occ_root)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vae().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    step = 0
    stop = False
    print(f"grids {len(data)} steps {total_steps} device {device}")
    for epoch in range(args.epochs):
        for batch in loader:
            batch = batch.to(device)
            opt.param_groups[0]["lr"] = lr_at(step, total_steps, args.warmup, args.lr)
            out = model(batch)
            loss, parts = vae_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            opt.step()
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {float(loss):.4f} ce {parts['ce']:.4f} lovasz {parts['lovasz']:.4f} kl {parts['kl']:.4f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
