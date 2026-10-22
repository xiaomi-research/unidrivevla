# WASP-style LoRA regularization.
#
# For a frozen pretrained linear weight W0, keep a standard LoRA update
#   Delta W = (alpha / r) * B A
# and softly penalize overlap with the dominant singular subspaces of W0:
#   R = (||U_k^T Delta W||_F^2 + ||Delta W V_k||_F^2)
#       / (||Delta W||_F^2 + eps)
from __future__ import annotations

try:
    import fcntl
except ImportError:  # the SVD cache lock is Unix-only; training runs on Linux
    fcntl = None
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_SVD_CACHE_VERSION = 1
_SVD_CACHE_ENV = "UNIDRIVE_WASP_SVD_CACHE_DIR"


def _compute_svd_basis(
    weight: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the same exact singular basis used by the original WASP code."""
    out_features, in_features = weight.shape
    k = min(int(top_k), out_features, in_features)
    weight_fp32 = weight.detach().float()
    u, _, vh = torch.linalg.svd(weight_fp32, full_matrices=False)
    return u[:, :k], vh[:k, :].t().contiguous()


def _weight_fingerprint(weight: torch.Tensor, sample_count: int = 2048) -> str:
    """Build a cheap, deterministic identity check for a pretrained weight."""
    flat = weight.detach().reshape(-1)
    count = min(int(sample_count), flat.numel())
    digest = hashlib.sha256()
    digest.update(str(tuple(weight.shape)).encode("ascii"))
    digest.update(str(weight.dtype).encode("ascii"))
    if count:
        if count == flat.numel():
            sample = flat
        elif count == 1:
            sample = flat[:1]
        else:
            indices = (
                torch.arange(count, device=flat.device, dtype=torch.long)
                * (flat.numel() - 1)
                // (count - 1)
            )
            sample = flat.index_select(0, indices)
        sample_bytes = (
            sample.float().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
        digest.update(sample_bytes)
    return digest.hexdigest()


def _build_cache_manifest(targets, top_k: int):
    layers = []
    for name, _, _, linear in targets:
        out_features, in_features = linear.weight.shape
        layers.append(
            {
                "name": name,
                "shape": [out_features, in_features],
                "dtype": str(linear.weight.dtype),
                "top_k": min(int(top_k), out_features, in_features),
                "fingerprint": _weight_fingerprint(linear.weight),
            }
        )
    encoded = json.dumps(layers, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return layers, hashlib.sha256(encoded).hexdigest()


def _resolve_cache_path(
    model: nn.Module,
    cfg: dict,
    top_k: int,
    manifest_hash: str,
) -> Optional[Path]:
    explicit_path = cfg.get("svd_cache_path")
    if explicit_path:
        return Path(os.path.expandvars(str(explicit_path))).expanduser()

    cache_dir = cfg.get("svd_cache_dir") or os.environ.get(_SVD_CACHE_ENV)
    if not cache_dir:
        return None

    namespace = str(cfg.get("svd_cache_namespace", model.__class__.__name__))
    namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", namespace).strip("._")
    namespace = namespace or "module"
    filename = f"{namespace}.top{int(top_k)}.{manifest_hash[:24]}.wasp_svd.pt"
    return Path(os.path.expandvars(str(cache_dir))).expanduser() / filename


def _load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_cache(
    payload,
    manifest,
    manifest_hash: str,
) -> Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _SVD_CACHE_VERSION:
        return None
    if payload.get("manifest_hash") != manifest_hash:
        return None
    if payload.get("manifest") != manifest:
        return None

    cached_layers = payload.get("layers")
    if not isinstance(cached_layers, dict):
        return None

    result = {}
    for layer in manifest:
        name = layer["name"]
        cached = cached_layers.get(name)
        if not isinstance(cached, dict):
            return None
        u_top = cached.get("U_top")
        v_top = cached.get("V_top")
        out_features, in_features = layer["shape"]
        k = layer["top_k"]
        if not isinstance(u_top, torch.Tensor) or not isinstance(v_top, torch.Tensor):
            return None
        if tuple(u_top.shape) != (out_features, k):
            return None
        if tuple(v_top.shape) != (in_features, k):
            return None
        result[name] = (u_top, v_top)
    return result


def _write_cache_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            torch.save(payload, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _load_or_build_svd_cache(
    model: nn.Module,
    targets,
    cfg: dict,
    top_k: int,
) -> Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    cache_requested = (
        cfg.get("svd_cache_path")
        or cfg.get("svd_cache_dir")
        or os.environ.get(_SVD_CACHE_ENV)
    )
    if not cache_requested:
        return None

    manifest, manifest_hash = _build_cache_manifest(targets, top_k)
    cache_path = _resolve_cache_path(model, cfg, top_k, manifest_hash)
    if cache_path is None:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{cache_path}.lock")
    start = time.monotonic()
    if cache_path.is_file():
        try:
            cached_basis = _validate_cache(
                _load_torch_file(cache_path), manifest, manifest_hash
            )
        except Exception:
            cached_basis = None
        if cached_basis is not None:
            elapsed = time.monotonic() - start
            print(
                f"[WASP SVD] cache hit: {cache_path} "
                f"({len(targets)} layers, {elapsed:.2f}s)"
            )
            return cached_basis

    with lock_path.open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached_basis = None
        if cache_path.is_file():
            try:
                cached_basis = _validate_cache(
                    _load_torch_file(cache_path), manifest, manifest_hash
                )
            except Exception as exc:
                print(f"[WASP SVD] Ignoring invalid cache {cache_path}: {exc}")

        if cached_basis is not None:
            elapsed = time.monotonic() - start
            print(
                f"[WASP SVD] cache hit: {cache_path} "
                f"({len(targets)} layers, {elapsed:.2f}s)"
            )
            return cached_basis

        build_start = time.monotonic()
        cached_basis = {}
        serialized_layers = {}
        for name, _, _, linear in targets:
            u_top, v_top = _compute_svd_basis(linear.weight, top_k)
            u_top = u_top.to(dtype=linear.weight.dtype, device="cpu")
            v_top = v_top.to(dtype=linear.weight.dtype, device="cpu")
            cached_basis[name] = (u_top, v_top)
            serialized_layers[name] = {"U_top": u_top, "V_top": v_top}

        payload = {
            "version": _SVD_CACHE_VERSION,
            "manifest": manifest,
            "manifest_hash": manifest_hash,
            "layers": serialized_layers,
        }
        _write_cache_atomic(cache_path, payload)
        elapsed = time.monotonic() - build_start
        print(
            f"[WASP SVD] cache built: {cache_path} "
            f"({len(targets)} layers, {elapsed:.2f}s)"
        )
        return cached_basis


class WASPLinear(nn.Module):
    """Drop-in nn.Linear replacement with standard LoRA plus WASP loss."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 64,
        top_k: int = 16,
        alpha: float = 128.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
        u_top: Optional[torch.Tensor] = None,
        v_top: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        out_features, in_features = base_layer.weight.shape

        self.in_features = in_features
        self.out_features = out_features
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.eps = float(eps)

        k = min(int(top_k), out_features, in_features)
        if (u_top is None) != (v_top is None):
            raise ValueError("u_top and v_top must be provided together.")
        if u_top is None:
            u_top, v_top = _compute_svd_basis(base_layer.weight, k)
        if tuple(u_top.shape) != (out_features, k):
            raise ValueError(
                f"Invalid U_top shape {tuple(u_top.shape)}; "
                f"expected {(out_features, k)}."
            )
        if tuple(v_top.shape) != (in_features, k):
            raise ValueError(
                f"Invalid V_top shape {tuple(v_top.shape)}; "
                f"expected {(in_features, k)}."
            )
        basis_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.register_buffer("U_top", u_top.detach().to(**basis_kwargs).contiguous())
        self.register_buffer("V_top", v_top.detach().to(**basis_kwargs).contiguous())

        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        x_drop = self.dropout(x)
        lora_out = (x_drop @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def _get_fp32_basis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (
            self.U_top.device,
            self.U_top._version,
            self.V_top.device,
            self.V_top._version,
        )
        if self.__dict__.get("_fp32_basis_cache_key") != cache_key:
            self.__dict__["_u_top_fp32_cache"] = self.U_top.float()
            self.__dict__["_v_top_fp32_cache"] = self.V_top.float()
            self.__dict__["_fp32_basis_cache_key"] = cache_key
        return (
            self.__dict__["_u_top_fp32_cache"],
            self.__dict__["_v_top_fp32_cache"],
        )

    def wasp_regularization(self) -> torch.Tensor:
        """Return the unweighted WASP ratio for this layer."""
        a = self.lora_A.float()
        b = self.lora_B.float()
        u, v = self._get_fp32_basis()

        aa_t = a @ a.t()
        bt_b = b.t() @ b

        # Compute the three Frobenius norms without materializing Delta W.
        delta_norm_sq = (bt_b * aa_t.t()).sum()
        u_b = u.t() @ b
        u_term = ((u_b.t() @ u_b) * aa_t.t()).sum()
        a_v = a @ v
        v_term = (bt_b * (a_v @ a_v.t()).t()).sum()

        scaling_sq = self.scaling * self.scaling
        return scaling_sq * (u_term + v_term) / (
            scaling_sq * delta_norm_sq + self.eps
        )

    def merge_weight(self) -> torch.Tensor:
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return self.weight + delta

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, top_k={self.U_top.shape[1]}, "
            f"scaling={self.scaling:.4f}"
        )


def _iter_target_layers(model: nn.Module, target_modules: List[str]):
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name not in target_modules:
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]
        yield name, parent, attr, module


def apply_wasp(model: nn.Module, cfg: dict) -> int:
    model.__dict__.pop("_wasp_regularization_modules", None)
    rank = int(cfg.get("rank", cfg.get("r", 64)))
    top_k = int(cfg.get("top_k", cfg.get("preserve_top_k", 16)))
    alpha = float(cfg.get("alpha", cfg.get("lora_alpha", 128)))
    dropout = float(cfg.get("dropout", cfg.get("lora_dropout", 0.0)))
    eps = float(cfg.get("eps", 1e-6))
    target_modules: List[str] = cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model.requires_grad_(False)
    targets = list(_iter_target_layers(model, target_modules))
    if not targets:
        raise RuntimeError(
            f"WASP did not match any Linear layers for targets={target_modules}."
        )

    cached_basis = _load_or_build_svd_cache(model, targets, cfg, top_k)
    for name, parent, attr, linear in targets:
        u_top, v_top = (None, None)
        if cached_basis is not None:
            u_top, v_top = cached_basis[name]
        wasp_layer = WASPLinear(
            base_layer=linear,
            rank=rank,
            top_k=top_k,
            alpha=alpha,
            dropout=dropout,
            eps=eps,
            u_top=u_top,
            v_top=v_top,
        ).to(device=linear.weight.device)
        setattr(parent, attr, wasp_layer)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[WASP] replaced {len(targets)} Linear layers. "
        f"trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / max(total, 1):.2f}%)"
    )
    return len(targets)


def compute_wasp_regularization(model: nn.Module) -> Optional[torch.Tensor]:
    wasp_modules = model.__dict__.get("_wasp_regularization_modules")
    if wasp_modules is None:
        wasp_modules = tuple(
            module for module in model.modules()
            if isinstance(module, WASPLinear)
        )
        model.__dict__["_wasp_regularization_modules"] = wasp_modules
    losses = [
        module.wasp_regularization()
        for module in wasp_modules
    ]
    if not losses:
        return None
    return torch.stack(losses).sum()


def merge_wasp(model: nn.Module) -> int:
    model.__dict__.pop("_wasp_regularization_modules", None)
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, WASPLinear):
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]

        merged_weight = module.merge_weight().detach()
        new_linear = nn.Linear(
            merged_weight.shape[1],
            merged_weight.shape[0],
            bias=module.bias is not None,
            device=merged_weight.device,
            dtype=merged_weight.dtype,
        )
        new_linear.weight = nn.Parameter(merged_weight)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.detach().clone())
        setattr(parent, attr, new_linear)
        merged += 1

    print(f"[WASP] merged {merged} WASPLinear layers back to nn.Linear.")
    return merged

