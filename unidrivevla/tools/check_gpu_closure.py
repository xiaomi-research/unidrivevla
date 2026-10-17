"""Prepare and run the short, guarded 2027 GPU closure checks.

This is a verification harness, not a training launcher.  ``compose`` is CPU-only;
actual cases require the explicit ``--execute`` switch and recheck every GPU
immediately before starting.  Each run writes into a fresh, owned setup bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = Path("/data/yongkangli/UniDriveVLA")
PYTHON = Path("/data/yongkangli/miniconda3/envs/navsim/bin/python")
VLM = Path("/data/yongkangli/ckpts/Qwen3-VL-2B-Instruct")
NUSC_INFOS = FORMAL_ROOT / "nuScenes/data/infos/mini"
NUSC_DATA = FORMAL_ROOT / "nuScenes/data/nuscenes"
B2D_INFOS = FORMAL_ROOT / "Bench2Drive/data/infos"
B2D_DATA = FORMAL_ROOT / "Bench2Drive/data/bench2drive"
NAVSIM_EXP = Path("/data/yongkangli/NAVSIM/exp")
NAVSIM_LOG = "2021.05.12.19.36.12_veh-35_00005_00204"
NAVSIM_CACHE = {
    "1.1": NAVSIM_EXP / "unidrivevla_sample_cache_navsim1.1_smoke",
    "2.0": NAVSIM_EXP / "unidrivevla_sample_cache_navsim2.0_smoke",
}
NAVSIM_DEVKIT = {
    "1.1": FORMAL_ROOT / "NavSim/navsim1.1",
    "2.0": FORMAL_ROOT / "NavSim/navsim2.0",
}
NAVSIM_ANCHORS = FORMAL_ROOT / "anchors/navsim"
DAF_SOURCE_SHA256 = {
    "models/ops/src/deformable_aggregation.cpp": "10dee0228ef8d815f181fd2b4a801a487174d49c01b44414572541a8477db3a6",
    "models/ops/src/deformable_aggregation_cuda.cu": "3b81926b2e0ebc00341d38bd9db32dee27300951e067479ab74693c0214bfcbd",
}
GPU_CASES = (
    "nuscenes_s1", "nuscenes_s2", "frozen_prefix",
    "b2d_s1", "b2d_s2", "navsim11_s1", "navsim11_s2",
    "navsim20_s1", "navsim20_s2", "multidomain_vqa",
)
SINGLE_CASES = {
    "nuscenes_s1": ("nuscenes", "1.1", "stage1"),
    "nuscenes_s2": ("nuscenes", "1.1", "stage2"),
    "b2d_s1": ("b2d", "1.1", "stage1"),
    "b2d_s2": ("b2d", "1.1", "stage2"),
    "navsim11_s1": ("navsim", "1.1", "stage1"),
    "navsim11_s2": ("navsim", "1.1", "stage2"),
    "navsim20_s1": ("navsim", "2.0", "stage1"),
    "navsim20_s2": ("navsim", "2.0", "stage2"),
}
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _json(path: Path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex[:8])
    _json(temporary, value, exclusive=True)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_root(raw: str) -> Path:
    if not raw:
        raise ValueError("--root must name the unique setup/codex_gpu_closure_<uuid> bundle")
    supplied = Path(raw)
    root = supplied.resolve()
    expected_parent = FORMAL_ROOT / "setup"
    if (supplied.is_symlink() or supplied.absolute() != root or root.parent != expected_parent or
            not root.name.startswith("codex_gpu_closure_")):
        raise ValueError(f"refusing non-owned bundle path: {root}")
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"owned bundle directory does not exist as a physical directory: {root}")
    return root


def _bundle_files(root: Path):
    return {
        "code": root / "code",
        "package": root / "code/unidrivevla",
        "logs": root / "logs",
        "exp": root / "exp",
        "evidence": root / "evidence",
        "artifacts": root / "artifacts",
    }


def _short_tmpdir(root: Path) -> Path:
    suffix = root.name.rsplit("_", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{12}", suffix):
        raise ValueError(f"owned bundle name must end in a 12-hex run id: {root.name}")
    return Path("/data/yongkangli/tmp") / ("u_" + suffix)


def _validate_tmpdir(tmpdir: Path):
    base = Path("/data/yongkangli/tmp")
    if base.is_symlink() or tmpdir.is_symlink() or tmpdir.resolve() != tmpdir:
        raise RuntimeError(f"multiprocessing TMPDIR must be a physical non-symlink directory: {tmpdir}")
    if not tmpdir.is_dir():
        raise FileNotFoundError(f"short owned TMPDIR was not created by the bundle stager: {tmpdir}")
    info = tmpdir.stat()
    if (info.st_mode & 0o777) != 0o700:
        raise PermissionError(f"short owned TMPDIR must have mode 0700, got {info.st_mode & 0o777:o}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise PermissionError("short owned TMPDIR is not owned by the current server account")
    if info.st_dev != base.stat().st_dev:
        raise RuntimeError("short TMPDIR must reside on the existing user /data temp filesystem")


def _multiprocessing_socket_smoke(tmpdir: Path):
    """Exercise the same AF_UNIX resource_sharer Listener used to pass DataLoader tensors."""
    import multiprocessing.util as mp_util
    import tempfile

    os.environ["TMPDIR"] = str(tmpdir)
    tempfile.tempdir = None
    mp_util._tempdir = None
    if tempfile.gettempdir() != str(tmpdir):
        raise RuntimeError(f"Python ignored the owned short TMPDIR: {tempfile.gettempdir()}")
    from multiprocessing import resource_sharer
    from multiprocessing.reduction import DupFd

    read_fd, write_fd = os.pipe()
    socket_path = None
    try:
        _duplicate = DupFd(read_fd)
        listener = resource_sharer._resource_sharer._listener
        socket_path = str(listener.address)
        path_bytes = len(os.fsencode(socket_path))
        if not socket_path.startswith(str(tmpdir) + os.sep):
            raise RuntimeError(f"resource_sharer socket escaped the owned TMPDIR: {socket_path}")
        if path_bytes >= 108:
            raise OSError(f"resource_sharer AF_UNIX path still exceeds 107 bytes: {path_bytes}")
    finally:
        resource_sharer._resource_sharer.stop()
        os.close(read_fd)
        os.close(write_fd)
    if socket_path and Path(socket_path).exists():
        raise RuntimeError(f"resource_sharer socket was not closed: {socket_path}")
    return {"tmpdir": str(tmpdir), "socket_path": socket_path,
            "socket_path_bytes": len(os.fsencode(socket_path)), "socket_path_max_bytes": 107,
            "listener_created_and_closed": True}


def _require_snapshot_code(root: Path):
    expected = _bundle_files(root)["code"].resolve()
    if CODE_ROOT.resolve() != expected:
        raise RuntimeError(f"run the harness and Hydra configs from the owned snapshot {expected}; got {CODE_ROOT}")
    if _bundle_files(root)["package"].resolve() == (FORMAL_ROOT / "unidrivevla").resolve():
        raise RuntimeError("formal package cannot be used as the GPU smoke code snapshot")


def _setup_python_path(root: Path, tree: str = "1.1"):
    files = _bundle_files(root)
    devkit = NAVSIM_DEVKIT[tree]
    for value in (str(files["code"]), str(devkit)):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _expected_environment(root: Path, tree: str = "1.1"):
    files = _bundle_files(root)
    env = os.environ.copy()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(key, None)
    env.update({
        "UNIDRIVE_EXP_ROOT": str(files["exp"]),
        "UNIDRIVE_ANCHORS_ROOT": str(FORMAL_ROOT / "anchors"),
        "UNIDRIVE_NATIVE_ANCHORS_ROOT": str(FORMAL_ROOT / "anchors"),
        "NUSCENES_INFOS": str(NUSC_INFOS),
        "NUSCENES_PATH_ROOT": str(FORMAL_ROOT / "nuScenes"),
        "NUSCENES_ROOT": str(NUSC_DATA),
        "NUSCENES_VERSION": "v1.0-mini",
        "B2D_INFOS": str(B2D_INFOS),
        "B2D_ROOT": str(B2D_DATA),
        "NAVSIM_DEVKIT_ROOT": str(NAVSIM_DEVKIT[tree]),
        "NAVSIM_EXP_ROOT": str(NAVSIM_EXP),
        "OPENSCENE_DATA_ROOT": str(Path("/data/yongkangli/NAVSIM/dataset")),
        "NUPLAN_MAPS_ROOT": "/data/yongkangli/NAVSIM/dataset/maps",
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "VLM_PRETRAINED_PATH": str(VLM),
        "PYTHONFAULTHANDLER": "1",
        "PYTHONPATH": os.pathsep.join((str(NAVSIM_DEVKIT[tree]), str(files["code"]))),
        "TMPDIR": str(_short_tmpdir(root)),
        "XDG_CACHE_HOME": str(root / "cache/xdg"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "cache/inductor"),
        "TRITON_CACHE_DIR": str(root / "cache/triton"),
        "CUDA_CACHE_PATH": str(root / "cache/cuda"),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def _anchor_paths(raw: str):
    if not raw:
        raise ValueError("--native-anchors must be the integration worker's development-only anchors root")
    root = Path(raw).resolve()
    if (root.name != "anchors" or root.parent.parent != FORMAL_ROOT / "setup" or
            not root.parent.name.startswith("codex_native_dev_anchors_")):
        raise ValueError("native anchors must come from the owned development-only fitting directory")
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root / "nuscenes", root / "b2d"


def _common_overrides(name: str, root: Path, tree: str, stage: str, suite_id: str, case: str,
                      nusc_anchor: Path, b2d_anchor: Path, vqa_jsonl: Path):
    values = [
        f"experiment_name={suite_id}_{case}", f"data={name}", f"stage={stage}",
        "train.max_epochs=1", "train.max_steps=3", "train.global_batch=0",
        "train.log_every=1", "train.skip_validation=false", "train.save_every_n_steps=0",
        "evaluator.every_n_epochs=0", "evaluator.on_end=false",
        "dataloader.params.batch_size=1", "dataloader.params.num_workers=2",
        "dataloader.params.pin_memory=true",
    ]
    if stage == "stage2":
        values.append(f"agent.checkpoint_path={_stage1_checkpoint(root, suite_id, name, tree)}")
    if name == "nuscenes":
        values += ["nuscenes.occ_root=null", "agent.vae_path=null", f"profile.anchors_dir={nusc_anchor}"]
    elif name == "b2d":
        values.append(f"profile.anchors_dir={b2d_anchor}")
    elif name == "navsim":
        values += [
            "train_test_split=navtrain", f"cache_path={NAVSIM_CACHE[tree]}",
            f'train_logs=["{NAVSIM_LOG}"]', f'val_logs=["{NAVSIM_LOG}"]',
            f"profile.anchors_dir={NAVSIM_ANCHORS}",
        ]
        if stage == "stage2":
            values.append("+stage2_max_epochs=1")
    return values


def _stage1_checkpoint(root: Path, suite_id: str, name: str, tree: str = "1.1"):
    case = {"nuscenes": "nuscenes_s1", "b2d": "b2d_s1",
            "navsim": "navsim11_s1" if tree == "1.1" else "navsim20_s1"}[name]
    return root / "exp" / f"{suite_id}_{case}" / "last.ckpt"


def _case_overrides(case: str, root: Path, native_anchors: Path, vqa_jsonl: Path, suite_id: str):
    nusc_anchor, b2d_anchor = _anchor_paths(str(native_anchors))
    if case in SINGLE_CASES:
        name, tree, stage = SINGLE_CASES[case]
        return tree, _common_overrides(name, root, tree, stage, suite_id, case,
                                       nusc_anchor, b2d_anchor, vqa_jsonl)
    if case == "multidomain_vqa":
        return "1.1", [
            f"experiment_name={suite_id}_{case}", "data=multidomain", "agent=multidomain",
            "stage=stage1", "vqa=combined", "vqa.enabled=true", f"vqa.jsonl_paths=[{vqa_jsonl}]",
            "vqa.batch_size=1", "vqa.num_workers=0", "vqa.max_length=128", "vqa.max_pixels=40000",
            "train.max_epochs=1", "train.max_steps=18", "train.global_batch=0", "train.log_every=1",
            "train.skip_validation=false", "train.save_every_n_steps=0",
            "evaluator.every_n_epochs=0", "evaluator.on_end=false",
            "dataloader.params.batch_size=1", "dataloader.params.num_workers=2",
            "dataloader.params.pin_memory=true", "nuscenes.occ_root=null",
            f"agent.vae_path=null", f"profiles.nuscenes.anchors_dir={nusc_anchor}",
            f"profiles.b2d.anchors_dir={b2d_anchor}", f"profiles.navsim.anchors_dir={NAVSIM_ANCHORS}",
            "seed=17", "train.exact_resume=false",
            f"cache_path={NAVSIM_CACHE['1.1']}", "train_test_split=navtrain",
            f'train_logs=["{NAVSIM_LOG}"]', f'val_logs=["{NAVSIM_LOG}"]',
        ]
    raise ValueError(f"unknown case {case!r}")


def _profile_summary(cfg):
    if cfg.get("profiles"):
        profiles = cfg.profiles
    else:
        profiles = {str(cfg.profile.name): cfg.profile}
    out = {}
    for name, profile in profiles.items():
        out[str(name)] = {
            "frame": str(profile.frame),
            "prefix_max_len": int(profile.prefix_max_len or 0),
            "official_attention_mask": bool(profile.get("official_attention_mask", False)),
            "lr_microstep": bool(profile.get("lr_microstep", False)),
            "anchors_dir": str(profile.anchors_dir),
        }
    return out


def _validate_recipe(name: str, stage: str, cfg):
    if stage == "stage1":
        assert bool(cfg.agent.train_vlm), "stage1 must train the VLM/LoRA"
        assert not bool(cfg.agent.with_motion), "stage1 must not construct motion"
    elif stage == "stage2":
        assert not bool(cfg.agent.train_vlm), "stage2 must freeze the VLM/LoRA"
        assert bool(cfg.agent.with_motion), "stage2 must enable motion"
    if name in ("nuscenes", "b2d"):
        profile = cfg.profile
        expected = 3740 if name == "nuscenes" else 3670
        assert int(profile.prefix_max_len) == expected, (name, profile.prefix_max_len, expected)
        assert bool(profile.official_attention_mask)
        assert str(profile.frame) == "lidar"
    if name == "nuscenes":
        assert cfg.nuscenes.occ_root is None and cfg.agent.vae_path is None


def _compose_one(root: Path, tree: str, overrides, name: str, stage: str, write=False):
    _setup_python_path(root, tree)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CODE_ROOT / "unidrivevla/config"), version_base=None):
        cfg = compose("train", overrides=overrides)
    if name != "multidomain":
        _validate_recipe(name, stage, cfg)
    else:
        assert stage == "stage1"
        assert len(cfg.profiles) == 3 and len(cfg.multidomain.sources) == 3
        assert cfg.vqa.enabled and cfg.vqa.mode == "combined" and cfg.vqa.jsonl_paths
        assert bool(cfg.agent.train_vlm) and not bool(cfg.agent.with_motion)
    # Resolve every config interpolation now; the composed profile is also the recipe evidence.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    info = {
        "case": name + ("_" + stage if name != "multidomain" else "_vqa"),
        "data_loader": str(cfg.data_loader),
        "stage": stage,
        "agent_train_vlm": bool(cfg.agent.train_vlm),
        "agent_with_motion": bool(cfg.agent.with_motion),
        "microbatch": int(cfg.dataloader.params.batch_size),
        "max_steps": int(cfg.train.max_steps),
        "profiles": _profile_summary(cfg),
        "resolved_config_sha256": hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")).hexdigest(),
    }
    if name == "multidomain":
        info["vqa"] = {"enabled": bool(cfg.vqa.enabled), "mode": str(cfg.vqa.mode),
                       "batch_size": int(cfg.vqa.batch_size), "jsonl_paths": list(cfg.vqa.jsonl_paths)}
        info["sources"] = [str(source.name) for source in cfg.multidomain.sources]
    if name == "navsim" and stage == "stage2":
        info["stage2_epoch_override"] = int(cfg.train.max_epochs)
        info["stage2_recipe_note"] = "NAVSIM stage2_max_epochs=1 is a smoke-only architecture/reload override"
    if name == "nuscenes":
        info["loss_limit"] = "Occ3D/OccWorld/VAE supervision absent: nuscenes.occ_root=null and agent.vae_path=null"
    if write:
        return info, resolved
    return info, None


def _gpu_inventory():
    gpu_command = ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                   "--format=csv,noheader,nounits"]
    process_command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                       "--format=csv,noheader,nounits"]
    try:
        gpu = subprocess.run(gpu_command, capture_output=True, text=True, timeout=10, check=True)
        proc = subprocess.run(process_command, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ready": False, "error": str(exc), "gpus": [], "compute_processes": []}
    rows = []
    try:
        for line in gpu.stdout.splitlines():
            pieces = [part.strip() for part in line.split(",")]
            if len(pieces) != 3:
                raise ValueError(f"malformed GPU inventory row: {line!r}")
            rows.append({"index": int(pieces[0]), "free_mib": int(pieces[1]), "util_percent": int(pieces[2])})
    except ValueError as exc:
        return {"ready": False, "error": str(exc), "gpus": rows, "compute_processes": []}
    process_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    expected_indices = [0, 1, 2, 3]
    ready = (len(rows) == 4 and [item["index"] for item in rows] == expected_indices
             and not process_lines and all(row["free_mib"] >= 23000 and row["util_percent"] < 5 for row in rows))
    return {"ready": ready, "gpus": rows, "compute_processes": process_lines,
            "criteria": {"cards": 4, "free_mib_each_min": 23000, "util_percent_each_max_exclusive": 5,
                         "compute_processes": 0}}


def _assert_physical_isolated(path: Path, formal_path: Path):
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"bundle file is absent or symlinked: {path}")
    if formal_path.exists() and formal_path.is_file():
        local_stat, formal_stat = path.stat(), formal_path.stat()
        if (local_stat.st_dev, local_stat.st_ino) == (formal_stat.st_dev, formal_stat.st_ino):
            raise RuntimeError(f"bundle file is a hardlink to the formal package: {path}")


def _preflight(root: Path, native_root: Path, vqa_image: Path, vqa_jsonl: Path):
    files = _bundle_files(root)
    tmpdir = _short_tmpdir(root)
    _validate_tmpdir(tmpdir)
    multiprocessing_smoke = _multiprocessing_socket_smoke(tmpdir)
    for key in ("code", "package", "logs", "exp", "evidence", "artifacts"):
        if not files[key].exists():
            raise FileNotFoundError(files[key])
    if files["package"].resolve() == (FORMAL_ROOT / "unidrivevla").resolve():
        raise RuntimeError("GPU smoke must use an independent physical source copy")
    _require_snapshot_code(root)
    if not PYTHON.exists() or not VLM.is_dir():
        raise FileNotFoundError(f"required navsim Python or Qwen2B weights missing: {PYTHON}, {VLM}")
    for directory in (NUSC_INFOS, NUSC_DATA, B2D_INFOS, B2D_DATA,
                      NAVSIM_DEVKIT["1.1"], NAVSIM_DEVKIT["2.0"],
                      NAVSIM_CACHE["1.1"], NAVSIM_CACHE["2.0"], NAVSIM_ANCHORS):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    nusc_anchor, b2d_anchor = _anchor_paths(str(native_root))
    anchor_report = {}
    for label, directory in (("nuscenes", nusc_anchor), ("b2d", b2d_anchor)):
        for filename in ("det_900.npy", "map_100.npy", "motion_6.npy"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(directory / filename)
        anchor_report[label] = str(directory)
    if not vqa_image.is_file():
        raise FileNotFoundError(vqa_image)
    if not str(vqa_image.resolve()).startswith(str(NUSC_DATA.resolve()) + os.sep):
        raise ValueError("synthetic VQA image must come from the verified nuScenes mini sample tree")
    if not vqa_jsonl.is_file():
        raise FileNotFoundError(vqa_jsonl)
    so = files["package"] / "models/ops/deformable_aggregation_ext.cpython-39-x86_64-linux-gnu.so"
    formal_so = FORMAL_ROOT / "unidrivevla" / so.relative_to(files["package"])
    _assert_physical_isolated(so, formal_so)
    formal_so_hash = _sha256(formal_so)
    if _sha256(so) != formal_so_hash:
        raise RuntimeError("the snapshot DAF binary differs from the precompiled formal server binary")
    helper = root / "nocudnn_train.py"
    formal_helper = FORMAL_ROOT / "setup/nocudnn_train.py"
    _assert_physical_isolated(helper, formal_helper)
    formal_helper_hash = _sha256(formal_helper)
    if _sha256(helper) != formal_helper_hash:
        raise RuntimeError("the owned cuDNN verification wrapper differs from the reviewed server helper")
    source_hashes = {}
    for relative, expected in DAF_SOURCE_SHA256.items():
        path = files["package"] / relative
        if _sha256(path) != expected:
            raise RuntimeError(f"DAF source differs from the compiled extension source: {relative}")
        source_hashes[relative] = expected
    _setup_python_path(root, "1.1")
    import torch
    from unidrivevla.models import ops

    if sys.version_info[:2] != (3, 9) or not torch.__version__.startswith("2.6."):
        raise RuntimeError(f"wrong training runtime: Python {sys.version}, torch {torch.__version__}")
    if torch.version.cuda != "12.4":
        raise RuntimeError(f"expected PyTorch CUDA 12.4 build, got {torch.version.cuda}")
    if not hasattr(ops, "deformable_aggregation_function"):
        raise RuntimeError("the copied precompiled DAF extension did not load")
    source_cases = [
        ("nuscenes", "1.1", ["data=nuscenes", "nuscenes.occ_root=null", "agent.vae_path=null",
                              f"profile.anchors_dir={nusc_anchor}"]),
        ("b2d", "1.1", ["data=b2d", f"profile.anchors_dir={b2d_anchor}"]),
        ("navsim11", "1.1", ["data=navsim", "train_test_split=navtrain",
                               f"cache_path={NAVSIM_CACHE['1.1']}", f'train_logs=["{NAVSIM_LOG}"]',
                               f'val_logs=["{NAVSIM_LOG}"]']),
        ("navsim20", "2.0", ["data=navsim", "train_test_split=navtrain",
                               f"cache_path={NAVSIM_CACHE['2.0']}", f'train_logs=["{NAVSIM_LOG}"]',
                               f'val_logs=["{NAVSIM_LOG}"]']),
    ]
    source_results = []
    for case, tree, overrides in source_cases:
        env = _expected_environment(root, tree)
        command = [str(PYTHON), str(files["package"] / "tools/check_sources.py"), *overrides,
                   "--split", "train", "--samples", "1"]
        run = subprocess.run(command, cwd=str(files["code"]), env=env, capture_output=True,
                             text=True, timeout=240)
        source_results.append({"case": case, "exit_code": int(run.returncode),
                               "stdout": run.stdout[-6000:], "stderr": run.stderr[-3000:]})
    supervision_env = _expected_environment(root, "1.1")
    supervision_env["CUDA_VISIBLE_DEVICES"] = ""
    supervision = subprocess.run([str(PYTHON), str(files["package"] / "tools/check_supervision.py")],
                                 cwd=str(files["code"]), env=supervision_env, capture_output=True,
                                 text=True, timeout=240)
    source_passed = (all(item["exit_code"] == 0 for item in source_results) and
                     supervision.returncode == 0)
    return {"python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_device_count_probe": "not run during CPU preflight",
            "daf_so": str(so), "daf_so_sha256": _sha256(so), "formal_daf_so_sha256": formal_so_hash,
            "nocudnn_helper": str(helper), "nocudnn_helper_sha256": _sha256(helper),
            "formal_nocudnn_helper_sha256": formal_helper_hash,
            "daf_source_sha256": source_hashes, "native_development_anchors": anchor_report,
            "real_source_checks": source_results,
            "vqa_image": str(vqa_image), "vqa_image_sha256": _sha256(vqa_image),
            "vqa_jsonl": str(vqa_jsonl), "vqa_jsonl_sha256": _sha256(vqa_jsonl),
            "check_supervision": {"exit_code": int(supervision.returncode),
                                  "stdout": supervision.stdout[-6000:],
                                  "stderr": supervision.stderr[-3000:]},
            "multiprocessing_resource_sharer_smoke": multiprocessing_smoke,
            "real_source_checks_passed": source_passed,
            "data_paths": {"nuscenes_infos": str(NUSC_INFOS), "b2d_infos": str(B2D_INFOS),
                           "navsim_cache_1.1": str(NAVSIM_CACHE["1.1"]),
                           "navsim_cache_2.0": str(NAVSIM_CACHE["2.0"]),
                           "navsim_devkit_1.1": str(NAVSIM_DEVKIT["1.1"]),
                           "navsim_devkit_2.0": str(NAVSIM_DEVKIT["2.0"])}}


def _prepare_vqa_fixture(root: Path, image: Path, destination: Path):
    if not image.is_file():
        raise FileNotFoundError(image)
    resolved = image.resolve()
    if not str(resolved).startswith(str(NUSC_DATA.resolve()) + os.sep):
        raise ValueError("verification VQA fixture uses a nuScenes mini image only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite verification fixture: {destination}")
    rows = []
    for index in range(18):
        rows.append({"image": str(resolved), "conversations": [
            {"from": "human", "value": "<image>\nName one broad category of object that can occur on a road."},
            {"from": "gpt", "value": "A vehicle."},
        ], "_verification_only": {"synthetic": True, "sample": index,
                                  "label": "not ground-truth; verifies the real Qwen image+AR training path"}})
    with destination.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"prepared verification-only VQA fixture: {destination} ({len(rows)} synthetic rows; image={resolved})")


def _plan(root: str = None, native: str = None, vqa_jsonl: str = None):
    self_check = {"paths": _self_check_plan(), "receipts": _self_check_receipt_contracts()}
    return {
        "intent": "short training and forward/backward closure checks; never full training",
        "gpu_start_policy": {"all_four_cards_must_be_idle": True, "free_mib_each_min": 23000,
                             "util_percent_each_below": 5, "compute_processes": 0,
                             "recheck_before_every_case": True},
        "single_profile_cases": [
            {"stage1": stage1, "stage2": stage2, "tree": tree, "optimizer_steps_each": 3,
             "stage2_initialization": "fresh complete stage1 checkpoint"}
            for stage1, stage2, tree in (("nuscenes_s1", "nuscenes_s2", "mini"),
                                         ("b2d_s1", "b2d_s2", "mini"),
                                         ("navsim11_s1", "navsim11_s2", "1.1"),
                                         ("navsim20_s1", "navsim20_s2", "2.0"))
        ],
        "prefix_case": "check_frozen_prefix.py using the fresh nuScenes stage2 checkpoint; same-checkpoint joint/cache loss+gradient equivalence only",
        "multi_vqa_case": {"steps_max": 18, "profile_heads": ["nuscenes", "b2d", "navsim"],
                           "torchrun_world_size": 4, "microbatch_per_rank": 1,
                           "derived_global_batch": 4, "single_profile_world_size": 1,
                           "vqa": "combined; 18 verification-only synthetic QA rows over one real nuScenes-mini camera image",
                           "must_observe_forward_backward_for_all_three_profiles": True},
        "profile_and_limits": {"single_stage_steps": "1 warmup + 2 measured optimizer steps",
                               "single_profile_microbatch": 1, "single_profile_global_batch": 1,
                               "multi_profile_global_batch": 4,
                               "fixed_native_prefix_caps": {"nuscenes": 3740, "b2d": 3670},
                               "full_heads": True, "no_head_or_prefix_cap_shrink": True,
                               "nusc_occ3d_vae": "disabled and reported absent in the available smoke inputs",
                               "navsim_stage2": "stage2_max_epochs=1 smoke-only override; not an official NAVSIM recipe",
                               "main_full_network_same_weight_loss_gradient_oracle": "still pending; frozen-prefix result is not that oracle"},
        "required_assets": {"package_snapshot": "final current source copied physically under this bundle/code",
                            "daf_source_sha256": DAF_SOURCE_SHA256,
                            "qwen2b": VLM.as_posix(), "nuscenes_mini": NUSC_INFOS.as_posix(),
                            "b2d_smoke": B2D_INFOS.as_posix(),
                            "navsim_devkits": {key: value.as_posix() for key, value in NAVSIM_DEVKIT.items()},
                            "navsim_cache": {key: value.as_posix() for key, value in NAVSIM_CACHE.items()},
                            "native_development_anchors": native, "synthetic_vqa_jsonl": vqa_jsonl,
                            "new_output_root": str(root).rstrip("/") + "/exp" if root else None},
        "files_owned_here": [str(Path(__file__).resolve()), "bundle code snapshot", "unique logs", "fresh checkpoints",
                             "one suite lock and per-case exit receipts"],
        "static_self_check": self_check,
    }


def _self_check_plan():
    suite = "gpu_plan_selfcheck"
    output_names = [f"{suite}_{case}" for case in SINGLE_CASES]
    output_names.append(f"{suite}_multidomain_vqa")
    if len(output_names) != len(set(output_names)):
        raise AssertionError("every stage/data case must receive its own experiment_name")
    expected = {
        "nuscenes_s2": _stage1_checkpoint(Path("/owned"), suite, "nuscenes", "1.1"),
        "b2d_s2": _stage1_checkpoint(Path("/owned"), suite, "b2d", "1.1"),
        "navsim11_s2": _stage1_checkpoint(Path("/owned"), suite, "navsim", "1.1"),
        "navsim20_s2": _stage1_checkpoint(Path("/owned"), suite, "navsim", "2.0"),
    }
    exact = {
        "nuscenes_s2": Path("/owned/exp/gpu_plan_selfcheck_nuscenes_s1/last.ckpt"),
        "b2d_s2": Path("/owned/exp/gpu_plan_selfcheck_b2d_s1/last.ckpt"),
        "navsim11_s2": Path("/owned/exp/gpu_plan_selfcheck_navsim11_s1/last.ckpt"),
        "navsim20_s2": Path("/owned/exp/gpu_plan_selfcheck_navsim20_s1/last.ckpt"),
    }
    if expected != exact:
        raise AssertionError(f"stage2 checkpoint provenance is mismatched: {expected}")
    return {"passed": True, "unique_experiment_names": len(output_names),
            "stage2_dependencies": {name: str(path) for name, path in expected.items()},
            "gpu_gate_cases": list(GPU_CASES)}


def _case_result_errors(case: str, receipt: dict):
    errors = []
    if receipt.get("status") != "passed" or receipt.get("exit_code") != 0:
        errors.append("receipt status/exit_code is not passed/0")
    if case == "frozen_prefix":
        if not receipt.get("log") or not receipt.get("stage2_checkpoint"):
            errors.append("prefix receipt lacks its actual log or stage2 checkpoint")
        if "not a main full-network same-weight oracle" not in receipt.get("interpretation", ""):
            errors.append("prefix oracle scope is not labelled")
        return errors
    steps = 18 if case == "multidomain_vqa" else 3
    if receipt.get("global_step") != steps or receipt.get("train_max_steps") != steps:
        errors.append("optimizer step count is incomplete")
    if not receipt.get("checkpoint_exists"):
        errors.append("fresh last.ckpt was not saved")
    if not receipt.get("parameter_counts", {}).get("qwen_prefix", {}).get("total", 0):
        errors.append("Qwen prefix parameter counts are absent")
    if not receipt.get("validation_profiles") or not receipt.get("validation_loss_finite"):
        errors.append("one-batch production inference/compute_loss did not pass")
    step_profiles = receipt.get("optimizer_step_profiles", [])
    if len(step_profiles) != steps:
        errors.append("per-step timing/profile rows are incomplete")
    elif any(not item.get("grad_norm_finite", False) for item in step_profiles):
        errors.append("a step reported a non-finite gradient norm")
    if receipt.get("nonfinite_loss_count", 0):
        errors.append("a training loss was non-finite")
    if receipt.get("nonfinite_vqa_gradient_profiles"):
        errors.append("a Qwen AR gradient was non-finite")
    if case == "multidomain_vqa":
        expected = {"nuscenes", "b2d", "navsim"}
        if set(receipt.get("profile_forwards", {})) != expected:
            errors.append("not all three planning profiles forwarded")
        if not expected.issubset(receipt.get("profile_gradient_hits", [])):
            errors.append("not all three planning heads received gradients")
        if receipt.get("vqa_batches") != steps or not expected.issubset(receipt.get("vqa_forward_profiles", [])):
            errors.append("the combined AR branch did not run for all profiles")
        if not expected.issubset(receipt.get("vqa_gradient_profiles", [])):
            errors.append("the Qwen AR branch did not receive gradients for all profiles")
        if receipt.get("world_size") != 4 or receipt.get("effective_global_batch") != 4:
            errors.append("multi-profile VQA did not use four DDP ranks at batch one per rank")
    else:
        if receipt.get("world_size") != 1 or receipt.get("effective_global_batch") != 1:
            errors.append("single-profile diagnostic did not use one rank at microbatch one")
        if case in ("nuscenes_s1", "nuscenes_s2", "b2d_s1", "b2d_s2"):
            if receipt.get("optimizer_lr_interval") != "micro":
                errors.append("native single-profile schedule did not advance at micro-step cadence")
    return errors


def _validate_rank_receipts(case: str, receipts, expected_world: int):
    errors = []
    if len(receipts) != expected_world:
        return False, [f"expected {expected_world} rank receipts, got {len(receipts)}"]
    by_rank = {int(item.get("rank", -1)): item for item in receipts}
    if set(by_rank) != set(range(expected_world)):
        errors.append(f"rank receipts do not cover [0,{expected_world}): {sorted(by_rank)}")
    for rank, item in by_rank.items():
        if int(item.get("world_size", -1)) != expected_world:
            errors.append(f"rank {rank} reports world_size={item.get('world_size')}")
        errors.extend([f"rank {rank}: {error}" for error in _case_result_errors(case, item)])
    if case == "multidomain_vqa" and by_rank:
        reference = by_rank[min(by_rank)]
        for rank, item in by_rank.items():
            for field in ("profile_forwards", "profile_gradient_hits", "vqa_forward_profiles",
                          "vqa_gradient_profiles", "nonfinite_vqa_gradient_profiles", "global_step",
                          "vqa_batches", "effective_global_batch"):
                if item.get(field) != reference.get(field):
                    errors.append(f"rank {rank} differs from rank {min(by_rank)} for {field}")
    return not errors, errors


def _self_check_receipt_contracts():
    def base(case, rank=0, world=1):
        steps = 18 if case == "multidomain_vqa" else 3
        return {"case": case, "status": "passed", "exit_code": 0, "rank": rank,
                "world_size": world, "global_step": steps, "train_max_steps": steps,
                "checkpoint_exists": True,
                "parameter_counts": {"qwen_prefix": {"total": 100, "trainable": 10, "frozen": 90}},
                "validation_profiles": [{"public_finite": True}], "validation_loss_finite": True,
                "optimizer_step_profiles": [{"grad_norm_finite": True} for _ in range(steps)],
                "nonfinite_loss_count": 0, "nonfinite_vqa_gradient_profiles": [],
                "profile_forwards": {}, "profile_gradient_hits": [], "vqa_batches": 0,
                "vqa_forward_profiles": [], "vqa_gradient_profiles": [], "effective_global_batch": 1,
                "optimizer_lr_interval": "micro"}

    for case in SINGLE_CASES:
        okay, errors = _validate_rank_receipts(case, [base(case)], 1)
        if not okay:
            raise AssertionError(f"valid {case} receipt rejected: {errors}")
    prefix = {"status": "passed", "exit_code": 0, "rank": 0, "world_size": 1,
              "log": "/owned/logs/prefix.log", "stage2_checkpoint": "/owned/exp/stage2/last.ckpt",
              "interpretation": "same-checkpoint result; not a main full-network same-weight oracle"}
    if not _validate_rank_receipts("frozen_prefix", [prefix], 1)[0]:
        raise AssertionError("valid frozen-prefix receipt rejected")
    incomplete = base("nuscenes_s1")
    incomplete["global_step"] = 2
    okay, errors = _validate_rank_receipts("nuscenes_s1", [incomplete], 1)
    if okay or not any("optimizer step count" in error for error in errors):
        raise AssertionError(f"receipt self-check failed to reject an incomplete single case: {errors}")

    names = ["nuscenes", "b2d", "navsim"]
    multi = []
    for rank in range(4):
        item = base("multidomain_vqa", rank, 4)
        item.update({"profile_forwards": {name: 6 for name in names},
                     "profile_gradient_hits": names,
                     "vqa_batches": 18, "vqa_forward_profiles": names,
                     "vqa_gradient_profiles": names, "effective_global_batch": 4,
                     "optimizer_lr_interval": "step"})
        multi.append(item)
    if not _validate_rank_receipts("multidomain_vqa", multi, 4)[0]:
        raise AssertionError("valid four-rank mixed VQA receipts rejected")
    for name, mutate, expected_error in (
        ("missing rank", lambda values: values[:-1], "expected 4 rank receipts"),
        ("rank nonzero exit", lambda values: [{**values[0], "exit_code": 7}, *values[1:]], "status/exit_code"),
        ("rank profile disagreement", lambda values: [{**values[0]}, {**values[1], "profile_forwards": {"nuscenes": 18}}, *values[2:]], "differs"),
        ("missing head gradient", lambda values: [{**values[0], "profile_gradient_hits": ["nuscenes", "b2d"]}, *values[1:]], "received gradients"),
        ("non-finite training", lambda values: [{**values[0], "nonfinite_loss_count": 1}, *values[1:]], "training loss"),
    ):
        okay, errors = _validate_rank_receipts("multidomain_vqa", mutate(multi), 4)
        if okay or not any(expected_error in error for error in errors):
            raise AssertionError(f"receipt self-check failed to reject {name}: {errors}")
    return {"passed": True, "single_profile_stage_cases": len(SINGLE_CASES),
            "prefix": True, "multi_world_size": 4, "negative_cases": 6}


def _compose_cases(root: Path, native_root: Path, vqa_jsonl: Path, suite_id: str):
    result = []
    for case in (*SINGLE_CASES.keys(), "multidomain_vqa"):
        tree = SINGLE_CASES.get(case, ("multidomain", "1.1", "stage1"))[1]
        env = _expected_environment(root, tree)
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                   "--native-anchors", str(native_root), "--vqa-jsonl", str(vqa_jsonl),
                   "--suite-id", suite_id, "compose-one", case]
        completed = subprocess.run(command, cwd=str(CODE_ROOT), env=env, capture_output=True,
                                   text=True, timeout=120)
        item = {"case_id": case, "tree": tree, "exit_code": int(completed.returncode)}
        if completed.returncode:
            item["stderr_tail"] = completed.stderr[-4000:]
            item["stdout_tail"] = completed.stdout[-1000:]
            result.append(item)
            continue
        item["composed"] = json.loads(completed.stdout.splitlines()[-1])
        result.append(item)
    return result


def _case_profile_counts(agent):
    from unidrivevla.adapters.multidomain import MultiProfilePolicy

    counts = {}
    if isinstance(agent, MultiProfilePolicy):
        heads = agent.heads
        shared = agent.shared_core
    else:
        heads = {str(getattr(agent.profile, "name", "single")): agent.head}
        shared = getattr(agent.head, "qwen3_vl_with_expert", None)
    prefix = getattr(shared, "qwen3_vl", None) if shared is not None else None
    params = list(agent.parameters())
    prefix_params = list(prefix.parameters()) if prefix is not None else []
    counts["model"] = {"total": sum(p.numel() for p in params),
                       "trainable": sum(p.numel() for p in params if p.requires_grad),
                       "frozen": sum(p.numel() for p in params if not p.requires_grad)}
    counts["qwen_prefix"] = {"module_name": ("shared_core.qwen3_vl" if isinstance(agent, MultiProfilePolicy)
                                               else "head.qwen3_vl_with_expert.qwen3_vl"),
                             "total": sum(p.numel() for p in prefix_params),
                             "trainable": sum(p.numel() for p in prefix_params if p.requires_grad),
                             "frozen": sum(p.numel() for p in prefix_params if not p.requires_grad)}
    counts["heads"] = {str(name): {"total": sum(p.numel() for p in head.parameters()),
                                  "trainable": sum(p.numel() for p in head.parameters() if p.requires_grad),
                                  "frozen": sum(p.numel() for p in head.parameters() if not p.requires_grad)}
                       for name, head in heads.items()}
    by_root = {}
    for name, parameter in agent.named_parameters():
        root_name = name.split(".", 1)[0]
        item = by_root.setdefault(root_name, {"total": 0, "trainable": 0, "frozen": 0})
        item["total"] += parameter.numel()
        item["trainable" if parameter.requires_grad else "frozen"] += parameter.numel()
    counts["named_parameter_roots"] = by_root
    return counts


class _OneBatchLoader:
    """A read-only proxy that keeps production validation code but yields one batch."""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        return itertools.islice(iter(self._loader), 1)

    def __len__(self):
        return min(1, len(self._loader))

    def __getattr__(self, name):
        return getattr(self._loader, name)


def _validate_inference_contract(agent, features, predictions, record):
    import torch

    if not isinstance(predictions, dict) or "pred_traj" not in predictions:
        raise RuntimeError("production inference forward must return public pred_traj")
    public = predictions["pred_traj"]
    if public.ndim != 3 or public.shape[0] < 1 or public.shape[-1] < 2:
        raise RuntimeError(f"bad public trajectory shape: {tuple(public.shape)}")
    if not torch.isfinite(public).all().item():
        raise RuntimeError("public pred_traj contains non-finite values")
    if hasattr(agent, "profile_for_source"):
        names = features.get("profile")
        name = str(names[0] if isinstance(names, (list, tuple)) else names)
        profile = agent.profile_for_source(name)
    else:
        profile = agent.profile
        name = str(profile.name)
    if public.shape[1] != int(profile.horizon):
        raise RuntimeError(f"public trajectory horizon mismatch for {name}: {public.shape[1]} != {profile.horizon}")
    item = {"profile": name, "public_shape": list(public.shape), "public_finite": True,
            "frame": str(profile.frame), "prefix_max_len": int(profile.prefix_max_len or 0)}
    if str(profile.frame) == "lidar":
        native = predictions.get("pred_traj_head")
        transform = features.get("native_to_ego")
        if native is None or transform is None:
            raise RuntimeError("native-frame evaluation must expose pred_traj_head and native_to_ego")
        expected = torch.einsum("bij,btj->bti", transform[:, :2, :2].to(native), native[..., :2])
        if not torch.allclose(public[..., :2], expected, atol=1e-4, rtol=1e-4):
            raise RuntimeError("public pred_traj is not the native prediction rotated into ego coordinates")
        item["native_head_shape"] = list(native.shape)
        item["public_ego_transform_max_abs"] = float((public[..., :2] - expected).abs().max().item())
    record["validation_profiles"].append(item)


def _install_observers(trainer, record):
    from types import MethodType

    agent = trainer.agent
    heads = getattr(agent, "heads", None)
    named_heads = list(heads.items()) if heads is not None else [("single", agent.head)]
    for name, head in named_heads:
        for parameter in head.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad, n=str(name): record["profile_gradient_hits"].add(n))
        method = getattr(head, "forward_ar_batch", None)
        if callable(method):
            original_ar = method

            def wrapped_ar(self, ar_batch, *args, _original=original_ar, _name=str(name), **kwargs):
                record["vqa_forward_profiles"].add(_name)
                result = _original(ar_batch, *args, **kwargs)
                loss = result.get("loss_ar") if isinstance(result, dict) else None
                if hasattr(loss, "requires_grad") and loss.requires_grad:
                    def observe_ar_gradient(grad, n=_name):
                        record["vqa_gradient_profiles"].add(n)
                        if not __import__("torch").isfinite(grad).all().item():
                            record["nonfinite_vqa_gradient_profiles"].add(n)
                        return grad
                    loss.register_hook(observe_ar_gradient)
                return result
            head.forward_ar_batch = MethodType(wrapped_ar, head)

    original_forward = agent.forward

    def wrapped_forward(self, features, targets=None, tokens=None):
        profile = features.get("profile", "single")
        if isinstance(profile, (list, tuple)):
            profile = str(profile[0]) if profile else "single"
        profile = str(profile)
        if self.training:
            record["profile_forwards"][profile] = record["profile_forwards"].get(profile, 0) + 1
            if "ar_batch" in features:
                record["vqa_batches"] += 1
        result = original_forward(features, targets, tokens)
        if self.training:
            loss = getattr(result, "loss", None)
            if loss is not None and hasattr(loss, "detach"):
                value = float(loss.detach().float().cpu().item())
                record["losses"].append(value)
                if not __import__("math").isfinite(value):
                    record["nonfinite_loss_count"] += 1
        else:
            _validate_inference_contract(self, features, result, record)
        return result
    agent.forward = MethodType(wrapped_forward, agent)

    original_initialize = agent.initialize
    if callable(original_initialize):
        def wrapped_initialize(self, *args, **kwargs):
            result = original_initialize(*args, **kwargs)
            record["parameter_counts"] = _case_profile_counts(agent)
            record["initialize_completed"] = True
            return result
        agent.initialize = MethodType(wrapped_initialize, agent)

    original_build_data = trainer.build_data

    def wrapped_build_data(self):
        result = original_build_data()
        record["effective_global_batch"] = int(trainer.global_batch)
        record["train_batches_per_epoch"] = len(trainer.train_loader)
        record["val_batches_per_rank"] = len(trainer.val_loader) if trainer.val_loader is not None else 0
        return result
    trainer.build_data = MethodType(wrapped_build_data, trainer)

    original_build_optimizer = trainer.build_optimizer

    def wrapped_build_optimizer(self):
        result = original_build_optimizer()
        record["optimizer_lr_interval"] = str(trainer.lr_interval)
        record["initial_lrs"] = [float(group["lr"]) for group in trainer.optimizer.param_groups]
        record["profile_lr_microstep"] = {name: value["lr_microstep"]
                                          for name, value in record["profile_contract"].items()}
        original_step = trainer.optimizer.step

        def observed_optimizer_step(*args, **kwargs):
            record["optimizer_lrs_used"].append({
                "update": int(trainer.global_step) + 1,
                "lr": [float(group["lr"]) for group in trainer.optimizer.param_groups],
            })
            return original_step(*args, **kwargs)
        trainer.optimizer.step = observed_optimizer_step
        return result
    trainer.build_optimizer = MethodType(wrapped_build_optimizer, trainer)

    original_log_train = trainer.log_train

    def wrapped_log_train(self, epoch, out, loss, grad_norm, window, batches):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = float(window["time"])
        start_wait = float(window["wait"])
        result = original_log_train(epoch, out, loss, grad_norm, window, batches)
        elapsed = max(0.0, (float(window["time"]) - start_time) * 1000.0)
        waited = max(0.0, (float(window["wait"]) - start_wait) * 1000.0)
        step = int(trainer.global_step)
        peak = None
        try:
            import torch
            if torch.cuda.is_available():
                peak = round(torch.cuda.max_memory_allocated() / (2 ** 30), 3)
        except Exception:
            pass
        record["optimizer_step_profiles"].append({
            "step": step, "phase": "warmup" if step == 1 else "measured",
            "wall_ms": round(elapsed, 3), "data_wait_ms": round(waited, 3),
            "lr_after_scheduler": [float(group["lr"]) for group in trainer.optimizer.param_groups],
            "peak_memory_gib": peak,
            "grad_norm_finite": (True if grad_norm is None else
                                 bool(torch.isfinite(grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor")
                                                     else grad_norm).all().item())),
        })
        record["whole_case_peak_memory_gib"] = max(record.get("whole_case_peak_memory_gib") or 0.0, peak or 0.0)
        if step == 1 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return result
    trainer.log_train = MethodType(wrapped_log_train, trainer)

    original_validate = trainer.validate

    def one_batch_validate(self):
        if trainer.val_loader is None:
            raise RuntimeError("inference smoke requires an evaluation DataLoader")
        original_loader = trainer.val_loader
        trainer.val_loader = _OneBatchLoader(original_loader)
        record["validation_batches_requested"] = 1
        try:
            value = original_validate()
            record["validation_loss"] = float(value)
            record["validation_loss_finite"] = bool(__import__("math").isfinite(float(value)))
            if not record["validation_loss_finite"] or not record["validation_profiles"]:
                raise RuntimeError("one-batch production validation did not return a finite score/trajectory")
            return value
        finally:
            trainer.val_loader = original_loader
    trainer.validate = MethodType(one_batch_validate, trainer)


def _compose_runtime(root: Path, tree: str, overrides):
    _setup_python_path(root, tree)
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CODE_ROOT / "unidrivevla/config"), version_base=None):
        return compose("train", overrides=overrides)


def _execute_train_case(root: Path, native_root: Path, vqa_jsonl: Path, suite_id: str, case: str):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world = 4 if case == "multidomain_vqa" else 1
    if world != expected_world:
        raise RuntimeError(f"{case} requires WORLD_SIZE={expected_world}, got {world}")
    evidence = _bundle_files(root)["evidence"]
    receipt = evidence / "cases" / f"{suite_id}_{case}_rank{rank:04d}.json"
    if receipt.exists():
        raise FileExistsError(f"case receipt already exists; observe it and do not repeat: {receipt}")
    details = {
        "suite_id": suite_id, "case": case, "status": "starting", "exit_code": None,
        "started_unix": time.time(), "rank": rank, "world_size": world,
        "code_root": str(CODE_ROOT), "microbatch": 1,
        "train_max_steps": 18 if case == "multidomain_vqa" else 3,
        "requested_global_batch": 0, "effective_global_batch": None,
        "profile_contract": {}, "profile_forwards": {}, "profile_gradient_hits": set(),
        "validation_profiles": [], "validation_batches_requested": 0,
        "vqa_forward_profiles": set(), "vqa_gradient_profiles": set(), "vqa_batches": 0,
        "nonfinite_vqa_gradient_profiles": set(), "nonfinite_loss_count": 0,
        "losses": [], "optimizer_step_profiles": [], "optimizer_lrs_used": [],
        "limits": _plan(str(root), str(native_root), str(vqa_jsonl))["profile_and_limits"],
    }
    evidence_cases = receipt.parent
    evidence_cases.mkdir(parents=True, exist_ok=True)
    _json(receipt, {key: (sorted(value) if isinstance(value, set) else value)
                    for key, value in details.items()}, exclusive=True)
    started = time.perf_counter()
    trainer = None
    run_id = f"{suite_id}_{case}"
    try:
        tree, overrides = _case_overrides(case, root, native_root, vqa_jsonl, suite_id)
        if case.endswith("_s2"):
            name = SINGLE_CASES[case][0]
            tree = SINGLE_CASES[case][1]
            checkpoint = _stage1_checkpoint(root, suite_id, name, tree)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"stage2 requires the complete fresh stage1 checkpoint: {checkpoint}")
            if checkpoint.stat().st_size < 1024 * 1024:
                raise RuntimeError(f"stage1 checkpoint is unexpectedly small: {checkpoint.stat().st_size}")
        if case == "multidomain_vqa" and not vqa_jsonl.is_file():
            raise FileNotFoundError(vqa_jsonl)
        name = "multidomain" if case == "multidomain_vqa" else SINGLE_CASES[case][0]
        stage = "stage1" if case == "multidomain_vqa" else SINGLE_CASES[case][2]
        cfg = _compose_runtime(root, tree, overrides)
        _validate_recipe(name, stage, cfg)
        details["profile_contract"] = _profile_summary(cfg)
        details["experiment_name"] = run_id
        details["stage1_checkpoint"] = (str(_stage1_checkpoint(root, suite_id, name, tree))
                                        if case.endswith("_s2") else None)
        details["native_anchor_note"] = "smoke-only development anchors; not official/full-data anchors"
        details["vqa_fixture_note"] = ("synthetic image/QA labels; real image exercises Qwen AR path, no VQA accuracy claim"
                                        if case == "multidomain_vqa" else None)

        # Multi-GPU ranks verify the suite's single parent gate and meet at a filesystem barrier before CUDA init.
        if world > 1:
            parent_suite = os.environ.get("UNI_GPU_CLOSURE_PARENT_SUITE")
            parent_case = os.environ.get("UNI_GPU_CLOSURE_PARENT_CASE")
            gate_record = _bundle_files(root)["evidence"] / f"{suite_id}_{case}.gpu_gate.json"
            lock = _bundle_files(root)["evidence"] / f"{suite_id}.lock"
            if parent_suite != suite_id or parent_case != case or not lock.is_file() or not gate_record.is_file():
                raise RuntimeError("multi-rank case is only allowed through its guarded parent suite")
            gate = json.loads(gate_record.read_text(encoding="utf-8"))
            if not gate.get("ready"):
                raise RuntimeError("parent GPU gate record does not authorize this case")
            ready_dir = _bundle_files(root)["evidence"] / f"{suite_id}_{case}.ranks_ready"
            ready_dir.mkdir(parents=True, exist_ok=True)
            ready = ready_dir / f"rank{rank:04d}.ready"
            with ready.open("x", encoding="ascii") as stream:
                stream.write(str(time.time()))
            wait_until = time.monotonic() + 180
            while time.monotonic() < wait_until:
                if all((ready_dir / f"rank{peer:04d}.ready").is_file() for peer in range(world)):
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError("not all four ranks reached the pre-CUDA readiness barrier")
            details["parent_gpu_gate"] = gate
            details["rank_start_barrier"] = str(ready_dir)

        from omegaconf import OmegaConf
        from unidrivevla.run_paths import prepare_run_paths
        from unidrivevla.train.trainer import Trainer

        os.environ["UNIDRIVE_RUN_ID"] = run_id
        if world > 1 and not os.environ.get("UNIDRIVE_INVOCATION_ID"):
            raise RuntimeError("torchrun suite did not provide the shared fresh invocation identity")
        output_dir, _ = prepare_run_paths("train", overrides)
        details["output_dir"] = str(output_dir)
        details["run_paths_claimed"] = True
        # For one-rank cases, this is the final all-card check just before Trainer initializes CUDA.
        if world == 1:
            gate = _gpu_inventory()
            if not gate["ready"]:
                details["status"] = "blocked_gpu_gate_before_trainer"
                details["gpu_gate_before_trainer"] = gate
                details["exit_code"] = 3
                _atomic_json(receipt, details)
                return 3
            details["gpu_gate_before_trainer"] = gate
        provenance = output_dir / "code/hydra" / os.environ["UNIDRIVE_INVOCATION_ID"]
        provenance.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            OmegaConf.save(cfg, provenance / "resolved.yaml", resolve=True)
        trainer = Trainer(cfg)
        _install_observers(trainer, details)
        details["status"] = "running"
        trainer.fit()
        details["status"] = "passed" if trainer.global_step == details["train_max_steps"] else "failed_step_count"
        details["exit_code"] = 0 if details["status"] == "passed" else 1
        details["global_step"] = int(trainer.global_step)
        details["output_last_checkpoint"] = str(output_dir / "last.ckpt")
        details["checkpoint_exists"] = (output_dir / "last.ckpt").is_file()
        details["checkpoint_bytes"] = (output_dir / "last.ckpt").stat().st_size if details["checkpoint_exists"] else None
        details["global_batch"] = details.get("effective_global_batch")
        if len(details["optimizer_step_profiles"]) != details["train_max_steps"]:
            details["status"] = "failed_step_profile_count"
            details["exit_code"] = 1
            raise RuntimeError("Trainer did not emit one timing record per optimizer step")
        if details["nonfinite_loss_count"] or any(not item["grad_norm_finite"]
                                                  for item in details["optimizer_step_profiles"]):
            details["status"] = "failed_nonfinite_training"
            details["exit_code"] = 1
            raise RuntimeError("training loss or gradient norm became non-finite")
        if not details["validation_profiles"] or not details.get("validation_loss_finite"):
            details["status"] = "failed_validation_smoke"
            details["exit_code"] = 1
            raise RuntimeError("the one-batch production inference/compute_loss smoke did not pass")
        if details["status"] != "passed" or not details.get("checkpoint_exists"):
            details["status"] = "failed_checkpoint_or_steps"
            details["exit_code"] = 1
            raise RuntimeError("smoke did not complete every optimizer step and save last.ckpt")
        if case == "multidomain_vqa":
            expected = {"nuscenes", "b2d", "navsim"}
            observed = set(details["profile_forwards"])
            gradient = set(details["profile_gradient_hits"])
            if observed != expected or not expected.issubset(gradient):
                details["status"] = "failed_profile_coverage"
                details["exit_code"] = 1
                raise RuntimeError(f"multidomain coverage incomplete: forward={sorted(observed)}, grad={sorted(gradient)}")
            if details["vqa_batches"] != details["global_step"] or not expected.issubset(details["vqa_forward_profiles"]):
                details["status"] = "failed_vqa_forward_coverage"
                details["exit_code"] = 1
                raise RuntimeError("combined VQA did not execute alongside every profile as planned")
            if not expected.issubset(details["vqa_gradient_profiles"]):
                details["status"] = "failed_vqa_gradient_coverage"
                details["exit_code"] = 1
                raise RuntimeError(f"Qwen AR gradient coverage incomplete: {details['vqa_gradient_profiles']}")
            if details["nonfinite_vqa_gradient_profiles"]:
                details["status"] = "failed_nonfinite_vqa_gradient"
                details["exit_code"] = 1
                raise RuntimeError(f"Qwen AR gradients were non-finite for: {details['nonfinite_vqa_gradient_profiles']}")
        contract_errors = _case_result_errors(case, details)
        if contract_errors:
            details["status"] = "failed_result_contract"
            details["exit_code"] = 1
            raise RuntimeError("; ".join(contract_errors))
    except BaseException as exc:
        if details.get("exit_code") is None:
            details["exit_code"] = 1
        if details.get("status") in ("starting", "running"):
            details["status"] = "failed"
        details["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        details["wall_seconds"] = time.perf_counter() - started
        for key in ("profile_gradient_hits", "vqa_forward_profiles", "vqa_gradient_profiles",
                    "nonfinite_vqa_gradient_profiles"):
            details[key] = sorted(details[key])
        details["measured_train_peak_memory_gib"] = max(
            (item["peak_memory_gib"] or 0.0 for item in details["optimizer_step_profiles"]
             if item["phase"] == "measured"), default=None)
        details["whole_case_peak_memory_gib"] = max(
            [details.get("whole_case_peak_memory_gib") or 0.0] +
            [item["peak_memory_gib"] or 0.0 for item in details["optimizer_step_profiles"]]) or None
        try:
            import torch
            if trainer is not None and torch.cuda.is_available():
                details["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / (2 ** 30), 3)
                details["whole_case_peak_memory_gib"] = max(details["whole_case_peak_memory_gib"] or 0.0,
                                                             details["peak_memory_gib"])
        except Exception:
            details["peak_memory_gib"] = None
        _atomic_json(receipt, details)
    print(json.dumps(details, default=str, sort_keys=True))


def _execute_prefix_case(root: Path, native_root: Path, suite_id: str):
    nusc_anchor, _ = _anchor_paths(str(native_root))
    checkpoint = root / "exp" / f"{suite_id}_nuscenes_s2" / "last.ckpt"
    receipt = _bundle_files(root)["evidence"] / "cases" / f"{suite_id}_frozen_prefix_rank0000.json"
    if receipt.exists():
        raise FileExistsError(f"prefix case already has a receipt; observe it and do not repeat: {receipt}")
    started = time.time()
    _json(receipt, {"suite_id": suite_id, "case": "frozen_prefix", "rank": 0,
                    "world_size": 1, "status": "starting", "exit_code": None,
                    "stage2_checkpoint": str(checkpoint)}, exclusive=True)
    tree = "1.1"
    name = f"{suite_id}_frozen_prefix"
    log = _bundle_files(root)["logs"] / f"{name}.log"
    result_code = 1
    detail = {"log": str(log), "stage2_checkpoint": str(checkpoint),
              "interpretation": "same-stage2-checkpoint joint versus frozen-prefix loss/gradient oracle only; not a main full-network same-weight oracle"}
    try:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"prefix oracle requires fresh stage2 checkpoint: {checkpoint}")
        if log.exists():
            raise FileExistsError(log)
        env = _expected_environment(root, tree)
        os.environ.update(env)
        _setup_python_path(root, tree)
        _validate_recipe("nuscenes", "stage2", _compose_runtime(root, tree, [
            "data=nuscenes", "stage=stage2", f"agent.checkpoint_path={checkpoint}",
            "nuscenes.occ_root=null", "agent.vae_path=null", f"profile.anchors_dir={nusc_anchor}",
            "train.skip_validation=true", "evaluator.every_n_epochs=0", "evaluator.on_end=false",
        ]))
        gate = _gpu_inventory()
        if not gate["ready"]:
            raise RuntimeError("GPU gate closed immediately before frozen-prefix oracle: " + json.dumps(gate))
        experiment = f"{suite_id}_prefix_oracle"
        overrides = ["data=nuscenes", "stage=stage2", f"agent.checkpoint_path={checkpoint}",
                     "nuscenes.occ_root=null", "agent.vae_path=null", f"profile.anchors_dir={nusc_anchor}",
                     "train.skip_validation=true", "evaluator.every_n_epochs=0", "evaluator.on_end=false",
                     "dataloader.params.batch_size=1", "dataloader.params.num_workers=0",
                     "dataloader.params.pin_memory=false", f"experiment_name={experiment}"]
        env.update({"CUDA_VISIBLE_DEVICES": "0", "NOCUDNN_ALL": "0", "LD_DEBUG": "libs",
                    "LD_DEBUG_OUTPUT": str(_bundle_files(root)["logs"] / f"{name}.lddebug")})
        script = CODE_ROOT / "unidrivevla/tools/check_frozen_prefix.py"
        helper = root / "nocudnn_train.py"
        command = [str(PYTHON), str(helper), str(script), *overrides]
        _bundle_files(root)["logs"].mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=str(CODE_ROOT), env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, text=True)
        result_code = int(result.returncode)
        detail["gpu_gate"] = gate
        if result_code:
            raise subprocess.CalledProcessError(result_code, command)
    except BaseException as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        detail["status"] = "failed"
        detail["exit_code"] = int(getattr(exc, "returncode", 1) or 1)
        raise
    finally:
        detail["status"] = "passed" if result_code == 0 else detail.get("status", "failed")
        detail["exit_code"] = result_code
        detail["ended_unix"] = time.time()
        detail["wall_seconds"] = time.time() - started
        current = json.loads(receipt.read_text(encoding="utf-8"))
        current.update(detail)
        _atomic_json(receipt, current)
    print(json.dumps(json.loads(receipt.read_text(encoding="utf-8")), sort_keys=True))


def _suite(root: Path, native_root: Path, vqa_image: Path, vqa_jsonl: Path, suite_id: str, execute: bool):
    if not execute:
        raise ValueError("GPU cases require explicit --execute")
    if not RUN_ID.fullmatch(suite_id):
        raise ValueError("--suite-id has unsafe characters")
    # Do not claim a suite while another user's job makes the launch gate false.
    initial_gate = _gpu_inventory()
    if not initial_gate["ready"]:
        print(json.dumps({"suite_id": suite_id, "status": "blocked_initial_gpu_gate",
                          "gpu_gate": initial_gate, "no_case_started": True}, sort_keys=True))
        return 3
    evidence = _bundle_files(root)["evidence"]
    evidence.mkdir(parents=True, exist_ok=True)
    lock = evidence / f"{suite_id}.lock"
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(f"pid={os.getpid()}\nstarted={time.time()}\n")
    state_path = evidence / f"{suite_id}.suite.json"
    _json(state_path, {"suite_id": suite_id, "status": "running", "started_unix": time.time(),
                       "cases": [], "rerun_policy": "never rerun an existing lock or case receipt"}, exclusive=True)
    cases = ["nuscenes_s1", "nuscenes_s2", "frozen_prefix", "b2d_s1", "b2d_s2",
             "navsim11_s1", "navsim11_s2", "navsim20_s1", "navsim20_s2", "multidomain_vqa"]
    if not vqa_jsonl.is_file():
        _prepare_vqa_fixture(root, vqa_image, vqa_jsonl)
    suite = json.loads(state_path.read_text(encoding="utf-8"))
    try:
        for case in cases:
            gate = _gpu_inventory()
            if not gate["ready"]:
                suite["status"] = "waiting_gpu_gate"
                suite["waiting_for_case"] = case
                suite["gate_before_wait"] = gate
                _atomic_json(state_path, suite)
                query_error = gate.get("error")
                repeated_errors = 1 if query_error else 0
                while not gate["ready"]:
                    time.sleep(30)
                    gate = _gpu_inventory()
                    if gate.get("error"):
                        repeated_errors = repeated_errors + 1 if gate["error"] == query_error else 1
                        query_error = gate["error"]
                        if repeated_errors >= 3:
                            suite["status"] = "failed_gpu_gate_query"
                            suite["gpu_gate_error"] = gate["error"]
                            _atomic_json(state_path, suite)
                            return 2
                    else:
                        repeated_errors = 0
                        query_error = None
                suite["status"] = "running"
                suite.pop("waiting_for_case", None)
                suite["last_gate_ready_unix"] = time.time()
                _atomic_json(state_path, suite)
            log = _bundle_files(root)["logs"] / f"{suite_id}_{case}.log"
            case_receipts = list((evidence / "cases").glob(f"{suite_id}_{case}_rank*.json"))
            if log.exists() or case_receipts:
                raise FileExistsError(f"case already started; observe its log/receipt and do not repeat: {case}")
            _bundle_files(root)["logs"].mkdir(parents=True, exist_ok=True)
            gate_record_path = evidence / f"{suite_id}_{case}.gpu_gate.json"
            gate_record = {**gate, "suite_id": suite_id, "case": case, "checked_unix": time.time()}
            _json(gate_record_path, gate_record, exclusive=True)
            if case == "multidomain_vqa":
                ready_dir = evidence / f"{suite_id}_{case}.ranks_ready"
                ready_dir.mkdir(exist_ok=False)
            if case == "frozen_prefix":
                started = time.time()
                try:
                    _execute_prefix_case(root, native_root, suite_id)
                    code = 0
                except BaseException as exc:
                    code = getattr(exc, "returncode", 1) or 1
                    suite["cases"].append({"case": case, "exit_code": code, "started_unix": started,
                                           "ended_unix": time.time(), "error": f"{type(exc).__name__}: {exc}"})
                    suite["status"] = "failed"
                    _atomic_json(state_path, suite)
                    raise
            else:
                tree = SINGLE_CASES.get(case, ("multidomain", "1.1", "stage1"))[1]
                env = _expected_environment(root, tree)
                world = 4 if case == "multidomain_vqa" else 1
                env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" if world == 4 else "0"
                env["WORLD_SIZE"] = str(world)
                env["UNIDRIVE_RUN_ID"] = f"{suite_id}_{case}"
                env["UNIDRIVE_INVOCATION_ID"] = uuid.uuid4().hex[:20]
                env["UNI_GPU_CLOSURE_PARENT_SUITE"] = suite_id
                env["UNI_GPU_CLOSURE_PARENT_CASE"] = case
                env.update({"NOCUDNN_ALL": "0", "LD_DEBUG": "libs",
                            "LD_DEBUG_OUTPUT": str(_bundle_files(root)["logs"] / f"{suite_id}_{case}.lddebug")})
                helper = root / "nocudnn_train.py"
                child = [str(helper), str(Path(__file__).resolve()), "--root", str(root),
                         "--native-anchors", str(native_root), "--vqa-image", str(vqa_image),
                         "--vqa-jsonl", str(vqa_jsonl), "--suite-id", suite_id,
                         "case", case, "--execute"]
                if world == 4:
                    master_port = 31000 + int(uuid.uuid4().hex[:4], 16) % 20000
                    command = [str(PYTHON), "-m", "torch.distributed.run", "--nnodes=1",
                               "--nproc_per_node=4", f"--master_port={master_port}", *child]
                else:
                    command = [str(PYTHON), *child]
                started = time.time()
                with log.open("x", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=str(CODE_ROOT), env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, text=True)
                code = int(result.returncode)
            event = {"case": case, "exit_code": code, "started_unix": started,
                     "ended_unix": time.time(), "log": str(log), "gpu_gate": gate}
            suite["cases"].append(event)
            ranks = 4 if case == "multidomain_vqa" else 1
            receipts = []
            for rank in range(ranks):
                receipt_path = evidence / "cases" / f"{suite_id}_{case}_rank{rank:04d}.json"
                if not receipt_path.is_file():
                    event.setdefault("unreported_ranks", []).append(rank)
                    continue
                receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
            event["rank_exit_codes"] = {str(item.get("rank")): item.get("exit_code") for item in receipts}
            for item in receipts:
                if item.get("exit_code") is None and item.get("rank") is not None:
                    event["rank_exit_codes"][str(item["rank"])] = "unreported; inspect child log"
                    event.setdefault("unreported_ranks", []).append(item["rank"])
            for rank in event.get("unreported_ranks", []):
                event["rank_exit_codes"][str(rank)] = "unreported; inspect child log"
            event["rank_receipts"] = [
                str(evidence / "cases" / f"{suite_id}_{case}_rank{item['rank']:04d}.json")
                for item in receipts if "rank" in item
            ]
            if code != 0:
                suite["status"] = "failed"
                event["process_exit_code"] = code
                if event.get("unreported_ranks"):
                    event["rank_exit_code_note"] = "a rank did not write a final receipt; see torchrun log for its process exit code"
                _atomic_json(state_path, suite)
                print(json.dumps(suite, sort_keys=True))
                return code or 1
            valid_receipts, result_errors = _validate_rank_receipts(case, receipts, ranks)
            if not valid_receipts:
                suite["status"] = ("failed_missing_rank_receipt" if event.get("unreported_ranks")
                                   else "failed_rank_receipt")
                event["error"] = result_errors
                _atomic_json(state_path, suite)
                return 1
            _atomic_json(state_path, suite)
        suite["status"] = "complete"
        suite["finished_unix"] = time.time()
        _atomic_json(state_path, suite)
        print(json.dumps(suite, sort_keys=True))
        return 0
    except BaseException as exc:
        suite["status"] = "failed"
        suite["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(state_path, suite)
        raise


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("UNI_GPU_CLOSURE_ROOT"))
    parser.add_argument("--native-anchors")
    parser.add_argument("--vqa-image")
    parser.add_argument("--vqa-jsonl")
    parser.add_argument("--suite-id")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("preflight")
    sub.add_parser("compose")
    sub.add_parser("gpu-gate")
    compose_one = sub.add_parser("compose-one", help=argparse.SUPPRESS)
    compose_one.add_argument("case_id", choices=tuple(SINGLE_CASES) + ("multidomain_vqa",))
    suite = sub.add_parser("suite")
    suite.add_argument("--execute", action="store_true")
    case = sub.add_parser("case")
    case.add_argument("case_name", choices=GPU_CASES)
    case.add_argument("--execute", action="store_true")
    return parser


def main():
    args = _cli().parse_args()
    if args.command == "plan":
        print(json.dumps(_plan(args.root, args.native_anchors, args.vqa_jsonl), indent=2, sort_keys=True))
        return 0
    if args.command == "gpu-gate":
        report = _gpu_inventory()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready"] else 3
    if args.command == "compose-one":
        root = bundle_root(args.root)
        if not args.native_anchors or not args.suite_id:
            raise ValueError("internal compose-one needs --native-anchors and --suite-id")
        native = Path(args.native_anchors)
        vqa = Path(args.vqa_jsonl) if args.vqa_jsonl else root / "artifacts/synthetic_vqa.jsonl"
        tree, overrides = _case_overrides(args.case_id, root, native, vqa, args.suite_id)
        if args.case_id == "multidomain_vqa":
            info, _ = _compose_one(root, tree, overrides, "multidomain", "stage1", write=True)
        else:
            name, _, stage = SINGLE_CASES[args.case_id]
            info, _ = _compose_one(root, tree, overrides, name, stage, write=True)
        print(json.dumps(info, sort_keys=True))
        return 0
    root = bundle_root(args.root)
    _require_snapshot_code(root)
    if not args.native_anchors:
        raise ValueError("--native-anchors is required")
    native = Path(args.native_anchors)
    vqa = Path(args.vqa_jsonl) if args.vqa_jsonl else root / "artifacts/synthetic_vqa.jsonl"
    if vqa.resolve().parent != (root / "artifacts").resolve() or vqa.is_symlink():
        raise ValueError("VQA verification JSONL must be a new file under this bundle's artifacts directory")
    image = Path(args.vqa_image) if args.vqa_image else None
    if args.command == "preflight":
        if image is None:
            raise ValueError("--vqa-image must point to a real nuScenes-mini camera image")
        if not vqa.is_file():
            _prepare_vqa_fixture(root, image, vqa)
        report = _preflight(root, native, image, vqa)
        report["exit_code"] = 0 if report["real_source_checks_passed"] else 1
        report["status"] = "passed" if report["exit_code"] == 0 else "failed_real_source_check"
        _json(_bundle_files(root)["evidence"] / "preflight.json", report, exclusive=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return report["exit_code"]
    if args.command == "compose":
        if image is None:
            raise ValueError("--vqa-image must point to a real nuScenes-mini camera image")
        if not vqa.is_file():
            _prepare_vqa_fixture(root, image, vqa)
        report = _compose_cases(root, native, vqa, "compose_" + uuid.uuid4().hex[:10])
        payload = {"cases": report,
                   "limits": _plan(str(root), str(native), str(vqa))["profile_and_limits"],
                   "all_configs_composed_on_cpu": all(item["exit_code"] == 0 for item in report),
                   "exit_code": 0 if all(item["exit_code"] == 0 for item in report) else 1}
        _json(_bundle_files(root)["evidence"] / "compose.json", payload, exclusive=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload["exit_code"]
    if args.command == "case":
        if not args.execute:
            raise ValueError("case command requires explicit --execute")
        if not args.suite_id:
            raise ValueError("case requires --suite-id")
        if image is None:
            raise ValueError("case requires --vqa-image")
        required_evidence = (_bundle_files(root)["evidence"] / "preflight.json",
                             _bundle_files(root)["evidence"] / "compose.json")
        if not all(path.is_file() for path in required_evidence):
            raise RuntimeError("case execution requires passing CPU preflight.json and compose.json")
        if any(json.loads(path.read_text(encoding="utf-8")).get("exit_code") != 0
               for path in required_evidence):
            raise RuntimeError("CPU preflight or Hydra compose did not report exit_code=0")
        _prepare_vqa_fixture(root, image, vqa) if not vqa.is_file() else None
        world = int(os.environ.get("WORLD_SIZE", "1"))
        expected_world = 4 if args.case_name == "multidomain_vqa" else 1
        if world != expected_world:
            raise RuntimeError(f"{args.case_name} requires WORLD_SIZE={expected_world}, got {world}")
        parent_suite = os.environ.get("UNI_GPU_CLOSURE_PARENT_SUITE")
        parent_case = os.environ.get("UNI_GPU_CLOSURE_PARENT_CASE")
        lock = _bundle_files(root)["evidence"] / f"{args.suite_id}.lock"
        if parent_suite != args.suite_id or parent_case != args.case_name or not lock.is_file():
            raise RuntimeError("case execution must be started by its unique locked suite")
        if world == 1:
            gate = _gpu_inventory()
            if not gate["ready"]:
                print(json.dumps({"case": args.case_name, "status": "blocked_gpu_gate",
                                  "gpu_gate": gate, "no_case_started": True}, sort_keys=True))
                return 3
        else:
            gate_record = _bundle_files(root)["evidence"] / f"{args.suite_id}_{args.case_name}.gpu_gate.json"
            if not gate_record.is_file():
                raise RuntimeError("parent GPU gate record is missing")
            gate = json.loads(gate_record.read_text(encoding="utf-8"))
            if gate.get("suite_id") != args.suite_id or gate.get("case") != args.case_name or not gate.get("ready"):
                raise RuntimeError("parent GPU gate record is missing or closed")
        if args.case_name == "frozen_prefix":
            _execute_prefix_case(root, native, args.suite_id)
            return 0
        return int(_execute_train_case(root, native, vqa, args.suite_id, args.case_name) or 0)
    if args.command == "suite":
        if not args.execute:
            raise ValueError("suite requires explicit --execute")
        if not args.suite_id or not image:
            raise ValueError("suite requires --suite-id and --vqa-image")
        required_evidence = (_bundle_files(root)["evidence"] / "preflight.json",
                             _bundle_files(root)["evidence"] / "compose.json")
        if not all(path.is_file() for path in required_evidence):
            raise RuntimeError("suite requires passing CPU preflight.json and compose.json from this owned bundle")
        if any(json.loads(path.read_text(encoding="utf-8")).get("exit_code") != 0
               for path in required_evidence):
            raise RuntimeError("suite requires CPU preflight and Hydra compose exit_code=0")
        return _suite(root, native, image, vqa, args.suite_id, args.execute)
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
