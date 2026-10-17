"""Pre-Hydra run identity and output-directory guards for the CLI entrypoints."""

import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_CHECKPOINT = re.compile(r"^step_\d{8}(?:_e\d{4}_b\d{8})?$")
_PATH_OVERRIDES = {"experiment_name", "output_dir"}
_UNSAFE_HYDRA_OVERRIDES = {
    "hydra.run.dir", "hydra.output_subdir", "hydra.job.chdir", "hydra.runtime.output_dir", "hydra.mode",
}


def _overrides(argv):
    from hydra.core.override_parser.overrides_parser import OverridesParser

    parser = OverridesParser.create()
    result = {}
    unsafe = []
    for arg in argv:
        if "=" not in arg or arg.startswith("-"):
            continue
        override = parser.parse_overrides([arg])[0]
        key, value = override.key_or_group, override.value()
        if key in _UNSAFE_HYDRA_OVERRIDES or key.startswith(("hydra.sweep.", "hydra.sweeper.", "hydra.launcher.")):
            unsafe.append(key)
        if key in _PATH_OVERRIDES or key in {"data", "data_loader", "train.resume"}:
            result[key] = value
    return result, sorted(unsafe)


def _path_override(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a literal string for the pre-Hydra output guard")
    if not value.strip() or "$" in value:
        raise ValueError(f"{name} must be a literal path/name for the pre-Hydra output guard")
    if name == "experiment_name" and (value in {".", ".."} or "/" in value or "\\" in value):
        raise ValueError("experiment_name must be a single path component")
    return value


def _has_complete_checkpoint(output_dir):
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.is_dir():
        return False
    return any(
        _CHECKPOINT.fullmatch(path.name) and (path / "trainer_state.pt").is_file()
        for path in checkpoints.iterdir()
        if path.is_dir()
    )


def _claim_directory(path, invocation_id):
    """Atomically reserve a path; ranks from this invocation share the same marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = ".unidrive_invocation"
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=str(path.parent)))
    (temporary / marker).write_text(invocation_id, encoding="ascii")
    try:
        os.rename(temporary, path)
    except OSError:
        owner = path / marker
        if not owner.is_file() or owner.read_text(encoding="ascii") != invocation_id:
            raise FileExistsError(f"output is already owned by another invocation: {path}")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _owned_by(path, invocation_id):
    marker = Path(path) / ".unidrive_invocation"
    return marker.is_file() and marker.read_text(encoding="ascii") == invocation_id


def _rank_layout():
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    rank_text = os.environ.get("RANK")
    if rank_text is None:
        if world_size > 1:
            raise ValueError("multi-rank CLI needs RANK set by torchrun")
        rank = 0
    else:
        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError("RANK must be an integer") from exc
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got {rank}/{world_size}")
    return rank, world_size


def _exclusive_claim(path, content, description):
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(description) from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as claim:
        claim.write(content)


def _claim_rank(provenance_dir, run_id, invocation_id, rank, world_size):
    claims = provenance_dir / ".unidrive_rank_claims"
    claims.mkdir(parents=True, exist_ok=True)
    rank_claim = claims / f"rank_{rank:05d}.claim"
    content = f"rank={rank}\nworld_size={world_size}\nrun_id={run_id}\ninvocation_id={invocation_id}\n"
    _exclusive_claim(rank_claim, content,
                     f"rank {rank}/{world_size} already claimed invocation {invocation_id}; use fresh IDs")


def _target_dir(overrides, run_id, mode):
    if "output_dir" in overrides:
        return Path(_path_override(overrides["output_dir"], "output_dir")).resolve()
    if "experiment_name" in overrides:
        name = _path_override(overrides["experiment_name"], "experiment_name")
    else:
        data = overrides.get("data_loader", overrides.get("data", "navsim"))
        data = _path_override(data, "data")
        name = f"unidrivevla_{data}_{mode}_{run_id}"
    root = Path(os.environ.get("UNIDRIVE_EXP_ROOT", "/data/yongkangli/UniDriveVLA/exp"))
    return (root / name).resolve()


def prepare_run_paths(mode, argv=None):
    """Reserve a pre-Hydra output; multi-rank callers must share fresh IDs on every rank and invocation."""
    if mode not in {"train", "infer"}:
        raise ValueError(f"unknown run mode: {mode!r}")
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-m", "--multirun", "--config-name", "-cn", "--config-path", "-cp"} or
           arg.startswith(("--config-name=", "--config-path=", "~hydra.")) for arg in argv):
        raise ValueError("multirun and alternate Hydra config paths are unsupported by the output guard")
    overrides, unsafe = _overrides(argv)
    if unsafe:
        raise ValueError(f"Hydra output/location overrides are disabled by the output guard: {unsafe}")
    rank, world_size = _rank_layout()

    run_id = os.environ.get("UNIDRIVE_RUN_ID")
    resume = mode == "train" and overrides.get("train.resume", False) is True
    explicit_path = "output_dir" in overrides or "experiment_name" in overrides
    if run_id is not None and not _RUN_ID.fullmatch(run_id):
        raise ValueError("UNIDRIVE_RUN_ID must be 1-96 letters, digits, dots, underscores, or hyphens")
    if run_id is None:
        if world_size > 1:
            raise ValueError("manual multi-rank CLI needs a fresh shared UNIDRIVE_RUN_ID for this invocation")
        else:
            run_id = uuid.uuid4().hex[:16]
    os.environ["UNIDRIVE_RUN_ID"] = run_id
    os.environ["UNIDRIVE_RUN_MODE"] = mode

    invocation_id = os.environ.get("UNIDRIVE_INVOCATION_ID")
    if world_size == 1:
        invocation_id = uuid.uuid4().hex[:16]
    elif invocation_id is None:
        raise ValueError("manual multi-rank CLI needs a fresh shared UNIDRIVE_INVOCATION_ID for this invocation")
    if not _RUN_ID.fullmatch(invocation_id):
        raise ValueError("UNIDRIVE_INVOCATION_ID must be 1-96 letters, digits, dots, underscores, or hyphens")
    os.environ["UNIDRIVE_INVOCATION_ID"] = invocation_id

    output_dir = _target_dir(overrides, run_id, mode)
    if mode == "train" and resume:
        if not explicit_path:
            raise ValueError("train.resume=true requires an explicit experiment_name or output_dir")
        if not _has_complete_checkpoint(output_dir):
            raise FileNotFoundError(f"train.resume=true needs a complete checkpoint under {output_dir}")
    else:
        if output_dir.exists():
            if not _owned_by(output_dir, invocation_id):
                raise FileExistsError(
                    f"{mode} output already exists: {output_dir}; choose a new path, "
                    "or use train.resume=true for a run with a complete checkpoint"
                )
        else:
            _claim_directory(output_dir, invocation_id)
    provenance_dir = output_dir / "code" / "hydra" / invocation_id
    _claim_directory(provenance_dir, invocation_id)
    _claim_rank(provenance_dir, run_id, invocation_id, rank, world_size)
    return output_dir, run_id
