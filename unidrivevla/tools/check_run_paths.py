"""Exercise the pre-Hydra output guard and Hydra's real run/provenance paths on CPU."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent))


def run(command, env, expected=0):
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=45)
    if result.returncode != expected:
        raise AssertionError(
            f"command exit={result.returncode}, expected={expected}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main():
    try:
        import hydra  # noqa: F401
    except ImportError as exc:
        raise SystemExit("check_run_paths.py requires the project's Hydra dependency") from exc

    with tempfile.TemporaryDirectory(prefix="unidrivevla-run-paths-") as temp:
        root = Path(temp)
        config = root / "config"
        config.mkdir()
        script = root / "hydra_probe.py"
        script.write_text(
            "import os, sys\n"
            "import time\n"
            "from pathlib import Path\n"
            "from unidrivevla.run_paths import prepare_run_paths\n"
            "if os.environ.get('RANK') == '1' and os.environ.get('PROBE_DELAY_RANK1') == '1': time.sleep(2)\n"
            "prepare_run_paths(os.environ['PROBE_MODE'], sys.argv[1:])\n"
            "import hydra\n"
            "from omegaconf import DictConfig\n"
            "@hydra.main(config_path='config', config_name='tiny', version_base=None)\n"
            "def main(cfg: DictConfig):\n"
            "    root = Path(cfg.output_dir)\n"
            "    (root / ('rank_' + os.environ.get('RANK', '0') + '.txt')).write_text(\n"
            "        cfg.output_dir + '\\n' + os.environ['UNIDRIVE_RUN_ID'] + '\\n' +\n"
            "        os.environ['UNIDRIVE_INVOCATION_ID'])\n"
            "if __name__ == '__main__': main()\n",
            encoding="utf-8",
        )
        (config / "tiny.yaml").write_text(
            "hydra:\n"
            "  run:\n"
            "    dir: ${output_dir}\n"
            "  output_subdir: code/hydra/${oc.env:UNIDRIVE_INVOCATION_ID}\n"
            "  job:\n"
            "    chdir: false\n"
            "experiment_name: unidrivevla_${data}_${oc.env:PROBE_MODE}_${oc.env:UNIDRIVE_RUN_ID}\n"
            "output_dir: ${oc.env:UNIDRIVE_EXP_ROOT}/${experiment_name}\n"
            "data: navsim\n"
            "train:\n"
            "  resume: false\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(PACKAGE_ROOT.parent) + os.pathsep + env.get("PYTHONPATH", ""),
            "UNIDRIVE_EXP_ROOT": str(root / "exp"),
            "PROBE_MODE": "train",
        })
        env.pop("UNIDRIVE_RUN_ID", None)
        env.pop("UNIDRIVE_INVOCATION_ID", None)
        env.pop("WORLD_SIZE", None)
        env.pop("RANK", None)
        run([sys.executable, str(script)], env)
        run([sys.executable, str(script)], env)
        rank_files = sorted((root / "exp").glob("unidrivevla_navsim_train_*/rank_0.txt"))
        assert len(rank_files) == 2
        first_id, second_id = [path.read_text().splitlines()[1] for path in rank_files]
        assert first_id != second_id, "fresh invocations must receive distinct IDs"
        for run_id in (first_id, second_id):
            output = root / "exp" / f"unidrivevla_navsim_train_{run_id}"
            invocation_id = (root / "exp" / f"unidrivevla_navsim_train_{run_id}" / "rank_0.txt").read_text().splitlines()[2]
            assert (output / "code" / "hydra" / invocation_id / "config.yaml").is_file()
        print("ok   real Hydra gives fresh single-process invocations distinct output and provenance paths")

        quoted_name = root / "exp" / "quoted name"
        run([sys.executable, str(script), 'experiment_name="quoted name"'], env)
        assert (quoted_name / "rank_0.txt").is_file()
        escaped_name = root / "exp" / "escaped name"
        run([sys.executable, str(script), r"experiment_name=escaped\ name"], env)
        assert (escaped_name / "rank_0.txt").is_file()
        literal_name = " leading and trailing "
        literal_output = root / "exp" / literal_name
        run([sys.executable, str(script), f'experiment_name="{literal_name}"'], env)
        literal_fields = (literal_output / "rank_0.txt").read_text().splitlines()
        assert literal_fields[0] == str(literal_output)
        assert (literal_output / ".unidrive_invocation").read_text() == literal_fields[2]
        assert not (root / "exp" / literal_name.strip()).exists()
        print("ok   quoted, escaped, and edge-space names match Hydra's exact output path")

        # Existing fresh output is rejected before Hydra can replace its config or touch checkpoints.
        existing = root / "exp" / "preserve"
        sentinel = existing / "checkpoints" / "last.ckpt"
        old_config = existing / "code" / "hydra" / "old-run" / "config.yaml"
        sentinel.parent.mkdir(parents=True)
        old_config.parent.mkdir(parents=True)
        sentinel.write_bytes(b"checkpoint sentinel")
        old_config.write_text("old provenance", encoding="utf-8")
        provenance_before = set((existing / "code" / "hydra").glob("*/config.yaml"))
        env["PROBE_MODE"] = "train"
        rejected = run([sys.executable, str(script), "experiment_name=preserve"], env, expected=1)
        assert "already exists" in rejected.stderr
        assert sentinel.read_bytes() == b"checkpoint sentinel"
        assert old_config.read_text(encoding="utf-8") == "old provenance"
        assert set((existing / "code" / "hydra").glob("*/config.yaml")) == provenance_before
        rejected = run([sys.executable, str(script), 'experiment_name="preserve"'], env, expected=1)
        assert "already exists" in rejected.stderr
        assert sentinel.read_bytes() == b"checkpoint sentinel"
        assert old_config.read_text(encoding="utf-8") == "old provenance"
        assert set((existing / "code" / "hydra").glob("*/config.yaml")) == provenance_before
        assert not (root / "exp" / '"preserve"').exists()
        spaced_existing = root / "exp" / " guarded edge "
        spaced_checkpoint = spaced_existing / "checkpoints" / "last.ckpt"
        spaced_old_config = spaced_existing / "code" / "hydra" / "old-run" / "config.yaml"
        spaced_checkpoint.parent.mkdir(parents=True)
        spaced_old_config.parent.mkdir(parents=True)
        spaced_checkpoint.write_bytes(b"spaced checkpoint sentinel")
        spaced_old_config.write_text("spaced old provenance", encoding="utf-8")
        spaced_provenance_before = set((spaced_existing / "code" / "hydra").glob("*/config.yaml"))
        rejected = run([sys.executable, str(script), f'experiment_name=" guarded edge "'], env, expected=1)
        assert "already exists" in rejected.stderr
        assert spaced_checkpoint.read_bytes() == b"spaced checkpoint sentinel"
        assert spaced_old_config.read_text(encoding="utf-8") == "spaced old provenance"
        assert set((spaced_existing / "code" / "hydra").glob("*/config.yaml")) == spaced_provenance_before
        assert not (root / "exp" / "guarded edge").exists()
        print("ok   fresh training refuses an existing run before touching checkpoint or Hydra provenance")

        explicit_output = root / "exp" / "output-override"
        explicit_output.mkdir()
        explicit_sentinel = explicit_output / "keep.txt"
        explicit_sentinel.write_text("keep", encoding="utf-8")
        rejected = run([sys.executable, str(script), f'output_dir="{explicit_output}"'], env, expected=1)
        assert "already exists" in rejected.stderr
        assert explicit_sentinel.read_text(encoding="utf-8") == "keep"
        print("ok   explicit output_dir refuses an existing directory before Hydra writes files")

        # An explicit valid resume reuses checkpoint output while writing new per-invocation provenance.
        rejected = run([sys.executable, str(script), "experiment_name=preserve", "train.resume=true"], env,
                       expected=1)
        assert "complete checkpoint" in rejected.stderr
        assert set((existing / "code" / "hydra").glob("*/config.yaml")) == provenance_before
        checkpoint = existing / "checkpoints" / "step_00000001" / "trainer_state.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"complete checkpoint sentinel")
        resumed = run([sys.executable, str(script), "experiment_name=preserve", "train.resume=true"], env)
        del resumed
        created = [p for p in (existing / "code" / "hydra").glob("*/config.yaml") if p.parent.name != "old-run"]
        assert len(created) == 1, "resume must preserve old Hydra config and add one invocation record"
        assert old_config.read_text(encoding="utf-8") == "old provenance"
        assert checkpoint.read_bytes() == b"complete checkpoint sentinel"
        resume_invocation = (existing / "rank_0.txt").read_text().splitlines()[2]
        rank_claim = existing / "code" / "hydra" / resume_invocation / ".unidrive_rank_claims" / "rank_00000.claim"
        assert "rank=0\nworld_size=1\n" in rank_claim.read_text(encoding="ascii")
        print("ok   resume requires a complete checkpoint and writes new provenance beside the old record")

        # Two real torchrun ranks share fresh per-invocation IDs and the same output/provenance path.
        env["PROBE_MODE"] = "infer"
        env["UNIDRIVE_RUN_ID"] = "shared-rank-id"
        env["UNIDRIVE_INVOCATION_ID"] = "shared-rank-id"
        env["PROBE_DELAY_RANK1"] = "1"
        multi = run([
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc_per_node=2", str(script),
        ], env)
        del multi
        output = root / "exp" / "unidrivevla_navsim_infer_shared-rank-id"
        assert (output / "rank_0.txt").is_file() and (output / "rank_1.txt").is_file()
        assert (output / "rank_0.txt").read_text().splitlines() == (output / "rank_1.txt").read_text().splitlines()
        assert (output / "code" / "hydra" / "shared-rank-id" / "config.yaml").is_file()
        print("ok   delayed rank 1 joins after rank 0 and both share the same output/provenance paths")

        # A fixed explicit directory cannot be reused by a later multi-rank invocation with fresh IDs.
        explicit = root / "exp" / "two-rank-explicit"
        env["UNIDRIVE_RUN_ID"] = "explicit-run-one"
        env["UNIDRIVE_INVOCATION_ID"] = "explicit-invocation-one"
        torchrun = [
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc_per_node=2", str(script), f"output_dir={explicit}",
        ]
        run(torchrun, env)
        preserved_config = explicit / "code" / "hydra" / "explicit-invocation-one" / "config.yaml"
        explicit_sentinel = explicit / "checkpoint-sentinel.ckpt"
        explicit_sentinel.write_bytes(b"preserve explicit run")
        config_before = preserved_config.read_bytes()
        rejected = run(torchrun, env, expected=1)
        assert "already claimed invocation" in rejected.stderr
        assert preserved_config.read_bytes() == config_before
        assert explicit_sentinel.read_bytes() == b"preserve explicit run"
        assert not (explicit / "code" / "hydra" / "explicit-invocation-one" / "config.yaml.tmp").exists()
        print("ok   exact two-rank replay with the same IDs is rejected by per-rank claims")
        env["UNIDRIVE_RUN_ID"] = "explicit-run-two"
        env["UNIDRIVE_INVOCATION_ID"] = "explicit-invocation-two"
        rejected = run(torchrun, env, expected=1)
        assert "already exists" in rejected.stderr
        assert preserved_config.read_bytes() == config_before
        assert explicit_sentinel.read_bytes() == b"preserve explicit run"
        assert not (explicit / "code" / "hydra" / "explicit-invocation-two" / "config.yaml").exists()
        print("ok   repeated two-rank launch with a new ID cannot reuse an explicit output directory")

        # Manual multi-rank runs require a fresh shared run ID and invocation ID, even with explicit output_dir.
        env.pop("UNIDRIVE_RUN_ID")
        env.pop("UNIDRIVE_INVOCATION_ID")
        env["WORLD_SIZE"] = "2"
        env["RANK"] = "0"
        env["PROBE_MODE"] = "infer"
        provenance_before = set((root / "exp").glob("unidrivevla_navsim_infer_*/code/hydra/*/config.yaml"))
        rejected = run([sys.executable, str(script)], env, expected=1)
        assert "UNIDRIVE_RUN_ID" in rejected.stderr
        assert set((root / "exp").glob("unidrivevla_navsim_infer_*/code/hydra/*/config.yaml")) == provenance_before
        print("ok   uncoordinated multi-rank direct invocation fails before Hydra creates output")
        rejected = run([sys.executable, str(script), "output_dir=" + str(root / "exp" / "no-id")], env,
                       expected=1)
        assert "UNIDRIVE_RUN_ID" in rejected.stderr
        assert not (root / "exp" / "no-id").exists()
        env["UNIDRIVE_RUN_ID"] = "manual-run-id"
        rejected = run([sys.executable, str(script), "output_dir=" + str(root / "exp" / "no-invocation-id")],
                       env, expected=1)
        assert "UNIDRIVE_INVOCATION_ID" in rejected.stderr
        assert not (root / "exp" / "no-invocation-id").exists()
        env["UNIDRIVE_INVOCATION_ID"] = "manual-invocation-id"
        env["RANK"] = "2"
        rejected = run([sys.executable, str(script), "output_dir=" + str(root / "exp" / "invalid-rank")],
                       env, expected=1)
        assert "RANK must be in [0, WORLD_SIZE)" in rejected.stderr
        assert not (root / "exp" / "invalid-rank").exists()
        env["RANK"] = "not-an-integer"
        rejected = run([sys.executable, str(script), "output_dir=" + str(root / "exp" / "malformed-rank")],
                       env, expected=1)
        assert "RANK must be an integer" in rejected.stderr
        assert not (root / "exp" / "malformed-rank").exists()
        env.pop("UNIDRIVE_RUN_ID")
        env.pop("UNIDRIVE_INVOCATION_ID")
        env.pop("RANK")
        print("ok   explicit multi-rank output still requires both fresh shared IDs")

        # Hydra CLI overrides must not redirect config writes or bypass the output claim.
        env.pop("WORLD_SIZE")
        for override in (
            "hydra.run.dir=" + str(existing),
            "hydra.output_subdir=" + str(existing),
            "hydra.job.chdir=true",
            "hydra.sweep.dir=" + str(existing),
            "-m",
            "--config-name=other",
        ):
            before = {str(p.relative_to(existing)) for p in existing.rglob("*")}
            rejected = run([sys.executable, str(script), "output_dir=" + str(existing), override], env, expected=1)
            assert "Hydra" in rejected.stderr or "multirun" in rejected.stderr
            assert {str(p.relative_to(existing)) for p in existing.rglob("*")} == before
        print("ok   Hydra path/chdir/sweep/config overrides fail before touching output")

        # Inference must reject any existing destination, even if it contains no checkpoint.
        env["PROBE_MODE"] = "infer"
        inference_dir = root / "exp" / "inference-existing"
        inference_dir.mkdir()
        (inference_dir / "sentinel.txt").write_text("keep", encoding="utf-8")
        rejected = run([sys.executable, str(script), "experiment_name=inference-existing"], env, expected=1)
        assert "already exists" in rejected.stderr
        assert (inference_dir / "sentinel.txt").read_text(encoding="utf-8") == "keep"
        print("ok   inference refuses existing output before Hydra writes evaluation artifacts")


if __name__ == "__main__":
    main()
