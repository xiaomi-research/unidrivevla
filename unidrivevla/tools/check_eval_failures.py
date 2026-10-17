"""CPU check that failed evaluators fail the run, retain old outputs, and reach every DDP rank."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.eval.registry import SubprocessEvaluator
from unidrivevla.train.trainer import Trainer


class CpuEvaluator(SubprocessEvaluator):
    def __init__(self, output_dir, mode):
        section = OmegaConf.create({"every_n_epochs": 0, "on_end": False})
        super().__init__(OmegaConf.create({}), section, Path(output_dir))
        self.benchmark = "cpu"
        self.root = Path(output_dir) / "eval" / "cpu"
        self.mode = mode

    def command(self, ckpt, out_dir, tag):
        score = out_dir / "score.json"
        if self.mode == "exit7":
            code = "import sys; sys.exit(7)"
        elif self.mode == "missing":
            code = "pass"
        elif self.mode == "sleeper":
            child_pid = out_dir / "child.pid"
            code = (f"import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',"
                    f"'import time; time.sleep(60)']); open({str(child_pid)!r},'x').write(str(child.pid)); "
                    "time.sleep(60)")
        else:
            code = f"import json; open({str(score)!r}, 'x').write(json.dumps({{'score': 0.75}}))"
        return [sys.executable, "-c", code]

    def read_result(self, out_dir):
        with open(out_dir / "score.json", encoding="utf-8") as stream:
            return json.load(stream)


class MemoryMonitor:
    def __init__(self):
        self.values = []

    def log(self, metrics, step, epoch):
        self.values.append((metrics, step, epoch))


def trainer_for(evaluator, rank=0, world_size=1, gloo_group=None):
    trainer = Trainer.__new__(Trainer)
    trainer.ctx = SimpleNamespace(is_main=rank == 0, rank=rank, world_size=world_size,
                                  gloo_group=gloo_group or (lambda: dist.group.WORLD))
    trainer.monitor = MemoryMonitor()
    trainer.global_step = 4
    trainer.epoch = 1
    trainer.evaluators = [evaluator]
    return trainer


def run_case(root, mode):
    evaluator = CpuEvaluator(root, mode)
    original = evaluator.root / "same_tag"
    original.mkdir(parents=True)
    sentinel = original / "eval.log"
    sentinel.write_text("old evaluation log must remain\n", encoding="utf-8")
    evaluator.submit(Path("unused.ckpt"), "same_tag")
    assert sentinel.read_text(encoding="utf-8") == "old evaluation log must remain\n"
    run_dir = evaluator.jobs[0][2]
    assert run_dir != original and run_dir.name == "same_tag_run_001"
    results = evaluator.collect(block=True)
    assert len(results) == 1 and results[0][0] == "same_tag"
    if mode == "valid":
        assert results[0][1] == {"score": 0.75}
        trainer_for(evaluator).report(evaluator, results)
        assert (evaluator.root / "same_tag" / "eval.log").read_text(encoding="utf-8").startswith("old")
        return
    assert "error" in results[0][1]
    trainer = trainer_for(evaluator)
    try:
        trainer.report(evaluator, results)
    except RuntimeError as exc:
        assert "evaluation failed" in str(exc)
    else:
        raise AssertionError(f"{mode} evaluator failure must fail the trainer")
    assert (run_dir / "eval_error.txt").is_file()
    assert sentinel.read_text(encoding="utf-8") == "old evaluation log must remain\n"


def rank_check(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        evaluator = CpuEvaluator(Path(rendezvous).parent / f"rank{rank}", "valid")
        trainer = trainer_for(evaluator, rank, 2)
        results = [("periodic", {"error": "exit 7, see eval.log"})] if rank == 0 else []
        try:
            trainer.report(evaluator, results)
        except RuntimeError as exc:
            assert "cpu/periodic" in str(exc) and "exit 7" in str(exc)
        else:
            raise AssertionError("the evaluator error must reach every rank")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def cancel_case(root):
    if os.name != "posix":
        return
    evaluator = CpuEvaluator(root, "sleeper")
    evaluator.submit(Path("unused.ckpt"), "owned")
    _, proc, out_dir = evaluator.jobs[0]
    child_pid_file = out_dir / "child.pid"
    deadline = time.monotonic() + 5
    while not child_pid_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid_file.is_file(), "owned scoring child did not start"
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    evaluator.cancel()
    assert not evaluator.jobs and proc.poll() is not None
    for _ in range(40):
        try:
            state = Path(f"/proc/{child_pid}/stat").read_text(encoding="utf-8").split()[2]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("cancel left an owned scoring child running")


def cli_failure(mode):
    with tempfile.TemporaryDirectory(prefix="unidrivevla-eval-cli-") as temp:
        evaluator = CpuEvaluator(temp, mode)
        evaluator.submit(Path("unused.ckpt"), "final")
        results = evaluator.collect(block=True)
        trainer_for(evaluator).report(evaluator, results)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--fatal":
        cli_failure(sys.argv[2])
        raise AssertionError("an evaluator error returned success")

    with tempfile.TemporaryDirectory(prefix="unidrivevla-eval-") as temp:
        for mode in ("exit7", "missing", "valid"):
            run_case(Path(temp) / mode, mode)
        cancel_case(Path(temp) / "cancel")
        mp.spawn(rank_check, args=(str(Path(temp) / "rendezvous"),), nprocs=2, join=True)
        for mode in ("exit7", "missing"):
            proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--fatal", mode],
                                  cwd=Path(__file__).resolve().parents[2],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            assert proc.returncode != 0, f"{mode} evaluation failure exited successfully"
            assert "evaluation failed" in proc.stderr
    print("OK: exit 7 and missing score fail CLI; valid score passes; old logs survive; DDP error broadcasts")


if __name__ == "__main__":
    main()
