"""Run the CPU regression scripts in separate processes; keep every result and fail if any check fails."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

CHECKS = ("checkpoint", "resume", "master_adamw", "vqa", "task_schedule", "targets", "streams", "mixture", "multidomain",
          "supervision", "image_pipeline", "official_heads", "planning_metrics", "closedloop", "external_sources",
          "external_eval", "eval_schedule", "b2d_closedloop", "insulation", "attention_contract", "run_paths",
          "eval_failures", "native_sources", "native_policy", "sequences", "motion_decoder", "optional_attention",
          "gradient_reduction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    parser.add_argument("--output", type=Path, help="new output directory; existing results are never overwritten")
    parser.add_argument("--timeout", type=int, default=300, help="per-check timeout in seconds")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    output = args.output or Path(tempfile.mkdtemp(prefix="unidrivevla-checks-"))
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    tools = Path(__file__).resolve().parent
    env["PYTHONPATH"] = str(tools.parents[1]) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    results = []
    for name in args.checks:
        path = tools / f"check_{name}.py"
        log = output / f"{name}.log"
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            try:
                run = subprocess.run([sys.executable, "-B", str(path)], cwd=str(tools.parents[1]), env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                code = run.returncode
            except subprocess.TimeoutExpired:
                code = 124
        results.append({"check": name, "exit_code": code, "seconds": round(time.monotonic() - start, 3),
                        "log": str(log.resolve())})
        print(f"{'PASS' if code == 0 else 'FAIL'} {name} ({code})", flush=True)
        (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results: {output / 'results.json'}", flush=True)
    return int(any(item["exit_code"] for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
