import json
import os
import tempfile
import unittest
from pathlib import Path

from slurm.evaluation.recover_remote_routes import (
    discover_candidates,
    promote_valid_results,
)


def _result(status, entry_status="Finished", eligible=True):
    return {
        "entry_status": entry_status,
        "eligible": eligible,
        "_checkpoint": {"records": [{"status": status}]},
    }


class RemoteRecoveryTest(unittest.TestCase):
    def _write_attempt(self, root, route_id, attempt, gpu, marker="Segmentation fault"):
        stdout_dir = root / "stdout"
        stderr_dir = root / "stderr"
        stdout_dir.mkdir(parents=True, exist_ok=True)
        stderr_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = stdout_dir / f"stdout_{route_id}_{attempt}.txt"
        stderr_path = stderr_dir / f"stderr_{route_id}_{attempt}.txt"
        stdout_path.write_text(
            f"Route {route_id} uses CUDA_VISIBLE_DEVICES={gpu}, "
            f"CARLA graphics adapter {gpu}, PORT=20000, TM_PORT=30000\n",
            encoding="utf-8",
        )
        stderr_path.write_text(marker, encoding="utf-8")
        os.utime(stdout_path, (100.0, 100.0))
        os.utime(stderr_path, (100.0, 100.0))

    def test_discovers_stale_terminal_failures_on_selected_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            for attempt in range(1, 4):
                self._write_attempt(source, "2903", attempt, gpu=0)
                self._write_attempt(source, "2904", attempt, gpu=1)

            candidates = discover_candidates(
                source,
                target_gpu=0,
                max_remote_attempts=3,
                stale_seconds=180,
                now=1000.0,
            )
            self.assertEqual([candidate.route_id for candidate in candidates], ["2903"])

    def test_does_not_recover_running_or_incomplete_attempt_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            for attempt in range(1, 3):
                self._write_attempt(source, "2050", attempt, gpu=0)
            for attempt in range(1, 4):
                self._write_attempt(source, "2084", attempt, gpu=0)
            os.utime(source / "stderr" / "stderr_2084_3.txt", (950.0, 950.0))

            candidates = discover_candidates(
                source,
                target_gpu=0,
                max_remote_attempts=3,
                stale_seconds=180,
                now=1000.0,
            )
            self.assertEqual(candidates, [])

    def test_valid_policy_failure_is_not_recovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            for attempt in range(1, 4):
                self._write_attempt(source, "1711", attempt, gpu=0)
            eval_dir = source / "eval"
            eval_dir.mkdir()
            (eval_dir / "1711.json").write_text(
                json.dumps(_result("Failed - Agent got blocked")),
                encoding="utf-8",
            )

            candidates = discover_candidates(
                source,
                target_gpu=0,
                max_remote_attempts=3,
                stale_seconds=180,
                now=1000.0,
            )
            self.assertEqual(candidates, [])

    def test_promotes_valid_result_and_replaces_invalid_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            recovery = root / "recovery"
            (source / "routes_all").mkdir(parents=True)
            (source / "routes_all" / "2903.xml").write_text("<routes/>", encoding="utf-8")
            (source / "eval").mkdir()
            (recovery / "eval").mkdir(parents=True)
            (source / "eval" / "2903.json").write_text(
                json.dumps(_result("Failed - Simulation crashed", entry_status="Started", eligible=False)),
                encoding="utf-8",
            )
            expected = _result("Failed - Agent got blocked")
            (recovery / "eval" / "2903.json").write_text(
                json.dumps(expected),
                encoding="utf-8",
            )

            self.assertEqual(promote_valid_results(recovery, source), ["2903"])
            self.assertEqual(
                json.loads((source / "eval" / "2903.json").read_text(encoding="utf-8")),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
