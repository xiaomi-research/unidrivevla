import json
import tempfile
import unittest
from pathlib import Path

from slurm.evaluation.evaluate_utils import did_record_crash, did_route_crash
from slurm.evaluation.merge_bench2drive_shards import (
    load_valid_result,
    record_has_software_failure,
)


def _route_record(status, score_route=0.0):
    return {
        "status": status,
        "scores": {
            "score_route": score_route,
            "score_penalty": 1.0,
            "score_composed": score_route,
        },
    }


def _result(status, score_route=0.0, entry_status="Finished", eligible=True):
    return {
        "entry_status": entry_status,
        "eligible": eligible,
        "_checkpoint": {
            "records": [_route_record(status, score_route)],
        },
    }


class ResultClassificationTest(unittest.TestCase):
    def test_policy_failures_are_not_software_crashes(self):
        statuses = [
            "Failed",
            "Failed - TickRuntime",
            "Failed - Agent got blocked",
            "Failed - Agent deviated from the route",
            "Failed - Agent timed out",
        ]
        for status in statuses:
            with self.subTest(status=status):
                record = _route_record(status)
                self.assertFalse(did_record_crash(record))
                self.assertFalse(record_has_software_failure(record))

    def test_infrastructure_failures_are_crashes(self):
        statuses = [
            "Failed - Agent couldn't be set up",
            "Failed - Simulation crashed",
            "Failed - Agent crashed",
            "Failed - Agent's sensors were invalid",
        ]
        for status in statuses:
            with self.subTest(status=status):
                record = _route_record(status, score_route=10.0)
                self.assertTrue(did_record_crash(record))
                self.assertTrue(record_has_software_failure(record))

    def test_zero_completion_blocked_route_does_not_retry(self):
        self.assertFalse(did_route_crash(_result("Failed - Agent got blocked")))

    def test_unfinished_entry_retries_even_with_a_record(self):
        result = _result(
            "Failed - Agent got blocked",
            entry_status="Started",
            eligible=False,
        )
        self.assertTrue(did_route_crash(result))

    def test_merge_accepts_finished_policy_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "1711.json"
            result_path.write_text(
                json.dumps(_result("Failed - Agent got blocked")),
                encoding="utf-8",
            )
            result, error = load_valid_result(result_path)
            self.assertIsNotNone(result)
            self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
