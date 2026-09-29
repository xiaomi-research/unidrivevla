import argparse
import json
import tempfile
import unittest
from pathlib import Path

from slurm.evaluation.run_manual_recovery_queue import parse_route_destination, promote_route


def valid_result(route_id):
    return {
        "entry_status": "Finished",
        "eligible": True,
        "_checkpoint": {
            "records": [
                {
                    "route_id": f"RouteScenario_{route_id}_rep0",
                    "status": "Completed",
                }
            ]
        },
    }


class ManualRecoveryQueueTest(unittest.TestCase):
    def test_parse_route_destination(self):
        route_id, destination = parse_route_destination("11715=/tmp/shard1")
        self.assertEqual(route_id, "11715")
        self.assertEqual(destination, Path("/tmp/shard1"))

    def test_parse_route_destination_rejects_invalid_route(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_route_destination("route11715=/tmp/shard1")

    def test_promote_route_replaces_started_result_atomically(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.json"
            destination = root / "shard1"
            destination_eval = destination / "eval"
            destination_eval.mkdir(parents=True)
            target = destination_eval / "11715.json"
            target.write_text(json.dumps({"entry_status": "Started", "eligible": False}))
            source.write_text(json.dumps(valid_result("11715")))

            promote_route(source, destination, "11715")

            self.assertEqual(json.loads(target.read_text()), valid_result("11715"))
            self.assertEqual(list(destination_eval.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
