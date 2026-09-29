import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from slurm.evaluation.sync_bench2drive_recovery import synchronize


def write_route(path: Path, route_id: str) -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "route", id=route_id)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_result(path: Path, route_id: str, valid: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "entry_status": "Finished" if valid else "Started",
                "eligible": valid,
                "_checkpoint": {
                    "records": (
                        [
                            {
                                "route_id": f"RouteScenario_{route_id}_rep0",
                                "status": "Completed",
                            }
                        ]
                        if valid
                        else []
                    )
                },
            }
        ),
        encoding="utf-8",
    )


class SyncBench2DriveRecoveryTest(unittest.TestCase):
    def test_seeds_promotes_and_reports_pending_routes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            shard0 = root / "shard0"
            shard1 = root / "shard1"
            recovery = root / "recovery"
            for directory in (shard0, shard1):
                (directory / "routes_all").mkdir(parents=True)
                (directory / "eval").mkdir()

            for route_id in range(220):
                shard = shard0 if route_id % 2 == 0 else shard1
                write_route(shard / "routes_all" / f"{route_id}.xml", str(route_id))

            write_result(shard0 / "eval" / "0.json", "0", valid=True)
            (recovery / "eval").mkdir(parents=True)
            write_result(recovery / "eval" / "1.json", "1", valid=True)
            write_result(shard0 / "eval" / "2.json", "2", valid=False)

            pending, seeded, promoted = synchronize(
                [shard0, shard1], recovery
            )

            self.assertEqual(seeded, ["0"])
            self.assertEqual(promoted, ["1"])
            self.assertNotIn("0", pending)
            self.assertNotIn("1", pending)
            self.assertIn("2", pending)
            self.assertEqual(len(pending), 218)
            self.assertTrue((recovery / "eval" / "0.json").is_file())
            self.assertTrue((shard1 / "eval" / "1.json").is_file())


if __name__ == "__main__":
    unittest.main()
