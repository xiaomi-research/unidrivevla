"""CPU check for masked external-data waypoint metrics, independent of simulator scoring."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.eval.external import waypoint_metrics


def main():
    gt = np.zeros((6, 3))
    pred = np.column_stack([np.arange(1, 7), np.zeros((6, 2))])
    full = ("full", pred, {"trajectory": gt, "mask": np.ones(6)})
    partial = ("partial", pred, {"trajectory": gt, "mask": np.array([1, 1, 0, 0, 0, 0])})
    missing = ("missing", pred, {"trajectory": gt, "mask": np.zeros(6)})
    result = waypoint_metrics([full, partial, missing])
    assert result["valid_waypoints"] == 8 and result["valid_samples"] == 2
    assert result["ADE"] == 3 and result["FDE"] == 4
    assert result["L2_1s"] == 2 and result["L2_2s"] == 4 and result["L2_3s"] == 6
    assert "ADE" not in waypoint_metrics([missing])
    try:
        waypoint_metrics([("short", pred[:2], full[2])])
    except ValueError:
        pass
    else:
        raise AssertionError("a shortened prediction must not hide later valid targets")
    invalid = pred.copy()
    invalid[0, 0] = np.nan
    try:
        waypoint_metrics([("bad", invalid, full[2])])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite supervised prediction must fail")
    print("OK: external ADE/FDE, per-horizon L2, masked and empty targets")


if __name__ == "__main__":
    main()
