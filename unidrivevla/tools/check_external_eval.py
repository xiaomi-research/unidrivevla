"""CPU check for masked external-data waypoint metrics, independent of simulator scoring."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.eval.external import waypoint_metrics


def main():
    gt = np.zeros((6, 3))
    pred = np.column_stack([np.arange(1, 7), np.zeros((6, 2))])
    full_info = {"trajectory": gt, "mask": np.ones(6), "sample_interval_us": 500_000}
    full = ("full", pred, full_info)
    partial = ("partial", pred, {**full_info, "mask": np.array([1, 1, 0, 0, 0, 0])})
    missing = ("missing", pred, {**full_info, "mask": np.zeros(6)})
    result = waypoint_metrics([full, partial, missing])
    assert result["valid_waypoints"] == 8 and result["valid_samples"] == 2
    assert result["ADE"] == 3 and result["FDE"] == 4
    assert result["L2_1s"] == 2 and result["L2_2s"] == 4 and result["L2_3s"] == 6
    assert "ADE" not in waypoint_metrics([missing])

    lead_gt = np.zeros((8, 3))
    lead_pred = np.column_stack([np.arange(1, 9), np.zeros((8, 2))])
    lead = ("lead-2s", lead_pred,
            {"trajectory": lead_gt, "mask": np.ones(8), "sample_interval_us": 250_000})
    lead_result = waypoint_metrics([lead])
    assert lead_result["L2_1s"] == 4 and lead_result["L2_2s"] == 8
    assert "L2_3s" not in lead_result

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
    try:
        waypoint_metrics([("missing-interval", pred, {"trajectory": gt, "mask": np.ones(6)})])
    except ValueError as exc:
        assert "sample_interval_us" in str(exc)
    else:
        raise AssertionError("external metric must not guess a target interval")
    print("OK: external metrics use per-cache interval; LEAD 2s has no synthetic 3s score")


if __name__ == "__main__":
    main()
