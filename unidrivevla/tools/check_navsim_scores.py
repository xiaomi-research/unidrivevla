"""CSV fixtures for version- and entry-specific NAVSIM aggregate row selection."""

import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.eval.navsim import NavsimEvaluator


def evaluator(root, split):
    section = OmegaConf.create({"every_n_epochs": 0, "on_end": True, "split": split})
    return NavsimEvaluator(OmegaConf.create({"experiment_name": "fixture"}), section, root)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("token", "valid", "score", "pdm_score", "comfort"))
        writer.writeheader()
        writer.writerows(rows)
    return path.parent


def read(root, split, rows):
    output = root / "scores"
    write_csv(output / "scores.csv", rows)
    with patch("unidrivevla.eval.navsim.devkit_root", return_value=root):
        return evaluator(root.parent, split).read_result(output)


def must_fail(root, split, rows):
    try:
        read(root, split, rows)
    except ValueError:
        return
    raise AssertionError(f"invalid/ambiguous aggregate rows accepted: {rows}")


def main():
    with tempfile.TemporaryDirectory(prefix="unidrivevla-navsim-score-") as temp:
        base = Path(temp)
        v11 = base / "navsim1.1"
        result = read(v11, "navtest", [
            {"token": "average", "score": "0.61", "pdm_score": "", "comfort": "0.8"},
            {"token": "average_expert_frames", "score": "0.99", "pdm_score": "", "comfort": "1.0"},
        ])
        assert result["score"] == 0.61 and result["comfort"] == 0.8

        v20 = base / "navsim2.0"
        rows = [
            {"token": "average_all_frames", "valid": "True", "score": "0.8", "pdm_score": "", "comfort": "0.7"},
            {"token": "pseudo_closed_loop", "valid": "True", "score": "0.4", "pdm_score": "", "comfort": "0.5"},
            {"token": "average_expert_frames", "score": "0.96", "pdm_score": "", "comfort": "1.0"},
        ]
        assert read(v20, "navtest", rows)["score"] == 0.8  # pinned v2.0 outputs both rows for this split
        assert read(v20, "navtest_two_stage", rows)["score"] == 0.4
        assert read(v20, "navtest", rows + [rows[1]])["score"] == 0.8  # unselected pseudo row is ignored
        assert read(v20, "navtest", [rows[0], {**rows[1], "valid": "False"}])["score"] == 0.8
        must_fail(v20, "navtest_two_stage", [rows[0], {**rows[1], "valid": "False"}])
        must_fail(v20, "navtest", [{**rows[0], "valid": "False"}, rows[1]])

        one_stage = base / "dedicated-one-stage" / "navsim2.0"
        one_stage.mkdir(parents=True)
        (one_stage / "navsim" / "planning" / "script").mkdir(parents=True)
        (one_stage / "navsim" / "planning" / "script" / "run_pdm_score_one_stage.py").write_text("", encoding="utf-8")
        assert read(one_stage, "navtest", rows)["score"] == 0.8

        extended = [{"token": "extended_pdm_score_combined", "score": "0.53", "pdm_score": "", "comfort": "0.6"}]
        assert read(v20, "navsafe_two_stage", extended)["score"] == 0.53
        must_fail(v20, "navsafe_two_stage", rows[:1] + rows[2:])  # do not fall back to averages
        must_fail(v20, "navtest", rows + [rows[0]])  # duplicate selected overall row
        must_fail(v20, "navsafe_two_stage", rows + extended)  # two overall aliases are ambiguous
        must_fail(v20, "navtest", [
            {"token": "average_all_frames", "score": "nan", "pdm_score": "", "comfort": "0.6"}
        ])
        must_fail(v20, "navtest", [
            {"token": "average_all_frames", "score": "", "pdm_score": "", "comfort": "0.6"}
        ])
        must_fail(one_stage, "navtest", rows + [
            {"token": "average", "score": "0.88", "pdm_score": "", "comfort": "0.9"}
        ])
    print("OK: NAVSIM 1.1 average; NAVSIM 2.0 scorer-entry aggregates; missing, duplicate and ambiguous rows fail")


if __name__ == "__main__":
    main()
