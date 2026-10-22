"""CoC text is part of the driving prefix, and the planner is allowed to see it."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from unidrivevla.models.unidrivevla.dense_heads.canbus_prompt import (  # noqa: E402
    build_driving_user_prompt, coc_language_target, load_coc_by_token,
)
from unidrivevla.models.unidrivevla.dense_heads.utils import make_att_2d_masks  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "coc.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"sample_token": "abc", "cot": "the car yields"}\n{"token": "skip"}\n')
        table = load_coc_by_token(path)
    assert coc_language_target(table["abc"]) == "the car yields"
    assert coc_language_target(table.get("missing")) == ""
    prompt = build_driving_user_prompt(
        nav_cmd="GO STRAIGHT", hist_traj=[[0, 0], [1, 0]],
        trajectory_prompt_template="hist {hist_traj_str} cmd {nav_cmd}",
        use_coc_language_prompt=True, include_ego_status_prompt=False,
    )
    assert "chain-of-causation" in prompt

    # prompt tokens, the causal answer, then one planning token. Each starts a visible segment.
    marks = torch.tensor([[1, 1, 1, 1, 1, 1]])
    padding = torch.ones_like(marks, dtype=torch.bool)
    visible = make_att_2d_masks(padding, marks)
    assert bool(visible[0, -1, 3]) and bool(visible[0, -1, 4])
    print("ok")


if __name__ == "__main__":
    main()
