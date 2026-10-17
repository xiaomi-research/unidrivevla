"""torchrun unidrivevla/train.py data=navsim|nuscenes|b2d [overrides]: the unified trainer
(scripts/train/run_unidrivevla_train.sh). The CLI claims a fresh output before Hydra writes provenance."""

import sys
from pathlib import Path

# Run as a script, this file's directory (the package) would be sys.path[0] and its subpackages (data, models, train,
# ...) would shadow top-level modules of those names; the package's parent goes there instead.
sys.path[0] = str(Path(__file__).resolve().parents[1])

# Hydra creates its run and output-subdir directories before calling main(). Guard them first.
if __name__ == "__main__":
    from unidrivevla.run_paths import prepare_run_paths

    prepare_run_paths("train", sys.argv[1:])

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from unidrivevla.train import Trainer  # noqa: E402


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
