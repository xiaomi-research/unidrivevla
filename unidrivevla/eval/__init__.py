from unidrivevla.eval.registry import EVALUATORS, SubprocessEvaluator, build_evaluators, register

import unidrivevla.eval.navsim  # noqa: F401  (registers "navsim")
import unidrivevla.eval.openloop  # noqa: F401  (registers "nuscenes", "b2d")
import unidrivevla.eval.nuscenes_pr10  # noqa: F401  (registers "nuscenes_pr10")

__all__ = ["EVALUATORS", "SubprocessEvaluator", "build_evaluators", "register"]
import unidrivevla.eval.external  # noqa: F401 (converted-data waypoint errors)
