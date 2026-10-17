from unidrivevla.eval.registry import EVALUATORS, SubprocessEvaluator, build_evaluators, register

import unidrivevla.eval.navsim  # noqa: F401  (registers "navsim")
import unidrivevla.eval.openloop  # noqa: F401  (registers "nuscenes", "b2d")

__all__ = ["EVALUATORS", "SubprocessEvaluator", "build_evaluators", "register"]
import unidrivevla.eval.external  # noqa: F401 (converted-data waypoint errors)
