"""Masked waypoint errors for converted datasets; these are not closed-loop benchmark scores."""

import numpy as np

from unidrivevla.eval.openloop import OpenLoopEvaluator
from unidrivevla.eval.registry import register


def waypoint_metrics(samples):
    errors, final, at_second = [], [], {s: [] for s in (1, 2, 3)}
    valid_samples = 0
    for token, prediction, info in samples:
        pred, gt = np.asarray(prediction), np.asarray(info["trajectory"])
        mask = np.asarray(info["mask"], dtype=bool)
        interval_us = info.get("sample_interval_us")
        if type(interval_us) is not int or interval_us <= 0:
            raise ValueError(f"missing or invalid sample_interval_us for {token}")
        if (pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] < 2 or gt.shape[1] < 2
                or len(pred) != len(gt) or mask.shape != (len(gt),)):
            raise ValueError(f"invalid trajectory/mask shape for {token}")
        length = min(len(pred), len(gt))
        valid = mask[:length]
        if not valid.any():
            continue
        if not np.isfinite(pred[:length, :2][valid]).all() or not np.isfinite(gt[:length, :2][valid]).all():
            raise ValueError(f"non-finite supervised waypoint in {token}")
        distances = np.linalg.norm(pred[:length, :2] - gt[:length, :2], axis=1)
        errors.extend(distances[valid].tolist())
        final.append(float(distances[np.flatnonzero(valid)[-1]]))
        valid_samples += 1
        for second, values in at_second.items():
            time_us = second * 1_000_000
            if time_us % interval_us:
                continue
            index = time_us // interval_us - 1
            if 0 <= index < length and valid[index]:
                values.append(float(distances[index]))
    result = {"samples": len(samples), "valid_samples": valid_samples, "valid_waypoints": len(errors)}
    if errors:
        result.update(ADE=float(np.mean(errors)), FDE=float(np.mean(final)))
    result.update({f"L2_{second}s": float(np.mean(values)) for second, values in at_second.items() if values})
    return result


class ExternalPlanningEvaluator(OpenLoopEvaluator):
    def score(self, samples):
        return {f"{self.source}/{key}": value for key, value in waypoint_metrics(samples).items()}


@register("physicalai")
class PhysicalAIEvaluator(ExternalPlanningEvaluator):
    source = "physicalai"


@register("lead")
class LEADEvaluator(ExternalPlanningEvaluator):
    source = "lead"
