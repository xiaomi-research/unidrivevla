"""Per-episode 0.5 s ego pose history in the current canonical frame."""

import math

import numpy as np


def _yaw(matrix):
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PoseHistory:
    """Retain measured poses and interpolate the model's four 0.5 s history slots.

    Before an episode has accumulated 1.5 s of observations, the earliest measured pose is repeated. This is the
    same left-padding convention used by the data adapter, without inventing unobserved motion.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []

    def history(self, timestamp, pose):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite nonnegative seconds")
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("ego2global must be a finite 4x4 matrix")
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError("timestamps must increase within an episode; call reset at episode boundaries")
        self.samples.append((timestamp, pose[:2, 3].copy(), _yaw(pose)))
        cutoff = timestamp - 2.0
        while len(self.samples) > 2 and self.samples[1][0] < cutoff:
            self.samples.pop(0)

        current_xy, current_yaw = self.samples[-1][1:]
        output = []
        for target in (timestamp - 1.5, timestamp - 1.0, timestamp - 0.5, timestamp):
            old_xy, old_yaw = self._at(target)
            delta = old_xy - current_xy
            c, s = math.cos(current_yaw), math.sin(current_yaw)
            output.append((c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1],
                           _wrap(old_yaw - current_yaw)))
        return np.asarray(output, dtype=np.float32)

    def _at(self, timestamp):
        if timestamp <= self.samples[0][0]:
            return self.samples[0][1], self.samples[0][2]
        for left, right in zip(self.samples, self.samples[1:]):
            if timestamp <= right[0]:
                fraction = (timestamp - left[0]) / (right[0] - left[0])
                yaw_delta = _wrap(right[2] - left[2])
                return (left[1] + fraction * (right[1] - left[1]),
                        _wrap(left[2] + fraction * yaw_delta))
        return self.samples[-1][1], self.samples[-1][2]
