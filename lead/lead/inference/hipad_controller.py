from collections import deque

import numpy as np


class PID:
    def __init__(self, kp, ki, kd, window):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.errors = deque([0.0] * int(window), maxlen=int(window))

    def step(self, error):
        error = float(error)
        previous = self.errors[-1]
        self.errors.append(error)
        integral = float(np.mean(self.errors))
        derivative = error - previous
        return self.kp * error + self.ki * integral + self.kd * derivative

    def reset(self):
        self.errors.extend([0.0] * self.errors.maxlen)


class HiPADDualPlanningController:
    """HiP-AD controller using temporal planning for speed and spatial planning for steer."""

    def __init__(
        self,
        *,
        temporal_hz=5.0,
        turn_kp=1.0,
        turn_ki=0.75,
        turn_kd=0.0,
        turn_window=10,
        speed_kp=5.0,
        speed_ki=0.5,
        speed_kd=1.0,
        speed_window=10,
        brake_speed=0.4,
        brake_ratio=1.1,
        clip_delta=0.25,
        max_throttle=0.75,
    ):
        self.temporal_hz = float(temporal_hz)
        if self.temporal_hz <= 0:
            raise ValueError("temporal_hz must be positive")
        self.turn_controller = PID(
            turn_kp, turn_ki, turn_kd, turn_window
        )
        self.speed_controller = PID(
            speed_kp, speed_ki, speed_kd, speed_window
        )
        self.brake_speed = float(brake_speed)
        self.brake_ratio = float(brake_ratio)
        self.clip_delta = float(clip_delta)
        self.max_throttle = float(max_throttle)

    @staticmethod
    def _validate_plan(name, plan):
        plan = np.asarray(plan, dtype=np.float64).reshape(-1, 2)
        if plan.shape[0] < 2:
            raise ValueError(f"{name} requires at least two points, got {plan.shape}")
        if not np.isfinite(plan).all():
            raise ValueError(f"{name} contains non-finite values")
        return plan

    @staticmethod
    def _normalized_steering_angle(vector):
        lateral, forward = float(vector[0]), float(vector[1])
        return float(
            np.degrees(np.pi / 2.0 - np.arctan2(forward, lateral)) / 90.0
        )

    def control(self, temporal_plan, spatial_plan, speed):
        temporal = self._validate_plan("temporal_plan", temporal_plan)
        spatial = self._validate_plan("spatial_plan", spatial_plan)
        speed = float(speed)
        if not np.isfinite(speed):
            raise ValueError(f"speed must be finite, got {speed}")

        temporal_segments = temporal[1:] - temporal[:-1]
        desired_speed = float(
            np.linalg.norm(temporal_segments, axis=1).mean() * self.temporal_hz
        )

        aim_distance = max(speed, 0.0)
        spatial_midpoints = (spatial[1:] + spatial[:-1]) / 2.0
        midpoint_distances = np.linalg.norm(spatial_midpoints, axis=1)
        aim_index = int(np.argmin(np.abs(midpoint_distances - aim_distance)))
        aim = spatial[aim_index]
        final_direction = spatial[-1] - spatial[-2]

        angle = self._normalized_steering_angle(aim)
        angle_last = self._normalized_steering_angle(final_direction)
        steer = float(np.clip(self.turn_controller.step(angle), -1.0, 1.0))

        speed_ratio = speed / max(desired_speed, 1e-6)
        brake = bool(
            desired_speed < self.brake_speed or speed_ratio > self.brake_ratio
        )
        delta = float(np.clip(desired_speed - speed, 0.0, self.clip_delta))
        throttle = float(
            np.clip(self.speed_controller.step(delta), 0.0, self.max_throttle)
        )
        if brake:
            throttle = 0.0

        metadata = {
            "speed": speed,
            "desired_speed": desired_speed,
            "steer": steer,
            "throttle": throttle,
            "brake": float(brake),
            "aim": aim.tolist(),
            "aim_index": aim_index,
            "aim_distance": aim_distance,
            "angle": angle,
            "angle_last": angle_last,
            "delta": delta,
            "temporal_plan": temporal.tolist(),
            "spatial_plan": spatial.tolist(),
        }
        return steer, throttle, float(brake), metadata

