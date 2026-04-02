from collections import deque
import numpy as np


class PID(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20, anti_windup=False,
                 integral_clamp=None):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._anti_windup  = anti_windup
        self._integral_clamp = integral_clamp  # v14: soft saturation clamp

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        # anti_windup: clear integral on sign flip to prevent accumulated bias
        if self._anti_windup and len(self._window) >= 1:
            last = self._window[-1]
            if last != 0 and error * last < 0:
                self._window = deque(
                    [0 for _ in range(self._window.maxlen)], maxlen=self._window.maxlen)

        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral   = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral   = 0.0
            derivative = 0.0

        # v14: integral clamp — soft saturation (alternative to hard zero-clear)
        if self._integral_clamp is not None:
            integral = np.clip(integral, -self._integral_clamp, self._integral_clamp)

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class PIDController(object):

    def __init__(self,
                 # ── PID gains (same defaults as vanilla) ─────────────────────
                 turn_KP=0.75, turn_KI=0.75, turn_KD=0.3, turn_n=40,
                 speed_KP=5.0, speed_KI=0.5, speed_KD=1.0, speed_n=40,
                 max_throttle=0.75, brake_speed=0.4, brake_ratio=1.1,
                 clip_delta=0.25, aim_dist=4.0, angle_thresh=0.3, dist_thresh=10,
                 # ── baseline toggle: match vanilla (False) vs production (True) ──
                 anti_windup=True,
                 # ── vanilla angle selection: use target when it has smaller angle ──
                 # True  = vanilla logic (use angle_target if |angle_target|<|angle|)
                 # False = always use predicted waypoint angle (production default)
                 use_target_to_aim=False,
                 # ── v0-v8 ablation params ─────────────────────────────────────
                 aim_dist_low=2.0, aim_dist_high=8.0,
                 max_steer_delta=None,
                 near_speed=False,
                 steer_deadzone=0.0,
                 # ── v9: curvature-adaptive aim dist ───────────────────────────
                 curvature_adapt=False,
                 curvature_lat_thresh=4.0,
                 curvature_min_factor=0.3,
                 # ── v10: speed-adaptive steer gain ────────────────────────────
                 speed_steer_alpha=0.0,
                 # ── v11: waypoint EMA smoothing (1.0 = off) ───────────────────
                 wp_ema_alpha=1.0,
                 # ── v12: predictive braking ────────────────────────────────────
                 predictive_brake=False,
                 predict_lat_thresh=1.5,
                 # ── v13: coast zone ────────────────────────────────────────────
                 coast_zone=0.0,
                 # ── v14: turn integral clamp (None = disabled) ─────────────────
                 turn_integral_clamp=None,
                 # ── v15: steering feedforward ──────────────────────────────────
                 # Adds angle_last * gain directly to steer output.
                 # Reduces entry-turn lag (Cases 1, 25).
                 steer_ff_gain=0.0,
                 # ── v16: asymmetric brake KP ───────────────────────────────────
                 # Scales down effective brake_ratio when decelerating, triggering
                 # braking earlier/harder (Cases 10, 11, 13, 18).
                 brake_kp_ratio=1.0,
                 # ── v17: aim time gap ──────────────────────────────────────────
                 # dynamic_aim_dist = clip(speed * aim_time_gap, low, high)
                 # 0.8 = shorter lookahead (turns), 1.5 = longer (highway).
                 aim_time_gap=1.0,
                 # ── v18: lateral acceleration limit ───────────────────────────
                 # Caps |angle_final| ≤ max_lat_accel / speed.
                 # Prevents high-speed tyre-slip (Cases 7, 21, 22).
                 # 0.0 = disabled.
                 max_lat_accel=0.0,
                 # ── v19: throttle rate limiter ─────────────────────────────────
                 # Limits |throttle - prev_throttle| per step.
                 # Prevents snap-throttle → rear-end collisions (Cases 1, 10, 23).
                 # None = disabled.
                 max_throttle_delta=None):
        """
        anti_windup       : expose vanilla vs production baseline
        use_target_to_aim : vanilla angle selection (use angle_target when |angle_target|<|angle|)
        steer_ff_gain     : feedforward from trailing-wp direction onto steer
        brake_kp_ratio    : effective_brake_ratio = brake_ratio / brake_kp_ratio
        aim_time_gap      : replaces hardcoded 1.0 in speed * 1.0
        max_lat_accel     : cap |angle_final| ≤ max_lat_accel / max(speed, 0.5)
        max_throttle_delta: throttle rate limiter (same idea as max_steer_delta)
        """
        self.turn_controller  = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n,
                                    anti_windup=anti_windup,
                                    integral_clamp=turn_integral_clamp)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)

        self.max_throttle  = max_throttle
        self.brake_speed   = brake_speed
        self.brake_ratio   = brake_ratio
        self.clip_delta    = clip_delta
        self.aim_dist      = aim_dist
        self.angle_thresh  = angle_thresh
        self.dist_thresh   = dist_thresh
        # v0-v8
        self.aim_dist_low       = aim_dist_low
        self.aim_dist_high      = aim_dist_high
        self.max_steer_delta    = max_steer_delta
        self.near_speed         = near_speed
        self.steer_deadzone     = steer_deadzone
        # v9
        self.curvature_adapt      = curvature_adapt
        self.curvature_lat_thresh = curvature_lat_thresh
        self.curvature_min_factor = curvature_min_factor
        # v10
        self.speed_steer_alpha = speed_steer_alpha
        # v11
        self.wp_ema_alpha = wp_ema_alpha
        # v12
        self.predictive_brake   = predictive_brake
        self.predict_lat_thresh = predict_lat_thresh
        # v13
        self.coast_zone = coast_zone
        # v15
        self.steer_ff_gain = steer_ff_gain
        # v16
        self.brake_kp_ratio = brake_kp_ratio
        # v17
        self.aim_time_gap = aim_time_gap
        # v18
        self.max_lat_accel = max_lat_accel
        # v19
        self.max_throttle_delta = max_throttle_delta
        # vanilla angle selection
        self.use_target_to_aim  = use_target_to_aim
        # state
        self._last_steer    = 0.0
        self._last_throttle = 0.0
        self._wp_ema        = None   # v11 EMA state

    def control_pid(self, waypoints, speed, target):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (array): predicted future positions relative to ego
            speed     (array): current speedometer reading (m/s)
            target    (array): route target point relative to ego
        '''

        # ── v11: waypoint EMA smoothing ───────────────────────────────────────
        if self.wp_ema_alpha < 1.0:
            wp_f = waypoints.astype(np.float64)
            if self._wp_ema is None:
                self._wp_ema = wp_f.copy()
            else:
                self._wp_ema = self.wp_ema_alpha * wp_f + (1.0 - self.wp_ema_alpha) * self._wp_ema
            waypoints = self._wp_ema

        num_pairs    = len(waypoints) - 1
        best_norm    = 1e5
        desired_speed = 0.0
        aim          = waypoints[0]

        # ── v9: curvature-adaptive aim distance ───────────────────────────────
        if self.curvature_adapt:
            max_lateral    = max(abs(float(waypoints[i][1])) for i in range(min(3, len(waypoints))))
            curvature_factor = float(np.clip(
                1.0 - max_lateral / self.curvature_lat_thresh,
                self.curvature_min_factor, 1.0))
        else:
            curvature_factor = 1.0

        # ── v17: aim time gap (replaces hard-coded 1.0) ───────────────────────
        dynamic_aim_dist = float(np.clip(
            float(speed) * self.aim_time_gap * curvature_factor,
            self.aim_dist_low, self.aim_dist_high))

        # ── speed estimation ──────────────────────────────────────────────────
        if self.near_speed:
            desired_speed = (0.75 * np.linalg.norm(waypoints[0]) * 2.0 +
                             0.25 * np.linalg.norm(waypoints[1] - waypoints[0]) * 2.0)

        for i in range(num_pairs):
            if not self.near_speed:
                desired_speed += np.linalg.norm(waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(dynamic_aim_dist - best_norm) > abs(dynamic_aim_dist - norm):
                aim       = waypoints[i]
                best_norm = norm

        # ── v12: predictive braking ────────────────────────────────────────────
        if self.predictive_brake and len(waypoints) >= 3:
            upcoming_lateral = abs(float(waypoints[2][1]))
            if upcoming_lateral > self.predict_lat_thresh:
                desired_speed = desired_speed * float(np.clip(
                    self.predict_lat_thresh / upcoming_lateral, 0.5, 1.0))

        aim_last = waypoints[-1] - waypoints[-2]

        angle        = np.degrees(np.pi / 2 - np.arctan2(aim[1],      aim[0]))      / 90
        angle_last   = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1],   target[0]))   / 90

        # steer dead-zone
        if self.steer_deadzone > 0.0 and abs(aim[1]) <= self.steer_deadzone:
            angle = 0.0

        # ── vanilla angle selection (use_target_to_aim) ───────────────────────
        # Vanilla picks angle_target when it's smaller (removes outlier predictions)
        # or when route command diverges sharply from the waypoint heading.
        if self.use_target_to_aim:
            use_target = np.abs(angle_target) < np.abs(angle)
            use_target = use_target or (
                np.abs(angle_target - angle_last) > self.angle_thresh
                and target[1] < self.dist_thresh)
            angle_final = float(angle_target) if use_target else float(angle)
        else:
            angle_final = float(angle)

        # ── v10: speed-adaptive steer gain ────────────────────────────────────
        if self.speed_steer_alpha > 0.0:
            angle_final = angle_final / (1.0 + self.speed_steer_alpha * float(speed))

        # ── v18: lateral acceleration hard limit ──────────────────────────────
        # cap: a_lat ≤ max_lat_accel  →  |steer| ≤ max_lat_accel / v
        if self.max_lat_accel > 0.0:
            max_angle = self.max_lat_accel / max(float(speed), 0.5)
            angle_final = float(np.clip(angle_final, -max_angle, max_angle))

        steer = self.turn_controller.step(angle_final)

        # ── v15: steering feedforward ─────────────────────────────────────────
        # angle_last represents the far-waypoint heading = geometric intent.
        # Adding it directly bypasses integral lag on turn entry.
        if self.steer_ff_gain > 0.0:
            steer = steer + self.steer_ff_gain * float(angle_last)

        steer = float(np.clip(steer, -1.0, 1.0))

        # steer rate limiter (v4)
        if self.max_steer_delta is not None:
            steer = float(np.clip(steer,
                                  self._last_steer - self.max_steer_delta,
                                  self._last_steer + self.max_steer_delta))
        self._last_steer = steer

        # always compute delta for metadata
        delta     = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        speed_err = float(desired_speed) - float(speed)

        # ── v16: asymmetric brake KP ───────────────────────────────────────────
        # brake_kp_ratio > 1 → effective_brake_ratio tighter → brake triggers earlier
        effective_brake_ratio = self.brake_ratio / max(self.brake_kp_ratio, 1e-3)

        # ── v13: coast zone / normal throttle-brake ────────────────────────────
        if self.coast_zone > 0.0 and abs(speed_err) < self.coast_zone:
            throttle = 0.0
            brake    = False
        else:
            brake    = (desired_speed < self.brake_speed or
                        float(speed) / max(float(desired_speed), 1e-3) > effective_brake_ratio)
            throttle = float(np.clip(self.speed_controller.step(delta), 0.0, self.max_throttle))
            throttle = throttle if not brake else 0.0

        # ── v19: throttle rate limiter ────────────────────────────────────────
        if self.max_throttle_delta is not None:
            throttle = float(np.clip(throttle,
                                     self._last_throttle - self.max_throttle_delta,
                                     self._last_throttle + self.max_throttle_delta))
        self._last_throttle = throttle

        metadata = {
            'speed':            float(speed.astype(np.float64)),
            'steer':            steer,
            'throttle':         throttle,
            'brake':            float(brake),
            'wp_4':             tuple(waypoints[3].astype(np.float64)),
            'wp_3':             tuple(waypoints[2].astype(np.float64)),
            'wp_2':             tuple(waypoints[1].astype(np.float64)),
            'wp_1':             tuple(waypoints[0].astype(np.float64)),
            'aim':              tuple(aim.astype(np.float64)),
            'target':           tuple(target.astype(np.float64)),
            'desired_speed':    float(desired_speed),
            'angle':            float(angle),
            'angle_last':       float(angle_last),
            'angle_target':     float(angle_target),
            'angle_final':      angle_final,
            'delta':            float(delta.astype(np.float64)),
            'curvature_factor': curvature_factor,
            'dynamic_aim_dist': dynamic_aim_dist,
        }

        return steer, throttle, brake, metadata
