from collections import deque
from copy import deepcopy

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype

import lead.common.common_utils as common_utils
from lead.expert.config_expert import ExpertConfig
from lead.inference.config_closed_loop import ClosedLoopConfig


class PIDController:
    """Classical PID controller for general control applications.

    Implements proportional-integral-derivative control with a sliding
    window for error history management.
    """

    @beartype
    def __init__(
        self,
        k_p: float = 1.0,
        k_i: float = 0.0,
        k_d: float = 0.0,
        n: int = 20,
    ) -> None:
        """Initialize the PID controller with gain parameters.

        Args:
            k_p: Proportional gain coefficient.
            k_i: Integral gain coefficient.
            k_d: Derivative gain coefficient.
            n: Size of the sliding window for error history.
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self._saved_window = deque([0 for _ in range(n)], maxlen=n)
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def reset_error_integral(self) -> None:
        """Reset the error integral by clearing the sliding window."""
        self._window = deque(len(self._window) * [0])

    def step(self, error: float) -> float:
        """Compute the PID control output for the given error.

        Args:
            error: Current error value.

        Returns:
            PID control output value.
        """
        self._window.append(error)
        if len(self._window) >= 2:
            integral = sum(self._window) / len(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self.k_p * error + self.k_i * integral + self.k_d * derivative

    def save(self) -> None:
        """Save the current state of the controller."""
        self._saved_window = deepcopy(self._window)

    def load(self) -> None:
        """Restore the previously saved controller state."""
        self._window = self._saved_window


class LateralPIDController:
    """PID controller for lateral (steering) control of the vehicle.

    Implements lateral control using PID with adaptive lookahead distance
    based on vehicle speed.
    """

    @beartype
    def __init__(self, config: ClosedLoopConfig) -> None:
        """Initialize the lateral PID controller.

        Args:
            config: Closed-loop configuration containing PID parameters.
        """
        self.k_p = config.lateral_k_p
        self.k_d = config.lateral_k_d
        self.k_i = config.lateral_k_i
        self.speed_scale = config.lateral_speed_scale
        self.speed_offset = config.lateral_speed_offset
        self.default_lookahead = config.lateral_default_lookahead
        self.speed_threshold = config.lateral_speed_threshold
        self.n = config.lateral_n

        self._saved_window = []
        self._window = []
        self.config = config

    @jt.jaxtyped(typechecker=beartype)
    def step(
        self,
        route: jt.Float[npt.NDArray, "n_points 2"],
        current_speed: float,
        ego_vehicle_location: float,
        ego_vehicle_rotation: float,
        sensor_agent_steer_correction: bool = False,
    ) -> float:
        """Compute steering control output based on route following.

        Args:
            route: Array of route waypoints.
            current_speed: Current vehicle speed in m/s.
            ego_vehicle_location: Current vehicle position.
            ego_vehicle_rotation: Current vehicle heading in radians.
            sensor_agent_steer_correction: Whether to apply correction for sensor-agent misalignment.
        Returns:
            Steering angle in range [-1, 1].
        """
        current_speed = current_speed * 3.6
        # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
        n_lookahead = (
            np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10
        )  # range [2.4, 10.5]
        n_lookahead = n_lookahead - 2  # range [0.4, 8.5]
        n_lookahead = int(
            min(n_lookahead, route.shape[0] - 1),
        )  # range [0, 8] - but 0 and 1 are never used because n_lookahead is overwritten below

        n_lookahead = min(n_lookahead, len(route) - 1)
        curvature = common_utils.waypoints_curvature(route)

        if sensor_agent_steer_correction:
            n_lookahead += np.clip(
                int(curvature * self.config.sensor_agent_steer_correction_param),
                0,
                2,
            )

        desired_heading_vec = route[n_lookahead] - ego_vehicle_location

        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = (yaw_path - ego_vehicle_rotation) % (2 * np.pi)
        heading_error = (
            heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        )

        # the scaling doesn't deserve any specific purpose but is a leftover from a previous less efficient implementation,
        # on which we optimized the parameters
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(heading_error)
        self._window = self._window[-self.n :]

        derivative = (
            0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        )
        integral = np.mean(self._window)

        steering = np.clip(
            self.k_p * heading_error + self.k_d * derivative + self.k_i * integral,
            -1.0,
            1.0,
        ).item()

        return round(float(np.clip(steering, -1.0, 1.0)), 3)

    def save(self) -> None:
        """Save the current state of the controller."""
        self._saved_window = self._window.copy()

    def load(self) -> None:
        """Restore the previously saved controller state."""
        self._window = self._saved_window.copy()


class ExpertLateralPIDController:
    """PID controller for lateral control specifically tuned for expert.

    Implements adaptive lateral control with speed-dependent lookahead
    distance and PID gains optimized for the expert driving model.
    """

    @beartype
    def __init__(self, config: ExpertConfig) -> None:
        """Initialize the expert lateral PID controller.

        Args:
            config: expert configuration containing PID parameters.
        """
        self.config = config

        self.lateral_pid_kp = self.config.lateral_pid_kp
        self.lateral_pid_kd = self.config.lateral_pid_kd
        self.lateral_pid_ki = self.config.lateral_pid_ki

        self.lateral_pid_speed_scale = self.config.lateral_pid_speed_scale
        self.lateral_pid_speed_offset = self.config.lateral_pid_speed_offset
        self.lateral_pid_default_lookahead = self.config.lateral_pid_default_lookahead
        self.lateral_pid_speed_threshold = self.config.lateral_pid_speed_threshold

        self.lateral_pid_window_size = self.config.lateral_pid_window_size
        self.lateral_pid_minimum_lookahead_distance = (
            self.config.lateral_pid_minimum_lookahead_distance
        )
        self.lateral_pid_maximum_lookahead_distance = (
            self.config.lateral_pid_maximum_lookahead_distance
        )

        # The following lists are used as deques
        self.error_history = []  # Sliding window to store past errors
        self.saved_error_history = []  # Saved error history for state loading

    def step(
        self,
        route_points: jt.Float[npt.NDArray, "n 2"],
        current_speed: float,
        vehicle_position: jt.Float[npt.NDArray, " 2"],
        vehicle_heading: float,
        inference_mode: bool = False,
    ) -> float:
        """Compute steering angle based on route following.

        Args:
            route_points: Array of (x, y) coordinates representing the route.
            current_speed: Current speed of the vehicle in m/s.
            vehicle_position: Array of (x, y) coordinates representing vehicle position.
            vehicle_heading: Current heading angle of the vehicle in radians.
            inference_mode: Controls whether TF or expert executes this method.

        Returns:
            Computed steering angle in the range [-1.0, 1.0].
        """
        current_speed_kph = current_speed * 3.6  # Convert speed from m/s to km/h

        # Compute the lookahead distance based on the current speed
        # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
        if inference_mode:
            lookahead_distance = (
                self.lateral_pid_speed_scale * current_speed
                + self.lateral_pid_speed_offset
            )
            lookahead_distance = (
                np.clip(
                    lookahead_distance,
                    self.lateral_pid_minimum_lookahead_distance,
                    self.lateral_pid_maximum_lookahead_distance,
                )
                / self.config.route_points
            )  # range [2.4, 10.5]
            lookahead_distance = lookahead_distance - 2  # range [0.4, 8.5]
        else:
            lookahead_distance = (
                self.lateral_pid_speed_scale * current_speed_kph
                + self.lateral_pid_speed_offset
            )
            lookahead_distance = np.clip(
                lookahead_distance,
                self.lateral_pid_minimum_lookahead_distance,
                self.lateral_pid_maximum_lookahead_distance,
            )

        lookahead_distance = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Calculate the desired heading vector from the lookahead point
        desired_heading_vec = route_points[lookahead_distance] - vehicle_position
        desired_heading_angle = np.arctan2(
            desired_heading_vec[1],
            desired_heading_vec[0],
        )

        # Calculate the heading error
        heading_error = (desired_heading_angle - vehicle_heading) % (2 * np.pi)
        heading_error = (
            heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        )

        # Scale the heading error (leftover from a previous implementation)
        heading_error = heading_error * 180.0 / np.pi / 90.0

        # Update the error history. Only use the last lateral_pid_window_size errors like in a deque.
        self.error_history.append(heading_error)
        self.error_history = self.error_history[-self.lateral_pid_window_size :]

        # Calculate the derivative and integral terms
        derivative = (
            0.0
            if len(self.error_history) == 1
            else self.error_history[-1] - self.error_history[-2]
        )
        integral = np.mean(self.error_history)

        # Compute the steering angle using the PID control law
        steering = np.clip(
            self.lateral_pid_kp * heading_error
            + self.lateral_pid_kd * derivative
            + self.lateral_pid_ki * integral,
            -1.0,
            1.0,
        ).item()

        return steering

    def save_state(self) -> None:
        """Save the current state of the controller by copying error history."""
        self.saved_error_history = self.error_history.copy()

    def load_state(self) -> None:
        """Load the previously saved state by restoring saved error history."""
        self.error_history = self.saved_error_history.copy()


class ExpertLongitudinalController:
    """Linear regression-based longitudinal controller for expert.

    Implements speed control using a linear regression model to determine
    optimal throttle and brake values based on speed error and current speed.
    """

    @beartype
    def __init__(self, config: ExpertConfig) -> None:
        """Initialize the longitudinal controller.

        Args:
            config: expert configuration containing regression parameters.
        """
        self.config = config
        self.minimum_target_speed = (
            self.config.longitudinal_linear_regression_minimum_target_speed
        )
        self.params = self.config.longitudinal_linear_regression_params
        self.maximum_acceleration = (
            self.config.longitudinal_linear_regression_maximum_acceleration
        )
        self.maximum_deceleration = (
            self.config.longitudinal_linear_regression_maximum_deceleration
        )

    def get_throttle_and_brake(
        self,
        hazard_brake: bool,
        target_speed: float,
        current_speed: float,
    ) -> tuple[float, bool]:
        """Get throttle and brake values using linear regression model.

        Args:
            hazard_brake: Flag indicating whether to apply hazard braking.
            target_speed: The desired target speed in m/s.
            current_speed: The current speed of the vehicle in m/s.

        Returns:
            A tuple containing the throttle and brake values.
        """
        if target_speed < 1e-5 or hazard_brake:
            return 0.0, True
        elif target_speed < self.minimum_target_speed:  # Avoid very small target speeds
            target_speed = self.minimum_target_speed

        current_speed = current_speed * 3.6
        target_speed = target_speed * 3.6
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1.0, False

        if current_speed / target_speed > params[-1] or hazard_brake:
            throttle, control_brake = 0.0, True
            return throttle, control_brake

        speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
        current_speed /= 100.0
        features = np.array(
            [
                current_speed,
                current_speed**2,
                100 * speed_error_cl,
                speed_error_cl**2,
                current_speed * speed_error_cl,
                current_speed**2 * speed_error_cl,
            ],
        )

        throttle, control_brake = np.clip(features @ params[:-1], 0.0, 1.0), False

        return throttle, control_brake

    def get_throttle_extrapolation(
        self,
        target_speed: float,
        current_speed: float,
    ) -> float:
        """Get throttle value for forecasting purposes.

        Computes throttle assuming no hazard brake condition, used for
        trajectory forecasting and planning.

        Args:
            target_speed: The desired target speed in m/s.
            current_speed: The current speed of the vehicle in m/s.

        Returns:
            The throttle value in range [0, 1].
        """
        current_speed = current_speed * 3.6  # Conversion to km/h
        target_speed = target_speed * 3.6  # Conversion to km/h
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1.0
        # Maximum deceleration -4.82 m/tick
        elif speed_error < self.maximum_deceleration:
            return 0.0

        throttle = 0.0
        # 0.1 to ensure small distances are overcome fast
        if target_speed < 0.1 or current_speed / target_speed > params[-1]:
            return throttle

        speed_error_cl = (
            np.clip(speed_error, 0.0, np.inf) / 100.0
        )  # The scaling is a leftover from the optimization
        current_speed /= 100.0  # The scaling is a leftover from the optimization
        features = np.array(
            [
                current_speed,
                current_speed**2,
                100 * speed_error_cl,
                speed_error_cl**2,
                current_speed * speed_error_cl,
                current_speed**2 * speed_error_cl,
            ],
        ).flatten()

        throttle = np.clip(features @ params[:-1], 0.0, 1.0)

        return throttle


@beartype
def get_throttle(
    brake: bool,
    target_speed: float,
    speed: float,
    config: ExpertConfig,
) -> tuple[float, bool]:
    """Compute throttle and brake values using expert longitudinal control.

    Standalone function for computing longitudinal control outputs using
    the expert linear regression approach.

    Args:
        brake: Whether hazard braking is active.
        target_speed: Desired target speed in m/s.
        speed: Current vehicle speed in m/s.
        config: expert configuration containing control parameters.

    Returns:
        A tuple containing:
            - Throttle value in range [0, 1]
            - Boolean indicating if braking should be applied
    """
    if target_speed < 1e-5 or brake:
        return 0.0, True
    elif target_speed < 1.0 / 3.6:  # to avoid very small target speeds
        target_speed = 1.0 / 3.6

    speed = speed * 3.6
    target_speed = target_speed * 3.6
    params = config.longitudinal_params
    speed_error = target_speed - speed

    # maximum acceleration 1.9 m/tick
    if speed_error > config.longitudinal_max_accelerations:
        return 1.0, False

    if speed / target_speed > params[-1] or brake:
        throttle, control_brake = 0.0, True
        return throttle, control_brake

    speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
    speed /= 100.0
    features = np.array(
        [
            speed,
            speed**2,
            100 * speed_error_cl,
            speed_error_cl**2,
            speed * speed_error_cl,
            speed**2 * speed_error_cl,
        ],
    ).squeeze()

    params = np.array(params[:-1])
    throttle, control_brake = np.clip(features @ params, 0.0, 1.0), False

    return float(throttle), control_brake
