import jaxtyping as jt
import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from lead.common.config_base import BaseConfig


class KinematicBicycleModel:
    """Kinematic bicycle model describing the motion of a car given its state and action.
    Tuned parameters are taken from World on Rails."""

    @beartype
    def __init__(self, config: BaseConfig) -> None:
        """Initialize the kinematic bicycle model with configuration parameters.

        Args:
            config: Object of the config for hyperparameters.
        """
        self.config = config

        self.time_step = self.config.time_step
        self.front_wheel_base = self.config.front_wheel_base
        self.rear_wheel_base = self.config.rear_wheel_base
        self.steering_gain = self.config.steering_gain
        self.brake_acceleration = self.config.brake_acceleration
        self.throttle_acceleration = self.config.throttle_acceleration
        self.throttle_values = self.config.throttle_values
        self.brake_values = self.config.brake_values
        self.throttle_threshold_during_forecasting = (
            self.config.throttle_threshold_during_forecasting
        )

    @beartype
    def forecast_other_vehicles(
        self,
        locations: jt.Float[NDArray, "... 3"],
        headings: jt.Float[NDArray, "..."],
        speeds: jt.Float[NDArray, "..."],
        actions: jt.Float[NDArray, "... 3"],
    ) -> tuple[
        jt.Float[NDArray, "... 3"],
        jt.Float[NDArray, "..."],
        jt.Float[NDArray, "..."],
    ]:
        """Forecast the future states of other vehicles based on their current states and actions.

        Args:
            locations: Array of (x, y, z) coordinates representing the locations of other vehicles.
            headings: Array of heading angles (in radians) for other vehicles.
            speeds: Array of speeds (in m/s) for other vehicles.
            actions: Array of actions (steer, throttle, brake) for other vehicles.

        Returns:
            A tuple containing the forecasted locations, headings, and speeds for other vehicles.
        """
        steers, throttles = actions[:, 0], actions[:, 1]
        with np.errstate(invalid="ignore"):
            brakes = actions[:, 2].astype(np.uint8)
        wheel_angles = self.steering_gain * steers
        slip_angles = np.arctan(
            self.rear_wheel_base
            / (self.front_wheel_base + self.rear_wheel_base)
            * np.tan(wheel_angles),
        )

        next_x = (
            locations[:, 0] + speeds * np.cos(headings + slip_angles) * self.time_step
        )
        next_y = (
            locations[:, 1] + speeds * np.sin(headings + slip_angles) * self.time_step
        )
        next_headings = (
            headings
            + speeds / self.rear_wheel_base * np.sin(slip_angles) * self.time_step
        )

        next_speeds = speeds + self.time_step * np.where(
            brakes,
            self.brake_acceleration,
            throttles * self.throttle_acceleration,
        )
        next_speeds = np.maximum(0.0, next_speeds)

        next_locations = np.column_stack([next_x, next_y, locations[:, 2]])

        return next_locations, next_headings, next_speeds

    @beartype
    def forecast_ego_vehicle(
        self,
        location: jt.Float[NDArray, "3"],
        heading: float,
        speed: float,
        action: jt.Float[NDArray, "3"],
    ) -> tuple[jt.Float[NDArray, "3"], float, float]:
        """Forecast the future state of the ego vehicle based on its current state and action.

        Args:
            location: Array of (x, y, z) coordinates representing the location of the ego vehicle.
            heading: Current heading angle (in radians) of the ego vehicle.
            speed: Current speed (in m/s) of the ego vehicle.
            action: Action (steer, throttle, brake) for the ego vehicle.

        Returns:
            A tuple containing the forecasted location, heading, and speed for the ego vehicle.
        """
        steer, throttle, brake = action
        steer = float(steer)
        throttle = float(throttle)
        brake = bool(brake)
        speed = float(speed)
        wheel_angle = self.steering_gain * steer
        slip_angle = np.arctan(
            self.rear_wheel_base
            / (self.front_wheel_base + self.rear_wheel_base)
            * np.tan(wheel_angle),
        )

        next_x = location[0] + speed * np.cos(heading + slip_angle) * self.time_step
        next_y = location[1] + speed * np.sin(heading + slip_angle) * self.time_step
        next_heading = (
            heading + speed / self.rear_wheel_base * np.sin(slip_angle) * self.time_step
        )

        # We use different polynomial models for estimating the speed depending on whether
        # the ego vehicle brakes or not.
        if brake:
            speed_kph = speed * 3.6
            features = speed_kph ** np.arange(1, 8)
            next_speed_kph = features @ self.brake_values
            next_speed = next_speed_kph / 3.6
        else:
            throttle = np.clip(throttle, 0.0, 1.0)

            # For a throttle value < 0.3 the car doesn't really accelerate and the
            # polynomial model below doesn't hold anymore.
            if throttle < self.throttle_threshold_during_forecasting:
                # If the throttle is low, the car does not accelerate, so we
                # assume constant speed.
                next_speed = speed
            else:
                # For a throttle value > 0.3 the car accelerates and we can
                # use the polynomial model below.
                speed_kph = speed * 3.6
                features = np.array(
                    [
                        speed_kph,
                        speed_kph**2,
                        throttle,
                        throttle**2,
                        speed_kph * throttle,
                        speed_kph * throttle**2,
                        speed_kph**2 * throttle,
                        speed_kph**2 * throttle**2,
                    ],
                ).T

                next_speed_kph = features @ self.throttle_values
                next_speed = next_speed_kph / 3.6

        next_speed = np.maximum(0.0, next_speed)
        next_location = np.array([float(next_x), float(next_y), float(location[2])])

        return next_location, next_heading, next_speed
