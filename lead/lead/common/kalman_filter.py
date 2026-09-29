import math
from collections import deque

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
from filterpy.kalman import UnscentedKalmanFilter as UKF
from numpy import dot, isscalar, outer, zeros

import lead.common.common_utils as common_utils
from lead.expert.config_expert import ExpertConfig
from lead.training.config_training import TrainingConfig


class KalmanFilter:
    """Unscented Kalman Filter for less noisy GPS localization."""

    @beartype
    def __init__(self, config: TrainingConfig | ExpertConfig):
        """Constructor.

        Args:
            config: Object containing the configuration parameters.
        """
        self.config = config
        self.points = MerweScaledSigmaPoints(
            n=4,
            alpha=0.00001,
            beta=2,
            kappa=0,
            subtract=self._residual_state_x,
        )

        self.ukf = UKF(
            dim_x=4,
            dim_z=4,
            fx=self._bicycle_model_forward,
            hx=self._measurement_function_hx,
            dt=self.config.carla_frame_rate,
            points=self.points,
            x_mean_fn=self._state_mean,
            z_mean_fn=self._measurement_mean,
            residual_x=self._residual_state_x,
            residual_z=self._residual_measurement_h,
        )

        # State noise, same as measurement because we
        # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
        # Used to set the filter state equal the first measurement
        self.filter_initialized = False

        maxlen = self.config.ego_num_temporal_data_points_saved + 1

        self.history_x = deque(maxlen=maxlen)  # filtered states
        self.history_P = deque(maxlen=maxlen)  # covariances
        self.history_steers = deque(maxlen=maxlen)  # steering commands
        self.history_throttles = deque(maxlen=maxlen)  # throttle commands
        self.history_brakes = deque(maxlen=maxlen)  # brake commands

        # Scaling factors to avoid working with large numbers
        self.start_x = None
        self.start_y = None

    @beartype
    def step(
        self,
        noisy_position: npt.NDArray,
        compass: float,
        speed: float,
        control: carla.VehicleControl,
    ) -> npt.NDArray[np.floating]:
        """Performs one iteration of predict and update of the UKF.

        Args:
            noisy_position: Carla coordinates in meters.
            compass: Unbounded compass angle in radians w.r.t world frame.
            speed: Speed in m/s.
            control: Object containing the last control command.
        Returns:
            npt.NDArray[np.floating]: The filtered state [x, y, angle, speed].
        """
        if self.start_x is None:
            self.start_x = noisy_position[0]
            self.start_y = noisy_position[1]

        # Create scale state
        z = np.array(
            [
                noisy_position[0] - self.start_x,
                noisy_position[1] - self.start_y,
                common_utils.normalize_angle(compass),
                speed,
            ],
        )

        if not self.filter_initialized:
            # apply ukf only to x and y coordinates, append z coordinate afterwards
            self.ukf.x = z
            self.filter_initialized = True
            self.history_steers.append(0.0)
            self.history_throttles.append(0.0)
            self.history_brakes.append(0.0)

        self.ukf.predict(
            steer=control.steer,
            throttle=control.throttle,
            brake=control.brake,
        )
        self.ukf.update(z)

        prediction = self.ukf.x.copy()

        # Rescale back to original coordinates
        prediction[0] += self.start_x
        prediction[1] += self.start_y

        self.history_x.append(prediction)
        self.history_P.append(self.ukf.P.copy())
        self.history_steers.append(control.steer)
        self.history_throttles.append(control.throttle)
        self.history_brakes.append(control.brake)

        return prediction

    @beartype
    def smooth(self) -> jt.Float[npt.NDArray, "N 4"]:
        """Applies the RTS smoother to the stored history of states and returns the smoothed states.

        Returns:
            The smoothed states [x, y, angle, speed].
        """
        history_x = np.array(self.history_x)

        # Scale to origin
        history_x[:, 0] -= self.start_x
        history_x[:, 1] -= self.start_y

        (xs, Ps, Ks) = rts_smoother(
            self=self.ukf,
            Xs=history_x,
            Ps=np.array(self.history_P),
            steers=np.array(self.history_steers),
            throttles=np.array(self.history_throttles),
            brakes=np.array(self.history_brakes),
        )

        # Rescale back to original coordinates
        xs[:, 0] += self.start_x
        xs[:, 1] += self.start_y

        return np.array(xs)

    @beartype
    def _bicycle_model_forward(
        self,
        x: jt.Float[npt.NDArray, " 4"],
        dt: float,
        steer: float,
        throttle: float,
        brake: float,
    ) -> jt.Float[npt.NDArray, " 4"]:
        """Leaderboard 1.0's kinematic bicycle model. Numbers are the tuned parameters from World on Rails.

        Args:
            x: State vector [x, y, yaw, speed].
            dt: Timestep in seconds.
            steer: Last step's steering command.
            throttle: Last step's throttle command.
            brake: Last step's brake command.

        Returns:
            npt.NDArray[np.floating]: The next predicted state [x, y, yaw, speed].
        """
        front_wb = -0.090769015
        rear_wb = 1.4178275

        steer_gain = 0.36848336
        brake_accel = -4.952399
        throt_accel = 0.5633837

        locs_0 = x[0]
        locs_1 = x[1]
        yaw = x[2]
        speed = x[3]

        if brake:
            accel = brake_accel
        else:
            accel = throt_accel * throttle

        wheel = steer_gain * steer

        beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
        next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
        next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
        next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
        next_speed = speed + accel * dt
        next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

        next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

        return next_state_x

    @beartype
    def _measurement_function_hx(
        self,
        vehicle_state: jt.Float[npt.NDArray, " 4"],
    ) -> jt.Float[npt.NDArray, " 4"]:
        """
        Identity measurement function.

        Args:
            vehicle_state: Vehicle state variable containing an internal state of the vehicle from the filter

        Returns:
            npt.NDArray: Output.
        """
        return vehicle_state

    @beartype
    def _state_mean(
        self,
        state: jt.Float[npt.NDArray, "N 4"],
        wm: npt.ArrayLike,
    ) -> jt.Float[npt.NDArray, " 4"]:
        """Averaging function.

        Args:
            state: States to be averaged.
            wm: Weights for the mean.
        Returns:
            The averaged state.
        Note: We use the arctan of the average of sin and cos of the angle to calculate the average of orientations.
        """
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
        sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
        x[0] = np.sum(np.dot(state[:, 0], wm))
        x[1] = np.sum(np.dot(state[:, 1], wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(state[:, 3], wm))

        return x

    @beartype
    def _measurement_mean(
        self,
        state: jt.Float[npt.NDArray, "N 4"],
        wm: npt.ArrayLike,
    ) -> jt.Float[npt.NDArray, " 4"]:
        """Averaging function.

        Args:
            state: States to be averaged.
            wm: Weights for the mean.
        Returns:
            The averaged state.
        Note: We use the arctan of the average of sin and cos of the angle to calculate the average of orientations.
        """
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
        sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
        x[0] = np.sum(np.dot(state[:, 0], wm))
        x[1] = np.sum(np.dot(state[:, 1], wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(state[:, 3], wm))

        return x

    @beartype
    def _residual_state_x(
        self,
        a: npt.NDArray,
        b: npt.NDArray,
    ) -> npt.NDArray[np.floating]:
        """Residual function

        Args:
            a: Predicted state.
            b: State to be subtracted from predicted state.

        Returns:
            The residual.
        """
        y = a - b
        y[2] = common_utils.normalize_angle(y[2])
        return y

    @beartype
    def _residual_measurement_h(
        self,
        a: npt.NDArray,
        b: npt.NDArray,
    ) -> npt.NDArray[np.floating]:
        """Residual function

        Args:
            a: Predicted state.
            b: State to be subtracted from predicted state.

        Returns:
            The residual.
        """
        y = a - b
        y[2] = common_utils.normalize_angle(y[2])
        return y


def rts_smoother(self, Xs, Ps, steers, throttles, brakes, Qs=None, dts=None, UT=None):
    """Copy-pasted from filterpy. Adapted to include control inputs."""
    # pylint: disable=too-many-locals, too-many-arguments

    if len(Xs) != len(Ps):
        raise ValueError("Xs and Ps must have the same length")

    n, dim_x = Xs.shape

    if dts is None:
        dts = [self._dt] * n
    elif isscalar(dts):
        dts = [dts] * n

    if Qs is None:
        Qs = [self.Q] * n

    if UT is None:
        UT = unscented_transform

    # smoother gain
    Ks = zeros((n, dim_x, dim_x))

    num_sigmas = self._num_sigmas

    xs, ps = Xs.copy(), Ps.copy()
    sigmas_f = zeros((num_sigmas, dim_x))

    for k in reversed(range(n - 1)):
        # create sigma points from state estimate, pass through state func
        sigmas = self.points_fn.sigma_points(xs[k], ps[k])
        for i in range(num_sigmas):
            sigmas_f[i] = self.fx(sigmas[i], dts[k], steers[k], throttles[k], brakes[k])

        xb, Pb = UT(sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x)

        # compute cross variance
        Pxb = 0
        for i in range(num_sigmas):
            y = self.residual_x(sigmas_f[i], xb)
            z = self.residual_x(sigmas[i], Xs[k])
            Pxb += self.Wc[i] * outer(z, y)

        # compute gain
        K = dot(Pxb, self.inv(Pb))

        # update the smoothed estimates
        xs[k] += dot(K, self.residual_x(xs[k + 1], xb))
        ps[k] += dot(K, ps[k + 1] - Pb).dot(K.T)
        Ks[k] = K

    return (xs, ps, Ks)
