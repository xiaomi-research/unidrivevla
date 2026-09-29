import logging
import math
import numbers
import typing
from numbers import Real

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from agents.navigation.global_route_planner import GlobalRoutePlanner
from beartype import beartype
from scipy.integrate import RK45
from scipy.optimize import linear_sum_assignment

from lead.common import constants
from lead.common.constants import (
    SEMANTIC_SEGMENTATION_CONVERTER,
    CameraPointCloudIndex,
    CarlaSemanticSegmentationClass,
)
from lead.common.pid_controller import PIDController
from lead.expert.config_expert import ExpertConfig

LOG = logging.getLogger(__name__)


class CarlaActorDummy:
    """
    Actor dummy structure used to simulate a CARLA actor for HD-Map perturbation
    """

    world = None
    bounding_box = None
    transform = None
    id = None

    def __init__(self, world, bounding_box, transform, id):  # pylint: disable=locally-disabled, redefined-builtin
        self.world = world
        self.bounding_box = bounding_box
        self.transform = transform
        self.id = id

    def get_world(self):
        return self.world

    def get_transform(self):
        return self.transform

    def get_bounding_box(self):
        return self.bounding_box


@beartype
def step_cached_property(func: typing.Callable) -> property:
    """Decorator to cache the result of a method based on the current step.
    This is useful for properties that are expensive to compute and
    should only be recalculated when the step changes.

    Args:
        func: Function to be decorated.

    Returns:
        A property that caches its value based on the step attribute of the instance.
    """
    cache_attr = f"_{func.__name__}_cache"
    step_attr = f"_{func.__name__}_step"

    def getter(self):
        if getattr(self, step_attr, None) != self.step:
            setattr(self, cache_attr, func(self))
            setattr(self, step_attr, self.step)
        return getattr(self, cache_attr)

    return property(getter)


@beartype
def cached_property_by(key_getter: typing.Callable):
    """Decorator to cache the result of a method based on a custom key.
    This is useful for properties that are expensive to compute and should only be
    recalculated when a specific attribute or value changes.

    Args:
        key_getter: Function that takes the instance and returns the cache key.

    Returns:
        A decorator that creates a property cached by the key_getter result.

    Example:
        @cached_property_by(lambda self: self.privileged_route_planner.route_index)
        def some_expensive_calculation(self):
            return expensive_computation()
    """

    def decorator(func: typing.Callable) -> property:
        cache_attr = f"_{func.__name__}_cache"
        key_attr = f"_{func.__name__}_cache_key"

        def getter(self):
            current_key = key_getter(self)
            if getattr(self, key_attr, None) != current_key:
                setattr(self, cache_attr, func(self))
                setattr(self, key_attr, current_key)
            return getattr(self, cache_attr)

        return property(getter)

    return decorator


@beartype
def get_horizontal_distance(actor1: carla.Actor, actor2: carla.Actor) -> float:
    """Get horizontal distance between two actors.

    Args:
        actor1: First actor.
        actor2: Second actor.

    Returns:
        float: Horizontal distance between the two actors.
    """
    location1, location2 = actor1.get_location(), actor2.get_location()

    # Compute the distance vector (ignoring the z-coordinate)
    diff_vector = carla.Vector3D(
        location1.x - location2.x,
        location1.y - location2.y,
        0,
    )

    norm: float = diff_vector.length()

    return norm


@beartype
def distance_location_to_route(route: npt.NDArray, location: npt.NDArray) -> float:
    """
    Project a location onto the closest point on a route.

    Args:
        route: (N, 3) np.array of route points
        location: (3,) np.array of the location to project

    Returns:
        Distance from the projected point to the location
    """
    # Compute the distance between the location and each point on the route
    distances = np.linalg.norm(route - location, axis=1)

    # Find the minimum distance (i.e., the closest point)
    min_distance = np.min(distances)

    return min_distance


@beartype
def compute_global_route(
    world: carla.World,
    source_location: carla.Location,
    sink_location: carla.Location,
    resolution: float = 1.0,
) -> jt.Float[npt.NDArray, "n 3"]:
    """
    Args:
        world: carla.World instance
        source_location: carla.Location of the source point
        sink_location: carla.Location of the sink point
        resolution: resolution for the global route planner
    Returns:
        array of waypoints in the format [[x, y, z], ...]
    """
    grp = GlobalRoutePlanner(world.get_map(), resolution)
    route = grp.trace_route(source_location, sink_location)
    return np.array(
        [
            [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
            for wp, _ in route
        ],
    )


@beartype
def intersection_of_routes(
    points_a: npt.NDArray,
    points_b: npt.NDArray,
    epsilon: float = 0.5,
) -> tuple[carla.Location | None, int | None]:
    """
    Args:
        points_a: (N, 3) np.array of route points for the first route
        points_b: (M, 3) np.array of route points for the second route
        epsilon: threshold distance to consider two points as intersecting
    Returns:
        The intersection point and its index if found, otherwise None.
    """
    points_a = np.array(points_a, dtype=np.float32)
    points_b = np.array(points_b, dtype=np.float32)
    diff = points_a[:, None, :2] - points_b[None, :, :2]  # (N, M, 2)
    dists = np.sqrt((diff**2).sum(axis=-1))  # (N, M)

    mask = dists < epsilon
    indices = np.argwhere(mask)

    if indices.shape[0] == 0:
        return None, None

    i, j = indices[0]
    z = (points_a[i, 2] + points_b[j, 2]) / 2.0
    x = (points_a[i, 0] + points_b[j, 0]) / 2.0
    y = (points_a[i, 1] + points_b[j, 1]) / 2.0
    return carla.Location(x=x, y=y, z=z), int(i)


@beartype
def dot_product(vector1: carla.Vector3D, vector2: carla.Vector3D) -> float:
    """
    Calculate the dot product of two vectors.

    Args:
        vector1: The first vector.
        vector2: The second vector.

    Returns:
        The dot product of the two vectors.
    """
    return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z


@beartype
def cross_product(vector1: carla.Vector3D, vector2: carla.Vector3D) -> carla.Vector3D:
    """
    Calculate the cross product of two vectors.

    Args:
        vector1: The first vector.
        vector2: The second vector.

    Returns:
        The cross product of the two vectors.
    """
    x = vector1.y * vector2.z - vector1.z * vector2.y
    y = vector1.z * vector2.x - vector1.x * vector2.z
    z = vector1.x * vector2.y - vector1.y * vector2.x

    return carla.Vector3D(x=x, y=y, z=z)


@beartype
def get_separating_plane(
    relative_position: carla.Vector3D,
    plane_normal: carla.Vector3D,
    obb1: carla.BoundingBox,
    obb2: carla.BoundingBox,
) -> bool:
    """
    Check if there is a separating plane between two oriented bounding boxes (OBBs).

    Args:
        relative_position: The relative position between the two OBBs.
        plane_normal: The normal vector of the plane.
        obb1: The first oriented bounding box.
        obb2: The second oriented bounding box.

    Returns:
        True if there is a separating plane, False otherwise.
    """
    # Calculate the projection of the relative position onto the plane normal
    projection_distance = abs(dot_product(relative_position, plane_normal))

    # Calculate the sum of the projections of the OBB extents onto the plane normal
    obb1_projection = (
        abs(
            dot_product(
                obb1.rotation.get_forward_vector() * obb1.extent.x,
                plane_normal,
            ),
        )
        + abs(
            dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal),
        )
        + abs(dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal))
    )

    obb2_projection = (
        abs(
            dot_product(
                obb2.rotation.get_forward_vector() * obb2.extent.x,
                plane_normal,
            ),
        )
        + abs(
            dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal),
        )
        + abs(dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal))
    )

    # Check if the projection distance is greater than the sum of the OBB projections
    return projection_distance > obb1_projection + obb2_projection


@beartype
def check_obb_intersection(obb1: carla.BoundingBox, obb2: carla.BoundingBox) -> bool:
    """
    Check if two 3D oriented bounding boxes (OBBs) intersect.

    Args:
        obb1: The first oriented bounding box.
        obb2: The second oriented bounding box.

    Returns:
        True if the two OBBs intersect, False otherwise.
    """
    relative_position = obb2.location - obb1.location

    # Check for separating planes along the axes of both OBBs
    if (
        get_separating_plane(
            relative_position,
            obb1.rotation.get_forward_vector(),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            obb1.rotation.get_right_vector(),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            obb1.rotation.get_up_vector(),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            obb2.rotation.get_forward_vector(),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            obb2.rotation.get_right_vector(),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            obb2.rotation.get_up_vector(),
            obb1,
            obb2,
        )
    ):
        return False

    # Check for separating planes along the cross products of the axes of both OBBs
    if (
        get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_forward_vector(),
                obb2.rotation.get_forward_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_forward_vector(),
                obb2.rotation.get_right_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_forward_vector(),
                obb2.rotation.get_up_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_right_vector(),
                obb2.rotation.get_forward_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_right_vector(),
                obb2.rotation.get_right_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_right_vector(),
                obb2.rotation.get_up_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_up_vector(),
                obb2.rotation.get_forward_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(
                obb1.rotation.get_up_vector(),
                obb2.rotation.get_right_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_up_vector()),
            obb1,
            obb2,
        )
    ):
        return False

    # If no separating plane is found, the OBBs intersect
    return True


@beartype
def get_angle_to(
    current_position: list[tuple[float, float]],
    current_heading: float,
    target_position: tuple | list,
) -> float:
    """
    Calculate the angle (in degrees) from the current position and heading to a target position.

    Args:
        current_position: A list of (x, y) coordinates representing the current position.
        current_heading: The current heading angle in radians.
        target_position: A tuple or list of (x, y) coordinates representing the target position.
    Returns:
        float: The angle (in degrees) from the current position and heading to the target position.
    """
    cos_heading = math.cos(current_heading)
    sin_heading = math.sin(current_heading)

    # Calculate the vector from the current position to the target position
    position_delta = target_position - current_position

    # Calculate the dot product of the position delta vector and the current heading vector
    aim_x = cos_heading * position_delta[0] + sin_heading * position_delta[1]
    aim_y = -sin_heading * position_delta[0] + cos_heading * position_delta[1]

    # Calculate the angle (in radians) from the current heading to the target position
    angle_radians = -math.atan2(-aim_y, aim_x)

    # Convert the angle from radians to degrees
    angle_degrees = float(math.degrees(angle_radians))

    return angle_degrees


def get_steer(
    config: ExpertConfig,
    turn_controller: PIDController,
    route_points: npt.NDArray,
    current_position: tuple[float, float],
    current_heading: float,
    current_speed: float,
) -> float:
    """
    Calculate the steering angle based on the current position, heading, speed, and the route points.

    Args:
        config: Configuration object containing PID controller parameters.
        turn_controller: An instance of the PIDController for steering.
        route_points: An array of (x, y) coordinates representing the route points.
        current_position: The current position (x, y) of the vehicle.
        current_heading: The current heading angle (in radians) of the vehicle.
        current_speed: The current speed of the vehicle (in m/s).

    Returns:
        The calculated steering angle.
    """
    # Calculate the steering angle using the turn controller
    steering_angle = turn_controller.step(
        route_points,
        current_speed,
        current_position,
        current_heading,
    )
    steering_angle = round(steering_angle, 3)
    return steering_angle


@beartype
def compute_target_speed_idm(
    config: ExpertConfig,
    desired_speed: numbers.Real,
    leading_actor_length: numbers.Real,
    ego_speed: numbers.Real,
    leading_actor_speed: numbers.Real,
    distance_to_leading_actor: numbers.Real,
    s0: numbers.Real = 4.0,
    T: numbers.Real = 0.5,
):
    """
    Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

    Args:
        config: Configuration object containing IDM parameters.
        desired_speed: The desired speed of the ego vehicle.
        leading_actor_length: The length of the leading actor (vehicle or obstacle).
        ego_speed: The current speed of the ego vehicle.
        leading_actor_speed: The speed of the leading actor.
        distance_to_leading_actor: The distance to the leading actor.
        s0: The minimum desired net distance.
        T: The desired time headway.

    Returns:
        The computed target speed for the ego vehicle.
    """

    a = config.idm_maximum_acceleration  # Maximum acceleration [m/s²]
    b = (
        config.idm_comfortable_braking_deceleration_high_speed
        if ego_speed > config.idm_comfortable_braking_deceleration_threshold
        else config.idm_comfortable_braking_deceleration_low_speed
    )  # Comfortable deceleration [m/s²]
    delta = config.idm_acceleration_exponent  # Acceleration exponent

    t_bound = config.idm_t_bound

    @beartype
    def idm_equations(t: float, x: jt.Float[npt.NDArray, " 2"]) -> list[float]:
        """
        Differential equations for the Intelligent Driver Model.

        Args:
            t: Time.
            x: State variables [position, speed].

        Returns:
            Derivatives of the state variables.
        """
        ego_position, ego_speed = x

        speed_diff = ego_speed - leading_actor_speed
        s_star = s0 + ego_speed * T + ego_speed * speed_diff / 2.0 / np.sqrt(a * b)
        # The maximum is needed to avoid numerical unstabilities
        s = max(
            0.1,
            distance_to_leading_actor
            + t * leading_actor_speed
            - ego_position
            - leading_actor_length,
        )
        dvdt = a * (1.0 - (ego_speed / desired_speed) ** delta - (s_star / s) ** 2)

        return [ego_speed, dvdt]

    # Set the initial conditions
    y0 = [0.0, ego_speed]

    # Integrate the differential equations using RK45
    rk45 = RK45(fun=idm_equations, t0=0.0, y0=y0, t_bound=t_bound)
    while rk45.status == "running":
        rk45.step()

    # The target speed is the final speed obtained from the integration
    target_speed = rk45.y[1]

    # Clip the target speed to non-negative values
    return np.clip(target_speed, 0, np.inf)


@beartype
def get_previous_road_lane_ids(
    config: ExpertConfig,
    starting_waypoint: carla.Waypoint,
) -> list[tuple[int, int]]:
    """
    Retrieves the previous road and lane IDs for a given starting waypoint.

    Args:
        config: Configuration object containing parameters.
        starting_waypoint: The starting waypoint.
    Returns:
        A list of tuples containing road IDs and lane IDs.
    """
    current_waypoint = starting_waypoint
    previous_lane_ids = [(current_waypoint.road_id, current_waypoint.lane_id)]

    # Traverse backwards up to 100 waypoints to find previous lane IDs
    for _ in range(config.previous_road_lane_retrieve_distance):
        previous_waypoints = current_waypoint.previous(1)

        # Check if the road ends and no previous route waypoints exist
        if len(previous_waypoints) == 0:
            break
        current_waypoint = previous_waypoints[0]

        if (
            current_waypoint.road_id,
            current_waypoint.lane_id,
        ) not in previous_lane_ids:
            previous_lane_ids.append(
                (current_waypoint.road_id, current_waypoint.lane_id),
            )

    return previous_lane_ids


@beartype
def compute_min_time_for_distance(
    config: ExpertConfig,
    distance: float,
    target_speed: float,
    ego_speed: float,
) -> float:
    """
    Computes the minimum time the ego vehicle needs to travel a given distance.

    Args:
        config: Configuration object containing vehicle dynamics parameters.
        distance: The distance to be traveled.
        target_speed: The target speed of the ego vehicle.
        ego_speed: The current speed of the ego vehicle.

    Returns:
        The minimum time needed to travel the given distance.
    """
    min_time_needed = 0.0
    remaining_distance = distance
    current_speed = ego_speed

    # Iterate over time steps until the distance is covered
    while True:
        if current_speed == 0 and target_speed == 0 and remaining_distance > 0:
            return np.inf
        # Takes less than a tick to cover remaining_distance with current_speed
        if remaining_distance - current_speed * config.fps_inv < 0:
            break

        remaining_distance -= current_speed * config.fps_inv
        min_time_needed += config.fps_inv

        # Values from kinematic bicycle model
        normalized_speed = current_speed / 120.0
        speed_change_params = config.compute_min_time_to_cover_distance_params
        speed_change = np.clip(
            speed_change_params[0]
            + normalized_speed * speed_change_params[1]
            + speed_change_params[2] * normalized_speed**2
            + speed_change_params[3] * normalized_speed**3,
            0.0,
            np.inf,
        )
        current_speed = np.clip(
            120 * (normalized_speed + speed_change),
            0,
            target_speed,
        )

    # Add remaining time at the current speed
    min_time_needed += remaining_distance / current_speed

    return min_time_needed


@beartype
def wps_next_until_lane_end(wp: carla.Waypoint) -> list[carla.Waypoint]:
    """
    Get all waypoints until the lane ends (i.e., road_id or lane_id changes).
    Args:
        wp: The starting waypoint.
    Returns:
        list[carla.Waypoint]: A list of waypoints until the lane ends.
    """
    try:
        road_id_cur = wp.road_id
        lane_id_cur = wp.lane_id
        road_id_next = road_id_cur
        lane_id_next = lane_id_cur
        curr_wp = [wp]
        next_wps = []
        # https://github.com/carla-simulator/carla/issues/2511#issuecomment-597230746
        while road_id_cur == road_id_next and lane_id_cur == lane_id_next:
            next_wp = curr_wp[0].next(1)
            if len(next_wp) == 0:
                break
            curr_wp = next_wp
            next_wps.append(next_wp[0])
            road_id_next = next_wp[0].road_id
            lane_id_next = next_wp[0].lane_id
    except:
        next_wps = []

    return next_wps


def get_night_mode(weather) -> bool:
    """Check whether or not the street lights need to be turned on"""
    SUN_ALTITUDE_THRESHOLD_1 = 15
    SUN_ALTITUDE_THRESHOLD_2 = 165

    # For higher fog and cloudness values, the amount of light in scene starts to rapidly decrease
    CLOUDINESS_THRESHOLD = 80
    FOG_THRESHOLD = 40

    # In cases where more than one weather conditition is active, decrease the thresholds
    COMBINED_THRESHOLD = 10

    altitude_dist = weather.sun_altitude_angle - SUN_ALTITUDE_THRESHOLD_1
    altitude_dist = min(
        altitude_dist,
        SUN_ALTITUDE_THRESHOLD_2 - weather.sun_altitude_angle,
    )
    cloudiness_dist = CLOUDINESS_THRESHOLD - weather.cloudiness
    fog_density_dist = FOG_THRESHOLD - weather.fog_density

    # Check each parameter independently
    if altitude_dist < 0 or cloudiness_dist < 0 or fog_density_dist < 0:
        return True

    # Check if two or more values are close to their threshold
    joined_threshold = int(altitude_dist < COMBINED_THRESHOLD)
    joined_threshold += int(cloudiness_dist < COMBINED_THRESHOLD)
    joined_threshold += int(fog_density_dist < COMBINED_THRESHOLD)

    if joined_threshold >= 2:
        return True

    return False


def get_vehicle_velocity_in_ego_frame(ego: carla.Vehicle, actor: carla.Actor):
    """
    Return the x, y velocity of actor but in ego frame.
    Args
        ego: the ego vehicle
        actor: the actor for which the velocity is computed
    Return
        array of x, y velocity in ego frame
    """
    # Get velocity in world coordinates
    vel = actor.get_velocity()
    vel_vec = np.array([vel.x, vel.y, vel.z])

    # Ego orientation
    ego_transform = ego.get_transform()
    yaw = np.radians(ego_transform.rotation.yaw)

    # World-to-ego rotation matrix (2D)
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

    # Project velocity onto ego frame
    vel_ego = R @ vel_vec[:2]
    return vel_ego.tolist()


@torch.jit.script
def perturbated_sensor_cfg(
    camera_rot: list[float],
    camera_pos: list[float],
    perturbation_translation: float,
    perturbation_rotation: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Original pose
    # Apply rigid transformation
    perturbation_rotation_rad = torch.deg2rad(torch.tensor(perturbation_rotation))

    # Create rotation matrix components
    cos_rot = torch.cos(perturbation_rotation_rad)
    sin_rot = torch.sin(perturbation_rotation_rad)

    # Build rotation matrix manually to avoid nested tensor issues
    rotation_matrix = torch.zeros(2, 2)
    rotation_matrix[0, 0] = cos_rot
    rotation_matrix[0, 1] = -sin_rot
    rotation_matrix[1, 0] = sin_rot
    rotation_matrix[1, 1] = cos_rot

    # Apply transformation
    camera_pos_2d = torch.tensor([camera_pos[0], camera_pos[1]])
    translation = torch.tensor([0.0, perturbation_translation])
    transformed_pos = rotation_matrix @ camera_pos_2d + translation

    # Extract x, y coordinates - convert to float to avoid tensor/float mixing
    x = float(transformed_pos[0])
    y = float(transformed_pos[1])

    camera_rot_new = torch.tensor(
        [camera_rot[0], camera_rot[1], camera_rot[2] + perturbation_rotation],
    )
    camera_pos_new = torch.tensor([x, y, camera_pos[2]])
    return camera_rot_new, camera_pos_new


@torch.jit.script
def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Convert Euler angles (in degrees) to rotation matrix"""
    # Convert to radians
    roll_rad = torch.deg2rad(torch.tensor(roll))
    pitch_rad = torch.deg2rad(torch.tensor(pitch))
    yaw_rad = torch.deg2rad(torch.tensor(yaw))

    # Compute trigonometric values
    cos_r, sin_r = torch.cos(roll_rad), torch.sin(roll_rad)
    cos_p, sin_p = torch.cos(pitch_rad), torch.sin(pitch_rad)
    cos_y, sin_y = torch.cos(yaw_rad), torch.sin(yaw_rad)

    # Create rotation matrices manually to avoid nested tensor issues
    # Roll matrix (rotation around x-axis)
    R_x = torch.zeros(3, 3)
    R_x[0, 0] = 1.0
    R_x[1, 1] = cos_r
    R_x[1, 2] = -sin_r
    R_x[2, 1] = sin_r
    R_x[2, 2] = cos_r

    # Pitch matrix (rotation around y-axis)
    R_y = torch.zeros(3, 3)
    R_y[0, 0] = cos_p
    R_y[0, 2] = sin_p
    R_y[1, 1] = 1.0
    R_y[2, 0] = -sin_p
    R_y[2, 2] = cos_p

    # Yaw matrix (rotation around z-axis)
    R_z = torch.zeros(3, 3)
    R_z[0, 0] = cos_y
    R_z[0, 1] = -sin_y
    R_z[1, 0] = sin_y
    R_z[1, 1] = cos_y
    R_z[2, 2] = 1.0

    # Combined rotation matrix (ZYX order - same as scipy's 'xyz' order)
    return R_z @ R_y @ R_x


@torch.jit.script
def unproject_camera(
    depth: torch.Tensor,
    camera_fov: float,
    camera_width: float,
    camera_height: float,
    camera_rot: list[float],
    camera_pos: list[float],
    perturbation_rotation: float,
    perturbation_translation: float,
    device: torch.device,
) -> torch.Tensor:
    camera_rot, camera_pos = perturbated_sensor_cfg(
        camera_rot,
        camera_pos,
        perturbation_translation,
        perturbation_rotation,
    )

    # Projecting instance segmentation to ego space
    fov, width, height = camera_fov, camera_width, camera_height
    f_u = f_v = width / (
        2 * torch.tan(torch.deg2rad(torch.tensor([fov / 2], device=device)))
    )
    c_u = torch.tensor(width / 2, device=device)
    c_v = torch.tensor(height / 2, device=device)

    # Pixel coordinates
    x_coords, y_coords = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    depth = depth.flatten().to(device).to(torch.float32)

    # Back-project to camera space
    x = (x_coords - c_u) * depth / f_u
    y = (y_coords - c_v) * depth / f_v
    z = depth
    points_cam = torch.stack([x, y, z], dim=1).to(torch.float32)

    # Transform to ego space - use custom euler to rotation matrix function
    rot = euler_to_rotation_matrix(
        float(camera_rot[0]),
        float(camera_rot[1]),
        float(camera_rot[2]),
    ).to(dtype=torch.float32, device=device)
    trans = camera_pos.to(dtype=torch.float32, device=device)

    # From camera frame (OpenCV) to CARLA ego frame
    T_camera_to_ego = torch.tensor(
        [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
        device=device,
        dtype=torch.float32,
    )

    points = (points_cam @ T_camera_to_ego.T) @ rot.T + trans

    points[:, 2] -= camera_pos[-1] / 2  # Not sure why we need this :/

    return points


@beartype
def semantics_camera_pc(
    depth: npt.NDArray,
    instance: npt.NDArray,
    camera_fov: Real,
    camera_width: Real,
    camera_height: Real,
    camera_rot: list[Real],
    camera_pos: list[Real],
    perturbation_rotation: Real,
    perturbation_translation: Real,
    config: ExpertConfig,
) -> torch.Tensor:
    """
    Unproject depth and instance segmentation to a point cloud in ego space with semantic and Unreal Engine instance IDs.

    Args:
        depth: Array of shape h x w
        instance: Array of shape h x w x 2
        camera_fov: Camera field of view
        camera_width: Camera width
        camera_height: Camera height
        camera_rot: Camera rotation (roll, pitch, yaw)
        camera_pos: Camera position (x, y, z)
        perturbation_rotation: perturbation rotation (degrees)
        perturbation_translation: perturbation translation (meters)
        config: Configuration object containing parameters.
    Returns:
        torch.Tensor: Array of shape N x 5 with attributes x, y, z,full CARLA semantic_id, unreal_id

    WARNING: id corresponds to Unreal Engine's instance IDs, not CARLA's actor IDs.
    Using match_unreal_engine_ids_to_carla_ids it is possible to map them to CARLA's actor IDs.
    """
    assert depth.ndim == 2
    assert instance.ndim == 3 and instance.shape[2] == 2
    device = torch.device("cpu")
    if config.unproject_on_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance_tensor = torch.from_numpy(instance).to(device)
    depth_tensor = torch.from_numpy(depth).to(device)
    pc = unproject_camera(
        depth_tensor,
        camera_fov,
        camera_width,
        camera_height,
        camera_rot,
        camera_pos,
        perturbation_rotation,
        perturbation_translation,
        device,
    )
    semantic_id = instance_tensor[..., 0].flatten()
    instance_id = instance_tensor[..., 1].flatten()
    assert semantic_id.max().item() < 100
    return torch.cat(
        [
            pc,
            semantic_id[:, None].to(torch.float32),
            instance_id[:, None].to(torch.float32),
        ],
        dim=1,
    ).cpu()


@beartype
def match_unreal_engine_ids_to_carla_bounding_boxes_ids(
    ego_matrix: npt.NDArray,
    boxes_original_semantic_id: int,
    carla_boxes: list[dict],
    ego_camera_pc: npt.NDArray,
    penalize_points_outside: bool = False,
) -> dict:
    """
    Return a dict mapping Unreal Engine's IDs to CARLA actor IDs for a given semantic class.
    Args:
        ego_matrix: Needed for transformation.
        boxes_original_semantic_id: Semantic ID to be replaced (e.g., CONE_AND_TRAFFIC_WARNING_SEMANTIC_ID)
        carla_boxes: list of CARLA bounding boxes
        ego_camera_pc: Nx5 array with x,y,z,semantic_id,unreal_id
        penalize_points_outside: If True, points outside the bounding box are penalized in the cost matrix.
    Returns:
        Mapping from Unreal Engine instance IDs to CARLA actor IDs
    """
    if len(carla_boxes) == 0:
        return {}
    # --- Consider only instance IDs of class we want to transform ---
    unreal_semantic_ids = ego_camera_pc[:, CameraPointCloudIndex.UNREAL_SEMANTICS_ID]
    unreal_instance_ids = ego_camera_pc[:, CameraPointCloudIndex.UNREAL_INSTANCE_ID]
    filtered_unreal_instance_ids = unreal_instance_ids[
        unreal_semantic_ids == boxes_original_semantic_id
    ]
    if filtered_unreal_instance_ids.shape[0] == 0:
        return {}

    # --- Group points by unreal instance id. ---
    map_unreal_id_to_points_clouds = {}
    for unreal_instance_id in np.unique(filtered_unreal_instance_ids):
        map_unreal_id_to_points_clouds[unreal_instance_id] = ego_camera_pc[
            unreal_instance_ids == unreal_instance_id
        ]
    unreal_instance_ids_list = list(map_unreal_id_to_points_clouds.keys())

    # --- Construct cost matrix mapping carla to unreal---
    cost_matrix = np.full(
        (len(carla_boxes), len(unreal_instance_ids_list)),
        0,
        dtype=np.float32,
    )
    for carla_index, carla_box in enumerate(carla_boxes):
        for instance_index, unreal_instance_id in enumerate(unreal_instance_ids_list):
            unreal_instance_pc = map_unreal_id_to_points_clouds[unreal_instance_id]

            filtered_unreal_instance_pc = unreal_instance_pc
            if penalize_points_outside:
                # If we penalize points outside of bb, we also want to minimize those penalties.
                # This is needed since Car windows are transparent and take depth of objects behind them
                dist = np.linalg.norm(unreal_instance_pc[..., :3], axis=1)
                filtered_unreal_instance_pc = unreal_instance_pc[
                    (np.quantile(dist, 0.15) < dist) & (dist < np.quantile(dist, 0.85))
                ]  # Remove outliers

            relevant_points = get_points_in_actor_frame_and_in_bbox(
                ego_matrix,
                np.array(carla_box["matrix"]),
                np.array(carla_box["extent"]),
                filtered_unreal_instance_pc,
                pad=True,
            )

            num_points_in = relevant_points.shape[0]
            num_points_out = filtered_unreal_instance_pc.shape[0] - num_points_in
            cost_matrix[carla_index, instance_index] = -num_points_in + 0.1 * (
                num_points_out if penalize_points_outside else 0
            )

    # --- Match camera point clouds with CARLA bounding boxes ---
    matched_carla_indices, matched_unreal_indices = linear_sum_assignment(cost_matrix)
    map_unreal_id_to_carla_id = {}
    for carla_index, unreal_index in zip(
        matched_carla_indices,
        matched_unreal_indices,
        strict=False,
    ):
        if cost_matrix[carla_index, unreal_index] < 0:
            map_unreal_id_to_carla_id[int(unreal_instance_ids_list[unreal_index])] = (
                carla_boxes[carla_index]["id"]
            )
    return map_unreal_id_to_carla_id


@beartype
def match_unreal_engine_ids_to_carla_actors_ids(
    ego_matrix: npt.NDArray,
    actors_original_semantic_id: int,
    carla_actors: list[carla.Actor],
    ego_camera_pc: npt.NDArray,
) -> dict:
    """
    Return a dict mapping Unreal Engine's IDs to CARLA actor IDs for a given semantic class.
    Args:
        ego_matrix: Needed for transformation.
        actors_original_semantic_id: Semantic ID to be replaced (e.g., CONE_AND_TRAFFIC_WARNING_SEMANTIC_ID)
        carla_actors: list of CARLA bounding boxes
        ego_camera_pc: Nx5 array with x,y,z,semantic_id,unreal_id
    Returns:
        Mapping from Unreal Engine instance IDs to CARLA actor IDs
    """
    carla_boxes = [
        {
            "id": actor.id,
            "matrix": actor.get_transform().get_matrix(),
            "extent": [
                actor.bounding_box.extent.x,
                actor.bounding_box.extent.y,
                actor.bounding_box.extent.z,
            ],
        }
        for actor in carla_actors
    ]
    return match_unreal_engine_ids_to_carla_bounding_boxes_ids(
        ego_matrix,
        actors_original_semantic_id,
        carla_boxes,
        ego_camera_pc,
    )


@beartype
def get_num_points_in_actor(
    ego: carla.Vehicle,
    actor: carla.Actor,
    ego_point_cloud: npt.NDArray,
    pad: bool = False,
) -> int:
    """
    Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
    :param ego: The ego vehicle actor
    :param actor: The vehicle actor for which the bounding box is checked
    :param ego_point_cloud: 3D point clouds in ego frame
    :return: Returns the number of LiDAR hits within the bounding box of the actor
    """
    return len(
        get_points_in_actor_frame_and_in_bbox(
            np.array(ego.get_transform().get_matrix()),
            np.array(actor.get_transform().get_matrix()),
            np.array(
                [
                    actor.bounding_box.extent.x,
                    actor.bounding_box.extent.y,
                    actor.bounding_box.extent.z,
                ],
            ),
            ego_point_cloud,
            pad,
        ),
    )


@beartype
def get_num_points_in_bbox(
    ego: carla.Vehicle,
    bbox: dict,
    ego_point_cloud: npt.NDArray,
    pad: bool = False,
) -> int:
    """
    Checks for a given bounding box in ego coordinate system, how many LiDAR hit there are in its bounding box.
    :param ego: The ego vehicle actor
    :param bbox: The bounding box dict with keys "matrix" and "extent"
    :param ego_point_cloud: 3D point clouds in ego frame
    :return: Returns the number of LiDAR hits within the bounding box
    """
    return len(
        get_points_in_actor_frame_and_in_bbox(
            np.array(ego.get_transform().get_matrix()),
            np.array(bbox["matrix"]),
            np.array(bbox["extent"]),
            ego_point_cloud,
            pad,
        ),
    )


@beartype
def transform_points_to_actor_frame(
    ego_matrix: npt.NDArray,
    actor_matrix: npt.NDArray,
    ego_point_cloud: npt.NDArray,
) -> npt.NDArray:
    if ego_point_cloud.shape[0] == 0:
        return np.array([])

    ego_point_cloud = ego_point_cloud[:, :3]  # Only xyz

    # Ego pose (rotation + translation)
    R_ego = ego_matrix[:3, :3]
    ego_pos = ego_matrix[:3, 3]

    # Actor pose (rotation + translation)
    R_actor = actor_matrix[:3, :3]
    actor_pos = actor_matrix[:3, 3]

    # Actor center in ego frame
    vehicle_pos = R_ego.T @ (actor_pos - ego_pos)

    # Rotation: world -> ego -> actor
    R_rel = R_ego.T @ R_actor  # ego -> actor frame

    # Transform LiDAR points into actor's local frame
    return (R_rel.T @ (ego_point_cloud - vehicle_pos).T).T


@beartype
def get_points_in_actor_frame_and_in_bbox(
    ego_matrix: npt.NDArray,
    actor_matrix: npt.NDArray,
    actor_extent: npt.NDArray,
    ego_point_cloud: npt.NDArray,
    pad: bool = False,
) -> npt.NDArray:
    """
    Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
    :param ego: The ego vehicle actor
    :param actor: The vehicle actor for which the bounding box is checked
    :param ego_point_cloud: 3D point clouds in ego frame
    :return: Returns the points within the bounding box of the actor
    """
    vehicle_lidar = transform_points_to_actor_frame(
        ego_matrix,
        actor_matrix,
        ego_point_cloud,
    )
    if len(vehicle_lidar) == 0:
        return np.array([])

    # Count points inside bounding box
    margin_x = 0.25 if pad else 0.0
    margin_y = 0.25 if pad else 0.0
    margin_z = 0.25 if pad else 0.0
    mask = (
        (np.abs(vehicle_lidar[:, 0]) < actor_extent[0] + margin_x)
        & (np.abs(vehicle_lidar[:, 1]) < actor_extent[1] + margin_y)
        & (np.abs(vehicle_lidar[:, 2]) < actor_extent[2] + margin_z)
    )
    return vehicle_lidar[mask]


@beartype
def enhance_semantics_segmentation(
    instance_image: npt.NDArray,
    semantics_segmentation_image: npt.NDArray,
    id_map: dict[int, int],
    boxes_new_semantic_id: int,
) -> npt.NDArray:
    """
    Enhance the semantics segmentation image by replacing specific semantic IDs based on instance segmentation.
    Args:
        instance_image: HxWx2 numpy array where the first channel is the semantic ID and the second channel is the instance ID.
        semantics_segmentation_image: HxW numpy array with the original semantic segmentation IDs.
        id_map: Dictionary mapping from original semantic IDs to new semantic IDs.
        boxes_new_semantic_id: The new semantic ID to assign to the specified original semantic IDs.
    Returns:
        npt.NDArray: Enhanced semantics segmentation image.
    """
    if len(id_map) == 0:
        return semantics_segmentation_image
    enhanced_image = semantics_segmentation_image.copy()
    for original_unreal_instance_id in id_map.keys():
        mask = instance_image[..., 1] == original_unreal_instance_id
        enhanced_image[mask] = boxes_new_semantic_id
    return enhanced_image


@beartype
def sample_sensor_perturbation_parameters(
    config: ExpertConfig,
    max_speed_limit_route: float,
    min_lane_width_route: float,
) -> tuple[float, float]:
    """
    Sample sensor perturbation parameters (translation and rotation) based on the route's speed limit and lane width.
    Args:
        config: Configuration object containing perturbation parameters.
        max_speed_limit_route: Maximum speed limit along the route (in m/s).
        min_lane_width_route: Minimum lane width along the route (in meters).
    Returns:
        tuple[float, float]: A tuple containing the perturbation translation (in meters) and perturbation rotation (in degrees).
    """
    safety_translation_perturbation_gap = (
        config.default_safety_translation_perturbation_penalty
    )
    if max_speed_limit_route < constants.URBAN_MAX_SPEED_LIMIT:
        safety_translation_perturbation_gap = (
            config.urban_safety_translation_perturbation_penalty
        )
    lateral_gap = max(
        min_lane_width_route / 2.0
        - config.ego_extent_y / 2
        - safety_translation_perturbation_gap,
        0.1,
    )
    tmax = min(config.camera_translation_perturbation_max, lateral_gap)

    # Pick perturbation translation shift
    perturbation_translation = np.random.choice([-1, 1]) * np.random.uniform(
        low=config.camera_translation_perturbation_min,
        high=tmax,
    )

    # Next, pick rotation perturbation ranges, depending on translation perturbation.
    # Interpolate perturbation rotation depends on translation perturbation to avoid unrealistic configurations.
    neg_range = (
        -config.camera_rotation_perturbation_min,
        config.camera_rotation_perturbation_max,
    )  # for t <= -1
    pos_range = (
        -config.camera_rotation_perturbation_max,
        config.camera_rotation_perturbation_min,
    )  # for t >= +1
    if perturbation_translation <= -config.camera_translation_perturbation_max:
        rmin, rmax = neg_range
    elif perturbation_translation >= config.camera_translation_perturbation_max:
        rmin, rmax = pos_range
    else:
        alpha = (
            perturbation_translation + config.camera_translation_perturbation_max
        ) / (2.0 * config.camera_translation_perturbation_max)  # maps to [0,1]
        rmin = -(-neg_range[0] * (1 - alpha) + -pos_range[0] * alpha)
        rmax = neg_range[1] * (1 - alpha) + pos_range[1] * alpha

    beta = tmax / config.camera_translation_perturbation_max  # in [0,1]
    rmin *= beta
    rmax *= beta

    # Rejection sampling of rotation perturbation to avoid useless small rotation angles
    perturbation_rotation = np.random.uniform(low=rmin, high=rmax)
    while abs(perturbation_rotation) < config.camera_rotation_epsilon:
        perturbation_rotation = np.random.uniform(low=rmin, high=rmax)

    LOG.info(f"Translation perturbation range: [-{tmax:.2f}m, {tmax:.2f}m].")
    LOG.info(f"Sampled translation: {perturbation_translation:.2f}m.")
    LOG.info(
        f"rotation range: [{rmin:.2f}°, {rmax:.2f}°], sampled rotation: {perturbation_rotation:.2f}°",
    )

    return perturbation_translation, perturbation_rotation


@beartype
def get_weather_name(
    weather_parameter: carla.WeatherParameters,
    config: ExpertConfig,
) -> str:
    """
    Return the name of the weather preset matching the given CARLA WeatherParameters.
    Args:
        weather_parameter: The CARLA WeatherParameters object describing current weather.
        config: An expert config containing a dictionary of known weather presets.

    Returns:
        str: The name of the matched preset if found, otherwise the name of the nearest preset.
    """
    best_name, best_dist = None, float("inf")
    NEAREST_NEIGHBOR_KEYS = [
        "cloudiness",
        "dust_storm",
        "fog_density",
        "precipitation",
        "precipitation_deposits",
        "sun_altitude_angle",
        "wetness",
        "wind_intensity",
    ]

    for name, preset in config.weather_settings.items():
        # Check exact match
        if all(
            math.isclose(getattr(weather_parameter, key), val, abs_tol=1e-2)
            for key, val in preset.items()
        ):
            return name

        # Compute distance for fallback (restricted keys)
        dist = 0.0
        for key in NEAREST_NEIGHBOR_KEYS:
            diff = getattr(weather_parameter, key) - preset[key]
            dist += diff * diff
        if dist < best_dist:
            best_name, best_dist = name, dist

    LOG.warning(f"Weather preset not found, using nearest {best_name}")
    return best_name


@beartype
def weather_parameter_to_dict(weather_parameter: carla.WeatherParameters) -> dict:
    """
    Convert a CARLA WeatherParameters object into a plain Python dictionary.

    This extracts the subset of weather-related attributes that are relevant
    for comparison or storage (e.g., matching against preset configurations).

    Args:
        weather_parameter: A CARLA WeatherParameters object.

    Returns:
        dict: A dictionary mapping attribute names to their corresponding float values.
    """
    keys = [
        "cloudiness",
        "dust_storm",
        "fog_density",
        "fog_distance",
        "fog_falloff",
        "mie_scattering_scale",
        "precipitation",
        "precipitation_deposits",
        "rayleigh_scattering_scale",
        "scattering_intensity",
        "sun_altitude_angle",
        "sun_azimuth_angle",
        "wetness",
        "wind_intensity",
    ]
    return {k: getattr(weather_parameter, k) for k in keys}


@beartype
def compute_camera_occlusion_score(
    pc: jt.Float[torch.Tensor, "N 5"],
    max_z: float = 1.0,
    min_dist: float = 6.0,
    max_dist: float = 48.0,
    quantile: float = 0.90,
) -> float:
    """
    Compute an occlusion score for a single camera's semantic point cloud.

    Args:
        pc: Semantic point cloud for one camera as a tensor, where the last channel encodes the semantic class ID.
        max_z: Maximum height threshold for valid points.
        min_dist: Minimum distance from ego to keep points.
        max_dist: Maximum distance from ego to keep points.
        quantile: Quantile of the point distances used to compute visibility range.

    Returns:
        float: Occlusion score in [0, 1], where higher values indicate
               better visibility (less occlusion).
    """
    CLASSES_OF_INTEREST = (
        CarlaSemanticSegmentationClass.Roads,
        CarlaSemanticSegmentationClass.SideWalks,
        CarlaSemanticSegmentationClass.Pedestrian,
        CarlaSemanticSegmentationClass.Rider,
        CarlaSemanticSegmentationClass.Car,
        CarlaSemanticSegmentationClass.Truck,
        CarlaSemanticSegmentationClass.Bus,
        CarlaSemanticSegmentationClass.Motorcycle,
        CarlaSemanticSegmentationClass.Bicycle,
        CarlaSemanticSegmentationClass.RoadLine,
    )

    if pc.shape[0] == 0:
        return 0.0

    # z filter
    pc = pc[pc[:, CameraPointCloudIndex.Z] < max_z]

    # distance filters
    norms = torch.linalg.norm(pc[:, : CameraPointCloudIndex.Z + 1], dim=1)
    mask = (norms > min_dist) & (norms < max_dist)
    pc = pc[mask]
    all_pc = pc

    if all_pc.shape[0] == 0:
        return 0.0

    # class filter
    class_mask = torch.isin(
        pc[:, CameraPointCloudIndex.UNREAL_SEMANTICS_ID].long(),
        torch.tensor(CLASSES_OF_INTEREST, device=pc.device),
    )
    pc = pc[class_mask][:, : CameraPointCloudIndex.Z + 1]

    if pc.shape[0] == 0:
        return 0.0

    # compute stats
    norms = torch.linalg.norm(pc, dim=1)
    q = torch.quantile(norms, quantile).item()
    c = norms.shape[0]
    n = all_pc.shape[0]

    return (q / max_dist) * (c / n)


@beartype
def rotate_point(point: carla.Vector3D, angle: float) -> carla.Vector3D:
    """Rotate a given point by a given angle.

    Args:
        point: The point to be rotated.
        angle: The angle in degrees.
    Returns:
        The rotated point.
    """
    x_ = (
        math.cos(math.radians(angle)) * point.x
        - math.sin(math.radians(angle)) * point.y
    )
    y_ = (
        math.sin(math.radians(angle)) * point.x
        + math.cos(math.radians(angle)) * point.y
    )
    return carla.Vector3D(x_, y_, point.z)


@beartype
def get_traffic_light_waypoints(traffic_light: carla.Actor, carla_map: carla.Map):
    """Get waypoints of roads controlled by a given traffic light.

    This function finds waypoints on roads that lead to intersections controlled
    by the traffic light. It discretizes the traffic light's trigger volume,
    finds corresponding road waypoints, and advances them toward the intersection
    while staying within the traffic light's influence area.

    Args:
        traffic_light: The traffic light object to analyze.
        carla_map: The CARLA map containing road network information.

    Returns:
        tuple: A tuple containing:
            - area_loc (carla.Location): The world location of the trigger volume center
            - wps (list[carla.Waypoint]): List of waypoints on roads controlled by this traffic light
    """
    base_transform = traffic_light.get_transform()
    base_loc = traffic_light.get_location()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    # Discretize the trigger box into points
    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(
        -0.9 * area_ext.x,
        0.9 * area_ext.x,
        1.0,
    )  # 0.9 to avoid crossing to adjacent lanes

    area = []
    for x in x_values:
        point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if (
            not ini_wps
            or ini_wps[-1].road_id != wpx.road_id
            or ini_wps[-1].lane_id != wpx.lane_id
        ):
            ini_wps.append(wpx)

    # Advance them until the intersection
    wps = []
    eu_wps = []
    for wpx in ini_wps:
        distance_to_light = base_loc.distance(wpx.transform.location)
        eu_wps.append(wpx)
        next_distance_to_light = distance_to_light + 1.0
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            next_distance_to_light = base_loc.distance(next_wp.transform.location)
            if (
                next_wp
                and not next_wp.is_intersection
                and next_distance_to_light <= distance_to_light
            ):
                eu_wps.append(next_wp)
                distance_to_light = next_distance_to_light
                wpx = next_wp
            else:
                break

        if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
            wps.append(eu_wps[-4])
        else:
            wps.append(wpx)

    return area_loc, wps


@beartype
def get_same_direction_lanes(
    waypoint: carla.Waypoint,
    max_lane_search: int = 10,
) -> list[carla.Waypoint]:
    """
    Find all waypoints in the same direction (left and right lanes)

    Args:
        waypoint: The reference waypoint
        max_lane_search: Maximum number of lanes to search in each direction

    Returns:
        List of waypoints in the same direction
    """
    same_direction_waypoints = []  # Include the original waypoint

    # Search left lanes
    current_wp = waypoint
    for _ in range(max_lane_search):
        left_wp = current_wp.get_left_lane()
        if left_wp is None:
            break
        if left_wp.lane_type != carla.LaneType.Driving:
            break
        if left_wp.lane_id == waypoint.lane_id:
            continue  # Skip if it's the same lane
        if left_wp.lane_id * waypoint.lane_id < 0:
            continue  # Skip if it's the same lane

        same_direction_waypoints.append(left_wp)
        current_wp = left_wp

    # Search right lanes
    current_wp = waypoint
    for _ in range(max_lane_search):
        right_wp = current_wp.get_right_lane()
        if right_wp is None:
            break
        if right_wp.lane_type != carla.LaneType.Driving:
            break
        if right_wp.lane_id == waypoint.lane_id:
            continue  # Skip if it's the same lane
        if right_wp.lane_id * waypoint.lane_id < 0:
            continue  # Skip if it's the same lane

        same_direction_waypoints.append(right_wp)
        current_wp = right_wp

    return same_direction_waypoints


@beartype
def get_stop_waypoints(
    ego_waypoint: carla.Waypoint,
    traffic_light: carla.TrafficLight,
) -> list[carla.Waypoint]:
    """
    Get all waypoints at the intersection controlled by this traffic light in the same direction.

    Args:
        ego_waypoint: The ego vehicle's current waypoint for lane matching
        traffic_light: The traffic light to get stop waypoints from

    Returns:
        List of waypoints at the intersection in the same direction
    """
    waypoints = traffic_light.get_stop_waypoints()

    # Push all waypoints forward to intersection
    intersection_waypoints = []
    for wp in waypoints:
        current = previous = wp
        while current.next(1.0) and not current.get_junction():
            previous = current
            current = current.next(1.0)[0]
        intersection_waypoints.append(previous)

    if not intersection_waypoints:
        return []

    # Use intersection waypoint with same lane id as ego, otherwise use first
    base_wp = intersection_waypoints[0]
    for wp in intersection_waypoints:
        if wp.lane_id == ego_waypoint.lane_id:
            base_wp = wp
            break
    same_direction = get_same_direction_lanes(base_wp)

    # Include the base waypoint itself
    if len(same_direction) + 1 < len(intersection_waypoints):
        return intersection_waypoints
    return [base_wp] + same_direction


@beartype
def create_bounding_box_for_waypoint(
    original_bbox: carla.BoundingBox,
    waypoint: carla.Waypoint,
) -> carla.BoundingBox:
    """
    Create a new bounding box positioned at the given waypoint

    Args:
        original_bbox: The original traffic light bounding box
        waypoint: The waypoint where to position the new bounding box

    Returns:
        New bounding box with updated location
    """
    new_bbox = carla.BoundingBox()
    new_bbox.location = waypoint.transform.location
    new_bbox.rotation = waypoint.transform.rotation
    new_bbox.extent = original_bbox.extent
    return new_bbox


class NegativeIdCounter:
    """ID generator that maps keys to negative IDs, creating new ones only when needed"""

    @beartype
    def __init__(self, start_value: int = -100000) -> None:
        """
        Initialize the ID generator

        Args:
            start_value: Starting value for new IDs (default: -100000)
        """
        self._id_map: dict[int, int] = {}
        self._current = start_value

    @beartype
    def __call__(self, key: int) -> int:
        """
        Get ID for a key. If key exists in map, return existing ID.
        If not, create and return a new negative ID.

        Args:
            key: The key to get an ID for

        Returns:
            The ID associated with the key
        """
        if key in self._id_map:
            return self._id_map[key]

        # Create new ID
        new_id = self._current
        self._id_map[key] = new_id
        self._current -= 1
        return new_id


@beartype
def convert_depth(data: jt.UInt8[npt.NDArray, "h w 3"]) -> jt.Float[npt.NDArray, "h w"]:
    """Compute normalized depth from a CARLA depth map.

    Converts CARLA's RGB-encoded depth values to actual depth in meters.

    Args:
        data: CARLA depth map as RGB array of shape (height, width, 3).

    Returns:
        Depth values in meters as array of shape (height, width).
    """
    # Optimized implementation:
    # 1. Avoids astype(float32) on the whole 3D array (saves 66% memory allocation)
    # 2. Computes per-channel contributions directly

    scale = 1000.0 / 16777215.0  # 1000 / (256**3 - 1)

    # R * 65536 * scale + G * 256 * scale + B * scale
    # Initialize with the most significant byte (R)
    depth = data[:, :, 0].astype(np.float32)
    depth *= 65536.0 * scale

    # Add Middle byte (G)
    depth += data[:, :, 1].astype(np.float32) * (256.0 * scale)

    # Add Least significant byte (B)
    depth += data[:, :, 2].astype(np.float32) * scale

    return depth


@beartype
def convert_instance_segmentation(
    data: jt.UInt8[npt.NDArray, "H W 3"],
) -> jt.Int32[npt.NDArray, "H W 2"]:
    """
    Args:
        data: Instance segmentation map from CARLA of shape (H, W, 3) with values in range [0, 255].
              R channel = semantic ID
              G+B*256 = instance ID
    Returns:
        data: Instance segmentation of shape (H, W, 2) with channels: [semantic_id, instance_id]
    """
    semantic_id = data[:, :, 0]
    semantic_id[semantic_id >= len(SEMANTIC_SEGMENTATION_CONVERTER)] = (
        0  # Carla semantic segmentation has some bug which give invalid semantic class.
    )
    instance_id = data[:, :, 1].astype(np.int32) + (data[:, :, 2].astype(np.int32) << 8)
    return np.stack([semantic_id, instance_id], axis=-1)


@beartype
def enhance_depth(
    depth: jt.Float[npt.NDArray, "H W"],
    semantic_segmentation: jt.UInt8[npt.NDArray, "H W"],
    instance_semantic_segmentation: jt.Int32[npt.NDArray, "H W 2"],
) -> jt.Float[npt.NDArray, "H W"]:
    """
    Make car windows not transparent in depth images.
    This function could be extended further for less noisy depth map but it's good for now.

    Args:
        depth: Metric depth
        semantic_segmentation: Semantic segmentation of the image
        instance_semantic_segmentation: First channel is semantic id and second channel is unreal engine instance id
    Returns:
        Repaired depth image where car windows are not transparent
    """
    CAR_SEMANTIC_ID = 14
    instance_id = instance_semantic_segmentation[..., 1]
    semantic_id = instance_semantic_segmentation[..., 0]
    instance_ids = np.unique(instance_id[semantic_id == CAR_SEMANTIC_ID])

    depth_repaired = depth.copy()
    for inst_id in instance_ids:
        inst_mask = instance_id == inst_id
        window_mask = inst_mask & (semantic_segmentation != CAR_SEMANTIC_ID)
        depth_repaired[window_mask] = depth_repaired[inst_mask].min()

    return depth_repaired


if __name__ == "__main__":
    config = ExpertConfig()
