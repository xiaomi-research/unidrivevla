import logging
import lzma
import math
import numbers
import pickle
import pickletools
from typing import Any

import carla
import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from agents.navigation.local_planner import RoadOption
from beartype import beartype
from scipy.optimize import fsolve

from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


def read_pickle(path: str) -> Any:
    """Read pickled data from a compressed file.

    Args:
        path: Path to the compressed pickle file.

    Returns:
        Unpickled data object.
    """
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path: str, data: Any) -> None:
    """Write data to a compressed pickle file.

    Args:
        path: Path where to save the compressed pickle file.
        data: Data object to pickle and compress.
    """
    with lzma.open(path, "wb") as f:
        pickle_str = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_str = pickletools.optimize(pickle_str)
        f.write(pickle_str)


def angle2class(angle: float, num_dir_bins: int) -> tuple[int, float]:
    """Convert continuous angle to discrete class and residual.

    Encodes a continuous angle into a discrete class and a small regression
    residual from the class center to the actual angle.

    Args:
        angle: Continuous angle in radians (0-2π or -π~π).
        num_dir_bins: Number of discrete direction bins for encoding.

    Returns:
        A tuple containing:
            - Discrete angle class as integer
            - Angle residual as float (difference from class center)
    """
    angle = angle % (2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_dir_bins)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    angle_cls = shifted_angle // angle_per_class
    angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
    return int(angle_cls), angle_res


@beartype
def convert_gps_to_carla(
    gps: npt.NDArray,
    lat_ref: float,
    lon_ref: float,
) -> npt.NDArray:
    """
    Converts GPS signal into the CARLA coordinate frame.

    Args:
        gps: GPS from GNSS sensor
        lat_ref: Latitude reference point of the map.
        lon_ref: Longitude reference point of the map.
    Returns:
        npt.NDArray: CARLA coordinates of the specific map in meters.
    """
    EARTH_RADIUS_EQUA = 6378137.0  # Constant from CARLA leaderboard GPS simulation

    lat, lon, _ = gps
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
    mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
    y = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
        - my
    )
    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    gps = np.array([x, y, gps[2]])

    return gps


@beartype
def find_gps_ref(
    global_plan_world_coord: list[tuple[carla.Transform, RoadOption]],
    global_plan: list[tuple[dict[str, float], RoadOption]],
) -> tuple[float, float]:
    """The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
    GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0.

    Starting from Leaderboard 2.0, Town13 has a different value in CARLA 0.9.15.
    The following code, adapted from Bench2DriveZoo estimates the lat, lon reference values by using
    the fact that the leaderboard exposes the route plan also in CARLA coordinates.

    The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale of the GPS.

    It seems to work reasonably well, so we use this workaround for now.

    Args:
        global_plan_world_coord: The global plan in CARLA world coordinates.
        global_plan: The global plan in GPS coordinates. The dicts have keys 'lat', 'lon', 'z'.

    Returns:
        tuple[float, float]: lat_ref, lon_ref
    """
    try:
        locx, locy = (
            global_plan_world_coord[0][0].location.x,
            global_plan_world_coord[0][0].location.y,
        )
        lon, lat = global_plan[0][0]["lon"], global_plan[0][0]["lat"]
        earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation

        def equations(variables):
            x, y = variables
            eq1 = (
                lon * math.cos(x * math.pi / 180.0)
                - (locx * x * 180.0) / (math.pi * earth_radius_equa)
                - math.cos(x * math.pi / 180.0) * y
            )
            eq2 = (
                math.log(math.tan((lat + 90.0) * math.pi / 360.0))
                * earth_radius_equa
                * math.cos(x * math.pi / 180.0)
                + locy
                - math.cos(x * math.pi / 180.0)
                * earth_radius_equa
                * math.log(math.tan((90.0 + x) * math.pi / 360.0))
            )
            return [eq1, eq2]

        initial_guess = [0.0, 0.0]
        solution = fsolve(equations, initial_guess)
        return solution[0], solution[1]
    except Exception as e:
        LOG.warning(e)
        return 0.0, 0.0


@beartype
def is_point_in_camera_frustum(
    x: float | int,
    y: float | int,
    config: TrainingConfig,
    center_x: float | int = 0,
    center_y: float | int = 0,
) -> bool:
    """Check if a point is within the camera's field of view frustum.

    Only an approximation, but sufficient for most use cases.

    Args:
        x: X coordinate of the point to check.
        y: Y coordinate of the point to check.
        config: Training configuration object containing camera parameters.
        center_x: X coordinate of the camera center.
        center_y: Y coordinate of the camera center.

    Returns:
        True if the point is within the camera frustum, False otherwise.
    """
    if (
        config.num_used_cameras >= 3
    ):  # TODO: only a hack for now. We assume if 3 or more cameras are used, they cover 360 degree FOV
        return True
    fov_deg = config.num_used_cameras * 60 + 5  # in degrees
    fov_rad = math.radians(fov_deg / 2)
    dx = x - center_x
    dy = y - center_y
    angle = math.atan2(dy, dx)
    return abs(angle) <= fov_rad


def is_box_in_camera_frustum(box: dict[str, Any], config: TrainingConfig) -> bool:
    """Check if a bounding box intersects with the camera frustum.

    Determines whether any corner of the bounding box is within the
    camera's field of view.

    Args:
        box: Dictionary containing bounding box information with keys:
            - 'position': [x, y, z] coordinates
            - 'extent': [width, height, depth] dimensions
            - 'yaw': rotation angle
        config: Training configuration object containing camera parameters.

    Returns:
        True if the bounding box intersects the camera frustum, False otherwise.
    """
    position = box["position"]  # x, y, z
    extent = box["extent"]
    yaw = box["yaw"]

    # Compute 4 corners of the box in 2D
    w, h = extent[0], extent[1]
    dx = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotated = dx @ rot.T
    corners = rotated + np.array([position[0], position[1]])

    # Check if any corner is within frustum
    for corner in corners:
        if is_point_in_camera_frustum(corner[0], corner[1], config):
            return True

    return False


def transform_lidar_to_bounding_box(
    lidar_points: jt.Float[npt.NDArray, "n 3"],
    bb_in_vehicle_system: jt.Float[npt.NDArray, "5"],
) -> jt.Float[npt.NDArray, "n 2"]:
    """Transform LiDAR points from ego coordinate system to bounding box local coordinates.

    Args:
        lidar_points: LiDAR points in ego coordinate system of shape (N, 3).
        bb_in_vehicle_system: Bounding box parameters [x, y, w, h, yaw].

    Returns:
        Transformed LiDAR points in bounding box local coordinate system of shape (N, 2).
    """
    # Extract bounding box parameters
    bb_x, bb_y, _, _, bb_yaw = bb_in_vehicle_system[:5]

    # Create inverse rotation matrix for yaw
    rotation_matrix = np.array(
        [[np.cos(bb_yaw), np.sin(bb_yaw)], [-np.sin(bb_yaw), np.cos(bb_yaw)]],
    )

    # Translate LiDAR points relative to the bounding box center
    translated_points = lidar_points[:, :2] - np.array([bb_x, bb_y])

    # Rotate points to align with the bounding box's local frame
    return np.dot(translated_points, rotation_matrix.T)


def filter_lidar_points_in_obb(
    lidar_points: jt.Float[npt.NDArray, "n 3"],
    bb: jt.Float[npt.NDArray, "5"],
) -> jt.Float[npt.NDArray, "... 3"]:
    """Filter LiDAR points to find those inside the oriented bounding box.

    Args:
        lidar_points: LiDAR points in ego coordinate system of shape (N, 3).
        bb: Bounding box parameters [x, y, w, h, yaw].

    Returns:
        LiDAR points that are inside the bounding box.
    """
    # Transform LiDAR points to the bounding box coordinate system
    local_points = transform_lidar_to_bounding_box(lidar_points, bb)

    # Check if points are within the bounding box's extents
    in_x = (local_points[:, 0] > -bb[2] / 2) & (local_points[:, 0] < bb[2] / 2)
    in_y = (local_points[:, 1] > -bb[3] / 2) & (local_points[:, 1] < bb[3] / 2)

    # Include z-range checks if needed
    in_box = in_x & in_y

    return lidar_points[in_box]


def normalize_angle(x: float) -> float:
    """
    Normalize an angle to the range [-π, π).

    This function takes an angle in radians and converts it to the standard
    range of [-π, π), which is commonly used in robotics, navigation, and
    other applications where angle wrapping is needed.

    Args:
        x: Angle in radians (can be any real number)

    Returns:
        float: Normalized angle in the range [-π, π)

    Examples:
        >>> normalize_angle(3 * np.pi)
        -3.141592653589793
        >>> normalize_angle(-3 * np.pi)
        3.141592653589793
        >>> normalize_angle(np.pi / 4)
        0.7853981633974483
        >>> normalize_angle(0)
        0.0
    """
    x = x % (2 * np.pi)  # Force angle into range [0, 2π)
    if x > np.pi:  # If angle > π, wrap to negative equivalent
        x -= 2 * np.pi  # Move to range [-π, π)
    return x


def normalize_angle_degree(x: float) -> float:
    """
    Normalize an angle to the range [-180°, 180°).

    This function takes an angle in degrees and converts it to the standard
    range of [-180°, 180°), which is the degree equivalent of the radian
    normalization. Useful for applications working with compass headings,
    geographic coordinates, or other degree-based angle systems.

    Args:
        x: Angle in degrees (can be any real number)

    Returns:
        float: Normalized angle in the range [-180°, 180°)

    Examples:
        >>> normalize_angle_degree(450)
        90.0
        >>> normalize_angle_degree(-270)
        90.0
        >>> normalize_angle_degree(180)
        -180.0
        >>> normalize_angle_degree(90)
        90.0
        >>> normalize_angle_degree(0)
        0.0
    """
    x = x % 360.0
    if x > 180.0:
        x -= 360.0
    return x


def euler_deg_to_mat(
    roll: float,
    pitch: float,
    yaw: float,
) -> jt.Float[npt.NDArray, "3 3"]:
    """Convert Euler angles in degrees to rotation matrix.

    Computes the 3D rotation matrix from roll, pitch, and yaw angles
    using the standard aerospace sequence (roll-pitch-yaw).

    Args:
        roll: Rotation around x-axis in degrees.
        pitch: Rotation around y-axis in degrees.
        yaw: Rotation around z-axis in degrees.

    Returns:
        3x3 rotation matrix combining all three rotations.
    """
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    # roll (x), pitch (y), yaw (z)
    return Rz @ Ry @ Rx


def lidar_to_ego_coordinate(
    lidar_rot: list[float],
    lidar_pos: list[float],
    lidar: jt.Float[npt.NDArray, "... 3"],
) -> jt.Float[npt.NDArray, "... 3"]:
    """Convert LiDAR points from sensor frame to ego vehicle coordinate system.

    Args:
        lidar_rot: LiDAR sensor rotation as [roll, pitch, yaw] in degrees.
        lidar_pos: LiDAR sensor position as [x, y, z] in meters.
        lidar: LiDAR point cloud as provided in the input of run_step.

    Returns:
        LiDAR points transformed to ego vehicle coordinate system.
    """
    rotation_matrix = euler_deg_to_mat(lidar_rot[0], lidar_rot[1], lidar_rot[2])
    translation = np.array(lidar_pos)
    lidar_points = (rotation_matrix @ lidar[1][:, :3].T).T + translation
    lidar_points[:, 2] = (
        lidar_points[:, 2] - lidar_pos[-1] / 2
    )  # Not sure why we need this :/
    return lidar_points


def radar_points_to_ego(
    raw_radar: jt.Float[npt.NDArray, "n 4"],
    sensor_pos: list[float],
    sensor_rot: list[float],
) -> jt.Float[npt.NDArray, "n 4"]:
    """Transform radar points from sensor frame to ego vehicle coordinate system.

    Args:
        raw_radar: Radar data of shape (N, 4) with [x, y, z, velocity] in sensor frame.
        sensor_pos: Sensor position [x, y, z] in ego frame (meters).
        sensor_rot: Sensor rotation [roll, pitch, yaw] in degrees (ego frame convention).

    Returns:
        Transformed radar points in ego coordinate system of shape (N, 4).
    """
    sensor_pos = [sensor_pos[0], sensor_pos[1], sensor_pos[2]]
    sensor_rot = [sensor_rot[0], sensor_rot[1], sensor_rot[2]]

    r = raw_radar[:, 0]
    alt = raw_radar[:, 1]
    az = raw_radar[:, 2]
    vel = raw_radar[:, 3]
    x = r * np.cos(az) * np.cos(alt)
    y = r * np.sin(az) * np.cos(alt)
    z = r * np.sin(alt)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    R_se = euler_deg_to_mat(*sensor_rot)
    pts_ego = (R_se @ pts.T).T + np.asarray(sensor_pos, dtype=np.float32).reshape(1, 3)

    pts_ego[:, 2] = pts_ego[:, 2] - sensor_pos[-1] / 2  # Not sure why we need this :/

    return np.concatenate([pts_ego, vel.reshape(-1, 1)], axis=1)


@beartype
def align_lidar(
    lidar: jt.Float[npt.NDArray, "n 3"],
    translation: jt.Float[npt.NDArray, " 3"],
    yaw: float,
) -> jt.Float[npt.NDArray, "n 3"]:
    """
    Translates and rotates a LiDAR into a new coordinate system.
    Rotation is inverse to translation and yaw

    Args:
        lidar: numpy LiDAR point cloud.
        translation: translations in meters.
        yaw: yaw angle in radians.

    Returns:
        numpy LiDAR point cloud in the new coordinate system.
    """
    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    return (rotation_matrix.T @ (lidar - translation).T).T


@beartype
def inverse_conversion_2d(
    point: jt.Float[npt.NDArray, " 2"],
    translation: jt.Float[npt.NDArray, " 2"],
    yaw: float,
) -> jt.Float[npt.NDArray, " 2"]:
    """
    Performs a forward coordinate conversion on a 2D point.

    Args:
        point: Point to be converted
        translation: 2D translation vector of the new coordinate system
        yaw: yaw in radian of the new coordinate system
    Returns:
        Converted point.
    """
    rotation_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
    )
    return rotation_matrix.T @ (point - translation)


@beartype
def conversion_2d(
    point: jt.Float[npt.NDArray, " 2"],
    translation: jt.Float[npt.NDArray, " 2"],
    yaw: float,
) -> jt.Float[npt.NDArray, " 2"]:
    """
    Performs a forward coordinate conversion on a 2D point

    Args:
        point: Point to be converted
        translation: 2D translation vector of the new coordinate system
        yaw: yaw in radian of the new coordinate system
    Returns:
        Converted point.
    """
    rotation_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
    )

    converted_point = rotation_matrix @ point + translation
    return converted_point


@beartype
def preprocess_compass(compass: float) -> float:
    """
    Checks the compass for Nans and rotates it into the default CARLA coordinate system with range [-pi,pi].

    Args:
        compass: compass value provided by the IMU, in radian

    Returns:
        float: yaw of the car in radian in the CARLA coordinate system.
    """
    if math.isnan(compass):  # simulation bug
        compass = 0.0
    # The minus 90.0 degree is because the compass sensor uses a different coordinate system then CARLA
    return normalize_angle(compass - np.deg2rad(90.0))


@beartype
def get_world_coordinate_2d(
    ego_transform: carla.Transform,
    local_location: carla.Location,
) -> carla.Location:
    """
    Ignore pitch and roll of car to get only 2D position with only yaw.
    """
    yaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    x = local_location.x
    y = local_location.y
    z = local_location.z

    world_x = ego_transform.location.x + cos_yaw * x - sin_yaw * y
    world_y = ego_transform.location.y + sin_yaw * x + cos_yaw * y
    world_z = ego_transform.location.z + z  # preserve vertical offset

    return carla.Location(x=world_x, y=world_y, z=world_z)


@beartype
def get_relative_transform(
    ego_matrix: npt.NDArray,
    vehicle_matrix: npt.NDArray,
) -> npt.NDArray:
    """Returns the position of the vehicle matrix in the ego coordinate system.

    Args:
        ego_matrix: 4x4 matrix of the ego vehicle in global coordinates.
        vehicle_matrix: 4x4 Matrix of another actor in global coordinates.
    Returns:
        3D position of the other vehicle in the ego coordinate system
    """
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    return rot @ relative_pos


def extract_yaw_from_matrix(matrix: jt.Float[npt.NDArray, "4 4"]) -> float:
    """Extract yaw angle from a CARLA world transformation matrix.

    Args:
        matrix: 4x4 transformation matrix from CARLA.

    Returns:
        Yaw angle in radians, normalized to [-π, π].
    """
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    yaw = normalize_angle(yaw)
    return yaw


def encode_depth_8bit(
    depth: jt.Float[npt.NDArray, "h w"],
) -> jt.UInt8[npt.NDArray, "h w"]:
    """Encode a depth map into 8-bit format for visualization.

    Clips depth values and scales them to fit within 0-255 range.

    Args:
        depth: Depth map with values in meters.

    Returns:
        8-bit encoded depth map with values scaled to 0-255.
    """
    depth = np.clip(depth, 0, 50)  # Clip to a maximum depth of 50 meters
    depth = (depth / 50) * 255  # Scale to 0-255
    return depth.astype(np.uint8)


def encode_depth_16bit(
    depth: jt.Float[npt.NDArray, "h w"],
) -> jt.UInt16[npt.NDArray, "h w"]:
    """Encode a depth map into 16-bit format for visualization.

    Clips depth values and scales them to fit within 0-65535 range.

    Args:
        depth: Depth map with values in meters.

    Returns:
        16-bit encoded depth map with values scaled to 0-65535.
    """
    depth = np.clip(depth, 0, 96)  # Clip to a maximum depth of 96 meters
    depth = (depth / 96) * (2**16 - 1)
    return depth.astype(np.uint16)


def decode_depth_16bit(
    encoded_depth: jt.UInt16[npt.NDArray, "h w"],
) -> jt.Float[npt.NDArray, "h w"]:
    """Decode a 16-bit encoded depth map back to original depth values.

    Args:
        encoded_depth: 16-bit encoded depth map.

    Returns:
        Decoded depth values in meters.
    """
    encoded_depth = encoded_depth.astype(np.float32)
    decoded_depth = (
        encoded_depth / (2**16 - 1)
    ) * 96  # Scale back to original depth range
    return decoded_depth


def decode_depth_8bit(
    encoded_depth: jt.UInt8[npt.NDArray, "h w"],
) -> jt.Float[npt.NDArray, "h w"]:
    """Decode an 8-bit encoded depth map back to original depth values.

    Args:
        encoded_depth: 8-bit encoded depth map.

    Returns:
        Decoded depth values in meters.
    """
    encoded_depth = encoded_depth.astype(np.float32)
    decoded_depth = (encoded_depth / 255) * 50  # Scale back to original depth range
    return decoded_depth


def decode_depth(
    encoded_depth: jt.Integer[npt.NDArray, "h w"],
) -> jt.Float[npt.NDArray, "h w"]:
    """Decode a depth map from 8-bit or 16-bit format back to original depth values.

    Automatically detects the input format and applies the appropriate decoding.

    Args:
        encoded_depth: Encoded depth map as uint8 or uint16 array.

    Returns:
        Decoded depth values in meters.

    Raises:
        ValueError: If the input data type is not uint8 or uint16.
    """
    if encoded_depth.dtype == np.uint8:
        return decode_depth_8bit(encoded_depth)
    elif encoded_depth.dtype == np.uint16:
        return decode_depth_16bit(encoded_depth)
    else:
        raise ValueError(
            "Unsupported data type for encoded depth. Expected uint8 or uint16.",
        )


@beartype
def waypoints_curvature(
    waypoints: jt.Float[torch.Tensor, "n 2"] | jt.Float[npt.NDArray, "n 2"],
) -> float:
    """Compute average absolute curvature of a waypoint trajectory.

    Args:
        waypoints: Trajectory waypoints as tensor of shape (N, 2).

    Returns:
        Average absolute curvature as scalar tensor.
    """
    if isinstance(waypoints, np.ndarray):
        waypoints = torch.from_numpy(waypoints)
    angles = torch.atan2(
        waypoints[:, 1],
        waypoints[:, 0],
    )  # Get angles in range [-pi, pi]
    return float(torch.mean(torch.abs(angles)))  # Compute average absolute angle


def waypoints_signed_curvature(waypoints: torch.Tensor) -> torch.Tensor:
    """Compute average signed curvature of a waypoint trajectory.

    Args:
        waypoints: Trajectory waypoints as tensor of shape (N, 2).

    Returns:
        Average signed curvature as scalar tensor.
    """
    angles = torch.atan2(
        waypoints[:, 1],
        waypoints[:, 0],
    )  # Get angles in range [-pi, pi]
    return torch.mean(angles)  # Compute average angle


@beartype
def average_displacement_error(
    predictions: torch.Tensor,
    observed_traj: torch.Tensor,
) -> float:
    """Compute L2 distance between proposed trajectories and ground truth.

    Args:
        predictions: A numpy array representing model predictions of size: [# batch, # time steps, spatial features].
        observed_traj: A tensor representing the observed trajectory in the logs of size [# batch, time steps, spatial features]

    Returns:
        float: L2 distance
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().float().numpy()
    if isinstance(observed_traj, torch.Tensor):
        observed_traj = observed_traj.detach().cpu().float().numpy()
    return float(
        np.linalg.norm(predictions - observed_traj, axis=-1).mean(axis=-1).mean(),
    )


@beartype
def final_displacement_error(
    predictions: torch.Tensor,
    observed_traj: torch.Tensor,
) -> float:
    """Compute final L2 distance between proposed trajectories and ground truth.

    Args:
        predictions: Model predictions of size: [# batch, # time steps, spatial features].
        observed_traj: Observed trajectory in the logs of size [# batch, time steps, spatial features]

    Returns:
        float: L2 distance
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().float().numpy()
    if isinstance(observed_traj, torch.Tensor):
        observed_traj = observed_traj.detach().cpu().float().numpy()
    return float(
        np.linalg.norm(predictions[:, -1] - observed_traj[:, -1], axis=-1).mean(),
    )


@beartype
def project_points_to_image(
    camera_rot: list[float],
    camera_pos: list[float],
    camera_fov: int | float,
    camera_width: int,
    camera_height: int,
    points: npt.NDArray,
) -> tuple[list[tuple[numbers.Real, numbers.Real]], list[bool]]:
    """
    Project 2D points (with z=0) to 2D image coordinates using camera parameters.

    Args:
        camera_rot: list of (roll, pitch, yaw) in degrees
        camera_pos: list of (x, y, z) camera position
        camera_fov: field of view in degrees (vertical FOV)
        camera_width: image width in pixels
        camera_height: image height in pixels
        points: numpy array of shape (N, 2) containing 2D points (z=0 assumed)

    Returns:
        list of (x, y) tuples for each point,
        list of booleans indicating if point is inside image bounds
    """

    # Convert inputs to numpy arrays
    camera_pos = np.array(camera_pos)
    points_2d = np.array(points)

    # Make points 3D
    points_3d = np.column_stack([points_2d, np.zeros(len(points_2d))])

    # Get rotation matrix using the provided function
    roll, pitch, yaw = camera_rot
    R = euler_deg_to_mat(roll, pitch, yaw)

    # Transform points to camera coordinate system
    # Translate points relative to camera position
    points_translated = points_3d - camera_pos

    # Rotate points to camera coordinate system
    points_camera = (R @ points_translated.T).T

    # Convert from world coordinates (x=forward, y=right, z=up)
    # to camera coordinates (x=right, y=down, z=forward)
    # This assumes your world coordinate system needs remapping
    points_cam_remapped = np.zeros_like(points_camera)
    points_cam_remapped[:, 0] = points_camera[:, 1]  # x_cam = y_world (right)
    points_cam_remapped[:, 1] = -points_camera[:, 2]  # y_cam = -z_world (down)
    points_cam_remapped[:, 2] = points_camera[:, 0]  # z_cam = x_world (forward)

    # Calculate camera intrinsic parameters
    fov_rad = np.radians(camera_fov)
    focal_length_y = camera_height / (2 * np.tan(fov_rad / 2))
    aspect_ratio = camera_width / camera_height
    focal_length_x = focal_length_y * aspect_ratio

    # Principal point (center of image)
    cx = camera_width / 2
    cy = camera_height / 2

    # Project points to image plane
    # Avoid division by zero for points behind camera
    z = points_cam_remapped[:, 2]
    valid_z = z > 1e-6  # Points must be in front of camera

    projected_points = []
    inside_image = []

    for i in range(len(points_cam_remapped)):
        if valid_z[i]:
            # Perspective projection
            x_img = (focal_length_x * points_cam_remapped[i, 0] / z[i]) + cx
            y_img = (focal_length_y * points_cam_remapped[i, 1] / z[i]) + cy

            # Check if point is inside image bounds
            inside = bool((0 <= x_img < camera_width) and (0 <= y_img < camera_height))

            projected_points.append((x_img, y_img))
            inside_image.append(inside)
        else:
            # Point is behind camera
            projected_points.append((0, 0))  # Default coordinate
            inside_image.append(False)

    return projected_points, inside_image


def rgb(r, g, b):
    """Help function to create RGB color tuples. Does not do much except for improving code readability with VSCode extension"""
    return (r, g, b)


def fov_crop(
    img: npt.NDArray,
    crop_pixels: int,
    chw: bool = False,
    interpolation: int = cv2.INTER_LINEAR,
) -> npt.NDArray:
    """Crop `crop_pixels` from left and right and resize back to original width.

    Supports (C, H, W) when chw=True, and (H, W) / (H, W, C) otherwise.
    """
    if chw:
        _, h, w = img.shape
        img = np.transpose(img, (1, 2, 0))  # -> (H, W_crop, C)
        img = img[:, crop_pixels:-crop_pixels, :]
        img = cv2.resize(img, (w, h), interpolation=interpolation)
        return np.transpose(img, (2, 0, 1))  # -> (C, H, W)
    else:
        h, w = img.shape[:2]
        img = (
            img[:, crop_pixels:-crop_pixels]
            if img.ndim == 2
            else img[:, crop_pixels:-crop_pixels, :]
        )
        return cv2.resize(img, (w, h), interpolation=interpolation)
