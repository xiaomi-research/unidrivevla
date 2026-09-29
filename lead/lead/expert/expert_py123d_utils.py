"""Utility functions for Py123D data conversion.

These utilities handle coordinate system conversions between CARLA (Unreal Engine)
and ISO 8855 (Py123D) coordinate systems.
"""

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype
from py123d.datatypes import EgoStateSE3Metadata, TrafficLightStatus
from py123d.geometry import (
    BoundingBoxSE3,
    EulerAngles,
    PoseSE3,
    PoseSE3Index,
    Quaternion,
    Vector3D,
)
from py123d.geometry.transform import (
    translate_se3_along_body_frame,
    translate_se3_along_z,
)
from py123d.parser.utils.sensor_utils.camera_conventions import (
    convert_camera_convention,
)


@beartype
def get_carla_lincoln_mkz_2020_metadata() -> EgoStateSE3Metadata:
    """Helper function to get CARLA Lincoln MKZ 2020 ego metadata."""
    # NOTE: These parameters are taken from the CARLA simulator vehicle model. The rear axles to center transform
    # parameters are calculated based on parameters from the CARLA simulator.
    return EgoStateSE3Metadata(
        vehicle_name="carla_lincoln_mkz_2020",
        width=1.83671,
        length=4.89238,
        height=1.49028,
        wheel_base=2.86048,
        center_to_imu_se3=PoseSE3(
            x=1.64855,
            y=0.0,
            z=0.38579,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


@beartype
def quaternion_from_carla_rotation(rotation: carla.Rotation) -> Quaternion:
    """Convert CARLA Euler rotation to Py123D quaternion in ISO 8855 coordinates.

    Args:
        rotation: CARLA rotation with roll, pitch, yaw in degrees.

    Returns:
        Quaternion with pitch and yaw inverted for ISO 8855 convention.
    """
    euler_angles = EulerAngles(
        roll=np.deg2rad(rotation.roll),
        pitch=-np.deg2rad(rotation.pitch),  # Invert for ISO 8855
        yaw=-np.deg2rad(rotation.yaw),  # Invert for ISO 8855
    )
    return Quaternion.from_rotation_matrix(euler_angles.rotation_matrix)


@beartype
def carla_actor_to_se3(
    actor: carla.Actor,
    half_height: float | None = None,
) -> PoseSE3:
    """Get SE3 pose for CARLA actor at geometric center in ISO 8855 coordinates.

    Args:
        actor: CARLA actor with transform and bounding box.
        half_height: Optional custom half-height to override actor's bounding box extent.

    Returns:
        PoseSE3 at actor's geometric center with Y-axis inverted.
    """
    transform = actor.get_transform()
    # Use provided half_height or default to actor's bounding box extent
    if half_height is None:
        half_height = actor.bounding_box.extent.z

    quaternion = quaternion_from_carla_rotation(transform.rotation)

    pose = PoseSE3(
        x=transform.location.x,
        y=-transform.location.y,  # Invert Y-axis for ISO 8855
        z=transform.location.z,
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    # Translate from bottom center to geometric center
    pose = translate_se3_along_z(pose, half_height)
    return pose


@beartype
def carla_velocity_to_vector3d(velocity: carla.Vector3D) -> Vector3D:
    """Convert CARLA velocity to Py123D Vector3D in ISO 8855 coordinates.

    Args:
        velocity: CARLA velocity vector.

    Returns:
        Vector3D with Y-axis inverted for ISO 8855 convention.
    """
    return Vector3D(
        x=velocity.x,
        y=-velocity.y,  # Invert Y-axis
        z=velocity.z,
    )


@beartype
def get_actor_bounding_box_se3(
    actor: carla.Actor,
    overwrite_extent: jt.Float[npt.NDArray, "3"] | list[float] | None = None,
) -> BoundingBoxSE3:
    """Extract bounding box in SE3 format from CARLA actor.

    Args:
        actor: CARLA actor with bounding box.
        overwrite_extent: Optional custom extent [x, y, z] to override actor's extent.

    Returns:
        BoundingBoxSE3 at actor's geometric center in ISO 8855 coordinates.
    """
    if overwrite_extent is None:
        extent = actor.bounding_box.extent
        extent_list = [extent.x, extent.y, extent.z]
        center_se3 = carla_actor_to_se3(actor)
    else:
        extent_list = overwrite_extent
        # Use overwrite_extent[2] as half_height for correct center calculation
        center_se3 = carla_actor_to_se3(actor, half_height=extent_list[2])

    return BoundingBoxSE3(
        center_se3=center_se3,
        length=extent_list[0] * 2.0,  # CARLA extent is half-dimensions
        width=extent_list[1] * 2.0,
        height=extent_list[2] * 2.0,
    )


@beartype
def get_bounding_box_se3(bb: dict) -> BoundingBoxSE3:
    """Extract bounding box in SE3 format from bounding box dictionary.

    Args:
        bb: Bounding box dict with 'matrix' (4x4 transform) and 'extent' [x, y, z].

    Returns:
        BoundingBoxSE3 in ISO 8855 coordinates.
    """
    # Extract world transformation matrix
    matrix = np.array(bb["matrix"])

    # Extract position from matrix
    x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]

    # Extract rotation from matrix and convert to quaternion
    # The rotation part is the upper-left 3x3 of the matrix
    rotation_matrix = matrix[:3, :3]

    # Extract yaw from the rotation matrix (assuming primarily yaw rotation)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Create quaternion from Euler angles (ISO 8855 convention: invert pitch and yaw)
    euler_angles = EulerAngles(
        roll=0.0,  # Assume no roll for static objects
        pitch=0.0,  # Assume no pitch for static objects
        yaw=-yaw,  # Invert for ISO 8855
    )
    quaternion = Quaternion.from_rotation_matrix(euler_angles.rotation_matrix)

    # Get extent (half-dimensions)
    extent = bb["extent"]
    half_height = extent[2]

    # Create pose at geometric center (translate from floor to center)
    pose = PoseSE3(
        x=x,
        y=-y,  # Invert Y-axis for ISO 8855
        z=z,
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    # Translate from floor center to geometric center
    pose = translate_se3_along_z(pose, half_height)

    return BoundingBoxSE3(
        center_se3=pose,
        length=extent[0] * 2.0,  # CARLA extent is half-dimensions
        width=extent[1] * 2.0,
        height=extent[2] * 2.0,
    )


@beartype
def get_bounding_box_velocity(bb: dict) -> Vector3D:
    """Get velocity from bounding box dictionary (zero for static objects).

    Args:
        bb: Bounding box dictionary.

    Returns:
        Vector3D with zero velocity.
    """
    # Static objects have zero velocity
    # If speed is available, we could approximate velocity along forward direction,
    # but for "static" and "static_prop_car" classes, they are not moving
    return Vector3D(x=0.0, y=0.0, z=0.0)


@beartype
def get_actor_velocity(actor: carla.Actor) -> Vector3D:
    """Get velocity from CARLA actor in ISO 8855 coordinates.

    Args:
        actor: CARLA actor.

    Returns:
        Vector3D velocity with Y-axis inverted.
    """
    velocity = actor.get_velocity()
    return carla_velocity_to_vector3d(velocity)


@beartype
def floor_center_to_rear_axle_translate(
    pose: PoseSE3,
    ego_metadata: EgoStateSE3Metadata,
) -> PoseSE3:
    """Translate pose from CARLA floor center to ISO 8855 rear axle frame.

    Args:
        pose: PoseSE3 in floor center reference.
        ego_metadata: Ego vehicle metadata with rear axle offsets.

    Returns:
        PoseSE3 translated to rear axle reference.
    """
    zero_pose = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    rear_axle_translate = translate_se3_along_body_frame(
        zero_pose,
        Vector3D(
            x=ego_metadata.rear_axle_to_center_longitudinal,
            y=0.0,
            z=-(ego_metadata.half_height) + ego_metadata.rear_axle_to_center_vertical,
        ),
    )
    pose.array[PoseSE3Index.XYZ] += rear_axle_translate.array[PoseSE3Index.XYZ]
    return pose


@beartype
def get_camera_extrinsic_as_iso(
    camera_pos: jt.Float[npt.NDArray, "3"] | list[float],
    camera_rot: jt.Float[npt.NDArray, "3"] | list[float],
    ego_metadata: EgoStateSE3Metadata,
) -> PoseSE3:
    """Convert camera extrinsic from Unreal Engine to ISO 8855 format.

    Args:
        camera_pos: Camera position [x, y, z] in Unreal coordinates.
        camera_rot: Camera rotation [roll, pitch, yaw] in degrees.
        ego_metadata: Ego vehicle metadata for rear axle offset.

    Returns:
        PoseSE3 camera extrinsic in ISO 8855/OpenCV convention.
    """
    # Create quaternion from rotation (pitch and yaw inverted for ISO 8855)
    quaternion = quaternion_from_carla_rotation(
        carla.Rotation(roll=camera_rot[0], pitch=camera_rot[1], yaw=camera_rot[2]),
    )

    # Create extrinsic pose with inverted Y-axis
    camera_extrinsic = PoseSE3(
        x=camera_pos[0],
        y=-camera_pos[1],  # Invert Y for ISO 8855
        z=camera_pos[2],
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    # Convert from floor center to rear axle reference
    camera_extrinsic = floor_center_to_rear_axle_translate(
        camera_extrinsic,
        ego_metadata,
    )

    # Convert camera convention from pXpZmY (Unreal) to pZmYpX (ISO 8855/OpenCV)
    camera_extrinsic = convert_camera_convention(
        camera_extrinsic,
        from_convention="pXpZmY",
        to_convention="pZmYpX",
    )

    return camera_extrinsic


@beartype
def carla_traffic_light_status_to_py123d(
    state: carla.TrafficLightState,
) -> TrafficLightStatus:
    """Convert CARLA TrafficLightState to Py123D TrafficLightStatus.

    Args:
        state: CARLA traffic light state.

    Returns:
        Corresponding Py123D TrafficLightStatus.
    """
    _MAP = {
        carla.TrafficLightState.Green: TrafficLightStatus.GREEN,
        carla.TrafficLightState.Yellow: TrafficLightStatus.YELLOW,
        carla.TrafficLightState.Red: TrafficLightStatus.RED,
        carla.TrafficLightState.Off: TrafficLightStatus.OFF,
        carla.TrafficLightState.Unknown: TrafficLightStatus.UNKNOWN,
    }
    return _MAP.get(state, TrafficLightStatus.UNKNOWN)
