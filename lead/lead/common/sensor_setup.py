import copy
import logging

import numpy as np
from beartype import beartype
from scipy.spatial.transform import Rotation as R

from lead.common import config_base
from lead.expert.config_expert import ExpertConfig
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


@beartype
def av_sensor_setup(
    config: ExpertConfig | TrainingConfig,
    perturbation_rotation: float,
    perturbation_translation: float,
    lidar: bool,
    perturbate: bool,
    sensor_agent: bool,
    radar: bool = False,
) -> list[dict]:
    """
    Function to set up sensors for an autonomous vehicle (AV) simulation.

    Args:
        config: Configuration object containing sensor parameters
        perturbation_rotation: Rotation perturbation in degrees
        perturbation_translation: Translation perturbation in meters
        lidar: Whether to include LiDAR sensors
        perturbate: Whether to create perturbated sensor variants
        sensor_agent: Whether this is for a sensor agent (affects which sensors are created)
        radar: Whether to include Radar sensors
    Returns:
        List of sensor configurations
    """
    result = camera_sensor_setup(
        config,
        perturbation_rotation,
        perturbation_translation,
        perturbate,
        sensor_agent,
    )
    if lidar:
        result.append(
            {
                "type": "sensor.lidar.ray_cast",
                "x": config.lidar_pos_1[0],
                "y": config.lidar_pos_1[1],
                "z": config.lidar_pos_1[2],
                "roll": config.lidar_rot_1[0],
                "pitch": config.lidar_rot_1[1],
                "yaw": config.lidar_rot_1[2],
                "id": "lidar1",
            },
        )
        LOG.info(
            f"Added sensor: {result[-1]['id']} at position ({config.lidar_pos_1[0]}, {config.lidar_pos_1[1]}, {config.lidar_pos_1[2]})",  # noqa: E501
        )
        if config.use_two_lidars:
            result.append(
                {
                    "type": "sensor.lidar.ray_cast",
                    "x": config.lidar_pos_2[0],
                    "y": config.lidar_pos_2[1],
                    "z": config.lidar_pos_2[2],
                    "roll": config.lidar_rot_2[0],
                    "pitch": config.lidar_rot_2[1],
                    "yaw": config.lidar_rot_2[2],
                    "id": "lidar2",
                },
            )
            LOG.info(
                f"Added sensor: {result[-1]['id']} at position ({config.lidar_pos_2[0]}, {config.lidar_pos_2[1]}, {config.lidar_pos_2[2]})",  # noqa: E501
            )
    if radar:
        for sensor_index, sensor_cfg in config.radar_calibration.items():
            result.append(
                {
                    "type": "sensor.other.radar",
                    "x": sensor_cfg["pos"][0],
                    "y": sensor_cfg["pos"][1],
                    "z": sensor_cfg["pos"][2],
                    "roll": sensor_cfg["rot"][0],
                    "pitch": sensor_cfg["rot"][1],
                    "yaw": sensor_cfg["rot"][2],
                    "horizontal_fov": sensor_cfg["horz_fov"],
                    "vertical_fov": sensor_cfg["vert_fov"],
                    "id": f"radar{sensor_index}",
                },
            )
            LOG.info(
                f"Added sensor: {result[-1]['id']} at position ({sensor_cfg['pos'][0]}, {sensor_cfg['pos'][1]}, {sensor_cfg['pos'][2]})",  # noqa: E501
            )
        if perturbate:
            for sensor_index, sensor_cfg in config.radar_calibration.items():
                result.append(
                    perturbated_sensor_cfg(
                        {
                            "type": "sensor.other.radar",
                            "x": sensor_cfg["pos"][0],
                            "y": sensor_cfg["pos"][1],
                            "z": sensor_cfg["pos"][2],
                            "roll": sensor_cfg["rot"][0],
                            "pitch": sensor_cfg["rot"][1],
                            "yaw": sensor_cfg["rot"][2],
                            "horizontal_fov": sensor_cfg["horz_fov"],
                            "vertical_fov": sensor_cfg["vert_fov"],
                            "id": f"radar{sensor_index}_perturbated",
                        },
                        perturbation_translation,
                        perturbation_rotation,
                    ),
                )
                LOG.info(
                    f"Added sensor: {result[-1]['id']} at position ({result[-1]['x']}, {result[-1]['y']}, {result[-1]['z']})",
                )

    result.append(
        {
            "type": "sensor.other.imu",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": config.carla_frame_rate,
            "id": "imu",
        },
    )
    LOG.info(f"Added sensor: {result[-1]['id']}")
    result.append(
        {
            "type": "sensor.other.gnss",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": 0.01,
            "id": "gps",
        },
    )
    LOG.info(f"Added sensor: {result[-1]['id']}")
    result.append(
        {
            "type": "sensor.speedometer",
            "reading_frequency": config.carla_fps,
            "id": "speed",
        },
    )
    LOG.info(f"Added sensor: {result[-1]['id']}")
    return result


def camera_sensor_setup(
    config: config_base.BaseConfig,
    perturbation_rotation: float,
    perturbation_translation: float,
    perturbate: bool,
    sensor_agent: bool,
) -> list:
    """Set up camera sensors for the given configuration.

    Args:
        config: Configuration object containing camera parameters
        perturbation_rotation: Rotation perturbation in degrees
        perturbation_translation: Translation perturbation in meters
        perturbate: Whether to create perturbated sensor variants
        sensor_agent: Whether this is for a sensor agent (affects which sensors are created)

    Returns:
        List of camera sensor configurations
    """
    result = []

    # Use dynamic camera indices based on num_cameras configuration
    camera_indices = list(range(1, config.num_cameras + 1))

    for idx in camera_indices:
        cam_config = config.camera_calibration[idx]
        cam_pos = cam_config["pos"]
        cam_rot = cam_config["rot"]
        cam_width = cam_config["width"]
        cam_height = cam_config["height"]
        camera_fov = cam_config["fov"]
        suffix = f"_{idx}"

        # RGB camera
        result.append(
            {
                "type": "sensor.camera.rgb",
                "x": cam_pos[0],
                "y": cam_pos[1],
                "z": cam_pos[2],
                "roll": cam_rot[0],
                "pitch": cam_rot[1],
                "yaw": cam_rot[2],
                "width": cam_width,
                "height": cam_height,
                "fov": camera_fov,
                "id": f"rgb{suffix}",
            },
        )
        LOG.info(
            f"Added sensor: {result[-1]['id']} at position ({cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]}), size: {cam_width}x{cam_height}px",  # noqa: E501
        )

        if not sensor_agent:
            # GT sensors
            for base_id, sensor_type in [
                ("semantics", "sensor.camera.semantic_segmentation"),
                ("depth", "sensor.camera.depth"),
                ("instance", "sensor.camera.instance_segmentation"),
            ]:
                result.append(
                    {
                        "type": sensor_type,
                        "x": cam_pos[0],
                        "y": cam_pos[1],
                        "z": cam_pos[2],
                        "roll": cam_rot[0],
                        "pitch": cam_rot[1],
                        "yaw": cam_rot[2],
                        "width": cam_width,
                        "height": cam_height,
                        "fov": camera_fov,
                        "id": f"{base_id}{suffix}",
                    },
                )
                LOG.info(
                    f"Added sensor: {result[-1]['id']} at position ({cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]}), size: {cam_width}x{cam_height}px",  # noqa: E501
                )

            # perturbated views
            if perturbate:
                for base_id, sensor_type in [
                    ("rgb", "sensor.camera.rgb"),
                    ("semantics", "sensor.camera.semantic_segmentation"),
                    ("depth", "sensor.camera.depth"),
                    ("instance", "sensor.camera.instance_segmentation"),
                ]:
                    result.append(
                        perturbated_sensor_cfg(
                            {
                                "type": sensor_type,
                                "x": cam_pos[0],
                                "y": cam_pos[1],
                                "z": cam_pos[2],
                                "roll": cam_rot[0],
                                "pitch": cam_rot[1],
                                "yaw": cam_rot[2],
                                "width": cam_width,
                                "height": cam_height,
                                "fov": camera_fov,
                                "id": f"{base_id}{suffix}_perturbated",
                            },
                            perturbation_translation,
                            perturbation_rotation,
                        ),
                    )
                    LOG.info(
                        f"Added perturbated sensor: {result[-1]['id']} at position ({result[-1]['x']}, {result[-1]['y']}, {result[-1]['z']}), size: {result[-1]['width']}x{result[-1]['height']}px",  # noqa: E501
                    )

    return result


@beartype
def perturbated_sensor_cfg(
    sensor_cfg: dict[str, float | str],
    perturbation_translation: float,
    perturbation_rotation: float,
    perturbation_roll: float = 0.0,
    perturbation_pitch: float = 0.0,
) -> dict[str, float | str]:
    """Apply a 3D rigid transformation (rotation + translation) to a sensor pose.

    Args:
        sensor_cfg: Dictionary containing the original sensor configuration
            with keys "x", "y", "z", "roll", "pitch", "yaw".
        perturbation_translation: Translation offset to apply along the Y-axis
            of the perturbation frame, in the same units as the position.
        perturbation_rotation: Rotation angle around the Z-axis (yaw), in degrees.
        perturbation_roll: Additional rotation angle around the X-axis (roll), in degrees.
        perturbation_pitch: Additional rotation angle around the Y-axis (pitch), in degrees.

    Returns:
        dict[str, float | str]: A new sensor configuration dictionary with updated
        "x", "y", "z", "roll", "pitch", and "yaw" values after applying
        the rigid transformation.
    """

    # Original pose
    pos = np.array([sensor_cfg["x"], sensor_cfg["y"], sensor_cfg["z"]])
    R0 = R.from_euler(
        "xyz",
        [sensor_cfg["roll"], sensor_cfg["pitch"], sensor_cfg["yaw"]],
        degrees=True,
    )

    # perturbation transform
    R_aug = R.from_euler(
        "xyz",
        [perturbation_roll, perturbation_pitch, perturbation_rotation],
        degrees=True,
    )
    t_aug = np.array(
        [0, perturbation_translation, 0],
    )  # translate along Y of perturbate frame

    # Apply 3D rigid transform
    pos_new = R_aug.apply(pos) + t_aug
    R_new = R_aug * R0  # compose rotations

    # Back to Euler
    roll, pitch, yaw = R_new.as_euler("xyz", degrees=True)

    sensor_cfg = copy.deepcopy(sensor_cfg)
    sensor_cfg.update(
        {
            "x": float(pos_new[0]),
            "y": float(pos_new[1]),
            "z": float(pos_new[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        },
    )
    return sensor_cfg


if __name__ == "__main__":
    # Example usage
    config = ExpertConfig()
    perturbation_rotation = 1.0  # degrees
    perturbation_translation = 1.0  # meters
    lidar = True
    perturbate = True
    sensor_agent = False
    radar = True

    sensors = av_sensor_setup(
        config,
        perturbation_rotation,
        perturbation_translation,
        lidar,
        perturbate,
        sensor_agent,
        radar,
    )
    for sensor in sensors:
        if "radar" not in sensor["type"]:
            continue
        print(sensor)
