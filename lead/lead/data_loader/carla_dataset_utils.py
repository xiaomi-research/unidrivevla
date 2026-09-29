from __future__ import annotations

import logging
from numbers import Real

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype

from lead.common import common_utils, constants, ransac
from lead.common.constants import (
    CONSTRUCTION_CONE_BB_SIZE,
    TRAFFIC_WARNING_BB_SIZE,
    RadarLabels,
    TransfuserBEVOccupancyClass,
    TransfuserBoundingBoxClass,
    TransfuserBoundingBoxIndex,
    TransfuserSemanticSegmentationClass,
)
from lead.data_loader.training_cache import SensorData
from lead.tfv6 import center_net_decoder as g_t
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


@beartype
def rasterize_lidar(
    config: TrainingConfig,
    lidar: jt.Float[npt.NDArray, "N 3"],
    remove_ground_plane: bool = False,
) -> jt.Float[npt.NDArray, "H W"]:
    """
    Convert LiDAR point cloud into pseudo-image.

    Args:
        config: Training configuration object.
        lidar: LiDAR point cloud.
        remove_ground_plane: whether to remove ground plane points.
    Returns:
        Sparse pseudo-image.
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(
            config.min_x_meter,
            config.max_x_meter,
            (config.max_x_meter - config.min_x_meter) * int(config.pixels_per_meter)
            + 1,
        )
        ybins = np.linspace(
            config.min_y_meter,
            config.max_y_meter,
            (config.max_y_meter - config.min_y_meter) * int(config.pixels_per_meter)
            + 1,
        )
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
        overhead_splat = hist / config.hist_max_per_pixel
        # The transpose here is an efficient axis swap.
        # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
        # (x height channel, y width channel)
        return overhead_splat.T

    # Remove points above the vehicle
    features = splat_points(lidar)
    lidar = lidar[
        (lidar[..., 2] <= config.max_height_lidar)
        & (config.min_height_lidar <= lidar[..., 2])
    ]
    if remove_ground_plane:
        is_ground_mask = ransac.remove_ground(
            lidar,
            config,
            parallel=True,
        )  # Torch parallel and dataloader seem to have issues with parallel numba.
        above = lidar[~is_ground_mask]
        features = splat_points(above)
    else:
        features = np.stack([splat_points(point_cloud=lidar)], axis=-1)
    return (features).squeeze().astype(np.float32)


@beartype
def image_augmenter(config: TrainingConfig, prob: float = 0.2):
    """Create an image augmenter for data perturbation.

    Args:
        config: Training configuration object.
        prob: Probability of applying each perturbation.

    Returns:
        Image augmenter.
    """
    import imgaug
    from imgaug import augmenters as ia

    imgaug.imgaug.seed(config.seed)
    perturbations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
        ia.Sometimes(
            prob,
            ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        ),
        ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),  # Strong
        ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
    ]
    return ia.Sequential(perturbations, random_order=True)


@beartype
def perturbate_route(
    route: jt.Float[npt.NDArray, "N 2"],
    y_perturbation: float = 0.0,
    yaw_perturbation: float = 0.0,
) -> jt.Float[npt.NDArray, "N 2"]:
    """Apply data perturbation to a route by rotating and translating waypoints.

    Args:
        route: Array of shape (N, 2) containing route waypoints in (x, y) coordinates.
        y_perturbation: Translation perturbation value in meters along y-axis.
        yaw_perturbation: Rotation perturbation value in degrees.

    Returns:
        Augmented route array of shape (N, 2) with transformed waypoints.
    """
    aug_yaw_rad = np.deg2rad(yaw_perturbation)
    rotation_matrix = np.array(
        [
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ],
    )

    translation = np.array([[0.0, y_perturbation]])
    route_aug = (rotation_matrix.T @ (route - translation).T).T
    return route_aug


@beartype
def perturbate_target_point(
    target_point: jt.Float[npt.NDArray, " 2"],
    y_perturbation: float = 0.0,
    yaw_perturbation: float = 0.0,
) -> jt.Float[npt.NDArray, " 2"]:
    """Apply data perturbation to a target point by rotating and translating it.

    Args:
        target_point: Array of shape (2,) containing target point coordinates (x, y).
        y_perturbation: Translation perturbation value in meters along y-axis.
        yaw_perturbation: Rotation perturbation value in degrees.

    Returns:
        Perturbated target point array of shape (2,) with transformed coordinates.
    """
    aug_yaw_rad = np.deg2rad(yaw_perturbation)
    rotation_matrix = np.array(
        [
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ],
    )

    translation = np.array([[0.0], [y_perturbation]])
    pos = np.expand_dims(target_point, axis=1)
    target_point_aug = rotation_matrix.T @ (pos - translation)
    return np.squeeze(target_point_aug)


@beartype
def perturbate_waypoints(
    waypoints: jt.Float[npt.NDArray, "N 2"],
    y_perturbation: float = 0.0,
    yaw_perturbation: float = 0.0,
) -> jt.Float[npt.NDArray, "N 2"]:
    """Apply data perturbation to waypoints by rotating and translating each point.

    Args:
        waypoints: Array of shape (N, 2) containing waypoint coordinates (x, y).
        y_perturbation: Translation perturbation value in meters along y-axis.
        yaw_perturbation: Rotation perturbation value in degrees.

    Returns:
        List of augmented waypoints with transformed coordinates.
    """
    # Data perturbation
    aug_yaw_rad = np.deg2rad(yaw_perturbation)
    rotation_matrix = np.array(
        [
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ],
    )

    translation = np.array([[0.0], [y_perturbation]])
    waypoints_aug = []
    for waypoint in waypoints:
        pos = np.expand_dims(waypoint, axis=1)
        waypoint_aug = rotation_matrix.T @ (pos - translation)
        waypoints_aug.append(np.squeeze(waypoint_aug))

    return np.array(waypoints_aug)


@beartype
def perturbate_yaws(
    yaws: jt.Float[npt.NDArray, " N"],
    yaw_perturbation: float = 0.0,
) -> npt.NDArray:
    """Apply data perturbation to yaw angles by subtracting the perturbation angle.

    Args:
        yaws: Array of shape (N,) containing yaw angles in radians.
        yaw_perturbation: Rotation perturbation value in degrees.

    Returns:
        Array of augmented yaw angles with normalized angles.
    """
    # Data perturbation
    yaws_aug = []
    aug_yaw_rad = np.deg2rad(yaw_perturbation)
    for yaw in yaws:
        yaw_aug = common_utils.normalize_angle(yaw - aug_yaw_rad)
        yaws_aug.append(yaw_aug)

    return np.array(yaws_aug)


@beartype
def bbox_json2array(
    bbox_dict: dict,
    perturbation_translation: float,
    perturbation_rotation: float,
    config: TrainingConfig,
) -> tuple[jt.Float[npt.NDArray, " 9"], jt.Float[npt.NDArray, " timesteps 2"], int]:
    """Extract and augment bounding box label from CARLA bounding box dictionary.

    Args:
        bbox_dict: Dictionary containing bounding box information from CARLA.
        perturbation_translation: Translation perturbation value in meters.
        perturbation_rotation: Rotation perturbation value in degrees.
        config: TrainingConfig object containing configuration parameters.

    Returns:
        Array with bounding boxs. Each row is a bounding box.
        Array with waypoints for each bounding box in ego-vehicle frame.
        Number of valid waypoints.
    """
    # perturbation
    aug_yaw_rad = np.deg2rad(perturbation_rotation)
    rotation_matrix = np.array(
        [
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ],
    )

    position = np.array([[bbox_dict["position"][0]], [bbox_dict["position"][1]]])
    translation = np.array([[0.0], [perturbation_translation]])
    num_radar_points = bbox_dict.get("num_radar_points", -1)

    position_aug = rotation_matrix.T @ (position - translation)

    x, y = position_aug[:2, 0]

    # center_x, center_y, w, h, yaw
    bbox = np.array(
        [
            x,
            y,
            bbox_dict["extent"][0],
            bbox_dict["extent"][1],
            0,
            0,
            0,
            0,
            num_radar_points,
        ],
        dtype=np.float32,
    )
    bbox[TransfuserBoundingBoxIndex.YAW] = common_utils.normalize_angle(
        bbox_dict["yaw"] - aug_yaw_rad,
    )

    if bbox_dict["class"] == "car":  # static class = parking vehicle = an implicit car
        bbox[TransfuserBoundingBoxIndex.VELOCITY] = bbox_dict["speed"]
        # check for nans
        if np.isnan(bbox_dict["brake"]):
            bbox[TransfuserBoundingBoxIndex.BRAKE] = 0
        else:
            bbox[TransfuserBoundingBoxIndex.BRAKE] = bbox_dict["brake"]
        if (
            "role_name" in bbox_dict
            and "scenario" in bbox_dict["role_name"]
            and bbox_dict["type_id"] in constants.EMERGENCY_MESHES
        ):
            if config.carla_leaderboard_mode:
                # this is an emergency vehicle that we need to yield to (or dodge in the RunningRedLight scenario)
                # so we give it a different label
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.SPECIAL
                )
            else:
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.VEHICLE
                )
        else:
            bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.VEHICLE
    elif bbox_dict["class"] == "walker":
        bbox[TransfuserBoundingBoxIndex.VELOCITY] = bbox_dict["speed"]
        bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.WALKER
    elif bbox_dict["class"] == "traffic_light":
        bbox[TransfuserBoundingBoxIndex.CLASS] = (
            TransfuserBoundingBoxClass.TRAFFIC_LIGHT
        )
    elif bbox_dict["class"] == "stop_sign":
        bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.STOP_SIGN
    elif bbox_dict["class"] == "static" and config.carla_leaderboard_mode:
        bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.PARKING

    waypoints = np.array(
        [(bbox_dict["position"][0], bbox_dict["position"][1])]
        * config.num_way_points_prediction,
        dtype=np.float32,
    )
    num_waypoints = 0
    if bbox_dict.get("future_positions") is not None:
        future_waypoint_indices = [config.waypoints_spacing]
        for _ in range(config.num_way_points_prediction - 1):
            future_waypoint_indices.append(
                future_waypoint_indices[-1] + config.waypoints_spacing,
            )

        # In CARLA, often vehicles disappear. But they are not removed but rather teleported far away.
        # To mitigate this, we check the distance between the last known position and the future waypoints.
        last_pos = np.array(
            [
                bbox_dict["position"][0],
                bbox_dict["position"][1],
            ],
        )
        last_valid_yaw = bbox_dict["yaw"]
        last_valid_speed = bbox_dict.get("speed", 0.0)
        for i, future_waypoint_index in enumerate(future_waypoint_indices):
            if future_waypoint_index < len(bbox_dict["future_positions"]):
                dist = np.linalg.norm(
                    last_pos - bbox_dict["future_positions"][future_waypoint_index],
                )
                if dist > config.max_distance_future_waypoint:
                    break
                waypoints[i] = bbox_dict["future_positions"][future_waypoint_index]
                num_waypoints += 1
                last_pos = bbox_dict["future_positions"][future_waypoint_index]
                last_valid_yaw = bbox_dict["future_yaws"][future_waypoint_index]
                last_valid_speed = (
                    bbox_dict["future_speeds"][future_waypoint_index]
                    if "future_speeds" in bbox_dict
                    else last_valid_speed
                )

        # Extrapolate last valid waypoint to mitigate disappearing boxes
        last_valid_waypoint = (
            waypoints[num_waypoints - 1]
            if num_waypoints > 0
            else np.array([bbox_dict["position"][0], bbox_dict["position"][1]])
        )
        if num_waypoints < config.num_way_points_prediction:
            dt = 1 / (config.waypoints_spacing - 1)  # seconds between waypoints
            for i in range(num_waypoints, config.num_way_points_prediction):
                # Extrapolate using constant velocity model
                dx = last_valid_speed * dt * np.cos(last_valid_yaw)
                dy = last_valid_speed * dt * np.sin(last_valid_yaw)

                # Update the waypoint position
                last_valid_waypoint = last_valid_waypoint + np.array([dx, dy])
                waypoints[i] = last_valid_waypoint.copy()

        waypoints = perturbate_waypoints(
            waypoints,
            y_perturbation=perturbation_translation,
            yaw_perturbation=perturbation_rotation,
        )
    return bbox, waypoints, num_waypoints


@beartype
def get_bbox_labels(
    data: dict,
    config: TrainingConfig,
    boxes: list[dict],
    current_measurement: dict,
    perturbation_translation: float,
    perturbation_rotation: float,
) -> tuple[
    jt.Float[npt.NDArray, "max_num_bbs 9"],
    jt.Float[npt.NDArray, "max_num_bbs timesteps 2"],
    jt.Int[npt.NDArray, " max_num_bbs"],
]:
    """Parse and filter bounding boxes from CARLA data.

    Args:
        data: The data dictionary containing scenario information.
        config: Configuration parameters for training.
        boxes: List of bounding box dictionaries from CARLA.
        current_measurement: Current measurement data, including scenario obstacle IDs.
        perturbation_translation: Translation perturbation value in meters.
        perturbation_rotation: Rotation perturbation value in degrees.

    Returns:
        Array containing parsed and filtered bounding boxes in image coordinates for center net.
    """
    bboxes, waypoints, num_waypoints = [], [], []

    for _, current_box in enumerate(boxes):
        bbox, waypoint, num_waypoint = bbox_json2array(
            current_box,
            perturbation_translation,
            perturbation_rotation,
            config,
        )
        if current_box["class"] in ["ego_car"]:
            continue

        # Occlusion check
        if "num_points" in current_box:
            num_points = current_box["num_points"]
            visible_pixels = -1
            if "visible_pixels" in current_box:
                visible_pixels = current_box["visible_pixels"]

            if (
                current_box["transfuser_semantics_id"]
                == TransfuserSemanticSegmentationClass.PEDESTRIAN
                and 0 <= num_points < config.pedestrian_min_num_lidar_points
                and 0 <= visible_pixels < config.pedestrian_min_num_visible_pixels
            ):
                continue
            if (
                current_box["transfuser_semantics_id"]
                == TransfuserSemanticSegmentationClass.VEHICLE
                and current_box["class"] == "static"
                and 0 <= num_points < config.parking_vehicle_min_num_lidar_points
                and 0 <= visible_pixels < config.parking_vehicle_min_num_visible_pixels
            ):
                continue
            if (
                current_box["transfuser_semantics_id"]
                == TransfuserSemanticSegmentationClass.VEHICLE
                and 0 <= num_points < config.vehicle_min_num_lidar_points
                and 0 <= visible_pixels < config.vehicle_min_num_visible_pixels
            ):
                continue

        # Only use/detect boxes that are red and affect the ego vehicle
        if current_box["class"] == "traffic_light":
            if not current_box["affects_ego"] or current_box["state"] == "Green":
                continue

        if current_box["class"] == "stop_sign":
            # Don't detect cleared stop signs.
            if not current_box["affects_ego"]:
                continue

        # Filter bb that are outside of the LiDAR after the perturbation.
        height = current_box["position"][2]
        if (
            bbox[TransfuserBoundingBoxIndex.X] <= config.min_x_meter
            or bbox[TransfuserBoundingBoxIndex.X] >= config.max_x_meter
            or bbox[TransfuserBoundingBoxIndex.Y] <= config.min_y_meter
            or bbox[TransfuserBoundingBoxIndex.Y] >= config.max_y_meter
            or height <= config.min_z
            or height >= config.max_z
        ):
            continue

        is_parking_vehicle = (
            current_box["class"] == "static"
            and current_box.get("mesh_path") is not None
            and "ParkedVehicles" in current_box["mesh_path"]
        )
        is_parking_vehicle = (
            is_parking_vehicle or current_box["class"] == "static_prop_car"
        )
        if (
            current_box["class"] == "static"
            and "type_id" in current_box
            and current_box["type_id"] not in config.data_bb_static_types_white_list
        ):
            if not is_parking_vehicle:
                continue

        if "type_id" in current_box:
            if current_box["type_id"] == "static.prop.trafficwarning":
                (
                    bbox[TransfuserBoundingBoxIndex.W],
                    bbox[TransfuserBoundingBoxIndex.H],
                ) = (
                    TRAFFIC_WARNING_BB_SIZE[0],
                    TRAFFIC_WARNING_BB_SIZE[1],
                )
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.OBSTACLE
                )
            elif current_box["type_id"] == "static.prop.constructioncone":
                (
                    bbox[TransfuserBoundingBoxIndex.W],
                    bbox[TransfuserBoundingBoxIndex.H],
                ) = (
                    CONSTRUCTION_CONE_BB_SIZE[0],
                    CONSTRUCTION_CONE_BB_SIZE[1],
                )
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.OBSTACLE
                )

        if "mesh_path" in current_box:
            if current_box["mesh_path"] in constants.LOOKUP_TABLE:
                bbox[TransfuserBoundingBoxIndex.W] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][0]
                bbox[TransfuserBoundingBoxIndex.H] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][1]
            if is_parking_vehicle and config.carla_leaderboard_mode:
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.PARKING
                )

        if current_box["class"] == "static_prop_car":
            if config.carla_leaderboard_mode:
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.PARKING
                )
            else:
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.VEHICLE
                )

        # In some CARLA's scenarios, special vehicles have sometimes wrong labels.
        # We relabel those labels to be consistent with the scene
        if (
            current_box["class"] == "car"
            and "role_name" in current_box
            and "scenario" in current_box["role_name"]
            and current_box["speed"] < 0.1
        ):
            if (
                data["scenario_type"] == "VehicleOpensDoorTwoWays"
                and current_box["id"] in current_measurement["scenario_obstacles_ids"]
            ):
                # If the car open door, we extend the bounding box's width to consider the door
                if current_measurement["vehicle_opened_door"]:
                    bbox[TransfuserBoundingBoxIndex.H] += (
                        config.car_open_door_extra_width / 2
                    )
                    if current_measurement["vehicle_door_side"] == "left":
                        bbox[TransfuserBoundingBoxIndex.Y] += (
                            config.car_open_door_extra_width / 2
                        )
                    else:
                        bbox[TransfuserBoundingBoxIndex.Y] -= (
                            config.car_open_door_extra_width / 2
                        )

                if (
                    bbox[TransfuserBoundingBoxIndex.CLASS]
                    != TransfuserBoundingBoxClass.SPECIAL
                ):
                    if config.carla_leaderboard_mode:
                        if current_measurement["vehicle_opened_door"]:
                            bbox[TransfuserBoundingBoxIndex.CLASS] = (
                                TransfuserBoundingBoxClass.OBSTACLE
                            )
                        else:
                            bbox[TransfuserBoundingBoxIndex.CLASS] = (
                                TransfuserBoundingBoxClass.PARKING
                            )
                    else:
                        bbox[TransfuserBoundingBoxIndex.CLASS] = (
                            TransfuserBoundingBoxClass.VEHICLE
                        )
            elif (
                bbox[TransfuserBoundingBoxIndex.CLASS]
                != TransfuserBoundingBoxClass.SPECIAL
            ):
                if config.carla_leaderboard_mode:
                    if (
                        data["scenario_type"]
                        in [
                            "Accident",
                            "AccidentTwoWays",
                            "BlockedIntersection",
                            "ParkedObstacle",
                            "ParkedObstacleTwoWays",
                        ]
                        and current_box["id"]
                        in current_measurement["scenario_obstacles_ids"]
                    ):
                        bbox[TransfuserBoundingBoxIndex.CLASS] = (
                            TransfuserBoundingBoxClass.OBSTACLE
                        )
                    elif (
                        data["scenario_type"]
                        in [
                            "ParkingCrossingPedestrian",
                            "ParkingCutIn",
                            "StaticCutIn",
                            "PedestrianCrossing",
                            "ParkingExit",
                        ]
                        and current_box["lane_type_str"] == "Parking"
                    ) and config.carla_leaderboard_mode:
                        bbox[TransfuserBoundingBoxIndex.CLASS] = (
                            TransfuserBoundingBoxClass.PARKING
                        )
                else:
                    bbox[TransfuserBoundingBoxIndex.CLASS] = (
                        TransfuserBoundingBoxClass.VEHICLE
                    )

        if current_box.get("type_id") in constants.BIKER_MESHES:
            bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.BIKER

        bbox = bb_vehicle_to_image_system(
            bbox.reshape(1, -1),
            config.pixels_per_meter,
            config.min_x_meter,
            config.min_y_meter,
        ).squeeze()
        if not (0 <= bbox[TransfuserBoundingBoxIndex.X] < config.lidar_width_pixel):
            LOG.warning(
                f"{bbox[TransfuserBoundingBoxIndex.X]=} is larger than {config.lidar_width_pixel=}",
            )
            continue
        if not (0 <= bbox[TransfuserBoundingBoxIndex.Y] < config.lidar_height_pixel):
            LOG.warning(
                f"{bbox[TransfuserBoundingBoxIndex.Y]=} is larger than {config.lidar_height_pixel=}",
            )
            continue
        bboxes.append(bbox)
        waypoints.append(waypoint)
        num_waypoints.append(num_waypoint)

    bounding_boxes_array = np.array(bboxes)
    waypoints_array = np.array(waypoints)
    num_waypoints_array = np.array(num_waypoints)

    # Pad bounding boxes to a fixed number
    padded_bounding_boxes_array = np.zeros((config.max_num_bbs, 9), dtype=np.float32)
    padded_waypoints_array = np.zeros(
        (config.max_num_bbs, config.num_way_points_prediction, 2),
        dtype=np.float32,
    )
    padded_num_waypoints_array = np.zeros((config.max_num_bbs,), dtype=np.int32)

    if bounding_boxes_array.shape[0] > 0:
        if bounding_boxes_array.shape[0] <= config.max_num_bbs:
            padded_bounding_boxes_array[: bounding_boxes_array.shape[0], :] = (
                bounding_boxes_array
            )
            padded_waypoints_array[: bounding_boxes_array.shape[0], :, :] = (
                waypoints_array
            )
            padded_num_waypoints_array[: bounding_boxes_array.shape[0]] = (
                num_waypoints_array
            )
        else:
            padded_bounding_boxes_array[: config.max_num_bbs, :] = bounding_boxes_array[
                : config.max_num_bbs
            ]
            padded_waypoints_array[: config.max_num_bbs, :, :] = waypoints_array[
                : config.max_num_bbs,
                :,
                :,
            ]
            padded_num_waypoints_array[: config.max_num_bbs] = num_waypoints_array[
                : config.max_num_bbs
            ]

    return (
        padded_bounding_boxes_array,
        padded_waypoints_array,
        padded_num_waypoints_array,
    )


@beartype
def get_centernet_labels(
    gt_bboxes: jt.Float[npt.NDArray, "N 9"],
    config: TrainingConfig,
    num_bb_classes: int,
) -> dict[str, npt.NDArray]:
    """
    Compute regression and classification targets for CenterNet.

    Args:
        gt_bboxes: Ground truth bboxes for each image with shape (N, 11). Coordinates in image frame.
        config: TrainingConfig object containing configuration parameters.
        num_bb_classes: Number of bounding box classes.
    Returns:
        A dictionary containing various target tensors for training the CenterNet model.
    """
    feat_h = config.lidar_height_meter
    feat_w = config.lidar_width_meter

    center_heatmap_target = np.zeros([num_bb_classes, feat_h, feat_w], dtype=np.float32)
    wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    pixel_weight = np.zeros(
        [2, feat_h, feat_w],
        dtype=np.float32,
    )  # 2 is the max of the channels above here.

    if not gt_bboxes.shape[0] > 0:
        return {
            "center_net_bounding_boxes": gt_bboxes,
            "center_net_heatmap": center_heatmap_target,
            "center_net_wh": wh_target,
            "center_net_yaw_class": yaw_class_target.squeeze(0),
            "center_net_yaw_res": yaw_res_target,
            "center_net_offset": offset_target,
            "center_net_velocity": velocity_target,
            "center_net_brake": brake_target.squeeze(0),
            "center_net_pixel_weight": pixel_weight,
            "center_net_avg_factor": np.array([1]),
        }

    center_x = (
        gt_bboxes[:, [TransfuserBoundingBoxIndex.X]] / config.bev_down_sample_factor
    )
    center_y = (
        gt_bboxes[:, [TransfuserBoundingBoxIndex.Y]] / config.bev_down_sample_factor
    )
    gt_centers = np.concatenate((center_x, center_y), axis=1)

    for j, ct in enumerate(gt_centers):
        ctx_int, cty_int = ct.astype(int)
        ctx, cty = ct
        if ctx_int < 0 or ctx_int >= feat_w or cty_int < 0 or cty_int >= feat_h:
            LOG.warning(
                f"Be cautious! Bounding box center {ct} is out of bounds for image size ({feat_h}, {feat_w}).",
                flush=True,
            )
            continue

        extent_x = (
            gt_bboxes[j, TransfuserBoundingBoxIndex.W] / config.bev_down_sample_factor
        )
        extent_y = (
            gt_bboxes[j, TransfuserBoundingBoxIndex.H] / config.bev_down_sample_factor
        )

        radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
        radius = max(2, int(radius))
        ind = gt_bboxes[j, TransfuserBoundingBoxIndex.CLASS].astype(int)
        if not config.carla_leaderboard_mode:
            ind = constants.SIM2REAL_BOUNDING_BOX_CLASS_CONVERTER[ind]

        g_t.gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

        wh_target[0, cty_int, ctx_int] = extent_x
        wh_target[1, cty_int, ctx_int] = extent_y

        yaw_class, yaw_res = common_utils.angle2class(
            gt_bboxes[j, TransfuserBoundingBoxIndex.YAW],
            config.num_dir_bins,
        )

        yaw_class_target[0, cty_int, ctx_int] = yaw_class
        yaw_res_target[0, cty_int, ctx_int] = yaw_res

        velocity_target[0, cty_int, ctx_int] = gt_bboxes[
            j,
            TransfuserBoundingBoxIndex.VELOCITY,
        ]
        # Brakes can potentially be continous but we classify them now.
        # Using mathematical rounding the split is applied at 0.5
        brake_target[0, cty_int, ctx_int] = int(
            round(gt_bboxes[j, TransfuserBoundingBoxIndex.BRAKE]),
        )

        offset_target[0, cty_int, ctx_int] = ctx - ctx_int
        offset_target[1, cty_int, ctx_int] = cty - cty_int
        # All pixels with a bounding box have a weight of 1 all others have a weight of 0.
        # Used to ignore the pixels without bbs in the loss.
        pixel_weight[:, cty_int, ctx_int] = 1.0

    avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
    return {
        "center_net_bounding_boxes": gt_bboxes,
        "center_net_heatmap": center_heatmap_target,
        "center_net_wh": wh_target,
        "center_net_yaw_class": yaw_class_target.squeeze(0),
        "center_net_yaw_res": yaw_res_target,
        "center_net_offset": offset_target,
        "center_net_velocity": velocity_target,
        "center_net_brake": brake_target.squeeze(0),
        "center_net_pixel_weight": pixel_weight,
        "center_net_avg_factor": avg_factor,
    }


@beartype
def build_bev_occupancy(
    data: dict,
    current_measurement: dict,
    json_boxes: list,
    config: TrainingConfig,
    y_perturbation: float,
    yaw_perturbation: float,
) -> jt.UInt8[npt.NDArray, "H W"]:
    """Build bird's eye view occupancy map from bounding box data.

    Creates a semantic occupancy map by projecting bounding boxes onto a grid
    and applying data perturbation. Handles various object types including vehicles,
    pedestrians, traffic lights, and construction zones.

    Args:
        data: Dictionary containing scenario information.
        current_measurement: Current measurement data including scenario obstacles.
        json_boxes: List of bounding box dictionaries from CARLA.
        config: Training configuration object.
        y_perturbation: Translation perturbation value in meters.
        yaw_perturbation: Rotation perturbation value in degrees.

    Returns:
        Bird's eye view occupancy map as integer array of shape.
    """
    scale = 4
    grid_size = 256 * scale
    bev = np.zeros((grid_size, grid_size), dtype=np.uint8)
    aug_yaw_rad = np.deg2rad(yaw_perturbation)
    rot_mat = np.array(
        [
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ],
    )
    translation = np.array([[0.0], [y_perturbation]])

    obstacle_scenario_corners = []
    cone_corners = []
    warning_corners = []
    normal_red_light_corners = []
    unnormal_red_light_corners = []
    green_light_corners = []

    for current_box in json_boxes:
        cls = current_box["class"]
        if cls not in ["car", "walker", "static", "static_prop_car", "traffic_light"]:
            continue
        extra_scale = 1.0
        min_extent = 0.0
        if cls in ["walker"] or (
            "type_id" in current_box
            and current_box["type_id"] in constants.BIKER_MESHES
        ):
            extra_scale = config.scale_pedestrian_bev_semantic_size
            min_extent = config.pedestrian_bev_min_extent
        # Apply perturbation to position
        pos = np.array([[current_box["position"][0]], [current_box["position"][1]]])
        pos = rot_mat.T @ (pos - translation)
        x, y = pos[:, 0]
        y = -y

        cx = (x + 128.0) * scale
        cy = (128.0 - y) * scale
        if not (0 <= cx < grid_size and 0 <= cy < grid_size):
            continue

        # Apply perturbation to yaw
        yaw = common_utils.normalize_angle(current_box["yaw"] - aug_yaw_rad)
        extent_x, extent_y = current_box["extent"][:2]
        extent_x = max(extent_x, min_extent)
        extent_y = max(extent_y, min_extent)
        if (
            current_box["class"] == "car"
            and "role_name" in current_box
            and "scenario" in current_box["role_name"]
            and current_box["speed"] < 0.1
            and current_box["id"] in current_measurement["scenario_obstacles_ids"]
        ):
            if data["scenario_type"] == "VehicleOpensDoorTwoWays":
                if current_measurement["vehicle_opened_door"]:
                    # This is a special case where we open the door, so we extend the bounding box's width
                    extent_y += config.car_open_door_extra_width / 2
                    if current_measurement["vehicle_door_side"] == "left":
                        cy += config.car_open_door_extra_width / 2 * scale
                    else:
                        cy -= config.car_open_door_extra_width / 2 * scale

        if "mesh_path" in current_box:
            if current_box["mesh_path"] in constants.LOOKUP_TABLE:
                extent_x = constants.LOOKUP_TABLE[current_box["mesh_path"]][0]
                extent_y = constants.LOOKUP_TABLE[current_box["mesh_path"]][1]
        elif "type_id" in current_box:
            if current_box["type_id"] == "static.prop.trafficwarning":
                extent_x, extent_y = (
                    TRAFFIC_WARNING_BB_SIZE[0],
                    TRAFFIC_WARNING_BB_SIZE[1],
                )
            elif current_box["type_id"] == "static.prop.constructioncone":
                extent_x, extent_y = (
                    CONSTRUCTION_CONE_BB_SIZE[0],
                    CONSTRUCTION_CONE_BB_SIZE[1],
                )
        rect = (
            (cx, cy),
            (extent_x * 2 * scale * extra_scale, extent_y * 2 * scale * extra_scale),
            np.rad2deg(yaw),
        )
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        # Assign label
        is_parking_car = False
        label = TransfuserBEVOccupancyClass.VEHICLE
        if cls in ["car"]:
            if (
                current_box.get("type_id", "") in constants.EMERGENCY_MESHES
                and config.carla_leaderboard_mode
            ):
                if (
                    data["scenario_type"]
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                ):
                    label = TransfuserBEVOccupancyClass.OBSTACLE
                    obstacle_scenario_corners.extend(box_pts.tolist())
                else:
                    label = TransfuserBEVOccupancyClass.SPECIAL_VEHICLE
            elif current_box.get("type_id", "") in constants.BIKER_MESHES:
                label = TransfuserBEVOccupancyClass.BIKER
            elif (
                "role_name" in current_box
                and "scenario" in current_box["role_name"]
                and current_box["speed"] < 0.1
                and config.carla_leaderboard_mode
            ):
                if (
                    data["scenario_type"]
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                ):
                    obstacle_scenario_corners.extend(box_pts.tolist())
                    label = TransfuserBEVOccupancyClass.OBSTACLE
                elif (
                    data["scenario_type"] in ["VehicleOpensDoorTwoWays"]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                ):
                    if current_measurement["vehicle_opened_door"]:
                        label = TransfuserBEVOccupancyClass.OBSTACLE
                        obstacle_scenario_corners.extend(box_pts.tolist())
                    else:
                        label = TransfuserBEVOccupancyClass.PARKING_VEHICLE
                elif (
                    data["scenario_type"]
                    in [
                        "ParkingCrossingPedestrian",
                        "ParkingCutIn",
                        "StaticCutIn",
                        "PedestrianCrossing",
                        "ParkingExit",
                    ]
                    and current_box["lane_type_str"] == "Parking"
                ):
                    label = TransfuserBEVOccupancyClass.PARKING_VEHICLE
        elif cls == "walker":
            label = TransfuserBEVOccupancyClass.WALKER
        elif cls == "static_prop_car" and config.carla_leaderboard_mode:
            label = TransfuserBEVOccupancyClass.PARKING_VEHICLE
        elif cls == "static":
            type_id = current_box.get("type_id", "")
            is_parking_car = (
                current_box.get("mesh_path") is not None
                and "ParkedVehicles" in current_box["mesh_path"]
            )
            if type_id in config.data_bb_static_types_white_list:
                label = TransfuserBEVOccupancyClass.OBSTACLE
                if type_id == "static.prop.constructioncone":
                    cone_corners.extend(box_pts.tolist())
                elif type_id == "static.prop.trafficwarning":
                    warning_corners.extend(box_pts.tolist())
            elif (
                type_id in constants.EMERGENCY_MESHES and config.carla_leaderboard_mode
            ):
                label = TransfuserBEVOccupancyClass.SPECIAL_VEHICLE
                obstacle_scenario_corners.extend(box_pts.tolist())
            elif is_parking_car and config.carla_leaderboard_mode:
                label = TransfuserBEVOccupancyClass.PARKING_VEHICLE
            else:
                continue
        elif cls == "traffic_light":
            if current_box["affects_ego"] and current_box["state"] in ["Red", "Yellow"]:
                if (
                    not current_measurement["over_head_traffic_light"]
                    and not current_measurement["europe_traffic_light"]
                ):
                    label = TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL
                    normal_red_light_corners.extend(box_pts.tolist())
                else:
                    label = TransfuserBEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL
                    unnormal_red_light_corners.extend(box_pts.tolist())
            elif current_box["affects_ego"]:
                label = TransfuserBEVOccupancyClass.TRAFFIC_GREEN
                green_light_corners.extend(box_pts.tolist())
            else:
                continue

        # Occlusion check
        if (
            cls == "walker"
            and (
                0 <= current_box["num_points"] < config.pedestrian_min_num_lidar_points
            )
            and (
                0
                <= current_box["visible_pixels"]
                < config.pedestrian_min_num_visible_pixels
            )
        ):
            continue
        if (
            cls == "car"
            and (0 <= current_box["num_points"] < config.vehicle_min_num_lidar_points)
            and (
                0
                <= current_box["visible_pixels"]
                < config.vehicle_min_num_visible_pixels
            )
        ):
            continue
        if (
            cls == "static"
            and (
                0
                <= current_box["num_points"]
                < config.parking_vehicle_min_num_lidar_points
            )
            and (
                0
                <= current_box["visible_pixels"]
                < config.parking_vehicle_min_num_visible_pixels
            )
        ):
            continue
        if (
            cls == "static_prop_car"
            and (
                0
                <= current_box["num_points"]
                < config.parking_vehicle_min_num_lidar_points
            )
            and (
                0
                <= current_box["visible_pixels"]
                < config.parking_vehicle_min_num_visible_pixels
            )
        ):
            continue
        cv2.fillPoly(bev, [box_pts], label)

    # Construction site detection
    if (
        len(warning_corners) >= 3
        and len(cone_corners) >= 24
        and config.carla_leaderboard_mode
    ):
        all_pts = np.array(cone_corners + warning_corners)
        mean = np.mean(all_pts, axis=0)
        dists = np.linalg.norm(all_pts - mean, axis=1)

        if np.all(dists <= 24 * scale):
            hull = cv2.convexHull(all_pts)
            cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.OBSTACLE)

    # Accident and obstacle scenario detection
    if len(obstacle_scenario_corners) >= 3 and config.carla_leaderboard_mode:
        all_pts = np.array(obstacle_scenario_corners)
        mean = np.mean(all_pts, axis=0)
        dists = np.linalg.norm(all_pts - mean, axis=1)

        if np.all(dists <= 48 * scale):
            hull = cv2.convexHull(all_pts)
            cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.OBSTACLE)

    # Red light detection
    if len(unnormal_red_light_corners) >= 3:
        hull = cv2.convexHull(np.array(unnormal_red_light_corners))
        cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL)
    elif len(normal_red_light_corners) >= 3:
        hull = cv2.convexHull(np.array(normal_red_light_corners))
        cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL)
    elif len(green_light_corners) >= 3:
        hull = cv2.convexHull(np.array(green_light_corners))
        cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.TRAFFIC_GREEN)
    return bev


@jt.jaxtyped(typechecker=beartype)
def bb_vehicle_to_image_system(
    box: jt.Float[npt.NDArray, "N D"],
    pixels_per_meter: Real,
    min_x: Real,
    min_y: Real,
) -> jt.Float[npt.NDArray, "N D"]:
    """
    Changed a bounding box from the vehicle coordinate system to the image coordinate system.

    Args:
        box: bounding box in the vehicle coordinate system.
        pixels_per_meter: scaling factor from meters to pixels
        min_x: minimum x value of the image in the vehicle coordinate system
        min_y: minimum y value of the image in the vehicle coordinate system

    Returns:
        box: bounding box in the image coordinate system.
    """
    box = box.copy()
    box[:, :2] = box[:, :2] - np.array([min_x, min_y])
    box[:, :4] = box[:, :4] * pixels_per_meter
    return box


@jt.jaxtyped(typechecker=beartype)
def bb_image_to_vehicle_system(
    box: jt.Float[npt.NDArray, "N D"],
    pixels_per_meter: Real,
    min_x: Real,
    min_y: Real,
) -> jt.Float[npt.NDArray, "N D"]:
    """Inverse of bb_vehicle_to_image_system.

    Args:
        box: bounding box in the image coordinate system.
        pixels_per_meter: scaling factor from meters to pixels
        min_x: minimum x value of the image in the vehicle coordinate system
        min_y: minimum y value of the image in the vehicle coordinate system

    Returns:
        box: bounding box in the vehicle coordinate system.
    """
    box = box.copy()
    box[:, :4] = box[:, :4] / pixels_per_meter
    box[:, :2] = box[:, :2] + np.array([min_x, min_y])
    return box


@beartype
def preprocess_radar_input(
    config: TrainingConfig,
    radar_data_dict: dict,
) -> list[jt.Float[npt.NDArray, "N 5"]]:
    """Preprocess radar input data for model inference.

    Args:
        config: Training configuration containing radar parameters.
        radar_data_dict: Dictionary containing radar data from sensors (e.g., {"radar1": array, "radar2": array, ...}).

    Returns:
        List of preprocessed radar data with sensor ID as last column.
    """
    if not config.use_radars:
        # Return empty array with correct shape
        return np.zeros((0, 5), dtype=np.float32)

    def filter_and_pad_radars(arr):
        # Filter points within spatial bounds
        x_mask = (arr[:, 0] >= config.min_x_meter) & (arr[:, 0] <= config.max_x_meter)
        y_mask = (arr[:, 1] >= config.min_y_meter) & (arr[:, 1] <= config.max_y_meter)
        valid_mask = x_mask & y_mask
        filtered_arr = arr[valid_mask]

        # Pad the filtered array
        n = filtered_arr.shape[0]
        if n >= config.num_radar_points_per_sensor:
            return filtered_arr[: config.num_radar_points_per_sensor]
        out = np.zeros(
            (config.num_radar_points_per_sensor, filtered_arr.shape[1]),
            dtype=np.float32,
        )
        out[:n] = filtered_arr.astype(np.float32)
        return out

    radar_list = []
    for i in range(1, config.num_radar_sensors + 1):
        padded_radar = filter_and_pad_radars(radar_data_dict[f"radar{i}"])

        # Add sensor identity column (0-indexed)
        sensor_id = np.full((padded_radar.shape[0], 1), float(i - 1), dtype=np.float32)
        radar_with_id = np.concatenate([padded_radar, sensor_id], axis=1)
        radar_list.append(radar_with_id)
    return radar_list


@beartype
def parse_radar_detection_labels(
    config: TrainingConfig,
    sensor_data: SensorData,
) -> jt.Float32[npt.NDArray, "num_queries features"]:
    """Parse and filter radar detection labels from sensor data for model training.

    This function extracts radar-based object detections from bounding box data, filtering
    and prioritizing detections based on radar point coverage, object velocity, and class
    importance. It converts detections to vehicle coordinates and outputs a fixed-size array
    suitable for model consumption.

    The selection process prioritizes detections by:
    1. Higher velocity (more relevant for collision avoidance)
    2. Class priority (SPECIAL > VEHICLE > WALKER > OBSTACLE > PARKING)
    3. More radar measurement points (higher confidence)

    Args:
        config: Training configuration containing radar parameters including:
            - use_radars: Whether radar processing is enabled
            - num_radar_queries: Maximum number of radar detections to output
            - pixels_per_meter, min_x_meter, min_y_meter: Coordinate system parameters
        sensor_data: Sensor data container with bounding boxes, waypoints, and metadata.
            Must have non-None boxes attribute if radar processing is enabled.

    Returns:
        Array of shape (num_radar_queries, num_features) containing radar detection labels.
        Each row represents one detection with features [x, y, velocity, valid_flag].
        Unused slots are zero-padded. Features are in vehicle coordinate system.
    """
    # Initialize default values (all zeros)
    radar_detections = np.zeros(
        (config.num_radar_queries, len(RadarLabels)),
        dtype=np.float32,
    )

    if (
        config.use_radars
        and sensor_data.boxes is not None
        and sensor_data.boxes.shape[0] > 0
    ):
        priority_classes = [
            TransfuserBoundingBoxClass.SPECIAL,
            TransfuserBoundingBoxClass.VEHICLE,
            TransfuserBoundingBoxClass.WALKER,
            TransfuserBoundingBoxClass.OBSTACLE,
            TransfuserBoundingBoxClass.PARKING,
        ]

        # Copy data
        loaded_boxes_image_system = sensor_data.boxes.copy()
        loaded_waypoints = sensor_data.boxes_waypoints.copy()
        loaded_num_waypoints = sensor_data.boxes_num_waypoints.copy()
        loaded_boxes_vehicle_system = bb_image_to_vehicle_system(
            loaded_boxes_image_system,
            config.pixels_per_meter,
            config.min_x_meter,
            config.min_y_meter,
        )

        # Remove zero-padded data
        non_zero_mask = (
            loaded_boxes_vehicle_system[:, TransfuserBoundingBoxIndex.X] != 0.0
        ) | (loaded_boxes_vehicle_system[:, TransfuserBoundingBoxIndex.Y] != 0.0)
        loaded_boxes_vehicle_system = loaded_boxes_vehicle_system[non_zero_mask]
        loaded_waypoints = loaded_waypoints[non_zero_mask]
        loaded_num_waypoints = loaded_num_waypoints[non_zero_mask]

        # Filter data with minimally one radar point
        radar_mask = (
            loaded_boxes_vehicle_system[:, TransfuserBoundingBoxIndex.NUM_RADAR_POINTS]
            > 0
        )
        loaded_boxes_vehicle_system = loaded_boxes_vehicle_system[radar_mask]
        loaded_waypoints = loaded_waypoints[radar_mask]
        loaded_num_waypoints = loaded_num_waypoints[radar_mask]

        selected_boxes = []

        if loaded_boxes_vehicle_system.shape[0] > 0:
            # Compute class priority index for each box
            class_priorities = {cls: i for i, cls in enumerate(priority_classes)}
            class_priority = np.array(
                [
                    class_priorities.get(int(c), len(priority_classes))
                    for c in loaded_boxes_vehicle_system[
                        :,
                        TransfuserBoundingBoxIndex.CLASS,
                    ]
                ],
            )

            # Stack into sortable array: (-velocity, class_priority, -num_radar_points)
            sortable = np.stack(
                [
                    -loaded_boxes_vehicle_system[
                        :,
                        TransfuserBoundingBoxIndex.VELOCITY,
                    ],
                    class_priority,
                    -loaded_boxes_vehicle_system[
                        :,
                        TransfuserBoundingBoxIndex.NUM_RADAR_POINTS,
                    ],
                ],
                axis=1,
            )

            # Sort lexicographically: we prioritize higher velocity, then class priority, then more radar points
            sorted_indices = np.lexsort(sortable.T[::-1])

            # Apply sorting to all three arrays
            sorted_boxes = loaded_boxes_vehicle_system[sorted_indices]

            # Take up to num_radar_queries
            selected_boxes = sorted_boxes[: config.num_radar_queries]

        if len(selected_boxes) > 0:
            # Extract [x, y, velocity]
            n_boxes = selected_boxes.shape[0]
            radar_detections[:n_boxes, RadarLabels.X] = selected_boxes[
                :,
                TransfuserBoundingBoxIndex.X,
            ]
            radar_detections[:n_boxes, RadarLabels.Y] = selected_boxes[
                :,
                TransfuserBoundingBoxIndex.Y,
            ]
            radar_detections[:n_boxes, RadarLabels.V] = selected_boxes[
                :,
                TransfuserBoundingBoxIndex.VELOCITY,
            ]
            radar_detections[:n_boxes, RadarLabels.VALID] = 1.0  # Valid box indicator

    return radar_detections


@beartype
def smooth_path(
    config: TrainingConfig,
    route: jt.Float[npt.NDArray, "N 2"],
    target_first_distance: float,
) -> jt.Float[npt.NDArray, "N 2"]:
    """Smooth a route by removing duplicates and creating evenly-spaced interpolated waypoints.

    This function preprocesses a route by removing duplicate waypoints while preserving the
    original path order, then generates a smoothed representation with evenly-spaced points
    using iterative line interpolation.

    The smoothing process:
    1. Identifies and removes duplicate consecutive waypoints
    2. Preserves the original route order (important for path following)
    3. Generates interpolated points at regular intervals along the cleaned route

    Args:
        config: Training configuration containing route smoothing parameters such as
            the number of interpolated points to generate.
        route: Array of shape (N, 2) containing input waypoints as (x, y) coordinates.
            May contain duplicate points.
        target_first_distance: Distance in meters for placing the first interpolated point
            from the origin (0, 0).

    Returns:
        Array of shape (num_route_points_smoothing, 2) containing the smoothed route
        with evenly-spaced waypoints. All duplicates are removed and spacing is regularized.
    """
    _, indices = np.unique(route, return_index=True, axis=0)
    # We need to remove the sorting of unique, because this algorithm assumes the order of the path is kept
    route = np.array(route)
    indices = np.sort(indices)
    indices = np.array(indices).astype(int)
    route = route[indices]
    interpolated_route_points = iterative_line_interpolation(
        config,
        route,
        target_first_distance,
    )

    return interpolated_route_points


@jt.jaxtyped(typechecker=beartype)
def circle_line_segment_intersection(
    circle_center: jt.Float[npt.NDArray, " 2"],
    circle_radius: float,
    pt1: jt.Float[npt.NDArray, " 2"],
    pt2: jt.Float[npt.NDArray, " 2"],
    full_line: bool = True,
    tangent_tol: float = 1e-9,
) -> list[tuple[float, float]]:
    """Find the intersection points between a circle and a line segment.

    Computes geometric intersections using the analytical solution for circle-line
    intersection. The function can return 0, 1, or 2 intersection points depending
    on whether the line misses, is tangent to, or crosses through the circle.

    Args:
        circle_center: The (x, y) coordinates of the circle center.
        circle_radius: The radius of the circle.
        pt1: The (x, y) coordinates of the first point of the line segment.
        pt2: The (x, y) coordinates of the second point of the line segment.
        full_line: If True, find intersections along the infinite line extending through
            pt1 and pt2. If False, only return intersections within the segment [pt1, pt2].
        tangent_tol: Numerical tolerance for determining if the line is tangent to the circle.
            When discriminant is below this threshold, treat as tangent case.

    Returns:
        A list of (x, y) tuples representing intersection points. The list contains:
        - 0 elements if no intersection exists
        - 1 element if the line is tangent to the circle
        - 2 elements if the line crosses through the circle
        Points are ordered along the direction from pt1 to pt2.

    Note:
        Implementation follows the analytical solution from:
        http://mathworld.wolfram.com/Circle-LineIntersection.html
        Credit: https://stackoverflow.com/a/59582674/9173068
    """
    if np.linalg.norm(pt1 - pt2) < 0.000000001:
        LOG.warning("Problem")

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx**2 + dy**2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius**2 * dr**2 - big_d**2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        # This makes sure the order along the segment is correct
        intersections = [
            (
                cx
                + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**0.5)
                / dr**2,
                cy + (-big_d * dx + sign * abs(dy) * discriminant**0.5) / dr**2,
            )
            for sign in ((1, -1) if dy < 0 else (-1, 1))
        ]
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [
                (xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy
                for xi, yi in intersections
            ]
            intersections = [
                pt
                for pt, frac in zip(intersections, fraction_along_segment)
                if 0 <= frac <= 1
            ]
        # If line is tangent to circle, return just one point (as both intersections have same location)
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections


@beartype
def iterative_line_interpolation(
    config: TrainingConfig,
    route: jt.Float[npt.NDArray, "n 2"],
    target_first_distance: float,
) -> jt.Float[npt.NDArray, "n 2"]:
    """Generate evenly-spaced interpolated points along a route using circle-line intersection.

    This function creates a smoothed route representation by iteratively finding points at
    fixed distances along the input route. It uses a geometric approach where each new point
    is found at the intersection of a circle (centered at the last interpolated point) with
    the line segments of the original route.

    The algorithm:
    1. Places the first point at `target_first_distance` from the origin (0, 0)
    2. Places subsequent points exactly 1.0 meter apart along the route
    3. When multiple intersections exist, selects the one in the forward direction
    4. Extrapolates beyond the route end if needed to reach the target number of points

    Args:
        config: Training configuration containing route planning parameters, specifically:
            - num_route_points_smoothing: Number of interpolated points to generate
            - dense_route_planner_min_distance: Initial minimum distance (unused, overwritten)
        route: Array of shape (N, 2) containing the input waypoints as (x, y) coordinates.
        target_first_distance: Distance in meters for the first interpolated point from origin.

    Returns:
        Array of shape (num_route_points_smoothing, 2) containing evenly-spaced interpolated
        waypoints along the route.

    Raises:
        Exception: If no intersection is found during interpolation (should not occur under
            normal circumstances).
    """
    interpolated_route_points = []

    # this value is actually not used anymore, it is overwritten in the loop
    min_distance = config.dense_route_planner_min_distance
    last_interpolated_point = np.array([0.0, 0.0])
    current_route_index = 0
    current_point = route[current_route_index]
    last_point = np.array([0.0, 0.0])
    first_iteration = True

    while len(interpolated_route_points) < config.num_route_points_smoothing:
        # First point should be target_first_distance away from the vehicle.
        if not first_iteration:
            current_route_index += 1
            last_point = current_point

        if current_route_index < route.shape[0]:
            current_point = route[current_route_index]
            intersection = circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=(
                    min_distance if not first_iteration else target_first_distance
                ),
                pt1=last_interpolated_point,
                pt2=current_point,
                full_line=True,
            )

        else:  # We hit the end of the input route. We extrapolate the last 2 points
            current_point = route[-1]
            last_point = route[-2]
            intersection = circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=min_distance,
                pt1=last_point,
                pt2=current_point,
                full_line=True,
            )

        # 3 cases: 0 intersection, 1 intersection, 2 intersection
        if len(intersection) > 1:  # 2 intersections
            # Take the one that is closer to current point
            point_1 = np.array(intersection[0])
            point_2 = np.array(intersection[1])
            direction = current_point - last_point
            dot_p1_to_last = np.dot(point_1, direction)
            dot_p2_to_last = np.dot(point_2, direction)

            if dot_p1_to_last > dot_p2_to_last:
                intersection_point = point_1
            else:
                intersection_point = point_2
            add_point = True
        elif len(intersection) == 1:  # 1 Intersections
            intersection_point = np.array(intersection[0])
            add_point = True
        else:  # 0 Intersection
            add_point = False
            raise Exception("No intersection found. This should never occur.")

        if add_point:
            last_interpolated_point = intersection_point
            interpolated_route_points.append(intersection_point)
            min_distance = 1.0  # After the first point we want each point to be 1 m away from the last.

        first_iteration = False

    interpolated_route_points = np.array(interpolated_route_points)
    return interpolated_route_points


@beartype
def command_to_one_hot(command: int) -> jt.Float[npt.NDArray, " 6"]:
    """Convert CARLA navigation command to one-hot encoded representation.

    Args:
        command: CARLA navigation command integer.

    Returns:
        One-hot encoded numpy array of shape (6,) with 1.0 at the command index and 0.0 elsewhere.bas
    """
    if command < 0:
        command = 4
    command -= 1
    if command not in [0, 1, 2, 3, 4, 5]:
        command = 3
    cmd_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cmd_one_hot[command] = 1.0

    return np.array(cmd_one_hot)
