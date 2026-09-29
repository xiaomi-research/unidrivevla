# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Strongly modified version of chauffeurnet.py
Utilities to render bird's eye view semantic segmentation maps.
Code adapted from https://github.com/zhejz/carla-roach
"""

from collections import deque
from enum import Enum
from pathlib import Path

import carla
import cv2 as cv
import h5py
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.carl import rl_utils as rl_u
from lead.carl.bev.obs_manager import ObsManagerBase
from lead.carl.bev.traffic_light import TrafficLightHandler

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREY = (128, 128, 128)
COLOR_ORANGE = (209, 134, 0)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)
COLOR_ALUMINIUM_7 = (70, 70, 70)


def tint(color, factor):
    r, g, b = color
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)


class BoxType(Enum):
    VEHICLE = 1
    PEDESTRIAN = 2
    SPEED_SIGN = 3
    EMERGENCY_LIGHT = 4
    LEFT_BLINKER = 5
    RIGHT_BLINKER = 6
    BRAKE_LIGHT = 7
    DOOR = 8


class ObsManager(ObsManagerBase):
    """
    Generates bev semantic segmentation maps.
    """

    def __init__(self, config):
        self.config = config
        self.width = int(self.config.bev_semantics_width)
        self.pixels_ev_to_bottom = self.config.pixels_ev_to_bottom
        self.pixels_per_meter = self.config.pixels_per_meter
        self.scale_bbox = self.config.scale_bbox
        self.scale_mask_col = self.config.scale_mask_col
        self.vehicle = None
        self.world = None
        self.world_map = None
        if self.config.use_history:
            maxlen_queue = max(
                max(config.history_idx_2) + 1,
                -min(config.history_idx_2),
            )
            self.history_queue = deque(maxlen=maxlen_queue)

        # Preallocate memory, so we do not have to recreate the array every time step.
        self.final_masks = np.zeros(
            (self.config.obs_num_channels, self.width, self.width),
            dtype=np.uint8,
        )

        self.map_dir = Path(__file__).resolve().parent.parent / self.config.map_folder

        super().__init__()

    def attach_ego_vehicle(self, vehicle, criteria_stop, world_map, route):
        self.vehicle = vehicle
        self.criteria_stop = criteria_stop
        self.world_map = world_map
        self.maximum_speed_signs = self.world_map.get_all_landmarks_of_type("274")

        current_world = self.vehicle.get_world()
        if self.world is None or current_world.id != self.world.id:
            self.world = current_world

            maps_h5_path = self.map_dir / (world_map.name.rsplit("/", 1)[1] + ".h5")
            hf = h5py.File(maps_h5_path, "r", libver="latest", swmr=True)
            self._world_offset = np.array(
                hf.attrs["world_offset_in_meters"],
                dtype=np.float32,
            )
            assert np.isclose(
                self.pixels_per_meter,
                float(hf.attrs["pixels_per_meter"]),
            )
            self._map_h, self._map_w = hf["road"].shape
            # OpenCV warpAffine requires src dimensions < SHRT_MAX (32767).
            # Large maps (e.g. Town11 at 43371x43371) exceed this limit and also
            # cause OOM if fully loaded. For these maps we keep the h5 file open
            # and read small crops around the ego vehicle each frame instead.
            self._need_crop = self._map_h >= 32767 or self._map_w >= 32767

            if self._need_crop:
                self._h5_file = hf
                self.hd_map_array = None
            else:
                parking = np.array(hf["parking"], dtype=np.uint16)
                road = np.array(hf["road"], dtype=np.uint16)
                if self.config.render_shoulder:
                    shoulder = np.array(hf["shoulder"], dtype=np.uint16)
                    full_road = np.clip((shoulder + parking + road), 0, 255).astype(
                        dtype=np.uint8,
                    )
                else:
                    full_road = np.clip((parking + road), 0, 255).astype(dtype=np.uint8)

                if self.config.use_shoulder_channel:
                    self.hd_map_array = np.stack(
                        (
                            full_road,
                            hf["lane_marking_all"],
                            hf["lane_marking_white_broken"],
                            hf["shoulder"],
                        ),
                        axis=2,
                    )
                else:
                    self.hd_map_array = np.stack(
                        (
                            full_road,
                            hf["lane_marking_all"],
                            hf["lane_marking_white_broken"],
                        ),
                        axis=2,
                    )

                self.hd_map_array = self.hd_map_array.astype(dtype=np.uint8)
                hf.close()
                self._h5_file = None

            self.distance_threshold = np.ceil(self.width / self.pixels_per_meter)

        try:
            self.route_map = CarlaDataProvider.get_map_route()
        except AttributeError:
            self.route_map = []
            for route_point in route:
                self.route_map.append(
                    self.world_map.get_waypoint(
                        route_point[0].location,
                        project_to_road=True,
                    ),
                )

        TrafficLightHandler.reset(
            self.world,
            self.world_map,
            self.route_map,
            self.config,
        )
        self.total_num_route_points = len(route)

    @staticmethod
    def get_stops(criteria_stop):
        stop_sign = criteria_stop.target_stop_sign
        stops = []
        if (stop_sign is not None) and (not criteria_stop.stop_completed):
            bb_loc = carla.Location(stop_sign.trigger_volume.location)
            bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            trans = stop_sign.get_transform()
            stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
        return stops

    def is_within_distance(self, ego_location, actor_location):
        c_distance = (
            ego_location.distance(actor_location) < self.distance_threshold
            and abs(ego_location.z - actor_location.z) < 8.0
        )
        return c_distance

    def get_observation(
        self,
        waypoint_route,
        vehicles_all,
        walkers_all,
        static_all,
        debug=False,
    ):
        ev_transform = self.vehicle.get_transform()
        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = self.vehicle.bounding_box

        nearby_speed_signs = []
        for speed_sign in self.maximum_speed_signs:
            speed_sign_location = speed_sign.transform.location

            if self.is_within_distance(ev_loc, speed_sign_location):
                bounding_box = carla.BoundingBox(
                    speed_sign_location,
                    carla.Vector3D(1.0, 1.0, 1.0),
                )
                bounding_box.rotation = speed_sign.transform.rotation
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bounding_box.extent)
                speed_limit = speed_sign.value / 3.6
                nearby_speed_signs.append(
                    (
                        carla.Transform(bounding_box.location, bounding_box.rotation),
                        bb_loc,
                        bb_ext,
                        speed_limit,
                        BoxType.SPEED_SIGN,
                    ),
                )

        vehicles = []
        for vehicle in vehicles_all:
            if vehicle.id == self.vehicle.id:
                continue

            traffic_transform = vehicle.get_transform()
            bb_location = traffic_transform.location + vehicle.bounding_box.location
            if self.is_within_distance(ev_loc, bb_location):
                bounding_box = carla.BoundingBox(
                    bb_location,
                    vehicle.bounding_box.extent,
                )
                bounding_box.rotation = carla.Rotation(
                    pitch=vehicle.bounding_box.rotation.pitch
                    + traffic_transform.rotation.pitch,
                    yaw=vehicle.bounding_box.rotation.yaw
                    + traffic_transform.rotation.yaw,
                    roll=vehicle.bounding_box.rotation.roll
                    + traffic_transform.rotation.roll,
                )
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bounding_box.extent)
                if self.config.scale_bbox:
                    bb_ext = bb_ext * self.config.scale_factor_vehicle
                    bb_ext.x = max(bb_ext.x, self.config.min_ext_bounding_box)
                    bb_ext.y = max(bb_ext.y, self.config.min_ext_bounding_box)

                speed = vehicle.get_velocity().length()
                vehicles.append(
                    (
                        carla.Transform(bounding_box.location, bounding_box.rotation),
                        bb_loc,
                        bb_ext,
                        speed,
                        BoxType.VEHICLE,
                    ),
                )

                if (
                    hasattr(CarlaDataProvider, "_vehicles_with_open_doors")
                    and vehicle.id in CarlaDataProvider._vehicles_with_open_doors
                ):
                    door_type = CarlaDataProvider._vehicles_with_open_doors[vehicle.id]
                    if door_type == carla.VehicleDoor.FL:
                        door_loc = carla.Location(x=0.7, y=-1.8, z=0.0)
                        door_etx = carla.Vector3D(0.2, 1.0, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                door_loc,
                                door_etx,
                                speed,
                                BoxType.DOOR,
                            ),
                        )

                    if door_type == carla.VehicleDoor.FR:
                        door_loc = carla.Location(x=0.7, y=1.8, z=0.0)
                        door_etx = carla.Vector3D(0.2, 1.0, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                door_loc,
                                door_etx,
                                speed,
                                BoxType.DOOR,
                            ),
                        )

                    if door_type == carla.VehicleDoor.RL:
                        door_loc = carla.Location(x=-0.7, y=-1.8, z=0.0)
                        door_etx = carla.Vector3D(0.2, 1.0, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                door_loc,
                                door_etx,
                                speed,
                                BoxType.DOOR,
                            ),
                        )

                    if door_type == carla.VehicleDoor.RR:
                        door_loc = carla.Location(x=-0.7, y=1.8, z=0.0)
                        door_etx = carla.Vector3D(0.2, 1.0, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                door_loc,
                                door_etx,
                                speed,
                                BoxType.DOOR,
                            ),
                        )

                try:
                    light_state = vehicle.get_light_state()
                except RuntimeError:
                    light_state = None

                if light_state is not None:
                    if (
                        light_state
                        & (
                            carla.VehicleLightState.Special1
                            | carla.VehicleLightState.Special2
                        )
                        and vehicle.attributes["special_type"] == "emergency"
                    ):
                        light_loc = carla.Location(x=0.0, y=0.0, z=0.0)
                        light_etx = carla.Vector3D(0.5, 0.5, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                light_loc,
                                light_etx,
                                0.0,
                                BoxType.EMERGENCY_LIGHT,
                            ),
                        )

                    if light_state & carla.VehicleLightState.LeftBlinker:
                        light_loc = carla.Location(x=0.5, y=-0.5, z=0.0)
                        light_etx = carla.Vector3D(0.5, 0.5, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                light_loc,
                                light_etx,
                                0.0,
                                BoxType.LEFT_BLINKER,
                            ),
                        )

                    if light_state & carla.VehicleLightState.RightBlinker:
                        light_loc = carla.Location(x=0.5, y=0.5, z=0.0)
                        light_etx = carla.Vector3D(0.5, 0.5, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                light_loc,
                                light_etx,
                                0.0,
                                BoxType.RIGHT_BLINKER,
                            ),
                        )

                    if light_state & carla.VehicleLightState.Brake:
                        light_loc = carla.Location(x=-1.5, y=0.0, z=0.0)
                        light_etx = carla.Vector3D(0.5, 0.5, 0.5)
                        vehicles.append(
                            (
                                carla.Transform(
                                    bounding_box.location,
                                    bounding_box.rotation,
                                ),
                                light_loc,
                                light_etx,
                                0.0,
                                BoxType.BRAKE_LIGHT,
                            ),
                        )

        walkers = []
        for walker in walkers_all:
            walker_transform = walker.get_transform()
            walker_location = walker_transform.location

            if self.is_within_distance(ev_loc, walker_location):
                bounding_box = carla.BoundingBox(
                    walker_transform.transform(walker.bounding_box.location),
                    walker.bounding_box.extent,
                )
                bounding_box.rotation = carla.Rotation(
                    pitch=walker.bounding_box.rotation.pitch
                    + walker_transform.rotation.pitch,
                    yaw=walker.bounding_box.rotation.yaw
                    + walker_transform.rotation.yaw,
                    roll=walker.bounding_box.rotation.roll
                    + walker_transform.rotation.roll,
                )

                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bounding_box.extent)
                if self.config.scale_bbox:
                    bb_ext = bb_ext * self.config.scale_factor_vehicle
                    bb_ext.x = max(bb_ext.x, self.config.min_ext_bounding_box)
                    bb_ext.y = max(bb_ext.y, self.config.min_ext_bounding_box)

                speed = walker.get_velocity().length()
                walkers.append(
                    (
                        carla.Transform(bounding_box.location, bounding_box.rotation),
                        bb_loc,
                        bb_ext,
                        speed,
                        BoxType.PEDESTRIAN,
                    ),
                )

        statics = []
        for static in static_all:
            static_transform = static.get_transform()
            static_location = static_transform.location

            if (
                self.is_within_distance(ev_loc, static_location)
                and "dirtdebris" not in static.type_id
            ):
                bounding_box = carla.BoundingBox(
                    static_transform.transform(static.bounding_box.location),
                    static.bounding_box.extent,
                )
                bounding_box.rotation = carla.Rotation(
                    pitch=static.bounding_box.rotation.pitch
                    + static_transform.rotation.pitch,
                    yaw=static.bounding_box.rotation.yaw
                    + static_transform.rotation.yaw,
                    roll=static.bounding_box.rotation.roll
                    + static_transform.rotation.roll,
                )

                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bounding_box.extent)
                if self.config.scale_bbox:
                    bb_ext = bb_ext * self.config.scale_factor_vehicle
                    bb_ext.x = max(bb_ext.x, self.config.min_ext_bounding_box)
                    bb_ext.y = max(bb_ext.y, self.config.min_ext_bounding_box)

                statics.append(
                    (
                        carla.Transform(bounding_box.location, bounding_box.rotation),
                        bb_loc,
                        bb_ext,
                    ),
                )

        if self.config.use_history:
            self.history_queue.append((vehicles, walkers))

        route_idx_len = self.config.num_route_points_rendered
        if len(waypoint_route) < self.config.num_route_points_rendered:
            route_idx_len = len(waypoint_route)

        current_route_idx = self.total_num_route_points - len(waypoint_route)
        tl_green, tl_yellow, tl_red, yellow_tl_actor = (
            TrafficLightHandler.get_stopline_vtx(
                ev_loc,
                self.distance_threshold,
                current_route_idx,
                -20,
                self.config.num_route_points_rendered,
            )
        )
        stops = self.get_stops(self.criteria_stop)

        m_warp = self.get_warp_transform(ev_loc, ev_rot)

        vehicle_mask = self.get_mask_from_moving_actor_list(vehicles, m_warp)
        walker_mask = self.get_mask_from_moving_actor_list(walkers, m_warp)
        if self.config.render_green_tl:
            tl_green_mask = self._get_mask_from_stopline_vtx(tl_green, m_warp, 0.315)
        tl_yellow_mask = self._get_mask_from_stopline_vtx(
            tl_yellow,
            m_warp,
            0.67,
            yellow_tl_actor,
        )
        tl_red_mask = self._get_mask_from_stopline_vtx(tl_red, m_warp, 1.0)
        stop_mask = self.get_mask_from_actor_list(stops, m_warp)
        static_mask = self.get_mask_from_actor_list(statics, m_warp)
        speed_sign_mask = self.get_mask_from_moving_actor_list(
            nearby_speed_signs,
            m_warp,
        )

        if self.config.use_history:
            past_vehicle_masks, past_walker_masks = self.get_history_masks(m_warp)

        if self._need_crop:
            # Crop a small region around the ego vehicle from the h5 map and adjust
            # the warp matrix so that source coordinates are relative to the crop.
            # The warp reads at most ~width*sqrt(1.25) px from ego; crop_half of
            # width*1.5 provides ample margin for interpolation and rounding.
            ev_px = self._world_to_pixel(ev_loc)
            crop_half = int(np.ceil(self.width * 1.5))
            col0 = max(int(ev_px[0]) - crop_half, 0)
            row0 = max(int(ev_px[1]) - crop_half, 0)
            col1 = min(int(ev_px[0]) + crop_half, self._map_w)
            row1 = min(int(ev_px[1]) + crop_half, self._map_h)
            hd_map_crop = self._read_h5_crop(row0, row1, col0, col1)
            # Shift the affine translation to map cropped-array coords to the same
            # destination pixels: M'*[x-col0, y-row0, 1]^T == M*[x, y, 1]^T
            m_warp_crop = m_warp.copy()
            m_warp_crop[0, 2] += m_warp[0, 0] * col0 + m_warp[0, 1] * row0
            m_warp_crop[1, 2] += m_warp[1, 0] * col0 + m_warp[1, 1] * row0
            warped_hd_map = cv.warpAffine(
                hd_map_crop,
                m_warp_crop,
                (self.width, self.width),
            )
        else:
            warped_hd_map = cv.warpAffine(
                self.hd_map_array,
                m_warp,
                (self.width, self.width),
            )

        route_mask = np.zeros((self.width, self.width), dtype=np.uint8)

        if self.config.condition_outside_junction:
            route_plan = np.array(
                [
                    [waypoint_route[i][0].location.x, waypoint_route[i][0].location.y]
                    for i in range(0, route_idx_len)
                ],
                dtype=np.float32,
            )
        else:
            start_idx = len(self.route_map) - len(waypoint_route)
            if start_idx < 0 or (start_idx + route_idx_len) > len(self.route_map):
                route_plan = np.array([], dtype=np.float32)
            else:
                route_plan = np.array(
                    [
                        [
                            self.route_map[i].transform.location.x,
                            self.route_map[i].transform.location.y,
                        ]
                        for i in range(start_idx, start_idx + route_idx_len)
                        if self.route_map[i].is_junction
                        or self.route_map[i].is_intersection
                    ],
                    dtype=np.float32,
                )

        if self.config.use_target_point:
            start_idx = len(self.route_map) - len(waypoint_route)
            target_point = np.array(
                [
                    self.route_map[-1].transform.location.x,
                    self.route_map[-1].transform.location.y,
                ],
                dtype=np.float32,
            )
            if start_idx < 0 or (start_idx + route_idx_len) > len(self.route_map):
                target_point = np.array([0.0, 0.0], dtype=np.float32)
            else:
                for i in range(start_idx, len(self.route_map)):
                    if (
                        self.route_map[i].is_junction
                        or self.route_map[i].is_intersection
                    ):
                        target_point = np.array(
                            [
                                self.route_map[i].transform.location.x,
                                self.route_map[i].transform.location.y,
                            ],
                            dtype=np.float32,
                        )
                        break
            ego_target_point = rl_u.inverse_conversion_2d(
                target_point,
                np.array((ev_loc.x, ev_loc.y)),
                np.deg2rad(ev_rot.yaw),
            )

        if route_plan.shape[0] > 0:
            route_in_pixel = self.world_to_pixel_batch(route_plan)[:, np.newaxis, :]

            route_warped = cv.transform(route_in_pixel, m_warp)

            for waypoint in np.round(route_warped).astype(np.int32):
                cv.circle(
                    route_mask,
                    center=waypoint[0],
                    radius=self.config.route_width // 2,
                    color=255,
                    thickness=cv.FILLED,
                    lineType=cv.FILLED,
                )

        ev_mask_col = self.get_mask_from_actor_list(
            [(ev_transform, ev_bbox.location, ev_bbox.extent * self.scale_mask_col)],
            m_warp,
        )

        if debug:
            road_mask = warped_hd_map[:, :, 0].astype(bool)
            lane_mask_all = warped_hd_map[:, :, 1].astype(bool)
            lane_broken_mask = warped_hd_map[:, :, 2].astype(bool)
            route_mask_bool = route_mask.astype(bool)
            vehicle_mask_bool = vehicle_mask.astype(bool)
            walker_mask_bool = walker_mask.astype(bool)
            speed_sign_mask_bool = speed_sign_mask.astype(bool)
            if self.config.use_shoulder_channel:
                shoulder_mask_bool = warped_hd_map[:, :, 3].astype(bool)

            ev_mask = self.get_mask_from_actor_list(
                [(ev_transform, ev_bbox.location, ev_bbox.extent)],
                m_warp,
            )

            image = np.zeros([self.width, self.width, 3], dtype=np.uint8)
            if self.config.use_shoulder_channel:
                image[shoulder_mask_bool] = COLOR_ALUMINIUM_7
            image[road_mask] = COLOR_ALUMINIUM_5
            image[route_mask_bool] = COLOR_ALUMINIUM_3
            image[lane_mask_all] = COLOR_MAGENTA
            image[lane_broken_mask] = COLOR_MAGENTA_2

            image[stop_mask] = COLOR_YELLOW_2
            if self.config.render_green_tl:
                image[tl_green_mask.astype(bool)] = COLOR_GREEN
            image[tl_yellow_mask.astype(bool)] = COLOR_YELLOW
            image[tl_red_mask.astype(bool)] = COLOR_RED

            image[static_mask] = COLOR_ALUMINIUM_0

            h_len = len(self.config.history_idx_2)
            if self.config.use_history:
                for i in range(len(past_vehicle_masks)):
                    past_vehicle_mask_bool = past_vehicle_masks[i].astype(bool)
                    vehicle_float = (past_vehicle_masks[i].astype(np.float32) / 255.0)[
                        :,
                        :,
                        np.newaxis,
                    ]
                    vehicle_color = (
                        vehicle_float[past_vehicle_mask_bool]
                        * np.array(
                            tint(COLOR_BLUE, (h_len - i + 1) * 0.2),
                            dtype=np.float32,
                        )[np.newaxis]
                    )
                    image[past_vehicle_mask_bool] = vehicle_color.astype(np.uint8)

                for i in range(len(past_walker_masks)):
                    past_walker_mask_bool = past_walker_masks[i].astype(bool)
                    walker_float = (past_walker_masks[i].astype(np.float32) / 255.0)[
                        :,
                        :,
                        np.newaxis,
                    ]

                    walker_color = (
                        walker_float[past_walker_mask_bool]
                        * np.array(
                            tint(COLOR_CYAN, (h_len - i + 1) * 0.2),
                            dtype=np.float32,
                        )[np.newaxis]
                    )
                    image[past_walker_mask_bool] = walker_color.astype(np.uint8)

            vehicle_float = (vehicle_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            vehicle_color = (
                vehicle_float[vehicle_mask_bool]
                * np.array(COLOR_BLUE, dtype=np.float32)[np.newaxis]
            )
            image[vehicle_mask_bool] = vehicle_color.astype(np.uint8)

            walker_float = (walker_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            walker_color = (
                walker_float[walker_mask_bool]
                * np.array(COLOR_CYAN, dtype=np.float32)[np.newaxis]
            )
            image[walker_mask_bool] = walker_color.astype(np.uint8)

            speed_sign_float = (speed_sign_mask.astype(np.float32) / 255.0)[
                :,
                :,
                np.newaxis,
            ]
            speed_sign_color = (
                speed_sign_float[speed_sign_mask_bool]
                * np.array(COLOR_GREY, dtype=np.float32)[np.newaxis]
            )
            image[speed_sign_mask_bool] = speed_sign_color.astype(np.uint8)

            image[ev_mask] = COLOR_WHITE

        c_road = warped_hd_map[:, :, 0]
        c_lane = warped_hd_map[:, :, 1].astype(bool) * np.array(255, dtype=np.uint8)
        lane_mask_broken = warped_hd_map[:, :, 2].astype(bool)
        c_lane[lane_mask_broken] = np.array(127, dtype=np.uint8)

        if self.config.use_shoulder_channel:
            c_shoulder = warped_hd_map[:, :, 3]

        c_tl = np.zeros([self.width, self.width], dtype=np.uint8)
        if self.config.render_green_tl:
            c_tl[tl_green_mask.astype(bool)] = tl_green_mask[tl_green_mask.astype(bool)]
        c_tl[tl_yellow_mask.astype(bool)] = tl_yellow_mask[tl_yellow_mask.astype(bool)]
        c_tl[tl_red_mask.astype(bool)] = tl_red_mask[tl_red_mask.astype(bool)]

        c_stop_sign = stop_mask * np.array(255, dtype=np.uint8)

        c_static = static_mask * np.array(255, dtype=np.uint8)

        self.final_masks[0] = c_road
        self.final_masks[1] = route_mask
        self.final_masks[2] = c_lane
        self.final_masks[3] = vehicle_mask
        self.final_masks[4] = walker_mask
        self.final_masks[5] = c_tl
        self.final_masks[6] = c_stop_sign
        if self.config.obs_num_channels > 7:
            self.final_masks[7] = speed_sign_mask
        if self.config.obs_num_channels > 8:
            self.final_masks[8] = c_static
        if self.config.obs_num_channels > 9:
            self.final_masks[9] = c_shoulder
        if self.config.obs_num_channels > 15 and self.config.use_history:
            self.final_masks[10] = past_vehicle_masks[0]
            self.final_masks[11] = past_vehicle_masks[1]
            self.final_masks[12] = past_vehicle_masks[2]
            self.final_masks[13] = past_walker_masks[0]
            self.final_masks[14] = past_walker_masks[1]
            self.final_masks[15] = past_walker_masks[2]

        obs_dict = {"bev_semantic_classes": self.final_masks}
        if debug:
            obs_dict["rendered"] = image

        obs_dict["collision_px"] = np.any(ev_mask_col & walker_mask.astype(bool))
        if self.config.use_shoulder_channel:
            non_drivable_area = np.logical_or(
                c_road.astype(bool),
                c_shoulder.astype(bool),
            )
        else:
            non_drivable_area = c_road.astype(bool)

        obs_dict["percentage_off_road"] = np.sum(
            ev_mask_col & np.logical_not(non_drivable_area),
        ) / np.sum(ev_mask_col)

        if self.config.use_target_point:
            obs_dict["target_point"] = ego_target_point

        return obs_dict

    def _get_mask_from_stopline_vtx(self, stopline_vtx, m_warp, factor, tl_actor=None):
        mask = np.zeros([self.width, self.width], dtype=np.uint8)
        for idx, sp_locs in enumerate(stopline_vtx):
            stopline_in_pixel = np.array(
                [[x.x, x.y] for x in sp_locs],
                dtype=np.float32,
            )
            stopline_in_pixel[:, 0:2] = self.pixels_per_meter * (
                stopline_in_pixel[:, 0:2] - self._world_offset[0:2]
            )
            stopline_in_pixel = stopline_in_pixel[:, np.newaxis, :]
            stopline_warped = cv.transform(stopline_in_pixel, m_warp)
            pt1 = (stopline_warped[0, 0] + 0.5).astype(np.int32)
            pt2 = (stopline_warped[1, 0] + 0.5).astype(np.int32)

            if tl_actor is not None and self.config.render_yellow_time:
                yellow_time = tl_actor[idx].get_yellow_time()
                elapsed_time = tl_actor[idx].get_elapsed_time()
                remaining_time = yellow_time - elapsed_time
                fraction = min(remaining_time / 3.0, 1.0)
                interp_length = 1.0 - factor
                yellow_factor = 1.0 - fraction * interp_length
                color = int(yellow_factor * 255)
            else:
                color = int(factor * 255)

            cv.line(
                mask,
                tuple(pt1),
                tuple(pt2),
                color=color,
                thickness=self.config.red_light_thickness,
            )
        return mask

    def get_mask_from_actor_list(self, actor_list, m_warp):
        mask = np.zeros([self.width, self.width], dtype=np.uint8)
        for actor_transform, bb_loc, bb_ext in actor_list:
            corners2 = [
                actor_transform.transform(
                    bb_loc + carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                ),
                actor_transform.transform(
                    bb_loc + carla.Location(x=bb_ext.x, y=-bb_ext.y),
                ),
                actor_transform.transform(
                    bb_loc + carla.Location(x=bb_ext.x, y=bb_ext.y),
                ),
                actor_transform.transform(
                    bb_loc + carla.Location(x=-bb_ext.x, y=bb_ext.y),
                ),
            ]
            corners3 = np.array(
                [[corner.x, corner.y] for corner in corners2],
                dtype=np.float32,
            )
            corners_in_pixel = self.world_to_pixel_batch(corners3)[:, np.newaxis, :]
            corners_warped = cv.transform(corners_in_pixel, m_warp)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(bool)

    def get_mask_from_moving_actor_list(self, actor_list, m_warp):
        mask = np.zeros([self.width, self.width], dtype=np.uint8)
        speed_range = self.config.max_speed_actor - self.config.min_speed_actor
        for actor_transform, bb_loc, bb_ext, speed, box_type in actor_list:
            clipped_speed = min(
                speed_range,
                max(0.0, speed - self.config.min_speed_actor),
            )
            fraction = clipped_speed / speed_range
            uint8_speed_value = 127 + int(fraction * 128)
            if box_type == BoxType.EMERGENCY_LIGHT:
                uint8_speed_value = 22
            if box_type == BoxType.LEFT_BLINKER:
                uint8_speed_value = 44
            if box_type == BoxType.RIGHT_BLINKER:
                uint8_speed_value = 77
            if box_type == BoxType.BRAKE_LIGHT:
                uint8_speed_value = 111

            corners2 = [
                actor_transform.transform(
                    bb_loc + carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                ),
                actor_transform.transform(
                    bb_loc + carla.Location(x=bb_ext.x, y=-bb_ext.y),
                ),
                actor_transform.transform(
                    bb_loc + carla.Location(x=bb_ext.x, y=bb_ext.y),
                ),
                actor_transform.transform(
                    bb_loc + carla.Location(x=-bb_ext.x, y=bb_ext.y),
                ),
            ]
            if (
                box_type in (BoxType.VEHICLE, BoxType.PEDESTRIAN)
                and self.config.render_speed_lines
            ):
                distance = max(1.0 / self.config.pixels_per_meter, speed)
                corners2.append(
                    actor_transform.transform(
                        bb_loc + carla.Location(x=bb_ext.x, y=0.0),
                    ),
                )
                corners2.append(
                    actor_transform.transform(
                        bb_loc + carla.Location(x=bb_ext.x + distance, y=0.0),
                    ),
                )

            corners3 = np.array(
                [[corner.x, corner.y] for corner in corners2],
                dtype=np.float32,
            )
            corners_in_pixel = self.world_to_pixel_batch(corners3)[:, np.newaxis, :]
            corners_warped = cv.transform(corners_in_pixel, m_warp)
            corners_warped = np.round(corners_warped).astype(np.int32)
            bounding_box = corners_warped[:4]

            cv.fillConvexPoly(mask, bounding_box, uint8_speed_value)
            if (
                box_type in (BoxType.VEHICLE, BoxType.PEDESTRIAN)
                and self.config.render_speed_lines
            ):
                speed_line = corners_warped[4:]
                cv.line(
                    mask,
                    (speed_line[0, 0, 0], speed_line[0, 0, 1]),
                    (speed_line[1, 0, 0], speed_line[1, 0, 1]),
                    uint8_speed_value,
                    thickness=1,
                )

        return mask

    def get_history_masks(self, m_warp):
        qsize = len(self.history_queue)
        vehicle_masks, walker_masks = [], []
        for idx in self.config.history_idx_2:
            idx = max(idx, -1 * qsize)

            vehicles, walkers = self.history_queue[idx]

            vehicle_masks.append(self.get_mask_from_moving_actor_list(vehicles, m_warp))
            walker_masks.append(self.get_mask_from_moving_actor_list(walkers, m_warp))

        return vehicle_masks, walker_masks

    def get_surrounding_actors(self, actor_list, scale=None):
        actors = []
        for actor in actor_list:
            bbox = actor[0]
            speed = actor[1]
            bb_loc = carla.Location()
            bb_ext = carla.Vector3D(bbox.extent)
            if self.config.scale_bbox:
                bb_ext = bb_ext * scale
                bb_ext.x = max(bb_ext.x, self.config.min_ext_bounding_box)
                bb_ext.y = max(bb_ext.y, self.config.min_ext_bounding_box)

            actors.append(
                (carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext, speed),
            )
        return actors

    def get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

        bottom_left = (
            ev_loc_in_px
            - self.pixels_ev_to_bottom * forward_vec
            - (0.5 * self.width) * right_vec
        )
        top_left = (
            ev_loc_in_px
            + (self.width - self.pixels_ev_to_bottom) * forward_vec
            - (0.5 * self.width) * right_vec
        )
        top_right = (
            ev_loc_in_px
            + (self.width - self.pixels_ev_to_bottom) * forward_vec
            + (0.5 * self.width) * right_vec
        )

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(
            np.float32,
        )
        dst_pts = np.array(
            [[0, self.width - 1], [0, 0], [self.width - 1, 0]],
            dtype=np.float32,
        )
        return cv.getAffineTransform(src_pts, dst_pts)

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self.pixels_per_meter * (location.x - self._world_offset[0])
        y = self.pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def world_to_pixel_batch(self, location):
        """Converts the world coordinates to pixel coordinates in batched form"""
        location[:, 0:2] = self.pixels_per_meter * (
            location[:, 0:2] - self._world_offset[0:2]
        )
        return location

    def _world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return self.pixels_per_meter * width

    def _read_h5_crop(self, row0, row1, col0, col1):
        """Read a crop from the h5 file and build the stacked hd_map channels."""
        hf = self._h5_file
        road = np.array(hf["road"][row0:row1, col0:col1], dtype=np.uint16)
        parking = np.array(hf["parking"][row0:row1, col0:col1], dtype=np.uint16)
        shoulder = None
        if self.config.render_shoulder or self.config.use_shoulder_channel:
            shoulder = np.array(hf["shoulder"][row0:row1, col0:col1], dtype=np.uint16)

        if shoulder is not None and self.config.render_shoulder:
            full_road = np.clip((shoulder + parking + road), 0, 255).astype(np.uint8)
        else:
            full_road = np.clip((parking + road), 0, 255).astype(np.uint8)

        lane_all = np.array(
            hf["lane_marking_all"][row0:row1, col0:col1],
            dtype=np.uint8,
        )
        lane_broken = np.array(
            hf["lane_marking_white_broken"][row0:row1, col0:col1],
            dtype=np.uint8,
        )

        if self.config.use_shoulder_channel:
            return np.stack(
                (full_road, lane_all, lane_broken, shoulder.astype(np.uint8)),
                axis=2,
            )
        return np.stack((full_road, lane_all, lane_broken), axis=2)

    def clean(self):
        self.vehicle = None
        self.world = None
        if hasattr(self, "_h5_file") and self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        if self.config.use_history:
            self.history_queue.clear()
