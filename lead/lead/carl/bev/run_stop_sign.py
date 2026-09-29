# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Tests if a stop sign is relevant for a particular vehicle.
Code adapted from https://github.com/zhejz/carla-roach
"""

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.carl.rl_utils import check_obb_intersection


class RunStopSign:
    PROXIMITY_THRESHOLD = 50.0
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 0.5

    def __init__(self, carla_world, carla_map):
        self.world = carla_world
        self.carla_map = carla_map
        self.list_stop_signs = []
        self.target_stop_sign = None
        self.stop_completed = False

        for actor in CarlaDataProvider.get_all_actors():
            if "traffic.stop" in actor.type_id:
                self.list_stop_signs.append(actor)

    def point_inside_boundingbox(self, point, bb_center, bb_extent, multiplier=1.2):
        A = carla.Vector2D(
            bb_center.x - multiplier * bb_extent.x,
            bb_center.y - multiplier * bb_extent.y,
        )
        B = carla.Vector2D(
            bb_center.x + multiplier * bb_extent.x,
            bb_center.y - multiplier * bb_extent.y,
        )
        D = carla.Vector2D(
            bb_center.x - multiplier * bb_extent.x,
            bb_center.y + multiplier * bb_extent.y,
        )
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def is_actor_affected_by_stop(self, wp_list, stop):
        stop_location = stop.get_transform().transform(stop.trigger_volume.location)
        actor_location = wp_list[0].transform.location
        if stop_location.distance(actor_location) > self.PROXIMITY_THRESHOLD:
            return False

        stop_extent = stop.trigger_volume.extent
        for actor_wp in wp_list:
            if self.point_inside_boundingbox(
                actor_wp.transform.location,
                stop_location,
                stop_extent,
            ):
                return True

        return False

    def scan_for_stop_sign(self, actor_transform, wp_list, actor_velocity):
        actor_direction = actor_transform.get_forward_vector()

        if actor_velocity.dot(actor_direction) < -0.17:
            return None

        for stop in self.list_stop_signs:
            if self.is_actor_affected_by_stop(wp_list, stop):
                return stop

        return None

    def get_waypoints(self, actor):
        wp_list = []
        steps = int(self.PROXIMITY_THRESHOLD / self.WAYPOINT_STEP)

        wp = self.carla_map.get_waypoint(actor.get_location())
        wp_list.append(wp)

        next_wp = wp
        for _ in range(steps):
            next_wps = next_wp.next(self.WAYPOINT_STEP)
            if not next_wps:
                break
            next_wp = next_wps[0]
            wp_list.append(next_wp)

        return wp_list

    def tick(self, vehicle):
        actor_transform = vehicle.get_transform()
        check_wps = self.get_waypoints(vehicle)
        actor_velocity = vehicle.get_velocity()

        if not self.target_stop_sign:
            self.target_stop_sign = self.scan_for_stop_sign(
                actor_transform,
                check_wps,
                actor_velocity,
            )
            return

        if not self.stop_completed:
            current_speed = CarlaDataProvider.get_velocity(vehicle)
            stop_location = self.target_stop_sign.get_transform().transform(
                self.target_stop_sign.trigger_volume.location,
            )
            stop_extent = self.target_stop_sign.trigger_volume.extent

            actor_bb_location = actor_transform.transform(
                carla.Location(
                    x=vehicle.bounding_box.location.x,
                    y=vehicle.bounding_box.location.y,
                    z=vehicle.bounding_box.location.z,
                ),
            )
            actor_bb = carla.BoundingBox(actor_bb_location, vehicle.bounding_box.extent)
            actor_bb.rotation = actor_transform.rotation

            stop_bb = carla.BoundingBox(stop_location, stop_extent)
            stop_bb.extent.z += 5.0
            stop_bb.rotation = self.target_stop_sign.get_transform().rotation

            vehicle_close_to_stop = check_obb_intersection(actor_bb, stop_bb)

            if current_speed < self.SPEED_THRESHOLD and vehicle_close_to_stop:
                self.stop_completed = True

        if not self.is_actor_affected_by_stop(check_wps, self.target_stop_sign):
            self.target_stop_sign = None
            self.stop_completed = False
