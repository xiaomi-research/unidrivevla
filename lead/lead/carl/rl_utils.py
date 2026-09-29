# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Utility functions for preprocessing traffic lights and coordinate transforms.
"""

import math

import carla
import numpy as np


def normalize_angle_degree(x):
    x = x % 360.0
    if x > 180.0:
        x -= 360.0
    return x


def rotate_point(point, angle):
    x_ = (
        math.cos(math.radians(angle)) * point.x
        - math.sin(math.radians(angle)) * point.y
    )
    y_ = (
        math.sin(math.radians(angle)) * point.x
        + math.cos(math.radians(angle)) * point.y
    )
    return carla.Vector3D(x_, y_, point.z)


def get_traffic_light_waypoints(traffic_light, carla_map):
    base_transform = traffic_light.get_transform()
    base_loc = traffic_light.get_location()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)

    area = []
    for x in x_values:
        point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        area.append(point_location)

    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        if (
            not ini_wps
            or ini_wps[-1].road_id != wpx.road_id
            or ini_wps[-1].lane_id != wpx.lane_id
        ):
            ini_wps.append(wpx)

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


def inverse_conversion_2d(point, translation, yaw):
    rotation_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
    )
    converted_point = rotation_matrix.T @ (point - translation)
    return converted_point


def dot_product(vector1, vector2):
    return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z


def cross_product(vector1, vector2):
    return carla.Vector3D(
        x=vector1.y * vector2.z - vector1.z * vector2.y,
        y=vector1.z * vector2.x - vector1.x * vector2.z,
        z=vector1.x * vector2.y - vector1.y * vector2.x,
    )


def get_separating_plane(r_pos, plane, obb1, obb2):
    return abs(dot_product(r_pos, plane)) > (
        abs(dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane))
        + abs(dot_product((obb1.rotation.get_right_vector() * obb1.extent.y), plane))
        + abs(dot_product((obb1.rotation.get_up_vector() * obb1.extent.z), plane))
        + abs(dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane))
        + abs(dot_product((obb2.rotation.get_right_vector() * obb2.extent.y), plane))
        + abs(dot_product((obb2.rotation.get_up_vector() * obb2.extent.z), plane))
    )


def check_obb_intersection(obb1, obb2):
    r_pos = obb2.location - obb1.location
    return not (
        get_separating_plane(r_pos, obb1.rotation.get_forward_vector(), obb1, obb2)
        or get_separating_plane(r_pos, obb1.rotation.get_right_vector(), obb1, obb2)
        or get_separating_plane(r_pos, obb1.rotation.get_up_vector(), obb1, obb2)
        or get_separating_plane(r_pos, obb2.rotation.get_forward_vector(), obb1, obb2)
        or get_separating_plane(r_pos, obb2.rotation.get_right_vector(), obb1, obb2)
        or get_separating_plane(r_pos, obb2.rotation.get_up_vector(), obb1, obb2)
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_forward_vector(),
                obb2.rotation.get_forward_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_forward_vector(),
                obb2.rotation.get_right_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_forward_vector(),
                obb2.rotation.get_up_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_right_vector(),
                obb2.rotation.get_forward_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_right_vector(),
                obb2.rotation.get_right_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_right_vector(),
                obb2.rotation.get_up_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_up_vector(),
                obb2.rotation.get_forward_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(
                obb1.rotation.get_up_vector(),
                obb2.rotation.get_right_vector(),
            ),
            obb1,
            obb2,
        )
        or get_separating_plane(
            r_pos,
            cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_up_vector()),
            obb1,
            obb2,
        )
    )


def local_bounding_box_to_global(actor, actor_bb):
    actor_transform = actor.get_transform()
    global_location = actor_transform.transform(actor_bb.location)
    global_bounding_box = carla.BoundingBox(global_location, actor_bb.extent)
    global_bounding_box.rotation = carla.Rotation(
        pitch=actor_transform.rotation.pitch + actor_bb.rotation.pitch,
        yaw=actor_transform.rotation.yaw + actor_bb.rotation.yaw,
        roll=actor_transform.rotation.roll + actor_bb.rotation.roll,
    )
    return global_bounding_box
