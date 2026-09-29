"""Route planning utilities for autonomous driving navigation.

This module provides route planning functionality including waypoint management,
trajectory interpolation, and coordinate transformations for CARLA simulation.
"""

import math
import xml.etree.ElementTree as ET
from collections import deque
from copy import deepcopy
from typing import Any

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from agents.navigation.global_route_planner import GlobalRoutePlanner

from lead.common import common_utils


class RoutePlanner:
    """Manages route planning and waypoint navigation for autonomous driving.

    This class handles route management including waypoint tracking, distance
    calculations, and route modifications based on vehicle position.
    """

    def __init__(self, min_distance: float, max_distance: float) -> None:
        """Initialize the route planner with distance constraints.

        Args:
            min_distance: Minimum distance threshold for route planning.
            max_distance: Maximum distance threshold for route planning.
        """
        self.saved_route: deque = deque()
        self.route: deque = deque()
        self.saved_route_distances: deque = deque()
        self.route_distances: deque = deque()

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False
        self.previous_target_points: list[jt.Float[npt.NDArray, 3]] = []
        self.previous_commands: list[Any] = []

    def set_route(
        self,
        global_plan: list[tuple[Any, Any]],
        gps: bool = False,
        carla_map: carla.Map | None = None,
        lat_ref: float | None = None,
        lon_ref: float | None = None,
    ) -> None:
        """Set the global route plan for navigation.

        Args:
            global_plan: List of tuples containing positions and commands.
            gps: Whether the positions are in GPS coordinates.
            carla_map: CARLA map object for waypoint extension.
            lat_ref: Latitude reference for GPS conversion.
            lon_ref: Longitude reference for GPS conversion.
        """
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos["lat"], pos["lon"], pos["z"]])
                pos = common_utils.convert_gps_to_carla(pos, lat_ref, lon_ref)
            else:
                # important to use the z variable, otherwise there are some rare bugs at carla.map.get_waypoint(carla.Location)
                pos = np.array([pos.location.x, pos.location.y, pos.location.z])

            self.route.append((pos, cmd))

        if carla_map is not None:
            for _ in range(50):
                loc = carla.Location(
                    x=self.route[-1][0][0],
                    y=self.route[-1][0][1],
                    z=self.route[-1][0][2],
                )
                next_loc = carla_map.get_waypoint(loc).next(1)[0].transform.location
                next_loc = np.array([next_loc.x, next_loc.y, next_loc.z])
                self.route.append((next_loc, self.route[-1][1]))

        # We do the calculations in the beginning once so that we don't have
        # to do them every time in run_step
        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff = self.route[i][0] - self.route[i - 1][0]
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            self.route_distances.append(distance)

    def run_step(self, gps: jt.Float[npt.NDArray, " 3"]) -> deque:
        """Execute one step of route planning based on current GPS position.

        Args:
            gps: Current GPS position as a 3D array (x, y, z).

        Returns:
            Updated route deque with waypoints and commands.
        """
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += self.route_distances[i]

            diff = self.route[i][0] - gps
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5

            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                to_pop = i
        self.pop(to_pop)
        return self.route

    def pop(self, to_pop: int, save_popped: bool = True) -> None:
        """Remove waypoints from the front of the route.

        Args:
            to_pop: Number of waypoints to remove from the route.
            save_popped: Whether to save the popped waypoints to history.
        """
        for _ in range(to_pop):
            if len(self.route) > 2:
                if save_popped:
                    self.previous_target_points.append(self.route[0][0])
                    self.previous_commands.append(self.route[0][1])
                self.route.popleft()
                self.route_distances.popleft()

    def save(self) -> None:
        """Save the current route state for later restoration.

        Creates a deep copy of the current route and distances to allow
        restoration via the load method.
        """
        # because self.route saves objects of traffic lights and traffic signs a deep copy is not possible
        self.saved_route = []
        for (
            loc,
            cmd,
            d_traffic,
            traffic,
            d_stop,
            stop,
            speed_limit,
            corrected_speed_limit,
        ) in self.route:
            self.saved_route.append(
                (
                    np.copy(loc),
                    cmd,
                    d_traffic,
                    traffic,
                    d_stop,
                    stop,
                    speed_limit,
                    corrected_speed_limit,
                ),
            )

        self.saved_route = deque(self.saved_route)
        self.saved_route_distances = deepcopy(self.route_distances)

    def load(self) -> None:
        """Restore the previously saved route state.

        Restores the route and distances from the last save operation
        and resets the route completion flag.
        """
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False


def interpolate_trajectory(
    world_map: carla.Map,
    waypoints_trajectory: list[carla.Location],
    hop_resolution: float = 1.0,
    max_len: int = 400,
) -> tuple[list[tuple[dict[str, float], Any]], list[tuple[carla.Transform, Any]]]:
    """Interpolate a dense trajectory from sparse keypoints.

    Given raw keypoints, interpolate a full dense trajectory to be used
    by the vehicle navigation system. Returns both GPS coordinates and
    original form trajectories.

    Args:
        world_map: Reference to the CARLA world map for route planning.
        waypoints_trajectory: Current coarse trajectory as waypoint list.
        hop_resolution: Resolution density for the interpolated trajectory.
        max_len: Maximum length for interpolated trace segments.

    Returns:
        A tuple containing:
            - GPS route as list of (GPS coordinates, connection) tuples
            - Original route as list of (transform, connection) tuples
    """

    grp = GlobalRoutePlanner(world_map, hop_resolution)
    # Obtain route plan
    lat_ref, lon_ref = _get_latlon_ref(world_map)

    route = []
    gps_route = []

    for i in range(len(waypoints_trajectory) - 1):
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        if waypoint.x != waypoint_next.x or waypoint.y != waypoint_next.y:
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            if len(interpolated_trace) > max_len:
                waypoints_trajectory[i + 1] = waypoints_trajectory[i]
            else:
                for wp, connection in interpolated_trace:
                    route.append((wp.transform, connection))
                    gps_coord = _location_to_gps(
                        lat_ref,
                        lon_ref,
                        wp.transform.location,
                    )
                    gps_route.append((gps_coord, connection))

    return gps_route, route


def _get_latlon_ref(world_map: carla.Map) -> tuple[float, float]:
    """Extract latitude and longitude reference from CARLA map.

    Converts from waypoints world coordinates to CARLA GPS coordinates
    by parsing the OpenDRIVE map data.

    Args:
        world_map: CARLA map object to extract reference from.

    Returns:
        Tuple containing latitude and longitude reference coordinates.
    """
    xodr = world_map.to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(" ")
                    for item in str_list:
                        if "+lat_0" in item:
                            lat_ref = float(item.split("=")[1])
                        if "+lon_0" in item:
                            lon_ref = float(item.split("=")[1])
    return lat_ref, lon_ref


def _location_to_gps(
    lat_ref: float,
    lon_ref: float,
    location: carla.Location,
) -> dict[str, float]:
    """Convert from world coordinates to GPS coordinates.

    Transforms CARLA world coordinates to GPS latitude/longitude
    coordinates using the provided reference points.

    Args:
        lat_ref: Latitude reference for the current map.
        lon_ref: Longitude reference for the current map.
        location: CARLA location to convert to GPS.

    Returns:
        Dictionary containing 'lat', 'lon', and 'z' GPS coordinates.
    """

    EARTH_RADIUS_EQUA = 6378137.0  # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    )
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {"lat": lat, "lon": lon, "z": z}
