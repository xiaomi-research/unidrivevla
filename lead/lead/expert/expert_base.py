"""Base class that provide basic properties and methods for expert agents."""

import logging
import os
import pathlib
from functools import cached_property

import carla
import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from agents.navigation.local_planner import RoadOption
from beartype import beartype
from leaderboard.autoagents import autonomous_agent, autonomous_agent_local
from privileged_route_planner import PrivilegedRoutePlanner
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import lead.common.common_utils as common_utils
from lead.common.base_agent import BaseAgent
from lead.common.constants import TransfuserSemanticSegmentationClass
from lead.expert import expert_utils
from lead.expert.expert_utils import cached_property_by, step_cached_property

LOG = logging.getLogger(__name__)


class ExpertBase(BaseAgent, autonomous_agent_local.AutonomousAgent):
    @beartype
    def expert_setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ):
        LOG.info("Setup")
        self.initialized = False

        self.recording = False
        self.config_path = path_to_conf_file
        self.step = -1
        self.save_path = None
        self.route_index = route_index
        self.scenario_name = pathlib.Path(path_to_conf_file).parent.name
        self.list_traffic_lights: list[
            tuple[carla.TrafficLight, carla.Location, list[carla.Waypoint]]
        ] = []
        self.close_traffic_lights: list[
            tuple[
                carla.TrafficLight,
                carla.BoundingBox,
                carla.TrafficLightState,
                int,
                bool,
            ]
        ] = []
        self.close_stop_signs = []

        self.track = autonomous_agent.Track.MAP
        # Privileged access
        self.ego_vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self.carla_world: carla.World = self.ego_vehicle.get_world()
        self.carla_world_map: carla.Map = CarlaDataProvider.get_map()

    @beartype
    def expert_init(self, hd_map: carla.Map | None):
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map: The map object of the CARLA world.
        """
        LOG.info("Init")

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(
            self.ego_vehicle.get_location(),
        )

        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2
        LOG.info(
            f"Vehicle starts {'with' if starts_with_parking_exit else 'without'} parking exit.",
        )

        # Set up the route planner and extrapolation
        self.privileged_route_planner = PrivilegedRoutePlanner(self.config_expert)
        self.privileged_route_planner.setup_route(
            self.org_dense_route_world_coord,
            self.carla_world,
            self.carla_world_map,
            starts_with_parking_exit,
            self.ego_vehicle.get_location(),
        )
        self.privileged_route_planner.save()
        LOG.info(
            f"Route setup with {len(self.privileged_route_planner.route_waypoints)} waypoints.",
        )

        # Preprocess traffic lights
        all_actors = self.carla_world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = expert_utils.get_traffic_light_waypoints(
                    actor,
                    self.carla_world_map,
                )
                self.list_traffic_lights.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()

    @step_cached_property
    def actors(self) -> carla.ActorList:
        return self.carla_world.get_actors()

    @step_cached_property
    def leading_vehicle_ids(self):
        return self.privileged_route_planner.compute_leading_vehicles(
            self.vehicles_inside_bev,
            self.ego_vehicle.id,
        )

    @step_cached_property
    def trailing_vehicle_ids(self):
        return self.privileged_route_planner.compute_trailing_vehicles(
            self.vehicles_inside_bev,
            self.ego_vehicle.id,
        )

    @step_cached_property
    def distance_to_next_traffic_light(self):
        return self.privileged_route_planner.distances_to_next_traffic_lights[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def next_traffic_light(self):
        return self.privileged_route_planner.next_traffic_lights[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def distance_to_next_stop_sign(self):
        return self.privileged_route_planner.distances_to_next_stop_signs[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def next_stop_sign(self):
        return self.privileged_route_planner.next_stop_signs[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def remaining_route(self):
        return self.route_waypoints_np[
            self.config_expert.tf_first_checkpoint_distance :
        ][:: self.config_expert.points_per_meter]

    @step_cached_property
    def remaining_route_original(self):
        return self.original_route_waypoints_np[
            self.config_expert.tf_first_checkpoint_distance :
        ][:: self.config_expert.points_per_meter]

    @step_cached_property
    @beartype
    def near_lane_change(self) -> bool:
        route_points = self.route_waypoints_np
        # Calculate the braking distance based on the ego velocity
        braking_distance = (
            ((self.ego_speed * 3.6) / 10.0) ** 2 / 2.0
        ) + self.config_expert.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        look_ahead_points = max(
            self.config_expert.minimum_lookahead_distance_to_compute_near_lane_change,
            min(
                route_points.shape[0],
                self.config_expert.points_per_meter * int(braking_distance),
            ),
        )
        current_route_index = self.privileged_route_planner.route_index
        max_route_length = len(self.privileged_route_planner.commands)

        from_index = max(
            0,
            current_route_index
            - self.config_expert.check_previous_distance_for_lane_change,
        )
        to_index = min(max_route_length - 1, current_route_index + look_ahead_points)
        # Iterate over the points around the current position, checking for lane change commands
        for i in range(from_index, to_index, 1):
            if self.privileged_route_planner.commands[i] in (
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ):
                return True

        return False

    @cached_property
    @beartype
    def min_lane_width_route(self) -> float:
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.carla_world = self.ego_vehicle.get_world()

        global_plan = self.org_dense_route_world_coord
        carla_map = self.carla_world.get_map()
        route_waypoints = [transform.location for transform, _ in global_plan]
        route_waypoints = [carla_map.get_waypoint(loc) for loc in route_waypoints]
        widths = []
        for waypoint in route_waypoints:
            if waypoint is not None and not waypoint.is_junction:
                widths.append(waypoint.lane_width)
        return max(min(widths), 2.75)

    @cached_property
    @beartype
    def max_speed_limit_route(self) -> float:
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.carla_world = self.ego_vehicle.get_world()
        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(
            self.ego_vehicle.get_location(),
        )
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2
        waypoint_planner = PrivilegedRoutePlanner(self.config_expert)
        waypoint_planner.setup_route(
            self.org_dense_route_world_coord,
            self.carla_world,
            self.carla_world_map,
            starts_with_parking_exit,
            self.ego_vehicle.get_location(),
        )
        way_point_planner = waypoint_planner
        waypoints = way_point_planner.route_waypoints
        if len(waypoints) == 0:
            return 30 / 3.6
        speed_limits = []
        for wp in waypoints:
            if wp is not None:
                # Get speed limit landmarks within a reasonable distance ahead
                landmarks = wp.get_landmarks(200.0, True)
                for landmark in landmarks:
                    # Check if the landmark is a MaximumSpeed sign
                    if landmark.type == carla.LandmarkType.MaximumSpeed:
                        # Extract speed limit value from landmark
                        try:
                            speed_limit = float(landmark.value)
                            if speed_limit > 0:
                                speed_limits.append(speed_limit)
                        except (ValueError, AttributeError):
                            pass

        # Return the maximum speed limit found, or default to 30 km/h
        if len(speed_limits) > 0:
            return max(speed_limits) / 3.6  # Convert km/h to m/s
        return 30 / 3.6

    @property
    def town(self):
        return self.carla_world.get_map().name.split("/")[-1]

    @cached_property
    def rep(self):
        return os.environ.get("REPETITION", "-1")

    @step_cached_property
    @beartype
    def privileged_ego_past_positions(self) -> list[list[float]]:
        ego_matrix_current = self.transform_queue[-1].get_matrix()
        T_world_to_current_ego = np.linalg.inv(ego_matrix_current)
        past_positions = []
        for transform in self.transform_queue:
            T_past = np.array(transform.get_matrix())
            pos_world = np.append(T_past[:3, 3], 1.0)
            pos_current_ego = T_world_to_current_ego @ pos_world
            past_positions.append(pos_current_ego[:2].tolist())
        return past_positions

    @step_cached_property
    @beartype
    def distance_to_intersection_index_ego(self) -> float:
        """
        Returns the index of the intersection point in the route waypoints.
        If no intersection point is found, returns None.
        """
        if self.current_active_scenario_type in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            intersection_index_ego = (
                CarlaDataProvider.get_current_scenario_memory().get(
                    "intersection_index_ego",
                    None,
                )
            )
            if intersection_index_ego is not None:
                return (
                    intersection_index_ego - self.privileged_route_planner.route_index
                ) / self.config_expert.points_per_meter
        return float("inf")

    @step_cached_property
    @beartype
    def last_encountered_speed_limit_sign(self) -> float:
        ret = self.ego_vehicle.get_speed_limit()
        if ret is not None:
            ret /= 3.6
        return ret

    @step_cached_property
    @beartype
    def speed_limit(self) -> float:
        if self.last_encountered_speed_limit_sign is not None:
            return self.last_encountered_speed_limit_sign
        return 30.0 / 3.6

    @step_cached_property
    @beartype
    def adversarial_actors_ids(self) -> tuple[list, list, list]:
        """
        Return a tuple of:
            - dangerous adversarial actors IDs: we should be very waried of them
            - safe adversarial actors IDs: we can treat their bounding boxes a bit smaller
            - ignored adversarial actors IDs: we can ignore them completely
        """
        # Obstacle scenarios: compute source and target lane once
        if self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
        ]:
            obstacle, direction = [
                CarlaDataProvider.get_current_scenario_memory()[key]
                for key in ["first_actor", "direction"]
            ]
            source_lane = self.carla_world_map.get_waypoint(
                obstacle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            target_lane = (
                source_lane.get_right_lane()
                if direction == "left"
                else source_lane.get_left_lane()
            )
            if source_lane and target_lane:
                CarlaDataProvider.get_current_scenario_memory()["source_lane"] = (
                    source_lane
                )
                CarlaDataProvider.get_current_scenario_memory()["target_lane"] = (
                    target_lane
                )

        if self.current_active_scenario_type in ["HazardAtSideLane"]:
            if CarlaDataProvider.get_current_scenario_memory()["bicycle_1"] is not None:
                target_lane = CarlaDataProvider.get_current_scenario_memory()[
                    "target_lane"
                ]
                source_lane = CarlaDataProvider.get_current_scenario_memory()[
                    "source_lane"
                ]
                if target_lane is None or source_lane is None:
                    bicycle_1 = CarlaDataProvider.get_current_scenario_memory()[
                        "bicycle_1"
                    ]
                    source_lane = self.carla_world_map.get_waypoint(
                        bicycle_1.get_location(),
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    target_lane = source_lane.get_left_lane()
                    CarlaDataProvider.get_current_scenario_memory()["target_lane"] = (
                        target_lane
                    )
                    CarlaDataProvider.get_current_scenario_memory()["soure_lane"] = (
                        source_lane
                    )

        # One way obstacle scenarios: adversarial actors are those on the target lane
        if (
            min(
                [
                    self.distance_to_accident_site,
                    self.distance_to_construction_site,
                    self.distance_to_parked_obstacle,
                ],
            )
            <= 40
            or self.current_active_scenario_type == "HazardAtSideLane"
        ):
            for scenario in [
                "Accident",
                "ConstructionObstacle",
                "ParkedObstacle",
                "HazardAtSideLane",
            ]:
                dangerous_adversarial_actors_ids = []
                safe_adversarial_actors_ids = []
                ignored_adversarial_actors_ids = []
                if self.current_active_scenario_type != scenario:
                    continue

                # Get memory for the current active scenario
                current_memory = CarlaDataProvider.get_current_scenario_memory()
                if current_memory is None:
                    continue

                if (
                    current_memory
                    and "source_lane" in current_memory
                    and "target_lane" in current_memory
                    and current_memory["source_lane"] is not None
                    and current_memory["target_lane"] is not None
                ):
                    target_lane = current_memory["target_lane"]
                    for actor in self.vehicles_inside_bev:
                        if actor.id == self.ego_vehicle.id:
                            continue
                        actor_lane = self.carla_world_map.get_waypoint(
                            actor.get_location(),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        if actor_lane and actor_lane.lane_id == target_lane.lane_id:
                            rel_loc = common_utils.get_relative_transform(
                                self.ego_matrix,
                                np.array(actor.get_transform().get_matrix()),
                            )
                            if self.speed_limit > 25:
                                dangerous_adversarial_actors_ids.append(actor.id)
                            else:
                                if 0.5 <= rel_loc[0]:  # actor in front, is safe
                                    safe_adversarial_actors_ids.append(actor.id)
                                else:  # Normal speed, be more careful
                                    dangerous_adversarial_actors_ids.append(actor.id)
                current_memory["dangerous_adversarial_actors_ids"] = (
                    dangerous_adversarial_actors_ids
                )
                current_memory["safe_adversarial_actors_ids"] = (
                    safe_adversarial_actors_ids
                )
                current_memory["ignored_adversarial_actors_ids"] = (
                    ignored_adversarial_actors_ids
                )

        # High speed merging scenarios
        for scenario in [
            "EnterActorFlow",
            "EnterActorFlowV2",
            "InterurbanAdvancedActorFlow",
        ]:
            if self.current_active_scenario_type != scenario:
                continue
            safe_adversarial_actors_ids = []
            ignored_adversarial_actors_ids = []
            dangerous_adversarial_actors_ids = []
            for adversarial_actor in CarlaDataProvider.get_current_scenario_memory()[
                "adversarial_actors"
            ]:
                try:
                    if not self.is_actor_inside_bev(adversarial_actor):
                        continue
                    adversarial_lane = self.carla_world_map.get_waypoint(
                        adversarial_actor.get_location(),
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    if self.ego_lane_id != adversarial_lane.lane_id:
                        continue
                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                except:
                    pass

            CarlaDataProvider.get_current_scenario_memory()[
                "dangerous_adversarial_actors_ids"
            ] = dangerous_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "safe_adversarial_actors_ids"
            ] = safe_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "ignored_adversarial_actors_ids"
            ] = ignored_adversarial_actors_ids

        # Priority scenarios
        for scenario in [
            "OppositeVehicleRunningRedLight",
            "OppositeVehicleTakingPriority",
        ]:
            if self.current_active_scenario_type != scenario:
                continue
            safe_adversarial_actors_ids = []
            ignored_adversarial_actors_ids = []
            dangerous_adversarial_actors_ids = []
            for adversarial_actor in CarlaDataProvider.get_current_scenario_memory()[
                "adversarial_actors"
            ]:
                try:
                    if (
                        not self.is_actor_inside_bev(adversarial_actor)
                        or adversarial_actor.get_velocity().length() < 0.1
                        or (
                            self.id2bb_map[adversarial_actor.id]["visible_pixels"] < 10
                            and self.id2bb_map[adversarial_actor.id]["num_points"] < 10
                        )
                    ):
                        continue
                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                except:
                    pass

            CarlaDataProvider.get_current_scenario_memory()[
                "dangerous_adversarial_actors_ids"
            ] = dangerous_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "safe_adversarial_actors_ids"
            ] = safe_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "ignored_adversarial_actors_ids"
            ] = ignored_adversarial_actors_ids

        # Unprotected left and right turns scenarios
        for scenario in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            if self.current_active_scenario_type != scenario:
                continue

            # Get memory for the current active scenario
            current_memory = CarlaDataProvider.get_current_scenario_memory()
            if current_memory is None:
                continue

            # Computer intersection point of ego route and adversarial route
            source_wp: carla.Waypoint = current_memory["source_wp"]
            sink_wp: carla.Waypoint = current_memory["sink_wp"]

            # Skip if waypoints are None (invalid spawn locations)
            if source_wp is None or sink_wp is None:
                continue

            opponent_traffic_route = current_memory["opponent_traffic_route"]
            if opponent_traffic_route is None:
                opponent_traffic_route = expert_utils.compute_global_route(
                    world=self.carla_world,
                    source_location=source_wp.transform.location,
                    sink_location=sink_wp.transform.location,
                )
                current_memory["opponent_traffic_route"] = opponent_traffic_route

            intersection_point = current_memory["intersection_point"]
            if opponent_traffic_route is not None and intersection_point is None:
                intersection_point, intersection_index_ego = (
                    expert_utils.intersection_of_routes(
                        points_a=self.route_waypoints_np[
                            : self.config_expert.draw_future_route_till_distance
                        ],  # Don't use full route otherwise too expensive
                        points_b=opponent_traffic_route,
                    )
                )
                if intersection_index_ego is not None:
                    intersection_index_ego += self.privileged_route_planner.route_index
                current_memory["intersection_index_ego"] = intersection_index_ego
                current_memory["intersection_point"] = intersection_point

            # Filter adversarial actors for unprotected left turns
            intersection_point = current_memory["intersection_point"]
            if intersection_point is not None:
                safe_adversarial_actors_ids = current_memory[
                    "safe_adversarial_actors_ids"
                ]  # We keep track safe adversarial actors over time, once they are safe, they won't be dangerous anymore
                ignored_adversarial_actors_ids = []
                dangerous_adversarial_actors_ids = []
                for adversarial_actor in current_memory["adversarial_actors"]:
                    if adversarial_actor.id == self.ego_vehicle.id:
                        continue
                    if adversarial_actor.id in safe_adversarial_actors_ids:
                        continue
                    try:
                        if not self.is_actor_inside_bev(adversarial_actor):
                            continue
                        if (
                            self.distance_to_next_junction < 10
                            and (
                                (
                                    self.distance_to_intersection_index_ego < 13
                                    and scenario
                                    in [
                                        "SignalizedJunctionLeftTurnEnterFlow",
                                        "NonSignalizedJunctionLeftTurnEnterFlow",
                                    ]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 13
                                    and scenario
                                    in [
                                        "NonSignalizedJunctionLeftTurn",
                                        "InterurbanActorFlow",
                                    ]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 18
                                    and scenario
                                    in [
                                        "SignalizedJunctionRightTurn",
                                        "NonSignalizedJunctionRightTurn",
                                    ]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 23
                                    and scenario in ["SignalizedJunctionLeftTurn"]
                                )
                            )
                            and not self.stop_sign_hazard
                            and not self.traffic_light_hazard
                        ):  # If only we are not near enough to the junction, we ignore adversarial actors. Smoother stopping
                            if scenario in [
                                "SignalizedJunctionLeftTurn",
                                "NonSignalizedJunctionLeftTurn",
                                "InterurbanActorFlow",
                            ]:
                                safe_threshold = (
                                    self.distance_to_intersection_index_ego * 1.1
                                )  # Safe threshold, the lower, the earlier we ignore an adversarial actor
                                if (
                                    scenario in ["SignalizedJunctionLeftTurn"]
                                ):  # Urban scenarios, we need to treat them a bit differently
                                    if self.distance_to_intersection_index_ego < 13:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.2
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                    else:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.6
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                safe_threshold = min(
                                    safe_threshold,
                                    22,
                                )  # Don't go too far, otherwise we ignore all actors
                                if (
                                    adversarial_actor.get_location().distance(
                                        intersection_point,
                                    )
                                    < safe_threshold
                                ):  # If actor is/was near enough to the intersection point, we can safely ignore it
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                                else:
                                    dangerous_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                            elif scenario in [
                                "SignalizedJunctionRightTurn",
                                "NonSignalizedJunctionRightTurn",
                            ]:
                                safe_threshold = (
                                    self.distance_to_intersection_index_ego * 1.1
                                )
                                if (
                                    scenario in ["SignalizedJunctionRightTurn"]
                                ):  # Urban scenarios, we need to treat them a bit differently
                                    if self.distance_to_intersection_index_ego < 13:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.2
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                    else:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.6
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                safe_threshold = min(
                                    safe_threshold,
                                    22,
                                )  # Don't go too far, otherwise we ignore all actors
                                if self.distance_to_intersection_index_ego < 5:
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )  # If we are very close to the intersection, we really want to commit
                                elif (
                                    adversarial_actor.get_location().distance(
                                        intersection_point,
                                    )
                                    < safe_threshold
                                ):  # If actor is/was near enough to the intersection point, we can safely ignore it
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                                else:
                                    dangerous_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                            elif scenario in [
                                "SignalizedJunctionLeftTurnEnterFlow",
                                "NonSignalizedJunctionLeftTurnEnterFlow",
                            ]:
                                adversarial_actor_location = (
                                    adversarial_actor.get_location()
                                )
                                if (
                                    expert_utils.distance_location_to_route(
                                        route=current_memory["opponent_traffic_route"],
                                        location=np.array(
                                            [
                                                adversarial_actor_location.x,
                                                adversarial_actor_location.y,
                                                adversarial_actor_location.z,
                                            ],
                                        ),
                                    )
                                    > 1.0
                                ):
                                    # If actor is further than the intersection point in the route, we can safely ignore it
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                                    LOG.info(
                                        "Adversarial actor went out of route. ignore",
                                    )
                                else:
                                    dangerous_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                        else:
                            ignored_adversarial_actors_ids = [
                                actor.id
                                for actor in CarlaDataProvider.get_current_scenario_memory()[
                                    "adversarial_actors"
                                ]
                            ]

                    except RuntimeError as e:
                        if "trying to operate on a destroyed actor" in str(e):
                            ignored_adversarial_actors_ids.append(adversarial_actor.id)
                            continue
                        else:
                            raise e

                current_memory["dangerous_adversarial_actors_ids"] = (
                    dangerous_adversarial_actors_ids
                )
                current_memory["safe_adversarial_actors_ids"] = (
                    safe_adversarial_actors_ids
                )
                current_memory["ignored_adversarial_actors_ids"] = (
                    ignored_adversarial_actors_ids
                )

        current_scenario_memory = CarlaDataProvider.get_current_scenario_memory()
        if (
            current_scenario_memory is not None
            and "dangerous_adversarial_actors_ids" in current_scenario_memory
        ):
            return (
                current_scenario_memory["dangerous_adversarial_actors_ids"],
                current_scenario_memory["safe_adversarial_actors_ids"],
                current_scenario_memory["ignored_adversarial_actors_ids"],
            )
        return [], [], []

    @step_cached_property
    def rear_adversarial_actor(self) -> carla.Actor | None:
        rear_adversarial_vehicle = None
        if (
            self.current_active_scenario_type
            in [
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
            ]
            and self.distance_to_intersection_index_ego < 2
        ):
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self.ego_vehicle.id:
                    continue
                if (
                    vehicle.id not in dangerous_adversarial_actors_ids
                    and vehicle.id not in safe_adversarial_actors_ids
                ):
                    continue
                rel_loc = common_utils.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        elif self.current_active_scenario_type in [
            "EnterActorFlow",
            "EnterActorFlowV2",
            "InterurbanAdvancedActorFlow",
        ]:
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self.ego_vehicle.id:
                    continue
                if vehicle.id not in dangerous_adversarial_actors_ids:
                    continue
                rel_loc = common_utils.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        elif self.current_active_scenario_type in [
            "OppositeVehicleRunningRedLight",
            "OppositeVehicleTakingPriority",
        ]:
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self.ego_vehicle.id:
                    continue
                if vehicle.id not in dangerous_adversarial_actors_ids:
                    continue
                rel_loc = common_utils.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        return rear_adversarial_vehicle

    @step_cached_property
    def target_lane_width(self):
        if self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
            "HazardAtSideLane",
        ]:
            target_lane = CarlaDataProvider.get_current_scenario_memory()["target_lane"]
            if target_lane is not None:
                return target_lane.lane_width

        if self.current_active_scenario_type in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            sink_wp = CarlaDataProvider.get_current_scenario_memory()["sink_wp"]
            if sink_wp is not None:
                return sink_wp.lane_width

        return None

    @step_cached_property
    def ego_lane_width(self) -> float:
        """
        Returns the width of the lane the ego vehicle is currently on.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        return self.ego_lane.lane_width

    @step_cached_property
    def transform_3rd_person_camera(self):
        ego_camera_location = carla.Location(
            x=self.config_expert.camera_3rd_person_calibration["x"],
            y=self.config_expert.camera_3rd_person_calibration["y"],
            z=self.config_expert.camera_3rd_person_calibration["z"],
        )
        world_camera_location = common_utils.get_world_coordinate_2d(
            self.ego_vehicle.get_transform(),
            ego_camera_location,
        )
        return carla.Transform(
            world_camera_location,
            carla.Rotation(
                pitch=self.config_expert.camera_3rd_person_calibration["pitch"],
                yaw=self.ego_vehicle.get_transform().rotation.yaw
                + self.config_expert.camera_3rd_person_calibration["yaw"],
            ),
        )

    @step_cached_property
    @beartype
    def ego_lane(self) -> carla.Waypoint:
        """
        Returns the current lane of the ego vehicle as a CARLA waypoint.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        return self.carla_world_map.get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    @step_cached_property
    def ego_lane_id(self) -> int:
        """
        Returns the current lane ID of the ego vehicle.
        """
        return self.ego_lane.lane_id

    @step_cached_property
    def ego_transform(self) -> carla.Transform:
        return self.ego_vehicle.get_transform()

    @step_cached_property
    def ego_location(self) -> carla.Location:
        return self.ego_vehicle.get_location()

    @property
    def ego_location_array(self) -> jt.Float[npt.NDArray, " 3"]:
        """
        Returns the ego vehicle's location as a numpy array.
        """
        location = self.ego_location
        return np.array([location.x, location.y, location.z])

    @step_cached_property
    def ego_speed(self) -> float:
        return self.ego_vehicle.get_velocity().length()

    @step_cached_property
    def ego_yaw_degree(self) -> float:
        return self.ego_vehicle.get_transform().rotation.yaw

    @step_cached_property
    def ego_orientation_rad(self) -> float | None:
        return self.compass

    @cached_property_by(lambda self: self.privileged_route_planner.route_index)
    @beartype
    def route_waypoints(self) -> list[carla.Waypoint]:
        return self.privileged_route_planner.route_waypoints[
            self.privileged_route_planner.route_index :
        ]

    @cached_property_by(lambda self: self.privileged_route_planner.route_index)
    def route_waypoints_np(self) -> jt.Float[npt.NDArray, "N 3"]:
        return self.privileged_route_planner.route_points[
            self.privileged_route_planner.route_index :
        ]

    @cached_property_by(lambda self: self.privileged_route_planner.route_index)
    def original_route_waypoints_np(self) -> jt.Float[npt.NDArray, "N 3"]:
        return self.privileged_route_planner.original_route_points[
            self.privileged_route_planner.route_index :
        ]

    @step_cached_property
    def signed_dist_to_lane_change(self) -> float:
        """
        Compute the signed distance to the next or previous lane change command in the route.

        Returns:
            float: Signed distance to the next lane change command. Positive if ahead, negative if behind.
            Inf if no lane change command found in proximity.
        """
        route_points = self.privileged_route_planner.route_points
        current_index = self.privileged_route_planner.route_index
        from_index = max(0, current_index - 250)
        to_index = min(len(route_points) - 1, current_index + 250)
        # Iterate over the points around the current position, checking for lane change commands

        def dist(index_a, index_b):
            index_min = min(index_a, index_b)
            index_max = max(index_a, index_b)
            d = 0
            for i in range(index_min, index_max):
                p1 = route_points[i]
                p2 = route_points[i + 1]
                d += np.linalg.norm(p2 - p1)
            if index_a < index_b:
                return d
            return -d

        min_dist = np.inf
        for i in range(from_index, to_index, 1):
            if self.privileged_route_planner.commands[i] in (
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ):
                considered_dist = dist(current_index, i)
                if abs(considered_dist) < abs(min_dist):
                    min_dist = considered_dist

        return min_dist / self.config_expert.points_per_meter

    @property
    def ego_wp(self) -> carla.Waypoint:
        return self.route_waypoints[0]

    @step_cached_property
    def ego_matrix(self) -> jt.Float[npt.NDArray, "4 4"]:
        return np.array(self.ego_vehicle.get_transform().get_matrix())

    @step_cached_property
    def inv_ego_matrix(self) -> jt.Float[npt.NDArray, "4 4"]:
        return np.linalg.inv(self.ego_matrix)

    @property
    def current_active_scenario_type(self) -> str | None:
        if len(CarlaDataProvider.active_scenarios) == 0:
            return None
        return CarlaDataProvider.active_scenarios[0].name

    @property
    def previous_active_scenario_type(self) -> str | None:
        if CarlaDataProvider.previous_active_scenario is not None:
            return CarlaDataProvider.previous_active_scenario.name
        return None

    @step_cached_property
    def distance_to_construction_site(self) -> float:
        if self.current_active_scenario_type in [
            "ConstructionObstacle",
            "ConstructionObstacleTwoWays",
        ] or self.previous_active_scenario_type in [
            "ConstructionObstacle",
            "ConstructionObstacleTwoWays",
        ]:
            num_cones = 0
            num_warning_traffic_signs = 0
            distances = []
            for static in self.static_inside_bev:
                if static.type_id == "static.prop.constructioncone":
                    num_cones += 1
                    distances.append(static.get_location().distance(self.ego_location))
                elif static.type_id == "static.prop.trafficwarning":
                    num_warning_traffic_signs += 1
                    distances.append(static.get_location().distance(self.ego_location))
            if num_cones > 0 and num_warning_traffic_signs > 0:
                distances = np.array(distances)
                distance = distances.mean()
                return float(distance)
        return float("inf")

    @step_cached_property
    def distance_to_scenario_obstacle(self) -> float:
        return min(
            [
                self.distance_to_accident_site,
                self.distance_to_construction_site,
                self.distance_to_parked_obstacle,
                self.distance_to_vehicle_opens_door,
            ],
        )

    @step_cached_property
    def distance_to_accident_site(self) -> float:
        if self.current_active_scenario_type in [
            "Accident",
            "AccidentTwoWays",
        ] or self.previous_active_scenario_type in [
            "Accident",
            "AccidentTwoWays",
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if (
                    "scenario" in actor.attributes["role_name"]
                    and self._get_actor_forward_speed(actor) == 0.0
                ):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def distance_to_parked_obstacle(self) -> float:
        if self.current_active_scenario_type in [
            "ParkedObstacle",
            "ParkedObstacleTwoWays",
        ] or self.previous_active_scenario_type in [
            "ParkedObstacle",
            "ParkedObstacleTwoWays",
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if (
                    "scenario" in actor.attributes["role_name"]
                    and self._get_actor_forward_speed(actor) == 0.0
                ):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def distance_to_vehicle_opens_door(self) -> float:
        if self.current_active_scenario_type in [
            "VehicleOpensDoorTwoWays",
        ] or self.previous_active_scenario_type in ["VehicleOpensDoorTwoWays"]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if (
                    "scenario" in actor.attributes["role_name"]
                    and self._get_actor_forward_speed(actor) == 0.0
                ):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def distance_to_cutin_vehicle(self) -> float:
        if not self.config_expert.datagen:
            return float("inf")
        if self.current_active_scenario_type in [
            "ParkingCutIn",
            "StaticCutIn",
            "HighwayCutIn",
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.cutin_actors:
                if self.is_actor_inside_bev(actor):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def distance_to_pedestrian(self) -> float:
        """
        Calculate the distance to the closest pedestrian within the BEV (Bird's Eye View) range.

        Returns:
            float: The distance to the closest pedestrian, or infinity if no pedestrians are inside the BEV.
        """
        pedestrians = self.walkers_inside_bev
        if not pedestrians:
            return float("inf")

        # Get the location of the ego vehicle
        ego_location = self.ego_vehicle.get_location()

        # Find the closest pedestrian
        closest_pedestrian = min(
            pedestrians,
            key=lambda p: ego_location.distance(p.get_location()),
        )
        return ego_location.distance(closest_pedestrian.get_location())

    @step_cached_property
    def distance_to_biker(self) -> float:
        """
        Calculate the distance to the closest biker within the BEV (Bird's Eye View) range.

        Returns:
            float: The distance to the closest biker, or infinity if no bikers are inside the BEV.
        """
        bikers = self.bikers_inside_bev
        if not bikers:
            return float("inf")

        # Get the location of the ego vehicle
        ego_location = self.ego_vehicle.get_location()

        # Find the closest biker
        closest_biker = min(
            bikers,
            key=lambda b: ego_location.distance(b.get_location()),
        )
        return ego_location.distance(closest_biker.get_location())

    @step_cached_property
    def route_left_length(self):
        route_points = self.route_waypoints_np
        dist_diff = np.diff(route_points[:, :2], axis=0)
        segment_lengths = np.linalg.norm(dist_diff, axis=1)
        return np.sum(segment_lengths)

    @step_cached_property
    def distance_ego_to_route(self):
        ego_wp = self.ego_wp
        route_wp = self.route_waypoints[0]
        return ego_wp.transform.location.distance(route_wp.transform.location)

    @step_cached_property
    def route_curvature(self):
        route_points = self.route_waypoints_np
        total_curvature = 0.0
        for i in range(
            1,
            min(
                len(route_points) - 1,
                self.config_expert.high_road_curvature_max_future_points,
            )
            - 1,
        ):
            loc_prev = route_points[i - 1]
            loc_current = route_points[i]
            loc_next = route_points[i + 1]
            curv = (
                (loc_current[0] - loc_prev[0]) * (loc_next[1] - loc_current[1])
                - (loc_current[1] - loc_prev[1]) * (loc_next[0] - loc_current[0])
            ) / (
                (
                    (loc_current[0] - loc_prev[0]) ** 2
                    + (loc_current[1] - loc_prev[1]) ** 2
                )
                ** 0.5
                * (
                    (loc_next[0] - loc_current[0]) ** 2
                    + (loc_next[1] - loc_current[1]) ** 2
                )
                ** 0.5
            )
            total_curvature += abs(curv)
        return total_curvature

    @step_cached_property
    @beartype
    def vehicles_inside_bev(self) -> list[carla.Actor]:
        vehicles = self.carla_world.get_actors().filter("*vehicle*")
        vehicles = [
            vehicle for vehicle in vehicles if self.is_actor_inside_bev(vehicle)
        ]
        if (
            self.config_expert.datagen and self.config_expert.vehicle_occlusion_check
        ):  # Can only perform occlusion check if we have sensor data
            vehicles = [
                vehicle
                for vehicle in vehicles
                if not (
                    0
                    <= self.id2bb_map[vehicle.id]["num_points"]
                    < self.config_expert.vehicle_occlusion_check_min_num_points
                    and 0
                    <= self.id2bb_map[vehicle.id]["visible_pixels"]
                    < self.config_expert.vehicle_min_num_visible_pixels
                )
            ]
        return vehicles

    @step_cached_property
    @beartype
    def walkers_inside_bev(self) -> list[carla.Actor]:
        walkers = self.carla_world.get_actors().filter("*walker*")
        walkers = [walker for walker in walkers if self.is_actor_inside_bev(walker)]
        if (
            self.config_expert.datagen
        ):  # Can only perform occlusion check if we have sensor data
            walkers = [
                walker
                for walker in walkers
                if not (
                    0
                    <= self.id2bb_map[walker.id]["visible_pixels"]
                    < self.config_expert.pedestrian_min_num_visible_pixels
                )
            ]
        return walkers

    @step_cached_property
    @beartype
    def bikers_inside_bev(self) -> list[carla.Actor]:
        bikers = self.carla_world.get_actors().filter("*vehicle*")
        bikers = [
            b
            for b in bikers
            if b.type_id
            in [
                "vehicle.diamondback.century",
                "vehicle.gazelle.omafiets",
                "vehicle.bh.crossbike",
            ]
        ]
        bikers = [biker for biker in bikers if self.is_actor_inside_bev(biker)]
        if (
            self.config_expert.datagen and self.config_expert.bikers_occlusion_check
        ):  # Can only perform occlusion check if we have sensor data
            bikers = [
                biker
                for biker in bikers
                if not (
                    0
                    <= self.id2bb_map[biker.id]["visible_pixels"]
                    < self.config_expert.bikers_occlusion_check_min_visible_pixels
                )
            ]
        return bikers

    @step_cached_property
    def static_inside_bev(self) -> list[carla.Actor]:
        """
        Get static actors inside the BEV (Bird's Eye View) range.
        This includes traffic lights and other static objects that are not vehicles or walkers.

        Returns:
            list: A list of static actors inside the BEV.
        """
        static_actors = self.carla_world.get_actors().filter("*static*")
        static_actors = [
            actor for actor in static_actors if self.is_actor_inside_bev(actor)
        ]
        return static_actors

    @step_cached_property
    @beartype
    def distance_to_next_junction(self) -> float:
        ego_wp = self.carla_world_map.get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Any,
        )
        next_wps = expert_utils.wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []
        if ego_wp.is_junction:
            distance_to_junction_ego = 0.0
            # get distance to ego vehicle
        elif len(next_lane_wps_ego) > 0 and next_lane_wps_ego[0].is_junction:
            distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(
                ego_wp.transform.location,
            )
        else:
            distance_to_junction_ego = np.inf

        return float(distance_to_junction_ego)

    @step_cached_property
    @beartype
    def scenario_actors(self) -> list[carla.Actor]:
        ret = []
        for actor in (
            self.vehicles_inside_bev + self.walkers_inside_bev + self.bikers_inside_bev
        ):
            if "scenario" in actor.attributes["role_name"]:
                ret.append(actor)
        return ret

    @step_cached_property
    @beartype
    def scenario_actors_ids(self) -> list[int]:
        """
        Get the IDs of the scenario actors that are currently inside the BEV (Bird's Eye View) range.

        Returns:
            list: A list of IDs of the scenario actors.
        """
        return [actor.id for actor in self.scenario_actors]

    @step_cached_property
    @beartype
    def scenario_obstacles(self) -> list[carla.Actor]:
        ret = []
        scenarios = [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
            "AccidentTwoWays",
            "ConstructionObstacleTwoWays",
            "ParkedObstacleTwoWays",
            "VehicleOpensDoorTwoWays",
            "InvadingTurn",
            "BlockedIntersection",
        ]
        if self.current_active_scenario_type in scenarios:
            ret = CarlaDataProvider.get_current_scenario_memory()["obstacles"]
        elif self.previous_active_scenario_type in scenarios:
            if CarlaDataProvider.previous_active_scenario is not None:
                obstacles = CarlaDataProvider.previous_active_scenario.meta["obstacles"]
                try:
                    obstacles = [
                        actor for actor in obstacles if self.is_actor_inside_bev(actor)
                    ]
                    ret = obstacles
                except RuntimeError as e:
                    if "trying to operate on a destroyed actor" in str(e):
                        # If the scenario obstacles were destroyed, return an empty list
                        ret = []
                    else:
                        raise e
        ret = [actor for actor in ret if self.is_actor_inside_bev(actor)]
        return ret

    @step_cached_property
    def scenario_obstacles_convex_hull(self):
        """
        Get the convex hull of the scenario obstacles' bounding box corners that are currently inside the BEV range.

        Returns:
            list: A list of (x, y) points representing the convex hull of the obstacles.
        """
        if not self.scenario_obstacles:
            return []

        points = []
        for actor in self.scenario_obstacles:
            bbox = actor.bounding_box
            actor_transform = actor.get_transform()

            # Get bounding box center in world coordinates
            bbox_center_world = actor_transform.transform(bbox.location)

            # Rotation matrix from actor yaw
            yaw = np.radians(actor_transform.rotation.yaw)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            extent = bbox.extent

            # Define corners in local coordinates
            local_corners = [
                (extent.x, extent.y),
                (-extent.x, extent.y),
                (-extent.x, -extent.y),
                (extent.x, -extent.y),
            ]

            # Transform corners to world coordinates
            for lx, ly in local_corners:
                x = bbox_center_world.x + lx * cos_yaw - ly * sin_yaw
                y = bbox_center_world.y + lx * sin_yaw + ly * cos_yaw
                points.append((x, y))

        if len(points) < 3:
            return points

        points_np = np.array(points, dtype=np.float32)
        hull = cv2.convexHull(points_np)
        return hull.squeeze().tolist()

    @step_cached_property
    def scenario_obstacles_ids(self):
        """
        Get the IDs of the scenario obstacles that are currently inside the BEV (Bird's Eye View) range.

        Returns:
            list: A list of IDs of the scenario obstacles.
        """
        return [actor.id for actor in self.scenario_obstacles]

    @step_cached_property
    def vehicle_opened_door(self):
        """
        Check if the vehicle opened its door in the current scenario.
        This is used to determine if the agent should react to a vehicle opening its door.
        """
        if self.current_active_scenario_type == "VehicleOpensDoorTwoWays":
            return CarlaDataProvider.get_current_scenario_memory()[
                "vehicle_opened_door"
            ]
        elif self.previous_active_scenario_type == "VehicleOpensDoorTwoWays":
            if CarlaDataProvider.previous_active_scenario is not None:
                try:
                    CarlaDataProvider.previous_active_scenario.meta["obstacles"][
                        0
                    ].get_location()
                    return CarlaDataProvider.previous_active_scenario.meta[
                        "vehicle_opened_door"
                    ]
                except RuntimeError as e:
                    if "trying to operate on a destroyed actor" in str(e):
                        return False
                    else:
                        raise e
        return False

    @step_cached_property
    def vehicle_door_side(self):
        """
        Get the side of the vehicle that opened its door in the current scenario.
        This is used to determine if the agent should react to a vehicle opening its door.
        """
        if self.current_active_scenario_type == "VehicleOpensDoorTwoWays":
            return CarlaDataProvider.get_current_scenario_memory()["vehicle_door_side"]
        elif self.previous_active_scenario_type == "VehicleOpensDoorTwoWays":
            if CarlaDataProvider.previous_active_scenario is not None:
                return CarlaDataProvider.previous_active_scenario.meta[
                    "vehicle_door_side"
                ]
        return None

    @step_cached_property
    def cutin_actors(self):
        if self.current_active_scenario_type in [
            "ParkingCutIn",
            "StaticCutIn",
            "HighwayCutIn",
        ]:
            return [CarlaDataProvider.get_current_scenario_memory()["cut_in_vehicle"]]
        return []

    @step_cached_property
    def cut_in_actors_ids(self):
        return [actor.id for actor in self.cutin_actors]

    @step_cached_property
    def two_way_obstacle_distance_to_cones_factor(self):
        if self.ego_lane_width <= 2.76:
            return 1.13
        elif self.ego_lane_width <= 3.01:
            return 1.12
        return 1.12

    @step_cached_property
    def two_way_vehicle_open_door_distance_to_center_line_factor(self):
        if self.ego_lane_width <= 2.76:
            return 1.0
        elif self.ego_lane_width <= 3.01:
            return 0.875
        return 0.75

    @step_cached_property
    def add_after_construction_obstacle_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_after_construction_obstacle_two_ways + 0.5
        return self.config_expert.add_after_construction_obstacle_two_ways

    @step_cached_property
    def add_before_construction_obstacle_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_before_construction_obstacle_two_ways + 0.5
        return self.config_expert.add_before_construction_obstacle_two_ways

    @step_cached_property
    def two_way_overtake_speed(self):
        return {
            "AccidentTwoWays": self.config_expert.default_overtake_speed,
            "ConstructionObstacleTwoWays": self.config_expert.default_overtake_speed,
            "ParkedObstacleTwoWays": self.config_expert.default_overtake_speed,
            "VehicleOpensDoorTwoWays": self.config_expert.default_overtake_speed
            if self.ego_lane_width > 3.01
            else self.config_expert.overtake_speed_vehicle_opens_door_two_ways,
        }[self.current_active_scenario_type]

    @step_cached_property
    def add_after_accident_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_after_accident_two_ways + 0.5
        return self.config_expert.add_after_accident_two_ways

    @step_cached_property
    def add_before_accident_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_before_accident_two_ways + 0.5
        return self.config_expert.add_before_accident_two_ways

    @step_cached_property
    def num_parking_vehicles_in_proximity(self):
        count = 0
        for bb in self.stored_bounding_boxes_of_this_step:
            if not (-8 <= bb["position"][0] <= 32 and abs(bb["position"][1]) <= 10):
                continue
            if (
                bb["class"] == "static"
                and bb["transfuser_semantics_id"]
                == TransfuserSemanticSegmentationClass.VEHICLE
            ):
                count += 1
            if bb["class"] == "static_prop_car":
                count += 1
        return count

    @step_cached_property
    def second_highest_speed(self):
        speeds = []
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self.ego_vehicle.id:
                continue
            speeds.append(vehicle.get_velocity().length())
        if len(speeds) == 0:
            return 0.0
        elif len(speeds) == 1:
            return speeds[0]
        speeds = sorted(speeds, reverse=True)
        return speeds[1]

    @step_cached_property
    def second_highest_speed_limit(self):
        speed_limits = []
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self.ego_vehicle.id:
                continue
            speed_limits.append(vehicle.get_speed_limit() / 3.6)
        if len(speed_limits) == 0:
            return 0.0
        elif len(speed_limits) == 1:
            return speed_limits[0]
        speed_limits = sorted(speed_limits, reverse=True)
        return speed_limits[1]
