"""Driving logic for expert autonomous agent. Meta information is also handled here."""

import cProfile
import io
import logging
import numbers
import pstats
import time
import typing

import carla
import matplotlib
import numpy as np
from agents.tools.misc import compute_distance, is_within_distance
from beartype import beartype
from shapely.geometry import Polygon
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import lead.common.common_utils as common_utils
import lead.expert.expert_utils as expert_utils
from lead.common import constants, weathers
from lead.common.constants import WeatherVisibility
from lead.common.logging_config import setup_logging
from lead.common.pid_controller import ExpertLateralPIDController
from lead.expert.expert_data import ExpertData
from lead.expert.scenario_sorter import ScenarioSorter

matplotlib.use("Agg")  # non-GUI backend for headless servers


setup_logging()
LOG = logging.getLogger(__name__)


def get_entry_point() -> str:
    return "Expert"


class Expert(ExpertData):
    """Actor class for the expert autonomous agent in CARLA simulations.

    Consists driving logic, sensor configurations, and utility functions for expert driving behavior.
    Also handles sensor data processing and saving for data generation purposes.
    """

    @beartype
    def setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ):
        super().setup()
        LOG.info("Setup")
        self.expert_setup(path_to_conf_file, route_index, traffic_manager)

        self._turn_controller = ExpertLateralPIDController(self.config_expert)
        self.cleared_stop_sign = False

        # Initialize scenario sorter (will be configured with route in _init)
        self.scenario_sorter = ScenarioSorter()

        if self.config_expert.profile_expert:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

    def destroy(self, results=None) -> None:
        if self.config_expert.profile_expert:
            self.profiler.disable()

            # Save to file for visualization
            try:
                self.profiler.dump_stats("expert_profile.prof")
                LOG.info(
                    "Profile stats saved to 'expert_profile.prof'. Run 'pip install snakeviz && snakeviz expert_profile.prof' to view.",
                )
            except Exception as e:
                LOG.error(f"Failed to dump profile stats: {e}")

            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats("cumulative")
            ps.print_stats("lead/lead/")
            print(s.getvalue())
        super().destroy(results)

    @beartype
    def _init(self, hd_map) -> None:
        LOG.info("Init")
        self.expert_init(hd_map)

    @beartype
    def run_step(
        self,
        input_data: dict,
        _: float,
        __: list[list[str | typing.Any]] | None,
    ) -> carla.VehicleControl:
        """
        Entry main function!
        Run a single step of the agent's control loop.

        Args:
            input_data: Input data for the current step.

        Returns:
            The control commands.

        Raises:
            RuntimeError: If the agent is not initialized before calling this method.
        """
        start_time = time.time()

        # Initialize the agent if not done yet
        if not self.initialized:
            self._init(None)

        self.step += 1
        ego_location = self.ego_vehicle.get_location()
        self.scenario_sorter.sort_scenarios(ego_location)

        if not self.config_expert.eval_expert:
            if self.config_expert.datagen:
                self.perturbate_camera()

        self.update_3rd_person_camera()

        input_data = self.tick(input_data)

        # Get the control commands and driving data for the current step
        target_speed, control, speed_reduced_by_obj = self._get_control()

        # Visualize routes if enabled
        if self.config_expert.visualize_route:
            self._visualize_route()
        if self.config_expert.visualize_original_route:
            self._visualize_original_route()

        if (
            not self.config_expert.eval_expert
            and not self.config_expert.py123d_data_format
        ):
            if input_data is not None and "bounding_boxes" in input_data:
                self.bounding_boxes.append(
                    (
                        self.step,
                        self.step // self.config_expert.data_save_freq,
                        input_data["bounding_boxes"],
                    ),
                )

            self.save_meta(
                control,
                target_speed,
                input_data,
                speed_reduced_by_obj,
            )

        if (
            self.step % self.config_expert.data_save_freq == 0
            and not self.config_expert.py123d_data_format
        ):
            if self.save_path is not None and self.config_expert.datagen:
                self.save_sensors(input_data)
        if self.step % self.config_expert.log_info_freq == 0:
            elapsed_time = time.time() - start_time
            LOG.info(f"Step: {self.step}, Time per step: {elapsed_time * 1000:.2f} ms")
        return control

    @beartype
    def perturbate_camera(self) -> None:
        # Update dummy vehicle
        if self.initialized and self.config_expert.perturbate_sensors:
            # We are still rendering the map for the current frame, so we need to use the translation from the last frame.
            last_translation = self.perturbation_translation
            last_rotation = self.perturbation_rotation
            bb_copy = carla.BoundingBox(
                self.ego_vehicle.bounding_box.location,
                self.ego_vehicle.bounding_box.extent,
            )
            transform_copy = carla.Transform(
                self.ego_vehicle.get_transform().location,
                self.ego_vehicle.get_transform().rotation,
            )
            perturbated_loc = transform_copy.transform(
                carla.Location(0.0, last_translation, 0.0),
            )
            transform_copy.location = perturbated_loc
            transform_copy.rotation.yaw = transform_copy.rotation.yaw + last_rotation
            self.perturbated_vehicle_dummy.bounding_box = bb_copy
            self.perturbated_vehicle_dummy.transform = transform_copy

    @beartype
    def _solve_vulnerable_vehicles_scenarios(
        self,
        target_speed: float,
    ) -> tuple[float, bool, bool, list | None]:
        """
        Check for vulnerable vehicles (pedestrians, cyclists). If there are any vulnerable vehicles close enough,
        and we are inside certain scenarios, we brake and set the target speed to 0.

        This function checks if there are visible walker or bicycle inside 6m and 30° FOV. If yes, set target_speed to 0.

        Args:
            target_speed: The current target speed of the ego vehicle.

        Returns:
            A tuple containing (adjusted target_speed, is_overridden, should_brake, details or None).
        """
        ego_matrix = self.ego_matrix
        ego_location = ego_matrix[:3, 3]

        if self.current_active_scenario_type in [
            "VehicleTurningRoute",
            "VehicleTurningRoutePedestrian",
            "DynamicObjectCrossing",
            "ParkingCrossingPedestrian",
            "PedestrianCrossing",
        ]:
            for actor in self.walkers_inside_bev + self.bikers_inside_bev:
                num_visible_pixel = (
                    self.id2bb_map[actor.id]["visible_pixels"]
                    if self.config_expert.datagen
                    else -1
                )
                actor_height = actor.bounding_box.extent.z
                threshold = (
                    50
                    * (self.config_expert.image_height * self.config_expert.image_width)
                    / ((384**2) * 3)
                    * (3 / self.config_expert.num_cameras)
                )
                if 0 < num_visible_pixel / (actor_height**2) < threshold:
                    continue
                actor_velocity = actor.get_velocity().length()
                if (
                    actor_velocity < 0.25
                    and not CarlaDataProvider.get_current_scenario_memory()[
                        "pedestrian_moved"
                    ][actor.id]
                ):
                    continue  # Actor did not move yet and is not moving, we come nearer to trigger the scenario
                elif (
                    actor_velocity >= 0.25
                ):  # Actor is moving, we mark it as moving and continue with the scenario
                    CarlaDataProvider.get_current_scenario_memory()["pedestrian_moved"][
                        actor.id
                    ] = True

                bb_location = np.array(actor.get_transform().get_matrix())[:3, 3]
                rel_vector = bb_location - ego_location
                distance = np.linalg.norm(rel_vector)

                local_coords = self.inv_ego_matrix @ np.append(bb_location, 1.0)
                x, y = local_coords[0], local_coords[1]
                angle = np.abs(np.degrees(np.arctan2(y, x)))

                if x < 0:
                    continue  # Behind ego vehicle

                LOG.info(
                    f"Found vulnerable vehicle in front: {angle}°, {distance}m, emergency braking.",
                )
                return 0.0, True, True, [0, actor.type_id, actor.id, distance]

        return target_speed, False, False, None

    @beartype
    def _get_control(self) -> tuple[float, carla.VehicleControl, list | None]:
        """
        Compute the control commands for the current frame.

        Returns:
            A tuple containing the target speed, control commands, and speed_reduced_by_obj.
        """
        # Reset hazard flags
        self.stop_sign_close = False
        self.walker_close = False
        self.walker_close_id = None
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_affecting_id = None
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False

        self.europe_traffic_light = self.over_head_traffic_light = False
        # Waypoint planning and route generation
        self.privileged_route_planner.run_step(self.ego_location_array)

        # --- Target speed limit, the highest we can go depending on weather, surrounding, etc ---
        self.target_speed_limit = self.speed_limit
        self.target_speed_limit = max(
            self.target_speed_limit,
            self.second_highest_speed_limit,
        )  # We don't to drive too slow
        if (
            self.second_highest_speed > 30.0 / 3.6
        ):  # We don't want to drive too fast either
            self.target_speed_limit = min(
                self.target_speed_limit,
                self.second_highest_speed / 0.9,
            )

        # --- Drive slower with bad weather, night time, junction and clutterness in city ---
        self.weather_setting = expert_utils.get_weather_name(
            self.carla_world.get_weather(),
            self.config_expert,
        )
        self.visual_visibility = int(
            weathers.WEATHER_VISIBILITY_MAPPING[self.weather_setting],
        )
        self.slower_bad_visibility = False
        self.slower_clutterness = False
        self.slower_occluded_junction = False
        if (
            self.current_active_scenario_type
            not in [
                "EnterActorFlow",
                "EnterActorFlowV2",
                "MergerIntoSlowTraffic",
                "MergerIntoSlowTrafficV2",
                "HighwayExit",
                "NonSignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
                "InterurbanActorFlow",
                "InterurbanAdvancedActorFlow",
            ]
            and min(self.target_speed_limit, self.speed_limit)
            < 60 / 3.6  # Assume urban = 60kmh
        ):
            if self.distance_to_next_junction > 0:
                if self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                    self.target_speed_limit -= 4.0
                    self.slower_bad_visibility = True
                if self.num_parking_vehicles_in_proximity >= 2:
                    self.target_speed_limit -= 2.0
                    self.slower_clutterness = True
                if (
                    self.distance_to_next_junction < 20
                    and min(
                        [
                            self.visibility_range_camera_1,
                            self.visibility_range_camera_2,
                            self.visibility_range_camera_3,
                        ],
                    )
                    < 0.2
                ):
                    self.slower_occluded_junction = True
                    self.target_speed_limit -= 4.0

            # Reduce target speed if there is a junction ahead
            for i in range(
                min(
                    self.config_expert.max_lookahead_to_check_for_junction,
                    len(self.route_waypoints),
                ),
            ):
                if self.route_waypoints[i].is_junction:
                    if self.speed_limit < 40 / 3.6:
                        self.target_speed_limit = min(
                            self.target_speed_limit,
                            self.config_expert.max_speed_in_junction_urban,
                        )

        else:
            self.target_speed_limit = max(
                self.target_speed_limit,
                40.0 / 3.6,
            )  # We don't want to drive too slow

        self.target_speed_limit = max(
            self.target_speed_limit,
            self.config_expert.min_target_speed_limit,
        )
        self.target_speed_limit = min(self.target_speed_limit, 20.0)

        # --- Target speed, begins with target speed limit, reduces further depending on scenarios ---
        target_speed = self.target_speed_limit

        # Manage route obstacle scenarios and adjust target speed
        target_speed_route_obstacle, obstacle_override, speed_reduced_by_obj = (
            self._solve_obstacle_scenarios(target_speed)
        )

        # Manage distance to pedestrians and bikers
        (
            target_speed_vulnerable_vehicles,
            vulnerable_vehicle_override,
            brake_vulnerable_vehicle,
            speed_reduced_by_obj_vulnerable_vehicles,
        ) = self._solve_vulnerable_vehicles_scenarios(target_speed)
        self.does_emergency_brake_for_pedestrians = vulnerable_vehicle_override

        assert int(obstacle_override) + int(vulnerable_vehicle_override) <= 1, (
            "Only one override can be active at a time."
        )

        # Specific cases
        if obstacle_override:
            # In obstacle override, we force the vehicle to drive no matter what.
            LOG.info(
                f"[Control Override] Obstacle override active, forcing drive with speed {target_speed_route_obstacle * 3.6:.1f} km/h",
            )
            brake, target_speed = False, target_speed_route_obstacle
            speed_reduced_by_obj = speed_reduced_by_obj
        elif vulnerable_vehicle_override:
            LOG.info(
                f"[Control Override] Vulnerable vehicle override active, adjusting speed to {min(target_speed, target_speed_vulnerable_vehicles) * 3.6:.1f} km/h, brake={brake_vulnerable_vehicle}",
            )
            brake, target_speed = (
                brake_vulnerable_vehicle,
                min(target_speed, target_speed_vulnerable_vehicles),
            )
            speed_reduced_by_obj = speed_reduced_by_obj_vulnerable_vehicles
        else:  # Generate cases
            brake, target_speed, speed_reduced_by_obj = self._solve_general_scenarios(
                target_speed,
                speed_reduced_by_obj,
            )

        target_speed = min(target_speed, target_speed_route_obstacle)

        self.emergency_brake_for_special_vehicle = False
        if self.current_active_scenario_type in [
            "OppositeVehicleTakingPriority",
            "OppositeVehicleRunningRedLight",
        ]:
            if len(self.adversarial_actors_ids[0]) > 0:
                LOG.info(
                    f"[Control] {self.current_active_scenario_type}: Reducing target speed to 0, waiting for dangerous adversarial vehicle to pass.",
                )
                target_speed = 0.0
                brake = True
        elif self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
        ]:
            if self.speed_limit > 25:
                if 25 < self.distance_to_scenario_obstacle < 50:
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Reducing target speed to 10.0 m/s, approaching visible obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 10.0)
            elif self.speed_limit > 20:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Reducing target speed to 7.5 m/s, approaching visible obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 7.5)
            else:
                if 25 < self.distance_to_scenario_obstacle < 40:
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Reducing target speed to {self.config_expert.min_target_speed_limit} m/s, approaching visible obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(
                        target_speed,
                        self.config_expert.min_target_speed_limit,
                    )
        elif self.current_active_scenario_type in [
            "ParkedObstacle",
        ]:
            if self.speed_limit > 25:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info(
                        f"[Control] ParkedObstacle: Reducing target speed to 10.0 m/s, approaching parked obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 10.0)
            elif self.speed_limit > 20:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info(
                        f"[Control] ParkedObstacle: Reducing target speed to 7.5 m/s, approaching parked obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 7.5)
            else:
                if 25 < self.distance_to_scenario_obstacle < 35:
                    LOG.info(
                        f"[Control] ParkedObstacle: Reducing target speed to {self.config_expert.min_target_speed_limit} m/s, approaching parked obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(
                        target_speed,
                        self.config_expert.min_target_speed_limit,
                    )

        # Attempt to try to avoid collision in some cases
        self.rear_danger_8 = self.rear_danger_16 = False
        if self.current_active_scenario_type in [
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionRightTurn",
            "EnterActorFlow",
            "EnterActorFlowV2",
            "InterurbanAdvancedActorFlow",
            "OppositeVehicleRunningRedLight",
            "OppositeVehicleTakingPriority",
            "OppositeVehicleTakingPriority",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "SignalizedJunctionLeftTurnEnterFlow",
        ]:
            rear_adversarial_actor = self.rear_adversarial_actor
            if rear_adversarial_actor:
                vehicle_behind_speed = rear_adversarial_actor.get_velocity().length()
                vehicle_behind_distance = self.ego_location.distance(
                    rear_adversarial_actor.get_location(),
                )
                if (
                    vehicle_behind_distance < 8
                    and vehicle_behind_speed > 5
                    and self.ego_speed > 5
                ):
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Rear danger (8m): Accelerating to avoid rear collision, vehicle behind speed: {vehicle_behind_speed * 3.6:.1f} km/h, adjusting target speed to {(vehicle_behind_speed + 5) * 3.6:.1f} km/h",
                    )
                    target_speed = max(target_speed, vehicle_behind_speed + 5)
                    self.rear_danger_8 = True
                    brake = False
                elif (
                    vehicle_behind_distance < 16
                    and vehicle_behind_speed > 5
                    and self.ego_speed > 5
                ):
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Rear danger (16m): Matching rear vehicle speed: {vehicle_behind_speed * 3.6:.1f} km/h",
                    )
                    target_speed = max(target_speed, vehicle_behind_speed)
                    self.rear_danger_16 = True
                    brake = False

        # Safety brake if some vehicles cutin from side. This make the imitation learning easier.
        self.brake_cutin = False
        if self.current_active_scenario_type in ["ParkingCutIn"]:
            assert len(self.cutin_actors) == 1
            cut_in_vehicle = self.cutin_actors[0]
            if (
                1 < cut_in_vehicle.get_velocity().length() < 4.25
                and not CarlaDataProvider.get_current_scenario_memory()["stopped"]
            ):
                LOG.info(
                    f"[Control] ParkingCutIn: Emergency braking for cut-in vehicle (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                self.brake_cutin = True
                brake = True
                target_speed = 0.0
            elif (
                not CarlaDataProvider.get_current_scenario_memory()["stopped"]
                and cut_in_vehicle.get_velocity().length() >= 5
            ):
                LOG.info(
                    f"[Control] ParkingCutIn: Cut-in vehicle cleared (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                CarlaDataProvider.get_current_scenario_memory()["stopped"] = True
        elif self.current_active_scenario_type in ["StaticCutIn"]:
            assert len(self.cutin_actors) == 1
            cut_in_vehicle = self.cutin_actors[0]
            if (
                2.1 < cut_in_vehicle.get_velocity().length() < 4.25
                and not CarlaDataProvider.get_current_scenario_memory()["stopped"]
            ):
                LOG.info(
                    f"[Control] StaticCutIn: Emergency braking for cut-in vehicle (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                self.brake_cutin = True
                brake = True
                target_speed = 0.0
            elif (
                not CarlaDataProvider.get_current_scenario_memory()["stopped"]
                and cut_in_vehicle.get_velocity().length() >= 5
            ):
                LOG.info(
                    f"[Control] StaticCutIn: Cut-in vehicle cleared (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                CarlaDataProvider.get_current_scenario_memory()["stopped"] = True

        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(
            brake,
            target_speed,
            self.ego_speed,
        )

        # Compute steering control
        steer = expert_utils.get_steer(
            self.config_expert,
            self._turn_controller,
            self.route_waypoints_np,
            self.ego_location_array,
            self.ego_orientation_rad,
            self.ego_speed,
        )

        # Create the control command
        self.control = carla.VehicleControl()
        self.control.steer = steer + self.config_expert.steer_noise * np.random.randn()
        self.control.throttle = throttle
        self.control.brake = float(brake or control_brake)

        # Apply brake if the vehicle is stopped to prevent rolling back
        if (
            self.control.throttle == 0
            and self.ego_speed
            < self.config_expert.minimum_speed_to_prevent_rolling_back
        ):
            self.control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        if self.ego_blocked_for_ticks >= self.config_expert.max_blocked_ticks:
            self.control.throttle = 1
            self.control.brake = 0

        # Run the command planner
        self._command_planner.run_step(self.ego_location_array)
        for v in self._command_planners_dict.values():
            v.run_step(self.ego_location_array)

        return float(target_speed), self.control, speed_reduced_by_obj

    @beartype
    def is_two_ways_overtaking_path_clear(
        self,
        from_index: int,
        to_index: int,
        target_speed: numbers.Real,
        previous_lane_ids: list,
        min_speed: numbers.Real = 50.0 / 3.6,
    ) -> bool:
        """
        Checks if the path between two route indices is clear for the ego vehicle to overtake in two ways scenarios.

        Args:
            from_index: The starting route index.
            to_index: The ending route index.
            target_speed: The target speed of the ego vehicle.
            previous_lane_ids: A list of tuples containing previous road IDs and lane IDs.
            min_speed: The minimum speed to consider for overtaking. Defaults to 50/3.6 km/h.

        Returns:
            True if the path is clear for overtaking, False otherwise.
        """
        # 10 m safety distance, overtake with max. 50 km/h
        to_location = self.privileged_route_planner.route_points[to_index]
        to_location = carla.Location(to_location[0], to_location[1], to_location[2])

        from_location = self.privileged_route_planner.route_points[from_index]
        from_location = carla.Location(
            from_location[0],
            from_location[1],
            from_location[2],
        )

        # Compute the distance and time needed for the ego vehicle to overtake
        ego_distance = (
            to_location.distance(self.ego_location)
            + self.ego_vehicle.bounding_box.extent.x * 2
            + self.config_expert.check_path_free_safety_distance
        )
        ego_time = expert_utils.compute_min_time_for_distance(
            self.config_expert,
            ego_distance,
            min(min_speed, target_speed),
            self.ego_speed,
        )

        path_clear = True

        if self.config_expert.visualize_internal_data:
            for vehicle in self.vehicles_inside_bev:
                # Sort out ego vehicle
                if vehicle.id == self.ego_vehicle.id:
                    continue

                vehicle_location = vehicle.get_location()
                vehicle_waypoint = self.carla_world_map.get_waypoint(vehicle_location)

                diff_vector = vehicle_location - self.ego_location
                dot_product = (
                    self.ego_vehicle.get_transform()
                    .get_forward_vector()
                    .dot(diff_vector)
                )
                # Draw dot_product above vehicle
                self.carla_world.debug.draw_string(
                    vehicle_location + carla.Location(x=1, y=0, z=2.5),
                    f"dot1={dot_product:.1f}",
                    color=carla.Color(255, 255, 0),
                    life_time=self.config_expert.draw_life_time,
                )

                # The overtaking path is blocked by vehicle
                diff_vector_2 = to_location - vehicle_location
                dot_product_2 = (
                    vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
                )
                self.carla_world.debug.draw_string(
                    vehicle_location + carla.Location(x=2, y=0, z=3.5),
                    f"dot2={dot_product_2:.1f}",
                    color=carla.Color(0, 255, 255),
                    life_time=self.config_expert.draw_life_time,
                )

        for vehicle in self.vehicles_inside_bev:
            # Only consider visible vehicles in the BEV to make TransFuser learn easier
            if not self.is_actor_inside_bev(vehicle):
                continue

            # Sort out ego vehicle
            if vehicle.id == self.ego_vehicle.id:
                continue

            vehicle_location = vehicle.get_location()
            vehicle_waypoint = self.carla_world_map.get_waypoint(vehicle_location)

            # Check if the vehicle is on the previous lane IDs
            if (
                vehicle_waypoint.road_id,
                vehicle_waypoint.lane_id,
            ) in previous_lane_ids:
                diff_vector = vehicle_location - self.ego_location
                dot_product = (
                    self.ego_vehicle.get_transform()
                    .get_forward_vector()
                    .dot(diff_vector)
                )

                # One TwoWay scenarios we can skip this vehicle since it's not on the overtaking path and behind
                # the ego vehicle. Otherwise in other scenarios it's coming from behind and is relevant
                if dot_product < 0 and self.current_active_scenario_type in [
                    "ConstructionObstacleTwoWays",
                    "AccidentTwoWays",
                    "ParkedObstacleTwoWays",
                    "VehicleOpensDoorTwoWays",
                    "HazardAtSideLaneTwoWays",
                ]:
                    continue

                # Allow earlier acceleration.
                # We only want predictable earlier acceleration from scratch that is why we have the ego_speed < 1.0 constraint.
                # Ignore vehicle that are close to the ego vehicle and is already almost out of the way
                if self.ego_speed < 1.0 and vehicle.get_velocity().length() > 3.0:
                    if self.current_active_scenario_type in [
                        "ConstructionObstacleTwoWays",
                    ]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["AccidentTwoWays"]:
                        threshold = 7
                        if self.ego_lane_width <= 2.76:
                            threshold = 7
                        elif self.ego_lane_width <= 3.01:
                            threshold = 8
                        elif self.ego_lane_width <= 3.51:
                            threshold = 9
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["ParkedObstacleTwoWays"]:
                        threshold = 7
                        if self.ego_lane_width <= 2.76:
                            threshold = 7
                        elif self.ego_lane_width <= 3.01:
                            threshold = 8
                        elif self.ego_lane_width <= 3.51:
                            threshold = 9
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in [
                        "VehicleOpensDoorTwoWays",
                    ]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in [
                        "HazardAtSideLaneTwoWays",
                    ]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue

                # The overtaking path is blocked by vehicle
                diff_vector_2 = to_location - vehicle_location
                dot_product_2 = (
                    vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
                )

                if dot_product_2 < 0:
                    path_clear = False
                    break

                other_vehicle_distance = (
                    to_location.distance(vehicle_location)
                    - vehicle.bounding_box.extent.x
                )
                other_vehicle_time = other_vehicle_distance / max(
                    1.0,
                    vehicle.get_velocity().length(),
                )

                # Add 200 ms safety margin
                # Vehicle needs less time to arrive at to_location than the ego vehicle
                if (
                    other_vehicle_time
                    < ego_time + self.config_expert.check_path_free_safety_time
                ):
                    path_clear = False
                    break

        return path_clear

    def _solve_general_scenarios(
        self,
        initial_target_speed: float,
        speed_reduced_by_obj: list | None,
    ) -> tuple[bool, float, list | None]:
        """
        Compute the brake command and target speed for the ego vehicle based on various factors.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            speed_reduced_by_obj: A list containing [reduced speed, object type, object ID, distance]
                    for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            A tuple containing the brake command, target speed, and the updated speed_reduced_by_obj list.
        """
        target_speed = initial_target_speed

        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = self.ego_transform.transform(
            self.ego_vehicle.bounding_box.location,
        )
        ego_bb_global = carla.BoundingBox(
            center_ego_bb_global,
            self.ego_vehicle.bounding_box.extent,
        )
        ego_bb_global.rotation = self.ego_transform.rotation

        if self.config_expert.visualize_bounding_boxes:
            self.carla_world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=self.config_expert.ego_vehicle_bb_color,
                life_time=self.config_expert.draw_life_time,
            )

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.config_expert.bicycle_frame_rate
            * (
                self.config_expert.forecast_length_lane_change
                if self.near_lane_change
                else self.config_expert.default_forecast_length
            ),
        )

        # Get future bounding boxes of pedestrians
        nearby_pedestrians, nearby_pedestrian_ids = self.forecast_walkers(
            num_future_frames,
        )

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bounding_boxes = self.forecast_ego_agent(
            num_future_frames,
            initial_target_speed,
        )

        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes = self.predict_other_actors_bounding_boxes(
            num_future_frames,
        )

        # Compute the target speed with respect to the leading vehicle
        target_speed_leading, speed_reduced_by_obj = (
            self.compute_target_speed_wrt_leading_vehicle(
                initial_target_speed,
                predicted_bounding_boxes,
                speed_reduced_by_obj,
            )
        )

        # Compute the target speeds with respect to all actors (vehicles, bicycles, pedestrians)
        (
            target_speed_bicycle,
            target_speed_pedestrian,
            target_speed_vehicle,
            speed_reduced_by_obj,
        ) = self.compute_target_speeds_wrt_all_actors(
            initial_target_speed,
            ego_bounding_boxes,
            predicted_bounding_boxes,
            speed_reduced_by_obj,
            nearby_pedestrians,
            nearby_pedestrian_ids,
        )

        # Compute the target speed with respect to the red light
        target_speed_red_light = self.ego_agent_affected_by_red_light(
            initial_target_speed,
        )

        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_red_light
        ):
            speed_reduced_by_obj = [
                target_speed_red_light,
                None
                if self.next_traffic_light is None
                else self.next_traffic_light.type_id,
                None if self.next_traffic_light is None else self.next_traffic_light.id,
                self.distance_to_next_traffic_light,
            ]

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self.ego_agent_affected_by_stop_sign(
            initial_target_speed,
        )
        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_stop_sign
        ):
            speed_reduced_by_obj = [
                target_speed_stop_sign,
                None if self.next_stop_sign is None else self.next_stop_sign.type_id,
                None if self.next_stop_sign is None else self.next_stop_sign.id,
                self.distance_to_next_stop_sign,
            ]

        # Compute the minimum target speed considering all factors
        target_speed = min(
            target_speed_leading,
            target_speed_bicycle,
            target_speed_vehicle,
            target_speed_pedestrian,
            target_speed_red_light,
            target_speed_stop_sign,
        )

        # Set the hazard flags based on the target speed and its cause
        if (
            target_speed == target_speed_pedestrian
            and target_speed_pedestrian != initial_target_speed
        ):
            self.walker_hazard = True
            self.walker_close = True
        elif (
            target_speed == target_speed_red_light
            and target_speed_red_light != initial_target_speed
        ):
            self.traffic_light_hazard = True
        elif (
            target_speed == target_speed_stop_sign
            and target_speed_stop_sign != initial_target_speed
        ):
            self.stop_sign_hazard = True
            self.stop_sign_close = True

        # Determine if the ego vehicle needs to brake based on the target speed
        brake = target_speed == 0
        return brake, target_speed, speed_reduced_by_obj

    def _solve_obstacle_scenarios(
        self,
        target_speed: float,
    ) -> tuple[float, bool, list]:
        """
        This method handles various obstacle and scenario situations that may arise during navigation.
        It adjusts the target speed, modifies the route, and determines if the ego vehicle should keep driving or wait.
        The method supports different scenario types such as InvadingTurn, Accident, ConstructionObstacle,
        ParkedObstacle, AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays, VehicleOpensDoorTwoWays,
        HazardAtSideLaneTwoWays, HazardAtSideLane, and YieldToEmergencyVehicle.

        Args:
            target_speed: The current target speed of the ego vehicle.

        Returns:
            A tuple containing the updated target speed, a boolean indicating whether to keep driving,
                and a list containing information about a potential decreased target speed due to an object.
        """

        keep_driving = False
        speed_reduced_by_obj = [
            target_speed,
            None,
            None,
            None,
        ]  # [target_speed, type, id, distance]

        # Only continue if there are some active scenarios available
        if len(CarlaDataProvider.active_scenarios) != 0:
            ego_location = self.ego_vehicle.get_location()

            # # Sort the scenarios by distance if there is more than one active scenario
            # if len(CarlaDataProvider.active_scenarios) != 1:
            #     sort_scenarios_by_distance(ego_location)

            if self.current_active_scenario_type == "InvadingTurn":
                # Get the first (current) InvadingTurn memory entry
                invading_turn_memory = (
                    CarlaDataProvider.get_current_scenario_memory()
                    if CarlaDataProvider.get_current_scenario_memory()
                    else {}
                )
                first_cone, last_cone, offset = [
                    invading_turn_memory[k]
                    for k in ["first_cone", "last_cone", "offset"]
                ]
                closest_distance = first_cone.get_location().distance(ego_location)
                if (
                    closest_distance
                    < self.config_expert.default_max_distance_to_process_scenario
                ):
                    LOG.info(
                        "[Scenario Action] InvadingTurn: Shifting route for invading turn maneuver",
                    )
                    self.privileged_route_planner.shift_route_for_invading_turn(
                        first_cone,
                        last_cone,
                        offset,
                    )
                    LOG.info(
                        "[Scenario Exit] InvadingTurn: Completed, cleaning up scenario",
                    )
                    CarlaDataProvider.clean_current_active_scenario()

            elif self.current_active_scenario_type in [
                "Accident",
                "ConstructionObstacle",
                "ParkedObstacle",
            ]:
                first_actor, last_actor, direction = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in ["first_actor", "last_actor", "direction"]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(
                    self.ego_vehicle,
                    first_actor,
                )

                LOG.info(
                    f"[{self.current_active_scenario_type}] Horizontal distance to obstacle: {horizontal_distance:.1f}m",
                )

                # Shift the route around the obstacles
                if (
                    horizontal_distance
                    < self.config_expert.default_max_distance_to_process_scenario
                    and not CarlaDataProvider.get_current_scenario_memory()[
                        "changed_route"
                    ]
                ):
                    add_before_length = {
                        "Accident": self.config_expert.add_before_accident,
                        "ConstructionObstacle": self.config_expert.add_before_construction_obstacle,
                        "ParkedObstacle": self.config_expert.add_before_parked_obstacle,
                    }[self.current_active_scenario_type]
                    add_after_length = {
                        "Accident": self.config_expert.add_after_accident,
                        "ConstructionObstacle": self.config_expert.add_after_construction_obstacle,
                        "ParkedObstacle": self.config_expert.add_after_parked_obstacle,
                    }[self.current_active_scenario_type]
                    transition_length = {
                        "Accident": self.config_expert.transition_smoothness_distance_accident,
                        "ConstructionObstacle": self.config_expert.transition_smoothness_factor_construction_obstacle,
                        "ParkedObstacle": self.config_expert.transition_smoothness_distance_parked_obstacle,
                    }[self.current_active_scenario_type]
                    if self.current_active_scenario_type not in [
                        "ConstructionObstacle",
                    ]:
                        if self.visual_visibility == WeatherVisibility.LIMITED:
                            add_before_length -= 5.0
                            transition_length -= int(
                                4.0 * self.config_expert.points_per_meter,
                            )
                        elif self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                            add_before_length -= 7.0
                            transition_length -= int(
                                6.0 * self.config_expert.points_per_meter,
                            )
                        LOG.info("Later switching because of bad weather")
                    if self.speed_limit < 15:
                        add_after_length = {
                            "Accident": 0.5,
                            "ConstructionObstacle": self.config_expert.add_after_construction_obstacle,
                            "ParkedObstacle": 0.5,
                        }[
                            self.current_active_scenario_type
                        ]  # Speed is low and we can switch earlier
                    lane_transition_factor = {
                        "Accident": 1.0,
                        "ConstructionObstacle": self.two_way_obstacle_distance_to_cones_factor,
                        "ParkedObstacle": 1.0,
                    }[self.current_active_scenario_type]
                    LOG.info(
                        f"[Route Change] {self.current_active_scenario_type}: Shifting route around obstacle, direction={direction}, distance={horizontal_distance:.1f}m",
                    )
                    from_index, _ = (
                        self.privileged_route_planner.shift_route_around_actors(
                            first_actor,
                            last_actor,
                            direction,
                            transition_length,
                            lane_transition_factor=lane_transition_factor,
                            extra_length_before=add_before_length,
                            extra_length_after=add_after_length,
                        )
                    )
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_route": True,
                            "from_index": from_index,
                        },
                    )

                elif CarlaDataProvider.get_current_scenario_memory()["changed_route"]:
                    first_actor_rel_pos = common_utils.get_relative_transform(
                        self.ego_matrix,
                        np.array(first_actor.get_transform().get_matrix()),
                    )
                    if first_actor_rel_pos[0] < 3:
                        LOG.info(
                            f"[Scenario Exit] {self.current_active_scenario_type}: Passed obstacle, cleaning up scenario",
                        )
                        CarlaDataProvider.clean_current_active_scenario()
            elif self.current_active_scenario_type in [
                "AccidentTwoWays",
                "ConstructionObstacleTwoWays",
                "ParkedObstacleTwoWays",
                "VehicleOpensDoorTwoWays",
            ]:
                (
                    first_actor,
                    last_actor,
                    direction,
                    changed_route,
                    from_index,
                    to_index,
                    path_clear,
                ) = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "first_actor",
                        "last_actor",
                        "direction",
                        "changed_route",
                        "from_index",
                        "to_index",
                        "path_clear",
                    ]
                ]

                # change the route if the ego is close enough to the obstacle
                horizontal_distance = expert_utils.get_horizontal_distance(
                    self.ego_vehicle,
                    first_actor,
                )

                # Shift the route around the obstacles
                if (
                    horizontal_distance
                    < self.config_expert.default_max_distance_to_process_scenario
                    and not changed_route
                ):
                    transition_length = {
                        "AccidentTwoWays": self.config_expert.transition_length_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config_expert.transition_length_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.transition_length_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.transition_length_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    add_before_length = {
                        "AccidentTwoWays": self.add_before_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.add_before_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.add_before_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.add_before_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    add_after_length = {
                        "AccidentTwoWays": self.add_after_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.add_after_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.add_after_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.add_after_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    factor = {
                        "AccidentTwoWays": self.config_expert.factor_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.two_way_obstacle_distance_to_cones_factor,
                        "ParkedObstacleTwoWays": self.config_expert.factor_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.two_way_vehicle_open_door_distance_to_center_line_factor,
                    }[self.current_active_scenario_type]
                    if self.current_active_scenario_type not in [
                        "ConstructionObstacleTwoWays",
                    ]:
                        if self.visual_visibility == WeatherVisibility.LIMITED:
                            add_before_length -= 0.5
                        elif self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                            add_before_length -= 1.0

                    LOG.info(
                        f"[Route Change] {self.current_active_scenario_type}: Initiating two-way overtaking maneuver, direction={direction}, distance={horizontal_distance:.1f}m",
                    )
                    from_index, to_index = (
                        self.privileged_route_planner.shift_route_around_actors(
                            first_actor,
                            last_actor,
                            direction,
                            transition_length,
                            factor,
                            add_before_length,
                            add_after_length,
                        )
                    )

                    changed_route = True
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_route": changed_route,
                            "from_index": from_index,
                            "to_index": to_index,
                            "path_clear": path_clear,
                        },
                    )

                # Check if the ego can overtake the obstacle
                if (
                    changed_route
                    and from_index - self.privileged_route_planner.route_index
                    < self.config_expert.max_distance_to_overtake_two_way_scnearios
                    and not path_clear
                ):
                    # Get previous roads and lanes of the target lane
                    target_lane = (
                        self.route_waypoints[0].get_left_lane()
                        if direction == "right"
                        else self.route_waypoints[0].get_right_lane()
                    )
                    if target_lane is None:
                        return target_speed, keep_driving, speed_reduced_by_obj
                    prev_road_lane_ids = expert_utils.get_previous_road_lane_ids(
                        self.config_expert,
                        target_lane,
                    )
                    path_clear = self.is_two_ways_overtaking_path_clear(
                        int(from_index),
                        int(to_index),
                        target_speed,
                        prev_road_lane_ids,
                        min_speed=self.two_way_overtake_speed,
                    )
                    CarlaDataProvider.get_current_scenario_memory()["path_clear"] = (
                        path_clear
                    )

                # If the overtaking path is clear, keep driving; otherwise, wait behind the obstacle
                if path_clear:
                    target_speed = self.two_way_overtake_speed
                    if (
                        self.privileged_route_planner.route_index
                        >= to_index
                        - self.config_expert.distance_to_delete_scenario_in_two_ways
                    ):
                        LOG.info(
                            f"[Scenario Exit] {self.current_active_scenario_type}: Overtaking completed, cleaning up scenario",
                        )
                        CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = True
                else:
                    offset = {
                        "AccidentTwoWays": 10,
                        "ConstructionObstacleTwoWays": 10,
                        "ParkedObstacleTwoWays": 10,
                        "VehicleOpensDoorTwoWays": 10,
                    }[
                        self.current_active_scenario_type
                    ]  # Move a bit to side instead of standing direct behind the obstacle
                    distance_to_merging_point = (
                        float(
                            from_index
                            + offset
                            - self.privileged_route_planner.route_index,
                        )
                        / self.config_expert.points_per_meter
                    )
                    target_speed = expert_utils.compute_target_speed_idm(
                        config=self.config_expert,
                        desired_speed=target_speed,
                        leading_actor_length=self.ego_vehicle.bounding_box.extent.x,
                        ego_speed=self.ego_speed,
                        leading_actor_speed=0,
                        distance_to_leading_actor=distance_to_merging_point,
                        s0=self.config_expert.idm_two_way_scenarios_minimum_distance,
                        T=self.config_expert.idm_two_way_scenarios_time_headway,
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed
                    ):
                        speed_reduced_by_obj = [
                            target_speed,
                            first_actor.type_id,
                            first_actor.id,
                            distance_to_merging_point,
                        ]

            elif self.current_active_scenario_type == "HazardAtSideLaneTwoWays":
                (
                    first_actor,
                    last_actor,
                    changed_route,
                    from_index,
                    to_index,
                    path_clear,
                ) = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "first_actor",
                        "last_actor",
                        "changed_route",
                        "from_index",
                        "to_index",
                        "path_clear",
                    ]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(
                    self.ego_vehicle,
                    first_actor,
                )

                if (
                    horizontal_distance
                    < self.config_expert.max_distance_to_process_hazard_at_side_lane_two_ways
                    and not changed_route
                ):
                    to_index = self.privileged_route_planner.get_closest_route_index(
                        int(self.privileged_route_planner.route_index),
                        last_actor.get_location(),
                    )

                    # Assume the bicycles don't drive too much during the overtaking process
                    to_index += 170
                    from_index = int(self.privileged_route_planner.route_index)

                    starting_wp = self.route_waypoints[0].get_left_lane()
                    prev_road_lane_ids = expert_utils.get_previous_road_lane_ids(
                        self.config_expert,
                        starting_wp,
                    )
                    path_clear = self.is_two_ways_overtaking_path_clear(
                        int(from_index),
                        int(to_index),
                        target_speed,
                        prev_road_lane_ids,
                        min_speed=self.config_expert.default_overtake_speed,
                    )

                    if path_clear:
                        transition_length = (
                            self.config_expert.transition_smoothness_distance
                        )
                        self.privileged_route_planner.shift_route_smoothly(
                            int(from_index),
                            int(to_index),
                            True,
                            transition_length,
                        )
                        changed_route = True
                        CarlaDataProvider.get_current_scenario_memory().update(
                            {
                                "changed_route": changed_route,
                                "from_index": int(from_index),
                                "to_index": int(to_index),
                                "path_clear": path_clear,
                            },
                        )

                # the overtaking path is clear
                if path_clear:
                    # Check if the overtaking is done
                    if self.privileged_route_planner.route_index >= to_index:
                        CarlaDataProvider.clean_current_active_scenario()
                    # Overtake with max. 50 km/h
                    target_speed, keep_driving = (
                        self.config_expert.default_overtake_speed,
                        True,
                    )

            elif self.current_active_scenario_type == "HazardAtSideLane":
                (
                    first_actor,
                    last_actor,
                    changed_first_part_of_route,
                    from_index,
                    to_index,
                ) = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "first_actor",
                        "last_actor",
                        "changed_first_part_of_route",
                        "from_index",
                        "to_index",
                    ]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(
                    self.ego_vehicle,
                    last_actor,
                )

                if (
                    horizontal_distance
                    < self.config_expert.max_distance_to_process_hazard_at_side_lane
                    and not changed_first_part_of_route
                ):
                    transition_length = (
                        self.config_expert.transition_smoothness_distance
                    )
                    LOG.info(
                        f"[Route Change] HazardAtSideLane: Shifting route right to avoid hazard, distance={horizontal_distance:.1f}m",
                    )
                    from_index, to_index = (
                        self.privileged_route_planner.shift_route_around_actors(
                            first_actor,
                            last_actor,
                            "right",
                            transition_length,
                        )
                    )

                    to_index -= transition_length
                    changed_first_part_of_route = True
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_first_part_of_route": changed_first_part_of_route,
                            "from_index": from_index,
                            "to_index": to_index,
                        },
                    )

                if changed_first_part_of_route:
                    to_idx_ = self.privileged_route_planner.extend_lane_shift_transition_for_hazard_at_side_lane(
                        last_actor,
                        to_index,
                    )
                    to_index = to_idx_
                    CarlaDataProvider.get_current_scenario_memory()["to_index"] = (
                        to_index
                    )
                if self.privileged_route_planner.route_index > to_index:
                    LOG.info(
                        "[Scenario Exit] HazardAtSideLane: Passed hazard, cleaning up scenario",
                    )
                    CarlaDataProvider.clean_current_active_scenario()

            elif self.current_active_scenario_type == "YieldToEmergencyVehicle":
                emergency_veh, changed_route, from_index, to_index, to_left = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "emergency_vehicle",
                        "changed_route",
                        "from_index",
                        "to_index",
                        "to_left",
                    ]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(
                    self.ego_vehicle,
                    emergency_veh,
                )

                if (
                    horizontal_distance
                    < self.config_expert.default_max_distance_to_process_scenario
                    and not changed_route
                ):
                    # Assume the emergency vehicle doesn't drive more than 20 m during the overtaking process
                    from_index = (
                        self.privileged_route_planner.route_index
                        + 30 * self.config_expert.points_per_meter
                    )
                    to_index = (
                        from_index
                        + int(2 * self.config_expert.points_per_meter)
                        * self.config_expert.points_per_meter
                    )

                    transition_length = (
                        self.config_expert.transition_smoothness_distance
                    )
                    to_left = (
                        self.route_waypoints[from_index].lane_change
                        != carla.LaneChange.Right
                    )
                    LOG.info(
                        f"[Route Change] YieldToEmergencyVehicle: Shifting route {'left' if to_left else 'right'} to yield to emergency vehicle",
                    )
                    self.privileged_route_planner.shift_route_smoothly(
                        int(from_index),
                        int(to_index),
                        to_left,
                        transition_length,
                    )

                    changed_route = True
                    to_index -= transition_length
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_route": changed_route,
                            "from_index": int(from_index),
                            "to_index": int(to_index),
                            "to_left": to_left,
                        },
                    )

                if changed_route:
                    to_idx_ = self.privileged_route_planner.extend_lane_shift_transition_for_yield_to_emergency_vehicle(
                        to_left,
                        to_index,
                    )
                    to_index = to_idx_
                    CarlaDataProvider.get_current_scenario_memory()["to_index"] = (
                        to_index
                    )

                    # Check if the emergency vehicle is in front of the ego vehicle
                    diff = emergency_veh.get_location() - ego_location
                    dot_res = (
                        self.ego_vehicle.get_transform().get_forward_vector().dot(diff)
                    )
                    if dot_res > 0:
                        LOG.info(
                            "[Scenario Exit] YieldToEmergencyVehicle: Emergency vehicle passed, cleaning up scenario",
                        )
                        CarlaDataProvider.clean_current_active_scenario()
                    else:
                        LOG.info(
                            f"[Control] YieldToEmergencyVehicle: Setting overtake speed to {self.config_expert.default_overtake_speed * 3.6:.1f} km/h",
                        )
                        target_speed, keep_driving = (
                            self.config_expert.default_overtake_speed,
                            True,
                        )

        # Negotiate gap
        self.construction_obstacle_two_ways_stuck = False
        self.accident_two_ways_stuck = False
        self.parked_obstacle_two_ways_stuck = False
        self.vehicle_opens_door_two_ways_stuck = False
        if self.current_active_scenario_type in ["ConstructionObstacleTwoWays"]:
            speed_constant = 2
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    common_utils.normalize_angle_degree(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.construction_obstacle_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["AccidentTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    common_utils.normalize_angle_degree(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.accident_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["ParkedObstacleTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    common_utils.normalize_angle_degree(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.parked_obstacle_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["VehicleOpensDoorTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    common_utils.normalize_angle_degree(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.vehicle_opens_door_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        return target_speed, keep_driving, speed_reduced_by_obj

    @beartype
    def predict_other_actors_bounding_boxes(self, num_future_frames: int) -> dict:
        """
        Predict the future bounding boxes of actors for a given number of frames.

        Args:
            num_future_frames: The number of future frames to predict.

        Returns:
            A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
        """
        predicted_bounding_boxes = {}

        # --- Determine how wider we should make dangerous adversarial actors' bounding boxes
        if (
            self.current_active_scenario_type
            in [
                "Accident",
                "ConstructionObstacle",
                "ParkedObstacle",
                "HazardAtSideLane",
            ]
        ):  # Adversarials drive in same direction, we make their bounding boxes not too wide
            adversarial_bb_extra_width = (
                max(
                    (self.ego_lane_width - self.ego_vehicle.bounding_box.extent.y) / 2,
                    0,
                )
                * 0.6
            )
        elif (
            self.current_active_scenario_type
            in [
                "SignalizedJunctionLeftTurn",
                "NonSignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
                "InterurbanActorFlow",
            ]
        ):  # Adversarials drive in opposite direction, we make their bounding boxes wider
            adversarial_bb_extra_width = (
                max(
                    (self.ego_lane_width - self.ego_vehicle.bounding_box.extent.y) / 2,
                    0,
                )
                * 0.8
            )
        elif (
            self.current_active_scenario_type
            in [
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
            ]
        ):  # Adversarials drive in same direction, we make their bounding boxes even wider
            adversarial_bb_extra_width = (
                max(
                    (self.ego_lane_width - self.ego_vehicle.bounding_box.extent.y) / 2,
                    0,
                )
                * 1.75
            )  # Right turn larger BB to avoid collisions bc of blinding fleck

        minimal_adversarial_bb_width = None
        if self.target_lane_width is not None:
            minimal_adversarial_bb_width = (
                adversarial_bb_extra_width + self.target_lane_width
            )

        # --- Determine which adversarial actors to consider, ignore or make dangerous
        (
            dangerous_adversarial_actors_ids,
            safe_adversarial_actors_ids,
            ignored_adversarial_actors_ids,
        ) = self.adversarial_actors_ids

        # Filter out nearby actors within the detection radius, excluding the ego vehicle
        nearby_actors = [
            actor
            for actor in self.vehicles_inside_bev
            if actor.id != self.ego_vehicle.id
        ]

        # If there are nearby actors, calculate their future bounding boxes
        if nearby_actors:
            # Get the previous control inputs (steering, throttle, brake) for the nearby actors
            previous_controls = [actor.get_control() for actor in nearby_actors]
            previous_actions = np.array(
                [
                    [control.steer, control.throttle, control.brake]
                    for control in previous_controls
                ],
            )

            # Get the current velocities, locations, and headings of the nearby actors
            velocities = []
            for actor in nearby_actors:
                actor_original_velocity = actor.get_velocity().length()
                velocities.append(actor_original_velocity)

            velocities = np.array(velocities)
            locations = np.array(
                [
                    [
                        actor.get_location().x,
                        actor.get_location().y,
                        actor.get_location().z,
                    ]
                    for actor in nearby_actors
                ],
            )
            headings = np.deg2rad(
                np.array(
                    [actor.get_transform().rotation.yaw for actor in nearby_actors],
                ),
            )

            # Initialize arrays to store future locations, headings, and velocities
            future_locations = np.empty(
                (num_future_frames, len(nearby_actors), 3),
                dtype="float",
            )
            future_headings = np.empty(
                (num_future_frames, len(nearby_actors)),
                dtype="float",
            )
            future_velocities = np.empty(
                (num_future_frames, len(nearby_actors)),
                dtype="float",
            )

            # Forecast the future locations, headings, and velocities for the nearby actors
            for i in range(num_future_frames):
                locations, headings, velocities = (
                    self.vehicle_model.forecast_other_vehicles(
                        locations,
                        headings,
                        velocities,
                        previous_actions,
                    )
                )
                future_locations[i] = locations.copy()
                future_velocities[i] = velocities.copy()
                future_headings[i] = headings.copy()
            # Convert future headings to degrees
            future_headings = np.rad2deg(future_headings)

            # Calculate the predicted bounding boxes for each nearby actor and future frame
            for actor_idx, actor in enumerate(nearby_actors):
                predicted_actor_boxes = []

                for i in range(num_future_frames):
                    # Calculate the future location of the actor
                    location = carla.Location(
                        x=future_locations[i, actor_idx, 0].item(),
                        y=future_locations[i, actor_idx, 1].item(),
                        z=future_locations[i, actor_idx, 2].item(),
                    )

                    # Calculate the future rotation of the actor
                    rotation = carla.Rotation(
                        pitch=0,
                        yaw=future_headings[i, actor_idx],
                        roll=0,
                    )

                    # Get the extent (dimensions) of the actor's bounding box
                    extent = actor.bounding_box.extent
                    # Otherwise we would increase the extent of the bounding box of the vehicle
                    extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                    # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                    # uncertainty during forecasting
                    s = (
                        self.config_expert.high_speed_min_extent_x_other_vehicle_lane_change
                        if self.near_lane_change
                        else self.config_expert.high_speed_min_extent_x_other_vehicle
                    )
                    length_factor = (
                        self.config_expert.slow_speed_extent_factor_ego
                        if future_velocities[i, actor_idx]
                        < self.config_expert.extent_other_vehicles_bbs_speed_threshold
                        else max(
                            s,
                            self.config_expert.high_speed_min_extent_x_other_vehicle
                            * float(i)
                            / float(num_future_frames),
                        )
                    )
                    width_factor = (
                        self.config_expert.slow_speed_extent_factor_ego
                        if future_velocities[i, actor_idx]
                        < self.config_expert.extent_other_vehicles_bbs_speed_threshold
                        else max(
                            self.config_expert.high_speed_min_extent_y_other_vehicle,
                            self.config_expert.high_speed_extent_y_factor_other_vehicle
                            * float(i)
                            / float(num_future_frames),
                        )
                    )

                    if self.current_active_scenario_type in ["CrossingBicycleFlow"]:
                        if actor.type_id in constants.BIKER_MESHES:
                            length_factor = 4.0
                            width_factor = 10.0
                    elif self.current_active_scenario_type in [
                        "NonSignalizedJunctionRightTurn",
                    ]:
                        if actor.id in dangerous_adversarial_actors_ids:
                            length_factor = 1.5
                            width_factor = minimal_adversarial_bb_width / (extent.y * 2)
                        elif actor.id in safe_adversarial_actors_ids:
                            length_factor = 1.0
                            width_factor = 1
                        elif actor.id in ignored_adversarial_actors_ids:
                            length_factor = 0
                            width_factor = 0
                    elif self.current_active_scenario_type in [
                        "SignalizedJunctionRightTurn",
                    ]:
                        if actor.id in dangerous_adversarial_actors_ids:
                            length_factor = 1.6
                            width_factor = minimal_adversarial_bb_width / (extent.y * 2)
                        elif actor.id in safe_adversarial_actors_ids:
                            length_factor = 1.0
                            width_factor = 1
                        elif actor.id in ignored_adversarial_actors_ids:
                            length_factor = 0
                    elif self.current_active_scenario_type in [
                        "SignalizedJunctionLeftTurnEnterFlow",
                        "NonSignalizedJunctionLeftTurnEnterFlow",
                    ]:
                        if actor.id in dangerous_adversarial_actors_ids:
                            length_factor = 1.25
                            width_factor = minimal_adversarial_bb_width / (extent.y * 2)
                        elif actor.id in safe_adversarial_actors_ids:
                            length_factor = 0.75
                            width_factor = 1
                        elif actor.id in ignored_adversarial_actors_ids:
                            length_factor = 0
                            width_factor = 0

                    extent.x *= length_factor
                    extent.y *= width_factor
                    # Create the bounding box for the future frame
                    bounding_box = carla.BoundingBox(location, extent)
                    bounding_box.rotation = rotation

                    # Append the bounding box to the list of predicted bounding boxes for this actor
                    predicted_actor_boxes.append(bounding_box)

                # Store the predicted bounding boxes for this actor in the dictionary
                predicted_bounding_boxes[actor.id] = predicted_actor_boxes

        self.visualize_forecasted_bounding_boxes(predicted_bounding_boxes)

        return predicted_bounding_boxes

    @beartype
    def compute_target_speed_wrt_leading_vehicle(
        self,
        initial_target_speed: float,
        predicted_bounding_boxes: dict,
        speed_reduced_by_obj: list | None,
    ) -> tuple[float, list | None]:
        """
        Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            predicted_bounding_boxes: A dictionary mapping actor IDs to lists of predicted bounding boxes.
            speed_reduced_by_obj: A list containing [reduced speed, object type, object ID, distance]
                for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            A tuple containing the target speed considering the leading vehicle and the updated speed_reduced_by_obj.
        """
        target_speed_wrt_leading_vehicle = initial_target_speed
        # _, _, ignored_adversarial_actors_ids = self.adversarial_actors_ids

        for vehicle_id, _ in predicted_bounding_boxes.items():
            # if vehicle_id in ignored_adversarial_actors_ids:
            #     continue
            if vehicle_id in self.leading_vehicle_ids and not self.near_lane_change:
                # Vehicle is in front of the ego vehicle
                ego_speed = self.ego_vehicle.get_velocity().length()
                vehicle = self.carla_world.get_actor(vehicle_id)
                other_speed = vehicle.get_velocity().length()
                distance_to_vehicle = self.ego_location.distance(vehicle.get_location())

                # Compute the target speed using the IDM
                target_speed_wrt_leading_vehicle = min(
                    target_speed_wrt_leading_vehicle,
                    expert_utils.compute_target_speed_idm(
                        config=self.config_expert,
                        desired_speed=initial_target_speed,
                        leading_actor_length=vehicle.bounding_box.extent.x * 2,
                        ego_speed=ego_speed,
                        leading_actor_speed=other_speed,
                        distance_to_leading_actor=distance_to_vehicle,
                        s0=self.config_expert.idm_leading_vehicle_minimum_distance,
                        T=self.config_expert.idm_leading_vehicle_time_headway,
                    ),
                )

                # Update the object causing the most speed reduction
                if (
                    speed_reduced_by_obj is None
                    or speed_reduced_by_obj[0] > target_speed_wrt_leading_vehicle
                ):
                    speed_reduced_by_obj = [
                        target_speed_wrt_leading_vehicle,
                        vehicle.type_id,
                        vehicle.id,
                        distance_to_vehicle,
                    ]

            if self.config_expert.visualize_bounding_boxes:
                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if (
                        vehicle_id in self.leading_vehicle_ids
                        and not self.near_lane_change
                    ):
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.leading_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )
                    elif vehicle_id in self.trailing_vehicle_ids:
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.trailing_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    @beartype
    def compute_target_speeds_wrt_all_actors(
        self,
        initial_target_speed: float,
        ego_bounding_boxes: list,
        predicted_bounding_boxes: dict,
        speed_reduced_by_obj: list | None,
        nearby_walkers: list,
        nearby_walkers_ids: list,
    ) -> tuple[float, float, float, list | None]:
        """
        Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles,
        and pedestrians) by checking for intersecting bounding boxes.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            ego_bounding_boxes: A list of bounding boxes for the ego vehicle at different future frames.
            predicted_bounding_boxes: A dictionary mapping actor IDs to lists of predicted bounding boxes.
            speed_reduced_by_obj: A list containing [reduced speed, object type,
                object ID, distance] for the object that caused the most speed reduction, or None if
                no speed reduction.
            nearby_walkers: A list of predicted bounding boxes of nearby pedestrians.
            nearby_walkers_ids: A list of IDs for nearby pedestrians.

        Returns:
            A tuple containing the target speeds for bicycles, pedestrians, vehicles, and the updated
                speed_reduced_by_obj list.
        """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        ego_vehicle_location = self.ego_vehicle.get_location()
        hazard_color = self.config_expert.ego_vehicle_forecasted_bbs_hazard_color
        normal_color = self.config_expert.ego_vehicle_forecasted_bbs_normal_color
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, bounding_boxes in predicted_bounding_boxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in self.leading_vehicle_ids and not self.near_lane_change:
                    continue
                elif (
                    vehicle_id in self.trailing_vehicle_ids
                    and not self.near_lane_change
                ):
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = expert_utils.check_obb_intersection(
                        ego_bounding_box,
                        bounding_boxes[i],
                    )

                    if intersects_with_ego:
                        blocking_actor = self.carla_world.get_actor(vehicle_id)
                        # Handle the case when the blocking actor is a bicycle
                        if (
                            "base_type" in blocking_actor.attributes
                            and blocking_actor.attributes["base_type"] == "bicycle"
                        ):
                            other_speed = blocking_actor.get_velocity().length()
                            distance_to_actor = ego_vehicle_location.distance(
                                blocking_actor.get_location(),
                            )

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                expert_utils.compute_target_speed_idm(
                                    config=self.config_expert,
                                    desired_speed=initial_target_speed,
                                    leading_actor_length=blocking_actor.bounding_box.extent.x
                                    * 2,
                                    ego_speed=self.ego_speed,
                                    leading_actor_speed=other_speed,
                                    distance_to_leading_actor=distance_to_actor,
                                    s0=self.config_expert.idm_bicycle_minimum_distance,
                                    T=self.config_expert.idm_bicycle_desired_time_headway,
                                ),
                            )

                            # Update the object causing the most speed reduction
                            if (
                                speed_reduced_by_obj is None
                                or speed_reduced_by_obj[0] > target_speed_bicycle
                            ):
                                speed_reduced_by_obj = [
                                    target_speed_bicycle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            self.vehicle_hazard = True  # Set the vehicle hazard flag
                            self.vehicle_affecting_id = vehicle_id  # Store the ID of the vehicle causing the hazard
                            color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                            target_speed_vehicle = (
                                0.0  # Set the target speed for vehicles to zero
                            )
                            distance_to_actor = blocking_actor.get_location().distance(
                                ego_vehicle_location,
                            )

                            # Update the object causing the most speed reduction
                            if (
                                speed_reduced_by_obj is None
                                or speed_reduced_by_obj[0] > target_speed_vehicle
                            ):
                                speed_reduced_by_obj = [
                                    target_speed_vehicle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for pedestrian_bb, pedestrian_id in zip(
                nearby_walkers,
                nearby_walkers_ids,
                strict=False,
            ):
                if expert_utils.check_obb_intersection(
                    ego_bounding_box,
                    pedestrian_bb[i],
                ):
                    color = hazard_color
                    blocking_actor = self.carla_world.get_actor(pedestrian_id)
                    distance_to_actor = ego_vehicle_location.distance(
                        blocking_actor.get_location(),
                    )

                    # Compute the target speed for pedestrians using the IDM
                    target_speed_pedestrian = min(
                        target_speed_pedestrian,
                        expert_utils.compute_target_speed_idm(
                            config=self.config_expert,
                            desired_speed=initial_target_speed,
                            leading_actor_length=0.5
                            + self.ego_vehicle.bounding_box.extent.x,
                            ego_speed=self.ego_speed,
                            leading_actor_speed=0.0,
                            distance_to_leading_actor=distance_to_actor,
                            s0=self.config_expert.idm_pedestrian_minimum_distance,
                            T=self.config_expert.idm_pedestrian_desired_time_headway,
                        ),
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_pedestrian
                    ):
                        speed_reduced_by_obj = [
                            target_speed_pedestrian,
                            blocking_actor.type_id,
                            blocking_actor.id,
                            distance_to_actor,
                        ]
            if self.config_expert.visualize_bounding_boxes:
                self.carla_world.debug.draw_box(
                    box=ego_bounding_box,
                    rotation=ego_bounding_box.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=self.config_expert.draw_life_time,
                )
        return (
            float(target_speed_bicycle),
            float(target_speed_pedestrian),
            float(target_speed_vehicle),
            speed_reduced_by_obj,
        )

    @beartype
    def forecast_ego_agent(
        self,
        num_future_frames: int,
        initial_target_speed: float,
    ) -> list:
        """
        Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
        check subsequently whether the ego vehicle would collide.

        Args:
            num_future_frames: The number of future frames to forecast.
            initial_target_speed: The initial target speed for the ego vehicle.

        Returns:
            A list of bounding boxes representing the future states of the ego vehicle.
        """
        self._turn_controller.save_state()
        self.privileged_route_planner.save()

        # Initialize the initial state without braking
        location = np.array(
            [
                self.ego_transform.location.x,
                self.ego_transform.location.y,
                self.ego_transform.location.z,
            ],
        )
        heading_angle = np.array([np.deg2rad(self.ego_transform.rotation.yaw)])
        speed = np.array([self.ego_speed])

        # Calculate the throttle command based on the target speed and current speed
        throttle = self._longitudinal_controller.get_throttle_extrapolation(
            initial_target_speed,
            self.ego_speed,
        )
        steering = self._turn_controller.step(
            self.route_waypoints_np,
            speed,
            location,
            heading_angle.item(),
        )
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bounding_boxes = []
        # Iterate over the future frames and forecast the ego agent's state
        for _ in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(
                location,
                float(heading_angle),
                float(speed),
                action,
            )

            # Update the route and extrapolate steering and throttle commands
            extrapolated_route, _, _, _, _, _, _ = (
                self.privileged_route_planner.run_step(location)
            )
            steering = self._turn_controller.step(
                extrapolated_route,
                speed,
                location,
                heading_angle.item(),
            )
            throttle = self._longitudinal_controller.get_throttle_extrapolation(
                initial_target_speed,
                speed,
            )
            action = np.array([steering, throttle, 0.0]).flatten()

            heading_angle_degrees = np.rad2deg(heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box
            # intersectinos at collisions.
            # In case of driving increase them for safety.
            extent = self.ego_vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= (
                self.config_expert.slow_speed_extent_factor_ego
                if self.ego_speed < self.config_expert.extent_ego_bbs_speed_threshold
                else self.config_expert.high_speed_extent_factor_ego_x
            )
            extent.y *= (
                self.config_expert.slow_speed_extent_factor_ego
                if self.ego_speed < self.config_expert.extent_ego_bbs_speed_threshold
                else self.config_expert.high_speed_extent_factor_ego_y
            )

            transform = carla.Transform(
                carla.Location(
                    x=location[0].item(),
                    y=location[1].item(),
                    z=location[2].item(),
                ),
            )

            ego_bounding_box = carla.BoundingBox(transform.location, extent)
            ego_bounding_box.rotation = carla.Rotation(
                pitch=0,
                yaw=heading_angle_degrees,
                roll=0,
            )

            future_bounding_boxes.append(ego_bounding_box)

        self._turn_controller.load_state()
        self.privileged_route_planner.load()

        return future_bounding_boxes

    @beartype
    def forecast_walkers(
        self,
        number_of_future_frames: int,
    ) -> tuple[list, list]:  # -> tuple[list, list]:
        """
        Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they
        keep their velocity and direction

        Args:
            number_of_future_frames: The number of future frames to forecast.

        Returns:
            A tuple containing two lists:
                1. A list of lists, where each inner list contains the future bounding boxes for a pedestrian.
                2. A list of IDs for the pedestrians whose locations were forecasted.
        """
        nearby_pedestrians_bbs, nearby_pedestrian_ids = [], []

        # Filter pedestrians within the detection radius
        pedestrians = self.walkers_inside_bev

        # If no pedestrians are found, return empty lists
        if not pedestrians:
            return nearby_pedestrians_bbs, nearby_pedestrian_ids

        # Extract pedestrian locations, speeds, and directions
        pedestrian_locations = np.array(
            [
                [ped.get_location().x, ped.get_location().y, ped.get_location().z]
                for ped in pedestrians
            ],
        )
        pedestrian_speeds = np.array(
            [ped.get_velocity().length() for ped in pedestrians],
        )
        pedestrian_speeds = np.maximum(
            pedestrian_speeds,
            self.config_expert.min_walker_speed,
        )
        pedestrian_directions = np.array(
            [
                [
                    ped.get_control().direction.x,
                    ped.get_control().direction.y,
                    ped.get_control().direction.z,
                ]
                for ped in pedestrians
            ],
        )

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = (
            pedestrian_locations[:, None, :]
            + np.arange(1, number_of_future_frames + 1)[None, :, None]
            * pedestrian_directions[:, None, :]
            * pedestrian_speeds[:, None, None]
            / self.config_expert.bicycle_frame_rate
        )

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(pedestrians):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation = carla.Rotation(
                pitch=bb.rotation.pitch + transform.rotation.pitch,
                yaw=bb.rotation.yaw + transform.rotation.yaw,
                roll=bb.rotation.roll + transform.rotation.roll,
            )
            extent = bb.extent
            extent.x = max(
                self.config_expert.pedestrian_minimum_extent,
                extent.x,
            )  # Ensure a minimum width
            extent.y = max(
                self.config_expert.pedestrian_minimum_extent,
                extent.y,
            )  # Ensure a minimum length

            pedestrian_future_bboxes = []
            for j in range(number_of_future_frames):
                location = carla.Location(
                    future_pedestrian_locations[i, j, 0],
                    future_pedestrian_locations[i, j, 1],
                    future_pedestrian_locations[i, j, 2],
                )

                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation
                pedestrian_future_bboxes.append(bounding_box)

            nearby_pedestrian_ids.append(ped.id)
            nearby_pedestrians_bbs.append(pedestrian_future_bboxes)

        self.visualize_pedestrian_bounding_boxes(nearby_pedestrians_bbs)

        return nearby_pedestrians_bbs, nearby_pedestrian_ids

    @beartype
    def ego_agent_affected_by_red_light(self, target_speed: float) -> float:
        """
        Handles the behavior of the ego vehicle when approaching a traffic light.

        Args:
            target_speed: The target speed for the ego vehicle.

        Returns:
            The adjusted target speed for the ego vehicle.
        """

        self.close_traffic_lights.clear()

        for (
            traffic_light,
            traffic_light_center,
            traffic_light_waypoints,
        ) in self.list_traffic_lights:
            center_loc = carla.Location(traffic_light_center)
            if center_loc.distance(self.ego_location) > self.config_expert.light_radius:
                continue

            for wp in traffic_light_waypoints:
                # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
                length_bounding_box = carla.Vector3D(
                    (wp.lane_width / 2.0) * 0.9,
                    traffic_light.trigger_volume.extent.y,
                    traffic_light.trigger_volume.extent.z,
                )
                length_bounding_box = carla.Vector3D(1.5, 1.5, 0.5)

                wp_location = wp.transform
                traffic_light_location = traffic_light.get_transform()
                traffic_light_pos_on_street = common_utils.get_relative_transform(
                    ego_matrix=np.array(wp_location.get_matrix()),
                    vehicle_matrix=np.array(traffic_light_location.get_matrix()),
                )
                traffic_light_bb_location = carla.Location(
                    x=wp_location.location.x,
                    y=wp_location.location.y,
                    z=wp_location.location.z
                    + traffic_light_pos_on_street[
                        -1
                    ],  # z of traffic light is relative to street
                )
                bounding_box = carla.BoundingBox(
                    traffic_light_bb_location,
                    length_bounding_box,
                )

                global_rot = traffic_light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(
                    pitch=global_rot.pitch,
                    yaw=global_rot.yaw,
                    roll=global_rot.roll,
                )

                affects_ego = (
                    self.next_traffic_light is not None
                    and traffic_light.id == self.next_traffic_light.id
                )

                self.close_traffic_lights.append(
                    [
                        traffic_light,
                        bounding_box,
                        traffic_light.state,
                        traffic_light.id,
                        affects_ego,
                    ],
                )

                self.visualize_traffic_lights(traffic_light, wp, bounding_box)

        # If being a sensory agent, we use the more consistent distance to traffic light using bounding box
        # Here we also perform check if a red light is normal and we should come as near as possible or if
        # the red light is not normal and we should keep a larger
        distance_to_traffic_light_stop_point = self.distance_to_next_traffic_light
        if (
            self.config_expert.datagen
            and self.next_traffic_light is not None
            and self.next_traffic_light.id in self.id2bb_map
        ):
            next_traffic_light_bb = self.id2bb_map[
                self.next_traffic_light.id
            ]  # Red light bounding box of the traffic light projected to ego road, doesn't have to be on same lane as ego
            if next_traffic_light_bb is not None:
                distance_ego_to_traffic_light_bb = next_traffic_light_bb["position"][
                    0
                ]  # Longitudinal distance

                if next_traffic_light_bb[
                    "is_over_head_traffic_light"
                ]:  # Overhead traffic light, must stand much further
                    distance_to_traffic_light_stop_point = max(
                        distance_ego_to_traffic_light_bb
                        - self.config_expert.idm_overhead_red_light_minimum_distance,
                        0,
                    )
                    self.over_head_traffic_light = True
                elif next_traffic_light_bb[
                    "is_europe_traffic_light"
                ]:  # Overhead traffic light, must stand much further
                    distance_to_traffic_light_stop_point = max(
                        distance_ego_to_traffic_light_bb
                        - self.config_expert.idm_europe_red_light_minimum_distance,
                        0,
                    )
                    self.europe_traffic_light = True
                else:  # Traffic light is on other side of intersection, we can stop normally
                    distance_to_traffic_light_stop_point = (
                        distance_ego_to_traffic_light_bb
                    )
            else:
                LOG.info(
                    "Warning: Could not find traffic light bounding box, using default distance to stop point.",
                )

        if self.next_traffic_light is not None:
            CarlaDataProvider._global_memory["next_traffic_light"] = (
                self.next_traffic_light
            )
        # If green light, just skip
        if (
            self.next_traffic_light is None
            or self.next_traffic_light.state == carla.TrafficLightState.Green
        ):
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        new_target_speed = expert_utils.compute_target_speed_idm(
            config=self.config_expert,
            desired_speed=target_speed,
            leading_actor_length=0.0,
            ego_speed=self.ego_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=distance_to_traffic_light_stop_point,
            s0=self.config_expert.idm_red_light_minimum_distance,
            T=self.config_expert.idm_red_light_desired_time_headway,
        )

        if (
            abs(new_target_speed - target_speed) > 1e-2
            and distance_to_traffic_light_stop_point < 30.0
        ):
            LOG.info(
                f"""[{self.step}] Ego target speed adjusted for red light in {distance_to_traffic_light_stop_point:.2f}m
    from {target_speed:.2f} to {new_target_speed:.2f}""",
            )

        return new_target_speed

    @beartype
    def ego_agent_affected_by_stop_sign(self, target_speed: float) -> float:
        """
        Handles the behavior of the ego vehicle when approaching a stop sign.

        Args:
            target_speed: The target speed for the ego vehicle.

        Returns:
            The adjusted target speed for the ego vehicle.
        """
        self.close_stop_signs.clear()
        stop_signs = self.get_nearby_object(
            self.actors.filter("*traffic.stop*"),
            self.config_expert.light_radius,
        )

        for stop_sign in stop_signs:
            center_bb_stop_sign = stop_sign.get_transform().transform(
                stop_sign.trigger_volume.location,
            )
            stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
            bounding_box_stop_sign = carla.BoundingBox(
                center_bb_stop_sign,
                stop_sign_extent,
            )
            rotation_stop_sign = stop_sign.get_transform().rotation
            bounding_box_stop_sign.rotation = carla.Rotation(
                pitch=rotation_stop_sign.pitch,
                yaw=rotation_stop_sign.yaw,
                roll=rotation_stop_sign.roll,
            )

            affects_ego = (
                self.next_stop_sign is not None
                and self.next_stop_sign.id == stop_sign.id
                and not self.cleared_stop_sign
            )
            self.close_stop_signs.append(
                [bounding_box_stop_sign, stop_sign.id, affects_ego],
            )

            self.visualize_stop_signs(bounding_box_stop_sign, affects_ego)

        if self.next_stop_sign is None:
            # No stop sign, continue with the current target speed
            return target_speed

        # Calculate the accurate distance to the stop sign
        distance_to_stop_sign = (
            self.next_stop_sign.get_transform()
            .transform(self.next_stop_sign.trigger_volume.location)
            .distance(self.ego_location)
        )

        # Reset the stop sign flag if we are farther than 10m away
        if distance_to_stop_sign > self.config_expert.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
            self.waiting_ticks_at_stop_sign = 0
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if (
                self.ego_speed < 0.1
                and distance_to_stop_sign
                < self.config_expert.clearing_distance_to_stop_sign
            ):
                self.waiting_ticks_at_stop_sign += 1
                if self.waiting_ticks_at_stop_sign > 25:
                    self.cleared_stop_sign = True
            else:
                self.waiting_ticks_at_stop_sign = 0

        # Set the distance to stop sign as infinity if the stop sign has been cleared
        distance_to_stop_sign = (
            np.inf if self.cleared_stop_sign else distance_to_stop_sign
        )

        # If being a sensory agent, we use the more consistent distance to stop sign using bounding box
        if (
            self.config_expert.datagen
            and distance_to_stop_sign < np.inf
            and self.next_stop_sign.id in self.id2bb_map
        ):
            next_traffic_light_bb = self.id2bb_map[self.next_stop_sign.id]
            distance_to_stop_sign = next_traffic_light_bb["distance"]

        # Compute the target speed using the IDM
        target_speed = expert_utils.compute_target_speed_idm(
            config=self.config_expert,
            desired_speed=target_speed,
            leading_actor_length=0,
            ego_speed=self.ego_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=distance_to_stop_sign,
            s0=self.config_expert.idm_stop_sign_minimum_distance,
            T=self.config_expert.idm_stop_sign_desired_time_headway,
        )

        # Return whether the ego vehicle is affected by the stop sign and the adjusted target speed
        return target_speed

    @beartype
    def _get_actor_forward_speed(self, actor: carla.Actor) -> float:
        return self._get_forward_speed(
            transform=actor.get_transform(),
            velocity=actor.get_velocity(),
        )

    @beartype
    def _get_forward_speed(
        self,
        transform: carla.Transform | None = None,
        velocity: carla.Vector3D | None = None,
    ) -> float:
        """
        Calculate the forward speed of the vehicle based on its transform and velocity.

        Args:
            transform: The transform of the vehicle. If not provided, it will be obtained from ego.
            velocity: The velocity of the vehicle. If not provided, it will be obtained from ego.

        Returns:
            The forward speed of the vehicle in m/s.
        """
        if not velocity:
            velocity = self.ego_vehicle.get_velocity()

        if not transform:
            transform = self.ego_vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array(
            [
                np.cos(pitch_rad) * np.cos(yaw_rad),
                np.cos(pitch_rad) * np.sin(yaw_rad),
                np.sin(pitch_rad),
            ],
        )

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return float(forward_speed)

    @beartype
    def _vehicle_obstacle_detected(
        self,
        max_distance: float | None = None,
        up_angle_th: float = 90,
        low_angle_th: float = 0,
        lane_offset: float = 0,
    ) -> tuple[bool, int | None, float | None]:
        """
        Check if there is a vehicle in front of the agent blocking its path.

        Args:
            max_distance: Maximum freespace to check for obstacles. If None, the base threshold value is used.
            up_angle_th: Upper angle threshold in degrees.
            low_angle_th: Lower angle threshold in degrees.
            lane_offset: Lane offset for checking adjacent lanes.

        Returns:
            A tuple containing (obstacle_detected, vehicle_id, distance_to_obstacle).
        """
        self._use_bbs_detection = False
        self._offset = 0

        def get_route_polygon():
            route_bb = []
            extent_y = self.ego_vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(
                    r_ext * r_vec.x,
                    r_ext * r_vec.y,
                )
                p2 = wp.transform.location + carla.Location(
                    l_ext * r_vec.x,
                    l_ext * r_vec.y,
                )
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self.carla_world_map.get_waypoint(
            ego_location,
            lane_type=carla.libcarla.LaneType.Any,
        )

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self.ego_vehicle.bounding_box.extent.x * ego_transform.get_forward_vector(),
        )

        opposite_invasion = (
            abs(self._offset) + self.ego_vehicle.bounding_box.extent.y
            > ego_wpt.lane_width / 2
        )
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in self.vehicles_inside_bev:
            if target_vehicle.id == self.ego_vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self.carla_world_map.get_waypoint(
                target_transform.location,
                lane_type=carla.LaneType.Any,
            )

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:
                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(
                    target_vehicle.get_transform(),
                )
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (
                        True,
                        target_vehicle.id,
                        float(
                            compute_distance(
                                target_vehicle.get_location(),
                                ego_location,
                            ),
                        ),
                    )

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                if (
                    target_wpt.road_id != ego_wpt.road_id
                    or target_wpt.lane_id != ego_wpt.lane_id + lane_offset
                ):
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(
                        steps=3,
                    )[0]
                    if not next_wpt:
                        continue
                    if (
                        target_wpt.road_id != next_wpt.road_id
                        or target_wpt.lane_id != next_wpt.lane_id + lane_offset
                    ):
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(
                    target_rear_transform,
                    ego_front_transform,
                    max_distance,
                    [low_angle_th, up_angle_th],
                ):
                    return (
                        True,
                        target_vehicle.id,
                        float(
                            compute_distance(
                                target_transform.location,
                                ego_transform.location,
                            ),
                        ),
                    )

        return False, None, -1.0

    @beartype
    def save_meta(
        self,
        control: carla.VehicleControl,
        target_speed: float,
        tick_data: dict,
        speed_reduced_by_obj: list | None,
    ) -> dict:
        """
        Save the driving data for the current frame.

        Args:
            control: The control commands for the current frame.
            target_speed: The target speed for the current frame.
            tick_data: Dictionary containing the current state of the vehicle.
            speed_reduced_by_obj: List containing information about the object that caused speed reduction.

        Returns:
            A dictionary containing the driving data for the current frame.
        """
        frame = self.step // self.config_expert.data_save_freq

        # Extract relevant data from inputs
        previous_target_points = [
            tp.tolist() for tp in self._command_planner.previous_target_points
        ]
        previous_commands = [int(i) for i in self._command_planner.previous_commands]
        next_target_points = [tp[0].tolist() for tp in self._command_planner.route]
        next_commands = [
            int(self._command_planner.route[i][1])
            for i in range(len(self._command_planner.route))
        ]

        # Get the remaining route points in the local coordinate frame
        dense_route = []
        dense_route_original = []
        remaining_route = self.remaining_route[
            : self.config_expert.num_route_points_saved
        ]
        remaining_route_original = self.remaining_route_original[
            : self.config_expert.num_route_points_saved
        ]

        changed_route = bool(
            (
                self.privileged_route_planner.route_points[
                    self.privileged_route_planner.route_index
                ]
                != self.privileged_route_planner.original_route_points[
                    self.privileged_route_planner.route_index
                ]
            ).any(),
        )
        for checkpoint, checkpoint_original in zip(
            remaining_route,
            remaining_route_original,
            strict=True,
        ):
            dense_route.append(
                common_utils.inverse_conversion_2d(
                    checkpoint[:2],
                    self.ego_location_array[:2],
                    self.ego_orientation_rad,
                ).tolist(),
            )
            dense_route_original.append(
                common_utils.inverse_conversion_2d(
                    checkpoint_original[:2],
                    self.ego_location_array[:2],
                    self.ego_orientation_rad,
                ).tolist(),
            )

        # Extract speed reduction object information
        (
            speed_reduced_by_obj_type,
            speed_reduced_by_obj_id,
            speed_reduced_by_obj_distance,
        ) = None, None, None
        if speed_reduced_by_obj is not None:
            (
                speed_reduced_by_obj_type,
                speed_reduced_by_obj_id,
                speed_reduced_by_obj_distance,
            ) = speed_reduced_by_obj[1:]
            # Convert numpy to float so that it can be saved to json.
            if speed_reduced_by_obj_distance is not None:
                speed_reduced_by_obj_distance = float(speed_reduced_by_obj_distance)

        ego_wp: carla.Waypoint = self.carla_world_map.get_waypoint(
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
            distance_to_junction_ego = None

        # how far is next junction
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
            distance_to_junction_ego = None

        next_road_ids_ego = []
        next_next_road_ids_ego = []
        for wp in next_lane_wps_ego:
            next_road_ids_ego.append(wp.road_id)
            next_next_wps = expert_utils.wps_next_until_lane_end(wp)
            try:
                next_next_lane_wps_ego = next_next_wps[-1].next(1)
                if len(next_next_lane_wps_ego) == 0:
                    next_next_lane_wps_ego = [next_next_wps[-1]]
            except:
                next_next_lane_wps_ego = []
            for wp2 in next_next_lane_wps_ego:
                if wp2.road_id not in next_next_road_ids_ego:
                    next_next_road_ids_ego.append(wp2.road_id)

        tl = self.carla_world.get_traffic_lights_from_waypoint(ego_wp, 50.0)
        if len(tl) == 0:
            tl_state = "None"
        else:
            tl_state = str(tl[0].state)

        privileged_yaw = np.radians(
            self.ego_vehicle.get_transform().rotation.yaw,
        )  # convert from degrees to radians
        (
            dangerous_adversarial_actors_ids,
            safe_adversarial_actors_ids,
            ignored_adversarial_actors_ids,
        ) = self.adversarial_actors_ids
        data = {
            "num_dangerous_adversarial": len(dangerous_adversarial_actors_ids),
            "num_safe_adversarial": len(safe_adversarial_actors_ids),
            "num_ignored_adversarial": len(ignored_adversarial_actors_ids),
            "rear_adversarial_id": -1
            if self.rear_adversarial_actor is None
            else self.rear_adversarial_actor.id,
            "town": self.town,
            "privileged_past_positions": np.array(
                self.privileged_ego_past_positions,
                dtype=np.float32,
            )[::-1],
            "past_positions": np.array(self.ego_past_positions, dtype=np.float32)[::-1],
            "past_filtered_state": np.array(
                self.ego_past_filtered_state,
                dtype=np.float32,
            )[::-1],
            "past_speeds": np.array(self.speeds_queue, dtype=np.float32)[::-1],
            "past_yaws": np.array(self.ego_past_yaws, dtype=np.float32)[::-1],
            "speed": tick_data["speed"],
            "accel_x": tick_data["accel_x"],
            "accel_y": tick_data["accel_y"],
            "accel_z": tick_data["accel_z"],
            "angular_velocity_x": tick_data["angular_velocity_x"],
            "angular_velocity_y": tick_data["angular_velocity_y"],
            "angular_velocity_z": tick_data["angular_velocity_z"],
            "pos_global": self.ego_location_array.tolist(),
            "filtered_pos_global": tick_data["filtered_state"][:2].tolist(),
            "noisy_pos_global": tick_data["noisy_state"][:2].tolist(),
            "theta": self.ego_orientation_rad,
            "privileged_yaw": privileged_yaw,
            "target_speed": target_speed,
            "speed_limit": self.speed_limit,
            "last_encountered_speed_limit_sign": self.last_encountered_speed_limit_sign,
            "previous_target_points": previous_target_points,
            "next_target_points": next_target_points,
            "previous_commands": previous_commands,
            "next_commands": next_commands,
            "route": np.array(dense_route, dtype=np.float32),
            "route_original": np.array(dense_route_original, dtype=np.float32),
            "changed_route": changed_route,
            "speed_reduced_by_obj_type": speed_reduced_by_obj_type,
            "speed_reduced_by_obj_id": speed_reduced_by_obj_id,
            "speed_reduced_by_obj_distance": speed_reduced_by_obj_distance,
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": bool(control.brake),
            "perturbation_translation": self.perturbation_translation,
            "perturbation_rotation": self.perturbation_rotation,
            "ego_matrix": np.array(
                self.ego_vehicle.get_transform().get_matrix(),
                dtype=np.float32,
            ),
            "scenario": self.scenario_name,
            "traffic_light_state": tl_state,
            "distance_to_next_junction": self.distance_to_next_junction,
            "ego_lane_id": self.ego_lane_id,
            "road_id": ego_wp.road_id,
            "lane_id": ego_wp.lane_id,
            "is_junction": ego_wp.is_junction,
            "is_intersection": ego_wp.is_intersection,
            "junction_id": ego_wp.junction_id,
            "next_road_ids": next_road_ids_ego,
            "next_next_road_ids_ego": next_next_road_ids_ego,
            "lane_change_str": str(ego_wp.lane_change),
            "lane_type_str": str(ego_wp.lane_type),
            "left_lane_marking_color_str": str(ego_wp.left_lane_marking.color),
            "left_lane_marking_type_str": str(ego_wp.left_lane_marking.type),
            "right_lane_marking_color_str": str(ego_wp.right_lane_marking.color),
            "right_lane_marking_type_str": str(ego_wp.right_lane_marking.type),
            "route_curvature": self.route_curvature,
            "dist_to_construction_site": self.distance_to_construction_site,
            "dist_to_accident_site": self.distance_to_accident_site,
            "dist_to_parked_obstacle": self.distance_to_parked_obstacle,
            "dist_to_vehicle_opens_door": self.distance_to_vehicle_opens_door,
            "dist_to_cutin_vehicle": self.distance_to_cutin_vehicle,
            "dist_to_pedestrian": self.distance_to_pedestrian,
            "dist_to_biker": self.distance_to_biker,
            "dist_to_junction": distance_to_junction_ego,
            "current_active_scenario_type": self.current_active_scenario_type,
            "previous_active_scenario_type": self.previous_active_scenario_type,
            "scenario_obstacles_ids": self.scenario_obstacles_ids,
            "scenario_actors_ids": self.scenario_actors_ids,
            "vehicle_opened_door": self.vehicle_opened_door,
            "vehicle_door_side": self.vehicle_door_side,
            "scenario_obstacles_convex_hull": self.scenario_obstacles_convex_hull,
            "cut_in_actors_ids": self.cut_in_actors_ids,
            "distance_to_intersection_index_ego": self.distance_to_intersection_index_ego,
            "ego_lane_width": self.ego_lane_width,
            "target_lane_width": self.target_lane_width,
            "weather_setting": self.weather_setting,
            "jpeg_storage_quality": self.jpeg_storage_quality,
            "route_left_length": self.route_left_length,
            "distance_ego_to_route": self.distance_ego_to_route,
            "weather_parameters": self.weather_parameters,
            "signed_dist_to_lane_change": self.signed_dist_to_lane_change,
            "visual_visibility": int(self.visual_visibility),
            "num_parking_vehicles_in_proximity": self.num_parking_vehicles_in_proximity,
            "second_highest_speed": self.second_highest_speed,
            "second_highest_speed_limit": self.second_highest_speed_limit,
            "dataset_information": {
                "save_depth_lower_resolution": self.config_expert.save_depth_lower_resolution,
                "save_depth_resolution_ratio": self.config_expert.save_depth_resolution_ratio,
                "save_depth_bits": self.config_expert.save_depth_bits,
                "save_only_non_ground_lidar": self.config_expert.save_only_non_ground_lidar,
                "target_dataset": int(self.config_expert.target_dataset),
                "data_save_freq": self.config_expert.data_save_freq,
                "save_grouped_semantic": self.config_expert.save_grouped_semantic,
            },
            "sensor_information": {
                "lidar_pos_1": self.config_expert.lidar_pos_1,
                "lidar_rot_1": self.config_expert.lidar_rot_1,
                "lidar_pos_2": self.config_expert.lidar_pos_2,
                "lidar_rot_2": self.config_expert.lidar_rot_2,
                "lidar_accumulation": self.config_expert.lidar_accumulation,
                "num_cameras": self.config_expert.num_cameras,
                "camera_calibration": self.config_expert.camera_calibration,
                "num_radar_sensors": self.config_expert.num_radar_sensors,
                "radar_calibration": self.config_expert.radar_calibration,
            },
            "europe_traffic_light": self.europe_traffic_light,
            "over_head_traffic_light": self.over_head_traffic_light,
            "emergency_brake_for_special_vehicle": self.emergency_brake_for_special_vehicle,
            "vehicle_hazard": bool(self.vehicle_hazard),
            "vehicle_affecting_id": self.vehicle_affecting_id,
            "light_hazard": bool(self.traffic_light_hazard),
            "walker_hazard": bool(self.walker_hazard),
            "walker_affecting_id": self.walker_affecting_id,
            "stop_sign_hazard": bool(self.stop_sign_hazard),
            "stop_sign_close": bool(self.stop_sign_close),
            "walker_close": bool(self.walker_close),
            "walker_close_id": self.walker_close_id,
            "target_speed_limit": self.target_speed_limit,
            "slower_bad_visibility": self.slower_bad_visibility,
            "slower_clutterness": self.slower_clutterness,
            "slower_occluded_junction": self.slower_occluded_junction,
            "does_emergency_brake_for_pedestrians": self.does_emergency_brake_for_pedestrians,
            "construction_obstacle_two_ways_stuck": self.construction_obstacle_two_ways_stuck,
            "accident_two_ways_stuck": self.accident_two_ways_stuck,
            "parked_obstacle_two_ways_stuck": self.parked_obstacle_two_ways_stuck,
            "vehicle_opens_door_two_ways_stuck": self.vehicle_opens_door_two_ways_stuck,
            "rear_danger_8": self.rear_danger_8,
            "rear_danger_16": self.rear_danger_16,
            "brake_cutin": self.brake_cutin,
        }

        previous_gps_target_points_dict = {}
        previous_gps_commands_dict = {}
        next_gps_target_points_dict = {}
        next_gps_commands_dict = {}
        for k, v in self.gps_waypoint_planners_dict.items():
            previous_gps_target_points_dict[k] = [
                tp.tolist() for tp in v.previous_target_points
            ]
            previous_gps_commands_dict[k] = [int(i) for i in v.previous_commands]
            next_gps_target_points_dict[k] = [tp[0].tolist() for tp in v.route]
            next_gps_commands_dict[k] = [
                int(v.route[i][1]) for i in range(len(v.route))
            ]

        for k, v in previous_gps_target_points_dict.items():
            data[f"previous_gps_target_points_{k}"] = v
        for k, v in previous_gps_commands_dict.items():
            data[f"previous_gps_commands_{k}"] = v
        for k, v in next_gps_target_points_dict.items():
            data[f"next_gps_target_points_{k}"] = v
        for k, v in next_gps_commands_dict.items():
            data[f"next_gps_commands_{k}"] = v

        previous_target_points_dict = {}
        previous_commands_dict = {}
        next_target_points_dict = {}
        next_commands_dict = {}
        for k, v in self._command_planners_dict.items():
            previous_target_points_dict[k] = [
                tp.tolist() for tp in v.previous_target_points
            ]
            previous_commands_dict[k] = [int(i) for i in v.previous_commands]
            next_target_points_dict[k] = [tp[0].tolist() for tp in v.route]
            next_commands_dict[k] = [int(v.route[i][1]) for i in range(len(v.route))]

        for k, v in previous_target_points_dict.items():
            data[f"previous_target_points_{k}"] = v
        for k, v in previous_commands_dict.items():
            data[f"previous_commands_{k}"] = v
        for k, v in next_target_points_dict.items():
            data[f"next_target_points_{k}"] = v
        for k, v in next_commands_dict.items():
            data[f"next_commands_{k}"] = v

        self.metas.append((self.step, frame, data))

        return data

    @beartype
    def _visualize_route(self) -> None:
        """Visualize the current route using CARLA debug drawing."""
        route_points = self.route_waypoints_np
        if len(route_points) == 0:
            return

        sampled_points = route_points[:1000:20]
        for point in sampled_points:
            location = carla.Location(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]) + 1.0,
            )
            self.carla_world.debug.draw_point(
                location,
                size=0.05,  # Smaller size
                color=self.config_expert.future_route_color,  # Green
                life_time=self.config_expert.draw_life_time,  # Short life time for continuous updates
            )

        # Draw lines between consecutive sampled route points
        for i in range(len(sampled_points) - 1):
            start = carla.Location(
                x=float(sampled_points[i][0]),
                y=float(sampled_points[i][1]),
                z=float(sampled_points[i][2]) + 1.0,
            )
            end = carla.Location(
                x=float(sampled_points[i + 1][0]),
                y=float(sampled_points[i + 1][1]),
                z=float(sampled_points[i + 1][2]) + 1.0,
            )
            self.carla_world.debug.draw_line(
                start,
                end,
                thickness=0.05,  # Thinner lines
                color=self.config_expert.future_route_color,  # Green
                life_time=self.config_expert.draw_life_time,
            )

    @beartype
    def _visualize_original_route(self) -> None:
        """Visualize the original route using CARLA debug drawing."""
        original_route_points = self.original_route_waypoints_np
        if len(original_route_points) == 0:
            return

        sampled_points = original_route_points[:1000:20]
        for point in sampled_points:
            location = carla.Location(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]) + 1.5,
            )
            self.carla_world.debug.draw_point(
                location,
                size=0.05,  # Even smaller size
                color=carla.Color(0, 0, 255),  # Blue
                life_time=self.config_expert.draw_life_time,  # Short life time for continuous updates
            )

        # Draw lines between consecutive sampled original route points
        for i in range(len(sampled_points) - 1):
            start = carla.Location(
                x=float(sampled_points[i][0]),
                y=float(sampled_points[i][1]),
                z=float(sampled_points[i][2]) + 1.5,
            )
            end = carla.Location(
                x=float(sampled_points[i + 1][0]),
                y=float(sampled_points[i + 1][1]),
                z=float(sampled_points[i + 1][2]) + 1.5,
            )
            self.carla_world.debug.draw_line(
                start,
                end,
                thickness=0.04,  # Thinner lines
                color=carla.Color(0, 0, 255),  # Blue
                life_time=self.config_expert.draw_life_time,
            )
