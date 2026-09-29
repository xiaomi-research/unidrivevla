import os
from enum import IntEnum, auto

import numpy as np
from tqdm import tqdm

from lead.data_buckets.abstract_bucket_collection import AbstractBucketCollection
from lead.data_buckets.bucket import Bucket
from lead.training.config_training import TrainingConfig


class WaymoSim2RealBuckets(IntEnum):
    # --- Scenario specific mining ---
    ACCIDENT_SCENARIO = 0
    ACCIDENT_TWO_WAYS_SCENARIO = auto()
    BLOCKED_INTERSECTION_SCENARIO = auto()
    CONSTRUCTION_OBSTACLE_SCENARIO = auto()
    CONSTRUCTION_OBSTACLE_TWO_WAYS_SCENARIO = auto()
    CROSSING_BICYCLE_FLOW_SCENARIO = auto()
    CROSS_JUNCTION_DEFECT_TRAFFIC_LIGHT_SCENARIO = auto()
    DYNAMIC_OBJECT_CROSSING_SCENARIO = auto()
    ENTER_ACTOR_FLOW_SCENARIO = auto()
    ENTER_ACTOR_FLOW_V2_SCENARIO = auto()
    HARD_BREAK_ROUTE_SCENARIO = auto()
    HAZARD_AT_SIDE_LANE_SCENARIO = auto()
    HAZARD_AT_SIDE_LANE_TWO_WAYS_SCENARIO = auto()
    HIGHWAY_CUT_IN_SCENARIO = auto()
    HIGHWAY_EXIT_SCENARIO = auto()
    INTERURBAN_ACTOR_FLOW_SCENARIO = auto()
    INTERURBAN_ADVANCED_ACTOR_FLOW_SCENARIO = auto()
    INVADING_TURN_SCENARIO = auto()
    MERGER_INTO_SLOW_TRAFFIC_SCENARIO = auto()
    MERGER_INTO_SLOW_TRAFFIC_V2_SCENARIO = auto()
    NON_SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO = auto()
    NON_SIGNALIZED_JUNCTION_LEFT_TURN_ENTER_FLOW_SCENARIO = auto()
    NON_SIGNALIZED_JUNCTION_RIGHT_TURN_SCENARIO = auto()
    OPPOSITE_VEHICLE_RUNNING_RED_LIGHT_SCENARIO = auto()
    OPPOSITE_VEHICLE_TAKING_PRIORITY_SCENARIO = auto()
    PARKED_OBSTACLE_SCENARIO = auto()
    PARKED_OBSTACLE_TWO_WAYS_SCENARIO = auto()
    PARKING_CROSSING_PEDESTRIAN_SCENARIO = auto()
    PARKING_CUT_IN_SCENARIO = auto()
    PARKING_EXIT_SCENARIO = auto()
    PEDESTRIAN_CROSSING_SCENARIO = auto()
    PRIORITY_AT_JUNCTION_SCENARIO = auto()
    RED_LIGHT_WITHOUT_LEAD_VEHICLE_SCENARIO = auto()
    SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO = auto()
    SIGNALIZED_JUNCTION_LEFT_TURN_ENTER_FLOW_SCENARIO = auto()
    SIGNALIZED_JUNCTION_RIGHT_TURN_SCENARIO = auto()
    STATIC_CUT_IN_SCENARIO = auto()
    VEHICLE_OPENS_DOOR_TWO_WAYS_SCENARIO = auto()
    VEHICLE_TURNING_ROUTE_SCENARIO = auto()
    VEHICLE_TURNING_ROUTE_PEDESTRIAN_SCENARIO = auto()
    FAR_TARGET_POINT = auto()
    # --- General mining ---
    TOWN15 = auto()
    CURRENT_TARGET_POINT_BEHIND_EGO = auto()
    OCCLUDED_JUNCTION = auto()
    ENTERING_JUNCTION = auto()
    CLOSE_TO_JUNCTION = auto()
    RED_OVERHEAD_TRAFFIC_LIGHT = auto()
    STOP_SIGN_HAZARD = auto()
    LARGE_LATERAL_DEVIATION = auto()
    VEHICLE_HAZARD = auto()
    RED_EUROPE_TRAFFIC_LIGHT = auto()
    RED_TRAFFIC_LIGHT = auto()
    HIGH_ACCELERATION = auto()
    MEDIUM_ACCELERATION = auto()
    LOW_ACCELERATION = auto()
    HIGH_ROUTE_CURVATURE = auto()
    MEDIUM_ROUTE_CURVATURE = auto()

    OTHERS = auto()

    @classmethod
    def __len__(cls):
        return len(cls.__members__)

    @classmethod
    def index_of(cls, member):
        return list(cls).index(member)

    @classmethod
    def member_at(cls, index):
        return list(cls)[index]


class WaymoBucketCollection(AbstractBucketCollection):
    def __init__(self, root: str | list[str], config: TrainingConfig):
        self.buckets = [Bucket(config) for _ in range(len(WaymoSim2RealBuckets))]
        super().__init__(root, config)
        print(f"Using Waymo buckets with {len(WaymoSim2RealBuckets)} buckets")

    def _build_buckets(self):
        from lead.data_loader.carla_dataset import CARLAData

        carla_data = CARLAData(
            root=self.root,
            config=self.config,
            training_session_cache=None,
            random=False,
            build_buckets=True,
        )
        self.trainable_routes = carla_data.bucket_collection.trainable_routes
        self.trainable_frames = carla_data.bucket_collection.trainable_frames
        self.total_routes = carla_data.bucket_collection.total_routes
        for sample_idx, sample in tqdm(enumerate(carla_data)):
            if sample_idx % 100000 == 0:
                self.print_statistic()
            route_dir = sample["route_dir"]
            frame_number = int(sample["frame_number"])

            if sample["scenario_type"] == "Accident" and self._check_accident_scenario(
                sample,
            ):
                self.buckets[WaymoSim2RealBuckets.ACCIDENT_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "scenario_type"
            ] == "AccidentTwoWays" and self._check_accident_scenario(sample):
                self.buckets[WaymoSim2RealBuckets.ACCIDENT_TWO_WAYS_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "current_active_scenario_type"
            ] == "BlockedIntersection" and self._check_blocked_intersection(sample):
                self.buckets[WaymoSim2RealBuckets.BLOCKED_INTERSECTION_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "scenario_type"
            ] == "ConstructionObstacle" and self._check_construction_obstacle_scenario(
                sample,
            ):
                self.buckets[WaymoSim2RealBuckets.CONSTRUCTION_OBSTACLE_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif (
                sample["scenario_type"] == "ConstructionObstacleTwoWays"
                and self._check_construction_obstacle_scenario(sample)
            ):
                self.buckets[
                    WaymoSim2RealBuckets.CONSTRUCTION_OBSTACLE_TWO_WAYS_SCENARIO
                ].add(route_dir, frame_number)
            elif sample[
                "current_active_scenario_type"
            ] == "CrossingBicycleFlow" and self._check_scenario_actor_close(
                sample,
                max_distance=10.0,
            ):
                self.buckets[WaymoSim2RealBuckets.CROSSING_BICYCLE_FLOW_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "current_active_scenario_type"
            ] == "CrossJunctionDefectTrafficLight" and self._check_scenario_actor_close(
                sample,
                max_distance=10.0,
            ):
                self.buckets[
                    WaymoSim2RealBuckets.CROSS_JUNCTION_DEFECT_TRAFFIC_LIGHT_SCENARIO
                ].add(route_dir, frame_number)
            elif sample[
                "current_active_scenario_type"
            ] == "DynamicObjectCrossing" and self._check_scenario_actor_close(
                sample,
                max_distance=20.0,
            ):
                self.buckets[WaymoSim2RealBuckets.DYNAMIC_OBJECT_CROSSING_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "current_active_scenario_type"
            ] == "HardBreakRoute" and self._check_is_hard_break_route(sample):
                self.buckets[WaymoSim2RealBuckets.HARD_BREAK_ROUTE_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "current_active_scenario_type"
            ] == "HazardAtSideLane" and self._check_scenario_actor_close(
                sample,
                max_distance=20.0,
            ):
                self.buckets[WaymoSim2RealBuckets.HAZARD_AT_SIDE_LANE_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "current_active_scenario_type"
            ] == "HazardAtSideLaneTwoWays" and self._check_scenario_actor_close(
                sample,
                max_distance=20.0,
            ):
                self.buckets[
                    WaymoSim2RealBuckets.HAZARD_AT_SIDE_LANE_TWO_WAYS_SCENARIO
                ].add(route_dir, frame_number)
            elif self._check_highway_cutin_scenario(sample):
                self.buckets[WaymoSim2RealBuckets.HIGHWAY_CUT_IN_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif (
                sample["current_active_scenario_type"] == "InterurbanActorFlow"
                and len(sample["scenario_actors_ids"]) > 0
                and float(sample["distance_to_next_junction"]) < 5.0
            ):
                self.buckets[WaymoSim2RealBuckets.INTERURBAN_ACTOR_FLOW_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif (
                sample["current_active_scenario_type"] == "InterurbanAdvancedActorFlow"
                and len(sample["scenario_actors_ids"]) > 0
                and float(sample["distance_to_next_junction"]) < 5.0
            ):
                self.buckets[
                    WaymoSim2RealBuckets.INTERURBAN_ADVANCED_ACTOR_FLOW_SCENARIO
                ].add(route_dir, frame_number)
            elif sample[
                "scenario_type"
            ] == "InvadingTurn" and self._check_invading_turn_scenario(sample):
                self.buckets[WaymoSim2RealBuckets.INVADING_TURN_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
                self.buckets[
                    WaymoSim2RealBuckets.MERGER_INTO_SLOW_TRAFFIC_V2_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"]
                == "NonSignalizedJunctionLeftTurn"
                and float(sample["distance_to_next_junction"]) < 5.0
                and len(sample["scenario_actors_ids"]) > 0
            ):
                self.buckets[
                    WaymoSim2RealBuckets.NON_SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"]
                == "NonSignalizedJunctionLeftTurnEnterFlow"
                and float(sample["distance_to_next_junction"]) < 5.0
                and len(sample["scenario_actors_ids"]) > 0
            ):
                self.buckets[
                    WaymoSim2RealBuckets.NON_SIGNALIZED_JUNCTION_LEFT_TURN_ENTER_FLOW_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"]
                == "NonSignalizedJunctionRightTurn"
                and float(sample["distance_to_next_junction"]) < 5.0
                and len(sample["scenario_actors_ids"]) > 0
            ):
                self.buckets[
                    WaymoSim2RealBuckets.NON_SIGNALIZED_JUNCTION_RIGHT_TURN_SCENARIO
                ].add(route_dir, frame_number)
            elif self._check_opposite_vehicle_running_red_light_scenario(sample):
                self.buckets[
                    WaymoSim2RealBuckets.OPPOSITE_VEHICLE_RUNNING_RED_LIGHT_SCENARIO
                ].add(route_dir, frame_number)
            elif self._check_opposite_vehicle_taking_priority_scenario(sample):
                self.buckets[
                    WaymoSim2RealBuckets.OPPOSITE_VEHICLE_TAKING_PRIORITY_SCENARIO
                ].add(route_dir, frame_number)
            elif sample[
                "scenario_type"
            ] == "ParkedObstacle" and self._check_parked_obstacle_scenario(sample):
                self.buckets[WaymoSim2RealBuckets.PARKED_OBSTACLE_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "scenario_type"
            ] == "ParkedObstacleTwoWays" and self._check_parked_obstacle_scenario(
                sample,
            ):
                self.buckets[
                    WaymoSim2RealBuckets.PARKED_OBSTACLE_TWO_WAYS_SCENARIO
                ].add(route_dir, frame_number)
            elif sample[
                "current_active_scenario_type"
            ] == "ParkingCrossingPedestrian" and self._check_scenario_actor_close(
                sample,
                max_distance=20.0,
            ):
                self.buckets[
                    WaymoSim2RealBuckets.PARKING_CROSSING_PEDESTRIAN_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"] == "ParkingCutIn"
                and float(sample["dist_to_cutin_vehicle"]) < 20
            ):
                self.buckets[WaymoSim2RealBuckets.PARKING_CUT_IN_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample["current_active_scenario_type"] == "ParkingExit":
                self.buckets[WaymoSim2RealBuckets.PARKING_EXIT_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif (
                sample["current_active_scenario_type"] == "PedestrianCrossing"
                and len(sample["scenario_actors_ids"]) > 0
            ):
                self.buckets[WaymoSim2RealBuckets.PEDESTRIAN_CROSSING_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif (
                sample["current_active_scenario_type"] == "PriorityAtJunction"
                and float(sample["distance_to_next_junction"]) < 5.0
            ):
                self.buckets[WaymoSim2RealBuckets.PRIORITY_AT_JUNCTION_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "current_active_scenario_type"
            ] == "RedLightWithoutLeadVehicle" and self._check_red_traffic_light(sample):
                self.buckets[
                    WaymoSim2RealBuckets.RED_LIGHT_WITHOUT_LEAD_VEHICLE_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"] == "SignalizedJunctionLeftTurn"
                and float(sample["distance_to_next_junction"]) < 5.0
                and len(sample["scenario_actors_ids"]) > 0
            ):
                self.buckets[
                    WaymoSim2RealBuckets.SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"]
                == "SignalizedJunctionLeftTurnEnterFlow"
                and float(sample["distance_to_next_junction"]) < 5.0
                and len(sample["scenario_actors_ids"]) > 0
            ):
                self.buckets[
                    WaymoSim2RealBuckets.SIGNALIZED_JUNCTION_LEFT_TURN_ENTER_FLOW_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"] == "SignalizedJunctionRightTurn"
                and float(sample["distance_to_next_junction"]) < 5.0
                and len(sample["scenario_actors_ids"]) > 0
            ):
                self.buckets[
                    WaymoSim2RealBuckets.SIGNALIZED_JUNCTION_RIGHT_TURN_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"] == "StaticCutIn"
                and float(sample["dist_to_cutin_vehicle"]) < 20
            ):
                self.buckets[WaymoSim2RealBuckets.STATIC_CUT_IN_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif sample[
                "scenario_type"
            ] == "VehicleOpensDoorTwoWays" and self._check_vehicle_opens_door_scenario(
                sample,
            ):
                self.buckets[
                    WaymoSim2RealBuckets.VEHICLE_OPENS_DOOR_TWO_WAYS_SCENARIO
                ].add(route_dir, frame_number)
            elif (
                sample["current_active_scenario_type"] == "VehicleTurningRoute"
                and len(sample["scenario_actors_ids"]) > 0
                and self._check_vehicle_turning_route(sample)
            ):
                self.buckets[WaymoSim2RealBuckets.VEHICLE_TURNING_ROUTE_SCENARIO].add(
                    route_dir,
                    frame_number,
                )
            elif (
                sample["current_active_scenario_type"]
                == "VehicleTurningRoutePedestrian"
                and len(sample["scenario_actors_ids"]) > 0
                and self._check_vehicle_turning_route_pedestrian(sample)
            ):
                self.buckets[
                    WaymoSim2RealBuckets.VEHICLE_TURNING_ROUTE_PEDESTRIAN_SCENARIO
                ].add(route_dir, frame_number)

            # Non-scenario-specific mining
            elif self._check_red_overhead_traffic_light(sample):
                self.buckets[WaymoSim2RealBuckets.RED_OVERHEAD_TRAFFIC_LIGHT].add(
                    route_dir,
                    frame_number,
                )
            elif self._check_red_europe_traffic_light(sample):
                self.buckets[WaymoSim2RealBuckets.RED_EUROPE_TRAFFIC_LIGHT].add(
                    route_dir,
                    frame_number,
                )
            elif (
                bool(sample["slower_occluded_junction"])
                and float(sample["distance_to_next_junction"]) < 5.0
            ):
                self.buckets[WaymoSim2RealBuckets.OCCLUDED_JUNCTION].add(
                    route_dir,
                    frame_number,
                )
            elif abs(float(sample["privileged_acceleration"])) > 17.5:
                self.buckets[WaymoSim2RealBuckets.HIGH_ACCELERATION].add(
                    route_dir,
                    frame_number,
                )
            elif self._check_large_lateral_deviation(sample):
                self.buckets[WaymoSim2RealBuckets.LARGE_LATERAL_DEVIATION].add(
                    route_dir,
                    frame_number,
                )
            elif (
                bool(sample["stop_sign_hazard"])
                and float(sample["distance_to_next_junction"]) < 10.0
            ):
                self.buckets[WaymoSim2RealBuckets.STOP_SIGN_HAZARD].add(
                    route_dir,
                    frame_number,
                )
            elif (
                0.0 < float(sample["distance_to_next_junction"]) < 3.5
                and float(sample["speed"]) > 2.0
                and float(sample["privileged_acceleration"]) > 2.0
            ):
                self.buckets[WaymoSim2RealBuckets.ENTERING_JUNCTION].add(
                    route_dir,
                    frame_number,
                )
            elif (
                float(sample["distance_to_next_junction"]) < 1.0
                and float(sample["speed_limit"]) < 50.0 / 3.6
            ):
                self.buckets[WaymoSim2RealBuckets.CLOSE_TO_JUNCTION].add(
                    route_dir,
                    frame_number,
                )
            elif sample["target_point"][0] < 0:
                self.buckets[WaymoSim2RealBuckets.CURRENT_TARGET_POINT_BEHIND_EGO].add(
                    route_dir,
                    frame_number,
                )
            elif bool(sample["vehicle_hazard"]):
                self.buckets[WaymoSim2RealBuckets.VEHICLE_HAZARD].add(
                    route_dir,
                    frame_number,
                )
            elif (
                self._check_red_traffic_light(sample)
                and float(sample["distance_to_next_junction"]) < 5.0
            ):
                self.buckets[WaymoSim2RealBuckets.RED_TRAFFIC_LIGHT].add(
                    route_dir,
                    frame_number,
                )
            elif sample["town"] == "Town15":
                self.buckets[WaymoSim2RealBuckets.TOWN15].add(route_dir, frame_number)
            elif (
                np.linalg.norm(sample["target_point"]) > 175.0
                and abs(sample["target_point"][0]) > 10
            ):
                self.buckets[WaymoSim2RealBuckets.FAR_TARGET_POINT].add(
                    route_dir,
                    frame_number,
                )
            elif abs(float(sample["privileged_acceleration"])) > 15.0:
                self.buckets[WaymoSim2RealBuckets.MEDIUM_ACCELERATION].add(
                    route_dir,
                    frame_number,
                )
            elif abs(float(sample["privileged_acceleration"])) > 12.5:
                self.buckets[WaymoSim2RealBuckets.LOW_ACCELERATION].add(
                    route_dir,
                    frame_number,
                )
            elif sample["route_labels_curvature"] > 0.15 and sample["speed"] > 0.1:
                self.buckets[WaymoSim2RealBuckets.HIGH_ROUTE_CURVATURE].add(
                    route_dir,
                    frame_number,
                )
            elif sample["route_labels_curvature"] > 0.075 and sample["speed"] > 0.1:
                self.buckets[WaymoSim2RealBuckets.MEDIUM_ROUTE_CURVATURE].add(
                    route_dir,
                    frame_number,
                )
            else:
                self.buckets[WaymoSim2RealBuckets.OTHERS].add(route_dir, frame_number)

        self.print_statistic()

    def _check_accident_scenario(self, sample: dict) -> bool:
        """Check if sample has accident obstacle actors positioned beyond -10m and within 30m distance."""
        if float(sample["dist_to_accident_site"]) >= 42:
            return False
        if len(sample["scenario_obstacles_ids"]) == 0:
            return False
        scenario_obstacle_ids = set(sample["scenario_obstacles_ids"])
        for box in sample["boxes"]:
            if box["id"] in scenario_obstacle_ids and box["position"][0] > 0.0:
                return True
        return False

    def _check_blocked_intersection(self, sample: dict) -> bool:
        """Check if sample is in a blocked intersection scenario."""
        if sample["current_active_scenario_type"] != "BlockedIntersection":
            return False
        if len(sample["scenario_obstacles_ids"]) == 0:
            return False
        scenario_actor_id = sample["scenario_obstacles_ids"][0]
        for box in sample["boxes"]:
            if (
                box["class"] == "car"
                and float(box["speed"]) < 0.1
                and box["id"] == scenario_actor_id
                and float(box["distance"]) < 15
            ):
                return True
        return False

    def _check_construction_obstacle_scenario(self, sample: dict) -> bool:
        """Check if sample has construction cones positioned beyond -10m and within 42m distance."""
        if float(sample["dist_to_construction_site"]) >= 42:
            return False
        for box in sample["boxes"]:
            if (
                box.get("type_id") == "static.prop.constructioncone"
                and box["position"][0] > -10.0
            ):
                return True
        return False

    def _check_scenario_actor_close(self, sample: dict, max_distance: float) -> bool:
        """Check if any scenario actor is within max_distance from ego."""
        if len(sample["scenario_actors_ids"]) == 0:
            return False
        scenario_actor_ids = set(sample["scenario_actors_ids"])
        for box in sample["boxes"]:
            if (
                box["id"] in scenario_actor_ids
                and float(box["distance"]) < max_distance
            ):
                return True
        return False

    def _check_is_hard_break_route(self, sample: dict) -> bool:
        """Check if sample is in a hard break route scenario."""
        if sample["current_active_scenario_type"] != "HardBreakRoute":
            return False
        if not bool(sample["brake"]) or bool(sample["vehicle_hazard"]):
            return False
        if sample["privileged_acceleration"] < -3.0 or float(sample["speed"]) == 0.0:
            return True
        return False

    def _check_highway_cutin_scenario(self, sample: dict) -> bool:
        """Check if sample is in a highway cut-in scenario."""
        if sample["current_active_scenario_type"] != "HighwayCutIn":
            return False
        if len(sample["scenario_actors_ids"]) == 0:
            return False
        scenario_actor_id = sample["scenario_actors_ids"][0]
        for box in sample["boxes"]:
            if (
                box["class"] == "car"
                and box["id"] == scenario_actor_id
                and float(box["distance"]) < 32
            ):
                return True
        return False

    def _check_invading_turn_scenario(self, sample: dict) -> bool:
        """Check if sample is in an invading turn scenario."""
        if sample["scenario_type"] != "InvadingTurn":
            return False
        for box in sample["boxes"]:
            if (
                box.get("type_id") == "static.prop.constructioncone"
                and 0 < box["position"][0] < 16
            ):
                return True
        return False

    def _check_opposite_vehicle_running_red_light_scenario(self, sample: dict) -> bool:
        """Check if sample is in an opposite vehicle running red light scenario."""
        if sample["current_active_scenario_type"] != "OppositeVehicleRunningRedLight":
            return False
        if len(sample["scenario_actors_ids"]) == 0:
            return False
        scenario_actor_id = sample["scenario_actors_ids"][0]
        for box in sample["boxes"]:
            if (
                box["class"] == "car"
                and box["id"] == scenario_actor_id
                and float(box["speed"]) > 0.1
            ):
                return True
        return False

    def _check_opposite_vehicle_taking_priority_scenario(self, sample: dict) -> bool:
        """Check if sample is in an opposite vehicle taking priority scenario."""
        if sample["current_active_scenario_type"] != "OppositeVehicleTakingPriority":
            return False
        if len(sample["scenario_actors_ids"]) == 0:
            return False
        scenario_actor_id = sample["scenario_actors_ids"][0]
        for box in sample["boxes"]:
            if (
                box["class"] == "car"
                and box["id"] == scenario_actor_id
                and float(box["speed"]) > 0.1
            ):
                return True
        return False

    def _check_parked_obstacle_scenario(self, sample: dict) -> bool:
        """Check if sample has parked vehicles positioned beyond -8m and within 35m distance."""
        if float(sample["dist_to_parked_obstacle"]) >= 35:
            return False
        for box in sample["boxes"]:
            if (
                box["class"] == "car"
                and float(box["speed"]) < 0.1
                and box["position"][0] > -8.0
            ):
                return True
        return False

    def _check_vehicle_opens_door_scenario(self, sample: dict) -> bool:
        """Check if sample has vehicle opening door within 24m distance and not more than 8m behind."""
        if float(sample["dist_to_vehicle_opens_door"]) >= 24:
            return False
        if len(sample["scenario_obstacles_ids"]) == 0:
            return False
        scenario_obstacle_ids = set(sample["scenario_obstacles_ids"])
        for box in sample["boxes"]:
            if box["id"] in scenario_obstacle_ids and box["position"][0] > -8.0:
                return True
        return False

    def _check_red_traffic_light(self, sample: dict) -> bool:
        """Check if sample has red traffic light affecting ego vehicle while stopped."""
        if not sample["light_hazard"]:
            return False
        if float(sample["speed"]) >= 0.1:
            return False

        boxes = sample["boxes"]
        for box in boxes:
            if (
                box["class"] == "traffic_light"
                and box["state"] == "Red"
                and box["affects_ego"]
            ):
                return True
        return False

    def _check_vehicle_turning_route(self, sample: dict) -> bool:
        """Check if sample has vehicle turning route scenario."""
        if sample["current_active_scenario_type"] != "VehicleTurningRoute":
            return False
        if len(sample["scenario_actors_ids"]) == 0:
            return False
        scenario_actor_id = sample["scenario_actors_ids"][0]
        for box in sample["boxes"]:
            if box["id"] == scenario_actor_id and float(box["distance"]) < 15:
                return True
        return False

    def _check_vehicle_turning_route_pedestrian(self, sample: dict) -> bool:
        """Check if sample has vehicle turning route scenario."""
        if sample["current_active_scenario_type"] != "VehicleTurningRoutePedestrian":
            return False
        if len(sample["scenario_actors_ids"]) == 0:
            return False
        scenario_actor_id = sample["scenario_actors_ids"][0]
        for box in sample["boxes"]:
            if box["id"] == scenario_actor_id and float(box["distance"]) < 15:
                return True
        return False

    def _check_red_overhead_traffic_light(self, sample: dict) -> bool:
        """Check if sample has red overhead traffic light affecting ego vehicle while stopped."""
        if not sample["light_hazard"]:
            return False
        if float(sample["speed"]) >= 0.1:
            return False

        boxes = sample["boxes"]
        for box in boxes:
            if (
                box["class"] == "traffic_light"
                and box["state"] == "Red"
                and box["affects_ego"]
                and box["is_over_head_traffic_light"]
            ):
                return True
        return False

    def _check_red_europe_traffic_light(self, sample: dict) -> bool:
        """Check if sample has red Europe traffic light affecting ego vehicle while stopped."""
        if not sample["light_hazard"]:
            return False
        if float(sample["speed"]) >= 0.1:
            return False

        boxes = sample["boxes"]
        for box in boxes:
            if (
                box["class"] == "traffic_light"
                and box["state"] == "Red"
                and box["affects_ego"]
                and box["is_europe_traffic_light"]
            ):
                return True
        return False

    def _check_large_lateral_deviation(self, sample: dict) -> bool:
        """Check if any future position has large lateral deviation."""
        route = sample["route"]
        for point in route[:20]:
            if abs(point[1]) > 6.5:
                return True
        return False

    def cache_file_path(self):
        """Return path for cache file"""
        return os.path.join(
            self.config.bucket_collection_path,
            f"waymo_{self.config.num_way_points_prediction}_{self.config.skip_first}_{self.config.skip_last}_{self.config.waypoints_spacing}.gz",
        )

    def buckets_mixture_per_epoch(self, _):
        """
        Define sampling ratios for each bucket depending on epoch. Current implementation uses fixed ratios across all epochs.
        """
        mixture = {bucket: 1.0 for bucket in range(len(self.buckets))}

        mixture[WaymoSim2RealBuckets.ENTER_ACTOR_FLOW_SCENARIO] = 0.0  # Useless
        mixture[WaymoSim2RealBuckets.ENTER_ACTOR_FLOW_V2_SCENARIO] = 0.0  # Useless
        mixture[WaymoSim2RealBuckets.HAZARD_AT_SIDE_LANE_TWO_WAYS_SCENARIO] = (
            0.0  # Useless
        )
        mixture[WaymoSim2RealBuckets.INTERURBAN_ACTOR_FLOW_SCENARIO] = 0.0  # Useless
        mixture[WaymoSim2RealBuckets.MERGER_INTO_SLOW_TRAFFIC_SCENARIO] = 0.0  # Useless
        mixture[WaymoSim2RealBuckets.MERGER_INTO_SLOW_TRAFFIC_V2_SCENARIO] = (
            0.0  # Useless
        )
        mixture[WaymoSim2RealBuckets.HAZARD_AT_SIDE_LANE_SCENARIO] = 0.0
        mixture[WaymoSim2RealBuckets.CROSSING_BICYCLE_FLOW_SCENARIO] = 0.0
        mixture[WaymoSim2RealBuckets.HIGHWAY_CUT_IN_SCENARIO] = 0.0
        mixture[WaymoSim2RealBuckets.PARKING_EXIT_SCENARIO] = 0.0
        mixture[WaymoSim2RealBuckets.VEHICLE_TURNING_ROUTE_SCENARIO] = 0.0
        mixture[WaymoSim2RealBuckets.FAR_TARGET_POINT] = 0.0

        mixture[WaymoSim2RealBuckets.NON_SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO] = 3.0
        mixture[WaymoSim2RealBuckets.PARKED_OBSTACLE_SCENARIO] = 3.0
        mixture[WaymoSim2RealBuckets.INVADING_TURN_SCENARIO] = 3.0
        mixture[WaymoSim2RealBuckets.ACCIDENT_SCENARIO] = 1.0
        mixture[WaymoSim2RealBuckets.CONSTRUCTION_OBSTACLE_SCENARIO] = 1.5
        mixture[WaymoSim2RealBuckets.RED_OVERHEAD_TRAFFIC_LIGHT] = 3.0
        mixture[WaymoSim2RealBuckets.CROSS_JUNCTION_DEFECT_TRAFFIC_LIGHT_SCENARIO] = 3.0
        mixture[WaymoSim2RealBuckets.NON_SIGNALIZED_JUNCTION_RIGHT_TURN_SCENARIO] = 3.0
        mixture[WaymoSim2RealBuckets.VEHICLE_TURNING_ROUTE_PEDESTRIAN_SCENARIO] = 3.0
        mixture[WaymoSim2RealBuckets.VEHICLE_OPENS_DOOR_TWO_WAYS_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.ENTERING_JUNCTION] = 4.0
        mixture[WaymoSim2RealBuckets.MEDIUM_ROUTE_CURVATURE] = 1.0
        mixture[WaymoSim2RealBuckets.STOP_SIGN_HAZARD] = 3.0
        mixture[WaymoSim2RealBuckets.PARKING_CUT_IN_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.STATIC_CUT_IN_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.CURRENT_TARGET_POINT_BEHIND_EGO] = 5.0
        mixture[WaymoSim2RealBuckets.RED_LIGHT_WITHOUT_LEAD_VEHICLE_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO] = 3.0
        mixture[
            WaymoSim2RealBuckets.SIGNALIZED_JUNCTION_LEFT_TURN_ENTER_FLOW_SCENARIO
        ] = 3.0
        mixture[WaymoSim2RealBuckets.SIGNALIZED_JUNCTION_RIGHT_TURN_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.VEHICLE_TURNING_ROUTE_PEDESTRIAN_SCENARIO] = 2.5
        mixture[WaymoSim2RealBuckets.RED_LIGHT_WITHOUT_LEAD_VEHICLE_SCENARIO] = 2.0
        mixture[WaymoSim2RealBuckets.OPPOSITE_VEHICLE_RUNNING_RED_LIGHT_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.OPPOSITE_VEHICLE_TAKING_PRIORITY_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.PEDESTRIAN_CROSSING_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.PRIORITY_AT_JUNCTION_SCENARIO] = 5.0
        mixture[WaymoSim2RealBuckets.OCCLUDED_JUNCTION] = 3.0
        mixture[WaymoSim2RealBuckets.RED_OVERHEAD_TRAFFIC_LIGHT] = (
            0.5  # A bit awkward with NavSim calibration
        )

        mixture[WaymoSim2RealBuckets.TOWN15] = 0.6
        mixture[WaymoSim2RealBuckets.VEHICLE_HAZARD] = 0.2
        mixture[WaymoSim2RealBuckets.ACCIDENT_TWO_WAYS_SCENARIO] = 0.2
        mixture[WaymoSim2RealBuckets.CONSTRUCTION_OBSTACLE_TWO_WAYS_SCENARIO] = 0.1
        mixture[WaymoSim2RealBuckets.CLOSE_TO_JUNCTION] = 1.0

        # Downsample uninteresting samples significantly
        mixture[WaymoSim2RealBuckets.OTHERS] = 0.1

        return mixture

    def print_statistic(self):
        print("\nBucket Statistics:")
        print("=" * 80)
        for i, bucket in enumerate(self.buckets):
            bucket_name = WaymoSim2RealBuckets.member_at(i).name
            print(f"Bucket {i:2d} - {bucket_name:60s}: {len(bucket.images):6d} samples")

        print("\n" + "=" * 80)
        print("Sampling Ratios:")
        print("=" * 80)
        for i, ratio in self.buckets_mixture_per_epoch(0).items():
            bucket_name = WaymoSim2RealBuckets.member_at(i).name
            print(
                f"Bucket {i:2d} - {bucket_name:60s}: {ratio:.2f} {int(ratio * len(self.buckets[i]))}",
            )


if __name__ == "__main__":
    config = TrainingConfig()
    # config.force_rebuild_bucket = True  # To avoid recursion, set to True for the first time. Subsequent runs can be False.
    config.use_carla_data = True
    config.use_waymo_e2e_data = True
    config.LTF = True
    config.use_planning_decoder = False
    config.carla_root = "data/carla_leaderboad2_v14/results/data/sensor_data/"
    collection = WaymoBucketCollection(root=config.carla_data, config=config)
    collection.print_statistic()

    config.force_rebuild_bucket = False
    from lead.data_loader.carla_dataset import CARLAData

    dataset = CARLAData(
        root=config.carla_data,
        config=config,
    )
    print(f"Dataset size with subsampling: {len(dataset)} samples")
