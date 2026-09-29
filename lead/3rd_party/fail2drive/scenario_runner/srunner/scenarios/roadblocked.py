# Scenario to spawn either an image or some object that blocks the entire road with no traffic,
# The agent needs to wait for a specific time and then the obstacle despawns

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      ActorTransformSetter,
                                                                      Idle)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import (ChangeOppositeBehavior,
                                              ChangeRoadBehavior)


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

def get_interval_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]['from']),
            p_type(config.other_parameters[name]['to'])
        ]
    else:
        return default
    
class RoadBlocked(BasicScenario):
    """
    Vehicle turning left at junction scenario, with actors coming in the opposite direction.
    The ego has to react to them, safely crossing the opposite lane
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._rng = CarlaDataProvider.get_random_seed()

        self._distance = get_value_parameter(config, 'distance', float, 100)
        self._wait_time = get_value_parameter(config, 'wait', float, 60)

        self.obstacle_transforms = []

        super().__init__("RoadBlocked",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """
        ego_location = config.trigger_points[0].location
        self._ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_location)

        self._obstacle_wp = self._ego_wp.next(self._distance)[0]

        start_transform = self._obstacle_wp.transform

        statics = self.config.other_parameters.get("objects", {}).values()
        statics = [{y.split("=")[0]: y.split("=")[1] for y in x.split(" ")} for x in statics]

        for static in statics:
            prop = static.get("id")
            x = float(static.get("x", 0))
            y = float(static.get("y", 0))
            z = float(static.get("z", 0))
            yaw = float(static.get("yaw", 0))
            pitch = float(static.get("pitch", 0))
            roll = float(static.get("roll", 0))
            transform = carla.Transform(
                start_transform.location,
                start_transform.rotation)
            
            transform.location += x * transform.rotation.get_forward_vector()
            transform.location += y * transform.rotation.get_right_vector()
            transform.location += z * transform.rotation.get_up_vector()
            transform.rotation.yaw += yaw
            transform.rotation.pitch += pitch
            transform.rotation.roll += roll

            static = CarlaDataProvider.request_new_actor(prop, transform)
            transform = static.get_transform()

            if "vehicle" in prop:
                static.apply_control(carla.VehicleControl(hand_brake=True))

            static.set_simulate_physics(False)
            static.set_location(transform.location + carla.Location(z=-200))

            self.obstacle_transforms.append([static, transform])

        sortedactors = sorted(zip(self.obstacle_transforms, statics), key=lambda l: float(l[1].get("x", 0))) # sorted by x
        CarlaDataProvider.active_scenarios.append(("RoadBlocked", [sortedactors[0][0][0], sortedactors[-1][0][0], None, False, 1e9, 1e9, False])) # added

    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="RoadBlocked")
        if self.route_mode:
            # Remove all traffic:
            root.add_child(ChangeRoadBehavior(0,0,200,200))
            root.add_child(ChangeOppositeBehavior(active=False))


        for actor, transform in self.obstacle_transforms:
            root.add_child(ActorTransformSetter(actor, transform, True))

        root.add_child(Idle(self._wait_time))

        for actor, transform in self.obstacle_transforms:
            root.add_child(ActorDestroy(actor))
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        # criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        # if not self.route_mode:
        #     criteria.append(CollisionTest(self.ego_vehicles[0]))
        return [] # TODO ?