import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      Idle, WaitForever,
                                                                      ActorTransformSetter,
                                                                      KeepVelocity)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               WaitUntilInFront)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, ChangeRoadBehavior

def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

class PedestriansOnRoad(BasicScenario):
    """
    Added the dangerous scene of ego vehicles driving on roads without sidewalks,
    with three bicycles encroaching on some roads in front.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._walker_duration = get_value_parameter(config, 'walker_duration', float, 20)
        self._walker_speed = get_value_parameter(config, 'walker_speed', float, 2)
        self._distance = get_value_parameter(config, 'distance', float, 20)
        self._pedestrians = config.other_parameters.get("pedestrians", {"a": "walker.pedestrian.*"}).values()
        self._end_distance = self._distance + self._walker_duration * self._walker_speed + 20

        self._scenario_timeout = 240

        self.spawn_transforms = []

        super().__init__("PedestriansOnRoad",
                         ego_vehicles,
                         config,
                         world,
                         randomize,
                         debug_mode,
                         criteria_enable=criteria_enable)

    # TODO: Pedestrian have an issue with large maps were setting them to dormant breaks them,
    # so all functions below are meant to patch it until the fix is done
    def _replace_walker(self, walker):
        """As the adversary is probably, replace it with another one"""
        type_id = walker.type_id
        walker.destroy()
        spawn_transform = self.ego_vehicles[0].get_transform()
        spawn_transform.location.z -= 50
        walker = CarlaDataProvider.request_new_actor(type_id, spawn_transform)
        if not walker:
            raise ValueError("Couldn't spawn the walker substitute")
        walker.set_simulate_physics(False)
        walker.set_location(spawn_transform.location + carla.Location(z=-50))
        return walker

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Spawn the first bicycle
        first_wp = self._starting_wp.next(self._distance)[0]
        offsets = [[0.5, 0.5], [0.0, 0.0], [1, -0.5]]
        for (offset_x, offset_y), walker in zip(offsets, self._pedestrians):
            spawn_transform = carla.Transform(first_wp.transform.location, first_wp.transform.rotation)
            spawn_transform.location += offset_x * spawn_transform.rotation.get_forward_vector()
            spawn_transform.location += offset_y * spawn_transform.rotation.get_right_vector()
            spawn_transform.location.z += 1

            walker_1 = CarlaDataProvider.request_new_actor(walker, spawn_transform)

            walker_1.set_location(spawn_transform.location + carla.Location(z=-200))
            walker_1 = self._replace_walker(walker_1)

            # Set its initial conditions
            walker_1.apply_control(carla.WalkerControl())
            self.other_actors.append(walker_1)
            self.spawn_transforms.append(spawn_transform)

    def _create_behavior(self):
        """
        Activate the bicycles and wait for the ego to be close-by before changing the side traffic.
        End condition is based on the ego behind in front of the bicycles, or timeout based.
        """
        root = py_trees.composites.Sequence(name="HazardAtSideLane")
        if self.route_mode:
            total_dist = self._distance + 50
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=total_dist))
            root.add_child(Idle(0.1))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # End condition
        end_condition = py_trees.composites.Sequence(name="End Condition")
        end_condition.add_child(WaitUntilInFront(self.ego_vehicles[0], self.other_actors[-1], check_distance=False))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        main_behavior.add_child(end_condition)

        # Or end after specific time
        main_behavior.add_child(Idle(self._walker_duration))

        for actor, spawn_tf in zip(self.other_actors, self.spawn_transforms):
            walker = py_trees.composites.Sequence(name="Walker behavior")
            walker.add_child(ActorTransformSetter(actor, spawn_tf, True))
            walker.add_child(KeepVelocity(actor, self._walker_speed, False, self._walker_duration - 5))
            walker.add_child(WaitForever())
            main_behavior.add_child(walker)

        root.add_child(main_behavior)
        if self.route_mode:
            root.add_child(ChangeRoadBehavior(extra_space=0))

        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria