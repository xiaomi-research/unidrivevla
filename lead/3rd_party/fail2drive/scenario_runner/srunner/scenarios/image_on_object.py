import carla
import py_trees

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      ActorTransformSetter,
                                                                      ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance)

def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

class ImageOnObject(BasicScenario):
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

        self.timeout = timeout
        self._scenario_timeout = 240

        self.spawn_distance = get_value_parameter(config, 'distance', float, 10)
        self.drive_distance = self.spawn_distance + 10

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        self._reference_waypoint = self._reference_waypoint.next(self.spawn_distance)[0]

        spawn_wp = self._reference_waypoint

        while spawn_wp.lane_type != carla.LaneType.Sidewalk:
            right = spawn_wp.get_right_lane()
            if right is None:
                break
            spawn_wp = right

        self.spawn_transform = carla.Transform(spawn_wp.transform.location, self._reference_waypoint.transform.rotation)


        if spawn_wp != carla.LaneType.Sidewalk:
            self.spawn_transform.location += self.spawn_transform.rotation.get_right_vector() * 0.7

        self.prop_type = get_value_parameter(config, 'prop', str, 'static.prop.advertisement')
        self.image_type = get_value_parameter(config, 'image', str, 'static.prop.barrel')

        self.offset_x = get_value_parameter(config, 'offset_x', float, 0.0)
        self.offset_y = get_value_parameter(config, 'offset_y', float, 0.0)
        self.offset_z = get_value_parameter(config, 'offset_z', float, 0.0)

        self.props = []

        super().__init__("ImageOnObject",
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

        spawn_transform = carla.Transform(self.spawn_transform.location, self.spawn_transform.rotation) 
        if self.prop_type == "static.prop.busstop":
            spawn_transform.rotation.yaw += 180
            x_shift = 1.7
        else:
            spawn_transform.rotation.yaw -= 90
            x_shift = 0.15

        adversary = CarlaDataProvider.request_new_actor(self.prop_type, spawn_transform)
        if adversary is None:
            raise ValueError("Couldn't spawn adversary")
        spawn_transform = adversary.get_transform()

        adversary.set_simulate_physics(False)
        adversary.set_location(spawn_transform.location + carla.Location(z=-200))
        self.props.append((adversary, spawn_transform))

        spawn_2 = carla.Transform(spawn_transform.transform(adversary.bounding_box.location), self.spawn_transform.rotation)
        spawn_2.rotation.yaw += 90
        spawn_2.rotation.roll += 90

        forward = self.spawn_transform.rotation.get_forward_vector()
        right = self.spawn_transform.rotation.get_right_vector()
        up = self.spawn_transform.rotation.get_up_vector()

        spawn_2.location += forward * (self.offset_x - x_shift)
        spawn_2.location += right * self.offset_y
        spawn_2.location += up * self.offset_z

        if self.image_type == "none":
            return

        image = CarlaDataProvider.request_new_actor(self.image_type, spawn_2)
        if image is None:
            raise Exception("Couldn't spawn the image")

        image.set_simulate_physics(False)
        image.set_location(spawn_2.location + carla.Location(z=-200))

        self.props.append((image, spawn_2))

    def _create_behavior(self):
        sequence = py_trees.composites.Sequence(name="FakeStopSignPedestrian")
        for prop, transform in self.props:
            sequence.add_child(ActorTransformSetter(prop, transform, False))

        end_condition = py_trees.composites.Parallel(name="Endcondition", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self.drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        sequence.add_child(end_condition)
        for prop, transform in self.props:
            sequence.add_child(ActorDestroy(prop, name="DestroyProp"))
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria
