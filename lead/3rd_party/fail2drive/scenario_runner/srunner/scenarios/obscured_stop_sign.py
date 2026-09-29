import carla
import py_trees

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      ActorTransformSetter,
                                                                      ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               WaitEndIntersection)
from srunner.tools.background_manager import HandleJunctionScenario

def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

class ObscuredStopSign(BasicScenario):
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

        self._end_distance = 10
        self.prop_name = get_value_parameter(config, 'prop', str, 'static.prop.barrel')
        self.offset_y = get_value_parameter(config, 'offset_y', float, 0.0)
        self.offset_z = get_value_parameter(config, 'offset_z', float, 0.0)

        self.max_stop_dist = 50

        self.props = []

        super().__init__("ObscuredStopSign",
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

        world = CarlaDataProvider.get_world()
        stop_signs = world.get_actors().filter("traffic.stop")

        trigger_locations = [(stop, stop.get_transform().transform(stop.trigger_volume.location)) for stop in stop_signs]

        # Coarse filter
        trigger_locations = [x for x in trigger_locations if ego_location.distance(x[1]) <= self.max_stop_dist]

        # Find the nearest stop sign that affects a future waypoint
        starting_wp = self._ego_wp
        stop_sign = None
        while not starting_wp.is_junction and ego_location.distance(starting_wp.transform.location) <= self.max_stop_dist:
            for stop, stop_loc in trigger_locations:
                if starting_wp.transform.location.distance_2d(stop_loc) <= 1.5:
                    stop_sign = stop
                    break
            if stop_sign is not None:
                break

            starting_wps = starting_wp.next(1.0)
            if len(starting_wps) == 0:
                raise ValueError("Failed to find stop sign as a waypoint with no next was detected")
            starting_wp = starting_wps[0]

        if stop_sign is None:
            raise ValueError("Unable to find stop sign before next junction")

        stop_sign_loc = stop_sign.get_location()

        # Find the corresponding bounding box, this is a bit overcomplicated because the carla bounding box is the same as the trigger volume
        all_signs = [(x, stop_sign_loc.distance(x.location)) for x in world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)]
        stop_bb, stop_dist = min(all_signs, key=lambda l: l[1])
        if stop_dist > 10:
            raise ValueError("Unable to find corresponding traffic sign for stop line")
        # for stop_bb, _ in all_signs:
        transform = carla.Transform(location=stop_bb.location, rotation=stop_bb.rotation)

        # transform = carla.Transform(location = stop_sign.location, rotation=stop_sign.rotation)
        transform.rotation.roll += 90
        transform.location += transform.rotation.get_right_vector() * -self.offset_z
        transform.location += transform.rotation.get_forward_vector() * self.offset_y
        transform.location += transform.rotation.get_up_vector() * 0.03

        spawn_tf = carla.Transform(transform.location, transform.rotation)
        # spawn_tf.location.z -= 200
        if self.prop_name != "none":
            static = CarlaDataProvider.request_new_actor(self.prop_name, spawn_tf)
            static.set_simulate_physics(False)

            self.props.append([static, transform])

    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="ObscuredStopSign")
        for actor, transform in self.props:
            root.add_child(ActorTransformSetter(actor, transform, False))
        root.add_child(HandleJunctionScenario(clear_junction=True, clear_ego_entry=True, stop_entries=True))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(py_trees.composites.Sequence(children=[WaitEndIntersection(self.ego_vehicles[0]), DriveDistance(self.ego_vehicles[0], self._end_distance)]))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name)) # TODO idk mit dem timeout!!!
        root.add_child(end_condition)
        for actor, transform in self.props:
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
