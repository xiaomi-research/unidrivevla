#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pedestrians crossing through the middle of the lane.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      WaitForever,
                                                                      ActorTransformSetter,
                                                                      ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

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

class PedestrianCrowd(BasicScenario):

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)
        self._rng = CarlaDataProvider.get_random_seed()

        self.pedestrians = []
        self.ais = []

        self._scenario_timeout = 240

        self._center = get_value_parameter(config, 'pedestrian_center', float, 40)
        self._radius = get_value_parameter(config, 'pedestrian_radius', float, 20)
        self._length = self._radius * 2
        self._num_pedestrians = get_value_parameter(config, 'pedestrians', int, 20)
        self._side = get_value_parameter(config, 'side', str, "right")

        lane = self._reference_waypoint.next(self._center)[0]

        if self._side.lower() == "right":
            i = 0
            while lane is not None and lane.lane_type != carla.LaneType.Sidewalk:
                lane = lane.get_right_lane()
                i += 1
                if i > 100:
                    raise ValueError("Failed to find sidewalk on the right")
        elif self._side.lower() == "left":
            i = 0
            lane_id = lane.lane_id
            while lane is not None and lane.lane_type != carla.LaneType.Sidewalk and (lane_id > 0) == (lane.lane_id > 0):
                lane = lane.get_left_lane()
                i += 1
                if i > 100:
                    raise ValueError("Failed to find sidewalk on the left")

            while lane is not None and lane.lane_type != carla.LaneType.Sidewalk:
                lane = lane.get_right_lane()
                i += 1
                if i > 100:
                    raise ValueError("Failed to find sidewalk on the left")
        else:
            raise ValueError("Unknown side: " + self._side)

        if lane is None:
            raise ValueError("Failed to find sidewalk")

        self.target_locs = []
        for _ in range(500):
            x = (self._rng.rand() - 0.5) * self._length
            y = self._rng.rand() * lane.lane_width - lane.lane_width/2
            y *= 0.8

            if self._side == "right":
                if x > 0:
                    wp = lane.next(x)[0]
                else:
                    wp = lane.previous(abs(x))[0]
            else:
                if x > 0:
                    wp = lane.previous(x)[0]
                else:
                    wp = lane.next(abs(x))[0]

            target_tf = wp.transform
            target_tf.location += target_tf.rotation.get_right_vector() * y
            self.target_locs.append(target_tf.location)

        self.spawn_point = lane

        super().__init__("PedestrianCrowd",
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

        for _ in range(self._num_pedestrians*2):
            if len(self.pedestrians) >= self._num_pedestrians:
                break

            x = (self._rng.rand() - 0.5) * self._length
            y = self._rng.rand() * self.spawn_point.lane_width - self.spawn_point.lane_width/2
            y *= 0.7
            yaw = self._rng.rand() * 360

            if self._side == "right":
                if x > 0:
                    wp = self.spawn_point.next(x)[0]
                else:
                    wp = self.spawn_point.previous(abs(x))[0]
            else:
                if x > 0:
                    wp = self.spawn_point.previous(x)[0]
                else:
                    wp = self.spawn_point.next(abs(x))[0]

            spawn_transform = wp.transform
            spawn_transform.location += spawn_transform.rotation.get_right_vector() * y
            spawn_transform.location.z += 0.6
            spawn_transform.rotation.yaw = yaw

            actor = CarlaDataProvider.request_new_actor('walker.pedestrian.*', spawn_transform)

            if actor is None:
                continue

            spawn_transform = actor.get_transform()
            actor.apply_control(carla.WalkerControl(speed=0))
            actor.set_simulate_physics(False)
            actor.set_location(spawn_transform.location + carla.Location(z=-200))
            actor = self._replace_walker(actor)

            self.pedestrians.append((actor, spawn_transform, self._rng.permutation(self.target_locs)))

    def _create_behavior(self):
        end_condition = py_trees.composites.Parallel(name="PedestrianCrowd", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        parallelais = py_trees.composites.Parallel(name="PedestrianAIs")
        parallelais.add_child(WaitForever())
        for actor, transform, locs in self.pedestrians: # locs could be used to make them move but it was very buggy
            parallelais.add_child(ActorTransformSetter(actor, transform, True))

        end_condition.add_child(parallelais)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._center + self._radius + 20)) # 10m after pedestrians

        sequence = py_trees.composites.Sequence()
        sequence.add_child(end_condition)
        for actor, transform, _ in self.pedestrians:
            sequence.add_child(ActorDestroy(actor))

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

    def _replace_walker(self, adversary):
        """As the adversary is probably, replace it with another one"""
        type_id = adversary.type_id
        adversary.destroy()
        spawn_transform = self.ego_vehicles[0].get_transform()
        spawn_transform.location.z -= 50
        adversary = CarlaDataProvider.request_new_actor(type_id, spawn_transform)
        if not adversary:
            raise ValueError("Couldn't spawn the walker substitute")
        adversary.set_simulate_physics(False)
        adversary.set_location(spawn_transform.location + carla.Location(z=-50))
        return adversary