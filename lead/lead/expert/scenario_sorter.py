"""Efficient scenario sorter that sorts scenarios by minimum euclidean distance to ego vehicle."""

import logging

import carla
from beartype import beartype
from srunner.scenariomanager.carla_data_provider import (
    ActiveScenario,
    CarlaDataProvider,
)

LOG = logging.getLogger(__name__)


class ScenarioSorter:
    """Sorts active scenarios by minimum euclidean distance of their actors to ego vehicle.

    For each active scenario, this sorter retrieves all actors from the scenario's memory
    and computes the euclidean distance to the ego vehicle. The minimum distance across
    all actors in a scenario is used as that scenario's distance for sorting.
    """

    @beartype
    def __init__(self):
        """Initialize the scenario sorter."""
        pass

    def _remove_ended_scenarios(self) -> None:
        """Remove scenarios that have ended (actors are no longer alive).

        This cleans up scenarios where the first_actor or last_actor is no longer alive,
        which typically indicates the scenario has completed or timed out.
        """
        active_scenarios = CarlaDataProvider.active_scenarios.copy()

        for scenario in active_scenarios:
            should_remove = False

            # Check if first actor is dead
            if scenario.first_actor is not None and not scenario.first_actor.is_alive:
                should_remove = True

            # Check if last actor exists and is dead
            if (
                not should_remove
                and scenario.last_actor is not None
                and not scenario.last_actor.is_alive
            ):
                should_remove = True

            if should_remove:
                CarlaDataProvider.remove_scenario(scenario)

    @beartype
    def _compute_scenario_distance(
        self,
        scenario: ActiveScenario,
        ego_location: carla.Location,
    ) -> float:
        """Compute minimum distance from scenario actors to ego vehicle.

        Args:
            scenario: The active scenario.
            ego_location: Current ego vehicle location.

        Returns:
            Minimum euclidean distance from any actor in the scenario to ego vehicle.
            Returns infinity if no actors found.
        """
        actors = scenario.get_actors_from_memory()

        if not actors:
            # No actors found, use a large distance
            return float("inf")

        min_distance = float("inf")
        for actor in actors:
            try:
                actor_location = actor.get_location()
                distance = ego_location.distance(actor_location)
                min_distance = min(min_distance, distance)
            except:
                # Skip actors that can't provide location
                continue

        return min_distance

    @beartype
    def sort_scenarios(self, ego_location: carla.Location) -> None:
        """Sort active scenarios by minimum distance of their actors to ego vehicle.

        For each scenario, retrieves all actors from memory and computes euclidean distance
        to ego vehicle. Scenarios are sorted by their minimum actor distance.

        Also removes scenarios that have ended (actors no longer alive).

        Args:
            ego_location: Current ego vehicle location.
        """
        # First, remove any scenarios that have ended
        self._remove_ended_scenarios()

        active_scenarios = CarlaDataProvider.active_scenarios

        if not active_scenarios:
            return

        # Compute distance for each scenario
        scenario_distances = []
        for scenario in active_scenarios:
            distance = self._compute_scenario_distance(scenario, ego_location)
            scenario_distances.append((scenario, distance))
            LOG.debug(
                f"Scenario '{scenario.name}' (id={scenario.scenario_id}): distance={distance:.2f}m",
            )

        # Sort by distance (ascending)
        scenario_distances.sort(key=lambda x: x[1])

        # Update the active scenarios list with sorted order
        CarlaDataProvider.active_scenarios = [
            scenario for scenario, _ in scenario_distances
        ]

        if scenario_distances:
            LOG.debug(
                f"Sorted {len(active_scenarios)} scenarios by actor distance. "
                f"Closest: {scenario_distances[0][0].name} ({scenario_distances[0][1]:.2f}m)",
            )
