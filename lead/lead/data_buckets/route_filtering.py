import logging
import os

import ujson
from beartype import beartype

LOG = logging.getLogger(__name__)


@beartype
def route_failed(route_path: str) -> bool:
    """Check if a route has failed or should be skipped.

    Args:
        route_path: Directory of the route.

    Returns:
        bool: True if the route has failed or should be skipped, False otherwise.
    """
    if not os.path.isfile(route_path + "/results.json"):
        LOG.info(f"\tSkipping route {route_path} due to missing results file.")
        return True

    if not os.path.isfile(route_path + "/metas/0000.pkl"):
        LOG.info(f"\tSkipping route {route_path} due to missing meta files right now.")
        return True

    # We skip data where the expert did not achieve perfect driving score (except for min speed infractions)
    with open(route_path + "/results.json", encoding="utf-8") as f:
        results_route = ujson.load(f)
    infraction = True
    if results_route["scores"]["score_composed"] == 100.0:
        infraction = False
    if results_route["scores"]["score_composed"] < 100.0 and results_route[
        "num_infractions"
    ] == len(results_route["infractions"]["min_speed_infractions"]):
        infraction = False
    if results_route["status"] == "Completed" and results_route["num_infractions"] == (
        len(results_route["infractions"]["min_speed_infractions"])
        + len(results_route["infractions"]["outside_route_lanes"])
    ):  # We ignore minor out of lanes if we still complete the route
        infraction = False
    if results_route["status"] == "Perfect":
        infraction = False

    agent_not_setup = results_route["status"] == "Failed - Agent couldn't be set up"
    failed = results_route["status"] == "Failed"
    simulation_crashed = results_route["status"] == "Failed - Simulation crashed"
    agent_crashed = results_route["status"] == "Failed - Agent crashed"

    if infraction or agent_not_setup or failed or simulation_crashed or agent_crashed:
        LOG.info(
            f"\tSkipping {route_path}: results {results_route['status']}, scores {results_route['scores']['score_composed']}.",
        )
        return True
    return False


@beartype
def route_not_finished(route_path: str) -> bool:
    if not os.path.isfile(route_path + "/results.json"):
        LOG.info(f"\tSkipping route {route_path} due to missing results file.")
        return True
    return False


@beartype
def route_completed_but_fail(route_path: str) -> bool:
    if not os.path.isfile(route_path + "/results.json"):
        LOG.info(f"\tSkipping route {route_path} due to missing results file.")
        return False

    if not os.path.isfile(route_path + "/metas/0000.pkl"):
        LOG.info(f"\tSkipping route {route_path} due to missing meta files right now.")
        return False

    with open(route_path + "/results.json", encoding="utf-8") as f:
        results_route = ujson.load(f)

    agent_not_setup = results_route["status"] == "Failed - Agent couldn't be set up"
    simulation_crashed = results_route["status"] == "Failed - Simulation crashed"
    agent_crashed = results_route["status"] == "Failed - Agent crashed"

    if agent_not_setup or simulation_crashed or agent_crashed:
        LOG.info(
            f"\tSkipping {route_path}: results {results_route['status']}, scores {results_route['scores']['score_composed']}.",
        )
        return False
    if not os.path.exists(route_path + "/metas"):
        LOG.info(f"\tSkipping route {route_path} due to missing metas data.")
        return False
    if results_route["scores"]["score_composed"] == 100.0:
        LOG.info(f"\tSkipping route {route_path} due to perfect score.")
        return False
    if (
        results_route["status"] == "Completed"
        and results_route["scores"]["score_composed"] < 100.0
        and results_route["num_infractions"]
        == len(results_route["infractions"]["min_speed_infractions"])
    ):
        LOG.info(f"\tSkipping route {route_path} due to only min speed infractions.")
        return False
    if results_route["status"] == "Completed" and results_route["num_infractions"] == (
        len(results_route["infractions"]["min_speed_infractions"])
        + len(results_route["infractions"]["outside_route_lanes"])
    ):
        LOG.info(
            f"\tSkipping route {route_path} due to only min speed and outside lane infractions.",
        )
        return False

    return True
