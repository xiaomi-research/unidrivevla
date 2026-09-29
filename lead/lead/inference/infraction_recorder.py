import json
import logging
from typing import Any

from lead.common.logging_config import setup_logging
from lead.inference.config_closed_loop import ClosedLoopConfig

setup_logging()
LOG = logging.getLogger(__name__)


class InfractionRecorder:
    """Track and persist scenario infractions during closed-loop rollout.

    This class mirrors scenario_runner behavior for discrete and continuous
    infractions while providing a consistent JSON artifact for downstream
    analysis.
    """

    def __init__(
        self,
        config_closed_loop: ClosedLoopConfig,
        agent_name: str,
    ) -> None:
        """Initialise recorder state.

        Args:
            config_closed_loop: Closed-loop config used for output settings.
            agent_name: Agent identifier used in log messages.
        """
        self.config_closed_loop = config_closed_loop
        self.agent_name = agent_name

        self.scenario: Any | None = None
        self.infractions_log: list[dict[str, Any]] = []
        self.tracked_infraction_ids: set[str | tuple[str, int]] = set()

    def set_scenario(self, scenario: Any) -> None:
        """Set scenario object used to fetch criteria events.

        Args:
            scenario: Scenario instance that exposes get_criteria().
        """
        self.scenario = scenario

    def check_infractions(self, step: int, meters_travelled: float) -> None:
        """Scan scenario criteria and append newly observed infractions.

        Args:
            step: Current rollout step.
            meters_travelled: Distance travelled by ego vehicle in meters.
        """
        if self.scenario is None:
            return

        try:
            criteria = self.scenario.get_criteria()

            for criterion in criteria:
                criterion_key = str(
                    getattr(criterion, "name", type(criterion).__name__),
                )
                events = getattr(criterion, "events", None)

                if events:
                    is_continuous = len(events) == 1

                    for event in events:
                        frame = (
                            int(event.get_frame())
                            if hasattr(event, "get_frame")
                            else -1
                        )
                        event_id = (
                            criterion_key if is_continuous else (criterion_key, frame)
                        )

                        if event_id in self.tracked_infraction_ids:
                            continue

                        self.tracked_infraction_ids.add(event_id)
                        infraction_info: dict[str, Any] = {
                            "step": step,
                            "infraction": criterion_key,
                            "frame": frame,
                            "message": (
                                event.get_message()
                                if hasattr(event, "get_message")
                                else ""
                            ),
                            "event_type": (
                                str(event.get_type())
                                if hasattr(event, "get_type")
                                else ""
                            ),
                            "meters_travelled": round(meters_travelled, 2),
                        }
                        self.infractions_log.append(infraction_info)
                        LOG.info(
                            "[%s] Infraction detected at step %d: %s",
                            self.agent_name,
                            step,
                            criterion_key,
                        )

                if not events and criterion_key in self.tracked_infraction_ids:
                    self.tracked_infraction_ids.discard(criterion_key)

            self._persist_to_json()
        except Exception as exc:
            LOG.warning("[%s] Error checking infractions: %s", self.agent_name, exc)

    def _persist_to_json(self) -> None:
        """Persist infractions to disk when output path is configured."""
        if self.config_closed_loop.save_path is None:
            return

        infractions_path = self.config_closed_loop.save_path / "infractions.json"
        infractions_data: dict[str, Any] = {
            "infractions": self.infractions_log,
            "video_fps": self.config_closed_loop.video_fps,
        }
        with open(infractions_path, "w", encoding="utf-8") as handle:
            json.dump(infractions_data, handle, indent=4)

        LOG.debug(
            "[%s] Saved %d infractions to %s",
            self.agent_name,
            len(self.infractions_log),
            infractions_path,
        )
