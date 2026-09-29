#!/usr/bin/env python3
"""
Stand-alone wrapper for debugging leaderboard.
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path

from lead.common.logging_config import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)


class LeaderboardType(Enum):
    """Type of leaderboard to use."""

    STANDARD = "standard"
    BENCH2DRIVE = "bench2drive"
    FAIL2DRIVE = "fail2drive"
    AUTOPILOT = "autopilot"


# Mode-specific constants
class ModeConfig:
    """Configuration constants for different evaluation modes."""

    @staticmethod
    def get_mode_config(
        is_expert: bool,
        is_carl: bool,
        is_bench2drive: bool,
        is_fail2drive: bool,
        checkpoint: str | None,
        routes: str,
        use_py123d: bool = False,
        is_plant: bool = False,
    ) -> tuple[LeaderboardType, str, str, str | None, str]:
        """Get mode configuration based on CLI arguments.

        Args:
            is_expert: Whether expert mode is selected
            is_carl: Whether CaRL agent mode is selected
            is_bench2drive: Whether bench2drive variant is selected
            is_fail2drive: Whether fail2drive variant is selected
            checkpoint: Model checkpoint path (None for expert)
            routes: Routes file path
            use_py123d: Whether to use expert_py123d.py instead of expert.py
            is_plant: Whether PlanT agent mode is selected

        Returns:
            Tuple of (leaderboard_type, agent, agent_config, checkpoint_dir, track)
        """
        if is_expert:
            agent_file = (
                "lead/expert/expert_py123d.py"
                if use_py123d
                else "lead/expert/expert.py"
            )
            return (
                LeaderboardType.AUTOPILOT,
                agent_file,
                routes,
                None,
                "MAP",
            )

        def _resolve_leaderboard_type() -> LeaderboardType:
            if is_fail2drive:
                return LeaderboardType.FAIL2DRIVE
            if is_bench2drive:
                return LeaderboardType.BENCH2DRIVE
            return LeaderboardType.STANDARD

        if is_plant:
            return (
                _resolve_leaderboard_type(),
                "lead/plant/plant_agent.py",
                checkpoint,
                checkpoint,
                "MAP",
            )

        if is_carl:
            return (
                _resolve_leaderboard_type(),
                "lead/carl/carl_agent.py",
                checkpoint,
                checkpoint,
                "MAP",
            )

        return (
            _resolve_leaderboard_type(),
            "lead/inference/sensor_agent.py",
            checkpoint,
            checkpoint,
            "SENSORS",
        )


class LeaderboardWrapper:
    """Wrapper for running CARLA leaderboard evaluations.

    Provides a unified Python interface for executing different types of CARLA
    leaderboard evaluations (Standard, Bench2Drive, Autopilot) with both expert
    agents and trained models.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the leaderboard wrapper.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.routes = Path(args.routes)

        # Resolve workspace root from environment variable
        self.workspace_root = Path(os.environ["LEAD_PROJECT_ROOT"]).resolve()

        # Parse scenario type from routes XML file and extract route ID
        self.scenario_type = self._parse_scenario_type_from_routes()
        self.route_id = self.routes.stem.split("_")[0]

    def _parse_scenario_type_from_routes(self) -> str:
        """Parse scenario type from the first scenario in the routes XML file.

        Returns:
            Scenario type from first scenario element, or "noScenario" if none found
        """
        try:
            tree = ET.parse(self.routes)
            root = tree.getroot()

            # Find the first scenario element
            scenario_element = root.find(".//scenario")
            if scenario_element is not None:
                scenario_type = scenario_element.get("type")
                if scenario_type:
                    return scenario_type

            # No scenarios found or no type attribute
            return "noScenarios"

        except (ET.ParseError, FileNotFoundError) as e:
            LOG.warning(f"Could not parse routes file {self.routes}: {e}")
            return "noScenarios"

    def _get_leaderboard_evaluator_paths(self) -> dict:
        """Get paths to leaderboard evaluator components for subprocess execution. [Subprocess setup]

        Returns paths needed to locate and run the leaderboard evaluator:
        - Where to find the evaluator script
        - Where to find scenario runner dependencies
        - Where to find CARLA Python API (for AUTOPILOT mode)

        These paths are used to build PYTHONPATH and execute the evaluator subprocess.

        Returns:
            Dictionary containing paths:
            - leaderboard_root: Root directory of leaderboard code
            - scenario_runner_root: Root directory of scenario runner
            - evaluator_script: Path to main evaluator script
            - evaluator_module: Python module path (kept for compatibility)
            - carla_path: (AUTOPILOT only) Path to CARLA Python API
        """
        if self.leaderboard_type == LeaderboardType.BENCH2DRIVE:
            return {
                "leaderboard_root": self.workspace_root
                / "3rd_party/Bench2Drive/leaderboard",
                "scenario_runner_root": self.workspace_root
                / "3rd_party/Bench2Drive/scenario_runner",
                "evaluator_script": self.workspace_root
                / "3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
                "evaluator_module": "leaderboard.leaderboard_evaluator",
                "carla_path": self.workspace_root
                / "3rd_party/CARLA_0915/PythonAPI/carla",
            }
        elif self.leaderboard_type == LeaderboardType.FAIL2DRIVE:
            return {
                "leaderboard_root": self.workspace_root
                / "3rd_party/fail2drive/leaderboard",
                "scenario_runner_root": self.workspace_root
                / "3rd_party/fail2drive/scenario_runner",
                "evaluator_script": self.workspace_root
                / "3rd_party/fail2drive/leaderboard/leaderboard/leaderboard_evaluator.py",
                "evaluator_module": "leaderboard.leaderboard_evaluator",
                "carla_path": self.workspace_root
                / "3rd_party/CARLA_F2D/PythonAPI/carla",
            }
        elif self.leaderboard_type == LeaderboardType.AUTOPILOT:
            return {
                "leaderboard_root": self.workspace_root
                / "3rd_party/leaderboard_autopilot",
                "scenario_runner_root": self.workspace_root
                / "3rd_party/scenario_runner_autopilot",
                "evaluator_script": self.workspace_root
                / "3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py",
                "evaluator_module": "leaderboard.leaderboard_evaluator_local",
                "carla_path": self.workspace_root
                / "3rd_party/CARLA_0915/PythonAPI/carla",
            }
        else:  # STANDARD
            return {
                "leaderboard_root": self.workspace_root / "3rd_party/leaderboard",
                "scenario_runner_root": self.workspace_root
                / "3rd_party/scenario_runner",
                "evaluator_script": self.workspace_root
                / "3rd_party/leaderboard/leaderboard/leaderboard_evaluator.py",
                "evaluator_module": "leaderboard.leaderboard_evaluator",
            }

    def _build_pythonpath(self, paths: dict) -> str:
        """Build PYTHONPATH string from leaderboard paths for subprocess environment.

        Constructs complete PYTHONPATH by combining:
        1. CARLA Python API (AUTOPILOT mode only)
        2. Leaderboard root directory
        3. Scenario runner root directory
        4. Existing PYTHONPATH from environment (preserved)

        Order matters: CARLA API first to ensure correct imports in AUTOPILOT mode.

        Args:
            paths: Dictionary of leaderboard paths from get_leaderboard_evaluator_paths()
                Must contain 'leaderboard_root' and 'scenario_runner_root' keys.
                May contain 'carla_path' for AUTOPILOT mode.

        Returns:
            Colon-separated PYTHONPATH string ready for subprocess environment
        """
        pythonpath_parts = [
            str(paths["leaderboard_root"]),
            str(paths["scenario_runner_root"]),
        ]
        if "carla_path" in paths:
            pythonpath_parts.insert(0, str(paths["carla_path"]))

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)

        return ":".join(pythonpath_parts)

    def _determine_evaluation_output_dir(self, output_dir: Path | None) -> Path:
        """Determine where to save evaluation results. [Main process logic]

        Args:
            output_dir: Explicitly provided output directory (takes precedence)

        Returns:
            Resolved output directory path
        """
        if output_dir is not None:
            return output_dir

        if self.args.expert:
            # Expert evaluation: debug directory
            return self.workspace_root / "outputs/expert_evaluation/"
        else:
            # Model evaluation: organize by scenario and route
            return self.workspace_root / f"outputs/local_evaluation/{self.route_id}"

    def _setup_leaderboard_environment(
        self,
        root_output_dir: Path | None = None,
        checkpoint_dir: str | None = None,
    ) -> dict:
        """Setup environment variables for leaderboard evaluator subprocess.

        Args:
            root_output_dir: User-provided root output directory (uses auto-detection if None)
            checkpoint_dir: Model checkpoint directory (None for expert mode)

        Returns:
            Dictionary of environment variables that were set in os.environ
        """
        paths = self._get_leaderboard_evaluator_paths()
        resolved_output_dir = self._determine_evaluation_output_dir(root_output_dir)

        # Build environment variables
        env_vars = {
            "PYTHONPATH": self._build_pythonpath(paths),
            "SCENARIO_RUNNER_ROOT": str(paths["scenario_runner_root"]),
            "LEADERBOARD_ROOT": str(paths["leaderboard_root"]),
            "ROUTES": str(self.routes.absolute()),
            "SCENARIO_TYPE": self.scenario_type,
            "BENCHMARK_ROUTE_ID": self.route_id,
            "ROUTE_NUMBER": self.route_id,
            "PYTHONUNBUFFERED": "1",
            "IS_BENCH2DRIVE": "1"
            if self.leaderboard_type == LeaderboardType.BENCH2DRIVE
            else "0",
            "OUTPUT_DIR": str(resolved_output_dir),
            "EVALUATION_OUTPUT_DIR": str(resolved_output_dir),
        }

        # Add agent mode specific variables
        if self.args.expert:
            env_vars.update(
                {
                    "SAVE_PATH": str(resolved_output_dir / "data" / self.scenario_type),
                    "DATAGEN": "1",
                    "DEBUG_CHALLENGE": "0",
                    "TEAM_CONFIG": str(self.routes.absolute()),
                },
            )
        else:
            save_path = resolved_output_dir
            env_vars.update(
                {
                    "CHECKPOINT_DIR": checkpoint_dir,
                    "SAVE_PATH": str(save_path),
                },
            )
            if getattr(self.args, "carl", False):
                # Required by PyTorch deterministic algorithms when using CuBLAS.
                env_vars["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG",
                    ":4096:8",
                )

        # Apply to os.environ
        for key, value in env_vars.items():
            os.environ[key] = value

        return env_vars

    def _prepare_checkpoint_paths(self, output_path: Path) -> tuple[Path, Path]:
        """Create checkpoint directories and return checkpoint file paths.

        Creates two types of checkpoint files for evaluation state management:
        1. checkpoint_endpoint.json: Main evaluation checkpoint for resume functionality
        2. debug_checkpoint_endpoint.txt: Debug checkpoint for detailed tracking

        Both directories are created automatically if they don't exist.

        Args:
            output_path: Base output directory for evaluation results

        Returns:
            Tuple of (checkpoint_path, debug_checkpoint_path) as Path objects

        Note:
            This method is currently unused as the logic was inlined in run()
            for better code organization. Consider removing if not needed elsewhere.
        """
        checkpoint_path = output_path / "checkpoint_endpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        debug_checkpoint_path = (
            output_path / "debug_checkpoint/debug_checkpoint_endpoint.txt"
        )
        debug_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        return checkpoint_path, debug_checkpoint_path

    def run(self) -> subprocess.CompletedProcess:
        """Execute CARLA leaderboard evaluation as subprocess.

        Main execution pipeline that:
        1. Determines evaluation mode (expert/model) and leaderboard type
        2. Sets up environment variables and output directories
        3. Builds command with all required arguments
        4. Executes leaderboard evaluator as subprocess
        5. Handles graceful shutdown on interruption

        Returns:
            subprocess.CompletedProcess: Result of leaderboard evaluation
                - returncode 0: Success
                - returncode != 0: Error during evaluation
                - KeyboardInterrupt: Graceful shutdown initiated

        Raises:
            SystemExit: On subprocess errors or keyboard interrupt
        """
        # Get mode configuration
        leaderboard_type, agent, agent_config, checkpoint_dir, track = (
            ModeConfig.get_mode_config(
                is_expert=self.args.expert,
                is_carl=self.args.carl,
                is_bench2drive=self.args.bench2drive,
                is_fail2drive=self.args.fail2drive,
                checkpoint=self.args.checkpoint,
                routes=str(self.routes),
                use_py123d=self.args.py123d if hasattr(self.args, "py123d") else False,
                is_plant=getattr(self.args, "plant", False),
            )
        )
        self.leaderboard_type = leaderboard_type

        # Setup environment
        root_output_dir = Path(self.args.output_dir) if self.args.output_dir else None
        env_vars = self._setup_leaderboard_environment(root_output_dir, checkpoint_dir)
        resolved_output_path = Path(env_vars["OUTPUT_DIR"])
        paths = self._get_leaderboard_evaluator_paths()

        env = os.environ.copy()
        env.update(env_vars)

        # Clean output directory (skip if resuming)
        if not self.args.resume and resolved_output_path.exists():
            LOG.info(f"Removing existing output directory: {resolved_output_path}")
            shutil.rmtree(resolved_output_path)

        # Build command directly
        checkpoint_path = resolved_output_path / "checkpoint_endpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        debug_checkpoint_path = (
            resolved_output_path / "debug_checkpoint/debug_checkpoint_endpoint.txt"
        )
        debug_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(paths["evaluator_script"]),
            "--routes",
            str(self.routes.absolute()),
            "--track",
            track,
            "--checkpoint",
            str(checkpoint_path),
            "--agent",
            str(self.workspace_root / agent),
            "--agent-config",
            agent_config or "",
            "--debug",
            str(self.args.debug),
            "--resume",
            str(int(self.args.resume)),
            "--port",
            str(self.args.port),
            "--traffic-manager-port",
            str(self.args.traffic_manager_port),
            "--traffic-manager-seed",
            str(self.args.traffic_manager_seed),
            "--repetitions",
            str(self.args.repetitions),
            "--timeout",
            str(self.args.timeout),
        ]

        # Add debug checkpoint if not autopilot
        if leaderboard_type != LeaderboardType.AUTOPILOT:
            cmd.extend(
                ["--debug-checkpoint", str(debug_checkpoint_path), "--record", "None"],
            )

        LOG.info("\n" + "=" * 80)
        LOG.info(
            f"Starting CARLA Leaderboard Evaluation ({self.leaderboard_type.value})",
        )
        LOG.info(f"Command: {' '.join(cmd)}")
        LOG.info("=" * 80)
        LOG.info(f"Routes: {self.routes}")
        LOG.info(f"Scenario Type: {self.scenario_type}")
        LOG.info(f"Route ID: {self.route_id}")
        LOG.info(f"Output Dir: {resolved_output_path}")
        for key, value in env_vars.items():
            LOG.info(f"{key}: {value}")
        LOG.info("=" * 80 + "\n")

        # Use Popen for better process control
        process = None
        try:
            process = subprocess.Popen(cmd, cwd=self.workspace_root, env=env)
            returncode = process.wait()
            return subprocess.CompletedProcess(cmd, returncode)

        except KeyboardInterrupt:
            LOG.info("\n" + "=" * 80)
            LOG.info("Received CTRL+C - initiating graceful shutdown...")
            LOG.info("=" * 80)

            if process:
                # Send SIGINT to subprocess to allow graceful cleanup
                try:
                    LOG.info("Sending interrupt signal to subprocess...")
                    process.send_signal(signal.SIGINT)

                    # Wait up to 30 seconds for graceful shutdown
                    LOG.info("Waiting for subprocess to clean up (max 30s)...")
                    for i in range(30):
                        if process.poll() is not None:
                            LOG.info(f"Subprocess exited cleanly after {i + 1} seconds")
                            break
                        time.sleep(1)
                    else:
                        # If still running after timeout, send SIGTERM
                        LOG.warning(
                            "Subprocess did not exit after 30s, sending SIGTERM...",
                        )
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                            LOG.info("Subprocess terminated successfully")
                        except subprocess.TimeoutExpired:
                            # Last resort: force kill
                            LOG.error("Subprocess did not terminate, forcing kill...")
                            process.kill()
                            process.wait()

                except Exception as e:
                    LOG.error(f"Error during cleanup: {e}")
                    if process and process.poll() is None:
                        process.kill()

            LOG.info("=" * 80)
            LOG.info("Shutdown complete")
            LOG.info("=" * 80)
            sys.exit(130)  # Standard exit code for SIGINT

        finally:
            # Ensure process is cleaned up
            if process and process.poll() is None:
                try:
                    process.kill()
                    process.wait()
                except:
                    pass


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure CLI argument parser with all options.

    Sets up argument parser with:
    - Mode selection (--checkpoint vs --expert)
    - Required arguments (--routes)
    - Leaderboard type (--bench2drive)
    - CARLA connection settings (ports)
    - Evaluation settings (repetitions, timeout, resume, debug)
    - Model-specific settings (gpu)
    - Output control (output-dir)

    Returns:
        Configured argument parser with usage examples
    """
    parser = argparse.ArgumentParser(
        description="Run CARLA Leaderboard Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Evaluate model on Town13
  python -m lead --checkpoint outputs/checkpoints/tfv6_resnet34 --routes data/benchmark_routes/Town13/0.xml

  # Evaluate model on Bench2Drive
  python -m lead --checkpoint outputs/checkpoints/tfv6_resnet34 --routes data/benchmark_routes/bench2drive/23687.xml --bench2drive

    # Evaluate CaRL agent on one route
    python -m lead --checkpoint outputs/checkpoints/CaRL --routes data/benchmark_routes/bench2drive/24240.xml --carl --timeout 900

  # Evaluate expert agent
  python -m lead --expert --routes data/benchmark_routes/Town13/1.xml

  # Evaluate expert for data generation
  python -m lead --expert --routes data/data_routes/lead/noScenarios/short_route.xml
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint directory (for model evaluation)",
    )
    mode_group.add_argument(
        "--expert",
        action="store_true",
        help="Run expert agent (for expert evaluation)",
    )

    # Required arguments
    parser.add_argument(
        "--routes",
        type=str,
        required=True,
        help="Path to the routes XML file",
    )

    # Leaderboard type
    parser.add_argument(
        "--bench2drive",
        action="store_true",
        help="Use Bench2Drive leaderboard",
    )
    parser.add_argument(
        "--fail2drive",
        action="store_true",
        help="Use Fail2Drive leaderboard (requires CARLA_F2D simulator)",
    )
    parser.add_argument(
        "--carl",
        action="store_true",
        help="Use lead/carl/carl_agent.py with MAP track (requires --checkpoint)",
    )
    parser.add_argument(
        "--plant",
        action="store_true",
        help="Use lead/plant/plant_agent.py with MAP track (requires --checkpoint)",
    )

    # Py123D option for expert mode
    parser.add_argument(
        "--py123d",
        action="store_true",
        help="Use expert_py123d.py (saves data in Py123D Arrow format). Only works with --expert.",
    )

    # CARLA settings
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=8000,
        help="Traffic manager port",
    )
    parser.add_argument(
        "--traffic-manager-seed",
        type=int,
        default=0,
        help="Traffic manager seed",
    )

    # Evaluation settings
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per route",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint",
    )
    parser.add_argument("--debug", type=int, default=0, help="Debug mode")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (for model evaluation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (auto-generated if not specified)",
    )

    return parser


def main() -> None:
    """CLI interface for running CARLA leaderboard evaluations.

    Command-line entry point that:
    1. Parses CLI arguments
    2. Sets GPU device for model evaluation
    3. Configures evaluation mode (expert vs model)
    4. Creates LeaderboardWrapper instance
    5. Runs evaluation with specified parameters

    Exit codes:
        0: Success
        1: Error during evaluation
        130: Keyboard interrupt (Ctrl+C)

    Examples:
        Model evaluation:
            $ python -m lead \\
                --checkpoint outputs/checkpoints/model \\
                --routes data/routes/town01.xml

        Expert data generation:
            $ python -m lead \\
                --expert \\
                --routes data/routes/short_route.xml

        Expert Py123D data generation:
            $ python -m lead \\
                --expert \\
                --py123d \\
                --routes data/routes/short_route.xml

        Bench2Drive evaluation:
            $ python -m lead \\
                --checkpoint outputs/checkpoints/model \\
                --routes data/routes/bench2drive.xml \\
                --bench2drive
    """

    parser = _create_argument_parser()
    args = parser.parse_args()

    # Validate py123d flag
    if args.py123d and not args.expert:
        LOG.warning(
            "--py123d flag is only valid with --expert mode. Ignoring --py123d flag.",
        )

    # Validate carl / plant mode flags
    if args.carl and not args.checkpoint:
        parser.error("--carl requires --checkpoint")
    if args.plant and not args.checkpoint:
        parser.error("--plant requires --checkpoint")

    # Set GPU for model evaluation
    if args.checkpoint and not args.carl:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Log mode information
    if args.expert:
        agent_type = "ExpertPy123D" if args.py123d else "Expert"
        LOG.info(f"Running in expert mode with {agent_type} agent")
    elif args.carl:
        LOG.info("Running in CaRL evaluation mode")

    # Create wrapper and run
    result = LeaderboardWrapper(args).run()
    if result.returncode != 0:
        LOG.error(
            f"Leaderboard evaluator exited with non-zero code {result.returncode}",
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
