#!/usr/bin/env python

import argparse
import glob
import os
import re
import shlex
from pathlib import Path


def is_on_tcml():
    return os.getenv("TCML") is not None


def make_env_exports():
    """Export only shell-safe environment variables into generated scripts."""
    safe_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    skip_prefixes = (
        "BASH_FUNC_",
    )
    skip_names = {
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "GITHUB_TOKEN",
    }
    lines = []
    for key, value in os.environ.items():
        if key in skip_names or key.startswith(skip_prefixes):
            continue
        if not safe_name.match(key):
            continue
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines)


def make_bash(bash_file, route_file, route_id, args):
    env_exports = make_env_exports()

    # Select evaluator scripts based on expert mode
    if args.expert:
        bench2drive_evaluator = "3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_local.py"
        fail2drive_evaluator = "3rd_party/fail2drive/leaderboard/leaderboard/leaderboard_evaluator_local.py"
        standard_evaluator = (
            "3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py"
        )
    else:
        bench2drive_evaluator = (
            "3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py"
        )
        fail2drive_evaluator = (
            "3rd_party/fail2drive/leaderboard/leaderboard/leaderboard_evaluator.py"
        )
        standard_evaluator = (
            "3rd_party/leaderboard/leaderboard/leaderboard_evaluator.py"
        )

    template = f"""#!/bin/bash
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time={args.slurm_timeout}
#SBATCH --gres=gpu{
        ":1080ti" if is_on_tcml() else ""
    }:1 # TCML has issue with 2080ti driver. Only 1080ti works.

set -e
set +x

# Export all current environment variables
{env_exports}

# Set environment variables
export BASE_CHECKPOINT_ENDPOINT={args.base_checkpoint_endpoint}/eval
export CHECKPOINT_ENDPOINT=$BASE_CHECKPOINT_ENDPOINT/{route_id}.json
export BASE_DEBUG_CHECKPOINT_ENDPOINT={args.base_checkpoint_endpoint}/debug_checkpoint
export DEBUG_CHECKPOINT_ENDPOINT=$BASE_DEBUG_CHECKPOINT_ENDPOINT/{route_id}.txt
export TEAM_AGENT={args.team_agent}
export TEAM_CONFIG={args.team_config}
export ROUTES={route_file}
export DEBUG_CHALLENGE=0
if [[ $EVALUATION_DATASET == "bench2drive" ]]; then
    export IS_BENCH2DRIVE=1
    export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
    export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
elif [[ $EVALUATION_DATASET == "fail2drive" ]]; then
    export IS_BENCH2DRIVE=0
    export SCENARIO_RUNNER_ROOT=3rd_party/fail2drive/scenario_runner
    export LEADERBOARD_ROOT=3rd_party/fail2drive/leaderboard
    export CARLA_ROOT=$LEAD_PROJECT_ROOT/3rd_party/CARLA_F2D
else
    export IS_BENCH2DRIVE=0
    export REPETITIONS=1
    export DEBUG_ENV_AGENT=0
    export RECORD=1
    export DATAGEN=0
    export SCENARIO_RUNNER_ROOT={
        "3rd_party/scenario_runner_autopilot"
        if args.expert
        else "3rd_party/scenario_runner"
    }
    export LEADERBOARD_ROOT={
        "3rd_party/leaderboard_autopilot" if args.expert else "3rd_party/leaderboard"
    }
fi
export PYTHONPATH=$SCENARIO_RUNNER_ROOT:$PYTHONPATH
export PYTHONPATH=$LEADERBOARD_ROOT:$PYTHONPATH
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/{route_id}
export BENCHMARK_ROUTE_ID={route_id}
export PORT=$(random_free_port.sh)
export TM_PORT=$(random_free_port.sh)
export PYTHONUNBUFFERED=1

# Before timer
date

# Create output folders and remove old eval file if exists
mkdir -p $BASE_CHECKPOINT_ENDPOINT
mkdir -p $BASE_DEBUG_CHECKPOINT_ENDPOINT
rm -f $CHECKPOINT_ENDPOINT
rm -f $DEBUG_CHECKPOINT_ENDPOINT

# Miniconda activating and debugging
which python3

# Debugging GPU availability
python -c "import torch; print('PyTorch CUDA is available?', torch.cuda.is_available())"
nvidia-smi

# Kill current carla running on the same GPU instance
clean_carla.sh

set -x
export PORT=$(random_free_port.sh)
export TM_PORT=$(random_free_port.sh)
bash $CARLA_ROOT/CarlaUE4.sh --world-port=$PORT -nosound -graphicsadapter=0 -RenderOffScreen &
sleep 180
nvidia-smi

if ! pgrep -f "CarlaUE4-Linux-Shipping.*--world-port=$PORT" >/dev/null; then
    echo "CarlaUE4-Linux-Shipping not found for port $PORT, exiting..."
    exit 1
fi

# Run evaluation
set +e

if [[ $EVALUATION_DATASET == "bench2drive" ]]; then
    CUDA_VISIBLE_DEVICES=0 python3 {bench2drive_evaluator} \
    --routes=$ROUTES \
    --track={args.track} \
    --checkpoint=$CHECKPOINT_ENDPOINT \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --debug=$DEBUG_CHALLENGE \
    --record=$RECORD_PATH \
    --resume=False \
    --port=$PORT \
    --traffic-manager-port=$TM_PORT \
    --timeout=120 \
    --debug-checkpoint=$DEBUG_CHECKPOINT_ENDPOINT \
    --traffic-manager-seed=$EXPERIMENT_SEED \
    --repetitions={args.repetitions}
elif [[ $EVALUATION_DATASET == "fail2drive" ]]; then
    CUDA_VISIBLE_DEVICES=0 python3 {fail2drive_evaluator} \
    --routes=$ROUTES \
    --track={args.track} \
    --checkpoint=$CHECKPOINT_ENDPOINT \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --debug=$DEBUG_CHALLENGE \
    --record=$RECORD_PATH \
    --resume=False \
    --port=$PORT \
    --traffic-manager-port=$TM_PORT \
    --timeout=120 \
    --debug-checkpoint=$DEBUG_CHECKPOINT_ENDPOINT \
    --traffic-manager-seed=$EXPERIMENT_SEED \
    --repetitions={args.repetitions}
else
    CUDA_VISIBLE_DEVICES=0 python3 {standard_evaluator} \
    --routes=$ROUTES \
    --track={args.track} \
    --checkpoint=$CHECKPOINT_ENDPOINT \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --debug=$DEBUG_CHALLENGE \
    --record=$RECORD_PATH \
    --resume=1 \
    --port=$PORT \
    --traffic-manager-port=$TM_PORT \
    --timeout=120 \
    --debug-checkpoint=$DEBUG_CHECKPOINT_ENDPOINT \
    --traffic-manager-seed=$EXPERIMENT_SEED \
    --repetitions={args.repetitions}
fi

# See what's going on with the GPU
nvidia-smi

# Kill only the Carla at the GPU we used
clean_carla.sh

# After timer
date
"""
    with open(bash_file, "w") as f:
        f.write(template)


def main():
    parser = argparse.ArgumentParser(description="Generate job scripts for evaluation.")
    parser.add_argument(
        "--base_checkpoint_endpoint",
        type=str,
        action="store",
        help="Root folder for saving job files.",
    )
    parser.add_argument(
        "--checkpoint_endpoint",
        type=str,
        action="store",
        help="Folder for TransFuser checkpoint.",
    )
    parser.add_argument(
        "--team_agent",
        type=str,
        action="store",
        help="Agent under test.",
    )
    parser.add_argument(
        "--team_config",
        type=str,
        action="store",
        help="Config of the agent",
    )
    parser.add_argument(
        "--route_folder",
        type=str,
        action="store",
        help="Folder containing route XML files.",
    )
    parser.add_argument(
        "--partition",
        default="2080-galvani" if "TCML" not in os.environ else "week",
        type=str,
        action="store",
        help="Partition to submit the job.",
    )
    parser.add_argument(
        "--track",
        default="SENSORS",
        type=str,
        action="store",
        help="Track to evaluate.",
    )
    parser.add_argument("--slurm_timeout", default="0-04:00:00", help="Slurm timeout.")
    parser.add_argument(
        "--repetitions",
        default=1,
        type=int,
        help="Number of repetitions.",
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help="Use autopilot scenario_runner and leaderboard with local evaluator.",
    )
    args = parser.parse_args()

    # Load all route XML files, including from subdirectories
    route_files: list[str] = glob.glob(
        os.path.join(args.route_folder, "**", "*.xml"),
        recursive=True,
    )
    bash_save_path = f"{args.base_checkpoint_endpoint}/scripts/"
    os.makedirs(bash_save_path, exist_ok=True)

    for route_file in route_files:
        route_id = Path(route_file).stem
        # Generate bash
        bash_file = os.path.join(bash_save_path, f"{route_id}.sh")
        make_bash(bash_file, route_file, route_id, args)


if __name__ == "__main__":
    main()
