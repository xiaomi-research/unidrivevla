#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT:-$SCRIPT_DIR}
PROJECT_DIR=${PROJECT_DIR:-$LEAD_PROJECT_ROOT/Bench2Drive}

export LEAD_PROJECT_ROOT
export PROJECT_DIR
export CONFIG=${CONFIG:-$PROJECT_DIR/projects/configs/unidrivevla_lead_stage2_motion_4hz8_2b.py}
export CHECKPOINT=${CHECKPOINT:-${LEAD_CHECKPOINT:?Set LEAD_CHECKPOINT or CHECKPOINT to the external LEAD final checkpoint}}
export SAVE_NAME=${SAVE_NAME:-latest_4hz_stage3_ds87_epoch9_comfort}
export EVALUATION_DATASET=bench2drive
export ROUTE_SOURCE=${ROUTE_SOURCE:-${LEAD_ROUTE_SOURCE:?Set LEAD_ROUTE_SOURCE or ROUTE_SOURCE to the external 220-route XML directory}}
BENCH2DRIVE_NUM_SHARDS=${BENCH2DRIVE_NUM_SHARDS:-1}
BENCH2DRIVE_SHARD_INDEX=${BENCH2DRIVE_SHARD_INDEX:-0}
export BENCH2DRIVE_NUM_SHARDS
export BENCH2DRIVE_SHARD_INDEX

export LOCAL_GPUS_PER_NODE=${LOCAL_GPUS_PER_NODE:-8}
export LEAD_LOCAL_MAX_PARALLEL_JOBS=${LEAD_LOCAL_MAX_PARALLEL_JOBS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export LEAD_RESUME_SKIP_FINISHED=${LEAD_RESUME_SKIP_FINISHED:-1}
export UNIDRIVEVLA_LEAD_CONTROL=${UNIDRIVEVLA_LEAD_CONTROL:-tfv6_route_speed}
export USE_B2D_NONROOT_CARLA=${USE_B2D_NONROOT_CARLA:-0}
export ROUTE_STARTUP_STAGGER=${ROUTE_STARTUP_STAGGER:-60}
export CARLA_STARTUP_SLEEP=${CARLA_STARTUP_SLEEP:-60}
export LEADERBOARD_TIMEOUT=${LEADERBOARD_TIMEOUT:-600}
export PORT_BASE=${PORT_BASE:-20000}
export TM_PORT_BASE=${TM_PORT_BASE:-30000}
export PORT_STRIDE=${PORT_STRIDE:-10}

for required_file in "$CONFIG" "$CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ ! -d "$ROUTE_SOURCE" ]]; then
    echo "Bench2Drive split-route directory not found: $ROUTE_SOURCE" >&2
    exit 1
fi

route_count=$(find "$ROUTE_SOURCE" -maxdepth 1 -type f -name '*.xml' | wc -l)
if [[ "$route_count" -ne 220 ]]; then
    echo "Expected 220 Bench2Drive route XML files, found $route_count in $ROUTE_SOURCE" >&2
    exit 1
fi

if ! [[ "$BENCH2DRIVE_NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$BENCH2DRIVE_NUM_SHARDS" -lt 1 ]]; then
    echo "BENCH2DRIVE_NUM_SHARDS must be a positive integer" >&2
    exit 1
fi
if ! [[ "$BENCH2DRIVE_SHARD_INDEX" =~ ^[0-9]+$ ]] || \
        [[ "$BENCH2DRIVE_SHARD_INDEX" -ge "$BENCH2DRIVE_NUM_SHARDS" ]]; then
    echo "BENCH2DRIVE_SHARD_INDEX must be in [0, $((BENCH2DRIVE_NUM_SHARDS - 1))]" >&2
    exit 1
fi

# Four concurrent CARLA/model workers are spread across the 8-GPU node. The
# local scheduler exclusively leases these physical GPU ids and derives a
# distinct CARLA RPC/TM port pair from each id.
if [[ -z "${LEAD_LOCAL_GPU_IDS:-}" ]]; then
    if ((BENCH2DRIVE_SHARD_INDEX % 2 == 0)); then
        LEAD_LOCAL_GPU_IDS="0,2,4,6"
    else
        LEAD_LOCAL_GPU_IDS="1,3,5,7"
    fi
fi
export LEAD_LOCAL_GPU_IDS

selected_route_count=$route_count
if [[ "$BENCH2DRIVE_NUM_SHARDS" -gt 1 ]]; then
    if [[ -n "${ROUTE_IDS:-}" || -n "${ROUTE_LIMIT:-}" ]]; then
        echo "Do not combine BENCH2DRIVE_NUM_SHARDS with ROUTE_IDS or ROUTE_LIMIT" >&2
        exit 1
    fi
    mapfile -t all_route_ids < <(
        find "$ROUTE_SOURCE" -maxdepth 1 -type f -name '*.xml' -printf '%f\n' \
            | sed 's/\.xml$//' \
            | sort -n
    )
    shard_route_ids=()
    for ((route_idx = 0; route_idx < ${#all_route_ids[@]}; route_idx++)); do
        if ((route_idx % BENCH2DRIVE_NUM_SHARDS == BENCH2DRIVE_SHARD_INDEX)); then
            shard_route_ids+=("${all_route_ids[$route_idx]}")
        fi
    done
    selected_route_count=${#shard_route_ids[@]}
    expected_shard_count=$(((route_count + BENCH2DRIVE_NUM_SHARDS - 1 - BENCH2DRIVE_SHARD_INDEX) / BENCH2DRIVE_NUM_SHARDS))
    if [[ "$selected_route_count" -ne "$expected_shard_count" ]]; then
        echo "Shard selection error: expected $expected_shard_count routes, got $selected_route_count" >&2
        exit 1
    fi
    export ROUTE_IDS="${shard_route_ids[*]}"
    export ALLOW_BENCH2DRIVE_SUBSET=1
elif [[ -n "${ROUTE_IDS:-}" || -n "${ROUTE_LIMIT:-}" ]]; then
    if [[ "${ALLOW_BENCH2DRIVE_SUBSET:-0}" != "1" ]]; then
        echo "ROUTE_IDS/ROUTE_LIMIT would make this a subset evaluation." >&2
        echo "Set ALLOW_BENCH2DRIVE_SUBSET=1 only for an intentional subset." >&2
        exit 1
    fi
fi

if [[ -z "${EXPERIMENT_RUN_ID:-}" ]]; then
    if [[ "$BENCH2DRIVE_NUM_SHARDS" -gt 1 ]]; then
        EXPERIMENT_RUN_ID=${SAVE_NAME}_Bench2Drive_full220_8gpu_v1_shard${BENCH2DRIVE_SHARD_INDEX}of${BENCH2DRIVE_NUM_SHARDS}
    else
        EXPERIMENT_RUN_ID=${SAVE_NAME}_Bench2Drive_full220_8gpu_v1
    fi
fi
export EXPERIMENT_RUN_ID

echo "UniDriveVLA LEAD final Stage3 Bench2Drive full evaluation"
echo "CONFIG=$CONFIG"
echo "CHECKPOINT=$CHECKPOINT"
echo "ROUTE_SOURCE=$ROUTE_SOURCE"
echo "ROUTES_AVAILABLE=$route_count"
echo "BENCH2DRIVE_SHARD=$BENCH2DRIVE_SHARD_INDEX/$BENCH2DRIVE_NUM_SHARDS"
echo "ROUTES_SELECTED=$selected_route_count"
echo "LOCAL_GPUS_PER_NODE=$LOCAL_GPUS_PER_NODE"
echo "LEAD_LOCAL_MAX_PARALLEL_JOBS=$LEAD_LOCAL_MAX_PARALLEL_JOBS"
echo "LEAD_LOCAL_GPU_IDS=${LEAD_LOCAL_GPU_IDS:-<all>}"
echo "UNIDRIVEVLA_LEAD_CONTROL=$UNIDRIVEVLA_LEAD_CONTROL"
echo "EXPERIMENT_RUN_ID=$EXPERIMENT_RUN_ID"

exec bash "$LEAD_PROJECT_ROOT/run_unidrivevla_lead_stage1_eval_8gpu.sh"
