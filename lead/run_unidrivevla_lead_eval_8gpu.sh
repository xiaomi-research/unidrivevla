#!/usr/bin/env bash
set -euo pipefail

# Single-node multi-GPU runner for LEAD official closed-loop evaluation.
# Default target is LEAD longest6 split routes, not Bench2Drive leaderboard eval.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT:-$SCRIPT_DIR}
PROJECT_DIR=${PROJECT_DIR:-$LEAD_PROJECT_ROOT/Bench2Drive}
UNIDRIVEVLA_ROOT=${UNIDRIVEVLA_ROOT:-$LEAD_PROJECT_ROOT/..}
MMCV_ROOT=${MMCV_ROOT:-${LEAD_MMCV_ROOT:-$UNIDRIVEVLA_ROOT/third_party/mmcv-1.7.2}}
MMDET3D_ROOT=${MMDET3D_ROOT:-${LEAD_MMDET3D_ROOT:-$UNIDRIVEVLA_ROOT/third_party/mmdetection3d-1.0.0rc6}}

CONFIG=${CONFIG:-$PROJECT_DIR/projects/configs/unidrivevla_lead_stage2_motion_4hz8_2b.py}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the external LEAD checkpoint}
SAVE_NAME=${SAVE_NAME:-lead_stage2_spatial_iter200}
VLM_PRETRAINED_PATH=${VLM_PRETRAINED_PATH:?Set VLM_PRETRAINED_PATH to the external Qwen3-VL-2B-Instruct directory}
VLM_VARIANT=${VLM_VARIANT:-2b}
SYNC_QWEN3_VL_FOR_EVAL=${SYNC_QWEN3_VL_FOR_EVAL:-1}
QWEN3_VL_SOURCE=${QWEN3_VL_SOURCE:-}
SYNC_QWEN_VL_UTILS_FOR_EVAL=${SYNC_QWEN_VL_UTILS_FOR_EVAL:-1}
QWEN_VL_UTILS_SOURCE=${QWEN_VL_UTILS_SOURCE:-}

PYTHON_BIN=${PYTHON_BIN:?Set PYTHON_BIN to the evaluation Python executable}
CARLA_ROOT=${CARLA_ROOT:?Set CARLA_ROOT to the external CARLA installation}
CARLA_PYTHON_BINDING=${CARLA_PYTHON_BINDING:-}
CARLA_FAKE_UID_LIB=${CARLA_FAKE_UID_LIB:-$LEAD_PROJECT_ROOT/scripts/libfake_nonroot_uid.so}
CARLA_VK_ICD=${CARLA_VK_ICD:-}
if [[ -z "$CARLA_VK_ICD" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    CARLA_VK_ICD=/etc/vulkan/icd.d/nvidia_icd.json
fi
CARLA_XDG_RUNTIME_DIR=${CARLA_XDG_RUNTIME_DIR:-}
if [[ -z "$CARLA_XDG_RUNTIME_DIR" ]]; then
    CARLA_XDG_RUNTIME_DIR=/tmp/xdg-lead-eval-${UID:-1001}
fi
CARLA_STARTUP_SLEEP=${CARLA_STARTUP_SLEEP:-180}
VULKAN_PREFLIGHT=${VULKAN_PREFLIGHT:-0}
CLEAN_EXISTING_CARLA=${CLEAN_EXISTING_CARLA:-0}
USE_B2D_NONROOT_CARLA=${USE_B2D_NONROOT_CARLA:-0}
CARLA_RUN_UID=${CARLA_RUN_UID:-1001}
CARLA_RUN_GID=${CARLA_RUN_GID:-1001}
SETPRIV_BIN=${SETPRIV_BIN:-/usr/bin/setpriv}

EVALUATION_DATASET=${EVALUATION_DATASET:-longest6}
ROUTE_SOURCE=${ROUTE_SOURCE:-$LEAD_PROJECT_ROOT/data/benchmark_routes/$EVALUATION_DATASET}
EXPERIMENT_RUN_ID=${EXPERIMENT_RUN_ID:-unidrivevla_lead_stage2_spatial_iter200_${EVALUATION_DATASET}_8gpu}
EVALUATION_OUTPUT_DIR=${EVALUATION_OUTPUT_DIR:-$LEAD_PROJECT_ROOT/outputs/evaluation/$EXPERIMENT_RUN_ID}
RUN_EVAL=${RUN_EVAL:-1}
LOCAL_GPUS_PER_NODE=${LOCAL_GPUS_PER_NODE:-8}
LEAD_LOCAL_MAX_PARALLEL_JOBS=${LEAD_LOCAL_MAX_PARALLEL_JOBS:-$LOCAL_GPUS_PER_NODE}
LEAD_LOCAL_GPU_IDS=${LEAD_LOCAL_GPU_IDS:-$(seq -s, 0 $((LOCAL_GPUS_PER_NODE - 1)))}
LEAD_FORCE_LOCAL=${LEAD_FORCE_LOCAL:-1}
PORT_BASE=${PORT_BASE:-20000}
TM_PORT_BASE=${TM_PORT_BASE:-30000}
PORT_STRIDE=${PORT_STRIDE:-10}
LEADERBOARD_TIMEOUT=${LEADERBOARD_TIMEOUT:-600}
CARLA_SEQUENTIAL_PREFLIGHT=${CARLA_SEQUENTIAL_PREFLIGHT:-1}
CARLA_PREFLIGHT_TIMEOUT=${CARLA_PREFLIGHT_TIMEOUT:-90}
CARLA_SEQUENTIAL_STARTUP=${CARLA_SEQUENTIAL_STARTUP:-1}
if [[ -z "${ROUTE_STARTUP_STAGGER+x}" ]]; then
    if [[ "$CARLA_SEQUENTIAL_STARTUP" == "1" ]]; then
        ROUTE_STARTUP_STAGGER=$((CARLA_STARTUP_SLEEP + 5))
    else
        ROUTE_STARTUP_STAGGER=0
    fi
fi
if [[ "$CARLA_SEQUENTIAL_STARTUP" == "1" ]]; then
    minimum_stagger=$((CARLA_STARTUP_SLEEP + 5))
    if (( ROUTE_STARTUP_STAGGER < minimum_stagger )); then
        ROUTE_STARTUP_STAGGER=$minimum_stagger
    fi
fi
ROUTE_LIMIT=${ROUTE_LIMIT:-}
ROUTE_IDS=${ROUTE_IDS:-}
LEAD_RESUME_SKIP_FINISHED=${LEAD_RESUME_SKIP_FINISHED:-1}
UNIDRIVEVLA_LEAD_DELTA_STATS=${UNIDRIVEVLA_LEAD_DELTA_STATS:-config}
UNIDRIVEVLA_LEAD_PLAN_COORD=${UNIDRIVEVLA_LEAD_PLAN_COORD:-lead_forward_lateral}
UNIDRIVEVLA_LEAD_CONTROL=${UNIDRIVEVLA_LEAD_CONTROL:-b2d_pid}
UNIDRIVEVLA_LEAD_JPEG_QUALITY=${UNIDRIVEVLA_LEAD_JPEG_QUALITY:-90}
UNIDRIVEVLA_LEAD_WAYPOINT_HZ=${UNIDRIVEVLA_LEAD_WAYPOINT_HZ:-auto}
UNIDRIVEVLA_LEAD_COMMAND_SOURCE=${UNIDRIVEVLA_LEAD_COMMAND_SOURCE:-current}
UNIDRIVEVLA_LEAD_ROUTE_PLANNER=${UNIDRIVEVLA_LEAD_ROUTE_PLANNER:-lead}
UNIDRIVEVLA_LEAD_ROUTE_POP_DISTANCE=${UNIDRIVEVLA_LEAD_ROUTE_POP_DISTANCE:-5.0}
UNIDRIVEVLA_LEAD_ROUTE_ADAPTIVE_POP=${UNIDRIVEVLA_LEAD_ROUTE_ADAPTIVE_POP:-1}
UNIDRIVEVLA_LEAD_TEMPORAL_MODE=${UNIDRIVEVLA_LEAD_TEMPORAL_MODE:-close_loop}

require_path() {
    local path="$1"
    local label="$2"
    if [[ ! -e "$path" ]]; then
        echo "$label not found: $path" >&2
        exit 1
    fi
}

require_path "$LEAD_PROJECT_ROOT" "LEAD_PROJECT_ROOT"
require_path "$PROJECT_DIR" "PROJECT_DIR"
require_path "$CONFIG" "CONFIG"
require_path "$CHECKPOINT" "CHECKPOINT"
require_path "$VLM_PRETRAINED_PATH" "VLM_PRETRAINED_PATH"
require_path "$CARLA_ROOT/CarlaUE4.sh" "CARLA executable"
require_path "$ROUTE_SOURCE" "ROUTE_SOURCE"
if [[ -n "$CARLA_PYTHON_BINDING" ]]; then
    require_path "$CARLA_PYTHON_BINDING" "CARLA_PYTHON_BINDING"
fi

PYTHON_BIN_RESOLVED=$(command -v "$PYTHON_BIN")
PYTHON_BIN_DIR=$(dirname "$PYTHON_BIN_RESOLVED")
PYTHON_SITE_PACKAGES=$("$PYTHON_BIN" - <<'PYSITE'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PYSITE
)
if [[ -z "$PYTHON_SITE_PACKAGES" ]]; then
    echo "Could not resolve site-packages for $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$VULKAN_PREFLIGHT" == "1" ]]; then
    if ! command -v vulkaninfo >/dev/null 2>&1; then
        echo "vulkaninfo not found; set VULKAN_PREFLIGHT=0 to skip this check." >&2
        exit 1
    fi
    echo "Checking Vulkan/NVIDIA ICD before starting CARLA..."
    VK_ICD_FILENAMES="$CARLA_VK_ICD" vulkaninfo --summary >/tmp/lead_vulkaninfo_summary.txt
    grep -E "GPU[0-9]+|deviceName|driverName|driverVersion" /tmp/lead_vulkaninfo_summary.txt || true
fi

if [[ "$SYNC_QWEN_VL_UTILS_FOR_EVAL" == "1" ]]; then
    if "$PYTHON_BIN" - <<'PYCHECK_QWEN_UTILS' >/dev/null 2>&1
import qwen_vl_utils
PYCHECK_QWEN_UTILS
    then
        echo "qwen_vl_utils already importable in eval Python."
    else
        if [[ -z "$QWEN_VL_UTILS_SOURCE" ]]; then
            for candidate in \
                "$PYTHON_SITE_PACKAGES/qwen_vl_utils"; do
                if [[ -f "$candidate/__init__.py" && -f "$candidate/vision_process.py" ]]; then
                    QWEN_VL_UTILS_SOURCE="$candidate"
                    break
                fi
            done
        fi
        require_path "$QWEN_VL_UTILS_SOURCE" "QWEN_VL_UTILS_SOURCE"
        export QWEN_VL_UTILS_SOURCE
        export PYTHON_SITE_PACKAGES
        "$PYTHON_BIN" - <<'PYUTILS'
import os
import shutil
from pathlib import Path

src = Path(os.environ["QWEN_VL_UTILS_SOURCE"]).resolve()
dst = Path(os.environ["PYTHON_SITE_PACKAGES"]) / "qwen_vl_utils"
required = ["__init__.py", "vision_process.py"]
missing = [name for name in required if not (src / name).is_file()]
if missing:
    raise SystemExit(f"QWEN_VL_UTILS_SOURCE is incomplete: missing {missing} under {src}")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
print(f"Synced qwen_vl_utils for eval: {src} -> {dst}")
PYUTILS
    fi
fi

cd "$LEAD_PROJECT_ROOT"

export LEAD_PROJECT_ROOT
export PROJECT_DIR
export VLM_PRETRAINED_PATH
export VLM_VARIANT
export EVALUATION_DATASET
export EVALUATION_OUTPUT_DIR
export EXPERIMENT_RUN_ID
export RECORD_PATH=${RECORD_PATH:-$EVALUATION_OUTPUT_DIR/records}
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTHONUNBUFFERED=1
export LEAD_FORCE_LOCAL
export LEAD_LOCAL_MAX_PARALLEL_JOBS
export LOCAL_GPUS_PER_NODE
export CARLA_ROOT
export CARLA_VK_ICD
export CARLA_XDG_RUNTIME_DIR
export CARLA_FAKE_UID_LIB
export CARLA_STARTUP_SLEEP
export PORT_BASE
export TM_PORT_BASE
export PORT_STRIDE
export LEADERBOARD_TIMEOUT
export ROUTE_STARTUP_STAGGER
export CARLA_SEQUENTIAL_PREFLIGHT
export CARLA_PREFLIGHT_TIMEOUT
export CARLA_SEQUENTIAL_STARTUP
export USE_B2D_NONROOT_CARLA
export CARLA_RUN_UID
export CARLA_RUN_GID
export SETPRIV_BIN
export LEAD_RESUME_SKIP_FINISHED
export UNIDRIVEVLA_LEAD_DELTA_STATS
export UNIDRIVEVLA_LEAD_PLAN_COORD
export UNIDRIVEVLA_LEAD_CONTROL
export UNIDRIVEVLA_LEAD_JPEG_QUALITY
export UNIDRIVEVLA_LEAD_WAYPOINT_HZ
export UNIDRIVEVLA_LEAD_COMMAND_SOURCE
export UNIDRIVEVLA_LEAD_ROUTE_PLANNER
export UNIDRIVEVLA_LEAD_ROUTE_POP_DISTANCE
export UNIDRIVEVLA_LEAD_ROUTE_ADAPTIVE_POP
export UNIDRIVEVLA_LEAD_TEMPORAL_MODE

export PATH="$PYTHON_BIN_DIR:$LEAD_PROJECT_ROOT/scripts:$LEAD_PROJECT_ROOT/3rd_party/Bench2Drive/tools:${PATH:-}"

TRANSFORMERS_SRC=${TRANSFORMERS_SRC:-$LEAD_PROJECT_ROOT/../third_party/transformers-4.57.1/src}
if [[ -d "$TRANSFORMERS_SRC/transformers" ]]; then
    export TRANSFORMERS_SRC
    if [[ "$SYNC_QWEN3_VL_FOR_EVAL" == "1" ]]; then
        qwen3_vl_complete=0
        for required_qwen3_file in \
            "$TRANSFORMERS_SRC/transformers/models/qwen3_vl/__init__.py" \
            "$TRANSFORMERS_SRC/transformers/models/qwen3_vl/configuration_qwen3_vl.py" \
            "$TRANSFORMERS_SRC/transformers/models/qwen3_vl/modeling_qwen3_vl.py" \
            "$TRANSFORMERS_SRC/transformers/models/qwen3_vl/processing_qwen3_vl.py" \
            "$TRANSFORMERS_SRC/transformers/models/qwen3_vl/video_processing_qwen3_vl.py"; do
            if [[ ! -f "$required_qwen3_file" ]]; then
                qwen3_vl_complete=1
                break
            fi
        done
        if [[ "$qwen3_vl_complete" == "0" ]]; then
            echo "Qwen3-VL transformers files already complete under $TRANSFORMERS_SRC."
        else
            if [[ -z "$QWEN3_VL_SOURCE" ]]; then
                for candidate in \
                    "$PYTHON_SITE_PACKAGES/transformers/models/qwen3_vl"; do
                    if [[ -f "$candidate/configuration_qwen3_vl.py" && -f "$candidate/processing_qwen3_vl.py" && -f "$candidate/video_processing_qwen3_vl.py" ]]; then
                        QWEN3_VL_SOURCE="$candidate"
                        break
                    fi
                done
            fi
            require_path "$QWEN3_VL_SOURCE" "QWEN3_VL_SOURCE"
            export QWEN3_VL_SOURCE
        "$PYTHON_BIN" - <<'PYSYNC'
import os
import shutil
from pathlib import Path

src = Path(os.environ["QWEN3_VL_SOURCE"]).resolve()
dst = Path(os.environ["TRANSFORMERS_SRC"]) / "transformers" / "models" / "qwen3_vl"
required = [
    "__init__.py",
    "configuration_qwen3_vl.py",
    "modeling_qwen3_vl.py",
    "processing_qwen3_vl.py",
    "video_processing_qwen3_vl.py",
]
missing = [name for name in required if not (src / name).is_file()]
if missing:
    raise SystemExit(f"QWEN3_VL_SOURCE is incomplete: missing {missing} under {src}")
dst.mkdir(parents=True, exist_ok=True)
for name in required:
    shutil.copy2(src / name, dst / name)

# The available LEAD evaluator runs under Python 3.8. Qwen3-VL source generated
# for the training env uses builtin generic annotations such as list[int],
# which Python 3.8 evaluates eagerly unless postponed.
for path in dst.glob("*.py"):
    text = path.read_text()
    if "from __future__ import annotations" in text:
        continue
    lines = text.splitlines()
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    while insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if stripped.startswith("#") or not stripped:
            insert_at += 1
            continue
        break
    lines.insert(insert_at, "from __future__ import annotations")
    path.write_text("\n".join(lines) + "\n")
print(f"Synced Qwen3-VL transformers files for eval: {src} -> {dst}")
PYSYNC
        fi
    fi
    export PYTHONPATH="$TRANSFORMERS_SRC:${PYTHONPATH:-}"
fi

export PYTHONPATH="$LEAD_PROJECT_ROOT:$PROJECT_DIR:$PROJECT_DIR/bench2drive/leaderboard:$MMCV_ROOT:$MMDET3D_ROOT:${PYTHONPATH:-}"

CARLA_EGG=$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name 'carla-0.9.15-py3.*-linux-x86_64.egg' | sort -r | head -n 1 || true)
if [[ -n "$CARLA_EGG" ]]; then
    export PYTHONPATH="$CARLA_EGG:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist:${PYTHONPATH}"
else
    export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist:${PYTHONPATH}"
fi
if [[ -n "$CARLA_PYTHON_BINDING" ]]; then
    export PYTHONPATH="$CARLA_PYTHON_BINDING:${PYTHONPATH}"
fi

"$PYTHON_BIN" - <<'PYCHECK'
missing = []
for mod in ["omegaconf", "wandb", "beartype", "jsonpickle", "filterpy", "ujson"]:
    try:
        __import__(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc}")
if missing:
    print("Missing LEAD eval Python dependencies:")
    for item in missing:
        print("  " + item)
    raise SystemExit(1)
PYCHECK

if [[ "$CLEAN_EXISTING_CARLA" == "1" ]]; then
    clean_carla.sh || true
fi

unset OPENAI_API_KEY
unset WANDB_API_KEY
unset HF_TOKEN
unset HUGGINGFACE_TOKEN
unset GITHUB_TOKEN


preflight_carla_gpu() {
    local gpu="$1"
    local port=$((PORT_BASE + gpu * PORT_STRIDE))
    local tm_port=$((TM_PORT_BASE + gpu * PORT_STRIDE))
    local log_file="$EVALUATION_OUTPUT_DIR/carla_preflight_gpu${gpu}.log"
    local pid=""
    local ready=0
    local elapsed=0
    local -a launch=(env)

    if [[ "$USE_B2D_NONROOT_CARLA" != "1" && -f "$CARLA_FAKE_UID_LIB" ]]; then
        launch+=("LD_PRELOAD=$CARLA_FAKE_UID_LIB")
    fi
    if [[ -n "$CARLA_VK_ICD" ]]; then
        launch+=("VK_ICD_FILENAMES=$CARLA_VK_ICD")
    fi
    if [[ -n "$CARLA_XDG_RUNTIME_DIR" ]]; then
        launch+=("XDG_RUNTIME_DIR=$CARLA_XDG_RUNTIME_DIR")
    fi
    launch+=("CUDA_VISIBLE_DEVICES=$gpu" "CARLA_GRAPHICS_ADAPTER=$gpu")
    if [[ "$USE_B2D_NONROOT_CARLA" == "1" ]]; then
        launch+=("$SETPRIV_BIN" --reuid "$CARLA_RUN_UID" --regid "$CARLA_RUN_GID" --clear-groups)
    fi
    launch+=(bash "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound
        "-carla-rpc-port=$port" "-graphicsadapter=$gpu")

    echo "[CARLA_PREFLIGHT] GPU $gpu starting on port $port" | tee "$log_file"
    pgrep -f "CarlaUE4-Linux-Shipping.*carla-rpc-port=$port" | xargs -r kill -9 || true
    pgrep -f "CarlaUE4.sh .*carla-rpc-port=$port" | xargs -r kill -9 || true
    mkdir -p "$CARLA_XDG_RUNTIME_DIR"
    "${launch[@]}" >>"$log_file" 2>&1 &
    pid=$!
    while (( elapsed < CARLA_PREFLIGHT_TIMEOUT )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        if "$PYTHON_BIN" - "$port" <<'PYCHECK_CARLA_PORT' >/dev/null 2>&1
import sys
import carla
client = carla.Client("127.0.0.1", int(sys.argv[1]))
client.set_timeout(2.0)
client.get_world()
PYCHECK_CARLA_PORT
        then
            ready=1
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    pgrep -f "CarlaUE4-Linux-Shipping.*carla-rpc-port=$port" | xargs -r kill -9 || true
    pgrep -f "CarlaUE4.sh .*carla-rpc-port=$port" | xargs -r kill -9 || true
    if [[ "$ready" == "1" ]]; then
        echo "[CARLA_PREFLIGHT] GPU $gpu healthy after ${elapsed}s"
        return 0
    fi
    echo "[CARLA_PREFLIGHT] GPU $gpu failed after ${elapsed}s; see $log_file" >&2
    return 1
}

run_carla_preflight() {
    if [[ "$RUN_EVAL" != "1" || "$CARLA_SEQUENTIAL_PREFLIGHT" != "1" ]]; then
        return 0
    fi
    local gpu
    local -a requested=()
    local -a healthy=()
    local configured_max="$LEAD_LOCAL_MAX_PARALLEL_JOBS"
    read -r -a requested <<< "${LEAD_LOCAL_GPU_IDS//,/ }"
    mkdir -p "$EVALUATION_OUTPUT_DIR" "$CARLA_XDG_RUNTIME_DIR"
    echo "[CARLA_PREFLIGHT] sequential GPU probe: ${requested[*]}"
    for gpu in "${requested[@]}"; do
        if [[ "$gpu" =~ ^[0-9]+$ ]] && preflight_carla_gpu "$gpu"; then
            healthy+=("$gpu")
        fi
    done
    if [[ "${#healthy[@]}" -eq 0 ]]; then
        echo "[CARLA_PREFLIGHT] no healthy GPU passed CARLA startup" >&2
        exit 1
    fi
    local joined
    joined=$(IFS=,; printf '%s' "${healthy[*]}")
    LEAD_LOCAL_GPU_IDS="$joined"
    if (( configured_max > ${#healthy[@]} )); then
        LEAD_LOCAL_MAX_PARALLEL_JOBS=${#healthy[@]}
    fi
    export LEAD_LOCAL_GPU_IDS LEAD_LOCAL_MAX_PARALLEL_JOBS
    echo "[CARLA_PREFLIGHT] healthy GPUs=$LEAD_LOCAL_GPU_IDS parallel_jobs=$LEAD_LOCAL_MAX_PARALLEL_JOBS"
}

route_finished() {
    local route_id="$1"
    local checkpoint_file="$EVALUATION_OUTPUT_DIR/eval/${route_id}.json"
    local metric_file="$EVALUATION_OUTPUT_DIR/${route_id}/metric_info.json"

    if [[ "$LEAD_RESUME_SKIP_FINISHED" != "1" ]]; then
        return 1
    fi

    "$PYTHON_BIN" - "$checkpoint_file" "$metric_file" "${LEAD_RESUME_REQUIRE_COMFORT_METRICS:-0}" <<'PYCHECK_ROUTE'
import json
import sys
from pathlib import Path

checkpoint_path = Path(sys.argv[1])
metric_path = Path(sys.argv[2])
require_comfort_metrics = sys.argv[3] == "1"
if not checkpoint_path.is_file():
    raise SystemExit(1)

try:
    data = json.loads(checkpoint_path.read_text())
except Exception:
    raise SystemExit(1)

from slurm.evaluation.evaluate_utils import did_route_crash
if did_route_crash(data):
    raise SystemExit(1)

if require_comfort_metrics:
    required_fields = {
        "acceleration",
        "angular_velocity",
        "forward_vector",
        "right_vector",
        "location",
        "rotation",
    }
    try:
        metric_data = json.loads(metric_path.read_text())
    except Exception:
        raise SystemExit(1)
    if not isinstance(metric_data, dict) or not metric_data:
        raise SystemExit(1)
    if any(
        not isinstance(frame, dict) or required_fields.difference(frame)
        for frame in metric_data.values()
    ):
        raise SystemExit(1)

raise SystemExit(0)
PYCHECK_ROUTE
}

prepare_route() {
    local route_file="$1"
    local route_id
    route_id="$(basename "$route_file" .xml)"
    local all_route_file="$ROUTE_ALL_FOLDER/$(basename "$route_file")"
    local dst_route_file="$ROUTE_FOLDER/$(basename "$route_file")"

    "$PYTHON_BIN" - "$route_file" "$all_route_file" "$route_id" <<'PYNORMALIZE_ROUTE'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
route_id = sys.argv[3]

tree = ET.parse(src)
root = tree.getroot()
routes = list(root.iter("route"))
if len(routes) != 1:
    raise SystemExit(f"Expected exactly one route in {src}, found {len(routes)}")
routes[0].set("id", route_id)
dst.parent.mkdir(parents=True, exist_ok=True)
tree.write(dst, encoding="utf-8", xml_declaration=True)
PYNORMALIZE_ROUTE

    if route_finished "$route_id"; then
        echo "Skipping finished route $route_id: $EVALUATION_OUTPUT_DIR/eval/${route_id}.json"
        return 0
    fi

    cp "$all_route_file" "$dst_route_file"
}

ROUTE_FOLDER="$EVALUATION_OUTPUT_DIR/routes"
ROUTE_ALL_FOLDER="$EVALUATION_OUTPUT_DIR/routes_all"
export ROUTE_ALL_FOLDER
rm -rf "$ROUTE_FOLDER" "$ROUTE_ALL_FOLDER" "$EVALUATION_OUTPUT_DIR/scripts"
mkdir -p "$ROUTE_FOLDER" "$ROUTE_ALL_FOLDER" "$EVALUATION_OUTPUT_DIR"

if [[ -d "$ROUTE_SOURCE" ]]; then
    if [[ -n "$ROUTE_IDS" ]]; then
        for route_id in $ROUTE_IDS; do
            route_file="$ROUTE_SOURCE/$route_id.xml"
            require_path "$route_file" "route XML"
            prepare_route "$route_file"
        done
    else
        count=0
        while IFS= read -r route_file; do
            prepare_route "$route_file"
            count=$((count + 1))
            if [[ -n "$ROUTE_LIMIT" && "$count" -ge "$ROUTE_LIMIT" ]]; then
                break
            fi
        done < <(find "$ROUTE_SOURCE" -maxdepth 1 -type f -name '*.xml' | sort)
    fi
else
    require_path "$ROUTE_SOURCE" "ROUTE_SOURCE"
    prepare_route "$ROUTE_SOURCE"
fi

route_count=$(find "$ROUTE_FOLDER" \( -type l -o -type f \) | wc -l)
if [[ "$route_count" -eq 0 ]]; then
    if [[ "$LEAD_RESUME_SKIP_FINISHED" == "1" ]]; then
        echo "No route XML files prepared under $ROUTE_FOLDER; all requested routes are already finished."
        "$PYTHON_BIN" - <<'PYAGG_FINISHED'
from slurm.evaluation import evaluate_utils

evaluate_utils.aggregate_metrics()
PYAGG_FINISHED
        exit 0
    fi
    echo "No route XML files prepared under $ROUTE_FOLDER" >&2
    exit 1
fi

run_carla_preflight

TEAM_CONFIG="${CONFIG}+${CHECKPOINT}+${SAVE_NAME}"

"$PYTHON_BIN" slurm/evaluation/evaluate_scripts_generator.py \
    --base_checkpoint_endpoint "$EVALUATION_OUTPUT_DIR" \
    --route_folder "$ROUTE_FOLDER" \
    --team_agent "$LEAD_PROJECT_ROOT/lead/inference/unidrivevla_lead_agent.py" \
    --team_config "$TEAM_CONFIG" \
    --track SENSORS

"$PYTHON_BIN" - <<'PYPATCH'
import os
from pathlib import Path

out_dir = Path(os.environ["EVALUATION_OUTPUT_DIR"])
scripts_dir = out_dir / "scripts"
fake_uid_lib = os.environ.get("CARLA_FAKE_UID_LIB", "")
vk_icd = os.environ.get("CARLA_VK_ICD", "")
xdg_runtime_dir = os.environ.get("CARLA_XDG_RUNTIME_DIR", "/tmp/xdg-lead-eval")
startup_sleep = os.environ.get("CARLA_STARTUP_SLEEP", "180")
leaderboard_timeout = os.environ.get("LEADERBOARD_TIMEOUT", "600")
use_b2d_nonroot_carla = os.environ.get("USE_B2D_NONROOT_CARLA", "0") == "1"
carla_run_uid = os.environ.get("CARLA_RUN_UID", "1001")
carla_run_gid = os.environ.get("CARLA_RUN_GID", "1001")
setpriv_bin = os.environ.get("SETPRIV_BIN", "/usr/bin/setpriv")
if use_b2d_nonroot_carla and not setpriv_bin:
    use_b2d_nonroot_carla = False
if use_b2d_nonroot_carla:
    # Match the working B2D local evaluator style: drop privileges and avoid
    # forcing Vulkan/XDG env vars that can make CARLA exit immediately.
    vk_icd = ""
    xdg_runtime_dir = ""

gpu_block = r'''
export LOCAL_GPUS_PER_NODE=${LOCAL_GPUS_PER_NODE:-8}
if [[ -n "${LEAD_ASSIGNED_LOCAL_GPU:-}" ]]; then
    if ! [[ "$LEAD_ASSIGNED_LOCAL_GPU" =~ ^[0-9]+$ ]]; then
        echo "Invalid LEAD_ASSIGNED_LOCAL_GPU=$LEAD_ASSIGNED_LOCAL_GPU" >&2
        exit 1
    fi
    export LOCAL_GPU_ID=$LEAD_ASSIGNED_LOCAL_GPU
elif [[ "$BENCHMARK_ROUTE_ID" =~ ^[0-9]+$ ]]; then
    export LOCAL_GPU_ID=$((10#$BENCHMARK_ROUTE_ID % LOCAL_GPUS_PER_NODE))
else
    export LOCAL_GPU_ID=$(python3 - <<'PYGPU'
import hashlib
import os
rid = os.environ["BENCHMARK_ROUTE_ID"]
n = int(os.environ.get("LOCAL_GPUS_PER_NODE", "8"))
print(int(hashlib.md5(rid.encode("utf-8")).hexdigest(), 16) % n)
PYGPU
)
fi
if [[ "$LOCAL_GPU_ID" -lt 0 || "$LOCAL_GPU_ID" -ge "$LOCAL_GPUS_PER_NODE" ]]; then
    echo "LOCAL_GPU_ID=$LOCAL_GPU_ID is outside [0, $((LOCAL_GPUS_PER_NODE - 1))]" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$LOCAL_GPU_ID
if [[ -n "${LEAD_ASSIGNED_LOCAL_GPU:-}" ]]; then
    export CARLA_GRAPHICS_ADAPTER=$LOCAL_GPU_ID
else
    export CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-$LOCAL_GPU_ID}
fi
if [[ "${LEAD_LOCAL_MAX_PARALLEL_JOBS:-1}" -gt 1 && "$PORT_STRIDE" -le 0 ]]; then
    echo "PORT_STRIDE must be positive for parallel local evaluation" >&2
    exit 1
fi
export PORT=$((PORT_BASE + LOCAL_GPU_ID * PORT_STRIDE))
export TM_PORT=$((TM_PORT_BASE + LOCAL_GPU_ID * PORT_STRIDE))
if [[ "$PORT" -lt 1 || "$PORT" -gt 65535 || "$TM_PORT" -lt 1 || "$TM_PORT" -gt 65535 ]]; then
    echo "Invalid CARLA ports for GPU slot $LOCAL_GPU_ID: PORT=$PORT TM_PORT=$TM_PORT" >&2
    exit 1
fi
echo "Route $BENCHMARK_ROUTE_ID uses CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, CARLA graphics adapter $CARLA_GRAPHICS_ADAPTER, PORT=$PORT, TM_PORT=$TM_PORT"
if [[ "${ROUTE_STARTUP_STAGGER:-0}" != "0" ]]; then
    if [[ -n "${LEAD_ASSIGNED_LOCAL_GPU:-}" ]]; then
        ROUTE_STAGGER_SLEEP=${LEAD_ASSIGNED_LOCAL_STARTUP_DELAY:-0}
    else
        ROUTE_STAGGER_SLEEP=$((LOCAL_GPU_ID * ROUTE_STARTUP_STAGGER))
    fi
    if [[ "$ROUTE_STAGGER_SLEEP" -gt 0 ]]; then
        echo "Route $BENCHMARK_ROUTE_ID sleeps ${ROUTE_STAGGER_SLEEP}s before CARLA/model startup"
        sleep "$ROUTE_STAGGER_SLEEP"
    fi
fi
'''

cleanup_block = r'''
CARLA_PID=""
cleanup_route_carla() {
    set +e
    if [[ -n "${CARLA_PID:-}" ]]; then
        kill -TERM "$CARLA_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$CARLA_PID" 2>/dev/null || true
    fi
    if [[ -n "${PORT:-}" ]]; then
        pgrep -f "CarlaUE4-Linux-Shipping.*carla-rpc-port=$PORT" | xargs -r kill -9 || true
        pgrep -f "CarlaUE4.sh .*carla-rpc-port=$PORT" | xargs -r kill -9 || true
    fi
}
trap cleanup_route_carla EXIT INT TERM
'''

carla_env = []
if os.geteuid() == 0 and fake_uid_lib and not use_b2d_nonroot_carla:
    carla_env.append(f"LD_PRELOAD={fake_uid_lib}")
if vk_icd:
    carla_env.append(f"VK_ICD_FILENAMES={vk_icd}")
if xdg_runtime_dir:
    carla_env.append(f"XDG_RUNTIME_DIR={xdg_runtime_dir}")
carla_prefix = " ".join(carla_env)
if carla_prefix:
    carla_prefix = "env " + carla_prefix + " "
if use_b2d_nonroot_carla:
    carla_prefix += f"{setpriv_bin} --reuid {carla_run_uid} --regid {carla_run_gid} --clear-groups "

for script_path in sorted(scripts_dir.glob("*.sh")):
    text = script_path.read_text()
    lines = text.splitlines(keepends=True)
    patched_lines = []
    inserted = False
    for line in lines:
        patched_lines.append(line)
        if not inserted and line.startswith("export BENCHMARK_ROUTE_ID="):
            patched_lines.append(gpu_block)
            patched_lines.append(cleanup_block)
            inserted = True
    if not inserted:
        raise RuntimeError(f"Could not find BENCHMARK_ROUTE_ID in {script_path}")
    text = "".join(patched_lines)
    text = text.replace("clean_carla.sh", 'echo "Skipping global clean_carla.sh for multi-GPU local eval."')
    text = text.replace("sleep 180", f"sleep {startup_sleep}")
    text = text.replace("--timeout=120", f"--timeout={leaderboard_timeout}")
    text = text.replace(
        "bash $CARLA_ROOT/CarlaUE4.sh --world-port=$PORT -nosound -graphicsadapter=0 -RenderOffScreen &",
        f"{carla_prefix}bash $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=$PORT -graphicsadapter=$CARLA_GRAPHICS_ADAPTER &\nCARLA_PID=$!",
    )
    text = text.replace(
        "pgrep -f \"CarlaUE4-Linux-Shipping.*--world-port=$PORT\"",
        "pgrep -f \"CarlaUE4-Linux-Shipping.*carla-rpc-port=$PORT\"",
    )
    text = text.replace(
        "CarlaUE4-Linux-Shipping not found for port $PORT, exiting...",
        "CarlaUE4-Linux-Shipping not found for rpc port $PORT, exiting...",
    )
    text = text.replace(
        'echo "CarlaUE4-Linux-Shipping not found for rpc port $PORT, exiting..."',
        'echo "CarlaUE4-Linux-Shipping not found for rpc port $PORT, exiting..."; exit 1',
    )
    text = text.replace("export PORT=$(random_free_port.sh)\n", "")
    text = text.replace("export TM_PORT=$(random_free_port.sh)\n", "")
    text = text.replace("CUDA_VISIBLE_DEVICES=0 python3", "CUDA_VISIBLE_DEVICES=$LOCAL_GPU_ID python3")
    script_path.write_text(text)
PYPATCH

echo "Generated LEAD official evaluation scripts: $EVALUATION_OUTPUT_DIR/scripts"
echo "Routes prepared: $route_count"
echo "Parallel local jobs: $LEAD_LOCAL_MAX_PARALLEL_JOBS"
echo "Leaderboard timeout: $LEADERBOARD_TIMEOUT"
echo "Route startup stagger: $ROUTE_STARTUP_STAGGER"
echo "Resume skip finished: $LEAD_RESUME_SKIP_FINISHED"
echo "UniDriveVLA LEAD delta stats: $UNIDRIVEVLA_LEAD_DELTA_STATS"
echo "UniDriveVLA LEAD plan coord: $UNIDRIVEVLA_LEAD_PLAN_COORD"
echo "TEAM_CONFIG=$TEAM_CONFIG"

if [[ "$RUN_EVAL" == "1" ]]; then
    if [[ -n "${CARLA_XDG_RUNTIME_DIR:-}" ]]; then
        mkdir -p "$CARLA_XDG_RUNTIME_DIR"
        if [[ "$USE_B2D_NONROOT_CARLA" == "1" ]]; then
            chown "$CARLA_RUN_UID:$CARLA_RUN_GID" "$CARLA_XDG_RUNTIME_DIR" || true
        fi
        chmod 700 "$CARLA_XDG_RUNTIME_DIR" || true
    fi
    "$PYTHON_BIN" slurm/evaluation/evaluate.py \
        --slurm_dir "$EVALUATION_OUTPUT_DIR/scripts" \
        --job_name "$EXPERIMENT_RUN_ID"
else
    echo "RUN_EVAL=0, generation only."
fi
