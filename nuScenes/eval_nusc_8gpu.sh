#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${NUSCENES_ROOT:-${SCRIPT_DIR}}
DATA_CWD=${NUSCENES_DATA_CWD:-${CODE_ROOT}}
cd "${DATA_CWD}"
export NUSCENES_ROOT="${CODE_ROOT}"

export PYTHON_BIN=${PYTHON_BIN:-python}
export MMCV_ROOT=${MMCV_ROOT:-${CODE_ROOT}/../third_party/mmcv-1.7.2}
export MMDET3D_ROOT=${MMDET3D_ROOT:-${CODE_ROOT}/../third_party/mmdetection3d-1.0.0rc6}
export PYTHONPATH=${MMCV_ROOT}:${MMDET3D_ROOT}:${CODE_ROOT}:${PYTHONPATH:-}

export OCCWORLD_VAE_PATH=${OCCWORLD_VAE_PATH:-/path/to/occvae_latest.pth}
export DATA_INFOS_ROOT=${DATA_INFOS_ROOT:-${DATA_CWD}/data/infos}
export NUSCENES_DATA_ROOT=${NUSCENES_DATA_ROOT:-${DATA_CWD}/data/nuscenes}
export NUSCENES_ANCHOR_ROOT=${NUSCENES_ANCHOR_ROOT:-${DATA_CWD}/data/kmeans}
export VLM_PRETRAINED_PATH=${VLM_PRETRAINED_PATH:-/path/to/UniDriveVLA_Stage1_Nuscenes_2B}
export COC_JSONL_PATH=${COC_JSONL_PATH:-${CODE_ROOT}/data/nuscenes/alpamayo_coc_val_with_nav.jsonl}
export VLA_ATTN_IMPLEMENTATION=${VLA_ATTN_IMPLEMENTATION:-sdpa}

CFG=${CFG:-${CODE_ROOT}/projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage2.py}
CKPT=${CKPT:-/path/to/stage2_checkpoint.pt}
OUT_DIR=${OUT_DIR:-${CODE_ROOT}/eval_results/nusc_stage2}

# Stage 2 and the can-bus Stage 2 recipes do not contain an object-motion
# head.  Keep their evaluation runnable while enabling motion automatically
# for the two Stage 3 recipes.  Users can still override this explicitly.
if [[ -z "${EVAL_WITH_MOTION+x}" ]]; then
    case "$(basename "${CFG}")" in
        *stage3*.py) EVAL_WITH_MOTION=1 ;;
        *) EVAL_WITH_MOTION=0 ;;
    esac
fi

export NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_PORT=${MASTER_PORT:-28951}

if [[ ! -f "${CFG}" ]]; then
    echo "ERROR: config not found: ${CFG}" >&2
    exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 3
fi
if [[ ! -f "${COC_JSONL_PATH}" ]]; then
    echo "ERROR: CoC validation JSONL not found: ${COC_JSONL_PATH}" >&2
    exit 4
fi
for anchor_file in kmeans_det_900.npy kmeans_map_100.npy kmeans_motion_6.npy; do
    if [[ ! -f "${NUSCENES_ANCHOR_ROOT}/${anchor_file}" ]]; then
        echo "ERROR: nuScenes anchor not found: ${NUSCENES_ANCHOR_ROOT}/${anchor_file}" >&2
        exit 5
    fi
done

mkdir -p "${OUT_DIR}/logs"
T=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUT_DIR}/logs/eval_nusc.${T}.log"

{
    echo "Config: ${CFG}"
    echo "Checkpoint: ${CKPT}"
    echo "Output dir: ${OUT_DIR}"
    echo "NUM_GPUS: ${NUM_GPUS}"
    echo "Tracking: instance IDs enabled; temporal ID ownership transfer is built into eval decoder"
    echo "Motion evaluation: ${EVAL_WITH_MOTION} (auto-selected from config basename unless overridden)"
    echo "CoC validation JSONL: ${COC_JSONL_PATH}"
} | tee "${LOG_FILE}"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    "${CODE_ROOT}/tools/test.py" \
    "${CFG}" \
    "${CKPT}" \
    --launcher pytorch \
    --out "${OUT_DIR}/results.pkl" \
    --eval bbox \
    --show-dir "${OUT_DIR}" \
    --cfg-options \
        data.samples_per_gpu=1 \
        data.workers_per_gpu=4 \
        model.planning_head.attn_implementation="${VLA_ATTN_IMPLEMENTATION}" \
        model.planning_head.enable_coc=True \
        model.planning_head.enable_coc_context=False \
        model.planning_head.enable_coc_generate=True \
        model.planning_head.coc_jsonl_path="${COC_JSONL_PATH}" \
        model.planning_head.unified_decoder_cfg.with_instance_id=True \
        evaluation.eval_mode.with_det=True \
        evaluation.eval_mode.with_tracking=True \
        evaluation.eval_mode.with_map=True \
        evaluation.eval_mode.with_motion="${EVAL_WITH_MOTION}" \
        evaluation.eval_mode.with_planning=True \
        evaluation.eval_mode.with_ego_status=True \
    2>&1 | tee -a "${LOG_FILE}"
