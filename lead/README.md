# LEAD final release

This directory contains only the training and evaluation sources for the
released LEAD final model. Checkpoints, CARLA installations, route XML files,
training data, and evaluation outputs stay external to this repository.

## Final training chain

The released model is produced by this chain:

1. Stage 1 initializes the ordinary-LoRA LEAD model for 3 epochs on 16 GPUs
   (2 nodes x 8 GPUs, micro batch 16, gradient accumulation 2). The final
   Stage 1 checkpoint used to initialize motion training is
   `iter_11358/global_step5679/mp_rank_00_model_states.pt`.
2. The final motion stage runs for 9 epochs on 32 GPUs (4 nodes x 8 GPUs,
   micro batch 16, gradient accumulation 1). Its final checkpoint is
   `iter_17037/global_step17037/mp_rank_00_model_states.pt`.

The initialization and task sequence is therefore:

```text
Qwen3-VL-2B base
  -> 3 epochs: det + map + ego + ego-trajectory planning (no object motion head)
  -> 9 epochs: det + map + ego + ego-trajectory planning + object motion head
```

The first stage is not a motion-free model in the broad planning sense: it
still learns the ego trajectory/planning output. “No motion” refers specifically
to the object future-motion task. The second stage loads the three-epoch model
weights with `LEAD_MOTION_LOAD_FROM`, adds the motion refinement head, and
starts a fresh optimizer/iteration schedule because `LEAD_MOTION_RESUME_FROM`
is empty.

The Stage 3 launcher loads the Stage 1 model with `LEAD_MOTION_LOAD_FROM`.
It deliberately leaves `LEAD_MOTION_RESUME_FROM` empty, so the Stage 1
optimizer and iteration state are not resumed. The 3-epoch and 6-epoch saves
are intermediate saves in this final 9-epoch motion run; they are not separate
released experiments.

The training scripts are:

```text
Bench2Drive/run_lead_stage1_4hz_full_16gpu.sh
Bench2Drive/run_lead_stage3_motion_4hz8_full_32gpu_bs16_accum1_9ep_from_stage2_ds87.sh
```

The four shell files at the top of `Bench2Drive` are the complete final
training chain, split into public launchers and execution layers:

```text
run_lead_stage1_4hz_full_16gpu.sh
  -> run_lead_stage1_8gpu.sh
run_lead_stage3_motion_4hz8_full_32gpu_bs16_accum1_9ep_from_stage2_ds87.sh
  -> run_lead_stage2_motion_4hz8_full_16gpu.sh
     -> run_lead_stage1_8gpu.sh
```

The two `*_8gpu.sh` files are not additional experiments. They contain the
shared single-node DeepSpeed setup called by the 16/32-GPU launchers.

The base file
`Bench2Drive/projects/configs/unidrivevla_b2d_stage1_unified_2b_no_cotraining.py`
is also retained because the final LEAD Stage 1 config inherits its model and
optimizer definitions from that file; it is not a separate training result.

## Environment

The final runs used the Conda environment named `b2d_zoo` (the original machine
had it under `/path/to/b2d_zoo`), with Python 3.8.19 and
pip CUDA wheels (`torch==2.3.1+cu118`). The reproducible package baseline is
in `environment.yml`; it includes the pinned PyTorch, OpenMMLab, Transformers,
DeepSpeed, and evaluation utility versions used for the final checkpoint.

The upstream `3rd_party/fail2drive/environment.yml` is not the environment
used by the final launcher: it describes a separate Python 3.10 Fail2Drive
setup. The original final launcher invokes the `b2d_zoo` interpreter directly,
without activating that upstream environment.

Create the environment with:

```bash
conda env create -f environment.yml
conda activate b2d_zoo
```

The YAML does not copy machine-specific assets into the repository. Training
and evaluation still require external LEAD 4 Hz data manifests, the Qwen3-VL
base, the Qwen3-VL Transformers source, MMDetection3D 1.0.0rc6, and a CARLA
installation. Set `LEAD_QWEN_BASE`, `LEAD_TRANSFORMERS_SRC`,
`LEAD_MMCV_ROOT`, `LEAD_MMDET3D_ROOT`, and `LEAD_CARLA_ROOT` (plus
`LEAD_CARLA_PYTHON_BINDING` for Fail2Drive) to those local paths. The final
machine's `qwen_vl_utils` module is provided by the local evaluation setup and
is synchronized by the evaluation wrapper when needed.

The release includes the two x86-64 custom-op binaries used by the original
Python 3.8/3.9 runs, together with their CUDA sources and `setup.py`. If the
binary does not match a new PyTorch/CUDA installation, rebuild it in place:

```bash
cd Bench2Drive/projects/mmdet3d_plugin/ops
${PYTHON_BIN:-python} setup.py build_ext --inplace
```

Training requires the same environment plus external data. Set
`PROJECT_DIR`, `VLM_PRETRAINED_PATH`, `LOCAL_TRANSFORMERS_SRC`, `MMCV_ROOT`,
and `MMDET3D_ROOT` for the local installation before launching training.

## Bench2Drive evaluation

The six shell files in the `lead` root are also layered rather than six
different evaluations:

```text
launch_latest_4hz_stage3_ds87_epoch9_comfort_full220.sh
  -> run_unidrivevla_lead_stage1_bench2drive_eval_8gpu.sh
     -> run_unidrivevla_lead_stage1_eval_8gpu.sh
        -> run_unidrivevla_lead_eval_8gpu.sh

launch_latest_4hz_stage3_ds87_epoch9_fail2drive_seed0_8gpu.sh
  -> run_unidrivevla_lead_eval_8gpu.sh

launch_latest_4hz_stage3_ds87_epoch9_fail2drive_seed0_single_lidar_creeping_visualization_generalization44_8gpu.sh
  -> launch_latest_4hz_stage3_ds87_epoch9_fail2drive_seed0_8gpu.sh
```

The `launch_*` files are the user-facing final commands. The
`run_unidrivevla_*` files validate routes, set the CARLA/Python environment,
generate local route workers, and aggregate metrics. They are shared by the
Bench2Drive and Fail2Drive final runs, so removing an execution layer would
break the corresponding public launcher.

The final checkpoint is evaluated with the tracking-fixed LEAD runtime in this
directory. Provide the final checkpoint, Qwen base, evaluation Python,
CARLA, the external 220-route XML directory, and the required Python source
trees:

```bash
export LEAD_CHECKPOINT=/path/to/iter_17037/global_step17037/mp_rank_00_model_states.pt
export LEAD_QWEN_BASE=/path/to/Qwen3-VL-2B-Instruct
export LEAD_PYTHON_BIN=/path/to/python
export LEAD_CARLA_ROOT=/path/to/carla
export LEAD_ROUTE_SOURCE=/path/to/bench2drive220_routes
export LEAD_TRAJECTORY_STATS_PATH=/path/to/Bench2Drive/data/lead/infos_4hz_full_motion4hz8_v15/lead_trajectory_stats.json
export LEAD_TRANSFORMERS_SRC=/path/to/transformers/src
export LEAD_MMCV_ROOT=/path/to/mmcv-1.7.2
export LEAD_MMDET3D_ROOT=/path/to/mmdetection3d-1.0.0rc6

LEAD_STAGE3_DS87_EPOCH9_GPU_IDS=0,1,2,3,4,5,6,7 \
bash launch_latest_4hz_stage3_ds87_epoch9_comfort_full220.sh
```

The route directory must contain the 220 split XML files. The launcher uses
the ordinary final motion configuration and the release-local motion anchor;
it does not require a checkpoint or work directory inside this repository.

## Fail2Drive evaluation

Fail2Drive is a separate evaluation dataset and has its own route source and
CARLA installation. The release includes the evaluator, scenario runner, the
required route-map metadata, the single-lidar creeping implementation, and
the same LEAD agent runtime used by Bench2Drive. It does not use the
`official_tfv6_creeping` launcher.

The requested Generalization44 visualization run is:

```bash
export LEAD_CHECKPOINT=/path/to/iter_17037/global_step17037/mp_rank_00_model_states.pt
export LEAD_QWEN_BASE=/path/to/Qwen3-VL-2B-Instruct
export LEAD_PYTHON_BIN=/path/to/python
export LEAD_CARLA_ROOT=/path/to/fail2drive-carla
export LEAD_CARLA_PYTHON_BINDING=/path/to/fail2drive-carla/PythonAPI/carla/py38
export LEAD_ROUTE_SOURCE=/path/to/fail2drive_routes
export LEAD_TRAJECTORY_STATS_PATH=/path/to/Bench2Drive/data/lead/infos_4hz_full_motion4hz8_v15/lead_trajectory_stats.json
export LEAD_TRANSFORMERS_SRC=/path/to/transformers/src
export LEAD_MMCV_ROOT=/path/to/mmcv-1.7.2
export LEAD_MMDET3D_ROOT=/path/to/mmdetection3d-1.0.0rc6

LEAD_STAGE3_DS87_EPOCH9_LONGEST6_GPU_IDS=0,1,2,3,4,5,6,7 \
bash launch_latest_4hz_stage3_ds87_epoch9_fail2drive_seed0_single_lidar_creeping_visualization_generalization44_8gpu.sh
```

The historical `LONGEST6_GPU_IDS` name is accepted by this wrapper for
compatibility and is forwarded to the Fail2Drive GPU selector. The wrapper
sets the creep point source to `single_lidar`, uses traffic-manager seed 0,
and selects the 44 Generalization routes. The underlying evaluator still
validates the complete 200-route Fail2Drive source before selecting that
subset.

Visualization records structured per-frame data in each route's
`metric_info.json`: predicted and reference routes, predicted detections and
map, ground-truth actors, control values, and comfort fields. It does not
change control or score computation and does not imply camera-video recording.

The tracking-ID repair is part of the single `lead/inference`
runtime/evaluator used by both CARLA benchmarks. There are no parallel
tracking implementations in this release. Resume validation requires the
comfort fields, and the Fail2Drive visualization run additionally requires
the visualization schema fields.

The expected reported metric sequences are:

```text
Bench2Drive: 200.27 26.64 87.16 94.50 80.77 84.44 88.33 50.00 91.40 78.99
Fail2Drive:  85.80 85.0 85.40 76.37 68.0 71.94
```

These values are reference results for the external final checkpoint and
matching CARLA/data versions; this source-only release does not ship those
runtime artifacts.
