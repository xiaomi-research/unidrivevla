# UniDriveVLA-Flash: 2B VisionZip/AdaNorm checkpoint

This directory contains the reproducible training chain for the selected
checkpoint:

```text
unidrivevla_main_fast_stage2_visionzip_keep25_adanorm_action_query_l2_20ep
```

It is the 2B, 20-epoch experiment with VisionZip (`keep25`), perception
AdaNorm, and supervised action queries trained with L2 loss. The action-query
settings are eight queries, L1 weight `0.0`, and L2 weight `1.0`. The launcher
uses eight GPUs, four samples per GPU, 879 iterations per epoch, and 17,580
total iterations. Detection, tracking, map, planning, and ego-status
evaluation are enabled at the final iteration; motion evaluation is disabled,
matching the original run.

NuScenes tracking stays on the evaluation path. Its formatter uses each
annotation's sample token and carries the predicted `instance_ids` into the
tracking `tracking_id`, so tracking is evaluated with the corrected ID handling
used by this runtime.

The `mmdet3d_plugin_flash` package is an isolated copy of the fast runtime
needed by this experiment. It contains the VisionZip/AdaNorm/action-query
implementation and does not replace the release's COC/WASP plugin. The
`transformers_replace` directory contains the Qwen3-VL VisionZip attention
implementation used by the run.

The second experiment,
`stage2_m3_tf32_sdpa_compile_cudagraphs_action_query_decoder_20ep`, is not
included here.

## External prerequisites

The release intentionally does not contain checkpoints, NuScenes data, or
third-party framework checkouts. Set these variables before launching:

```bash
export FLASH_MMCV_ROOT=/path/to/mmcv-1.7.2
export FLASH_MMDET3D_ROOT=/path/to/mmdetection3d-1.0.0rc6
export FLASH_DATA_ROOT=/path/to/nuScenes/data/nuscenes
export FLASH_DATA_INFOS_ROOT=/path/to/nuScenes/data/infos
export FLASH_ANCHOR_ROOT=/path/to/nuScenes/data/kmeans
export FLASH_VLM_PRETRAINED_PATH=/path/to/UniDriveVLA_Stage1_Nuscenes_2B
export FLASH_OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth
```

`FLASH_ANCHOR_ROOT` must contain `kmeans_det_900.npy`, `kmeans_map_100.npy`,
and `kmeans_motion_6.npy`; these dataset assets remain external to the
release.

The active environment must provide PyTorch, MMCV/MMDetection/MMDetection3D,
Transformers with Qwen3-VL support, DeepSpeed, and `peft`. The launcher copies
the bundled Qwen3-VL replacement into the active Transformers installation and
checks that VisionZip support is present before starting distributed training.
Build the Flash deformable-aggregation op in the active CUDA environment before
the first run:

```bash
cd nuScenes/projects/mmdet3d_plugin_flash/ops
FORCE_CUDA=1 python setup.py build_ext --inplace
cd ../../..
```

## Launch

From `nuScenes/`:

```bash
bash projects/configs/UniDriveVLA/UniDriveVLA-Flash/train_stage2_visionzip_keep25_adanorm_action_query_20ep_8gpu.sh
```

`FLASH_CONFIG`, `FLASH_EXP`, `FLASH_SAMPLES_PER_GPU`,
`FLASH_DEEPSPEED_CONFIG`, and `FLASH_QWEN3_REPLACEMENT` can be overridden for
local layouts. The launcher resumes the newest DeepSpeed `iter_*` directory
under the selected work directory when one exists.

No checkpoint, log, work directory, metric file, or dataset is part of this
release directory.
