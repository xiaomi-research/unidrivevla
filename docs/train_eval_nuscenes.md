# Training and Evaluation on nuScenes

## Training

Training uses a two-stage pipeline. Run all commands from the `nuScenes/` directory.

### Configuration

Before training, set the following paths in the config file (or export them as environment variables):

| Variable | Description |
|---|---|
| `VLM_PRETRAINED_PATH` | Path to the pretrained VLM weights (e.g. `UniDriveVLA_Nusc_Base_Stage1` for 2B, or `UniDriveVLA_Nusc_Large_Stage1` for 8B) |
| `OCCWORLD_VAE_PATH` | Path to `occvae_latest.pth`, download from [Google Drive](https://drive.google.com/drive/folders/1D1HugOG7JurEqmnQo4XbW_-Ji0chEq-e) |
| `DEEPSPEED_CONFIG` | Path to DeepSpeed config, defaults to `zero_configs/adam_zero1_bf16.json` (already provided) |
| `NUSCENES_DATA_ROOT` | Path to the external nuScenes data tree (images, sweeps, maps, and `v1.0-trainval`) |
| `DATA_INFOS_ROOT` | Path to the annotation pkl files, defaults to `data/infos` |
| `NUSCENES_ANCHOR_ROOT` | Path to external `kmeans_det_900.npy`, `kmeans_map_100.npy`, and `kmeans_motion_6.npy` |
| `DRIVING_JSONL_ROOT` | *(co-training only)* Directory containing driving QA jsonl files packed at `max_length=2048` (LingoQA, DriveLM, Senna) |
| `STAGE1_CHECKPOINT` | *(Stage 2 only)* Path to the Stage 1 checkpoint `.pt` file |

You can either edit the `# ===== User Configuration =====` block at the top of the config file directly, or pass variables at runtime:

```bash
export VLM_PRETRAINED_PATH=/path/to/UniDriveVLA_Nusc_Large_Stage1
export OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth
export NUSCENES_DATA_ROOT=/path/to/nuscenes
export DATA_INFOS_ROOT=data/infos
export NUSCENES_ANCHOR_ROOT=/path/to/nuScenes/data/kmeans
```

---

### Stage 1

The VQA/VLM pretraining stage is unchanged and remains under
`pretrain_script/`. The release does not include the nuScenes camera data or
generated annotation files; provide those externally through
`NUSCENES_DATA_ROOT` and `DATA_INFOS_ROOT`.

### Stage 2

Run the fixed Stage 2 launchers from `nuScenes/`:

```bash
bash train_stage2_coc_language_8gpu_workdirs2.sh
bash train_stage2_coc_weight01_2b_canbus_sft3ep_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_8gpu_workdirs2.sh
bash train_stage2_large_8b_16gpu_2node.sh vision_beta0005_llm_beta0003
bash train_stage2_coc_weight01_8b_canbus_sft3ep_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_8gpu_workdirs2.sh
```

The with-ego launchers require the corresponding three-epoch full-parameter
can-bus SFT output as `VLM_PRETRAINED_PATH`. Use
`pretrain_script/unidrivevla_nusc_2b_full_ego_canbus_pure_sft_3ep.sh` or
`pretrain_script/unidrivevla_nusc_8b_full_ego_canbus_pure_sft_3ep.sh` for that
prerequisite. Stage 3 loads the epoch-20 Stage 2 checkpoint:

```bash
STAGE1_CHECKPOINT=/path/to/2b_stage2/iter_17580/global_step4395/mp_rank_00_model_states.pt \
bash train_unidrivevla_stage2_2b_coc_motion_15ep_8gpu.sh

STAGE2_CHECKPOINT=/path/to/8b_stage2/iter_17580/global_step4395/mp_rank_00_model_states.pt \
bash train_unidrivevla_stage3_8b_large_r128_top16_motion_15ep_16gpu_2node.sh
```

The complete six-recipe table and output directory names are in
`nuScenes/RELEASE_TRAINING.md`.

---

## Evaluation

### Full evaluation and tracking

Use `eval_nusc_8gpu.sh` for every released checkpoint. Set `CFG` to one of the
six `unidrivevla_nusc_*.py` configs and `CKPT` to the external checkpoint:

```bash
CFG=projects/configs/UniDriveVLA/unidrivevla_nusc_2b_stage2.py \
CKPT=/path/to/checkpoint/mp_rank_00_model_states.pt \
OUT_DIR=/path/to/eval/nusc_2b_stage2 \
bash eval_nusc_8gpu.sh
```

Set `NUSCENES_DATA_ROOT`, `DATA_INFOS_ROOT`, `VLM_PRETRAINED_PATH`,
`OCCWORLD_VAE_PATH`, and `COC_JSONL_PATH` for external assets. The evaluator
enables detection, tracking, map, motion, planning, and ego-status tasks and
forces `with_instance_id=True`. Temporal ID ownership transfer in the
published decoder is the tracking-ID repair; there is no separate tracking
variant. The motion-only checks embedded in Stage 3 training intentionally set
`with_tracking=False` and are not the full evaluation.
