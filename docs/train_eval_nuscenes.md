# Training and Evaluation on nuScenes

## Training

Training uses a two-stage pipeline. Run all commands from the `nuScenes/` directory.

### Configuration

Before training, set the following paths in the config file (or export them as environment variables):

| Variable | Description |
|---|---|
| `VLM_PRETRAINED_PATH` | Path to the pretrained VLM weights (e.g. `UniDriveVLA_Nusc_Base_Stage1` for 2B, or `UniDriveVLA_Nusc_Large_Stage1` for 8B) |
| `OCCWORLD_VAE_PATH` | Path to `occvae_latest.pth`, download from [Google Drive](https://drive.google.com/drive/folders/1D1HugOG7JurEqmnQo4XbW_-Ji0chEq-e) |
| `DEEPSPEED_CONFIG` | Path to the DeepSpeed JSON. The Stage 1 configs use the corresponding file in `zero_configs/` by default. |
| `DATA_INFOS_ROOT` | Directory for standard and VAD annotation PKLs in Stage 1, defaults to `data/infos`. |
| `DRIVING_JSONL_ROOT` | *(co-training only)* Directory for the default [driving and general VQA manifests](vlm_pretraining.md#stage-1-co-training-manifests). |
| `BATCH_SIZE` | Samples per GPU for 2B Stage 1 without co-training, defaults to 4. |
| `STAGE1_CHECKPOINT` | *(Stage 2 only)* Path to the Stage 1 checkpoint `.pt` file |

You can either edit the `# ===== User Configuration =====` block at the top of the config file directly, or pass variables at runtime:

```bash
export VLM_PRETRAINED_PATH=/path/to/UniDriveVLA_Nusc_Large_Stage1
export OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth
export DATA_INFOS_ROOT=data/infos
```

---

### Stage 1

The launcher uses a single node by default, with up to eight GPUs from the `GPUS` argument. For multi-node training, set the `MLP_WORKER_*` environment variables. Run the [Stage 1 data check](data_preparation.md#4-generate-info-pkl-files) before launching.

Train the perception and planning experts with the [co-training manifests](vlm_pretraining.md#stage-1-co-training-manifests).

```bash
# UniDriveVLA-Base (2B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage1_2b_cotraining.py \
    8 unidrivevla_stage1_2b

# UniDriveVLA-Large (8B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage1_8b_cotraining.py \
    8 unidrivevla_stage1_8b
```

To train without co-training data (recommended for faster training):

```bash
# UniDriveVLA-Base (2B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage1_2b_no_cotraining.py \
    8 unidrivevla_stage1_2b_no_cotrain

# UniDriveVLA-Large (8B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage1_8b_no_cotraining.py \
    8 unidrivevla_stage1_8b_no_cotrain
```

#### Smaller GPU memory budget

Start with one sample per GPU and keep the recipe's BF16 and gradient checkpointing enabled. This example measures the training path on a single GPU with validation disabled.

```bash
BATCH_SIZE=1 bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage1_2b_no_cotraining.py \
    1 unidrivevla_stage1_2b_small --no-validate
```

Check peak GPU memory through an optimizer update. Image token counts, the trainable modules and optimizer state affect the minimum, so a single VRAM figure would depend on the run. The launcher sets `NUM_GPUS` from the topology and DeepSpeed computes the effective batch from the microbatch, GPU count and gradient accumulation. If optimizer state is the bottleneck, an advanced option is DeepSpeed optimizer offloading to CPU. The trainer supports `zero_optimization.offload_optimizer.device: "cpu"` in a separate DeepSpeed JSON passed via `DEEPSPEED_CONFIG`.

> **Tip:** If compute resources are limited, reduce training epochs to 20 by adding `--cfg-options total_epochs=20` to the command.

### Stage 2

Fine-tune with the Stage 1 checkpoint. Set `STAGE1_CHECKPOINT` to the `.pt` file saved under `work_dirs/<stage1_exp>/`:

```bash
export STAGE1_CHECKPOINT=work_dirs/unidrivevla_stage1_2b/iter_XXXX/global_stepXXXX/mp_rank_00_model_states.pt
```

```bash
# UniDriveVLA-Base (2B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py \
    8 unidrivevla_stage2_2b

# UniDriveVLA-Large (8B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage2_8b.py \
    8 unidrivevla_stage2_8b
```

Checkpoints are saved to `work_dirs/<EXP_NAME>/`. Training automatically resumes from the latest checkpoint if interrupted.

---

## Evaluation

### Open-Loop Planning (nuScenes)

```bash
cd nuScenes

bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py \
    /path/to/checkpoint.pth \
    8
```

Results are saved to `work_dirs/<config_name>/`.

### Multi-Task Perception

To evaluate detection, mapping, and motion forecasting, add the corresponding `--eval` flags:

```bash
bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py \
    /path/to/checkpoint.pth \
    8 \
    --eval bbox map motion
```
