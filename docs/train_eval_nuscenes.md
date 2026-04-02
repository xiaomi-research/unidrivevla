# Training and Evaluation on nuScenes

## Training

Training uses a two-stage pipeline. Run all commands from the `nuScenes/` directory.

### Stage 1

Train the perception and planning experts with co-training:

```bash
cd nuScenes

# UniDriveVLA-Base (2B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage1_2b_cotraining.py \
    8 unidrivevla_stage1_2b

# UniDriveVLA-Large (8B)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_stage1_8b_cotraining.py \
    8 unidrivevla_stage1_8b
```

To train without co-training data:

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

### Stage 2

Fine-tune with the stage1 checkpoint (set `load_from` in the config or via `--cfg-options`):

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
