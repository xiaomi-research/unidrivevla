# Training and Evaluation on Bench2Drive

## Training

### Stage 1

```bash
cd Bench2Drive
bash tools/dist_train.sh projects/configs/unidrivevla_b2d_stage1_unified_2b.py 8
```

### Stage 2

```bash
cd Bench2Drive
bash tools/dist_train.sh projects/configs/unidrivevla_b2d_stage2_unified_2b.py 8
```

---

## Closed-Loop Evaluation

### 1. Create Multi-Route Splits

```bash
python ./bench2drive/tools/split_xml.py
```

### 2. Configure Paths

Set `WORK_DIR` and `CARLA_ROOT` in the evaluation script:

```bash
# bench2drive/leaderboard/scripts/run_evaluation.sh
export WORK_DIR=/path/to/unidrivevla/Bench2Drive
export CARLA_ROOT=/path/to/unidrivevla/Bench2Drive/carla
```

### 3. Run Evaluation

Before running, set the following variables in `bench2drive/leaderboard/scripts/run_evaluation_multi_unidrivevla.sh`, or pass them as environment variables:

| Variable | Description |
|----------|-------------|
| `WORK_DIR` | Absolute path to the `Bench2Drive/` directory |
| `CHECKPOINT` | Path to the model weights (`.pt` file) |
| `SAVE_PATH` | Output directory for evaluation results (default: `evaluation/unidrivevla_b2d`) |

```bash
cd Bench2Drive
WORK_DIR=/path/to/unidrivevla/Bench2Drive \
CHECKPOINT=/path/to/UniDriveVLA_Stage2_Bench2Drive_2B.pt \
SAVE_PATH=evaluation/unidrivevla_b2d \
bash bench2drive/leaderboard/scripts/run_evaluation_multi_unidrivevla.sh
```

### 4. Compute Metrics

```bash
python ./bench2drive/tools/statistic_route_json.py \
    --route_dir evaluation/unidrivevla_b2d
```

### 5. Generate Videos (Optional)

```bash
python ./bench2drive/tools/generate_video.py \
    --folder_dir evaluation/unidrivevla_b2d
```
