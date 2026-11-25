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

Set up the [Python 3.8 CARLA environment](installation.md#bench2drive-closed-loop-evaluation) and prepare the anchors from the [Bench2Drive data guide](data_preparation.md#bench2drive).

### 1. Create Multi-Route Splits

```bash
cd Bench2Drive
python ./bench2drive/tools/split_xml.py
```

### 2. Configure Paths

Set `CARLA_ROOT` to the simulator directory containing `CarlaUE4.sh` if you installed it outside `Bench2Drive/carla`.

```bash
export CARLA_ROOT=/path/to/carla_0.9.15
```

The launcher finds the [bundled evaluator](../Bench2Drive/bench2drive/leaderboard/leaderboard/leaderboard_evaluator.py) from its own location. The second `leaderboard` in the path is the Python package.

### 3. Run Evaluation

Pass the checkpoint and any path overrides as environment variables.

| Variable | Description |
|----------|-------------|
| `WORK_DIR` | Optional absolute path to the `Bench2Drive/` directory |
| `VLM_PRETRAINED_PATH` | Local path to the [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) model used to build the policy |
| `CHECKPOINT` | Path to the model weights (`.pt` file) |
| `SAVE_PATH` | Output directory for evaluation results (default: `evaluation/unidrivevla_b2d`) |

```bash
VLM_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct \
CHECKPOINT=/path/to/UniDriveVLA_Stage3_Bench2drive_2B.pt \
SAVE_PATH=evaluation/unidrivevla_b2d \
bash bench2drive/leaderboard/scripts/run_evaluation_multi_unidrivevla.sh
```

The default launcher uses eight GPUs and the eight splits from step 1. It reports failed workers and keeps a log for each split under `SAVE_PATH`.

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
