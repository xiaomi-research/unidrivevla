# VLM Pretraining for UniDriveVLA

This guide covers the VLM pretraining stage (Stage 0) that produces the pretrained VLM weights used as initialization for Stage 1 training.

## Overview

VLM pretraining uses [ms-swift](https://github.com/modelscope/ms-swift) with Megatron-LM backend to pretrain Qwen3-VL on a mixture of driving-specific and general vision-language datasets. This stage produces the `UniDriveVLA_Nusc_Base_Stage1` / `UniDriveVLA_Nusc_Large_Stage1` checkpoints.

**Note:** Pretraining is optional. You can skip this step and directly use our released Stage 1 checkpoints for Stage 1/2 training.

---

## Data Preparation

### 1. Trajectory Planning Data

Download the trajectory planning QA datasets from HuggingFace:

```bash
# Download from https://huggingface.co/datasets/owl10/UniDriveVLA_Data
cd ${DATASET_ROOT}
git lfs install
git clone https://huggingface.co/datasets/owl10/UniDriveVLA_Data
```

This provides:
- `nuscenes_traj_train.jsonl` — nuScenes planning QA (train)
- `nuscenes_traj_val.jsonl` — nuScenes planning QA (val)
- `b2d_traj_train.jsonl` — Bench2Drive planning QA (train)

### 2. Driving VQA Data (ReCogDrive)

Download the ReCogDrive pretraining datasets:

```bash
# Download from https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining
git clone https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining
```

This includes 11 driving-specific VQA datasets:
- `dataset_nuinstruct.jsonl` — NuInstruct
- `dataset_coda_lm.jsonl` — CODA-LM
- `dataset_drivegpt4.jsonl` — DriveGPT4
- `dataset_drama.jsonl` — DRAMA
- `dataset_lingoqa.jsonl` — LingoQA
- `dataset_drivelm.jsonl` — DriveLM (8B only)
- `dataset_sutd.jsonl` — SUTD-TrafficQA
- `dataset_talk2car.jsonl` — Talk2Car
- `dataset_nuscenes_qa.jsonl` — NuScenes-QA
- `dataset_omnidrive.jsonl` — OmniDrive
- `dataset_senna.jsonl` — Senna
- `dataset_maplm.jsonl` — MapLM

### 3. General VQA Data

**Option A: FineVision (Recommended)**

Download from [HuggingFace FineVision](https://huggingface.co/spaces/HuggingFaceM4/FineVision):

```bash
# Follow instructions at https://huggingface.co/spaces/HuggingFaceM4/FineVision
# Save as dataset_finevision.jsonl
```

**Option B: LLaVA (Alternative)**

You can substitute with LLaVA-Instruct datasets if FineVision is unavailable.

### 4. Bench2Drive-Specific Data (B2D pretraining only)

Download Orion data for Bench2Drive pretraining:

```bash
# Download from https://huggingface.co/datasets/poleyzdk/Chat-B2D
git clone https://huggingface.co/datasets/poleyzdk/Chat-B2D
# Use train_converted_processed.jsonl and output_final_modified_finalview_processed.jsonl
```

---

## Environment Setup

Pretraining requires the `qwenvl3/` directory with ms-swift installed:

```bash
cd qwenvl3
pip install ms-swift
# Ensure swift/cli/_megatron/sft.py is available
```

---

## Running Pretraining

### Configuration

Set the following environment variables before running:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct  # or Qwen3-VL-8B-Instruct
export DATASET_ROOT=/path/to/datasets            # root directory containing all jsonl files
export SWIFT_ROOT=/path/to/ms-swift-main         # directory with tokens.txt and system.txt
export SAVE_DIR=megatron_output/UniDriveVLA_Nusc_Base_Stage1

# Multi-node training (optional, for cluster environments)
export MLP_WORKER_GPU=8           # GPUs per node
export MLP_WORKER_NUM=4           # Number of nodes
export MLP_ROLE_INDEX=0           # Current node rank
export MLP_WORKER_0_HOST=<master_ip>
export MLP_WORKER_0_PORT=29500
```

### nuScenes Pretraining

**2B Model:**

```bash
cd qwenvl3
bash ../pretrain_script/unidrivevla_nusc_2b_stage1.sh
```

**8B Model:**

```bash
cd qwenvl3
bash ../pretrain_script/unidrivevla_nusc_8b_stage1.sh
```

### Bench2Drive Pretraining

**2B Model:**

```bash
cd qwenvl3
bash ../pretrain_script/unidrivevla_b2d_2b_stage1.sh
```

---

## Output

Checkpoints are saved to `${SAVE_DIR}/` in Megatron format. The resulting checkpoint can be used directly as `VLM_PRETRAINED_PATH` in Stage 1 training configs.

---

## Notes

- **Training Time:** Pretraining takes ~3 days on 32x A100 (2B) or 64x A100 (8B) with the full dataset mixture.
- **Dataset Mixture:** The 2B model uses 12 driving datasets + FineVision. The 8B model additionally includes DriveLM.
- **Hyperparameters:** See the shell scripts for detailed training arguments (lr=4e-5, 3 epochs, global_batch_size=128).
- **Resuming:** Training automatically resumes from the latest checkpoint if interrupted.
