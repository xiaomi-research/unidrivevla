# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/xiaomi-research/unidrivevla.git
cd unidrivevla
```

## 2. Create a Conda Environment

```bash
conda create -n unidrivevla python=3.9
conda activate unidrivevla
```

## 3. Install PyTorch

PyTorch >= 2.5.1 is required.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install MMCV (from third-party)

We provide a modified version of MMCV adapted for DeepSpeed. Build it from source:

```bash
cd third_party/mmcv-1.7.2
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MMCV_NO_Compiler_CHECK=1
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
cd ../..
```

## 5. Install MMDetection3D (from third-party)

```bash
cd third_party/mmdetection3d-1.0.0rc6
pip install -e .
cd ../..
```

## 6. Install Project Dependencies

For nuScenes:

```bash
pip install -r requirements_nusc.txt
```

For Bench2Drive:

```bash
pip install -r requirements_b2d.txt
```

## 7. Install Training Dependencies

```bash
# DeepSpeed (for distributed training)
pip install deepspeed>=0.14.0

# PEFT (for LoRA)
pip install peft>=0.7.0
```

## 8. Install UniDriveVLA

```bash
pip install -e .
```

---

## CARLA Setup (for Bench2Drive closed-loop evaluation only)

### 1. Install CARLA 0.9.15

```bash
mkdir -p carla && cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz

tar -xvf CARLA_0.9.15.tar.gz
cd Import && tar -xvf ../AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
```

Add CARLA to your Python path:

```bash
export CARLA_ROOT=/path/to/unidrivevla/carla
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> $CONDA_PREFIX/lib/python3.9/site-packages/carla.pth
```

### 2. Create Multi-Route Splits

```bash
python ./bench2drive/tools/split_xml.py
```

### 3. Configure Paths

Set `project_dir` in your config file:

```python
# projects/configs/unidrivevla/unidrivevla_b2d_stage2.py
project_dir = "/path/to/unidrivevla"
```

Set `WORK_DIR` and `CARLA_ROOT` in the evaluation script:

```bash
# bench2drive/leaderboard/scripts/run_evaluation.sh
export WORK_DIR=/path/to/unidrivevla
export CARLA_ROOT=/path/to/unidrivevla/carla
```

### 4. Run Closed-Loop Evaluation

```bash
bash ./bench2drive/leaderboard/scripts/run_evaluation_multi.sh
```

### 5. Evaluate Results

```bash
python ./bench2drive/tools/statistic_route_json.py
```

### 6. Generate Videos (Optional)

```bash
python ./bench2drive/tools/generate_video.py
```
