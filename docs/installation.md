# Installation


## Common Setup

### 1. Clone the Repository

```bash
git clone https://github.com/xiaomi-research/unidrivevla.git
cd unidrivevla
```

### 2. Create a Conda Environment

> **Python version note:**
> - nuScenes training/evaluation and Bench2Drive training support **Python 3.9+**
> - Bench2Drive closed-loop evaluation (CARLA 0.9.15) requires **Python 3.8**
>
> We recommend using two separate environments: Python 3.9 for training and Python 3.8 for Bench2Drive evaluation. Alternatively, you can use Python 3.8 throughout if you only need one environment.

**Training environment (recommended):**
```bash
conda create -n unidrivevla python=3.9
conda activate unidrivevla
```

**Evaluation environment (for Bench2Drive closed-loop only):**
```bash
conda create -n unidrivevla_eval python=3.8
conda activate unidrivevla_eval
```

### 3. Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Transformers

```bash
pip install transformers==4.57.1
```

Then replace the model files with our modified version for Qwen3-VL support:

```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.9/site-packages/transformers/
cp -r qwenvl3/transformers_replace/models ${TRANSFORMERS_DIR}
```

### 6. Install MMCV (from third-party)

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

### 7. Install MMDetection3D (from third-party)

```bash
cd third_party/mmdetection3d-1.0.0rc6
pip install -e .
cd ../..
```

### 8. Install Training Dependencies

```bash
pip install deepspeed==0.14.4
pip install peft
```

---

## nuScenes

### 1. Install Dependencies

```bash
pip install -r requirements/requirements_nusc.txt
```

### 2. Build Custom Ops

```bash
cd nuScenes/projects/mmdet3d_plugin/ops
pip install -e .
cd ../../../..
```

---

## Bench2Drive

> **Note:** Training uses Python 3.9+. Closed-loop evaluation with CARLA 0.9.15 requires a **separate Python 3.8 environment**.

### 1. Install Training Dependencies (Python 3.9+)

```bash
pip install -r requirements/requirements_b2d.txt
```

### 2. Create Evaluation Environment (Python 3.8)

CARLA 0.9.15 requires Python 3.8. Create a separate conda environment for evaluation:

```bash
conda create -n unidrivevla_eval python=3.8
conda activate unidrivevla_eval
pip install -r requirements/requirements_b2d.txt
```

Install MMCV and MMDetection3D from third-party:

```bash
cd third_party/mmcv-1.7.2
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MMCV_NO_Compiler_CHECK=1
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
cd ../..

cd third_party/mmdetection3d-1.0.0rc6
pip install -e .
cd ../..
```

Install our modified transformers (adapted for Python 3.8) from third-party:

```bash
cd third_party/transformers-4.57.1
pip install -e .
cd ../..
```

### 3. Install CARLA 0.9.15

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

