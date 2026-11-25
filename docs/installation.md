# Installation

The commands below are for Linux. Use a CUDA toolkit and driver compatible with the PyTorch wheels when building the CUDA extensions. The examples use CUDA 12.1 (`cu121`). See the [PyTorch version matrix](https://pytorch.org/get-started/previous-versions/) for the other available CUDA builds.

```bash
git clone https://github.com/xiaomi-research/unidrivevla.git
cd unidrivevla
```

## Training and nuScenes evaluation

Use Python 3.9 and PyTorch 2.5.1 for the training recipes with FlexAttention.

```bash
conda create -n unidrivevla python=3.9
conda activate unidrivevla
python -m pip install wheel ninja packaging
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Choose the requirements for the benchmark in this environment.
python -m pip install -r requirements/requirements_nusc.txt
# For Bench2Drive training, use requirements/requirements_b2d.txt instead.

python -m pip install deepspeed==0.14.4 peft==0.13.2 transformers==4.57.1
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
TRANSFORMERS_DIR=$(python -c 'from pathlib import Path; import transformers; print(Path(transformers.__file__).parent)')
cp -R qwenvl3/transformers_replace/models/. "$TRANSFORMERS_DIR/models/"
```

Continue with [building the extensions](#build-the-extensions).

## Bench2Drive closed-loop evaluation

Use a separate Python 3.8 environment for the CARLA 0.9.15 evaluation. The CUDA 12.1 index provides Python 3.8 wheels for PyTorch 2.4.1 and torchvision 0.19.1.

```bash
conda create -n unidrivevla_eval python=3.8
conda activate unidrivevla_eval
python -m pip install wheel ninja packaging
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/requirements_b2d.txt
python -m pip install deepspeed==0.14.4 peft==0.13.2
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -m pip install -e third_party/transformers-4.57.1 --no-build-isolation
python -m pip install carla==0.9.15
```

The [vendored Transformers](../third_party/transformers-4.57.1) includes the Python 3.8 changes and the Qwen3-VL visual return values used by UniDriveVLA. The training environment gets the corresponding [Qwen3-VL model files](../qwenvl3/transformers_replace/models) from the copy step above.

Continue with [building the extensions](#build-the-extensions) for Bench2Drive, then install the CARLA simulator and maps. Run these commands from the repository root.

```bash
mkdir -p Bench2Drive/carla
cd Bench2Drive/carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
tar -xf CARLA_0.9.15.tar.gz
mkdir -p Import
tar -xf AdditionalMaps_0.9.15.tar.gz -C Import
bash ImportAssets.sh
export CARLA_ROOT="$PWD"
python -c 'import carla; print(carla.__file__)'
cd ../..
```

## Build the extensions

Run this section in the environment for your benchmark, from the repository root. The benchmark requirements pin MMDetection, MMSegmentation, Shapely 1.8.5 and the Qwen VL utilities. Build the OpenMMLab packages and deformable attention ops after installing those requirements.

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 MMCV_NO_Compiler_CHECK=1 \
    python -m pip install -e third_party/mmcv-1.7.2 --no-build-isolation
python -m pip install -e third_party/mmdetection3d-1.0.0rc6 --no-build-isolation

# For nuScenes:
python -m pip install -e nuScenes/projects/mmdet3d_plugin/ops --no-build-isolation
# For Bench2Drive, use Bench2Drive/projects/mmdet3d_plugin/ops instead.

python -m pip check
python -c 'import torch, mmcv, mmdet, mmdet3d, mmseg, shapely, transformers; import flash_attn, qwen_vl_utils; print(torch.__version__, torch.version.cuda, transformers.__file__)'
```

The released nuScenes configs register `QwenVL3APlanningHead` from the [active dense heads](../nuScenes/projects/mmdet3d_plugin/unidrivevla/dense_heads). `shared-weight_decoder/` holds an experimental implementation with a different model layout. You can use the released configs and checkpoints directly.
