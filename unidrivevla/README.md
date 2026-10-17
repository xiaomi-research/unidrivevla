# UniDriveVLA

一个 Qwen3-VL 自动驾驶 VLA 训练包。单 profile 配方覆盖 NAVSIM 1.1 / 2.0、nuScenes、Bench2Drive；另有多 profile 混训、PhysicalAI-AV / LEAD 离线缓存适配，以及 NeuroNCAP / HUGSIM 闭环客户端。完整仓库根目录是上一层 `Code/`，这里是 `Code/unidrivevla/` 包目录。以下训练命令在包目录执行；在 `Code/` 执行时使用 `python unidrivevla/train.py` 或 `python unidrivevla/tools/...`。`train.py` / `infer.py` / `tools/*.py` 会将包的父目录放进 `sys.path`，也可在包目录 `pip install -e .`。

官方训练栈保留在上层 `Code/nuScenes/` 与 `Code/Bench2Drive/`，本包不通过它们的 MMCV runner 训练。这里重新实现了 stage 1 / stage 2 配方（见文末差异）；NAVSIM、PhysicalAI-AV 和 LEAD 也不是 main 的官方训练 recipe。main 代码对照固定为 commit `a93c175`。代码级 oracle、合成 CPU 检查和真实数据/硬件验收是不同证据，当前没有完整的 GPU 收敛与性能对照，所以不宣称与 main 全量数值等价或达到最优性能。

当前边界：nuScenes / Bench2Drive 的原生 LiDAR 输入、head 配置和开环 metric helper 有 main 对照；小规模真实样本检查不代表完整训练。多 profile、PhysicalAI-AV、LEAD 是本仓新训练路径；NeuroNCAP / HUGSIM 是已实现协议客户端的闭环评测系统，不是数据集。合成 CPU fixture 不代表许可数据转换、完整训练、闭环 rollout 或官方分数通过。

完整七项需求、源码版本与证据范围、失败历史、使用方式及后续 GPU 验收见 [终版交付与验收说明](docs/ACCEPTANCE.md)。

| 能力 | 当前实现 | 尚未验收 |
|---|---|---|
| nuScenes | main 的 raw LiDAR frame / 固定 prompt cap、训练 profile、strict/loose metric helper；真实 mini train/val 抽样对照 | 完整 trainval GPU 训练、loss/gradient/prediction parity；loose 指标还需官方 segmentation PKL |
| Bench2Drive | main 的 raw LiDAR frame / 固定 prompt cap、数据与 ST-P3 metric helper；真实信息文件样本对照；CARLA Leaderboard adapter | 完整数据 GPU 训练；CARLA rollout 与官方分数 |
| NAVSIM 1.1 / 2.0 | 各自 cache/profile/agent 和对应 PDMS / EPDMS 子进程入口 | 真实训练配置和各版本官方 score replay；两个版本、split 与协议不能混报 |
| 多来源 / 多 profile | 一个共享 Qwen/VLM 与 experts，profile 私有 head；同一 global batch 只采一个来源；单 profile checkpoint 可显式迁移 | 真实多来源 GPU 混训、收敛和最终评测；CPU 双 rank 精确续训通过 |
| PhysicalAI-AV | 七相机 profile、官方 SDK 离线转换器、只读 canonical cache、masked planning loss | 许可数据全量转换、训练与官方评分 |
| LEAD | 官方 SceneLoader/Py123D 离线转换器；输入使用定位观测，未来标签以当前精确 IMU pose 为原点 | 许可数据转换、训练和官方评分 |
| VQA + planning | main JSONL AR 解析/collator、combined / alternate 调度、VQA-only 路径 | 真实语料上的多卡联合训练和生成质量 |
| Bench2Drive CARLA / NeuroNCAP / HUGSIM | 已实现 Leaderboard、HTTP Model Node、FIFO 客户端；有合成协议与坐标检查 | simulator rollout 和官方得分均待跑 |

`nuscenes.yaml` 与 `b2d.yaml` 的输入采用 main 原生 LiDAR 坐标约定；这不表示 MMCV 原始 checkpoint 能直接载入本包。旧 ego-axis UniDriveVLA 权重使用 `nuscenes_ego.yaml` / `b2d_ego.yaml`，不要把两种 frame 的 anchor 混用。新 checkpoint 保存 resolved profile/frame；旧 checkpoint 没有 frame 元数据时显式设置 `agent.checkpoint_frame=ego|lidar`。nuScenes 与 B2D 的默认 profile 还保留 main 固定的 3740 / 3670 prefix 上限和可见 PAD attention 行为。Native nuScenes / B2D 的 main trajectory-AR loss weight 分别是 0.1 / 1.0；`b2d_ego` 保留旧统一配置的 0.1。

## 布局

| 路径 | 内容 |
|---|---|
| `train.py` / `infer.py` | 训练、对已有 checkpoint 做评测 |
| `config/` | Hydra：`train.yaml`，`agent/`，`data/`，`profile/`，`stage/stage1.yaml`，`stage/stage2.yaml` |
| `scripts/train/run_unidrivevla_train.sh` | 单机多卡 `torchrun` 启动器 |
| `scripts/infer/` | NeuroNCAP HTTP 与 HUGSIM FIFO 模型进程启动器 |
| `models/` | 规划头、感知 decoder、VAE；`models/ops/` 是 Deformable Aggregation 的 CUDA 扩展 |
| `adapters/` | 把模型接到 trainer；`navsim_agent.py` 接到 NAVSIM 的缓存和打分入口 |
| `data/` | 各数据集读成同一种 sample；`external.py` 提供 PhysicalAI-AV / LEAD 的安全 cache 边界 |
| `adapters/multidomain.py` | 一个共享 Qwen3-VL / experts，加每个 profile 私有的 processor 和 planning head |
| `distributed/` | DDP / FSDP2、checkpoint |
| `train/` | trainer；`train/optim.py` 的 MasterAdamW 用 fp32 master weight |
| `eval/` | NAVSIM 官方评分入口；nuScenes / Bench2Drive 开环指标；PhysicalAI-AV / LEAD 通用 masked waypoint 指标 |
| `serve.py` / `closedloop/` | NeuroNCAP HTTP Model Node 与 HUGSIM FIFO 推理客户端 |
| `tools/` | 统一 CPU 检查、离线 cache 转换、anchor 和 info 转换。`tools/data_converter/` 是从官方仓库 vendored 的脚本（文件头注明来源和许可） |

数据源只需要实现 [canonical sample contract](data/sample.py)：声明样本坐标系和来源原点，输出各自存在的标注。新的 source 要把 `task_valid` 六个布尔值（planning、det、map、motion、occupancy、geometry）都写入 targets；`false` 表示未知/无标注，真正已标注的空场景用 `true` 加空目标表示。训练 loss 会跳过未知任务，不会把它们当作负样本。旧 source 缺少映射时保留原五类监督兼容；occupancy 只有在存在有效 `occ_path` 时才生效。

NAVSIM 旧 `CacheOnlyDataset` 缓存没有 `task_valid` 字段；NAVSIM loader 会从现存目标数组推断规划、检测、地图、运动与几何标签，并在缺少 `occ_path` 时显式关闭 occupancy 监督。这个转换发生在数据源边界，不改写缓存文件。

nuScenes / Bench2Drive source 仍使用 main 的既有监督字段；在没有 occupancy GT path 的抽样检查或训练配置下，validator 按旧契约保留规划/检测/地图/运动/几何监督并关闭 occupancy。这不会自动合成 occupancy labels。

## 环境

训练环境按已跑通的 conda env 记录（Python 3.9.25）。不要把 mmcv 装进这个环境。

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
# Only when explicitly selecting a FlashAttention module:
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`import cv2` 的版本是 4.11.0，对应 `opencv-python-headless==4.11.0.86`。`requirements.txt` 里的其余 pin 来自同一环境的 `pip freeze` / `import`：transformers 4.57.6、peft 0.17.1、accelerate 1.10.1、hydra-core 1.2.0、omegaconf 2.3.1、numpy 1.26.4、einops 0.8.2、timm 1.0.29、safetensors 0.7.0、qwen-vl-utils 0.0.14、Pillow 11.3.0、shapely 2.0.7、scipy 1.13.1、scikit-learn 1.6.1、pyquaternion 0.9.9、packaging 26.3、tqdm 4.70.0。

不需要官方文档里的 `qwenvl3/transformers_replace`。本仓库用 stock transformers 4.57.6；视觉塔在 deepstack / 最后一层上挂 forward hook，取官方补丁里那个第三返回值的同一批 tensor（`models/unidrivevla/dense_heads/unidrivevla_vlm_qwenvl3.py`）。

FlashAttention 是可选依赖，pin 同时声明在包的 `flash` extra。默认模型注册和 SDPA 路径不加载其 CUDA 扩展；显式构造 FlashAttention 模块时才加载，缺少依赖会在分配 CUDA 参数前报错。`agent.attn_implementation` 默认是 `sdpa`（RTX 3090 / sm86 上 flex attention 在 head_dim 128 没有 kernel）。A100 / H100 可以覆盖成 `flex`。上述命令用于准备新环境，本轮检查不会安装或更改既有服务器环境。

CUDA 扩展（没有它时 `DeformableFeatureAggregation` 走 `grid_sample`）：

```bash
cd models/ops
CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

`TORCH_CUDA_ARCH_LIST=8.6` 是 sm86（例如 RTX 3090）。按自己的 GPU 改。

数据转换用另一个 env，不要和训练混装。已用过的 env 是 Python 3.9.25、mmcv 1.7.2、CPU torch 2.5.1、nuscenes-devkit 1.1.10、numpy 1.23.4、Shapely 1.8.5（见 `requirements-data.txt`）。转换只用 `mmcv.dump`，不要编 mmcv 的 CUDA op（`MMCV_WITH_OPS=0`）。

```bash
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-data.txt
```

## 外部依赖

### NAVSIM devkit

`config/train.yaml` 的 `hydra.searchpath` 含 `pkg://navsim.planning.script.config.common` 和 `pkg://navsim.planning.script.config.training`。`data=navsim` 和包含 NAVSIM 的 `data=multidomain` 会读这些配置（`/default_common` 等）。`data=nuscenes` 可以在没有 devkit 的情况下把配置解析完。NAVSIM 的 `default_common` 把 `train_test_split` 设成必填；启动器传 `train_test_split=${TRAIN_TEST_SPLIT:-navtrain}`。

本仓库对照过的树（只改了下面这个补丁，工作区相对该 commit 是 dirty 的）：

| 树 | 远程 | commit | 分支 |
|---|---|---|---|
| NAVSIM 1.1 | `https://github.com/autonomousvision/navsim` | `0811876c274e8b058ab2be9b3dcd4d37bd23f177` | `v1.1` |
| NAVSIM 2.0 | 同上 | `cf08b01f457a10fbf26f756770ff5f37a4dccb71` | `v2.0` |

nuplan-devkit 是 **1.2.0**（NAVSIM `requirements.txt` 里的 git tag `nuplan-devkit-v1.2`）。克隆后的目录名必须是 `navsim1.1` 或 `navsim2.0`：`eval/navsim.py` 用 `Path(navsim.__file__).parents[1].name` 选择打分脚本。

两个树都要打同一个补丁，否则 `data/navsim.py` 的 `SensorConfig.load_image_path` 不存在，缓存会把图像解码成数组。只改 `navsim/common/dataclasses.py`：

- `SensorConfig` 增加字段 `load_image_path: bool = False`。`get_sensors_at_iteration` 里跳过这个名字，它不是传感器。
- `Cameras.from_camera_dict` 增加参数 `load_image_path: bool = False`。为真时 `Camera.image` 存文件路径，不调用 `Image.open`。
- `AgentInput` 和 `Scene` 里所有调用 `Cameras.from_camera_dict` 的地方传入 `load_image_path=sensor_config.load_image_path`。NAVSIM 2.0 的 `Scene` 有两处（含 synthetic scene）。

`data/navsim.py`、`adapters/navsim_agent.py`、`eval/navsim.py`、`eval/navsim_replay.py` 用到的符号：`SensorConfig`、`AgentInput`、`Trajectory`、`SceneLoader`（2.0 多一个 `synthetic_scenes_path`）、`CacheOnlyDataset`、`Dataset`、`AbstractAgent`（2.0 的 `__init__` 要 `trajectory_sampling`）、`run_dataset_caching.py`、`run_pdm_score.py`，以及 devkit 里存在时的 `run_pdm_score_one_stage.py`（见下文“评测”）。

OpenScene 日志、sensor blobs、nuPlan maps 按 NAVSIM 自己的 `docs/install.md` 下载。环境变量：

| 变量 | 启动器缺省 | 作用 |
|---|---|---|
| `NAVSIM_DEVKIT_ROOT` | 若存在 `../NavSim/navsim${NAVSIM_TREE}` 则用它 | devkit 根目录 |
| `NAVSIM_TREE` | `1.1` | `1.1` 或 `2.0`，只影响缺省 devkit 路径和 cache 目录名 |
| `OPENSCENE_DATA_ROOT` | `/data/yongkangli/NAVSIM/dataset` | `navsim_logs/`、`sensor_blobs/` |
| `NUPLAN_MAPS_ROOT` | `$OPENSCENE_DATA_ROOT/maps` | nuPlan 地图 |
| `NUPLAN_MAP_VERSION` | `nuplan-maps-v1.0` | |
| `NAVSIM_EXP_ROOT` | `/data/yongkangli/NAVSIM/exp` | sample cache、metric cache、实验输出的缺省根 |

这些 `/data/yongkangli/...` 只是作者机器上的缺省值，导出同名变量即可换路径。

### nuScenes / Bench2Drive 的 info

训练读的 pkl 由 vendored 转换脚本生成，不需要把官方 `nuScenes/`、`Bench2Drive/` 树放在旁边。`--code` 仍可指向官方树（和现在作者的调用方式兼容）。

`vad_nuscenes_converter.py` 顶层 import `mmdet3d`。上面的 data env 里没有 mmdet3d。trainer 不读它生成的 `vad_nuscenes_infos_temporal_*.pkl`。

## 数据

路径都用环境变量。不设的话，配置里的缺省就是作者机器上的目录。

| 变量 | 缺省 | 配置项 |
|---|---|---|
| `VLM_PRETRAINED_PATH` | `/data/yongkangli/ckpts/Qwen3-VL-2B-Instruct` | `agent.vlm_path` |
| `UNIDRIVE_EXP_ROOT` | `/data/yongkangli/UniDriveVLA/exp` | `output_dir` 的前缀 |
| `UNIDRIVE_ANCHORS_ROOT` | `/data/yongkangli/UniDriveVLA/anchors` | `profile.anchors_dir` |
| `NUSCENES_INFOS` | `.../nuScenes/data/infos` | `nuscenes.info_dir` |
| `NUSCENES_PATH_ROOT` | `.../nuScenes` | `nuscenes.path_root`（pkl 里图像相对路径的起点，即跑转换时的 cwd） |
| `NUSCENES_ROOT` | `.../nuScenes/data/nuscenes` | `nuscenes.data_root`（`v1.0-trainval/scene.json`，只在有 occ 时读） |
| `NUSCENES_VERSION` | `v1.0-trainval` | `nuscenes.version` |
| `NUSCENES_OCC_ROOT` | 无缺省，必须设置或覆盖成 `null` | `nuscenes.occ_root` |
| `OCCWORLD_VAE_PATH` | `agent` 组缺省 `null`；`data=nuscenes` 会把它改成必填 | `agent.vae_path` |
| `STAGE1_CHECKPOINT` | 无缺省 | `stage=stage2` 时的 `agent.checkpoint_path` |
| `B2D_INFOS` | `.../Bench2Drive/data/infos` | `b2d.info_dir` |
| `B2D_ROOT` | `.../Bench2Drive/data/bench2drive` | `b2d.data_root` |
| `PHYSICALAI_CACHE_ROOT` | `/data/yongkangli/PhysicalAI-AV/unidrivevla_cache` | `physicalai.cache_root` |
| `LEAD_CACHE_ROOT` | `/data/yongkangli/LEAD/unidrivevla_cache` | `lead.cache_root` |

### NAVSIM sample cache

先打上面的 `load_image_path` 补丁，再在训练 env 里：

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/navsim1.1
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NUPLAN_MAPS_ROOT=/path/to/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/exp
NAVSIM_TREE=1.1 DATA=navsim CACHE=1 bash scripts/train/run_unidrivevla_train.sh
```

`CACHE=1` 时启动器先跑：

```text
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
  hydra.searchpath=[pkg://navsim.planning.script.config.common,file://<本仓库>/config] \
  agent=unidrivevla_navsim experiment_name=unidrivevla_caching \
  train_test_split=$TRAIN_TEST_SPLIT cache_path=$CACHE_PATH force_cache_computation=false
```

`TRAIN_TEST_SPLIT` 缺省 `navtrain`。`CACHE_PATH` 缺省 `$NAVSIM_EXP_ROOT/unidrivevla_sample_cache_navsim${NAVSIM_TREE}`。`config/data/navsim.yaml` 里的 `cache_path` 没有版本后缀；启动器会覆盖它。`CACHE_OVERRIDES` 追加到缓存命令上。2.0 把 `NAVSIM_TREE=2.0`，`NAVSIM_DEVKIT_ROOT` 指向 `navsim2.0`。

PDMS 还要 metric cache。两个树的 `scripts/evaluation/run_metric_caching.sh` 参数名不同：

```bash
# NAVSIM 1.1
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache

# NAVSIM 2.0
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

训练时把这个目录传给启动器：`METRIC_CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache`。不传的话，打分脚本自己的配置也缺省到 `$NAVSIM_EXP_ROOT/metric_cache`。

### nuScenes

下载 nuScenes（含 CAN bus expansion）和 [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) 的 `gts/{scene}/{token}/labels.npz`。OccWorld VAE 权重 `occvae_latest.pth` 用官方说明里的文件（`docs` 中的 Google Drive 项 `OCCWORLD_VAE_PATH`）。

在 data env 里，工作目录就是以后的 `NUSCENES_PATH_ROOT`（pkl 里的图像路径相对这里）：

```bash
python tools/convert.py nuscenes -- \
  --root-path ./data/nuscenes \
  --canbus ./data/nuscenes \
  --version v1.0 \
  --out-dir ./data/infos \
  --extra-tag nuscenes
```

`--version v1.0` 时转换脚本会拼成 `v1.0-trainval`，写出 `nuscenes_infos_train.pkl` 和 `nuscenes_infos_val.pkl`。`--version v1.0-mini` 写到 `out-dir/mini/`。trainer 读的就是这两个非 temporal 文件（`data/nuscenes.py` 里的 `nuscenes_infos_{train,val}.pkl`）。时间上的邻帧用 pkl 里的 `scene_token` 自己排，不读 VAD temporal pkl。

官方 `scripts/create_data.sh` 还会跑 temporal 转换。需要那份 pkl 时（本 trainer 不用）：

```bash
python tools/convert.py vad -- \
  --root-path ./data/nuscenes \
  --canbus ./data/nuscenes \
  --version v1.0 \
  --out-dir ./data/infos \
  --extra-tag vad_nuscenes
```

输出 `vad_nuscenes_infos_temporal_{train,val}.pkl`。这一步需要 mmdet3d，data env 里目前没有。

`NUSCENES_OCC_ROOT` 指到 `gts` 这一层（下面是场景名目录）。不要 occ 监督时，训练命令加 `nuscenes.occ_root=null agent.vae_path=null`。

### Bench2Drive

Bench2Drive 0.0.4 是比官方转换脚本 `__main__` 里那份 base split 更新的数据发布。vendored `bench2drive_converter.py` 的 `__main__` 只跑 `generate_map`（info 两行被注释掉），并且读 `./data/splits/bench2drive_base_train_val_split.json`。不要直接跑那个 `__main__`。用 `tools/convert.py`，自己列出 clip 目录名（`B2D_ROOT/v1/<clip>` 下的文件夹名，不含 `v1/`）：

```bash
python tools/convert.py b2d \
  --data-root /path/to/bench2drive \
  --map-root /path/to/bench2drive/maps \
  --out-dir /path/to/infos \
  --train Town01_xxx Town02_xxx \
  --val Town03_xxx
```

写出 `b2d_infos_train.pkl`、`b2d_infos_val.pkl` 和 `b2d_map_infos.pkl`。`--map-root` 下是各地图的 `.npz`。图像路径相对 `--data-root`。0.0.4 若 clip 命名或地图文件和 base 版不同，先确认 `v1/<clip>` 和 `maps/*.npz` 还在，再填 `--train` / `--val`。

### PhysicalAI-AV 与 LEAD 的离线缓存

这两个 reader 只读本仓库的 canonical cache，不会在训练进程里导入官方 SDK。先用数据各自要求的 Python/SDK 版本离线转换；明确给出的 clip/log list 决定 split，cache split 已存在时转换器会报错，不覆盖它。

PhysicalAI-AV 使用[官方 toolkit](https://github.com/NVlabs/physical_ai_av)和[官方数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)，需要 Python >=3.11、已接受许可并在本地缓存的数据。转换器强制 HuggingFace 离线模式，不下载模型或数据：

```bash
PHYSICALAI_REVISION='REPLACE_WITH_IMMUTABLE_DATASET_REVISION'
python tools/convert_physicalai.py \
  --dataset-root /path/to/local-physicalai-files \
  --cache-root /path/to/physicalai-cache \
  --clip-ids /path/to/train-clips.txt --split train \
  --revision "$PHYSICALAI_REVISION"
python tools/convert_physicalai.py \
  --dataset-root /path/to/local-physicalai-files \
  --cache-root /path/to/physicalai-cache \
  --clip-ids /path/to/val-clips.txt --split val \
  --revision "$PHYSICALAI_REVISION"
```

LEAD（[官方仓库](https://github.com/kesai-labs/lead)）转换使用 main 的 Python 3.10 环境和 py123d 0.7.0，读取官方 123D Arrow logs。train/val/test 分别提供互不重叠的 log list；不启动 CARLA：

```bash
python tools/convert_lead.py \
  --data-root /path/to/lead-data \
  --cache-root /path/to/lead-cache \
  --log-list /path/to/train-logs.txt --split train \
  --scene-split normal_view
python tools/convert_lead.py \
  --data-root /path/to/lead-data \
  --cache-root /path/to/lead-cache \
  --log-list /path/to/val-logs.txt --split val \
  --scene-split normal_view
```

转换后在训练环境中使用 `DATA=physicalai` 或 `DATA=lead` 和对应 cache 环境变量。当前转换器只标记实际具有规划监督的样本；不会补造检测、地图、运动或 occupancy 标签。对应 evaluator 报告 mask 上的 `ADE`、`FDE`、`L2_1s`、`L2_2s`、`L2_3s` 和有效样本/waypoint 数。这些是本仓通用开环误差，不是 PhysicalAI-AV 或 LEAD 的官方 benchmark 分数。目前这两类数据没有设定 stage-2 epoch recipe；真实许可数据转换和训练尚未完成。

```bash
PHYSICALAI_CACHE_ROOT=/path/to/physicalai-cache GPUS=4 DATA=physicalai \
  bash scripts/train/run_unidrivevla_train.sh
LEAD_CACHE_ROOT=/path/to/lead-cache GPUS=4 DATA=lead \
  bash scripts/train/run_unidrivevla_train.sh
```

### Anchors

在完整训练集上拟合，不要加 `--stride`（子集的中心不是官方 anchor）。

```bash
python tools/fit_anchors.py data=nuscenes nuscenes.occ_root=null agent.vae_path=null
python tools/fit_anchors.py data=b2d
python tools/fit_anchors.py data=navsim
```

写出 `det_900.npy`、`map_100.npy`、`motion_6.npy` 到当前 profile 的 `profile.anchors_dir`（或 `--out DIR`）；nuScenes / Bench2Drive native LiDAR profile 默认写入 `anchors/native/{nuscenes,b2d}`，旧 ego-frame profile 使用旧目录。Anchor 文件与 frame 绑定，不能跨 frame 复用。已有文件默认拒绝覆盖。NAVSIM 这一条要能 import devkit，并且 `OPENSCENE_DATA_ROOT` / `NAVSIM_EXP_ROOT` 指向已经做好的 sample cache 所用的那套路径；启动器之外跑时，cache 目录用配置里的 `cache_path`，需要的话覆盖 `cache_path=...`。

子集拟合只能用于 smoke construction，不是 main 的正式 anchors。例如在已有 mini / 抽样数据上，把输出写到新的临时目录，并显式标记缺少运动类别的零 anchor：

```bash
python tools/fit_anchors.py --out /tmp/unidrivevla-native-smoke/nuscenes \
  --development-fill-missing-motion data=nuscenes nuscenes.info_dir=/path/to/mini/infos \
  nuscenes.occ_root=null agent.vae_path=null
```

正式配方必须从完整训练数据重新拟合，不要把上述 smoke anchors 搬到正式实验。

`tools/data_stats.py` 用同一套 Hydra 覆盖，打印 ego 位移的 mean / std，对应 profile 里的 `delta_mean` / `delta_std`。profile 里的数已经填好，只有在换数据时才需要重跑。

## 训练

在 `Code/unidrivevla/` 包目录、训练 env 已激活时：

```bash
export VLM_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct
export UNIDRIVE_EXP_ROOT=/path/to/exp
export UNIDRIVE_ANCHORS_ROOT=/path/to/anchors
export NAVSIM_DEVKIT_ROOT=/path/to/navsim1.1
```

Stage 1 训练视觉塔和语言模型 LoRA（`r=64`，`lora_alpha=128`，目标模块 `q,k,v,o,gate,up,down`），任务是 det + map + ego，不开 motion。Stage 2 加载 stage 1 的 `last.ckpt`，打开 motion，冻结 VLM（含 LoRA），`lr=1e-4`。

`train.global_batch=128` 时 `grad_accum = 128 / (batch_size × GPU 数)`。每卡 batch 缺省为官方的 4：8 卡累积 4 步，4 卡累积 8 步。24 GB 卡的冒烟检查显式设置 `dataloader.params.batch_size=1`；这会改变时序 batch 的组成，不能把它当作完整官方配方的等价验证。`train.global_batch=0` 沿用 `train.grad_accum`（缺省 1）。Native nuScenes / B2D 的 scheduler 按 main 的 micro-iteration 频率更新；`warmup_steps=125/250` 是以 optimizer-step 配方为参照，并按实际 `grad_accum` 转成 micro-step 数。累计 4 次时分别是 500/1000 次前向；每个累积窗口最后一个 micro-step 的 LR 用于参数更新，所以第一次更新对应 `t=3`，不是 `t=0`。累计步数随 GPU 数变化时，固定 `global_batch` 仍保持相同 warmup 样本预算。旧 ego-frame、NAVSIM 与多 profile 配方仍按 optimizer step 更新；`batch_size=1` 的 smoke 配置不会复现主配方的批内文本行为。

```bash
# nuScenes stage 1（30 epoch）。occ 与 VAE 不设环境变量就会在解析配置时失败
export NUSCENES_INFOS=/path/to/infos
export NUSCENES_PATH_ROOT=/path/to/nuscenes_repo_cwd
export NUSCENES_ROOT=/path/to/nuscenes
export NUSCENES_OCC_ROOT=/path/to/gts
export OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth
GPUS=8 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh

# nuScenes stage 2（15 epoch）
export STAGE1_CHECKPOINT=/path/to/actual-stage1-run/last.ckpt
GPUS=8 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh stage=stage2

# 不监督 occupancy
GPUS=8 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh \
  nuscenes.occ_root=null agent.vae_path=null

# Bench2Drive stage 1（10 epoch）/ stage 2（5 epoch）
export B2D_INFOS=/path/to/infos
export B2D_ROOT=/path/to/bench2drive
GPUS=8 DATA=b2d bash scripts/train/run_unidrivevla_train.sh
export STAGE1_CHECKPOINT=/path/to/actual-b2d-stage1-run/last.ckpt
GPUS=8 DATA=b2d bash scripts/train/run_unidrivevla_train.sh stage=stage2

# NAVSIM 没有官方 epoch。stage 1 缺省 10。stage 2 的配置里没有 stage2_max_epochs，要用 +stage2_max_epochs
NAVSIM_TREE=1.1 GPUS=8 DATA=navsim bash scripts/train/run_unidrivevla_train.sh
export STAGE1_CHECKPOINT=/path/to/actual-navsim-stage1-run/last.ckpt
NAVSIM_TREE=1.1 GPUS=8 DATA=navsim bash scripts/train/run_unidrivevla_train.sh \
  stage=stage2 +stage2_max_epochs=10
NAVSIM_TREE=2.0 GPUS=8 DATA=navsim bash scripts/train/run_unidrivevla_train.sh

# 冒烟：1 卡，不按 128 的 global batch 累积，只跑 2 个 optimizer step
GPUS=1 MAX_STEPS=2 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh train.global_batch=0 dataloader.params.batch_size=1
```

`experiment_name` 缺省 `unidrivevla_${data_loader}_${UNIDRIVE_RUN_MODE}_${UNIDRIVE_RUN_ID}`，启动器为每个新 run 生成 ID，输出在 `$UNIDRIVE_EXP_ROOT/<experiment_name>/`。Stage 2 要指向实际 Stage 1 输出中的 `last.ckpt`；不要按旧的静态实验名猜路径。也可显式设置一个尚不存在的 `experiment_name`。启动器把额外参数原样传给 Hydra。`MASTER_PORT` 缺省 29517。

不用启动器时（同样的覆盖）：

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29517 train.py data=nuscenes
```

这时要自己保证 `navsim` 和本包都在 `PYTHONPATH` 上。`pip install -e .` 之后本包不必再靠目录名。

### 多 profile 混合训练

`DATA=multidomain` 会自动选择 `agent=multidomain`。默认 `config/data/multidomain.yaml` 同时配置 nuScenes、Bench2Drive 和 NAVSIM；启动前要准备三个来源的数据、缓存、anchors 和 profile 依赖。`NAVSIM_TREE` / `NAVSIM_DEVKIT_ROOT` 仍需设置，即使当前要调的数据源列表未包含 NAVSIM。

```bash
NAVSIM_TREE=1.1 GPUS=4 DATA=multidomain bash scripts/train/run_unidrivevla_train.sh
```

模型只有一份共享的 Qwen3-VL、perception/action experts 和 view-token embedding；每个 profile 保留自己的 planning head、image processor、prompt、坐标/标定解释和预测 horizon。sampler 每次为整个 global batch 抽一个来源，所有 rank 的本地 batch 因而 profile 同质；单个 batch 混入多个 profile 会报错。默认三来源列表和权重见 `config/data/multidomain.yaml`，要增删来源或改权重就改这里，不要把异构样本手工拼在一个 batch 里。

stage 2 可沿用该数据配置的 `stage2_max_epochs` 和 `STAGE1_CHECKPOINT`：

```bash
STAGE1_CHECKPOINT=/path/to/single-profile/last.ckpt \
  NAVSIM_TREE=1.1 GPUS=4 DATA=multidomain \
  bash scripts/train/run_unidrivevla_train.sh stage=stage2 agent.initial_profile=nuscenes
```

从单 profile checkpoint 初始化必须明确 `agent.initial_profile`；只把该 profile 的私有 head 与共享骨干迁入，其他 profile head 从头初始化。已保存的 MultiProfilePolicy checkpoint 则含共享骨干和所有 profile heads。多 profile 架构是本仓新增训练路径，不能据此声称等价于 main 的单数据训练，也未完成真实多数据 GPU 收敛对照。

### 续训

新 `last.ckpt` / `best.ckpt` 和 DCP 状态保留完整模型权重，包括 stage 1 已训练、stage 2 冻结的视觉塔和 LoRA。旧版 stage 2 导出缺失的冻结权重无法凭旧导出文件恢复，需要原 stage 1 checkpoint；旧 DCP 续训仍保留初始化时的冻结权重。新 DCP 带完整状态标记，缺少模型权重会报错，不会静默退回旧格式。

`train.resume=true` 读的是 `$output_dir/checkpoints/step_*/trainer_state.pt`（torch.distributed.checkpoint 的分片，含 optimizer 和 step），不是 `last.ckpt`。如果指定的目录没有完整状态，入口会失败；从头训练时省略 `train.resume=true` 并使用新 run 目录。

```bash
GPUS=8 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh \
  experiment_name=YOUR_EXISTING_RUN_NAME train.resume=true
```

`agent.checkpoint_path` 只加载模型权重（stage 2 用 stage 1 的 `last.ckpt`，motion 模块对不上的键会空着初始化）。它不恢复 optimizer step。

精确的 mid-epoch 恢复是显式 opt-in，单 profile 和 `MultiProfilePolicy` 都使用 `train.exact_resume=true`。checkpoint 保存每 rank 的 RNG、source/sequence sampler 与 VQA task 位置、各 profile 的 temporal runtime state，以及梯度累积窗口内已产生的梯度；恢复时校验训练配置签名、world size、rank 布局和设备类型。planning 与 VQA 的 DataLoader 都必须 `num_workers=0`。FSDP 只在 epoch batch 数能整除 `train.grad_accum`、不需要保存 sharded partial gradients 时可用；否则用 DDP 或调整累积窗口。只有以 `exact_resume=true` 写出的新 resume checkpoint 才能精确恢复；改配置、world size 或设备类型会拒绝恢复。单进程与双 rank CPU 中断/恢复对照通过；完整 GPU 长训恢复尚未验收。默认 `false` 的 resume 仍是 best-effort，不能视为逐 batch 完全复现。

VQA 的独立循环跨 planning epoch 连续推进，精确恢复也保留这个位置。异步 checkpoint 要经过 `wait()` 确认各 rank 的数据与完成标记写入成功后才可恢复；DCP 或磁盘写入失败会报错，不生成有效完成标记。

若需跨优化器 step 留下恢复点，设置 `train.save_every_n_steps`；不设置时 exact resume 只在 epoch 边界保存。checkpoint 目录必须新建且与当前训练 run 对应，`last.ckpt` 本身不是 resume state。训练和推理入口会在 Hydra 写配置前声明独立输出目录，拒绝覆盖旧 run；手工启动多 rank 时，各 rank 必须共享本次新的 `UNIDRIVE_RUN_ID` 和 `UNIDRIVE_INVOCATION_ID`。使用已有 checkpoint 的 `train.resume=true` 需要明确指定已有实验目录。

新 checkpoint 的 `agent_config` 会还原 profile frame。旧 ego-frame nuScenes / B2D checkpoint 示例：

```bash
python infer.py data=nuscenes profile=nuscenes_ego agent.checkpoint_frame=ego \
  agent.checkpoint_path=/path/to/legacy-last.ckpt
python infer.py data=b2d profile=b2d_ego agent.checkpoint_frame=ego \
  agent.checkpoint_path=/path/to/legacy-last.ckpt
```

`agent.checkpoint_frame` 只用于核实没有 frame 元数据的旧 checkpoint；新 checkpoint 的元数据与显式值不一致时会报错。

## VQA 与 planning

`vqa=combined` 保留 main 的 AR hook 语义：每个 planning micro-batch 另取一个 VQA batch，在同一次前向里求和，再按原梯度累积规则更新一次优化器。`vqa=alternate` 则依次执行 planning、VQA micro-iteration，epoch 的迭代数加倍；它会改变优化器与学习率轨迹，不能当作官方联合模式的等价加速。

```bash
python train.py data=nuscenes vqa=combined \
  'vqa.jsonl_paths=[/path/to/lingoqa.jsonl,/path/to/drivelm.jsonl]' vqa.batch_size=2
python train.py data=b2d vqa=combined \
  'vqa.jsonl_paths=[/path/to/b2d_qa.jsonl]' vqa.batch_size=4
python train.py data=b2d vqa=alternate \
  'vqa.jsonl_paths=[/path/to/b2d_qa.jsonl]' vqa.batch_size=4
```

JSONL 使用 main 的 `image`（单路径、路径列表或无图片）和 `conversations`（`from`/`value`）格式。图片路径按运行目录解析，建议写绝对路径。标签仅监督 assistant 回复，padding 为 -100；每个 rank 采用相同任务顺序。VQA 要求 `train_vlm=true` 和可训练的 VLM 参数，stage 2 冻结 VLM 时会拒绝启用。当前端到端 GPU 联合训练尚待验证。

## 评测和推理

`infer.py` 不训练，对 `agent.checkpoint_path` 跑该数据配置里的 evaluator。开环在本进程里跑；NAVSIM PDMS 等本进程放掉 GPU 之后再调官方入口。

额外的验证 loss 和开环预测从清空的时序缓存开始，结束时恢复原来的单/多 profile runtime 和各模块 train/eval 状态；前向异常也执行恢复。这样一次验证不会给下一次评测或训练留下历史。帧间缓存算法仍沿用 main；这项隔离只处理额外评测过程的边界。恢复快照保留张量原设备，尚未测量其完整模型的 GPU 显存和耗时。

新 checkpoint 自带 resolved `agent_config`，`infer.py` 自动恢复单/多 profile 与 motion 架构，并检查与所选数据源是否相容；显式 `agent.vlm_path` 或 `VLM_PRETRAINED_PATH` 可覆盖权重目录。旧 checkpoint 仍需给出原来的 profile 和 stage 配置。推理、NAVSIM agent 与闭环服务都要求完整模型权重，缺失冻结权重或架构不符会报错。stage 1→stage 2 初始化完整恢复已有共享、冻结和 head 权重，只允许新加入的 motion 参数从新初始化开始。

训练导出的新 checkpoint 还保存 `compute_parameter_dtypes`：DDP 记录实际参数 dtype，FSDP 记录分片参数的执行 dtype 和忽略参数的原 dtype。推理严格加载全部权重后校验完整映射，再恢复各浮点参数的精度，保留共享参数身份及 buffer dtype；不允许错误或缺失条目静默降级。旧 checkpoint 没有此映射时保持原加载行为。

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29517 infer.py \
  data=nuscenes agent.checkpoint_path=/path/to/last.ckpt
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29517 infer.py \
  data=b2d agent.checkpoint_path=/path/to/last.ckpt
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29517 infer.py \
  data=navsim agent.checkpoint_path=/path/to/last.ckpt \
  train_test_split=navtest evaluator.split=navtest
```

nuScenes / Bench2Drive 的 evaluator 是开环，按本仓 main 的实际入口分别计算，结果写到 `output_dir/eval/<name>/<tag>/*_metrics.json`；重复 tag 会分配 `_run_NNN` 新目录并保留旧结果。评分子进程非零退出、缺少分数文件或分布式任一 rank 报错都会让训练/推理命令以非零码结束，具体记录在 `eval.log` / `eval_error.txt`。nuScenes strict 使用旋转自车框与原始 LiDAR 未来框；若存在 `NUSCENES_PLANNING_SEG_PKL` 或 `evaluator.seg_pkl_path` 指向的官方 segmentation PKL，还计算 main 的 loose 指标。默认寻找 `nuscenes.info_dir/planing_gt_segmentation_val`（保留官方文件名拼写）；缺少时明确标记 loose 不可用。Bench2Drive 使用当前帧经官方过滤的 actor 及 future/yaw 标签构建 ST-P3 occupancy。PhysicalAI-AV / LEAD 的 evaluator 仅计算 cache 内已标记为有效的 planning waypoint `ADE`、`FDE` 和分秒 `L2`；这些是通用开环诊断，不是官方数据集 benchmark metric。CARLA 闭环使用下面的独立官方 evaluator 入口。

精确复用官方栅格 footprint 需要 `scikit-image==0.24.0`（已加入依赖）。当前服务器的旧 navsim 环境缺少它；本轮 oracle 只在临时 dependency 目录中验证，没有修改该环境。正式运行新开环评测前应满足这项依赖。

NAVSIM：1.1 用 `run_pdm_score.py`（PDMS）。2.0 目录名是 `navsim2.0` 且 split 以 `two_stage` 结尾时用 `run_pdm_score.py`（两阶段 EPDMS）；单阶段 split 在 devkit 有 `run_pdm_score_one_stage.py` 时用它（区分一、二阶段打分的 2.x 版本），否则用 `run_pdm_score.py`（v2.0 本身，即 commit `cf08b01`）。训练过程中的评测在训练 GPU 上出轨迹，CPU 上用 replay agent 打分。`evaluator.split` 缺省 `navtest`。`evaluator.every_n_epochs=0` 表示训练中不评，`on_end=true` 表示结束时评。

### Bench2Drive CARLA 闭环客户端

`closedloop/bench2drive.py` 实现 Leaderboard agent。传感器和静态标定直接读取本仓 main 的 agent，RoutePlanner 与 PID 复用同一份 Bench2Drive devkit；模型加载和预处理使用新包。保留 JPEG quality 20、20 Hz 状态/控制调用和 10 组轮换历史状态，使每组缓存间隔为 0.5 s。多个槽位共享模型参数，不复制十份网络；每个 route 开始清空全部槽位。轨迹由 canonical 的 x前/y左转换成 PID 的 x右/y前。

需要 CARLA 0.9.15、完整 Bench2Drive/Leaderboard 场景依赖，以及与当前 Python 兼容的 CARLA Python API。脚本调用原 `run_evaluation.sh`，不安装依赖；原脚本追加的 py3.7 egg 不能代替 `navsim` Python 3.9 所需的兼容客户端。只有实际 simulator rollout 和官方输出才能算闭环验收。

```bash
WORK_DIR=/data/yongkangli/UniDriveVLA/Bench2Drive \
CARLA_ROOT=/path/to/carla_0.9.15 \
ROUTES=/path/to/official_routes.xml \
CHECKPOINT=/path/to/b2d/last.ckpt \
SAVE_PATH=/path/to/new-b2d-evaluation \
GPU_RANK=0 bash scripts/infer/run_b2d.sh
```

输出目录、record 目录和结果 JSON 必须不存在；脚本拒绝覆盖旧实验。直接交给官方 evaluator 时，设置 `--agent` 为 `closedloop/bench2drive.py`、`--agent-config` 为 checkpoint，并把 `Code` 加入 `PYTHONPATH`、`B2D_LEADERBOARD_ROOT` 指向 devkit 的 `bench2drive/leaderboard`。CPU 检查覆盖主分支像素处理、标定、命令、控制后处理及十槽状态；尚未运行 CARLA 正式评分。

没有 `agent_config` 的旧 Bench2Drive checkpoint 必须显式声明已经核实的坐标来源：旧 ego-axis 配方设置 `UNIDRIVE_CHECKPOINT_FRAME=ego`；原生 LiDAR 配方设置 `lidar`。旧 stage-2 checkpoint 还要设置 `UNIDRIVE_WITH_MOTION=1`。新 checkpoint 默认从 metadata 恢复架构；这两个变量可以不设置，显式值与 metadata 冲突时会拒绝加载。`UNIDRIVE_WITH_MOTION=0` 明确要求无 motion 架构，只接受 `0` 或 `1`。

### NeuroNCAP / HUGSIM 闭环客户端

闭环服务只加载模型做推理，不初始化 trainer 或 DDP。NeuroNCAP 提供官方 `/alive`、`/infer`、`/reset` HTTP endpoints。新 checkpoint 的 `agent_config` 用于恢复 profile 和 stage-2 motion architecture；NeuroNCAP 固定使用 `nuscenes` profile。旧 checkpoint 没有此 metadata 时，必须用 `--checkpoint-frame ego|lidar` 声明已经核实的坐标来源；`ego` 会选择完整的旧 `nuscenes_ego` 预设，包括原来的归一化、锚点和 attention 语义，不只旋转坐标。旧 stage 2 还要加 `--with-motion`。HUGSIM shell 对应设置 `UNIDRIVE_CHECKPOINT_FRAME` 和 `UNIDRIVE_WITH_MOTION=1`。显式值不能覆盖冲突的新 checkpoint metadata。这两个闭环协议当前只接入 nuScenes 六相机配置；PhysicalAI-AV 训练 profile 使用官方七相机，包括 front-tele。NeuroNCAP 请求必须提供官方的 `images`、`ego2world`、16 项 `canbus`、微秒 `timestamp`、`command`，以及每相机 `camera2image` / `camera2ego` 和 `lidar2ego` 标定；图片按官方 renderer 的 base64 torch-saved uint8 tensor 读取，使用 `weights_only=True`。服务只返回 ego-frame `(x forward, y left)` trajectory。

在模型环境中启动 NeuroNCAP Model Node：

```bash
bash scripts/infer/run_neuro_ncap.sh --checkpoint /path/to/last.ckpt --device cuda
# verified legacy ego-axis stage-2 checkpoint without agent_config:
bash scripts/infer/run_neuro_ncap.sh --checkpoint /path/to/stage2.ckpt --device cuda \
  --checkpoint-frame ego --with-motion
```

HUGSIM 客户端从 `info.cam_params` 和运行 simulator 时使用的 camera YAML 核验真实的 `v2c`、`l2c`、内参、`cam_rect`，并从同一 kinematic YAML 读取 wheelbase。它读取官方本地 `obs_pipe` / `plan_pipe` FIFO；不要用网络数据替代此受信任的 simulator pickle。输出转换到 HUGSIM LiDAR `(x right, y forward)`，保留模型实际预测的 horizon，不补造额外 waypoint。HUGSIM 每 0.25 s 更新，nuScenes profile 的历史和 trajectory 按 0.5 s 采样；历史按实测 pose 插值，temporal cache 会收到每个 simulator tick，这个频率差异还没有真实 rollout 验收。把以下命令配置成 HUGSIM 的外部 AD shell；官方 launcher 会传入 GPU id 与 scenario 输出目录：

```bash
export UNIDRIVE_CHECKPOINT=/path/to/last.ckpt
export UNIDRIVE_PYTHON=/path/to/navsim/bin/python
export HUGSIM_CAMERA_CONFIG=/path/to/HUGSIM/configs/sim/nuscenes_camera.yaml
export HUGSIM_KINEMATIC_CONFIG=/path/to/HUGSIM/configs/sim/kinematic.yaml
zsh /path/to/Code/unidrivevla/scripts/infer/run_hugsim.sh 0 /path/to/scenario-output
```

真实启动需要 HUGSIM 自己建立该输出目录和两个 FIFO。CPU 检查用合成图像/标定核验坐标、相机变换、`Done` 结束和 FIFO 类型；当前没有运行 NeuroNCAP 或 HUGSIM simulator，也没有报告它们的 benchmark 分数。[NeuroNCAP Model Node](https://github.com/atonderski/neuro-ncap/blob/main/docs/framework.md) · [HUGSIM closed-loop protocol](https://github.com/hyzhou404/HUGSIM/blob/main/closed_loop.py) · [HUGSIM camera calibration](https://github.com/hyzhou404/HUGSIM/blob/main/configs/sim/nuscenes_camera.yaml)。

## 检查

在 `Code/` 以 `python unidrivevla/tools/...` 跑，或在包目录以 `python tools/...` 跑。统一 runner 默认串行运行所有 CPU 回归，隐藏 CUDA，给每项单独日志和真实退出码；不启动训练、下载数据或计算正式 benchmark 分数：

```bash
python tools/run_checks.py
python tools/run_checks.py --checks multidomain external_sources external_eval closedloop resume
# --output 必须是一个尚不存在的新目录；省略时 runner 使用临时目录
python tools/run_checks.py --checks closedloop --output /tmp/unidrivevla-check-closedloop-1
```

默认 timeout 是每项 300 秒，可用 `--timeout SECONDS` 改。结束时打印每项 PASS/FAIL，并在目录里保存 `<check>.log` 与 `results.json`。返回码非零表示至少一个子检查失败；检查名字见 `tools/run_checks.py` 的 `CHECKS`。

多卡 FSDP2 要求评估时各 rank 的前向次数和 profile 顺序相同。非填充验证分片不满足时会在前向前报错；使用默认 DDP 可保留每个样本只计一次的验证分片。

### 2027 GPU 短冒烟

`tools/check_gpu_closure.py` 的 `preflight` 和 `compose` 是 CPU-only。GPU 冒烟从独立物理 bundle 运行；bundle 中的 `code/` 是这次的源快照，不要把正式 `unidrivevla/` 目录当成 GPU smoke 根目录。先对新的 bundle 做配置与数据检查：

```bash
PY=/data/yongkangli/miniconda3/envs/navsim/bin/python
BUNDLE=/data/yongkangli/UniDriveVLA/setup/codex_gpu_closure_<new-id>
CHECKER=$BUNDLE/code/unidrivevla/tools/check_gpu_closure.py
ANCHORS=/data/yongkangli/UniDriveVLA/setup/codex_native_dev_anchors_<run-id>/anchors
IMAGE=/data/yongkangli/UniDriveVLA/nuScenes/data/nuscenes/samples/CAM_FRONT/<image>.jpg
SUITE_ID=<new-unique-id>

$PY -B $CHECKER --root $BUNDLE --native-anchors $ANCHORS --vqa-image $IMAGE preflight
$PY -B $CHECKER --root $BUNDLE --native-anchors $ANCHORS --vqa-image $IMAGE compose
$PY -B $CHECKER --root $BUNDLE --native-anchors $ANCHORS gpu-gate
```

只有前三步都通过且 `gpu-gate` 报告四张卡空闲时才运行短冒烟：

```bash
$PY -B $CHECKER --root $BUNDLE --native-anchors $ANCHORS --vqa-image $IMAGE \
  --suite-id $SUITE_ID suite --execute
```

每个 case 启动前都会重新检查四张卡无计算进程、每张至少 23,000 MiB 空闲且 GPU 利用率低于 5%；门禁关闭时该 case 不启动。suite 覆盖 nuScenes、Bench2Drive、NAVSIM 1.1/2.0 的 stage 1/2 短步数、frozen-prefix 和共享 Qwen 多 profile/VQA 路径，输出写入 bundle 的新实验目录。NAVSIM stage 2 的一轮设置只验证架构与重载，不是 NAVSIM 正式 recipe；VQA 用验证用合成问答检查模型路径，不代表真实语料质量；所有 GPU smoke 都不是完整训练或官方 benchmark 分数。

| 命令 | 何时跑 |
|---|---|
| `python tools/check_checkpoint.py` | CPU：完整冻结权重导出与实际 initialize 的往返、新旧 DCP 恢复，以及损坏的新格式拒绝加载 |
| `python tools/check_motion_decoder.py` | CPU：直接加载固定 main 的 box/检测/motion decoder，对照 top-k、质量重排、阈值、ID、输出层和坐标；不运行神经网络或 motion benchmark |
| `python tools/check_gradient_reduction.py` | CPU / 两 rank Gloo：显式通信 dtype、缩放顺序、参数转换、Hydra 配方和旧 metadata 回退；不等于完整 DeepSpeed CUDA 对照 |
| `python tools/check_optional_attention.py` | CPU：模拟缺少 FlashAttention 的默认注册，显式选择缺依赖时在 CUDA 参数分配前报错；不测 CUDA kernel |
| `python tools/check_targets.py` | CPU：超出旧 padding 上限、小样本和空样本的 box/map 目标完整保留 |
| `python tools/check_vqa.py` | CPU：main AR tokenizer/collator 对照、图片顺序、assistant 标签、纯 VQA 与联合 loss 梯度 |
| `python tools/check_task_schedule.py` | CPU：联合 loss 求和、VQA 数据循环和任务顺序；不是完整网络多卡验证 |
| `python tools/check_multidomain.py` | CPU、tiny stub：共享骨干/私有 heads、source routing、同质 batch、checkpoint 迁移与参数去重；不构造 Qwen 全模型 |
| `python tools/check_resume.py` | CPU / 小型 DDP fixture：RNG、sampler、task cursor 与梯度窗口恢复约束；不验证多小时真实训练 |
| `python tools/check_eval_schedule.py` | CPU：FSDP2 评估前检查各 rank 前向次数和 profile 顺序；不一致时明确失败，避免 collective 等待 |
| `python tools/check_streams.py` | CPU：显式混合来源切换时清空 temporal cache，避免跨来源读取历史 |
| `python tools/check_mixture.py` | CPU：相容 profile 的多 rank 序列混合、跨 epoch/重复抽样隔离和来源权重；不相容 profile 的单 policy 混合会拒绝，异构 profile 使用 `agent=multidomain` |
| `python tools/check_external_sources.py` | CPU 合成 fixture：PhysicalAI-AV FTheta 投影、LEAD 时间/坐标映射、cache 和 adapter；不读取许可真实数据 |
| `python tools/check_external_eval.py` | CPU 合成轨迹：mask-aware ADE/FDE/L2 的计算与无有效标签处理，不是官方数据集 metric |
| `python tools/check_closedloop.py` | CPU 合成请求：NeuroNCAP HTTP schema/calibration、HUGSIM coordinate/FIFO contracts；不启动 simulator 或产出分数 |
| `python tools/check_b2d_closedloop.py` | CPU：B2D main 传感器/标定、JPEG/像素边界、状态/命令、PID坐标与十槽历史状态；不启动 CARLA |
| `python tools/check_planning_metrics.py` | CPU：对照 main 的 nuScenes strict/loose 与 B2D metric helper；包括转弯、GT 碰撞、无效未来和汇总 |
| `python tools/check_image_pipeline.py` | 改相机预处理之后。CPU。对照官方 ResizeCropFlipImage / PhotoMetricDistortion |
| `python tools/check_master_adamw.py` | 改 `train/optim.py` 之后。有 GPU 用 fused，否则 CPU foreach |
| `python tools/check_sequences.py` | 改 sampler、几何增广或 lr schedule 之后。CPU |
| `python tools/check_deformable.py` | 编完 CUDA op 之后。对不上 `grid_sample` 或 op 没编就失败 |
| `python tools/check_pixels.py [VLM路径]` | 改像素打包之后。路径缺省 `VLM_PRETRAINED_PATH`，再缺省作者机器上的 Qwen3-VL-2B-Instruct。要 GPU 才和 CUDA 路径比；无 GPU 时走 CPU |
| `torchrun --nproc_per_node=1 tools/check_frozen_prefix.py data=nuscenes ...` | 改 stage 2 的 frozen-prefix 缓存之后。要一卡和一份能读的数据 |
| `python tools/check_sources.py data=navsim\|nuscenes\|b2d` | 数据能读、形状和投影对得上之后。`--token` 加 `--old-cache` 可对旧 NAVSIM feature cache |
| `python tools/check_official_nuscenes.py --split both` | 改 nuScenes 数据源或 `prepare_sample` 之后。CPU，训练 env。把官方 `NuScenes3DDataset` 和 train / test pipeline（mmcv、mmdet 用桩模块代替）与本数据源在同一批样本上逐字段对比：图像、投影、框、标签、地图、agent / 自车未来轨迹、指令、ego status、`hist_traj`、`T_global`、时间戳、序列分组、occ 路径 |
| `python tools/check_official_heads.py` | 改 planning head 的 prompt、状态输入或 profile 开关之后。CPU，训练 env。导入官方 nuScenes / Bench2Drive head（mmcv、mmdet 用桩模块代替），对同一批输入比对 tokenized prompt、状态特征，以及 `head_kwargs` 生成的 occ query、历史 token、delta 归一化、关键点来源、log-size clamp。官方仓库根目录由 `NUSCENES_OFFICIAL` / `B2D_OFFICIAL` 指定 |
| `python tools/check_official_b2d.py --samples 6` | 改 Bench2Drive 数据源或 `prepare_sample` 之后。CPU，专用 data env（mmcv 1.7.2）与 `requirements-data.txt` 声明的纯 Python 配置依赖；缺少时可放到隔离目录并通过 `PYTHONPATH` 提供，不要替换或重装 mmcv。对比官方 `Bench2DriveDataset` 和 pipeline，字段同上，另含 `ego_status_mask`；官方源码目录由 `OFFICIAL_B2D` 指定 |

`tools/train_occ_vae.py` 是在 NAVSIM occupancy 网格上训练 `VAERes3D`（`--occ-root` 缺省 `NAVSIM_OCC_ROOT`，`--out` 缺省 `OCCWORLD_VAE_PATH` 或 `occvae_navsim.pth`）。nuScenes 配方用的是外部的 OccWorld `occvae_latest.pth`，不是这个脚本的输出。`--self-check` 只查 loss 和稀疏网格，不读数据。

## 配置开关

- `data=navsim|nuscenes|b2d|multidomain|physicalai|lead`，`stage=stage1|stage2`。`DATA=multidomain` 启动器会自动选 `agent=multidomain`；直接调用 `train.py` 时要同时写 `data=multidomain agent=multidomain`。
- `agent.train_vlm`：stage 1 为 true（视觉塔 + LoRA）。stage 2 为 false。
- `agent.with_motion`：stage 2 为 true。
- `profile.ar_loss_weight`：main nuScenes `0.1`、main Bench2Drive `1.0`；旧 `b2d_ego` checkpoint profile 保持 `0.1`。`profile.enable_knowledge_insulation` 是另一项显式开关，不会随该权重默认开启。
- `profile.lr_microstep`：main native nuScenes/B2D profile 开启，按 micro-batch 更新学习率；旧 ego-frame 与 NAVSIM profile 默认按 optimizer step 更新。warmup micro-step horizon 按运行时 `grad_accum` 计算。
- `agent.lora`：`r=64`，`lora_alpha=128`，`lora_dropout=0`，`bias=none`。
- `agent.lr` / `weight_decay` / `warmup_steps` / `warmup_start` / `min_lr_ratio`：nuScenes stage 1 是 `2e-4`、`1e-7`、125、`1/3`、`1e-3`。Bench2Drive 在 `config/data/b2d.yaml` 里改成 `weight_decay=0.01`、`warmup_steps=250`、`warmup_start=0.01`、`min_lr_ratio=0.05`。`vlm_lr_mult=0.5`。Native profile 的 LR scheduler 按 micro-step 更新以匹配 main；旧 ego-frame profile 保留旧 optimizer-step 行为。
- `agent.attn_implementation`：`sdpa` 或 `flex`。
- `agent.gradient_checkpointing`：stage 1 开。stage 2 关（冻结的 VLM 走缓存前缀）。
- `agent.cache_frozen_prefix`：VLM 冻结时把前缀当 KV cache。`train_vlm=true` 时不用。
- `train.strategy=ddp`（`precision=native` 不能配 FSDP2）。`train.grad_clip=1.0` 对齐 main 默认 DeepSpeed JSON 的有效设置；该运行路径绕过 MMCV 的 `optimizer_config.grad_clip`，不能只按配置文件中的 None/5 判断实际裁剪。
- 原生 nuScenes / Bench2Drive 默认 `train.precision=bf16`、`train.ddp_reduce_dtype=bf16`、`train.ddp_cast_frozen_parameters=true`：DDP 转换全部浮点参数，包括冻结参数，保留 buffer 原 dtype；通信在 BF16 中求和后除以 rank 数。旧 ego 预设、NAVSIM 和混训默认保留 `native` 精度及 `auto` 归约。设置从 `agent.profile` 读取，因此旧 checkpoint metadata 缺少这些字段时仍使用旧默认值。
- `train.ddp_reduce_dtype=auto|native|fp32|bf16`：`auto` 保留旧 hook 选择；`native` 使用 DDP 原生 bucket dtype；另外两项显式选择通信 dtype。`ddp_cast_frozen_parameters` 只作用于 DDP；FSDP2 保留 FP32 master 与 FP32 归约，不用于声称与 ZeRO-1 完全数值等价。
- `train.exact_resume=false` 默认关闭；单 profile 与 MultiProfilePolicy 都实现了运行态 hook，要求 planning/VQA worker 数为 0、使用新 exact checkpoint 且 config/rank/world/device 保持一致。单进程与双 rank CPU 恢复对照通过，GPU 长训恢复尚未验收。
- `train.ema_decay`：三个数据都是 0。
- `dataloader.params.batch_size=4`，`sequence_split=2`（nuScenes / Bench2Drive 按官方 GroupInBatchSampler 把序列切成两段喂给 batch 槽；0 表示打乱帧）。
- `dataloader.eval_batch_size=1` 独立于训练 batch，保持 main 逐帧时序评测；改大此值会改变每个槽看到的时间间隔。
- `evaluator.every_n_epochs` / `evaluator.on_end` / `evaluator.split` / `evaluator.metric_cache_path`。

## 和官方实现的差别

- AR 数据通过 `vqa=combined` 和 `vqa.jsonl_paths` 接入；未配置数据时默认关闭。联合模式复用 main 的 AR 路径；交替模式会改变更新顺序，属于显式实验设置。
- 每卡 micro-batch 默认 4；在小显存卡上改为 1 时，累计 batch 可以保持 128，但时序样本构成和官方 prompt 的批内读取行为不完全相同。
- 原生 nuScenes / Bench2Drive 的 DDP 配方按 BF16 参数、累积与通信对齐 main 的配置；`fp32` 归约仍可显式开启，属于改变舍入路径的设置。FP32 master/moment 的更新由 `MasterAdamW` 完成；完整模型的 CUDA optimizer 顺序、梯度与预测仍需对照，不能由 CPU 归约检查推断全量等价。main 没有锁定 DeepSpeed runtime 版本。
- 原生 LiDAR frame 的 nuScenes stage 2 启用 main 的无参数 `SparseMotionDecoder`，把 `trajs_3d` / `trajs_score` 附加到逐检测结果。Policy 通过 `det` 返回这些结果，并用 `det_coordinate_frame=lidar` 标记原生坐标；公开规划 `pred_traj` 仍为 ego-frame。stage 1、Bench2Drive 和旧 ego 预设保留原输出。开环 planning evaluator 继续只读取规划轨迹；motion 输出字段不等于已取得官方 motion benchmark 分数。
- 坐标系：main nuScenes / Bench2Drive profile 默认保留原始 LiDAR frame（x 右、y 前）和 LiDAR 原点；`nuscenes_ego` / `b2d_ego` 是旧 ego-axis checkpoint 兼容预设。样本 metadata、source batches 与 checkpoint 都会检查 frame；开环 metric 将预测转回官方评测轴。
- Bench2Drive 的时间戳在每个 clip 上加了 `1e4 s × clip 序号`，保证不同 clip 永远落在 instance bank 的 2 s 时窗外；同一 clip 内的差值和官方相同。
- 验证集多给检测 / 地图目标（开环 evaluator 用），官方 test pipeline 不收这些键；共有的键全部一致。
- VLM 默认固定 nuScenes/B2D prompt cap（3740/3670），并保留 main attention 中可见的 PAD key 行为；其他 profile 默认按 batch 长度 padding。`official_attention_mask=false` 或动态 cap 是显式实验变化，不用于声称 main parity。
- nuScenes / Bench2Drive 默认 `official_command_text: true`，保留 main 对 one-hot 做 `command.view(-1)` 后取前 B 个值的读法。设为 false 才按每个样本自己的指令生成文本；这是改变 prompt 的消融设置。每卡 batch 1 无法复现官方 batch 4 的全部批内读取行为。
- Bench2Drive 走官方 B2D head 的行为：速度/加速度 prompt、无历史 token 和 occ query、anchor embed 关键点、log-size clamp `[-5, 5]`、detach 的几何轨迹、eager 推理注意力和 main trajectory-AR loss weight `1.0`。旧 `b2d_ego` profile 保留 `0.1`。`enable_knowledge_insulation` 的 prompt-token CE 是独立配置项，默认关闭。默认 `summed_planning_loss: true` 保留 main 返回 aggregate `loss` 并与各分项重复求和的目标；设为 false 属于改变目标函数的消融。
- nuScenes 的 `hist_traj` 按 VAD 转换脚本的规则（`his_ts=4`，缺历史时重复下一步的位移）由 `scene_token` 顺序现算，不读 VAD temporal pkl；和官方 `LoadAnnotations3D_E2E` 的结果差 3e-6 m。
- EMA 关掉。官方 `EMAHook` 挂在 train epoch 上，`IterBasedRunner` 两种 epoch hook 都不会调，平均权重从未换进模型。`ema_decay=0` 之后，验证和 `last.ckpt` 都是训练权重。
