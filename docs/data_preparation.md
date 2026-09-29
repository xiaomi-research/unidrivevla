# Data Preparation

## nuScenes

### 1. Download Dataset

Download the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion. Create a symbolic link:

```bash
cd unidrivevla
mkdir -p nuScenes/data/nuscenes
ln -s /path/to/nuscenes/* /path/to/unidrivevla/nuScenes/data/nuscenes
```

Restore the included CoC training and evaluation JSONL files from their gzip
archives before running the nuScenes recipes:

```bash
python nuScenes/data/nuscenes/restore_alpamayo_jsonl.py
```

### 2. Download Occ3D Annotations

Download the [Occ3D-nuScenes](https://drive.google.com/drive/folders/1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc) occupancy ground truth and place it under `nuScenes/data/nuscenes/gts/`.

### 3. Download Evaluation PKLs

Download additional evaluation-related pkl files from [Google Drive](https://drive.google.com/drive/folders/1Dt7od4aH-tSOjrDGZX9b43I8Fa7TZI3F) and place them under `nuScenes/data/infos/`.

### 4. Prepare Info PKL and Anchor Files

The GitHub release intentionally does not ship the nuScenes sensor data,
annotation PKLs, or k-means assets. Prepare these with the official
nuScenes/SparseDrive data conversion pipeline, or use the matching prepared
assets from the release dataset bundle. Set their locations explicitly:

```bash
export NUSCENES_DATA_ROOT=/path/to/nuScenes/data/nuscenes
export DATA_INFOS_ROOT=/path/to/nuScenes/data/infos
export NUSCENES_ANCHOR_ROOT=/path/to/nuScenes/data/kmeans
```

The anchor directory must contain `kmeans_det_900.npy`,
`kmeans_map_100.npy`, and `kmeans_motion_6.npy`. The release launchers check
these files before training or evaluation. `MOTION_ANCHOR` may override only
the motion file when reproducing a specific experiment.

### 6. File Structure

```
nuScenes/
└── data/
    ├── infos/
    │   ├── nuscenes_infos_train.pkl
    │   ├── nuscenes_infos_val.pkl
    │   ├── planing_gt_segmentation_val/
    │   ├── vad_gt_seg.pkl
    │   ├── vad_nuscenes_infos_temporal_train.pkl
    │   └── vad_nuscenes_infos_temporal_val.pkl
    ├── kmeans/
    │   ├── kmeans_det_900.npy
    │   ├── kmeans_map_100.npy
    │   ├── kmeans_motion_6.npy
    │   └── kmeans_plan_6.npy
    └── nuscenes/
        ├── can_bus/
        ├── gts/
        ├── lidarseg/
        ├── maps/
        ├── nuscenes_caption/
        ├── samples/
        ├── sweeps/
        ├── v1.0-mini/
        ├── v1.0-test/
        └── v1.0-trainval/
```

---

## Bench2Drive

### 1. Download Dataset

Download the [Bench2Drive Base](https://github.com/Thinklab-SJTU/Bench2Drive) dataset following the official [data preparation guide](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/DATA_PREP.md). Create a symbolic link:

```bash
cd unidrivevla/Bench2Drive
mkdir -p data
ln -s /path/to/bench2drive/* ./data/bench2drive
```

### 2. Generate Info PKL Files

```bash
python ./tools/data_converter/bench2drive_converter.py
```

### 3. Generate Anchors by K-means

```bash
bash ./tools/kmeans/kemans.sh
```

### 4. File Structure

```
Bench2Drive/
├── bench2drive/
├── data/
│   ├── bench2drive/
│   │   ├── maps/
│   │   └── v1/
│   ├── infos/
│   │   ├── b2d_infos_train.pkl
│   │   ├── b2d_infos_val.pkl
│   │   ├── b2d_map_anno.pkl
│   │   └── b2d_map_infos.pkl
│   ├── kmeans/
│   │   ├── b2d_det_900.npy
│   │   ├── b2d_map_100.npy
│   │   ├── b2d_motion_6.npy
│   │   ├── b2d_plan_spat_6x8_2m.npy
│   │   └── b2d_plan_spat_6x8_5m.npy
│   └── splits/
├── projects/
└── tools/
```
