# Data Preparation

## nuScenes

### 1. Download Dataset

Download the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion. Create a symbolic link:

```bash
cd UniDriveVLA/nuScenes
mkdir -p data
ln -s /path/to/nuscenes ./data/nuscenes
```

### 2. Download Occ3D Annotations

Download the [Occ3D-nuScenes](https://drive.google.com/drive/folders/1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc) occupancy ground truth and place it under `data/nuscenes/gts/`.

### 3. Download Evaluation PKLs

Download additional evaluation-related pkl files from [Google Drive](https://drive.google.com/drive/folders/1Dt7od4aH-tSOjrDGZX9b43I8Fa7TZI3F) and place them under `data/infos/`.

### 4. Generate Info PKL Files

```bash
sh scripts/create_data.sh
```

### 5. Generate Anchors by K-means

```bash
sh scripts/kmeans.sh
```

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
    ├── nuscenes/
    │   ├── can_bus/
    │   ├── gts/
    │   ├── lidarseg/
    │   ├── maps/
    │   ├── nuscenes_caption/
    │   ├── samples/
    │   ├── sweeps/
    │   ├── v1.0-mini/
    │   ├── v1.0-test/
    │   └── v1.0-trainval/
    └── others/
        └── motion_anchor_infos_mode6.pkl
```

---

## Bench2Drive

### 1. Download Dataset

Download the [Bench2Drive Base](https://github.com/Thinklab-SJTU/Bench2Drive) dataset following the official [data preparation guide](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/DATA_PREP.md). Create a symbolic link:

```bash
cd UniDriveVLA/Bench2Drive
mkdir -p data
ln -s /path/to/bench2drive ./data/bench2drive
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
