import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

CLASSES = [
    'car',
    'van',
    'truck',
    'bicycle',
    'traffic_sign',
    'traffic_cone',
    'traffic_light',
    'pedestrian',
    'others'
]

NameMapping = {
    # =================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    # =========================================

    # =================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',

    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    # =========================================

    # ===================Construction===========
    "static.prop.warningconstruction": 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    # ===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    # =================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}

def lidar2agent(trajs_offset, boxes):
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = - boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum('aij,jka->aik', trajs, rot_mat_T)
    trajs_new = trajs_new[:, 1:]
    return trajs_new

def get_agent_trajs(data_infos, idx, future_frames, sample_rate=1):
    fut_idx_list = range(idx, idx+(future_frames + 1)*sample_rate, sample_rate)

    cur_info = data_infos[idx]

    cur_ids = cur_info['gt_ids']
    cur_boxes = cur_info['gt_boxes']
    world2lidar = cur_info['sensors']['LIDAR_TOP']['world2lidar']

    future_track =  np.zeros((len(cur_boxes), future_frames+1, 2))
    future_mask = np.zeros((len(cur_boxes), future_frames+1))

    for i, (cur_id, cur_box) in enumerate(zip(cur_ids, cur_boxes)):
        for j, fut_idx in enumerate(fut_idx_list):
            if  0 <= fut_idx and fut_idx < len(data_infos):
                adj_info = data_infos[fut_idx]
                adj_ids =  adj_info['gt_ids']

                if adj_info['folder'] != cur_info['folder']:
                    break

                if len(np.where(adj_ids == cur_id)[0])==0:
                    continue

                adj_id = np.where(adj_ids == cur_id)[0][0]
                adj2lidar = world2lidar @ adj_info['npc2world'][adj_id]
                adj_xy = adj2lidar[0:2, 3]

                future_track[i, j, :] = adj_xy
                future_mask[i, j] = 1

    future_track_offset = future_track[:, 1:, :] - future_track[:, :-1, :]
    future_mask_offset = future_mask[:, 1:]

    return future_track_offset, future_mask_offset

K = 6
future_frames = 6
sample_rate = 5
DIS_THRESH = 55

fp = 'data/infos/b2d_infos_train.pkl'
data_infos = mmcv.load(fp)

for info in data_infos:
    for i in range(len(info['gt_names'])):
        if info['gt_names'][i] in NameMapping.keys():
            info['gt_names'][i] = NameMapping[info['gt_names'][i]]

intention = dict()
for i in range(len(CLASSES)):
    intention[i] = []

for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    boxes = info['gt_boxes']
    names = info['gt_names']
    boxes[:, 6] = - (boxes[:, 6] + np.pi / 2)

    gt_fut_trajs, gt_fut_masks = get_agent_trajs(data_infos, idx, future_frames=future_frames, sample_rate=sample_rate)

    labels = []
    for cat in names:
        if cat in CLASSES:
            labels.append(CLASSES.index(cat))
        else:
            labels.append(-1)
    labels = np.array(labels)
    if len(boxes) == 0:
        continue

    for i in range(len(CLASSES)):
        cls_mask = (labels == i)
        box = boxes[cls_mask]

        fut_mask = gt_fut_masks[cls_mask]
        fut_traj = gt_fut_trajs[cls_mask]

        mask = np.logical_and(
            fut_mask.sum(axis=1) == future_frames,
            np.linalg.norm(box[:, :2], axis=1) < DIS_THRESH,
        )
        box = box[mask]
        fut_traj = fut_traj[mask]

        trajs_agent = lidar2agent(fut_traj, box)
        if trajs_agent.shape[0] == 0:
            continue
        intention[i].append(trajs_agent)

clusters = []
for i in range(len(CLASSES)):
    intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, future_frames*2)
    if intention_cls.shape[0] < K:
        continue
    cluster = KMeans(n_clusters=K).fit(intention_cls).cluster_centers_
    cluster = cluster.reshape(-1, future_frames, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
    plt.savefig(f'data/kmeans/b2d_motion_{CLASSES[i]}_{K}', bbox_inches='tight')
    plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data/kmeans/b2d_motion_{K}.npy', clusters)