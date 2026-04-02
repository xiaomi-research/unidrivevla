import os
import mmcv
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import KMeans
from shapely.geometry import LineString

MAP_CLASSES = ['Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign']
map_element_class = {map_class: map_label for map_label, map_class in enumerate(MAP_CLASSES)}
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

def get_map_info(ann_info, map_infos):
    gt_labels = []

    town_name = ann_info['town_name']
    map_info = map_infos[town_name]

    lane_points = map_info['lane_points']
    lane_types = map_info['lane_types']
    lane_sample_points = map_info['lane_sample_points']

    trigger_volumes_points = map_info['trigger_volumes_points']
    trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']
    trigger_volumes_types = map_info['trigger_volumes_types']
    world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
    ego_xy = np.linalg.inv(world2lidar)[0:2, 3]
    max_distance = 50
    chosed_idx = []

    for idx in range(len(lane_sample_points)):
        single_sample_points = lane_sample_points[idx]
        distance = np.linalg.norm((single_sample_points[:, 0:2] - ego_xy), axis=-1)
        if np.min(distance) < max_distance:
            chosed_idx.append(idx)

    polylines = []
    for idx in chosed_idx:
        if not lane_types[idx] in map_element_class.keys():
            continue
        points = lane_points[idx]
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        points_in_lidar = (world2lidar @ points.T).T
        mask = ((points_in_lidar[:, 0] > point_cloud_range[0]) &
                (points_in_lidar[:, 0] < point_cloud_range[3]) &
                (points_in_lidar[:, 1] > point_cloud_range[1]) &
                (points_in_lidar[:, 1] < point_cloud_range[4]))
        points_in_lidar_range = points_in_lidar[mask, 0:2]
        if len(points_in_lidar_range) > 1:
            polylines.append(LineString(points_in_lidar_range))
            gt_label = map_element_class[lane_types[idx]]
            gt_labels.append(gt_label)

    for idx in range(len(trigger_volumes_points)):
        if not trigger_volumes_types[idx] in map_element_class.keys():
            continue
        points = trigger_volumes_points[idx]
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        points_in_lidar = (world2lidar @ points.T).T
        mask = ((points_in_lidar[:, 0] > point_cloud_range[0]) &
                (points_in_lidar[:, 0] < point_cloud_range[3]) &
                (points_in_lidar[:, 1] > point_cloud_range[1]) &
                (points_in_lidar[:, 1] < point_cloud_range[4]))
        points_in_lidar_range = points_in_lidar[mask, 0:2]
        if mask.all():
            polylines.append(LineString(np.concatenate((points_in_lidar_range, points_in_lidar_range[0:1]), axis=0)))
            gt_label = map_element_class[trigger_volumes_types[idx]]
            gt_labels.append(gt_label)

    map_geoms = {}
    for label, polyline in zip(gt_labels, polylines):
        if label not in map_geoms:
            map_geoms[label] = []
        map_geoms[label].append(polyline)

    return map_geoms

def geom2anno(map_geoms):
    annos = {}
    for label, geom_list in map_geoms.items():
        annos[label] = []
        for geom in geom_list:
            anno = np.array(geom.xy).T
            annos[label].append(anno)
    return annos

K = 100
num_sample = 20
y_ratio = 0.7  # control ratio lanes in y axis

map_path = 'data/infos/b2d_map_infos.pkl'
data_path = 'data/infos/b2d_infos_train.pkl'

map_infos = mmcv.load(map_path)
data_infos = mmcv.load(data_path)[::100]

center = []
for idx in tqdm(range(len(data_infos))):
    ann_info = data_infos[idx]
    map_annos = geom2anno(get_map_info(ann_info, map_infos))

    for cls, geoms in map_annos.items():
        for geom in geoms:
            center.append(geom.mean(axis=0))


center = np.stack(center, axis=0)
center = KMeans(n_clusters=K).fit(center).cluster_centers_
vecs = []
for k in range(K):
    length = np.random.uniform(8, 20, 10)[0]
    if np.random.uniform()<y_ratio:
        delta_y = np.linspace(-length/2, length/2, num_sample)
        delta_x = np.zeros([num_sample])
    else:
        delta_y = np.zeros([num_sample])
        delta_x = np.linspace(-length/2, length/2, num_sample)
    delta = np.stack([delta_x, delta_y], axis=-1)
    vec = center[k, np.newaxis] + delta
    vecs.append(vec)
vecs = np.array(vecs)

for i in range(K):
    x = vecs[i, :, 0]
    y = vecs[i, :, 1]
    plt.plot(x, y, linewidth=1, marker='o', linestyle='-', markersize=2)
plt.savefig(f'data/kmeans/b2d_map_{K}', bbox_inches='tight')
np.save(f'data/kmeans/b2d_map_{K}.npy', vecs)