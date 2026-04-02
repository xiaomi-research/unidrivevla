import os
import mmcv
import pickle
import warnings
import numpy as np
import pyquaternion
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import KMeans
from numpy.polynomial import Polynomial

def command2hot(command, max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot


def get_ego_spatial_trajs(
        data_infos,
        idx,
        sample_points=6,
        sample_strategy=None,
        with_fitting=False,
):
    if sample_strategy["mode"] == 'LID':
        start_point = sample_strategy["start_distance"]
        end_point = sample_strategy["end_distance"]
        index = np.arange(0, sample_points, step=1)
        index_1 = index + 1
        bin_size = (end_point - start_point) / (sample_points * (1 + sample_points))
        sample_distances = start_point + bin_size * index * index_1
    elif sample_strategy["mode"] == "uniform":
        sample_distance = sample_strategy["sample_distance"]
        sample_distances = [i * sample_distance for i in range(1, sample_points + 1)]
    else:
        raise NotImplementedError

    cur_frame = data_infos[idx]
    world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']

    adj_idx = idx
    adj_positions = []
    while True:
        adj_idx += 1
        if 0 <= adj_idx and adj_idx < len(data_infos):
            adj_frame = data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2, 3]
            adj_positions.append(xy)
        else:
            break

    adj_positions = np.array(adj_positions)

    spatial_mask = np.array([0 for _ in range(sample_points)], np.float32)
    spatial_trajs = np.array([(-1, -1) for _ in range(sample_points)], np.float32)
    spatial_offset = np.array([(-1, -1) for _ in range(sample_points)], np.float32)

    if with_fitting and len(adj_positions) > 1:
        x = adj_positions[:, 1]
        y = adj_positions[:, 0]

        y_errs = []
        p_list = []
        for degree in range(1, 6):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                p = Polynomial.fit(x, y, degree)
            y_err = np.linalg.norm(p(x) - y)
            y_errs.append(y_err)
            p_list.append(p)
        p = p_list[np.argmin(y_errs)]

        x_fit = np.linspace(np.min(x), np.max(x), len(x)*10)
        y_fit = p(x_fit)
        adj_positions = np.stack([y_fit, x_fit], axis=1)

    adj_index = -1
    if len(adj_positions) > 0:
        for sample_idx, sample_distance in enumerate(sample_distances):
            pre_diff = sample_distance if sample_idx == 0 else sample_distance - sample_distances[sample_idx - 1]
            adj_diff = np.abs(sample_distance - np.linalg.norm(adj_positions, axis=1))
            min_index = np.argmin(adj_diff)
            if min_index > adj_index and adj_diff[min_index] < pre_diff * 0.25:
                spatial_trajs[sample_idx] = adj_positions[min_index]
                spatial_mask[sample_idx] = 1
                adj_index = min_index

        for i in range(sample_points):
            if spatial_mask[i]:
                spatial_offset[i] = spatial_trajs[i] if i == 0 else spatial_trajs[i] - spatial_trajs[i - 1]

    command = command2hot(cur_frame['command_near'])

    return spatial_offset, spatial_mask, command


N = 6 # num of command
K = 8 # num of trajectory on each command

sample_points = 6
sample_distance = 5

uniform_strategy = dict(
    mode="uniform",
    sample_distance=5,
)

LID_strategy = dict(
    mode="LID",
    start_distance=1,
    end_distance=30,
)

sample_strategy = uniform_strategy

fp = './data/infos/b2d_infos_train.pkl'
data_infos = mmcv.fileio.io.load(fp)
navi_trajs = [[] for _ in range(N)]
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]

    ego_fut_trajs, ego_fut_masks, command = (
        get_ego_spatial_trajs(data_infos, idx, sample_points, sample_strategy, with_fitting=True)
    )

    plan_traj = ego_fut_trajs.cumsum(axis=-2)
    plan_mask = ego_fut_masks
    cmd = command.astype(np.int32)
    cmd = cmd.argmax(axis=-1)
    if not plan_mask.sum() == 6:
        continue
    navi_trajs[cmd].append(plan_traj)

clusters = []
for trajs in navi_trajs:
    trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
    cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])

plt.savefig(f'data/kmeans/b2d_plan_spat_{N}x{K}_{sample_distance}m', bbox_inches='tight')
plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data/kmeans/b2d_plan_spat_{N}x{K}_{sample_distance}m.npy', clusters)