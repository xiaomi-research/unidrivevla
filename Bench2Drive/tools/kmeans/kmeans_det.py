import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

os.makedirs('data/kmeans', exist_ok=True)

K = 900
DIS_THRESH = 55

pkl_path = 'data/infos/b2d_infos_train.pkl'
data_infos = mmcv.load(pkl_path)

center = []
for idx in tqdm(range(len(data_infos))):
    boxes = data_infos[idx]['gt_boxes'][:,:3]
    if len(boxes) == 0:
        continue
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    center.append(boxes[distance < DIS_THRESH])
center = np.concatenate(center, axis=0)
print("start clustering det, may take a few minutes.")
cluster = KMeans(n_clusters=K).fit(center).cluster_centers_

plt.scatter(cluster[:,0], cluster[:,1])
plt.savefig(f'data/kmeans/b2d_det_{K}', bbox_inches='tight')
others = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].repeat(K, axis=0)
cluster = np.concatenate([cluster, others], axis=1)
np.save(f'data/kmeans/b2d_det_{K}.npy', cluster)