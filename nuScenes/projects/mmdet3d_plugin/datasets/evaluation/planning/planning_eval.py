import os
import pickle

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon

from mmcv.utils import print_log
from mmdet.datasets import build_dataset, build_dataloader

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


# ---------------------------------------------------------------------------
# Strict metric helpers (original SparseDrive 3D-box approach)
# ---------------------------------------------------------------------------

def check_collision(ego_box, boxes):
    '''
        ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
        boxes: tensor with shape [N, 7]
    '''
    if boxes.shape[0] == 0:
        return False

    # follow uniad, add a 0.5m offset
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly = Polygon([(point[0], point[1]) for point in corners_box[i]])
        if ego_poly.intersects(box_poly):
            return True
    return False


def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0] + 1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]


class PlanningMetric():
    """Strict metric: 3D box Shapely collision, skip incomplete GT (original SparseDrive)."""
    def __init__(self, n_future=6):
        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes, safe_incomplete=False):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)

        available_timesteps = len(fut_boxes)
        for t in range(n_future):
            if safe_incomplete and t >= available_timesteps:
                collision[t] = False
                continue
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes, safe_incomplete=False):
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert B == 1, 'only support bs=1'
        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], fut_boxes, safe_incomplete)
            box_coll = self.evaluate_single_coll(trajs[i], fut_boxes, safe_incomplete)
            box_coll = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))

            obj_coll_sum += gt_box_coll.long()
            obj_box_coll_sum += box_coll.long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        return torch.sqrt(
            (((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)
        )

    def update(self, trajs, gt_trajs, gt_trajs_mask, fut_boxes, safe_incomplete=False):
        assert trajs.shape == gt_trajs.shape
        trajs = trajs.clone()
        gt_trajs = gt_trajs.clone()
        trajs[..., 0] = -trajs[..., 0]
        gt_trajs[..., 0] = -gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(
            trajs[:, :, :2], gt_trajs[:, :, :2], fut_boxes, safe_incomplete
        )
        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total += len(trajs)

    def compute(self):
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2': self.L2 / self.total,
        }


# ---------------------------------------------------------------------------
# Loose metric (UniAD/STP-3 BEV occupancy approach, aligned with evaluation.py)
# ---------------------------------------------------------------------------

def _gen_dx_bx(xbound, ybound):
    dx = torch.tensor([xbound[2], ybound[2]])
    bx = torch.tensor([xbound[0] + xbound[2] / 2.0, ybound[0] + ybound[2] / 2.0])
    nx = int((xbound[1] - xbound[0]) / xbound[2])
    ny = int((ybound[1] - ybound[0]) / ybound[2])
    return dx, bx, (nx, ny)


class PlanningMetricLoose():
    """
    Loose metric: BEV occupancy-map collision, aligned with
    UniDriveVLA/infer_nusc_vlm/eval/evaluation.py + metric.py.
    Does NOT skip incomplete GT samples.
    """
    def __init__(self, n_future=6):
        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future

        # BEV grid parameters (same as metric.py)
        self.dx, self.bx, (self.bev_h, self.bev_w) = _gen_dx_bx(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5]
        )
        self.bev_dimension = np.array([self.bev_h, self.bev_w])

        # Pre-compute ego footprint in pixel space (constant across timesteps)
        pts = np.array([
            [-self.H / 2. + 0.5,  self.W / 2.],
            [ self.H / 2. + 0.5,  self.W / 2.],
            [ self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.numpy()) / self.dx.numpy()
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        self.rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)  # (N_pts, 2)

        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, segmentation):
        """
        traj:         (n_future, 2) in metres, ego-centric
        segmentation: (n_future, bev_h, bev_w) binary occupancy
        Returns:      bool tensor (n_future,)
        """
        n_future = traj.shape[0]
        trajs = traj.view(n_future, 1, 2).clone()
        trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]          # swap x↔y for BEV
        trajs = trajs / self.dx.to(traj.device)
        trajs = trajs.cpu().numpy() + self.rc               # (n_future, N_pts, 2)

        r = np.clip(trajs[:, :, 0].astype(np.int32), 0, self.bev_dimension[0] - 1)
        c = np.clip(trajs[:, :, 1].astype(np.int32), 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            collision[t] = np.any(segmentation[t, r[t], c[t]].cpu().numpy())

        return torch.from_numpy(collision)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        """
        trajs/gt_trajs: (B, n_future, 2)
        segmentation:   (B, n_future, bev_h, bev_w)
        """
        B, n_future, _ = trajs.shape
        # Align with evaluation.py: negate X in evaluate_coll
        trajs    = trajs    * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum     = torch.zeros(n_future)
        obj_box_coll_sum = torch.zeros(n_future)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])

            # obj_col: center-point based (aligned with metric.py)
            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()
            m1 = (
                (yi >= 0) & (yi < self.bev_dimension[0]) &
                (xi >= 0) & (xi < self.bev_dimension[1]) &
                (~gt_box_coll)
            )
            ti = torch.arange(n_future)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            # obj_box_col: full-box based (aligned with metric.py)
            m2 = ~gt_box_coll
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            obj_box_coll_sum[ti[m2]] += box_coll[ti[m2]].long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        return torch.sqrt(
            (((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)
        )

    def update(self, trajs, gt_trajs, gt_trajs_mask, segmentation):
        """
        trajs/gt_trajs:  (B, n_future, 2)
        gt_trajs_mask:   (B, n_future, 2)
        segmentation:    (B, n_future, bev_h, bev_w)
        Note: no X negation here, only in evaluate_coll (aligned with evaluation.py).
        """
        assert trajs.shape == gt_trajs.shape
        trajs     = trajs.clone()
        gt_trajs  = gt_trajs.clone()

        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(
            trajs[:, :, :2], gt_trajs[:, :, :2], segmentation
        )

        self.obj_col     += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2          += L2.sum(dim=0)
        self.total       += len(trajs)

    def compute(self):
        return {
            'obj_col':     self.obj_col     / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2':          self.L2          / self.total,
        }


# ---------------------------------------------------------------------------
# Print helpers — two display conventions
# ---------------------------------------------------------------------------

def _print_uniad_format(planning_results, logger, title):
    """
    UniAD display convention (GPT-Driver):
      columns = 1s, 2s, 3s using RAW per-timestep values at indices 1, 3, 5.
      avg = mean(v[1], v[3], v[5]).
    """
    from prettytable import PrettyTable
    tab = PrettyTable()
    tab.field_names = ["metrics", "1s", "2s", "3s", "avg"]
    metric_dict = {}
    for key, tensor in planning_results.items():
        value = tensor.tolist()
        v1s, v2s, v3s = value[1], value[3], value[5]
        avg = (v1s + v2s + v3s) / 3.0
        metric_dict[key] = avg
        fmt = (lambda v: '%.3f%%' % (v * 100)) if 'col' in key else (lambda v: '%.4f' % v)
        tab.add_row([key, fmt(v1s), fmt(v2s), fmt(v3s), fmt(avg)])
    print_log(f'\n--- {title} ---', logger=logger)
    print_log('\n' + str(tab), logger=logger)
    return metric_dict


def _print_stp3_format(planning_results, logger, title):
    """
    STP-3 display convention (GPT-Driver):
      columns = 1s, 2s, 3s using CUMULATIVE AVERAGES:
        1s = mean(v[0:2]), 2s = mean(v[0:4]), 3s = mean(v[0:6]).
      avg = mean(cumavg@1s, cumavg@2s, cumavg@3s).
    """
    from prettytable import PrettyTable
    tab = PrettyTable()
    tab.field_names = ["metrics", "1s", "2s", "3s", "avg"]
    metric_dict = {}
    for key, tensor in planning_results.items():
        value = tensor.tolist()
        v1s = float(np.mean(value[:2]))
        v2s = float(np.mean(value[:4]))
        v3s = float(np.mean(value[:6]))
        avg = (v1s + v2s + v3s) / 3.0
        metric_dict[key] = avg
        fmt = (lambda v: '%.3f%%' % (v * 100)) if 'col' in key else (lambda v: '%.4f' % v)
        tab.add_row([key, fmt(v1s), fmt(v2s), fmt(v3s), fmt(avg)])
    print_log(f'\n--- {title} ---', logger=logger)
    print_log('\n' + str(tab), logger=logger)
    return metric_dict


def _print_strict_format(planning_results, logger, title):
    """
    SparseDrive strict display: all 6 half-second steps + cumulative avg.
    (kept for completeness; not used by GPT-Driver reporting)
    """
    from prettytable import PrettyTable
    tab = PrettyTable()
    tab.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "avg"]
    metric_dict = {}
    for key, tensor in planning_results.items():
        value = tensor.tolist()
        cumavg = [float(np.mean(value[:i + 1])) for i in range(len(value))]
        avg = (cumavg[1] + cumavg[3] + cumavg[5]) / 3.0
        metric_dict[key] = avg
        fmt = (lambda v: '%.3f%%' % (v * 100)) if 'col' in key else (lambda v: '%.4f' % v)
        tab.add_row([key] + [fmt(v) for v in cumavg] + [fmt(avg)])
    print_log(f'\n--- {title} ---', logger=logger)
    print_log('\n' + str(tab), logger=logger)
    return metric_dict


# ---------------------------------------------------------------------------
# Default paths for BEV segmentation GT
# ---------------------------------------------------------------------------
_SPARSEDRIVE_ROOT = os.path.join(os.path.dirname(__file__), '../../../../..')
_DEFAULT_SEG_PKL  = os.path.join(_SPARSEDRIVE_ROOT, 'data/infos/planing_gt_segmentation_val')


def _load_occ_map(seg_pkl_path, logger, label):
    """Load seg pkl, convert to tensors, apply flip[-1]. Returns dict or None."""
    if not os.path.exists(seg_pkl_path):
        print_log(f'[WARNING] {label} seg pkl not found at {seg_pkl_path}', logger=logger)
        return None
    with open(seg_pkl_path, 'rb') as f:
        raw = pickle.load(f)
    for token in raw.keys():
        if not isinstance(raw[token], torch.Tensor):
            raw[token] = torch.tensor(raw[token])
        raw[token] = torch.flip(raw[token], [-1])   # standard flip for vehicle-only pkl
    print_log(f'Loaded {label} seg maps from {seg_pkl_path} ({len(raw)} tokens)',
              logger=logger)
    return raw


def _get_occupancy(occ_map, token):
    """Retrieve, squeeze batch dim, strip current frame, clip to 6. Returns None if invalid."""
    occ = occ_map.get(token, None)
    if occ is None:
        return None
    if occ.ndim == 4:
        occ = occ.squeeze(0)          # (1,T,200,200) → (T,200,200)
    if occ.shape[0] % 2 == 1:        # strip current-frame prefix
        occ = occ[1:]
    if occ.shape[0] < 6:
        return None
    return occ[:6].unsqueeze(0)       # (1, 6, 200, 200)


def planning_eval(results, eval_config, logger, seg_pkl_path=None):
    """
    Run planning metrics and print results in three display conventions.

    Both GPT-Driver displays (UniAD and STP-3) use the same underlying
    planing_gt_segmentation_val PKL and the same accumulated scores — only
    the display format differs, exactly as in evaluation.py.

      [SparseDrive — STRICT]
        3D-box Shapely collision; skips samples with incomplete GT.
        Display: cumulative avg at each 0.5s step.

      [GPT-Driver — UniAD]  (same data as STP-3)
        BEV-occupancy collision, vehicle-only PKL.
        Display: raw per-timestep values at indices 1,3,5 (= 1s/2s/3s).

      [GPT-Driver — STP-3]  (same data as UniAD)
        BEV-occupancy collision, vehicle-only PKL.
        Display: cumulative average up to 1s/2s/3s.
    """
    dataset = build_dataset(eval_config)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False
    )

    if seg_pkl_path is None:
        seg_pkl_path = _DEFAULT_SEG_PKL
    occ_map = _load_occ_map(seg_pkl_path, logger, 'BEV')

    m_strict = PlanningMetric()
    m_loose  = PlanningMetricLoose() if occ_map is not None else None

    for i, data in enumerate(tqdm(dataloader)):
        sdc_planning      = data['gt_ego_fut_trajs'].cumsum(dim=-2).unsqueeze(1)
        sdc_planning_mask = data['gt_ego_fut_masks'].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        fut_boxes         = data['fut_boxes']
        res               = results[i]
        pred_sdc_traj     = res['img_bbox']['final_planning'].unsqueeze(0)

        gt_traj = sdc_planning[0, :, :6, :2]        # (1, 6, 2)
        gt_mask = sdc_planning_mask[0, :, :6, :2]   # (1, 6, 2)
        pred    = pred_sdc_traj[:, :6, :2]           # (1, 6, 2)

        # ---- [SparseDrive STRICT] 3D-box, skip incomplete GT ----
        if sdc_planning_mask.all():
            m_strict.update(pred.clone(), gt_traj.clone(), gt_mask.clone(), fut_boxes)

        # ---- [GPT-Driver LOOSE] BEV-occupancy, all samples ----
        if m_loose is not None:
            token = dataset.data_infos[i]['token']
            seg = _get_occupancy(occ_map, token)
            if seg is not None:
                m_loose.update(pred.clone(), gt_traj.clone(), gt_mask.clone(), seg)

    # ---- [SparseDrive STRICT] ----
    strict_results = m_strict.compute()
    _print_strict_format(
        strict_results, logger,
        'Planning Metrics  [SparseDrive — STRICT]  '
        '3D-box Shapely collision | skip incomplete GT'
    )
    m_strict.reset()

    # ---- [GPT-Driver] UniAD + STP-3 from same scores ----
    metric_dict = {}
    if m_loose is not None:
        loose_results = m_loose.compute()
        m_loose.reset()

        # UniAD format: raw per-timestep at 1s/2s/3s (indices 1, 3, 5)
        _print_uniad_format(
            loose_results, logger,
            'Planning Metrics  [GPT-Driver — UniAD]  '
            'BEV-occupancy vehicle-only | raw value at 1s/2s/3s | all samples'
        )

        # STP-3 format: cumulative avg up to 1s/2s/3s (same scores, different display)
        # returned as the final metric_dict
        metric_dict = _print_stp3_format(
            loose_results, logger,
            'Planning Metrics  [GPT-Driver — STP-3]  '
            'BEV-occupancy vehicle-only | cumul avg at 1s/2s/3s | all samples'
        )
    else:
        metric_dict = {
            k: float((v[1] + v[3] + v[5]) / 3)
            for k, v in strict_results.items()
        }

    return metric_dict
