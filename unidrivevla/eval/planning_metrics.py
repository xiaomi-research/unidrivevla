"""CPU planning metrics adapted from this repository's active main evaluators.

nuScenes strict geometry follows ``nuScenes/.../planning_eval.py`` and its
``box3d_to_corners`` helper. Its optional loose path follows
``PlanningMetricLoose`` there. Bench2Drive occupancy and collision indexing
follow ``Bench2Drive/.../metric_stp3.py`` and ``sparse_head.py``.
"""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

SECONDS = (1, 2, 3)
EGO_WIDTH, EGO_LENGTH = 1.85, 4.084
VEHICLE_CLASSES = set(range(14, 24))
HUMAN_CLASSES = set(range(2, 9))


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_lidar(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Undo the per-frame LiDAR-to-ego XY map used by the new data pipeline."""
    points = np.asarray(points)
    out = np.linalg.solve(np.asarray(axes, np.float64), np.asarray(points, np.float64)[..., :2].T).T
    return out.astype(points.dtype, copy=False) if np.issubdtype(points.dtype, np.floating) else out


def _get_yaw(traj: np.ndarray) -> np.ndarray:
    """Active nuScenes PlanningMetric.get_yaw, including its stationary fallback."""
    traj = np.asarray(traj, np.float64)
    if np.linalg.norm(traj[-1] - traj[0]) < 0.5:
        return np.full(len(traj), np.pi / 2, np.float64)
    padded = np.concatenate([np.zeros((1, 2), np.float64), traj], axis=0)
    yaw = np.zeros(len(traj) + 1, np.float64)
    yaw[1:-1] = np.arctan2(padded[2:, 1] - padded[:-2, 1], padded[2:, 0] - padded[:-2, 0])
    yaw[-1] = np.arctan2(padded[-1, 1] - padded[-2, 1], padded[-1, 0] - padded[-2, 0])
    return yaw[1:]


def _box_corners(boxes: np.ndarray) -> np.ndarray:
    """Active main ``box3d_to_corners`` geometry, without its framework imports."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 7)
    order = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                      [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]])
    order = order[[0, 1, 3, 2, 4, 5, 7, 6]] - np.array([0.5, 0.5, 0.5])
    corners = boxes[:, None, [3, 4, 5]] * order[None]
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rotation = np.zeros((len(boxes), 3, 3), np.float64)
    rotation[:, 0, 0], rotation[:, 0, 1] = c, -s
    rotation[:, 1, 0], rotation[:, 1, 1] = s, c
    rotation[:, 2, 2] = 1
    return np.einsum("nij,nkj->nki", rotation, corners) + boxes[:, None, :3]


def _strict_collision_at(point: np.ndarray, yaw: float, boxes: np.ndarray) -> bool:
    from shapely.geometry import Polygon

    boxes = np.asarray(boxes, np.float64).reshape(-1, 7)
    if not len(boxes):
        return False
    x, y = point
    ego = np.array([[x + 0.5 * np.cos(yaw), y + 0.5 * np.sin(yaw), 0,
                     EGO_LENGTH, EGO_WIDTH, 1.56, yaw]], np.float64)
    ego_poly = Polygon(_box_corners(ego)[0, [0, 3, 7, 4], :2])
    return any(ego_poly.intersects(Polygon(corners))
               for corners in _box_corners(boxes)[:, [0, 3, 7, 4], :2])


def _nusc_strict(samples: Sequence[tuple]) -> Dict[str, float]:
    l2_sum = np.zeros(6, np.float64)
    gt_hit_sum = np.zeros(6, np.float64)
    pred_hit_sum = np.zeros(6, np.float64)
    count = 0
    for _token, pred, info in samples:
        gt, mask = _numpy(info["trajectory"]), _numpy(info["mask"]).reshape(-1)
        if len(mask) < 6 or not np.all(mask[:6] == 1):
            continue
        axes = _numpy(info["axes"])
        pred_lidar = _to_lidar(_numpy(pred)[:6], axes)
        gt_lidar = _to_lidar(gt[:6], axes)
        boxes = info["strict_boxes"]
        if len(boxes) < 6:
            raise ValueError(f"complete nuScenes future has only {len(boxes)} box frames")
        gt_yaw, pred_yaw = _get_yaw(gt_lidar), _get_yaw(pred_lidar)
        gt_hit = np.array([_strict_collision_at(gt_lidar[t], gt_yaw[t], boxes[t]) for t in range(6)])
        pred_hit = np.array([_strict_collision_at(pred_lidar[t], pred_yaw[t], boxes[t]) for t in range(6)])
        l2_sum += np.linalg.norm(pred_lidar - gt_lidar, axis=1)
        gt_hit_sum += gt_hit
        pred_hit_sum += pred_hit & ~gt_hit
        count += 1
    out: Dict[str, float] = {"samples": len(samples), "strict_samples": count}
    if not count:
        return out
    l2, gt_col, pred_col = l2_sum / count, gt_hit_sum / count, pred_hit_sum / count
    for i in range(6):
        key = f"nuscenes/strict/{(i + 1) / 2:g}s"
        out[f"{key}/L2"] = float(l2[i])
        out[f"{key}/obj_col"] = float(gt_col[i])
        out[f"{key}/obj_box_col"] = float(pred_col[i])
    for metric, values in (("L2", l2), ("obj_col", gt_col), ("obj_box_col", pred_col)):
        out[f"nuscenes/strict/{metric}_avg_1_2_3s"] = float(np.mean(values[[1, 3, 5]]))
    return out


def _load_occupancy(raw) -> Optional[np.ndarray]:
    occ = _numpy(raw)
    if occ.ndim == 4:
        if occ.shape[0] != 1:
            return None
        occ = occ[0]
    if occ.ndim != 3:
        return None
    occ = np.flip(occ, axis=-1)
    if occ.shape[0] % 2:
        occ = occ[1:]
    return occ[:6].copy() if occ.shape[0] >= 6 else None


def load_nuscenes_segmentation(path: str) -> dict:
    """Load the optional main ``planing_gt_segmentation_val`` token->BEV pickle."""
    import pickle

    with open(path, "rb") as stream:
        raw = pickle.load(stream)
    if not isinstance(raw, Mapping):
        raise TypeError(f"nuScenes planning segmentation must be a token mapping: {path}")
    return {str(token): occ for token, value in raw.items() if (occ := _load_occupancy(value)) is not None}


def _loose_collision(traj: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    """Active main PlanningMetricLoose.evaluate_single_coll occupancy-grid sampling."""
    from skimage.draw import polygon

    dx, bx = 0.5, -49.75
    pts = np.array([[-EGO_LENGTH / 2 + 0.5, EGO_WIDTH / 2],
                    [EGO_LENGTH / 2 + 0.5, EGO_WIDTH / 2],
                    [EGO_LENGTH / 2 + 0.5, -EGO_WIDTH / 2],
                    [-EGO_LENGTH / 2 + 0.5, -EGO_WIDTH / 2]])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr, cc = polygon(pts[:, 1], pts[:, 0])
    path = np.asarray(traj, np.float64)[:, [1, 0]][:, None, :] / dx
    pixels = path + np.stack([rr, cc], axis=1)[None]
    rows = np.clip(pixels[:, :, 0].astype(np.int32), 0, 199)
    cols = np.clip(pixels[:, :, 1].astype(np.int32), 0, 199)
    return np.array([np.any(segmentation[t, rows[t], cols[t]]) for t in range(len(traj))], bool)


def _loose_center_collision(traj: np.ndarray, gt_box_collision: np.ndarray,
                            segmentation: np.ndarray) -> np.ndarray:
    """Active main PlanningMetricLoose center-point collision and GT collision mask."""
    trajectory = np.asarray(traj, np.float64).copy()
    trajectory[:, 0] *= -1
    x, y = trajectory[:, 0], trajectory[:, 1]
    row = ((y + 49.75) / 0.5).astype(np.int64)
    col = ((x + 49.75) / 0.5).astype(np.int64)
    valid = ((row >= 0) & (row < 200) & (col >= 0) & (col < 200) & ~gt_box_collision)
    result = np.zeros(len(traj), np.float64)
    indexes = np.arange(len(traj))[valid]
    result[valid] = segmentation[indexes, row[valid], col[valid]]
    return result


def _nusc_loose(samples: Sequence[tuple], segmentation: Mapping[str, np.ndarray]) -> Dict[str, float]:
    l2_sum = np.zeros(6, np.float64)
    gt_hit_sum = np.zeros(6, np.float64)
    pred_hit_sum = np.zeros(6, np.float64)
    count = 0
    for token, pred, info in samples:
        occ = segmentation.get(token)
        if occ is None:
            continue
        pred_lidar = _to_lidar(_numpy(pred)[:6], _numpy(info["axes"]))
        gt = _to_lidar(_numpy(info["trajectory"])[:6], _numpy(info["axes"]))
        mask = np.asarray(_numpy(info["mask"])[:6], np.float64).reshape(-1, 1)
        gt_for_collision = gt * np.array([-1.0, 1.0])
        pred_for_collision = pred_lidar * np.array([-1.0, 1.0])
        gt_hit = _loose_collision(gt_for_collision, occ)
        pred_hit = _loose_collision(pred_for_collision, occ)
        l2_sum += np.sqrt(np.sum((pred_lidar - gt) ** 2 * mask, axis=1))
        gt_hit_sum += _loose_center_collision(pred_lidar, gt_hit, occ)
        pred_hit_sum += pred_hit & ~gt_hit
        count += 1
    out: Dict[str, float] = {"loose_samples": count}
    if not count:
        return out
    l2, gt_col, pred_col = l2_sum / count, gt_hit_sum / count, pred_hit_sum / count
    for i in range(6):
        key = f"nuscenes/loose/{(i + 1) / 2:g}s"
        out[f"{key}/L2"] = float(l2[i])
        out[f"{key}/obj_col"] = float(gt_col[i])
        out[f"{key}/obj_box_col"] = float(pred_col[i])
    for metric, values in (("L2", l2), ("obj_col", gt_col), ("obj_box_col", pred_col)):
        values_at_seconds = [float(np.mean(values[: 2 * sec])) for sec in SECONDS]
        out[f"nuscenes/loose/{metric}_avg_1_2_3s"] = float(np.mean(values_at_seconds))
    return out


def _b2d_occupancy(main_boxes: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    import cv2

    boxes = np.asarray(main_boxes, np.float64).reshape(-1, 9).copy()
    attrs = np.asarray(attrs, np.float64).reshape(-1, 34)
    # Match sparse_head.py: undo dataset remap before metric_stp3.get_label remaps yaw again.
    boxes[:, [3, 4]] = boxes[:, [4, 3]]
    boxes[:, 6] = -boxes[:, 6] - np.pi / 2
    # metric_stp3.get_birds_eye_view_label's in-place yaw conversion.
    boxes[:, 6] = -(boxes[:, 6] + np.pi / 2)
    seg = np.zeros((6, 200, 200), np.float64)
    pedestrian = np.zeros_like(seg)
    for i in range(len(boxes)):
        mask = attrs[i, 12:18]
        delta = attrs[i, :12].reshape(6, 2)
        positions = boxes[i, None, :2] + np.cumsum(delta, axis=0)
        yaws = boxes[i, 6] + np.cumsum(attrs[i, 28:34])
        cls = int(attrs[i, 27])
        length, width = boxes[i, 4], boxes[i, 3]
        for t in range(6):
            if mask[t] != 1:
                continue
            x, y, yaw = positions[t, 0], positions[t, 1], yaws[t]
            corners = np.array([[length / 2, -length / 2, -length / 2, length / 2],
                                [width / 2, width / 2, -width / 2, -width / 2]])
            c, s = np.cos(yaw), np.sin(yaw)
            xy = np.array([[c, -s], [s, c]]) @ corners + np.array([[x], [y]])
            cv_xy = np.stack([xy[0], -xy[1]], axis=-1)
            cv_xy = np.round((cv_xy - np.array([-49.75, -49.75]) + 0.25) / 0.5).astype(np.int32)
            if cls in VEHICLE_CLASSES:
                cv2.fillPoly(seg[t], [cv_xy], 1.0)
            if cls in HUMAN_CLASSES:
                cv2.fillPoly(pedestrian[t], [cv_xy], 1.0)
    return np.logical_or(seg, pedestrian)


def _b2d_collision(traj: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    from skimage.draw import polygon

    pts = np.array([[-EGO_LENGTH / 2 + 0.5, EGO_WIDTH / 2],
                    [EGO_LENGTH / 2 + 0.5, EGO_WIDTH / 2],
                    [EGO_LENGTH / 2 + 0.5, -EGO_WIDTH / 2],
                    [-EGO_LENGTH / 2 + 0.5, -EGO_WIDTH / 2]])
    dx, bx = np.array([0.5, 0.5]), np.array([-49.75, -49.75])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr, cc = polygon(pts[:, 1], pts[:, 0])
    trajs = np.asarray(traj, np.float64)[:, None, [1, 0]] / dx
    pix = trajs + np.stack([rr, cc], axis=1)[None]
    rows = np.clip((200 - pix[:, :, 0]).astype(np.int32), 0, 199)
    cols = np.clip(pix[:, :, 1].astype(np.int32), 0, 199)
    return np.array([np.any(segmentation[t, rows[t], cols[t]]) for t in range(len(traj))], bool)


def _b2d_eval_coll(pred: np.ndarray, gt: np.ndarray, occ: np.ndarray):
    center = np.zeros(len(pred), np.float64)
    gt_box = _b2d_collision(gt, occ)
    pred_box = _b2d_collision(pred, occ)
    xx, yy = pred[:, 0], pred[:, 1]
    xi = ((49.75 / 2 - yy) / 0.5).astype(np.int64)
    yi = ((49.75 / 2 + xx) / 0.5).astype(np.int64)
    in_bounds = (xi >= 0) & (xi < 200) & (yi >= 0) & (yi < 200) & ~gt_box
    center[in_bounds] = occ[np.arange(len(pred))[in_bounds], xi[in_bounds], yi[in_bounds]]
    return center, (pred_box & ~gt_box).astype(np.float64)


def _b2d(samples: Sequence[tuple]) -> Dict[str, float]:
    totals = {f"plan_{metric}_{sec}s": 0.0 for metric in ("L2", "obj_col", "obj_box_col") for sec in SECONDS}
    valid = 0
    for _token, pred, info in samples:
        gt, mask = _numpy(info["trajectory"])[:6, :2], _numpy(info["mask"]).reshape(-1)[:6]
        if len(mask) == 6 and np.all(mask == 1):
            pred = _numpy(pred)[:6, :2]
            axes = _numpy(info["axes"])
            pred, gt = _to_lidar(pred, axes), _to_lidar(gt, axes)
            occ = _b2d_occupancy(info["planning_boxes"], info["planning_attr"])
            valid += 1
            for sec in SECONDS:
                steps = 2 * sec
                totals[f"plan_L2_{sec}s"] += float(np.linalg.norm(pred[:steps] - gt[:steps], axis=1).mean())
                center, box = _b2d_eval_coll(pred[:steps], gt[:steps], occ[:steps])
                totals[f"plan_obj_col_{sec}s"] += float(center.mean())
                totals[f"plan_obj_box_col_{sec}s"] += float(box.mean())
    out: Dict[str, float] = {"samples": len(samples), "valid_samples": valid}
    if not valid:
        return out
    out.update({f"b2d/{key}": value / valid for key, value in totals.items()})
    return out


def planning_metrics(samples: Sequence[tuple], source: str,
                     segmentation: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, float]:
    """Score ``(token, predicted_trajectory, eval_info)`` records with the active main protocol."""
    if source == "nuscenes":
        out = _nusc_strict(samples)
        if segmentation is not None:
            out.update(_nusc_loose(samples, segmentation))
        return out
    if source == "b2d":
        return _b2d(samples)
    raise ValueError(f"no main-consistent open-loop metric for {source!r}")


def _demo() -> None:
    assert np.allclose(_to_lidar(np.array([[2.0, 1.0]]), np.array([[0.8, -0.2], [0.1, 0.9]])),
                       [[2.7027027027027026, 0.8108108108108109]])
    assert np.allclose(_get_yaw(np.zeros((6, 2))), np.full(6, np.pi / 2))


if __name__ == "__main__":
    _demo()
