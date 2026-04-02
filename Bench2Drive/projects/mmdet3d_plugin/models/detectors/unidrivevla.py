import copy
import torch
import numpy as np
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet.models.detectors.base import BaseDetector
from projects.mmdet3d_plugin.datasets.evaluation import PlanningMetric


@DETECTORS.register_module()
class UniDriveVLA(BaseDetector):
    def __init__(
        self,
        planning_head=None,
        task_loss_weight=dict(planning=1.0),
        **kwargs,
    ):
        super().__init__()
        self.planning_head = build_head(planning_head) if planning_head is not None else None
        self.task_loss_weight = task_loss_weight
        self.instance_bank = None
        self.planning_metric = PlanningMetric()

    @property
    def with_planning_head(self):
        return hasattr(self, "planning_head") and self.planning_head is not None

    def extract_feat(self, img):
        return None

    def simple_test(self, img, **kwargs):
        if not self.with_planning_head:
            raise RuntimeError("planning_head is required.")

        img_metas = kwargs.get('img_metas', [{}])
        ego_motion = img.new_tensor([m.get('ego_motion', np.eye(4)) for m in img_metas])

        pred = self.planning_head.forward_test(
            img=img,
            ego_motion=ego_motion,
            **kwargs,
        )

        result = [dict()]
        result[0]["planning"] = pred

        return result

    def aug_test(self, imgs, **kwargs):
        img = imgs[0]
        return self.simple_test(img, **kwargs)

    def forward(self, return_loss=True, ar_batch=None, **kwargs):
        if return_loss:
            if ar_batch is None:
                ar_batch = getattr(self, '_current_ar_batch', None)
            return self.forward_train(ar_batch=ar_batch, **kwargs)
        return self.forward_test(**kwargs)

    def loss_weighted_and_prefixed(self, loss_dict, prefix=""):
        factor = float(self.task_loss_weight.get(prefix, 1.0))
        return {f"{prefix}.{k}": v * factor for k, v in loss_dict.items()}

    def _select_last_from_queue(self, img):
        # SparseDrive pipeline may provide (B, 6, 3, H, W) or (B, T, 6, 3, H, W).
        if torch.is_tensor(img) and img.dim() == 6:
            return img[:, -1]
        return img

    def forward_train(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        gt_occ_dense=None,
        hist_traj=None,
        ar_batch=None,
        **kwargs,
    ):
        # Adapter for Bench2Drive/HiP-AD interface
        if gt_ego_fut_trajs is None and 'gt_ego_fut_trajs_2hz' in kwargs:
            gt_ego_fut_trajs = kwargs['gt_ego_fut_trajs_2hz']
        if gt_ego_fut_masks is None and 'gt_ego_fut_masks_2hz' in kwargs:
            gt_ego_fut_masks = kwargs['gt_ego_fut_masks_2hz']

        if not self.with_planning_head:
            raise RuntimeError("planning_head is required.")

        img_last = self._select_last_from_queue(img)

        img_metas = kwargs.get('img_metas', [{}])
        ego_motion = img_last.new_tensor([m.get('ego_motion', np.eye(4)) for m in img_metas])

        ret = self.planning_head.forward_train(
            img=img_last,
            timestamp=timestamp,
            projection_mat=projection_mat,
            image_wh=image_wh,
            gt_depth=gt_depth,
            focal=focal,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_map_labels=gt_map_labels,
            gt_map_pts=gt_map_pts,
            gt_agent_fut_trajs=gt_agent_fut_trajs,
            gt_agent_fut_masks=gt_agent_fut_masks,
            gt_ego_fut_trajs=gt_ego_fut_trajs,
            gt_ego_fut_masks=gt_ego_fut_masks,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            ego_status=ego_status,
            gt_occ_dense=gt_occ_dense,
            ego_motion=ego_motion,
            ar_batch=ar_batch,
            **kwargs,
        )

        planning_losses = ret["losses"] if isinstance(ret, dict) and "losses" in ret else ret
        losses = self.loss_weighted_and_prefixed(planning_losses, prefix="planning")

        for k, v in list(losses.items()):
            if not torch.is_tensor(v):
                v = img_last.new_tensor(v)
            losses[k] = torch.nan_to_num(v)

        return losses

    @torch.no_grad()
    def forward_test(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_trajs_2hz=None,
        gt_ego_fut_masks_2hz=None,
        gt_attr_labels=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        gt_occ_dense=None,
        hist_traj=None,
        **kwargs,
    ):
        # Adapter for Bench2Drive/HiP-AD interface
        if gt_ego_fut_trajs is None and 'gt_ego_fut_trajs_2hz' in kwargs:
            gt_ego_fut_trajs = kwargs['gt_ego_fut_trajs_2hz']
        if gt_ego_fut_masks is None and 'gt_ego_fut_masks_2hz' in kwargs:
            gt_ego_fut_masks = kwargs['gt_ego_fut_masks_2hz']

        if not self.with_planning_head:
            raise RuntimeError("planning_head is required.")
        img_last = self._select_last_from_queue(img)

        img_metas = kwargs.get('img_metas') or [{}]
        ego_motion = img_last.new_tensor([m.get('ego_motion', np.eye(4)) for m in img_metas])

        pred = self.planning_head.forward_test(
            img=img_last,
            timestamp=timestamp,
            projection_mat=projection_mat,
            image_wh=image_wh,
            gt_depth=gt_depth,
            focal=focal,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_map_labels=gt_map_labels,
            gt_map_pts=gt_map_pts,
            gt_agent_fut_trajs=gt_agent_fut_trajs,
            gt_agent_fut_masks=gt_agent_fut_masks,
            gt_ego_fut_trajs=gt_ego_fut_trajs,
            gt_ego_fut_masks=gt_ego_fut_masks,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            ego_status=ego_status,
            ego_motion=ego_motion,
            **kwargs,
        )


        # SparseDrive-style packaging: return [{'img_bbox': result_dict}].
        # If planning_head returns {'traj':..., 'det':..., 'map':...}, flatten det/map into img_bbox.

        # Unify output packaging to always include metric_results
        if isinstance(pred, dict):
            traj = pred.get("planning", pred.get("traj"))
            det_list = pred.get("det")
            map_list = pred.get("map")
        else:
            # Fallback for planning-only tensor output
            traj = pred
            det_list = None
            map_list = None

        # Determine batch size from available outputs.
        batch_size = None
        if torch.is_tensor(traj) and traj.dim() == 3:
            batch_size = traj.shape[0]
        elif isinstance(det_list, list) and isinstance(map_list, list):
            batch_size = max(len(det_list), len(map_list))
        elif isinstance(det_list, list):
            batch_size = len(det_list)
        elif isinstance(map_list, list):
            batch_size = len(map_list)
        else:
            batch_size = 1

        outputs = []
        for i in range(int(batch_size)):
            img_bbox = {}

            # Only include planning output; skip det/map fields to avoid
            # bench2drive_dataset._format_bbox crashing on detection-specific keys
            # (boxes_3d, trajs_3d per-box, etc.) that we don't produce.
            # planning_eval.py expects img_bbox['final_planning'] as per-sample (T, 2/3).
            if torch.is_tensor(traj) and traj.dim() == 3:
                traj_cpu = traj.detach().cpu().float()
                img_bbox["final_planning"] = traj_cpu[i]
            elif traj is not None:
                img_bbox["final_planning"] = traj

            out_dict = {"img_bbox": img_bbox}

            # Compute planning metric_results (required by bench2drive_dataset.evaluate)
            metric_results = self._compute_planning_metric(
                i, img_bbox,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_ego_fut_trajs_2hz=gt_ego_fut_trajs_2hz,
                gt_ego_fut_masks_2hz=gt_ego_fut_masks_2hz,
                gt_attr_labels=gt_attr_labels,
            )
            out_dict["metric_results"] = metric_results

            outputs.append(out_dict)

        return outputs



    def _compute_planning_metric(self, bs, img_bbox, gt_bboxes_3d=None,
                                  gt_ego_fut_trajs_2hz=None, gt_ego_fut_masks_2hz=None,
                                  gt_attr_labels=None):
        """Compute per-sample planning metrics.
        Fully mirrors sparse_head.py::compute_planner_metric_stp3.
        pred key: img_bbox['final_planning'] instead of results[bs]['plan_temp_2hz']
        """
        # pred: (T, 2) → wrap to (1, T, 2) to match HiP-AD interface
        pred_ego_fut_trajs = img_bbox["final_planning"].unsqueeze(0)  # (1, T, 2)

        if gt_ego_fut_trajs_2hz is None or gt_ego_fut_masks_2hz is None or gt_bboxes_3d is None or gt_attr_labels is None:
            metric_dict = dict(fut_valid_flag=False)
            for i in range(1, 4):
                metric_dict['plan_L2_{}s'.format(i)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i)] = 0.0
            return metric_dict

        gt_ego_fut_trajs = gt_ego_fut_trajs_2hz.cumsum(dim=-2)  # (B, T, 2)
        gt_agent_boxes = gt_bboxes_3d[bs]
        temp = copy.deepcopy(gt_agent_boxes[:, 3])
        gt_agent_boxes[:, 3] = gt_agent_boxes[:, 4]
        gt_agent_boxes[:, 4] = temp
        gt_agent_boxes[:, 6] = -gt_agent_boxes[:, 6] - np.pi / 2
        gt_agent_feats = gt_attr_labels
        # Check validity for this specific sample (index bs), not the whole batch
        fut_valid_flag = (gt_ego_fut_masks_2hz[bs] == 1).all().item()

        metric_dict = dict()
        metric_dict['fut_valid_flag'] = fut_valid_flag

        segmentation, pedestrian = self.planning_metric.get_label(gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        future_second = 3
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i + 1) * 2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[bs, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[bs:bs+1, :cur_time],
                    occupancy
                )
                metric_dict['plan_L2_{}s'.format(i + 1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = 0.0

        return metric_dict


def tinfo(name, x):
    if x is None:
        print(name, None); return
    if isinstance(x, (list, tuple)):
        print(name, type(x), len(x)); return
    if isinstance(x, dict):
        print(name, "dict", list(x.keys())[:10]); return
    if torch.is_tensor(x):
        print(name, tuple(x.shape), x.dtype, x.device, x.min().item() if x.numel() else None, x.max().item() if x.numel() else None)
        return
    print(name, type(x))
