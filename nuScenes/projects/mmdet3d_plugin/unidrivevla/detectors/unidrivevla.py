import torch
import numpy as np
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet.models.detectors.base import BaseDetector


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
        # NOTE: SparseDrive semantics: InstanceBank is maintained inside heads.
        # Do not persist temporal cache at detector level.
        self.instance_bank = None

    @property
    def with_planning_head(self):
        return hasattr(self, "planning_head") and self.planning_head is not None

    def extract_feat(self, img):
        return None

    def simple_test(self, img, **kwargs):
        if not self.with_planning_head:
            raise RuntimeError("planning_head is required.")

        img_metas = kwargs.get('img_metas', [{}])

        pred = self.planning_head.forward_test(
            img=img,
            **kwargs,
        )

        result = [dict()]
        result[0]["planning"] = pred

        return result

    def aug_test(self, imgs, **kwargs):
        img = imgs[0]
        return self.simple_test(img, **kwargs)

    def forward(self, return_loss=True, ar_batch=None, **kwargs):
        """
        Forward pass with optional AR cotraining support

        Args:
            return_loss: Whether to compute losses
            ar_batch: Optional AR batch for cotraining (from ARCotrainingHook)
            **kwargs: Other arguments

        Returns:
            losses or predictions
        """
        if return_loss:
            # Get ar_batch from model attribute (set by ARCotrainingHook)
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
        ar_batch=None,  # AR Cotraining: JSONL data batch
        **kwargs,
    ):
        """
        Forward training pass with optional AR cotraining

        Args:
            ar_batch: Optional AR batch for cotraining
            ... (other standard arguments)

        Returns:
            losses: Dict of weighted and prefixed losses
        """
        if not self.with_planning_head:
            raise RuntimeError("planning_head is required.")

        img_last = self._select_last_from_queue(img)

        img_metas = kwargs.get('img_metas', [{}])

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
            hist_traj=hist_traj,
            ar_batch=ar_batch,  # Pass AR batch to planning head
            **kwargs,
        )

        planning_losses = ret["losses"] if isinstance(ret, dict) and "losses" in ret else ret
        losses = self.loss_weighted_and_prefixed(planning_losses, prefix="planning")

        for k, v in list(losses.items()):
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=img.device if img is not None else torch.device('cuda'))
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
        gt_ego_fut_cmd=None,
        ego_status=None,
        gt_occ_dense=None,
        hist_traj=None,
        **kwargs,
    ):
        if not self.with_planning_head:
            raise RuntimeError("planning_head is required.")
        img_last = self._select_last_from_queue(img)

        img_metas = kwargs.get('img_metas', [{}])


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
            hist_traj=hist_traj,
            **kwargs,
        )

        if isinstance(pred, dict) and ("det" in pred or "map" in pred):
            traj = pred.get("planning", pred.get("traj"))

            det_list = pred.get("det")
            map_list = pred.get("map")

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

                if isinstance(det_list, list) and i < len(det_list) and isinstance(det_list[i], dict):
                    img_bbox.update(det_list[i])
                if isinstance(map_list, list) and i < len(map_list) and isinstance(map_list[i], dict):
                    img_bbox.update(map_list[i])

                # planning_eval.py expects img_bbox['final_planning'] as per-sample (T, 2/3) in meters.
                # trajs_3d comes from motion decoder via det_list[i] (per-agent trajectories).
                # Do NOT overwrite trajs_3d with the ego planning trajectory.
                if torch.is_tensor(traj) and traj.dim() == 3:
                    img_bbox["final_planning"] = traj.detach().cpu()[i]
                elif traj is not None:
                    img_bbox["final_planning"] = traj

                outputs.append({"img_bbox": img_bbox})

            return outputs

        # Fallback (planning-only): keep old behavior but still follow SparseDrive wrapper.
        return [{"img_bbox": {"trajs_3d": pred}}]



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