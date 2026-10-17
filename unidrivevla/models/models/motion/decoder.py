"""Inference-only SparseDrive agent-motion output decoding."""

import torch

from unidrivevla.models.compat import BBOX_CODERS
from unidrivevla.models.models.detection3d.decoder import SparseBox3DDecoder, decode_box


@BBOX_CODERS.register_module()
class SparseMotionDecoder(SparseBox3DDecoder):
    """Attach future agent trajectories to the exact detections selected by main.

    This mirrors nuScenes main's ``SparseMotionDecoder`` output contract. It has no learned state: trajectory modes
    come from the motion head and are gathered using the same score / quality ordering as the detection decoder.
    """

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        if motion_output is None:
            raise ValueError("motion_output is required to decode agent trajectories")

        squeeze_cls = instance_id is not None
        cls_scores = cls_scores[output_idx].sigmoid()
        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        batch_size, _, num_classes = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality is not None and quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., 0]
            centerness = torch.gather(centerness, 1, indices // num_classes)
            cls_scores = cls_scores * centerness.sigmoid()
            cls_scores, order = torch.sort(cls_scores, dim=1, descending=True)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, order)
            indices = torch.gather(indices, 1, order)

        trajectories = motion_output["prediction"][-1]
        trajectory_scores = motion_output["classification"][-1].sigmoid()
        results = []
        for batch_idx in range(batch_size):
            selected = indices[batch_idx] // num_classes
            boxes = decode_box(box_preds[batch_idx, selected])
            traj = trajectories[batch_idx, selected]
            traj_score = trajectory_scores[batch_idx, selected]
            if self.score_threshold is not None:
                valid = mask[batch_idx]
                boxes = boxes[valid]
                traj = traj[valid]
                traj_score = traj_score[valid]
            # Predictions are per-step displacements relative to each selected box's native-frame XY anchor.
            traj = traj.cumsum(dim=-2) + boxes[:, None, None, :2]
            results.append({"trajs_3d": traj.cpu(), "trajs_score": traj_score.cpu()})
        return results
