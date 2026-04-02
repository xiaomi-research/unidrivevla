import copy
import warnings
import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head
from typing import List, Optional, Tuple, Union

from projects.mmdet3d_plugin.datasets.evaluation import PlanningMetric

@HEADS.register_module()
class SparseHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        init_cfg=None,
        det_head=None,
        map_head=None,
        motion_plan_head=None,
        onedecoder_head=None,
        evaluate_bench2dive=False,
        **kwargs,
    ):
        super(SparseHead, self).__init__(init_cfg)
        self.task_config = task_config
        # original branch
        if self.task_config.get('with_det', False):
            self.det_head = build_head(det_head)
        if self.task_config.get('with_map', False):
            self.map_head = build_head(map_head)
        if self.task_config.get('with_motion_plan', False):
            self.motion_plan_head = build_head(motion_plan_head)
        if self.task_config.get('with_onedecoder', False):
            self.onedecoder_head = build_head(onedecoder_head)

        self.evaluate_bench2dive = evaluate_bench2dive

        if self.evaluate_bench2dive:
            self.planning_metric = PlanningMetric()

    def init_weights(self):
        if self.task_config.get('with_det', False):
            self.det_head.init_weights()
        if self.task_config.get('with_map', False):
            self.map_head.init_weights()
        if self.task_config.get('with_det_map', False):
            self.det_map_head.init_weights()
        if self.task_config.get('with_motion_plan', False):
            self.motion_plan_head.init_weights()
        if self.task_config.get('with_onedecoder', False):
            self.onedecoder_head.init_weights()

    def forward(self, img, feature_maps, metas: dict):
        det_output, map_output, ego_output, plan_output, motion_output, scenes_output \
            = None, None, None, None, None, None

        # onedecoder
        if self.task_config.get('with_onedecoder', False):
            det_output, map_output, ego_output, plan_output, motion_output, scenes_output \
                = self.onedecoder_head(img, feature_maps, metas)

        # original branch
        else:
            if self.task_config['with_det']:
                det_output = self.det_head(feature_maps, metas)
            if self.task_config['with_map']:
                map_output = self.map_head(feature_maps, metas)
            if self.task_config['with_motion_plan']:
                motion_output, plan_output = self.motion_plan_head(det_output, map_output, feature_maps, metas,
                                                                   self.det_head.anchor_encoder,
                                                                   self.det_head.instance_bank.mask,
                                                                   self.det_head.instance_bank.anchor_handler)
            else:
                motion_output, plan_output = None, None

        return det_output, map_output, ego_output, plan_output, motion_output, scenes_output

    def loss(self, model_outs, data):
        losses = dict()
        det_output, map_output, ego_output, plan_output, motion_output, scenes_output = model_outs

        # merge det, map, lane, motion and plan
        if self.task_config.get('with_onedecoder', False):
            loss_onedecoder = self.onedecoder_head.loss(
                det_output, map_output, ego_output, plan_output, motion_output, scenes_output, data)
            losses.update(loss_onedecoder)

        # original branch
        else:
            if self.task_config['with_det']:
                loss_det = self.det_head.loss(det_output, data)
                losses.update(loss_det)

            if self.task_config['with_map']:
                loss_map = self.map_head.loss(map_output, data)
                losses.update(loss_map)

            if self.task_config['with_motion_plan']:
                motion_loss_cache = dict(indices=self.det_head.sampler.indices)
                loss_motion = self.motion_plan_head.loss(motion_output, plan_output, data, motion_loss_cache)
                losses.update(loss_motion)

        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, ego_output, plan_output, motion_output, scenes_output = model_outs

        # merge det, map, motion and plan
        if self.task_config.get('with_onedecoder', False):
            det_result, map_result, ego_result, plan_result, motion_result = self.onedecoder_head.post_process(
                det_output, map_output, ego_output, plan_output, motion_output, data)

            task = self.onedecoder_head.task_select[0]
            if task == "det": batch_size = len(det_result)
            if task == "map": batch_size = len(map_result)
            if task == "plan": batch_size = len(plan_result)
            if task == "motion": batch_size = len(motion_result)

        # original branch
        else:
            if self.task_config['with_det']:
                det_result = self.det_head.post_process(det_output)
                batch_size = len(det_result)

            if self.task_config['with_map']:
                map_result = self.map_head.post_process(map_output)
                batch_size = len(map_result)

            if self.task_config['with_motion_plan']:
                motion_result, plan_result = self.motion_plan_head.post_process(
                    det_output, motion_output, plan_output, data)

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config.get('with_onedecoder', False):
                for task in self.onedecoder_head.task_select:
                    if task == "det": results[i].update(det_result[i])
                    if task == "map": results[i].update(map_result[i])
                    if task == "plan": results[i].update(plan_result[i])
                    if task == "motion": results[i].update(motion_result[i])
            else:
                if self.task_config['with_det']:
                    results[i].update(det_result[i])
                if self.task_config['with_map']:
                    results[i].update(map_result[i])
                if self.task_config['with_motion_plan']:
                    results[i].update(plan_result[i])
                    results[i].update(motion_result[i])

        if self.evaluate_bench2dive:
            results = self.evaluate_metric(results, data)

        return results

    def evaluate_metric(self, results, data):
        for bs in range(len(results)):
            metric_dict = self.compute_planner_metric_stp3(bs, results, data)
            results[bs]['metric_results'] = metric_dict
        return results

    def compute_planner_metric_stp3(self, bs, results, data, remap_box=True):
        pred_ego_fut_trajs = results[bs]['plan_temp_2hz'][None]
        gt_ego_fut_trajs = data["gt_ego_fut_trajs_2hz"].cumsum(dim=-2)
        gt_agent_boxes = data["gt_bboxes_3d"][bs]
        if remap_box:
            temp = copy.deepcopy(gt_agent_boxes[:, 3])
            gt_agent_boxes[:, 3] = gt_agent_boxes[:, 4]
            gt_agent_boxes[:, 4] = temp
            gt_agent_boxes[:, 6] = - gt_agent_boxes[:, 6] - np.pi / 2
        gt_agent_feats = data["gt_attr_labels"]
        fut_valid_flag = (data["gt_ego_fut_masks_2hz"] == 1).all().item()

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
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
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

    def compute_planner_metric_stp3_ori(self, bs, results, data, remap_box=True):
        pred_ego_fut_trajs = results[bs]['final_planning'][None]
        gt_ego_fut_trajs = data["gt_ego_fut_trajs"].cumsum(dim=-2)
        gt_agent_boxes = data["gt_bboxes_3d"][bs]
        if remap_box:
            temp = copy.deepcopy(gt_agent_boxes[:, 3])
            gt_agent_boxes[:, 3] = gt_agent_boxes[:, 4]
            gt_agent_boxes[:, 4] = temp
            gt_agent_boxes[:, 6] = - gt_agent_boxes[:, 6] - np.pi / 2
        gt_agent_feats = data["gt_attr_labels"]
        fut_valid_flag = (data["gt_ego_fut_masks"] == 1).all().item()

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
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
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
