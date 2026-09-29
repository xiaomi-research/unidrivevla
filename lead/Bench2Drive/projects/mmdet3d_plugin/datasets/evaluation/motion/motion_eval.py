# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
import tqdm
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes.eval.detection.algo import calc_tp
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList

from .motion_utils import MotionBox, accumulate
MOTION_TP_METRICS = ['min_ade_err', 'min_fde_err', 'miss_rate_err']


def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))

class MotionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 gt_boxes,
                 pred_boxes,
                 config: DetectionConfig,
                 verbose: bool = True):

        self.verbose = verbose
        self.cfg = config
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        self.cfg['class_names'] = ['car', 'pedestrian']
        self.cfg['dist_ths'] = [2.0]

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        metrics = {}
        for class_name in self.cfg['class_names']:
            for dist_th in self.cfg['dist_ths']:
                md, EPA, EPA_ = accumulate(self.gt_boxes, self.pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                metrics[f'{class_name}_EPA'] = EPA_

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        for class_name in self.cfg['class_names']:
            # Compute TP metrics.
            for metric_name in MOTION_TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.cfg['min_recall'], metric_name)
                metrics[f'{class_name}_{metric_name}']  = tp

        return metrics, metric_data_list