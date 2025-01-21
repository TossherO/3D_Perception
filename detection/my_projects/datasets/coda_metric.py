# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from mmengine import load
from mmengine.logging import MMLogger, print_log
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from .coda_eval import kitti_eval

@METRICS.register_module()
class CodaMetric(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Kitti metric'
        super(CodaMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')
    
    def convert_gt_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        """Convert loading annotations to Kitti annotations.

        Args:
            data_infos (dict): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of Kitti annotations.
        """
        data_annos = data_infos['data_list']
        if not self.format_only:
            cat2label = data_infos['metainfo']['categories']
            label2cat = dict((v, k) for (k, v) in cat2label.items())
            assert 'instances' in data_annos[0]
            for i, annos in enumerate(data_annos):
                if len(annos['instances']) == 0:
                    kitti_annos = {
                        'name': np.array([]),
                        'occluded': np.array([]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                    }
                else:
                    kitti_annos = {
                        'name': [],
                        'occluded': [],
                        'location': [],
                        'dimensions': [],
                        'rotation_y': [],
                    }
                    for instance in annos['instances']:
                        label = instance['bbox_label_3d']
                        kitti_annos['name'].append(label2cat[label])
                        kitti_annos['occluded'].append(instance['is_occluded'])
                        kitti_annos['location'].append(instance['bbox_3d'][:3])
                        kitti_annos['dimensions'].append(instance['bbox_3d'][3:6])
                        kitti_annos['rotation_y'].append(instance['bbox_3d'][6])
                    for name in kitti_annos:
                        kitti_annos[name] = np.array(kitti_annos[name])
                data_annos[i]['kitti_annos'] = kitti_annos
        return data_annos
    
    def convert_dt_annos_to_kitti_annos(self, results: List[dict]) -> List[dict]:
        """Convert loading annotations to Kitti annotations.

        Args:
            results (List[dict]): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of Kitti annotations.
        """
        dt_annos = []
        for i, annos in enumerate(results):
            num_instance = len(annos['pred_instances_3d']['scores_3d'])
            if num_instance == 0:
                kitti_annos = {
                    'name': np.array([]),
                    'occluded': np.array([]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
            else:
                kitti_annos = {
                    'name': [],
                    'occluded': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': [],
                }
                for j in range(num_instance):
                    label = annos['pred_instances_3d']['labels_3d'][j]
                    kitti_annos['name'].append(self.classes[label])
                    kitti_annos['occluded'].append('None')
                    location = annos['pred_instances_3d']['bboxes_3d'].tensor[j][:3]
                    location[2] = location[2] + annos['pred_instances_3d']['bboxes_3d'].tensor[j][5] / 2
                    kitti_annos['location'].append(location)
                    kitti_annos['dimensions'].append(annos['pred_instances_3d']['bboxes_3d'].tensor[j][3:6])
                    kitti_annos['rotation_y'].append(annos['pred_instances_3d']['bboxes_3d'].tensor[j][6])
                    kitti_annos['score'].append(annos['pred_instances_3d']['scores_3d'][j])
                for name in kitti_annos:
                    kitti_annos[name] = np.array(kitti_annos[name])
            kitti_annos['sample_idx'] = np.array([annos['sample_idx']] * len(kitti_annos['score']), dtype=np.int64)
            dt_annos.append(kitti_annos)
        return dt_annos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations and convert to Kitti format
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        self.data_infos = self.convert_gt_annos_to_kitti_annos(pkl_infos)

        # convert results to Kitti format
        dt_annos = self.convert_dt_annos_to_kitti_annos(results)

        # extract ground truth
        gt_annos = [self.data_infos[result['sample_idx']]['kitti_annos'] for result in results]

        eval_types = ['bev', '3d']
        ap_result_str, ap_dict = kitti_eval(gt_annos, dt_annos, self.classes, eval_types=eval_types)
        print_log(f'Results of pred_instances_3d:\n' + ap_result_str, logger=logger)
        return ap_dict