# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import numpy as np
from torch import Tensor
import numba

from mmdet.models.task_modules import BaseBBoxCoder
from mmdet3d.registry import TASK_UTILS
from .util import denormalize_bbox


@numba.jit(nopython=True)
def circle_nms(dets: Tensor, thresh: float, post_max_size: int = 100) -> Tensor:
    """Circular NMS.

    An object is only counted as positive if no other center with a higher
    confidence exists within a radius r using a bird-eye view distance metric.

    Args:
        dets (Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept.
            Defaults to 83.

    Returns:
        Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    # highest->lowest, 并转换为numpy
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return keep[:post_max_size]

    return keep


@TASK_UTILS.register_module()
class NMSFreeBBoxCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 nms_radius=None,
                 score_threshold=None,
                 num_classes=10):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, cz, w, l, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.reshape(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 
        
        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            final_box_preds = final_box_preds[thresh_mask]
            final_scores = final_scores[thresh_mask]
            final_preds = final_preds[thresh_mask]

        # use circle nms
        if self.nms_radius is not None:
            boxes_for_nms = torch.cat([final_box_preds[:, :2], final_scores[:, None]], dim=1).detach().cpu().numpy()
            selected = circle_nms(boxes_for_nms, self.nms_radius)
            final_box_preds = final_box_preds[selected]
            final_scores = final_scores[selected]
            final_preds = final_preds[selected]

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, cz, w, l, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        if preds_dicts.get('vel') is not None:
            pred_bbox = torch.cat(
                    (preds_dicts['center'][-1], preds_dicts['height'][-1],
                    preds_dicts['dim'][-1], preds_dicts['rot'][-1],
                    preds_dicts['vel'][-1]),
                    dim=-1)
        else:
            pred_bbox = torch.cat(
                (preds_dicts['center'][-1], preds_dicts['height'][-1],
                preds_dicts['dim'][-1], preds_dicts['rot'][-1]),
                dim=-1)
        pred_logits = preds_dicts['cls_logits'][-1]

        batch_size = pred_logits.shape[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(pred_logits[i], pred_bbox[i]))
        return predictions_list