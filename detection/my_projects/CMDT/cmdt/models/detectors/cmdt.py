# ------------------------------------------------------------------------
# Modified from CMT (https://github.com/junjie18/CMT)
# Copyright (c) 2023 Yan, Junjie
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence
from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from ..utils.grid_mask import GridMask
from ...ops import SPConvVoxelization


@MODELS.register_module()
class CmdtDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 **kwargs):
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs.pop('pts_voxel_layer', None)
        self.return_pts_feat = kwargs.get('return_pts_feat', False)
        kwargs.pop('return_pts_feat', None)
        super(CmdtDetector, self).__init__(**kwargs)
        
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    def init_weights(self):
        """Initialize model weights."""
        super(CmdtDetector, self).init_weights()

    # @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    # @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, imgs, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(imgs, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    @torch.no_grad()
    # @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                    (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        img_metas = [item.metainfo for item in batch_data_samples]
        gt_bboxes_3d = [item.get('gt_instances_3d')['bboxes_3d'] for item in batch_data_samples]
        gt_labels_3d = [item.get('gt_instances_3d')['labels_3d'] for item in batch_data_samples]

        img_feats, pts_feats = self.extract_feat(
            points, imgs=imgs, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.pts_bbox_head_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas)
            losses.update(losses_pts)
        return losses

    # @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def pts_bbox_head_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        img_metas = [item.metainfo for item in batch_data_samples]

        img_feats, pts_feats = self.extract_feat(
            points, imgs=imgs, img_metas=img_metas)
    
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.pts_bbox_head_test(pts_feats, img_feats, img_metas)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 data_instances_3d = bbox_pts,
                                                 data_instances_2d = None)
        if self.return_pts_feat:
            return detsamples, pts_feats[0]
        else:
            return detsamples
    
    # @force_fp32(apply_to=('x', 'x_img'))
    def pts_bbox_head_test(self, pts_feats, img_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = []
        for bboxes, scores, labels in bbox_list:
            results = InstanceData()
            results.bboxes_3d = bboxes.to('cpu')
            results.scores_3d = scores.cpu()
            results.labels_3d = labels.cpu()
            bbox_results.append(results)

        return bbox_results