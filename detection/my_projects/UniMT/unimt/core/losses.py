# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.models.losses.utils import weighted_loss
from mmdet3d.registry import MODELS


@weighted_loss
def truncated_l1_loss(pred: Tensor, target: Tensor, 
                      truncated_threshold: Optional[Tensor] = None) -> Tensor:
    """Truncated L1 loss.

    Args:
        pred (Tensor): [N, 7] The prediction.
        target (Tensor): [N, 7] The learning target of the prediction.
        truncated_threshold (Tensor, optional): [7] The relative threshold for x, y, z, l, w, h, yaw.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    if truncated_threshold is None:
        loss = torch.abs(pred - target)
        
    else:
        assert target.shape[-1] == 8
        _truncated_threshold = torch.zeros_like(target)
        _truncated_threshold[:, 0] = truncated_threshold[0] * target[:, 3]
        _truncated_threshold[:, 1] = truncated_threshold[1] * target[:, 4]
        _truncated_threshold[:, 2] = truncated_threshold[2] * target[:, 5]
        _truncated_threshold[:, 3] = truncated_threshold[3] * target[:, 3]
        _truncated_threshold[:, 4] = truncated_threshold[4] * target[:, 4]
        _truncated_threshold[:, 5] = truncated_threshold[5] * target[:, 5]
        _truncated_threshold[:, 6] = truncated_threshold[6]
        _truncated_threshold[:, 7] = truncated_threshold[6]
        
        loss = torch.abs(pred - target) - _truncated_threshold
        loss = torch.clamp(loss, min=0)
    
    return loss


@MODELS.register_module()
class Truncated_L1Loss(nn.Module):
    """Truncated L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                truncated_threshold: Optional[Tensor] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * truncated_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, truncated_threshold=truncated_threshold)
        return loss_bbox