# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from depth.models.builder import LOSSES

@LOSSES.register_module()
class MSELoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=False,
                 loss_weight=1.0,
                 max_depth=None):
        super(MSELoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.1 # avoid grad explode

    def mseloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        # input = input + self.eps
        # target = target + self.eps
        loss = nn.MSELoss()
        return loss(input, target)

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.mseloss(
            depth_pred,
            depth_gt,
            )
        return loss_depth
