# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from depth.models.builder import LOSSES

@LOSSES.register_module()
class MAELoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=False,
                 loss_weight=1.0,
                 max_depth=None):
        super(MAELoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.1 # avoid grad explode

    def maeloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        # input = input + self.eps
        # target = target + self.eps

        loss = nn.L1Loss()
        return loss(input, target)

    def DepthNorm(self, depth, maxDepth=10000.0): 
        return maxDepth / depth

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""
        if self.valid_mask:
            valid_mask = depth_gt > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(depth_gt > 0, depth_gt <= self.max_depth)
            depth_pred = depth_pred[valid_mask]
            depth_gt = depth_gt[valid_mask]
        
        depth_gt *= 1000.0
        depth_pred *= 1000.0

        loss_depth = self.loss_weight * self.maeloss(depth_pred, depth_gt)
        loss_depth_inverse = self.loss_weight * self.maeloss(self.DepthNorm(depth_pred), self.DepthNorm(depth_gt))

        return loss_depth, loss_depth_inverse
        #return loss_depth_inverse
        #return loss_depth
