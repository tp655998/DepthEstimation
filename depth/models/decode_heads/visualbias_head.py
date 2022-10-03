from inspect import CO_VARARGS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding
from torch.nn.modules import conv

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F
from depth.models.utils import UpConvBlock, BasicConvBlock
from depth.models.builder import build_loss
# from depth.models.decode_heads import DenseDepthHead
from depth.ops import resize
# import wandb



class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))

@HEADS.register_module()
class VisualBiasHead(DepthBaseDecodeHead):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
        fpn (bool): Whether apply FPN head.
            Default: False
        conv_dim (int): Default channel of features in FPN head.
            Default: 256.
    """

    def __init__(self,
                 up_sample_channels,
                 fpn=False,
                 loss_vb=dict(type='VisualBiasLoss', loss_weight=1.0),
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 loss_mae=dict(type='MAELoss', loss_weight=1.0),
                 loss_vnl=dict(type='VNL_Loss', loss_weight=1.0),
                 conv_dim=256,
                 **kwargs):
        super(VisualBiasHead, self).__init__(**kwargs)

        self.up_sample_channels = up_sample_channels[::-1]
        self.loss_vb = build_loss(loss_vb)
        self.loss_mse = build_loss(loss_mse)
        self.loss_mae = build_loss(loss_mae)
        self.in_channels = self.in_channels[::-1]
        self.loss_vnl = build_loss(loss_vnl)


        self.conv_list = nn.ModuleList()
        up_channel_temp = 0

        self.fpn = fpn
        if self.fpn:
            self.num_fpn_levels = len(self.in_channels)

            # construct the FPN
            self.lateral_convs = nn.ModuleList()
            self.output_convs = nn.ModuleList()

            for idx, in_channel in enumerate(self.in_channels[:self.num_fpn_levels]):
                lateral_conv = ConvModule(
                    in_channel, conv_dim, kernel_size=1, norm_cfg=self.norm_cfg
                )
                output_conv = ConvModule(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

        else:
            for index, (in_channel, up_channel) in enumerate(
                    zip(self.in_channels, self.up_sample_channels)):
                if index == 0:
                    self.conv_list.append(
                        ConvModule(
                            in_channels=in_channel,
                            out_channels=up_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            act_cfg=None
                        ))
                else:
                    self.conv_list.append(
                        UpSample(skip_input=in_channel + up_channel_temp,
                                 output_features=up_channel,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg))

                # save earlier fusion target
                up_channel_temp = up_channel

    def forward_train(self, img, inputs, img_metas, depth_gt, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        depth_pred = self.forward(inputs, img_metas)
        # losses = self.losses(depth_pred, depth_gt)
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)

        losses = dict()
        # print('depth_pred', depth_pred.shape)
        # print('depth_gt', depth_gt.shape)
        losses["loss_depth"] = self.loss_decode(depth_pred, depth_gt)
        #losses["loss_mae"], losses["loss_mae_inv"] = self.loss_mae(depth_pred, depth_gt)
        #losses["loss_mae"] = self.loss_mae(depth_pred, depth_gt)
        # losses["loss_mse"], losses["loss_mse_inv"] = self.loss_mse(depth_pred, depth_gt)
        # losses["loss_vb"] = self.loss_vb(depth_pred, depth_gt) #=============================
        #losses["loss_vb_d"] = self.loss_vb(depth_pred, depth_gt)
        #losses["loss_vb"], losses["loss_vb_d"] = self.loss_vb(depth_pred, depth_gt) #=============================
        # losses["loss_vnl"] = self.loss_vnl(depth_pred, depth_gt) #=============================

        #losses["loss_vb"] = self.loss_vb(depth_pred, depth_gt) #=============================

        log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
        losses.update(**log_imgs)
        #wandb.log({"sig loss": losses["loss_depth"]}) #==========================================
        #wandb.log({"mse loss": losses["loss_mse"]}) #==========================================
        #wandb.log({"vb loss": losses["loss_vb"]}) #==========================================
        #wandb.log({"vb(low) loss": losses["loss_vb"]}) #==========================================
        #wandb.log({"vb(high) loss": losses["loss_vb_d"]}) #==========================================
        #wandb.log({"vnl loss": losses["loss_vnl"]}) #==========================================
        #wandb.log({"total loss": losses["loss_depth"]+losses["loss_vb"]}) #==========================================
        #wandb.log({"total loss": losses["loss_vb"]+losses["loss_vb_d"]}) #==========================================

        return losses

    def forward(self, inputs, img_metas):
        """Forward function."""

        temp_feat_list = []
        if self.fpn:
            
            for index, feat in enumerate(inputs[::-1]):
                x = feat
                lateral_conv = self.lateral_convs[index]
                output_conv = self.output_convs[index]
                cur_fpn = lateral_conv(x)

                # Following FPN implementation, we use nearest upsampling here. Change align corners to True.
                if index != 0:
                    y = cur_fpn + F.interpolate(temp_feat_list[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
                else:
                    y = cur_fpn
                    
                y = output_conv(y)
                temp_feat_list.append(y)

        else:
            temp_feat_list = []
            for index, feat in enumerate(inputs[::-1]):
                if index == 0:
                    temp_feat = self.conv_list[index](feat)
                    temp_feat_list.append(temp_feat)
                else:
                    skip_feat = feat
                    up_feat = temp_feat_list[index-1]
                    temp_feat = self.conv_list[index](up_feat, skip_feat)
                    temp_feat_list.append(temp_feat)

        output = self.depth_pred(temp_feat_list[-1])
        return output
