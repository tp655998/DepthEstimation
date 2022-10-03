# Copyright (c) OpenMMLab. All rights reserved.
from matplotlib.pyplot import vlines
import torch
import torch.nn as nn
import numpy as np
from depth.models.builder import LOSSES
import cv2 as cv
from matplotlib import pyplot as plt
# import torch.nn.functional as F

'''
O distance
X norm
O edge
O /fx
'''

@LOSSES.register_module()
class VisualBiasLoss(nn.Module):
    """VisulaBiasLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, 
                 fx=None, 
                 fy=None, 
                 input_size=(416, 544), #nyu
                #  input_size=(352, 704), #kitti
                 valid_mask=False,
                 loss_weight=1.0,
                 max_depth=None):
        super(VisualBiasLoss, self).__init__()
        self.input_size = input_size
        self.fx = torch.tensor(fx, dtype=torch.float32).cuda()
        self.fy = torch.tensor(fy, dtype=torch.float32).cuda()
        self.u0 = torch.tensor(input_size[1]//2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0]//2, dtype=torch.float32).cuda()
        # print('u0', self.u0)
        # print('v0', self.v0)
        self.u = 0
        self.v = 0
        self.batch_size = 6
        self.init_image_coor()
        self.build_gt_coor() #self.gt_uv_list
        # self.init_image_coor()
        # self.u0 = torch.tensor(u0, dtype=torch.float32).cuda()
        # self.v0 = torch.tensor(v0, dtype=torch.float32).cuda()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.scale = 1000.0

        self.eps = 0.001 # avoid grad explode

    def init_image_coor(self):
        '''
        pixel coordinate's (0,0) move to (u0, v0)
        '''


        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u = x
        self.u_u0 = x - self.u0
        # print('u0', self.u0)

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v = y
        self.v_v0 = y - self.v0

    def build_gt_coor(self):
        '''
        gt uv_list
        '''
        # print('self.u', self.u.shape) # self.u torch.Size([1, 416, 544])
        u_list = torch.flatten(self.u, start_dim=1) # torch.Size([1, 226304])
        v_list = torch.flatten(self.v, start_dim=1) # torch.Size([1, 226304])
        u_u0_list = torch.flatten(self.u_u0, start_dim=1) # torch.Size([1, 226304])
        v_v0_list = torch.flatten(self.v_v0, start_dim=1) # torch.Size([1, 226304])

        batch_size = self.batch_size #============================

        #226304 nyu
        #247808 kitti
        total_pixel = self.input_size[0] * self.input_size[1]
        gt_u_list = torch.zeros([batch_size, 1, total_pixel], dtype=torch.float32) # create black image h x w 
        gt_v_list = torch.zeros([batch_size, 1, total_pixel], dtype=torch.float32) # create black image h x w 
        gt_u_u0_list = torch.zeros([batch_size, 1, total_pixel], dtype=torch.float32) # create black image h x w 
        gt_v_v0_list = torch.zeros([batch_size, 1, total_pixel], dtype=torch.float32) # create black image h x w 

        for batch in range(batch_size):
            gt_u_list[batch, :, :] = u_list
            gt_v_list[batch, :, :] = v_list
            gt_u_u0_list[batch, :, :] = u_u0_list
            gt_v_v0_list[batch, :, :] = v_v0_list


        self.u = gt_u_list.cuda()
        self.v = gt_v_list.cuda()
        self.gt_u_u0 = gt_u_u0_list.cuda()
        self.gt_v_v0 = gt_v_v0_list.cuda()

        # gt_u_list = torch.cat((u_list, v_list), dim=0)
        # self.gt_uv_list = gt_uv.cuda()

    def get_z(self, x, y, d):
        z = d / torch.sqrt(pow(x, 2) + pow(y, 2) + 1)
        return z

    def transfer_xyz(self, gt, pd, valid_mask, batch_size):
        '''
        pixel to world coordinate
        and use pd to z
        '''
        #edge_mask = self.find_edge(rgb)

        # flatten_gt = torch.flatten(gt, start_dim=2) 
        # valid_mask = torch.logical_and(flatten_gt > 0, flatten_gt <= self.max_depth) #depth value is valid

        #valid_mask = torch.logical_and(edge_mask, valid_mask) # not edge 、 depth value is valid
        # valid_flatten_gt = flatten_gt[valid_mask]


        # print('self.u_u0', self.u_u0.shape)
        # self.u_u0 = torch.flatten(self.u_u0, start_dim=1)
        # self.v_v0 = torch.flatten(self.v_v0, start_dim=1)
        # print('flatten self.u_u0', self.u_u0.shape)
        # print('u_u0', self.u_u0.shape) # torch.Size([1, 416, 544])
        # print('edge flatten gt', gt.shape) # torch.Size([678620])

        u_u0 = self.u_u0
        v_v0 = self.v_v0

        u_u0 = torch.unsqueeze(u_u0, dim=0).repeat(batch_size, 1, 1, 1) 
        v_v0 = torch.unsqueeze(v_v0, dim=0).repeat(batch_size, 1, 1, 1) 

        # print('repeat u_uo', u_u0.shape)




        x = torch.flatten(u_u0, start_dim=2)[valid_mask] * gt / self.fx 
        y = torch.flatten(v_v0, start_dim=2)[valid_mask] * gt / self.fy 
        z = pd #use predict depth insted gt

        # print('gt min', x.min(), y.min(), z.min())

        # print('x', x.shape)
        # print('y', y.shape)
        # print('z', z.shape)
        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)
        z = torch.unsqueeze(z, dim=0)

        pcd = torch.cat((x, y, z), 0)
        return pcd
        # return pcd, valid_mask, valid_flatten_gt


    # def cal_distance(self, pd, valid_mask, pd_d, gt_d):
    def cal_distance(self, pd, valid_mask):
        gt_u = self.u[valid_mask]
        gt_v = self.v[valid_mask]

        gt_u_0 = torch.abs(self.gt_u_u0[valid_mask]) + self.norm_level #1e-5
        gt_v_0 = torch.abs(self.gt_v_v0[valid_mask]) + self.norm_level

        # print('gt_u_0', gt_u_0)
        # gt_u_0 = gt_u_0 
        # gt_v_0 = gt_v_0 
        # print('pd u', torch.isnan(pd_u).int().sum())
        # print('pd v', torch.isnan(pd_v).int().sum())
        # print('gt u', torch.isnan(gt_u).int().sum())
        # print('gt v', torch.isnan(gt_v).int().sum())

        pd_u = pd[0, :]
        pd_v = pd[1, :]

        # u = torch.log(gt_u+1) - torch.log(pd_u+1)
        # v = torch.log(gt_v+1) - torch.log(pd_v+1)
        u = gt_u - pd_u
        v = gt_v - pd_v

        # distance = self.sig(pd_u, gt_u, pd_v, gt_v)

        # print(distance)

        # d = gt_d - pd_d
        # u = torch.abs(u)
        # v = torch.abs(v)

        # gt_u_0 = torch.abs(gt_u_0)
        # gt_v_0 = torch.abs(gt_v_0)
        # print('u', u)
        # print('u0', gt_u_0)
        # u = torch.mean(u)
        # v = torch.mean(v)

        # print('u', torch.mean(u))

        u = u / gt_u_0 
        v = v / gt_v_0



        # print('u /', torch.mean(u))
        # distance = torch.pow(u, 2) + torch.pow(v, 2)
        # distance = torch.sqrt(torch.pow(torch.mean(u), 2) + torch.pow(torch.mean(v), 2) + self.eps) 
        distance = torch.sqrt(torch.pow(u, 2) + torch.pow(v, 2) + self.eps) 
        # distance = torch.round(distance, decimals=4) 

        # distance = torch.sqrt(torch.pow(torch.log(u + 1), 2) + torch.pow(torch.log(v + 1), 2) + self.eps) 

        # distance = torch.var(distance) + 0.15 * torch.pow(torch.mean(distance), 2)
        # distance = torch.sqrt(distance)

        # distance = torch.log(distance + 1) # 取 log
        # print('u', distance)

        # print('log u', torch.log(distance))

        # print('distance', distance)
        distance = torch.mean(distance)
        # print(distance)


        # distance = torch.sqrt(torch.pow(torch.mean(u), 2) + torch.pow(torch.mean(v), 2) + torch.pow(torch.mean(d), 2)) 
        # distance = torch.log10(distance+self.eps)
        # distance = torch.mean(distance)
        # distance = torch.sqrt(torch.pow(torch.mean(u), 2) + torch.pow(torch.mean(v), 2)) 
        # return torch.log10(distance)
        # print('mean', distance)
        # return distance
        return distance

    # def bias_point_to_uv(self, point_cloud, valid_mask, gt):
    def bias_point_to_uv(self, point_cloud, valid_mask):
        # point_cloud = torch.flatten(point_cloud, start_dim=2)


        # print('flatten pointcloud',point_cloud.shape)
        x = point_cloud[0,:] # x
        y = point_cloud[1,:] # y
        z = point_cloud[2,:] # z

        # d = self.get_depth(x, y, z)

        u = (x * self.fx / z) + self.u0
        v = (y * self.fy / z) + self.v0

        # print('u shape', u.shape) # u shape torch.Size([4, 226304])


        # u = torch.round(u).int()
        # v = torch.round(v).int()


        # batch_size = u.shape[0]
        # print(batch_size)

        u = torch.unsqueeze(u, dim=0) 
        v = torch.unsqueeze(v, dim=0)
        # z = torch.unsqueeze(z, dim=1)
        # print('u unsqueeze shape', u.shape) #u unsqueeze shape torch.Size([4, 1, 226304])


        # u = u[valid_mask]
        # v = v[valid_mask]
        # pd = z[valid_mask]
        # print('pd', pd.shape) #pd torch.Size([840611])
        # print('gt', gt.shape) #gt torch.Size([840611])
        # print('u valid', u.shape) #u valid torch.Size([752561])


        # print(u.shape)
        # print(v.shape)

        # u = torch.unsqueeze(u, dim=0) 
        # v = torch.unsqueeze(v, dim=0)
        #u: torch.Size([1, 752561])


        pd_uv_list = torch.cat((u, v), dim=0)
        #uv: torch.Size([2, 752561])

        distance = self.cal_distance(pd_uv_list, valid_mask)
        # distance = self.cal_distance(pd_uv_list, valid_mask, pd, gt)
        # print('distance', distance)



        # print('distance', mean_distance)


        return distance



    def VBloss(self, input, target, valid_mask, batch_size):
        # print('target shape', target.shape)
        # self.input_size = (target.shape[2],target.shape[3]) # input shape
        # print('input',  self.input_size)


        
        # input = input + self.eps
        # target = target + self.eps
        # invalid = input < 0
        # input[invalid] = 0.001


        bias_point_cloud = self.transfer_xyz(target, input, valid_mask, batch_size) 
        vb_loss = self.bias_point_to_uv(bias_point_cloud, valid_mask)
        # bias_point_cloud, valid_mask, valid_flatten_gt = self.transfer_xyz(target, input) 
        # loss = self.bias_point_to_uv(bias_point_cloud, valid_mask, valid_flatten_gt)

        # g = torch.log(input) - torch.log(target)
        # Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        
        # sig = torch.sqrt(Dg)

        # return (sig * 5.0), (vb_loss * 0.2)
        return vb_loss



        # return torch.log10(loss)
    def DepthNorm(self, depth): 
        maxDepth = self.max_depth * self.scale
        return maxDepth - depth

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""

        batch_size = depth_gt.shape[0]
        # print('gt shape', depth_gt.shape)


        flatten_gt = torch.flatten(depth_gt, start_dim=2) 
        #valid_mask = flatten_gt > 0
        valid_mask = torch.logical_and(flatten_gt > 0, flatten_gt <= self.max_depth) #depth value is valid

        u_non_zero = torch.abs(self.gt_u_u0) > 0
        v_non_zero = torch.abs(self.gt_v_v0) > 0
        non_zero_mask = torch.logical_and(u_non_zero, v_non_zero) # cross
        
        valid_mask = torch.logical_and(valid_mask, non_zero_mask)

        
        # valid_mask_center = (torch.abs(self.gt_u_u0)+torch.abs(self.gt_v_v0)) > 20
        
        # valid_mask = torch.logical_and(valid_mask, valid_mask_center)



        depth_gt = flatten_gt[valid_mask]
        depth_pred = torch.flatten(depth_pred, start_dim=2)[valid_mask]

        # self.scale = 1000.0
        self.scale = 256.0


        depth_gt *= self.scale
        depth_pred *= self.scale

        self.norm_level = 0.001

        # self.loss_weight *

        vb = self.loss_weight * self.VBloss(depth_pred, depth_gt, valid_mask, batch_size)
        vb_d =  self.loss_weight * self.VBloss(self.DepthNorm(depth_pred), self.DepthNorm(depth_gt), valid_mask, batch_size)

        return vb*0.5, vb_d*0.5
        # return (vb+vb_d)/2

