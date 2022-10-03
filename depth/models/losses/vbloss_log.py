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
O log
X norm
X edge
X /fx
'''
# @LOSSES.register_module()
class VisualBiasLoss(nn.Module):
    """VisulaBiasLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, 
                 fx=None, 
                 fy=None, 
                 input_size=(416, 544),
                 valid_mask=False,
                 loss_weight=1.0,
                 max_depth=None):
        super(VisualBiasLoss, self).__init__()
        self.input_size = input_size
        self.fx = torch.tensor(fx, dtype=torch.float32).cuda()
        self.fy = torch.tensor(fy, dtype=torch.float32).cuda()
        self.u0 = torch.tensor(272, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(208, dtype=torch.float32).cuda()
        # print('u0', self.u0)
        # print('v0', self.v0)
        self.u = 0
        self.v = 0
        self.init_image_coor()
        self.build_gt_coor() #self.gt_uv_list
        # self.init_image_coor()
        # self.u0 = torch.tensor(u0, dtype=torch.float32).cuda()
        # self.v0 = torch.tensor(v0, dtype=torch.float32).cuda()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.0001 # avoid grad explode

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

        batch_size = 8 #============================

        gt_u_list = torch.zeros([batch_size, 1, 226304], dtype=torch.float32) # create black image h x w 
        gt_v_list = torch.zeros([batch_size, 1, 226304], dtype=torch.float32) # create black image h x w 
        gt_u_u0_list = torch.zeros([batch_size, 1, 226304], dtype=torch.float32) # create black image h x w 
        gt_v_v0_list = torch.zeros([batch_size, 1, 226304], dtype=torch.float32) # create black image h x w 

        for batch in range(batch_size):
            gt_u_list[batch, :, :] = u_list
            gt_v_list[batch, :, :] = v_list
            gt_u_u0_list[batch, :, :] = u_u0_list
            gt_v_v0_list[batch, :, :] = v_v0_list


        self.u = gt_u_list.cuda()
        self.v = gt_v_list.cuda()
        self.gt_u_u0 = gt_u_u0_list.cuda()
        self.gt_v_v0 = gt_v_v0_list.cuda()

    def plot_canny(self, img, edges, titleA, titleB):
        plt.subplot(121),plt.imshow(img,cmap = 'gray') # (hight, width)
        plt.title(titleA), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap = 'gray')
        plt.title(titleB), plt.xticks([]), plt.yticks([])
        plt.show()

    def auto_canny(self, image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv.Canny(image, lower, upper)
        return edged

    def find_edge(self, rgb):
        # print('rgb', rgb.shape) #torch.Size([4, 3, 416, 544])
        batch_size = rgb.shape[0] #0

        # gray = torch.mean(rgb, dim=1) # 轉灰階 torch.Size([4, 416, 544])
        # gray = torch.unsqueeze(gray, dim=1) # torch.Size([4, 1, 416, 544])

        edge_mask = torch.zeros([4, 1, 416, 544], dtype=torch.float32) # create black image torch.Size([4, 1, 416, 544]) 

        for batch in range(batch_size):
            img =  rgb[batch, :, :, :].permute(1, 2, 0) # torch.Size([3, 416, 544]) 
            # print('permute img', img.shape)
            # img = cv.blur(img)
            img = img.cpu().detach().numpy()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(np.uint8(gray), (5, 5), 0)
            # print('img', img.shape)
            # edge = cv.Canny(blur, 30, 150)
            edge = self.auto_canny(blur) #黑色背景 白色edge
            # self.plot_canny(img, edge, 'RGB', 'edge')

            # 膨胀范围
            kernel = np.ones((3,3), np.uint8)
            # 迭代次数
            dilate_edge = cv.dilate(edge, kernel, iterations = 2)
            # self.plot_canny(edge, dilate_edge, 'edge', 'dilate_edge')
            background = dilate_edge < 1 #背景(黑色)部分
            # self.plot_canny(dilate_edge, background, 'dilate_edge', 'background')
            # print('background', background.sum()/226304)
            background = torch.tensor(np.float32(background))
            # print('edge', edge.shape)

            background = torch.unsqueeze(background, dim=0) # 1 416, 544
            edge_mask[batch, :, :, :] = background
            # cv.imshow('Lap', edge)
            # cv.waitKey(0)

        # edge_mask = edge_rgb > 0
        # print('edge mask', edge_mask.shape)
        edge_mask = torch.flatten(edge_mask, start_dim=2)
        return edge_mask.cuda()

    def transfer_xyz(self, gt, pd):
        '''
        pixel to world coordinate
        and use pd to z
        '''
        # edge_mask = self.find_edge(rgb)

        flatten_gt = torch.flatten(gt, start_dim=2) 
        valid_mask = torch.logical_and(flatten_gt > 0, flatten_gt <= self.max_depth) #depth value is valid

        # valid_mask = torch.logical_and(edge_mask, valid_mask) # not edge 、 depth value is valid
        # valid_flatten_gt = flatten_gt[valid_mask]


        # print('self.u_u0', self.u_u0.shape)
        # self.u_u0 = torch.flatten(self.u_u0, start_dim=1)
        # self.v_v0 = torch.flatten(self.v_v0, start_dim=1)
        # print('flatten self.u_u0', self.u_u0.shape)

        x = self.u_u0 * gt / self.fx 
        y = self.v_v0 * gt / self.fy 
        z = pd #use predict depth insted gt

        # print('gt min', x.min(), y.min(), z.min())

        # print('x', x.shape)
        # print('y', y.shape)
        # print('z', z.shape)
        pcd = torch.cat((x, y, z), 1)
        return pcd, valid_mask
        # return pcd, valid_mask, valid_flatten_gt


        

    def cal_distance(self, pd, valid_mask):
        gt_u = torch.abs(self.u[valid_mask])
        gt_v = torch.abs(self.v[valid_mask])



        pd_u = torch.abs(pd[0, :])
        pd_v = torch.abs(pd[1, :])

        u = torch.log(gt_u+1) - torch.log(pd_u+1)
        v = torch.log(gt_v+1) - torch.log(pd_v+1)


        # print('u /', torch.mean(u))
        # distance = torch.pow(u, 2) + torch.pow(v, 2)
        # distance = torch.sqrt(torch.pow(torch.mean(u), 2) + torch.pow(torch.mean(v), 2) + self.eps) 
        distance = torch.sqrt(torch.pow(u, 2) + torch.pow(v, 2) + self.eps) 
        # distance = torch.sqrt(torch.pow(torch.log(u + 1), 2) + torch.pow(torch.log(v + 1), 2) + self.eps) 
        distance = torch.mean(distance)
        # print(distance)

        return distance

    # def bias_point_to_uv(self, point_cloud, valid_mask, gt):
    def bias_point_to_uv(self, point_cloud, valid_mask):
        point_cloud = torch.flatten(point_cloud, start_dim=2)


        # print('flatten pointcloud',point_cloud.shape)
        x = point_cloud[:, 0,:] # x
        y = point_cloud[:, 1,:] # y
        z = point_cloud[:, 2,:] # z

        u = (x * self.fx / z) + self.u0
        v = (y * self.fy / z) + self.v0

        # print('u shape', u.shape) # u shape torch.Size([4, 226304])


        # u = torch.round(u).int()
        # v = torch.round(v).int()


        # batch_size = u.shape[0]
        # print(batch_size)

        u = torch.unsqueeze(u, dim=1) 
        v = torch.unsqueeze(v, dim=1)
        # z = torch.unsqueeze(z, dim=1)
        # print('u unsqueeze shape', u.shape) #u unsqueeze shape torch.Size([4, 1, 226304])


        u = u[valid_mask]
        v = v[valid_mask]
        # pd = z[valid_mask]
        # print('pd', pd.shape) #pd torch.Size([840611])
        # print('gt', gt.shape) #gt torch.Size([840611])
        # print('u valid', u.shape) #u valid torch.Size([752561])


        # print(u.shape)
        # print(v.shape)

        u = torch.unsqueeze(u, dim=0) 
        v = torch.unsqueeze(v, dim=0)
        #u: torch.Size([1, 752561])


        pd_uv_list = torch.cat((u, v), dim=0)
        #uv: torch.Size([2, 752561])

        distance = self.cal_distance(pd_uv_list, valid_mask)
        # distance = self.cal_distance(pd_uv_list, valid_mask, pd, gt)
        # print('distance', distance)



        # print('distance', mean_distance)

        return distance



    def VBloss(self, input, target):
        # print('target shape', target.shape)
        # self.input_size = (target.shape[2],target.shape[3]) # input shape
        # print('input',  self.input_size)


        
        # input = input + self.eps
        # target = target + self.eps
        invalid = input < 0
        input[invalid] = 0.001



        # bias_point_cloud = self.RGBD_to_Point_Cloud(rgb, target, input) 
        bias_point_cloud, valid_mask = self.transfer_xyz(target, input) 
        loss = self.bias_point_to_uv(bias_point_cloud, valid_mask)
        # bias_point_cloud, valid_mask, valid_flatten_gt = self.transfer_xyz(target, input) 
        # loss = self.bias_point_to_uv(bias_point_cloud, valid_mask, valid_flatten_gt)


        return loss

        # return torch.log10(loss)

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""

        depth_pred /= 10.0
        depth_gt /= 10.0
        loss_depth = self.loss_weight * self.VBloss(depth_pred, depth_gt)

        return loss_depth

