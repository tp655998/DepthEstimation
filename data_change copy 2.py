from os import mkdir
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
from path import Path
from tqdm import tqdm

class kitti_data_path:
    # 建構式
    def __init__(self, rgb_path=None, gt_path=None):
        self.rgb_path = rgb_path
        self.gt_path = gt_path
        self.fname = ''
    
    def set_data(self, line):
        rgb_path, gt_path, focal = line.split(' ')
        fname = rgb_path.split('/')[-1] # 0000000768.jpg
        # fname = fname.split('.')[0]
         
        self.rgb_path = rgb_path 
        self.gt_path = gt_path 
        self.fname = fname



def save_color_img(depth_map, save_path):

    # plt.imsave(save_path, depth_map, cmap='plasma') 
    plt.imsave(save_path, depth_map, vmin=1e-3, vmax=80, cmap='plasma') 

    # print('save_path', save_path)

def save_crop_img(depth_map, save_path):
    cv2.imwrite(save_path, depth_map)

def read_txt(txt_path):
    f = open(txt_path)
    data_list = []
    for index, line in enumerate(f):
        data_list.append(kitti_data_path())
        data_list[index].set_data(line)
    return data_list

kitti_path = 'data/kitti/'

train_txt_path = os.path.join(kitti_path, 'kitti_eigen_test.txt')

test_list = read_txt(train_txt_path)



os.makedirs(kitti_path + '/' + 'test/RGB', exist_ok=True)
os.makedirs(kitti_path + '/' + 'test/GT', exist_ok=True)

save_path_rgb = 'data/kitti/test/RGB/'
save_path_gt = 'data/kitti/test/GT/'

scale = 256

for item in test_list:
    # print(item.rgb_path, item.gt_path, item.fname)
    if item.gt_path != 'None':
        rgb = cv2.imread(kitti_path+'input/' + item.rgb_path)
        # depth = cv2.imread(kitti_path+'gt_depth/'+ item.gt_path, -1) / scale
        depth = cv2.imread(kitti_path+'gt_depth/'+ item.gt_path, -1).astype(np.uint16)
        cv2.imwrite(os.path.join(save_path_rgb, item.fname), rgb)
        cv2.imwrite(os.path.join(save_path_gt, item.fname.replace('jpg', 'png')), depth)
        # save_color_img(depth, os.path.join(save_path_gt, item.fname))

