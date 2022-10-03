from os import mkdir
import os
import random



def read_txt(txt_path):
    f = open(txt_path)
    data_list = []
    for line in f:
        data_list.append(line)
    return data_list

kitti_path = 'data/kitti/'

train_txt_path = os.path.join(kitti_path, 'kitti_eigen_train.txt')

train_list = read_txt(train_txt_path)

new_list = random.sample(train_list, 1602)

txt_path = os.path.join(kitti_path,'kitti_eigen_train_mini.txt')

f = open(txt_path, 'w')

for item in new_list:

    # print(item[0], len(item[1])) 
    # num = int(len(item[1]) / 14)
    # for index in range(num):
    #     lines = item[1][index]
    f.writelines(item)

f.close()

