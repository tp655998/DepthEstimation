from os import mkdir
import os

def cal_train_valid(nyu_path):
    train_num = 0
    test_num = 0
    train_scene_list = []
    train = []
    test_scene_list = []
    test = []
    for scene in os.listdir(nyu_path):
        if os.path.isdir(os.path.join(nyu_path, scene)):

            scene_path = os.path.join(nyu_path, scene)
            scene_img_num = len(os.listdir(scene_path))
            # print(scene, scene_img_num)
            # print(scene)
            # print(scene.split('_')[-1][:4])
            # print(scene.split('_')[-1][:4].isnumeric())

            if scene.split('_')[-1][:4].isnumeric(): #train
                train_scene_list.append(scene) # 紀錄scene
                train.append([scene, []]) # 紀錄scene
                train_num += scene_img_num
            
            else: #test
                test_scene_list.append(scene) # 紀錄scene
                test.append([scene, []]) # 紀錄scene
                test_num += scene_img_num
        else:
            print(scene, 'is file!')

    print('Train、Test', train_num, test_num) # Train、Test 73192 908
    print('train_scene_list, test_scene_list', len(train_scene_list), len(test_scene_list))
    return train, train_scene_list, test, test_scene_list


def cal_item(txt_path, dataset, scene_list):
    f = open(txt_path)
    data_num = 0
    for line in f:
        item = line.split(' ')[1] # ['/kitchen_0028b/rgb_00045.jpg', '/kitchen_0028b/sync_depth_00045.png', '518.8579\n']
        item = item.split('/')[1] # ['', 'kitchen_0028b', 'sync_depth_00045.png']
        item_index = scene_list.index(item) 
        dataset[item_index][1].append(line)
        data_num+=1
    print('train num:', data_num)
    return dataset

nyu_path = 'data/nyu/'

train, train_scene_list, test, test_scene_list = cal_train_valid(nyu_path)

train_txt_path = os.path.join(nyu_path, 'nyu_train.txt')
test_txt_path = os.path.join(nyu_path, 'nyu_test.txt')


dataset = cal_item(train_txt_path, train, train_scene_list)

txt_path = os.path.join(nyu_path,'nyu_train_mini_14.txt')
# f = open(txt_path, 'w')
# for item in dataset:

    # print(item[0], len(item[1])) 
    # num = int(len(item[1]) / 14)
    # for index in range(num):
    #     lines = item[1][index]
    #     f.writelines(lines)
# f.close()

