import os
import re
import cv2
import glob
import h5py
import math
import time
import numpy as np
import scipy.spatial
import scipy.io as io
from shutil import copyfile
from scipy.ndimage.filters import gaussian_filter

root = '../dataset/RSOC'

test_label_pth = os.path.join(root, 'val/labelTxt-v1.5/DOTA-v1.5_val_hbb')
test_img_pth = os.path.join(root, 'val/images')
train_label_pth = os.path.join(root, 'train/labelTxt-v1.5/DOTA-v1.5_train_hbb')
train_img_pth = os.path.join(root, 'train/images')

test_data_images_pth = os.path.join(root, 'test_data', 'images')
test_data_map_pth = os.path.join(root, 'test_data', 'gt_density_map')
test_data_show_pth = os.path.join(root, 'test_data', 'gt_show')
train_data_images_pth = os.path.join(root, 'train_data', 'images')
train_data_map_pth = os.path.join(root, 'train_data', 'gt_density_map')
train_data_show_pth = os.path.join(root, 'train_data', 'gt_show')

if not os.path.exists(test_data_images_pth):
    os.makedirs(test_data_images_pth)
if not os.path.exists(test_data_map_pth):
    os.makedirs(test_data_map_pth)
if not os.path.exists(test_data_show_pth):
    os.makedirs(test_data_show_pth)
if not os.path.exists(train_data_images_pth):
    os.makedirs(train_data_images_pth)
if not os.path.exists(train_data_map_pth):
    os.makedirs(train_data_map_pth)
if not os.path.exists(train_data_show_pth):
    os.makedirs(train_data_show_pth)

def search(root, target):
    path_buf = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            path_buf += search(path, target)
        elif os.path.splitext(path)[1] == target:
            path_buf.append(path)
    return path_buf

def load_gt_bbox(filepath):
    with open(filepath) as f:
        file = f.readlines()
    gthBBs = []
    for idx, data in enumerate(file[2:]):
        label_line = data.split()
        gthBBs.append([])
        for label in label_line[:-1]:
            gthBBs[idx].append(label)
    return gthBBs

def find_the_num(target, category):
    for idx,name in enumerate(category):
        if str(target).find(name) >= 0:
            return idx
    return -1

def resize(input, target_size, mode='img'):
    if mode == 'img':
        rate = target_size/max(input.shape[0], input.shape[1])
        if rate<1:
            input = cv2.resize(input, (math.floor(input.shape[1]*rate), math.floor(input.shape[0]*rate)))
        return input
    elif mode == 'coordinate':
        rate = target_size/max(input[0][0], input[0][1])
        if(rate<1):
            new_x = math.floor(input[1]*rate)
            new_y = math.floor(input[2]*rate)
        else:
            new_x = input[1]
            new_y = input[2]
        return new_x, new_y
    else:
        print('Error resize mode')

def feature_test(feature, save_pth, category):
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    for i in range(feature.shape[0]):
        np.seterr(divide='ignore', invalid='ignore')
        save_data = 255 * feature[i,:,:] / np.max(feature[i,:,:])
        save_data = save_data.astype(np.uint8)
        save_data = cv2.applyColorMap(save_data, 2)
        cv2.imwrite(os.path.join(save_pth, '{}{}'.format(category[i], '.png')), save_data)

' small-vehicle, large-vehicle 属于同一类 '
# dota_category = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
#                  'ground-track-field', 'harbor', 'bridge', 'vehicle', 'helicopter','roundabout',
#                  'soccer-ball-field', 'swimming-pool', 'container-crane']
dota_category = ['small-vehicle','large-vehicle']


path_sets = [test_label_pth, train_label_pth]
img_paths=[]
for path in path_sets:
    img_paths+=search(path, '.txt')
img_paths.sort()

print('begin convert')
with open("warning_message.txt", "w") as f:
    f.write('begin convert')
space_num = 0 # 记录最多能存多少图
for pth in img_paths:
    starttime = time.time()

    hbbs = load_gt_bbox(pth)
    # 读图片，初始化 zero map， 画点
    if str(pth).find('train') > 0:
        img_pth = pth.replace('labelTxt-v1.5/DOTA-v1.5_train_hbb','images').replace('.txt','.png')
        target_pth = img_pth.replace('train', 'train_data')
    if str(pth).find('val') > 0:
        img_pth = pth.replace('labelTxt-v1.5/DOTA-v1.5_val_hbb','images').replace('.txt','.png')
        target_pth = img_pth.replace('val', 'test_data')
    img = cv2.imread(img_pth)
    source_shape = img.shape
    img = resize(img, 2048, 'img')

    kpoint = np.zeros((len(dota_category), img.shape[0], img.shape[1])).astype(np.int8)
    for idx, hbb in enumerate(hbbs):
        num = find_the_num(hbb[-1], dota_category)
        if num != -1:
            center_x=int((float(hbb[1])+float(hbb[5]))/2.0)
            center_y=int((float(hbb[0])+float(hbb[4]))/2.0)
            new_x, new_y = resize((source_shape, center_x, center_y), 2048, 'coordinate')
            try:
                kpoint[num,new_x,new_y] = 1
            except:
                with open("warning_message.txt", "a") as f:
                    f.write('{}{}{}{}\n'.format( 'x:',(float(hbb[1]), float(hbb[5])),'y:',(float(hbb[0]), float(hbb[4]))))
                    f.write('center:{}\n'.format(center_x, center_y))
                    f.write('new:{}\n'.format(new_x, new_y))
                    f.write('img.shape:{}\n'.format(img.shape))
                    f.write('source_shape:{}\n'.format(source_shape))
                    f.write('kpoint.shape:{}\n'.format(kpoint.shape))
                    f.write('img_pth:{}\n'.format(img_pth))
                    f.write('hbb:{}\n\n'.format(hbb))

    kernel_size = 8
    density_map = kpoint.copy().astype(np.float32)
    for i in range(len(dota_category)):
        density_map[i,:,:] = gaussian_filter(kpoint[i,:,:].astype(np.float32), kernel_size)


    distance_map = (255 * (1 - kpoint[0, :, :].copy())).astype(np.uint8)
    ship=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    distance_map = (255 * (1 - kpoint[1, :, :].copy())).astype(np.uint8)
    large_vehicle=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)

    spatial_mask = np.array([ship, large_vehicle])

    distance = 5
    spatial_mask[(spatial_mask >= 0) & (spatial_mask < 1 * distance)] = 0
    spatial_mask[(spatial_mask >= 1 * distance) & (spatial_mask < 2 * distance)] = 1
    spatial_mask[(spatial_mask >= 2 * distance) & (spatial_mask < 3 * distance)] = 2
    spatial_mask[(spatial_mask >= 3 * distance) & (spatial_mask < 4 * distance)] = 3
    spatial_mask[(spatial_mask >= 4 * distance) & (spatial_mask < 5 * distance)] = 4
    spatial_mask[(spatial_mask >= 5 * distance) & (spatial_mask < 6 * distance)] = 5
    spatial_mask[(spatial_mask >= 6 * distance) & (spatial_mask < 8 * distance)] = 6
    spatial_mask[(spatial_mask >= 8 * distance) & (spatial_mask < 12 * distance)] = 7
    spatial_mask[(spatial_mask >= 12 * distance) & (spatial_mask < 18 * distance)] = 8
    spatial_mask[(spatial_mask >= 18 * distance) & (spatial_mask < 28 * distance)] = 9
    spatial_mask[(spatial_mask >= 28 * distance)] = 10


    cv2.imwrite(target_pth, img)
    feature_test(density_map, target_pth.replace('images', 'gt_show').replace('.png', ''), dota_category)
    with h5py.File(target_pth.replace('images', 'gt_density_map').replace('.png', '.h5'), 'w') as hf:
        # hf['kpoint'] = kpoint
        hf['density_map'] = density_map
        hf['mask'] = spatial_mask

    endtime = time.time()
    dtime = endtime - starttime
    space_num = space_num + 1
    print(space_num, 'run_time:', dtime, pth)
    # break
print('end convert')
