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
from PIL import Image, ImageDraw, ImageFont
from shutil import copyfile
from scipy.ndimage.filters import gaussian_filter

root = '/data1/weixu/Pascal_VOC_2007'

test_label_pth = os.path.join(root, 'VisDrone2019-DET-val/annotations')
test_img_pth = os.path.join(root, 'VisDrone2019-DET-val/images')
train_label_pth = os.path.join(root, 'VisDrone2019-DET-train/annotations')
train_img_pth = os.path.join(root, 'VisDrone2019-DET-train/images')

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
    for idx, data in enumerate(file):
        label_line = data.split(',')
        gthBBs.append([])
        for label in label_line:
            gthBBs[idx].append(label.replace('\n',''))
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
        cv2.imwrite(os.path.join(save_pth, '{}{}'.format(category[i+1], '.png')), save_data)
