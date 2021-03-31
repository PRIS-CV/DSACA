import os
import re
import cv2
import glob
import h5py
import math
import time
import shutil
import numpy as np
import scipy.spatial
import scipy.io as io
from shutil import copyfile
from scipy.ndimage.filters import gaussian_filter

root_DOTA = '/dssg/weixu/data_wei/RSOC'
source_train_images_pth = os.path.join(root_DOTA, 'train/images')
source_val_images_pth = os.path.join(root_DOTA, 'val/images')
source_train_label_pth = os.path.join(root_DOTA, 'train/labelTxt-v1.5/DOTA-v1.5_train_hbb')
source_val_label_pth = os.path.join(root_DOTA, 'val/labelTxt-v1.5/DOTA-v1.5_val_hbb')

root_RSOC = '/dssg/weixu/data_wei/RSOC'
target_train_images_pth = os.path.join(root_RSOC, 'train_RSOC/images')
target_val_images_pth = os.path.join(root_RSOC, 'val_RSOC/images')
target_train_label_pth = os.path.join(root_RSOC, 'train_RSOC/labelTxt-v1.5/DOTA-v1.5_train_hbb')
target_val_label_pth = os.path.join(root_RSOC, 'val_RSOC/labelTxt-v1.5/DOTA-v1.5_val_hbb')

RSOC_test_large_vehicle = os.path.join(root_RSOC, 'test_large-vehicle.txt')
RSOC_test_ship = os.path.join(root_RSOC, 'test_ship.txt')
RSOC_test_small_vehicle = os.path.join(root_RSOC, 'test_small-vehicle.txt')
RSOC_train_large_vehicle = os.path.join(root_RSOC, 'train_large-vehicle.txt')
RSOC_train_ship = os.path.join(root_RSOC, 'train_ship.txt')
RSOC_train_small_vehicle = os.path.join(root_RSOC, 'train_small-vehicle.txt')

if not os.path.exists(target_train_images_pth):
    os.makedirs(target_train_images_pth)
if not os.path.exists(target_val_images_pth):
    os.makedirs(target_val_images_pth)
if not os.path.exists(target_train_label_pth):
    os.makedirs(target_train_label_pth)
if not os.path.exists(target_val_label_pth):
    os.makedirs(target_val_label_pth)

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

path_sets = [source_train_label_pth, source_val_label_pth]
label_paths=[]
for path in path_sets:
    label_paths+=search(path, '.txt')
label_paths.sort()

path_sets = [source_train_images_pth, source_val_images_pth]
image_paths=[]
for path in path_sets:
    image_paths+=search(path, '.png')
image_paths.sort()

image_list = []
txt_list = [RSOC_test_large_vehicle, RSOC_test_small_vehicle, RSOC_train_large_vehicle, RSOC_train_small_vehicle]
path_list = [target_val_images_pth, target_val_label_pth, target_train_images_pth, target_train_label_pth]
for idx, txt in enumerate(txt_list):
    with open(txt) as f:
        image_list.append(f.readlines())

for idx in range(len(image_list)):
    if idx < 2:
        for image in image_list[idx]:
            source_img_pth = os.path.join(source_val_images_pth, image.replace('\n',''))
            target_img_pth = os.path.join(target_val_images_pth, image.replace('\n',''))
            source_label_pth = os.path.join(source_val_label_pth, image.replace('png', 'txt').replace('\n',''))
            target_label_pth = os.path.join(target_val_label_pth, image.replace('png', 'txt').replace('\n',''))
            # print(target_label_pth)
            try:
                shutil.copyfile(source_img_pth, target_img_pth)
                shutil.copyfile(source_label_pth, target_label_pth)
            except:
                print(image.replace('\n',''))
    else:
        for image in image_list[idx]:
            source_img_pth = os.path.join(source_train_images_pth, image.replace('\n',''))
            target_img_pth = os.path.join(target_train_images_pth, image.replace('\n',''))
            source_label_pth = os.path.join(source_train_label_pth, image.replace('png', 'txt').replace('\n',''))
            target_label_pth = os.path.join(target_train_label_pth, image.replace('png', 'txt').replace('\n',''))
            # print(target_label_pth)
            try:
                shutil.copyfile(source_img_pth, target_img_pth)
                shutil.copyfile(source_label_pth, target_label_pth)
            except:
                print(image.replace('\n',''))
print('successful')





















