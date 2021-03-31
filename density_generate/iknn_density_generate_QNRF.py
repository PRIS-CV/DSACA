import os
import time

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import math

from sklearn import preprocessing

root = '/data/weixu/UCF-QNRF_ECCV18'
img_train_path = root + '/Train/'
gt_train_path = root + '/Train/'
img_test_path = root + '/Test/'
gt_test_path = root + '/Test/'

save_train_img_path = root + '/train_data/images/'
save_train_gt_path = root + '/train_data/gt_density_map_ori/'
save_test_img_path = root + '/test_data/images/'
save_test_gt_path = root + '/test_data/gt_density_map_ori/'

distance = 1
img_train = []
gt_train = []
img_test = []
gt_test = []

for file_name in os.listdir(img_train_path):
    if file_name.split('.')[1] == 'jpg':
        img_train.append(file_name)

for file_name in os.listdir(gt_train_path):
    if file_name.split('.')[1] == 'mat':
        gt_train.append(file_name)

for file_name in os.listdir(img_test_path):
    if file_name.split('.')[1] == 'jpg':
        img_test.append(file_name)

for file_name in os.listdir(gt_test_path):
    if file_name.split('.')[1] == 'mat':
        gt_test.append(file_name)

img_train.sort()
gt_train.sort()
img_test.sort()
gt_test.sort()

print(len(img_train),len(gt_train), len(img_test),len(gt_test))

f = open('test.txt','w+')
for k in range(len(img_test)):
    Img_data = cv2.imread(img_test_path + img_test[k])
    Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])
    Gt_data = Gt_data['annPoints']

    f.write('{} {} '.format(k+1, len(Gt_data)))
    for data in Gt_data:
        f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), 15, 20, 1))
        # f.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f.write('\n')
    print(gt_test_path + gt_test[k])
f.close()

