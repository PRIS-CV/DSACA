import h5py
import time
import torch
import numpy as np
from PIL import Image
import cv2

def load_data_visdrone_class_2(img_path,train = True):
    while True:
        try:
            gt_path = img_path.replace('.jpg','.h5').replace('images','gt_density_map')

            img = Image.open(img_path).convert('RGB')

            torch.cuda.synchronize()
            begin_time_test_7 = time.time()

            gt_file = h5py.File(gt_path)

            torch.cuda.synchronize()
            end_time_test_7 = time.time()
            run_time_7 = end_time_test_7 - begin_time_test_7
            # print('该循环程序运行时间7：', run_time_7)

            # mask_map = np.asarray(gt_file['mask'][()])
            target = np.asarray(gt_file['density_map'][()][1:3,:,:])
            mask = np.asarray(gt_file['mask'][()][1:3,:,:])
            # k = np.asarray(gt_file['kpoint'][()])
            break
        except IOError:
            cv2.waitKey(5)

    img=img.copy()
    mask=mask.copy()
    mask[mask<=4]=1
    mask[mask>4]=0
    target=target.copy()
    k = 0

    return img, target, k, mask

def load_data_visdrone_class_3(img_path,train = True):
    while True:
        try:
            gt_path = img_path.replace('.jpg','.h5').replace('images','gt_density_map')

            img = Image.open(img_path).convert('RGB')

            torch.cuda.synchronize()
            begin_time_test_7 = time.time()

            gt_file = h5py.File(gt_path)

            torch.cuda.synchronize()
            end_time_test_7 = time.time()
            run_time_7 = end_time_test_7 - begin_time_test_7
            # print('该循环程序运行时间7：', run_time_7)

            # mask_map = np.asarray(gt_file['mask'][()])
            target = np.asarray(gt_file['density_map'][()][1:4,:,:])
            mask = np.asarray(gt_file['mask'][()][1:4,:,:])
            # k = np.asarray(gt_file['kpoint'][()])
            break
        except IOError:
            cv2.waitKey(5)

    img=img.copy()
    mask=mask.copy()
    mask[mask<=4]=1
    mask[mask>4]=0
    target=target.copy()
    k = 0

    return img, target, k, mask

def load_data_visdrone_class_4(img_path,train = True):
    while True:
        try:
            gt_path = img_path.replace('.jpg','.h5').replace('images','gt_density_map')

            img = Image.open(img_path).convert('RGB')

            torch.cuda.synchronize()
            begin_time_test_7 = time.time()

            gt_file = h5py.File(gt_path)

            torch.cuda.synchronize()
            end_time_test_7 = time.time()
            run_time_7 = end_time_test_7 - begin_time_test_7
            # print('该循环程序运行时间7：', run_time_7)

            # mask_map = np.asarray(gt_file['mask'][()])
            target = np.asarray(gt_file['density_map'][()][0:4,:,:])
            mask = np.asarray(gt_file['mask'][()][0:4,:,:])
            # k = np.asarray(gt_file['kpoint'][()])
            break
        except IOError:
            cv2.waitKey(5)

    img=img.copy()
    mask=mask.copy()
    mask[mask<=4]=1
    mask[mask>4]=0
    target=target.copy()
    k = 0

    return img, target, k, mask

def load_data_visdrone_class_8(img_path,train = True):
    while True:
        try:
            gt_path = img_path.replace('.jpg','.h5').replace('images','gt_density_map')

            img = Image.open(img_path).convert('RGB')

            torch.cuda.synchronize()
            begin_time_test_7 = time.time()

            gt_file = h5py.File(gt_path)

            torch.cuda.synchronize()
            end_time_test_7 = time.time()
            run_time_7 = end_time_test_7 - begin_time_test_7
            # print('该循环程序运行时间7：', run_time_7)

            # mask_map = np.asarray(gt_file['mask'][()])
            target = np.asarray(gt_file['density_map'][()][0:8,:,:])
            mask = np.asarray(gt_file['mask'][()][0:8,:,:])
            # k = np.asarray(gt_file['kpoint'][()])
            break
        except IOError:
            cv2.waitKey(5)

    img=img.copy()
    mask=mask.copy()
    mask[mask<=4]=1
    mask[mask>4]=0
    target=target.copy()
    k = 0

    return img, target, k, mask

def load_data_dota_class_2(img_path,train = True):
    while True:
        try:
            gt_path = img_path.replace('.png','.h5').replace('images','gt_density_map')

            img = Image.open(img_path).convert('RGB')

            torch.cuda.synchronize()
            begin_time_test_7 = time.time()

            gt_file = h5py.File(gt_path)

            torch.cuda.synchronize()
            end_time_test_7 = time.time()
            run_time_7 = end_time_test_7 - begin_time_test_7
            # print('该循环程序运行时间7：', run_time_7)

            # mask_map = np.asarray(gt_file['mask'][()])
            target = np.asarray(gt_file['density_map'][()][0:2,:,:])
            mask = np.asarray(gt_file['mask'][()][0:2,:,:])
            # k = np.asarray(gt_file['kpoint'][()])
            break
        except IOError:
            cv2.waitKey(5)

    img=img.copy()
    mask=mask.copy()
    mask[mask<=4]=1
    mask[mask>4]=0
    target=target.copy()
    k = 0

    return img, target, k, mask
