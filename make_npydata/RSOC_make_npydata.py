import os
import numpy as np

if not os.path.exists('../npydata'):
    os.makedirs('../npydata')

'''please set your dataset path'''
try:
    DOTA_train_path='../dataset/RSOC/train_data/images/'
    DOTA_test_path='../dataset/RSOC/test_data/images/'

    train_list = []
    for filename in os.listdir(DOTA_train_path):
        if filename.split('.')[1] == 'png':
            train_list.append(DOTA_train_path.replace('..','.')+filename)
    train_list.sort()
    np.save('../npydata/RSOC_train.npy', train_list)


    test_list = []
    for filename in os.listdir(DOTA_test_path):
        if filename.split('.')[1] == 'png':
            test_list.append(DOTA_test_path.replace('..','.')+filename)
    test_list.sort()
    np.save('../npydata/RSOC_test.npy', test_list)
    print("Generate RSOC image list successfully")
except:
    print("The RSOC dataset path is wrong. Please check your path.")

