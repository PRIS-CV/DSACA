import os
import numpy as np

if not os.path.exists('../npydata'):
    os.makedirs('../npydata')

'''please set your dataset path'''
try:
    VisDrone_train_path='../dataset/VisDrone/train_data_class8/images/'
    VisDrone_test_path='../dataset/VisDrone/test_data_class8/images/'

    train_list = []
    for filename in os.listdir(VisDrone_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(VisDrone_train_path.replace('..','.')+filename)
    train_list.sort()
    np.save('../npydata/VisDrone_train.npy', train_list)


    test_list = []
    for filename in os.listdir(VisDrone_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(VisDrone_test_path.replace('..','.')+filename)
    test_list.sort()
    np.save('../npydata/VisDrone_test.npy', test_list)
    print("Generate VisDrone image list successfully")
except:
    print("The VisDrone dataset path is wrong. Please check your path.")

