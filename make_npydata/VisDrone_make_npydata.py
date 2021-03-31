import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''
try:
    VisDrone_train_path='/dssg/weixu/data_wei/VisDrone/train_data/images/'
    VisDrone_test_path='/dssg/weixu/data_wei/VisDrone/test_data/images/'

    train_list = []
    for filename in os.listdir(VisDrone_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(VisDrone_train_path+filename)
    train_list.sort()
    np.save('./npydata/VisDrone_train.npy', train_list)


    test_list = []
    for filename in os.listdir(VisDrone_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(VisDrone_test_path+filename)
    test_list.sort()
    np.save('./npydata/VisDrone_test.npy', test_list)
    print("Generate VisDrone image list successfully")
except:
    print("The VisDrone dataset path is wrong. Please check your path.")


# '''please set your dataset path'''
# try:
#     ShanghaiTech_train_path='/dssg/weixu/data_wei/ShanghaiTech/part_A_final/train_data/images/'
#     ShanghaiTech_test_path='/dssg/weixu/data_wei/ShanghaiTech/part_A_final/test_data/images/'
#
#     train_list = []
#     for filename in os.listdir(ShanghaiTech_train_path):
#         if filename.split('.')[1] == 'jpg':
#             train_list.append(ShanghaiTech_train_path+filename)
#     train_list.sort()
#     np.save('./npydata/ShanghaiTech_train.npy', train_list)
#
#
#     test_list = []
#     for filename in os.listdir(ShanghaiTech_test_path):
#         if filename.split('.')[1] == 'jpg':
#             test_list.append(ShanghaiTech_test_path+filename)
#     test_list.sort()
#     np.save('./npydata/ShanghaiTech_test.npy', test_list)
#     print("Generate ShanghaiTech image list successfully")
# except:
#     print("The ShanghaiTech dataset path is wrong. Please check your path.")
