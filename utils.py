import h5py
import torch
import shutil
import numpy as np
import cv2
import os


def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    density_map[density_map < 0] = 0
    input_img = input_img[0][0].astype(np.uint8)
    # print(gt_data.shape, density_map.shape)
    # result_img = np.hstack((gt_data[0][0].astype(np.uint8),density_map[0][0].astype(np.uint8)))

    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)
    # if density_map.shape[1] != input_img.shape[1]:
    #     density_map = cv2.resize(density_map, (input_img.shape[1],input_img.shape[0]))
    #     gt_data = cv2.resize(gt_data, (input_img.shape[1],input_img.shape[0]))

    # cv2.imwrite(os.path.join('.',output_dir,fname).replace('.h5','gt.bmp').replace('.jpg','gt.bmp'),gt_data)
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)
    # cv2.imwrite(os.path.join('.',output_dir,fname).replace('.h5','refine.bmp').replace('.jpg','refine.bmp'),density_map)

    # ori_img = cv2.imread('/data/weixu/ShanghaiTech/part_A_final/test_data/images/' + fname)

    # result_img = np.hstack((gt_data,density_map))
    result_img = density_map

    cv2.imwrite(os.path.join('.', output_dir, fname).replace('.h5', 'fuse.jpg').replace('.jpg', '.jpg'), result_img)


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, visi, is_best, task_id, filename='checkpoint.pth'):
    torch.save(state, './' + str(task_id) + '/' + filename)
    if is_best:
        shutil.copyfile('./' + str(task_id) + '/' + filename, './' + str(task_id) + '/' + 'model_best.pth')

    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, str(task_id), fname[0])
