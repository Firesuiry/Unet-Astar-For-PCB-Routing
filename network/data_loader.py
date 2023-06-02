import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2

# Your Data Path
img_dir = '/home/jyz/Downloads/classify_example/val/骏马/'
anno_file = '/home/jyz/Downloads/classify_example/val/label.txt'


class MyDataset(Dataset):
    def __init__(self, data_dir, max_num=-1):
        self.data_list = []
        self.data_dir = data_dir
        if data_dir is None:
            return
        path = Path(data_dir)
        num = 0
        for file_dir in path.glob('*'):
            i = 0
            while os.path.exists(str(file_dir) + '/data_' + str(i) + '.json'):
                self.data_list.append((file_dir.name, i))
                i += 1
                num += 1
                if num % 5000 == 0:
                    logging.info(f'load {num} data')
                if 0 < max_num <= num:
                    break
        logging.info(f'load {len(self.data_list)} data [finished]')

    def get_train_and_test_Dataset(self, test_size=0.2):
        train_dataset = MyDataset(None)
        train_dataset.data_dir = self.data_dir
        test_dataset = MyDataset(None)
        test_dataset.data_dir = self.data_dir
        for data in self.data_list:
            if random.random() < test_size:
                test_dataset.data_list.append(data)
            else:
                train_dataset.data_list.append(data)
        return train_dataset, test_dataset

    # need to overload
    def __len__(self):
        return len(self.data_list)

    # need to overload
    def __getitem__(self, idx):
        dir_name, i = self.data_list[idx]
        data_file = self.data_dir + dir_name + '/data_' + str(i) + '.json'
        feature_map_file = self.data_dir + dir_name + '/feature_map_' + str(i) + '.npy'
        result_file = self.data_dir + dir_name + '/result_' + str(i) + '.npy'
        # load data with json
        data = json.load(open(data_file, 'r'))
        # load feature map with npy
        feature_map = np.load(feature_map_file)
        # load result with npy
        result = np.load(result_file)
        # x_size = math.ceil(feature_map.shape[1] / 32) * 32
        # y_size = math.ceil(feature_map.shape[2] / 32) * 32
        x_size = math.ceil(5000 / 8 / 32) * 32
        y_size = math.ceil(5000 / 8 / 32) * 32
        new_result = np.zeros((result.shape[0], x_size, y_size), dtype=np.float32)
        new_result[:, :result.shape[1], :result.shape[2]] = result
        result = new_result
        img = np.ones((feature_map.shape[0] + 1, x_size, y_size), dtype=np.float32)
        img[1:, :feature_map.shape[1], :feature_map.shape[2]] = feature_map
        start = data['start']
        goal = data['goal']
        img[0] = point_feature_generate(img[0], goal, start)
        return img, result
        # return img, label


def point_feature_generate(img, goal, start):
    img[:, :] = 0
    img[start[1], start[2]] = 1
    img[goal[1], goal[2]] = 1
    sigma = math.hypot(start[1] - goal[1], start[2] - goal[2]) * 0.3
    img = gaussian_filter(img, sigma=sigma)
    img = img / img.max() if img.max() > 0 else img
    return img


# dataset = MyDataset(img_dir, anno_file)
# dataloader = DataLoader(dataset=dataset, batch_size=2)
#
# # display
# for img_batch, label_batch in dataloader:
#     img_batch = img_batch.numpy()
#     print(img_batch.shape)
#     # img = np.concatenate(img_batch, axis=0)
#     if img_batch.shape[0] == 2:
#         img = np.hstack((img_batch[0], img_batch[1]))
#     else:
#         img = np.squeeze(img_batch, axis=0)  # 最后一张图时，删除第一个维度
#     print(img.shape)
#     cv2.imshow(label_batch[0], img)
#     cv2.waitKey(0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    dataset = MyDataset(r'D:\\develop\\PCB\\network\\dataset\\')
    img, result = dataset[79]
    for i in range(img.shape[0]):
        cv2.imwrite(f'img{i}.jpg', img[i])
    for i in range(result.shape[0]):
        cv2.imwrite(f'result{i}.jpg', result[i].astype(np.uint8) * 255)
