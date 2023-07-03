import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import tqdm
import multiprocessing as mp

# Your Data Path
img_dir = '/home/jyz/Downloads/classify_example/val/骏马/'
anno_file = '/home/jyz/Downloads/classify_example/val/label.txt'


def item_exist(file_dir, idx):
    return os.path.exists(str(file_dir) + '/data_' + str(idx) + '.json') and \
        os.path.exists(str(file_dir) + '/feature_map_' + str(idx) + '.npy') and \
        os.path.exists(str(file_dir) + '/result_' + str(idx) + '.npy')


class MyDataset(Dataset):
    def __init__(self, data_dir, max_num=-1, check=True):
        self.data_list = []
        self.data_dir = data_dir
        if data_dir is None:
            return
        path = Path(data_dir)
        num = 0
        for file_dir in path.glob('*'):
            i = 0
            for index in range(100):
                if item_exist(file_dir, index):
                    i = index
                    break
            while item_exist(file_dir, i):
                self.data_list.append((file_dir.name, i))
                i += 1
                num += 1
                if num % 5000 == 0:
                    logging.info(f'load {num} data')
                if 0 < max_num <= num:
                    break
        logging.info(f'load {len(self.data_list)} data [finished]')
        if check:
            self.check_data()
            logging.info(f'check {len(self.data_list)} data [finished]')

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

    def check_data(self):
        new_data_list = []
        for idx in tqdm.tqdm(range(len(self.data_list))):
            try:
                data, feature_map, result = self.get_item(idx)
                new_data_list.append(self.data_list[idx])
            except:
                dir_name, i = self.data_list[idx]
                data_file = self.data_dir + dir_name + '/data_' + str(i) + '.json'
                feature_map_file = self.data_dir + dir_name + '/feature_map_' + str(i) + '.npy'
                result_file = self.data_dir + dir_name + '/result_' + str(i) + '.npy'
                os.remove(data_file)
                os.remove(feature_map_file)
                os.remove(result_file)
                logging.info(f'error in {idx}')
        self.data_list = new_data_list

    # need to overload
    def __getitem__(self, idx):
        data, feature_map, result = self.get_item(idx)
        # x_size = math.ceil(feature_map.shape[1] / 32) * 32
        # y_size = math.ceil(feature_map.shape[2] / 32) * 32
        x_size = int(2048 / 8)
        y_size = int(2048 / 8)
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

    def get_item(self, idx, dir_name=None, i=None):
        process = True
        if dir_name is None or i is None:
            dir_name, i = self.data_list[idx]
            dir_name = self.data_dir + dir_name
            process = False
        data_file = dir_name + '/data_' + str(i) + '.json'
        feature_map_file = dir_name + '/feature_map_' + str(i) + '.npy'
        result_file = dir_name + '/result_' + str(i) + '.npy'
        # load data with json
        data = json.load(open(data_file, 'r'))
        # load feature map with npy
        feature_map = np.load(feature_map_file)
        # load result with npy
        result = np.load(result_file)
        img = feature_map
        if process:
            x_size = int(2048 / 8)
            y_size = int(2048 / 8)
            new_result = np.zeros((result.shape[0], x_size, y_size), dtype=np.float32)
            new_result[:, :result.shape[1], :result.shape[2]] = result
            result = new_result
            img = np.ones((feature_map.shape[0] + 1, x_size, y_size), dtype=np.float32)
            img[1:, :feature_map.shape[1], :feature_map.shape[2]] = feature_map
            start = data['start']
            goal = data['goal']
            img[0] = point_feature_generate(img[0], goal, start)
        return data, img, result


def point_feature_generate(img, goal, start):
    img[:, :] = 0
    img[start[1], start[2]] = 1
    img[goal[1], goal[2]] = 1
    sigma = math.hypot(start[1] - goal[1], start[2] - goal[2]) * 0.3
    img = gaussian_filter(img, sigma=sigma)
    img = img / img.max() if img.max() > 0 else img
    return img


def write_data(path, data):
    with open(path, 'wb') as f:
        f.write(data)


def copy_file(src, dst):
    with open(src, 'rb') as f:
        data = f.read()
        with open(dst, 'wb') as f2:
            f2.write(data)
    return src

need_copy_files = []

def copy_file_callback(src):
    global need_copy_files
    if src in need_copy_files:
        need_copy_files.remove(src)
    len_files = len(need_copy_files)
    if len_files % 100 == 0 and len_files > 0:
        print(f'剩余{len_files}个文件')

def data_copy(source_path, target_path):
    global need_copy_files
    source_path = Path(source_path)
    target_path = Path(target_path)
    pool = mp.Pool(8)
    spls = list(source_path.glob('*'))
    random.shuffle(spls)
    for file_dir in tqdm.tqdm(spls):
        need_copy_files = list(file_dir.glob('*'))
        if len(need_copy_files) == 0:
            continue
        # logging.info(f'copy {file_dir.name}')
        target_file_dir = target_path / file_dir.name
        if not target_file_dir.exists():
            target_file_dir.mkdir()
        for file in need_copy_files:
            target_file = target_file_dir / file.name
            if file.is_dir():
                continue
            if 'search_area' in file.name:
                continue
            if os.path.exists(target_file):
                continue
            pool.apply_async(copy_file, args=(file, target_file), callback=copy_file_callback)
            need_copy_files.append(file)
            # with open(file, 'rb') as f:
            #     data = f.read()
            #     p = mp.Process(target=write_data, args=(target_file, data))
            #     p.start()
            # with open(target_file, 'wb') as f:
            #     f.write(data)
            # shutil.copy(file, target_file)
            # os.system(f'copy {file} {target_file}')
    pool.close()
    pool.join()


def test_dataset():
    dataset = MyDataset(r'D:\\develop\\PCB\\network\\dataset\\')
    img, result = dataset[79]
    for i in range(img.shape[0]):
        cv2.imwrite(f'img\\img{i}.jpg', img[i] * 255)
    for i in range(result.shape[0]):
        cv2.imwrite(f'img\\result{i}.jpg', result[i].astype(np.uint8) * 255)


def test_dataset2():
    dataset = MyDataset(r'D:\dataset\\')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    # test_dataset()
    # data_copy(r'Z:\network\dataset\\', r'D:\dataset\\')
    test_dataset2()
