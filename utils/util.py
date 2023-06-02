import numpy as np
from scipy.ndimage import gaussian_filter
import logging

def pin_hot_value_calculate(pins, max_x, max_y):
    logging.info('Calculating pin hot value...')
    # 创建一个二维数组，大小为max_x*max_y
    pin_hot_value = np.zeros((max_x, max_y))
    # 遍历所有的pin，将pin的位置的值加1
    poss = []
    for pin in pins:
        position = pin['x'], pin['y']
        pin_hot_value[position] += 1
        poss.append(position)
    # 计算所有pin的位置的标准差
    std = np.mean(np.std(poss, axis=0))
    # 将数组进行高斯模糊
    pin_hot_value = gaussian_filter(pin_hot_value, sigma=0.2 * std)
    # 将数组进行归一化
    pin_hot_value = pin_hot_value / np.max(pin_hot_value)
    # 将每个pin的hot_value加入到pin中
    for pin in pins:
        position = pin['x'], pin['y']
        pin['hot_value'] = pin_hot_value[position]
    return pin_hot_value
