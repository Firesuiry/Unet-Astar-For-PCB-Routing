import json
import logging
import os

import cv2
import numpy as np

from problem import RandomProblem


def imwrite(name, img):
    # 逆时针旋转90度
    img = np.rot90(img, 3)
    # 左右翻转
    img = np.fliplr(img)
    if '/' in name or '\\' in name:
        logging.info(f'save image to {name}')
        cv2.imwrite(name, img)
    else:
        cv2.imwrite('img/' + name, img)


def sample_display(save_path, net_id):
    if save_path[-1] != '/':
        save_path += '/'
    # load data from json
    if not os.path.exists(save_path + f'data_{net_id}.json'):
        logging.error(f'no data_{net_id}.json')
        return
    with open(save_path + f'data_{net_id}.json', 'r') as f:
        data = json.load(f)
    # load feature map from npy
    feature_map = np.load(save_path + f'feature_map_{net_id}.npy')
    result = np.load(save_path + f'result_{net_id}.npy')
    # load problem from pkl
    problem = RandomProblem.load(save_path + f'problem.pkl')
    # load nets from json
    with open(save_path + f'nets.json', 'r') as f:
        nets = json.load(f)

    # display
    goal = data['goal']
    start = data['start']
    l = feature_map.shape[0]
    img_save_path = save_path + f'img/'
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    display(goal, l, nets, result, start, net_id, img_save_path)
    # save image of feature map
    for i in range(l):
        cv2.imwrite(img_save_path + f'feature_map_{net_id}_{i}.png', feature_map[i] * 255)


def display(goal, l, nets, result, start, delete_net_id, save_path=''):
    normal_result = result / np.max(result) * 255 if np.max(result) != 0 else result
    img = np.uint8(normal_result)
    for layer in range(l):
        # imwrite(save_path + 'result_' + str(layer) + '.png', img[layer])
        new_img = cv2.cvtColor(img[layer], cv2.COLOR_GRAY2BGR)

        # 画出path
        for net_id in range(len(nets)):
            path = nets[net_id]['path']
            for i in range(len(nets[net_id]['path'])):
                if path[i][0] == layer and i > 0:
                    cv2.line(new_img, (path[i - 1][2], path[i - 1][1]), (path[i][2], path[i][1]), (255, 0, 0),
                             1)
        # 画出start和goal
        cv2.circle(new_img, (start[2], start[1]), 2, (0, 0, 255), -1)
        cv2.circle(new_img, (goal[2], goal[1]), 2, (0, 255, 0), -1)

        img_name = save_path + f'result_{delete_net_id}_{layer}_with_path.png'
        imwrite(img_name, new_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    for i in range(80):
        sample_display(R'D:\develop\PCB\network\network\dataset\634fb1aaf564dc758a9415b267dbc82e', i)