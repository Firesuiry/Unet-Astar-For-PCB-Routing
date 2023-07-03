import json
import logging
import os
import time

import cv2
import numpy as np
import torch

from network.data_loader import MyDataset
from network.unet import ResNetUNet
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


def sample_display(save_path, net_id, dataset, model):
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
    search_area = None
    if os.path.exists(save_path + f'search_area_{net_id}.npy'):
        search_area = np.load(save_path + f'search_area_{net_id}.npy')
    pred = None
    if model is not None:
        data, infer_fm, result0 = dataset.get_item(0, dir_name=save_path, i=net_id)
        for iii in range(infer_fm.shape[0]):
            cv2.imwrite(img_save_path + f'infer_fm_{net_id}_{iii}.png', (infer_fm[iii] * 255).astype(np.uint8))
        infer_fm = torch.from_numpy(infer_fm).unsqueeze(0).float()
        infer_fm = infer_fm.to('cuda')
        with torch.no_grad():
            s = time.time()
            pred = model(infer_fm)
            print(f'infer time: {time.time() - s}')
        pred = pred.squeeze(0).cpu().numpy()
        # 将预测结果通过sigmoid映射到0-1
        pred = 1 / (1 + np.exp(-pred))
        pred = pred[:, :feature_map.shape[1], :feature_map.shape[2]]
        for iii in range(pred.shape[0]):
            cv2.imwrite(img_save_path + f'infer_pred_{net_id}_{iii}.png', (pred[iii] * 255).astype(np.uint8))
    display(goal, l, nets, result, start, net_id, img_save_path, feature_map=feature_map, search_area=search_area,
            infer_result=pred)
    # save image of feature map
    # for i in range(l):
    #     cv2.imwrite(img_save_path + f'feature_map_{net_id}_{i}.png', feature_map[i] * 255)


def display(goal, l, nets, result, start, delete_net_id, save_path='', feature_map=None, search_area=None,
            infer_result=None, net_path=None):
    normal_result = result / np.max(result) * 255 if np.max(result) != 0 else result
    img = np.uint8(normal_result)
    for layer in range(l):
        # imwrite(save_path + 'result_' + str(layer) + '.png', img[layer])
        new_img = cv2.cvtColor(img[layer], cv2.COLOR_GRAY2BGR)
        new_img[feature_map[layer] == 1] = np.array([255, 0, 0])
        # 画出search_area
        if search_area is not None:
            # transposition search_area
            sa = np.transpose(search_area[layer])
            new_img[sa] = 0.5 * new_img[sa] + 0.5 * np.array([0, 0, 255])
        if infer_result is not None:
            new_img[infer_result[layer] > 0.5] = 0.5 * new_img[infer_result[layer] > 0.5] + \
                                                 0.5 * np.array([0, 255, 0])
        # 画出path
        for net_id in range(len(nets)):
            path = nets[net_id].get('path')
            if path is None and net_path is not None and net_id == delete_net_id:
                path = net_path
            if path is None:
                continue
            color = (255, 0, 0) if net_id != delete_net_id else (0, 255, 255)
            for i in range(len(path)):
                if path[i][0] == layer and i > 0:
                    cv2.line(new_img, (path[i - 1][2], path[i - 1][1]), (path[i][2], path[i][1]), color,
                             1)
        new_img[img[layer] == 255] = 0.5 * new_img[img[layer] == 255] + 0.5 * np.array([255, 255, 255])
        # 画出start和goal
        cv2.circle(new_img, (start[2], start[1]), 2, (0, 0, 255), -1)
        cv2.circle(new_img, (goal[2], goal[1]), 2, (0, 255, 0), -1)

        # if the size of new_img is too small, enlarge it so that its width and height is larger than 1000
        while new_img.shape[0] < 1000 or new_img.shape[1] < 1000:
            new_img = cv2.resize(new_img, (new_img.shape[1] * 2, new_img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = save_path + f'result_{delete_net_id}_{layer}_with_path.png'
        imwrite(img_name, new_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    model = dataset = None
    dataset = MyDataset(r'D:\develop\PCB\network\dataset\\', check=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(in_ch=3, out_ch=2).to(device)
    model.load_state_dict(torch.load(r'D:\develop\PCB\\best_val_model.pth', map_location=device))
    d1 = R'D:\develop\PCB\network\dataset\1ae9d019d912d9441e5c8a26678a168c'
    d2 = r'D:\dataset\000b4560cc2a244ecd10c87feaa36621'
    for i in range(0, 100):
        sample_display(d2, i, dataset=dataset,
                       model=model)
