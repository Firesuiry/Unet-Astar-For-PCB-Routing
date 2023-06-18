import copy
import json
import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
import logging
import math

import torch

from network.data_loader import point_feature_generate
from network.utils.obs_feature_map_generate import obs_feature_map_generate
from network.unet import ResNetUNet
from solver.astar import AStar, T
from utils.critic_path import key_path2path, path2key_path
from utils.net_flag import update_net_flag
from utils.net_layer_assign import layer_assign
from multiprocessing import shared_memory
from solver.astar.multilayer_astar import MazeSolver as MultiLayerAstar
from sklearn.cluster import KMeans

from utils.pin_group_divide import pin_group_divide

import multiprocessing as mp

from utils.util import pin_hot_value_calculate

NET_FLAG_RESOLUTION = 10
Infinite = float("inf")
DIRECTIONS = np.array([
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
])

NINE_DIRECTIONS = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
    ]
)


class AstarNode:
    __slots__ = ("data", "gscore", "fscore", "closed", "came_from", "out_openset")

    def __init__(
            self, data: T, gscore: float = Infinite, fscore: float = Infinite
    ) -> None:
        self.data = data
        self.gscore = gscore
        self.fscore = fscore
        self.closed = False
        self.out_openset = True
        self.came_from = None

    def __lt__(self, b: "AStar.SearchNode") -> bool:
        return self.fscore < b.fscore


class FlagDict(dict):

    def __missing__(self, key):
        v = []
        self.__setitem__(key, v)
        return v


def tuple_reverse(t):
    return t[1], t[0]


def shared_memory_to_numpy(name, shape, dtype):
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return arr


def numpy_to_shared_memory(arr, name, data_dict):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    shm.buf[:] = arr.tobytes()
    return shm


def _route(ra, _solver, ori_position, target_pos, net_id, save_path_flag=False):
    s = time.time()
    _found_path, search_area = _solver.astar(ori_position, target_pos)
    msg = f'{net_id}布线结束  耗时{time.time() - s:.2f}s ra:{ra is not None}'
    file_name = f'logs/ra0.txt' if ra is None else f'logs/ra1.txt'
    with open(file_name, 'a+', encoding='utf-8') as f:
        f.write(datetime.now().strftime("%Y-%m-%d-%H-%M-%S|") + msg + '\n')
    logging.info(msg)
    if save_path_flag:
        with open(f'data/net_path/net{net_id}_{ra is not None}_found_path.pickle', 'wb') as f:
            pickle.dump(_found_path, f)
    return _found_path, search_area


class BaseSolver:
    name = "base_solver"

    def __init__(self, problem, **kwargs):
        self.search_areas = {}
        self.net_groups = None
        self.problem = problem
        self.max_x = self.problem.max_x
        self.max_y = self.problem.max_y
        self.layer_num = self.problem.layer_num
        self.grid = self.problem.grid
        self.clearance = self.problem.clearance
        self.line_width = self.problem.line_width
        self.resolution = 1
        if hasattr(self.problem, 'via_radius'):
            self.via_radius = self.problem.via_radius
        else:
            self.problem.via_radius = self.via_radius = 0
        self.pins = copy.deepcopy(self.problem.pins)
        self.padstacks = self.problem.padstacks
        self.padstack_range = {}
        self.nets = self.problem.nets
        self.cross_punish = 0

        self.run_time = 0

        self.steiner_nets = []
        self.pending_nets = []
        # 将多个节点的连接切成两两连接
        self.steiner_tree_net_divide()

        # 标记障碍物情况 -1为无障碍物
        self.obstacle = None
        # 标记某条线是否经过某区域
        self.net_flags = None
        self.net_flag_details = {}
        self.cross_relation = np.zeros((len(self.steiner_nets), len(self.steiner_nets)), dtype=bool)
        self.net_assemble_flag = np.zeros((len(self.steiner_nets), len(self.steiner_nets)), dtype=float) + 1e9
        self.net_assemble_check()
        for i in range(len(self.steiner_nets)):
            self.steiner_nets[i]['index'] = i
            self.pending_nets.append(i)
        for i in range(len(self.problem.nets)):
            self.net_flag_details[i] = FlagDict()

        try:
            self.shm = shared_memory.SharedMemory(name='solver_data')
        except:
            self.shm = shared_memory.SharedMemory(create=True, size=2 ** 30, name='solver_data')

        self.hx_multi_rate = kwargs.get('hx_multi_rate', 1)
        self.jps_search_rate = kwargs.get('jps_search_rate', 0.1)
        self.model = None
        self.device = None
        if kwargs.get('model_path', None):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ResNetUNet(2).to(self.device)
            self.model.load_state_dict(torch.load(kwargs['model_path'], map_location=self.device))
        # self.resolution_change(2)
        self.pin_groups = []

    def get_name(self):
        return f'{self.name}_hx{self.hx_multi_rate}_jps{self.jps_search_rate}'

    def resolution_change(self, resolution=1):
        old_resolution = self.resolution
        self.resolution = resolution
        self.line_width = math.ceil(self.problem.line_width / resolution)
        self.clearance = math.ceil(self.problem.clearance / resolution)
        self.via_radius = math.ceil(self.problem.via_radius / resolution)
        self.max_x = int(self.problem.max_x / resolution)
        self.max_y = int(self.problem.max_y / resolution)
        logging.info(f'分辨率修改为{resolution}, max_x={self.max_x}, max_y={self.max_y}')

        self.pins = copy.deepcopy(self.problem.pins)
        for pin in self.pins:
            pin['x'] = int(pin['x_int'] / resolution)
            pin['y'] = int(pin['y_int'] / resolution)
        pin_hot_value_calculate(self.pins, self.max_x, self.max_y)
        for i in range(len(self.pin_groups)):
            pin_group = self.pin_groups[i]
            for pin_index in pin_group:
                self.pins[pin_index]['pin_group'] = i
        for k, v in self.padstacks.items():
            for kk, vv in v.items():
                vv['detail_p'] = [math.ceil(d / resolution) for d in vv['detail']]

        self.generate_obstacle()
        for i in range(len(self.steiner_nets)):
            self.net_refine(i)
        self.generate_net_flag()

    def route(self, net_id, **kwargs):
        i = net_id
        old_index = self.steiner_nets[i]['old_index']
        foundPath = self.net_route(i, cross_punish=self.cross_punish, **kwargs)
        # 更新到敏感列表上
        # self.net_flags[0, :, :, old_index] = False
        # self.net_flag_details[old_index].clear()
        if foundPath:
            foundPath = key_path2path(foundPath)
            key_path = path2key_path(foundPath)
            update_net_flag(resolution=NET_FLAG_RESOLUTION, path=foundPath,
                            old_index=old_index, obstacle=self.obstacle, line_width=self.line_width,
                            via_radius=self.via_radius
                            )
            # plot self.obstacle and save it to img/obstacle/{net_id}.png
            # img = (self.obstacle.copy() != -1).astype(np.uint8) * 255
            # for layer in range(self.problem.layer_num):
            #     cv2.imwrite(f'img/obstacle/{net_id}_{layer}.png', img[layer])
            # save (self.obstacle.copy() != -1) to img/obstacle/{net_id}.npy
            # np.save(f'img/obstacle/{net_id}.npy', (self.obstacle.copy() != -1).astype(bool))
            self.steiner_nets[i]['path'] = key_path
            self.steiner_nets[i]['resolution'] = self.resolution
        self.save_data()

    def resolution_solve(self, resolution=1, cross_punish=0):
        logging.info(f'开始分辨率为{resolution},惩罚为{cross_punish}的解算')
        self.resolution_change(resolution)
        self.cross_check()
        self.cross_punish = cross_punish
        t = time.time()
        for i in self.pending_nets:
            # if i != 170:
            #     continue
            self.route(i)
        logging.info(f'分辨率为{resolution},惩罚为{cross_punish}的解算完成, 耗时{time.time() - t:.2f}s')
        self.cross_check()
        ...

    def net_group_generate0(self, assembly_max=0.1):
        self.net_groups = []
        processed_nets = []
        pending_nets = list(range(len(self.steiner_nets)))
        for net_id in pending_nets:
            if net_id in processed_nets:
                continue
            nets = self.get_assemble_nets(net_id, assembly_max)
            if len(nets) <= 1:
                continue
            self.net_groups.append(nets)
            processed_nets.extend(nets)
        self.net_groups = sorted(self.net_groups, key=lambda ls: len(ls), reverse=True)
        ...

    def net_group_generate(self):
        self.net_groups = []

        class NetGroupDict(dict):
            def __missing__(self, key):
                self[key] = []
                return self[key]

        net_group_dict = NetGroupDict()
        for i in range(len(self.steiner_nets)):
            net = self.steiner_nets[i]
            pin1 = net['pins'][0]
            pin2 = net['pins'][1]
            pin_group1 = self.pins[pin1]['pin_group']
            pin_group2 = self.pins[pin2]['pin_group']
            # 将pin_group1和pin_group2合并成从小到大的tuple
            if pin_group1 > pin_group2:
                pin_group1, pin_group2 = pin_group2, pin_group1
            pin_group = (pin_group1, pin_group2)
            net_group_dict[pin_group].append(i)
        for k, v in net_group_dict.items():
            if len(v) > 1:
                data = {
                    'nets': v,
                    'pin_group': k
                }
                self.net_groups.append(data)
        self.net_groups = sorted(self.net_groups, key=lambda d: len(d['nets']), reverse=True)

    def net_group_route(self, net_group_id, **kwargs):
        net_group = self.net_groups[net_group_id]
        # collect all points in one layer
        points = []
        for net_id in net_group:
            pin0, pin1 = self.steiner_nets[net_id]['pins'][0], self.steiner_nets[net_id]['pins'][1]
            points.append((self.pins[pin0]['x'], self.pins[pin0]['y']))
            points.append((self.pins[pin1]['x'], self.pins[pin1]['y']))
        # kmeans
        os.environ['OMP_NUM_THREADS'] = '1'
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(points)
        # get cluster center
        center0 = kmeans.cluster_centers_[0]
        center1 = kmeans.cluster_centers_[1]
        # 计算nets距离center的偏离程度并进行排序
        net_shift = []
        for net_id in net_group:
            pin0, pin1 = self.steiner_nets[net_id]['pins'][0], self.steiner_nets[net_id]['pins'][1]
            x0, y0 = self.pins[pin0]['x'], self.pins[pin0]['y']
            x1, y1 = self.pins[pin1]['x'], self.pins[pin1]['y']
            distance1 = np.min([math.sqrt((x0 - center0[0]) ** 2 + (y0 - center0[1]) ** 2),
                                math.sqrt((x1 - center0[0]) ** 2 + (y1 - center0[1]) ** 2)])
            distance2 = np.min([math.sqrt((x0 - center1[0]) ** 2 + (y0 - center1[1]) ** 2),
                                math.sqrt((x1 - center1[0]) ** 2 + (y1 - center1[1]) ** 2)])
            net_shift.append((net_id, distance1 + distance2))
        net_shift = sorted(net_shift, key=lambda x: x[1])

        # 根据偏离程度从小到大进行route
        for net_id, _ in net_shift:
            self.route(net_id, **kwargs)
            self.pending_nets.remove(net_id)

    def solve(self):
        # self.resolution_solve(512, 0)
        # self.resolution_solve(128, 10)
        # self.resolution_solve(32, 30)
        s = time.time()
        self.resolution_solve(2, 100)
        ...
        # if self.cross_check():
        #     return
        # self.net_layer_assign()
        # for i in range(len(self.steiner_nets)):
        #     self.route(i, cross_punish=100)
        # self.resolution_change(8)
        #
        # for i in self.pending_nets:
        #     self.route(i, cross_punish=100)
        # self.generate_net_flag()
        # if self.cross_check():
        #     return
        # self.net_layer_assign()
        # self.cross_check()
        # for i in self.pending_nets:
        #     self.route(i, cross_punish=100)
        # self.generate_net_flag()
        # if self.cross_check():
        #     return
        # self.resolution_change(8)
        # for i in self.pending_nets:
        #     self.route(i, cross_punish=100)
        # self.cross_check()
        # self.resolution_change(1)
        self.run_time = time.time() - s
        return self.steiner_nets, self.resolution

    def cross_check(self):
        self.save_data()
        self.generate_net_flag()
        logging.info('检测是否有重叠')
        need_remove = []
        no_complete_nets = []
        line_width = int((self.line_width + self.clearance))
        logging.debug(f'line_width={line_width}')
        for net_id in range(len(self.steiner_nets)):
            net = self.steiner_nets[net_id]
            old_index = net['old_index']
            logging.debug(f'开始检测net{net_id}是否有重叠')
            if net.get('path'):
                cross = False
                for point in key_path2path(net['path']):
                    layer_id, x, y = point
                    min_distance = Infinite
                    flagx0, flagy0 = int(x / NET_FLAG_RESOLUTION), int(y / NET_FLAG_RESOLUTION)
                    for i in range(9):
                        flagx = flagx0 + NINE_DIRECTIONS[i][0]
                        flagy = flagy0 + NINE_DIRECTIONS[i][1]
                        if flagx < 0 or flagx >= self.net_flags.shape[1] or flagy < 0 or flagy >= self.net_flags.shape[
                            2]:
                            continue
                        for neighbor_net in np.where(self.net_flags[layer_id, flagx, flagy, :])[0]:
                            if old_index == neighbor_net:
                                continue
                            neighbor_points = self.net_flag_details[neighbor_net][(layer_id, flagx, flagy)]
                            _neighbor_points = neighbor_points
                            for d in neighbor_points:
                                point = d['point']
                                point_net_id = d['net_id']
                                distance = math.hypot(x - point[1], y - point[2])
                                if distance <= line_width:
                                    logging.info(
                                        f'线{net_id}与{point_net_id}重叠 交点{point}|{(layer_id, x, y)} 距离{distance}|{line_width}')
                                    cross = True
                                if distance < min_distance:
                                    min_distance = distance
                logging.debug(f'线{net_id}重叠{cross}')
                net['cross'] = cross
                if not cross:
                    net['cross'] = False
                    need_remove.append(net_id)
                    logging.debug(f'线{net_id}完成')
                else:
                    if net_id not in self.pending_nets:
                        logging.debug(f'线{net_id}重叠 加入待完成列表')
                        self.pending_nets.append(net_id)
            else:
                logging.debug(f'线{net_id}未完成')
                if net_id not in self.pending_nets:
                    self.pending_nets.append(net_id)
                no_complete_nets.append(net_id)
        logging.debug(f'need_remove={need_remove}')
        for net_id in need_remove:
            if net_id in self.pending_nets:
                self.pending_nets.remove(net_id)
        logging.info(f'剩余{len(self.pending_nets)}条线未完成 {self.pending_nets[:20]}')
        if len(no_complete_nets) > 0:
            logging.warning(f'未完成的线数量：{len(no_complete_nets)} {no_complete_nets}')
            return False
        if len(self.pending_nets) == 0:
            logging.info('所有线完成')
            return True
        else:
            return False

    def running_result(self):
        # 统计 线路长度 VIA数量 连通率 运行时间 设计规则违例
        data = {
            '线路长度': 0,
            'VIA数量': 0,
            '连通率': 0,
            '运行时间': self.run_time,
            '设计规则违例': -1,
        }
        未完成的线数量 = 0
        for net in self.steiner_nets:
            if net.get('path'):
                path = net['path']
                for i in range(1, len(path)):
                    data['线路长度'] += math.hypot(path[i][1] - path[i - 1][1], path[i][2] - path[i - 1][2])
                    if path[i][0] != path[i - 1][0]:
                        data['VIA数量'] += 1
            else:
                未完成的线数量 += 1
        data['连通率'] = (len(self.steiner_nets) - 未完成的线数量) / len(self.steiner_nets)
        return data

    def cross_relation_generate(self):
        self.cross_relation[:] = False
        line_width = int((self.line_width + self.clearance))
        for net_id in range(len(self.steiner_nets)):
            net = self.steiner_nets[net_id]
            old_index = net['old_index']
            if net.get('path'):
                for point in key_path2path(net['path']):
                    x, y = point
                    flagx0, flagy0 = int(x / NET_FLAG_RESOLUTION), int(y / NET_FLAG_RESOLUTION)
                    for i in range(9):
                        flagx = flagx0 + NINE_DIRECTIONS[i][0]
                        flagy = flagy0 + NINE_DIRECTIONS[i][1]
                        for layer in range(self.layer_num):
                            for neighbor_net in np.where(self.net_flags[layer, flagx, flagy, :])[0]:
                                if old_index == neighbor_net:
                                    continue
                                neighbor_points = self.net_flag_details[neighbor_net][(layer, flagx, flagy)]
                                for d in neighbor_points:
                                    point = d['point']
                                    point_net_id = d['net_id']
                                    distance = math.hypot(x - point[0], y - point[1])
                                    if distance < line_width:
                                        self.cross_relation[net_id, point_net_id] = \
                                            self.cross_relation[point_net_id, net_id] = True

    def net_layer_assign(self):
        self.cross_relation_generate()
        net_layer = layer_assign(self.cross_relation, layer_max=self.problem.layer_num - 1)
        for i in range(len(self.steiner_nets)):
            self.steiner_nets[i]['layer_id'] = net_layer[i]

    def generate_net_flag(self):
        logging.info('生成网线标记')
        self.net_flags = np.zeros((self.problem.layer_num, math.ceil(self.max_x / NET_FLAG_RESOLUTION) + 1,
                                   math.ceil(self.max_y / NET_FLAG_RESOLUTION) + 1,
                                   len(self.nets)), dtype=bool)
        for i in range(len(self.nets)):
            self.net_flag_details[i].clear()
        for i in range(len(self.steiner_nets)):
            old_index = self.steiner_nets[i]['old_index']
            if not self.steiner_nets[i].get('path'):
                continue
            path = key_path2path(self.steiner_nets[i]['path'])
            update_net_flag(resolution=NET_FLAG_RESOLUTION, path=path,
                            old_index=old_index, obstacle=self.obstacle, line_width=self.line_width,
                            via_radius=self.via_radius, )

    def generate_obstacle(self):
        logging.info('生成障碍物')
        self.obstacle = np.zeros((self.layer_num, self.max_x + 1, self.max_y + 1), dtype=np.int16) - 1
        for pin in self.pins:
            center = np.array([pin['x'], pin['y']], dtype=int)
            pin_id = pin['index']
            pad_key = pin['shape']
            pad_data = self.padstacks[pad_key]
            pin['layers'] = []
            for k, v in pad_data.items():
                layer_id = int(k) - 1
                pin['layers'].append(layer_id)
                shape = v['shape']
                detail = v['detail_p']
                if shape == 'circle':
                    radius = int(detail[0] / 2)
                    delta_x, delta_y = 0, 0
                    if len(detail) > 1:
                        delta_x = detail[1]
                        delta_y = detail[2]
                    delta_pos = np.array([delta_x, delta_y], dtype=int)
                    if not self.padstack_range.get(pad_key):
                        self.padstack_range[pad_key] = {'x_min': -radius, 'x_max': radius, 'y_min': -radius,
                                                        'y_max': radius}
                    cv2.circle(self.obstacle[layer_id, :, :], tuple_reverse(center + delta_pos), radius,
                               (pin.get('net_id', -2),),
                               -1)  # 填充
                elif shape == 'polygon':
                    point_num = int(len(detail) / 2)
                    pts = np.zeros((point_num, 2), dtype=np.int32)  # 数据类型必须为 int32
                    for i in range(point_num):
                        pts[i, 1] = center[0] + detail[i * 2 + 1]
                        pts[i, 0] = center[1] + detail[i * 2 + 2]
                    if not self.padstack_range.get(pad_key):
                        self.padstack_range[pad_key] = {'x_min': np.min(pts[:, 1]) - center[0],
                                                        'x_max': np.max(pts[:, 1]) - center[0],
                                                        'y_min': np.min(pts[:, 0]) - center[1],
                                                        'y_max': np.max(pts[:, 0]) - center[1]}
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(self.obstacle[layer_id, :, :], [pts], color=(pin.get('net_id', -2),))

    def get_assemble_nets(self, net_id, assembly_max, nets=None):
        if nets is None:
            nets = []
        if np.sum(self.net_assemble_flag[net_id] < assembly_max) < 1:
            return [net_id]
        if net_id not in nets:
            nets.append(net_id)
        for i in range(len(self.steiner_nets)):
            if i in nets:
                continue
            if self.net_assemble_flag[net_id, i] < assembly_max:
                new_nets = self.get_assemble_nets(i, assembly_max, nets)
                for net in new_nets:
                    if net not in nets:
                        nets.append(net)
        return nets

    def net_group_display(self, net_group_id):
        img = self.net_img.copy()
        w = 1

        def plot_net(net, color):
            start = self.pins[net['pins'][0]]['y'], self.pins[net['pins'][0]]['x']
            end = self.pins[net['pins'][1]]['y'], self.pins[net['pins'][1]]['x']
            cv2.line(img, start, end, color, thickness=w, lineType=cv2.LINE_4)
            logging.debug(f'划线 {start} {end} {color} {w}')

        nets = self.net_groups[net_group_id]['nets']
        for net in nets:
            plot_net(self.steiner_nets[net], (0, 255, 255))

        mat270 = np.rot90(img, 1)
        file_name = f'img/net{net_group_id}.png'
        cv2.imwrite(file_name, mat270)
        logging.info(f'画图结束 保存为{file_name}')

    def assemble_net_display(self, net_id):
        assembly_max = 0.1
        if np.sum(self.net_assemble_flag[net_id] < assembly_max) < 1:
            return
        if not hasattr(self, 'ploted_nets'):
            self.ploted_nets = []
        if net_id in self.ploted_nets:
            return
        img = self.net_img.copy()
        w = net_display_weight = 1
        net = self.steiner_nets[net_id]

        def plot_net(net, color):
            start = self.pins[net['pins'][0]]['y'], self.pins[net['pins'][0]]['x']
            end = self.pins[net['pins'][1]]['y'], self.pins[net['pins'][1]]['x']
            cv2.line(img, start, end, color, thickness=w, lineType=cv2.LINE_4)
            logging.debug(f'划线 {start} {end} {color} {w}')

        nets = self.get_assemble_nets(net_id, assembly_max)
        i = 0
        while True:
            if i >= len(nets):
                break
            new_nets = self.get_assemble_nets(nets[i], assembly_max)
            for net in new_nets:
                if net not in nets:
                    nets.append(net)
            i += 1
        self.ploted_nets.extend(nets)
        for net in nets:
            plot_net(self.steiner_nets[net], (0, 255, 255))

        mat270 = np.rot90(img, 1)
        file_name = f'img/net{net_id}.png'
        cv2.imwrite(file_name, mat270)
        logging.info(f'画图结束 保存为{file_name}')

    def save_data(self):
        data = {
            'pins': self.pins,
            'steiner_nets': self.steiner_nets,
            'padstacks': self.padstacks,
            'layer_num': self.layer_num,
            'max_x': self.max_x,
            'max_y': self.max_y,
        }
        binary_data = pickle.dumps(data)
        data_len = np.int64(len(binary_data))
        self.shm.buf[:8] = data_len.tobytes()
        self.shm.buf[8:8 + data_len] = binary_data

    def display(self, base_img=None, layers=None, save_path=None):
        if layers is None:
            layers = list(range(self.layer_num))
        img = np.zeros(((self.max_x + 1), (self.max_y + 1), 3), dtype=np.uint8)
        for layer in range(self.layer_num):

            # cv2.rectangle(img, (-1, -1), (self.max_y + 1, self.max_x + 1), (255, 255, 255), 1)

            img[np.where(self.obstacle[layer, :, :] != -1)] = (120, 120, 120)
            w = net_display_weight = 1
            for net in self.steiner_nets:
                if net.get('layer_id', 0) != layer:
                    continue
                foundPath = net.get('path')
                if foundPath:
                    path = key_path2path(foundPath)
                    for layer_id, x, y in path:
                        x0 = x - w if x >= w else x
                        x1 = x + w if x <= self.max_x - w else x
                        y0 = y - w if y >= w else y
                        y1 = y + w if y <= self.max_y - w else y
                        img[x0:x1, y0:y1] = (0, 200, 200) if net.get('cross') else [0, 200, 0]
                        # if x <= 0 or x > self.max_x or y <= 0 or y > self.max_y:
                        #     logging.info(f'超出范围 {x} {y} 网络ID {net["index"]}')

            w = pin_display_weight = 1
            for pin in self.pins:
                if layer in pin['layers']:
                    x, y = pin['x'], pin['y']
                    img[x - w:x + w, y - w:y + w] = (0, 0, 200)

        if base_img is not None:
            for i in [0, 2]:
                img[:, :, i] = base_img[:, :] * 50 + img[:, :, i]

        mat270 = np.rot90(img, 1)
        if save_path is None:
            save_path = 'img_layer{layer}.jpg'
        cv2.imwrite(save_path, mat270)
        logging.info(f'画图结束 保存为{save_path}')
        # self.net_display_init()
        # for i in range(len(self.steiner_nets)):
        #     self.net_display(i)

    def net_display_init(self):
        self.net_img = np.zeros(((self.max_x + 1), (self.max_y + 1), 3), dtype=np.uint8)
        self.net_img[np.where(self.obstacle[0, :, :] != -1)] = (120, 120, 120)
        w = pin_display_weight = 2
        for pin in self.pins:
            x, y = pin['x'], pin['y']
            self.net_img[x - w:x + w, y - w:y + w] = (0, 0, 255)

    def net_display(self, net_id):
        img = self.net_img.copy()

        w = net_display_weight = 1
        net = self.steiner_nets[net_id]
        foundPath = net.get('path')
        if foundPath:
            path = key_path2path(foundPath)
            for x, y in path:
                x0 = x - w if x >= w else x
                x1 = x + w if x <= self.max_x - w else x
                y0 = y - w if y >= w else y
                y1 = y + w if y <= self.max_y - w else y
                img[x0:x1, y0:y1] = (0, 255, 255) if net.get('cross') else [0, 255, 0]
                # if x <= 0 or x > self.max_x or y <= 0 or y > self.max_y:
                #     logging.info(f'超出范围 {x} {y} 网络ID {net["index"]}')

        mat270 = np.rot90(img, 1)
        file_name = f'img/net{net_id}.png'
        cv2.imwrite(file_name, mat270)
        logging.info(f'画图结束 保存为{file_name}')

    def get_solver(self, **kwargs):
        return MultiLayerAstar(**kwargs)

    def net_route(self, net_id, **kwargs):
        save_path_flag = False
        test_recommand_area = False
        net = self.steiner_nets[net_id]
        old_index = net['old_index']
        pin0, pin1 = net['pins'][0], net['pins'][1]
        if self.pins[pin0].get('hot_value', 0) < self.pins[pin1].get('hot_value', 0):
            pin0, pin1 = pin1, pin0
        ori_position = (self.pins[pin0]['layers'][0], self.pins[pin0]['x'], self.pins[pin0]['y'])
        target_pos = (self.pins[pin1]['layers'][0], self.pins[pin1]['x'], self.pins[pin1]['y'])
        logging.info(f'开始布线 id:{net_id}/{len(self.steiner_nets)} {ori_position} to {target_pos}')

        recommend_area = self.generate_recommend_area(net_id, pass_small_net=True) if self.model else None
        # A*算法
        p = None
        if save_path_flag and recommend_area is not None:
            with open(f'data/net_path/net{net_id}_recommend_area.pickle', 'wb') as f:
                pickle.dump(recommend_area, f)
        if test_recommand_area and recommend_area is not None:
            _solver = self.get_solver(width=self.max_x, height=self.max_y, solver=self, old_index=old_index,
                                      layer_max=self.layer_num, start_layers=self.pins[pin0]['layers'],
                                      end_layers=self.pins[pin1]['layers'],
                                      line_width=self.line_width, net_id=net_id,
                                      hx_multi_rate=self.hx_multi_rate,
                                      jps_search_rate=self.jps_search_rate,
                                      recommend_area=None,
                                      speed_test=False,
                                      obstacle=self.obstacle,
                                      clearance=self.clearance,
                                      via_radius=self.via_radius,
                                      **kwargs)
            p = mp.Process(target=_route, args=(None, _solver, ori_position, target_pos, net_id))
            p.start()
        _solver = self.get_solver(width=self.max_x, height=self.max_y, solver=self, old_index=old_index,
                                  layer_max=self.layer_num, start_layers=self.pins[pin0]['layers'],
                                  end_layers=self.pins[pin1]['layers'],
                                  line_width=self.line_width, net_id=net_id,
                                  hx_multi_rate=self.hx_multi_rate,
                                  jps_search_rate=self.jps_search_rate,
                                  recommend_area=recommend_area,
                                  speed_test=False,
                                  obstacle=self.obstacle,
                                  clearance=self.clearance,
                                  via_radius=self.via_radius,
                                  **kwargs)
        # TODO：跨层情况下，对平面方向不连续的方向是否可以裁剪
        _found_path, search_area = _route(recommend_area, _solver, ori_position, target_pos, net_id)
        self.search_areas[net_id] = search_area
        if p is not None:
            p.join()
        if _found_path:
            foundPath = list(_found_path)
            # save found path to file: "data/net_id_path.pickle"

            # 修改第一个点的层
            if len(foundPath) > 1:
                for layer in self.pins[pin0]['layers']:
                    if layer == foundPath[1][0]:
                        foundPath[0] = (layer, foundPath[0][1], foundPath[0][2])
        else:
            foundPath = None
            logging.warning(f'布线失败 id:{net_id}/{len(self.steiner_nets)}')
            self.save(f'data/fail_solver/net{net_id}.pickle')
        return foundPath

    def generate_pin_group(self, display=True):
        pin_groups = pin_group_divide(self.pins)
        for i in range(len(pin_groups)):
            data = {
                'index': i,
                'pins': pin_groups[i],
            }
            for pin_index in pin_groups[i]:
                pin_data = self.pins[pin_index]
                pin_data['pin_group'] = i
            pin_group = pin_groups[i]
            boundary = self.get_pin_group_boundary(pin_group)
            data['range'] = boundary
            self.pin_groups.append(data)
            logging.info(f'pin_group {i} {data}')
        if display:
            for pin_group in self.pin_groups:
                base_img = np.zeros(((self.max_x + 1), (self.max_y + 1)), dtype=np.uint8)
                for pin_index in pin_group['pins']:
                    pin = self.pins[pin_index]
                    x, y = pin['x'], pin['y']
                    base_img[x, y] = 1
                    # 在base_img上的x，y位置画一个圆, 半径为5，颜色为1，线宽为-1（表示填充）
                    cv2.circle(base_img, (y, x), 10, 2, -1)
                # 在base_img上画一个矩形 (minx, miny) (maxx, maxy)
                miny, minx, maxy, maxx = pin_group['range']['y_min'], pin_group['range']['x_min'], \
                    pin_group['range']['y_max'], pin_group['range']['x_max']
                cv2.rectangle(base_img, (miny, minx), (maxy, maxx), 1, 2)
                self.display(base_img, save_path=f'img/pin_group{pin_group["index"]}.jpg')

    def get_pin_group_boundary(self, pin_group):
        minx, miny = Infinite, Infinite
        maxx, maxy = 0, 0
        for pin_index in pin_group:
            pin_data = self.pins[pin_index]
            x, y = pin_data['x'], pin_data['y']
            padstack_range = self.padstack_range[pin_data['shape']]
            if x + padstack_range['x_min'] < minx:
                minx = x + padstack_range['x_min']
            if x + padstack_range['x_max'] > maxx:
                maxx = x + padstack_range['x_max']
            if y + padstack_range['y_min'] < miny:
                miny = y + padstack_range['y_min']
            if y + padstack_range['y_max'] > maxy:
                maxy = y + padstack_range['y_max']
        boundary = {
            'x_min': minx - 1,
            'x_max': maxx + 1,
            'y_min': miny - 1,
            'y_max': maxy + 1,
        }
        return boundary

    def net_refine(self, net_id, **kwargs):  # TODO: 低分辨率映射到高分辨率
        net = self.steiner_nets[net_id]
        available_area = np.zeros((self.layer_num, self.max_x + 1, self.max_y + 1), dtype=np.uint8)
        # 生成可用区域
        old_resolution, target_resolution = net.get('resolution'), self.resolution
        if old_resolution is None:
            # logging.warning(f'网络{net_id}没有分辨率信息')
            return
        if old_resolution <= target_resolution:
            logging.warning(f'网络{net_id}分辨率过低 old:{old_resolution} target:{target_resolution}')
            return
        if not net.get('path'):
            logging.warning(f'网络{net_id}没有路径信息')
            return
        logging.info(f'开始优化 id:{net_id}')
        multi_rate = int(old_resolution / target_resolution)
        path = key_path2path(net['path'])
        for layer, x, y in path:
            new_x = x * multi_rate
            new_y = y * multi_rate
            available_area[layer, new_x:new_x + multi_rate, new_y:new_y + multi_rate] = 1

        old_index = net['old_index']
        pin0, pin1 = net['pins'][0], net['pins'][1]

        ori_position = (self.pins[pin0]['layers'][0], self.pins[pin0]['x'], self.pins[pin0]['y'])
        target_pos = (self.pins[pin1]['layers'][0], self.pins[pin1]['x'], self.pins[pin1]['y'])

        # A*算法
        _solver = MultiLayerAstar(width=self.max_x, height=self.max_y, solver=self, old_index=old_index,
                                  layer_max=self.layer_num, start_layers=self.pins[pin0]['layers'],
                                  end_layers=self.pins[pin1]['layers'], available_range=available_area,
                                  cross_punish=0, **kwargs)
        _found_path = _solver.astar(ori_position, target_pos)
        foundPath = list(_found_path) if _found_path else None
        if foundPath:
            net['path'] = path2key_path(foundPath)
            net['resolution'] = target_resolution
        return foundPath

    def steiner_tree_net_divide(self):
        logging.info('开始切分net')
        self.steiner_nets.clear()
        for net in self.problem.nets:
            if len(net['pins']) <= 2:
                net['old_index'] = net['index']
                self.steiner_nets.append(copy.deepcopy(net))
                continue
            ori_pins = net['pins'].copy()
            index = 0
            new_pins = []
            while ori_pins:
                pin = ori_pins.pop()
                min_distance = np.inf
                target_pin = None
                for new_pin in new_pins:
                    distance = np.abs(self.problem.pins[new_pin]['x_int'] - self.problem.pins[pin]['x_int']) + \
                               np.abs(self.problem.pins[new_pin]['y_int'] - self.problem.pins[pin]['y_int'])
                    if distance < min_distance:
                        min_distance = distance
                        target_pin = new_pin
                new_pins.append(pin)
                if target_pin:
                    new_net = {'name': f"{net['name']}_{index}",
                               'pins': [pin, target_pin],
                               'old_index': net['index'],
                               }
                    self.steiner_nets.append(new_net)
                    index += 1
        for i in range(len(self.steiner_nets)):
            pin0 = self.steiner_nets[i]['pins'][0]
            pin1 = self.steiner_nets[i]['pins'][1]
            distance = np.abs(self.problem.pins[pin0]['x_int'] - self.problem.pins[pin1]['x_int']) + \
                       np.abs(self.problem.pins[pin0]['y_int'] - self.problem.pins[pin1]['y_int'])
            self.steiner_nets[i]['distance'] = distance
        self.steiner_nets.sort(key=lambda d: d['distance'])
        logging.info('切分net完成')

    def net_assemble_check(self):
        # 计算路线相似度
        logging.info('开始计算路线相似度')
        for i in range(len(self.steiner_nets)):
            for j in range(i + 1, len(self.steiner_nets)):
                if i == j:
                    continue
                assemble_score = 0
                net1_points = []
                net2_points = []
                net1 = self.steiner_nets[i]
                net2 = self.steiner_nets[j]
                for pin in net1['pins']:
                    net1_points.append((self.pins[pin]['x'], self.pins[pin]['y']))
                for pin in net2['pins']:
                    net2_points.append((self.pins[pin]['x'], self.pins[pin]['y']))
                # 计算欧式距离
                if math.hypot(net1_points[0][0] - net2_points[0][0], net1_points[0][1] - net2_points[0][1]) > \
                        math.hypot(net1_points[0][0] - net2_points[1][0], net1_points[0][1] - net2_points[1][1]):
                    net2_points.reverse()
                l1 = math.hypot(net1_points[0][0] - net2_points[0][0],
                                net1_points[0][1] - net2_points[0][1])
                l2 = math.hypot(net1_points[1][0] - net2_points[1][0],
                                net1_points[1][1] - net2_points[1][1])
                assemble_score += max(l1, l2)
                assemble_score = assemble_score / np.mean(
                    [math.hypot(net1_points[0][0] - net1_points[1][0], net1_points[0][1] - net1_points[1][1]),
                     math.hypot(net2_points[0][0] - net2_points[1][0], net2_points[0][1] - net2_points[1][1])])
                self.net_assemble_flag[i][j] = self.net_assemble_flag[j][i] = assemble_score

    def cross_point_check(self, net_id1, net_id2):
        net1 = self.steiner_nets[net_id1]
        net2 = self.steiner_nets[net_id2]
        path1 = key_path2path(net1['path'])
        path2 = key_path2path(net2['path'])
        for p1 in path1:
            for p2 in path2:
                if p1 == p2:
                    print(f'{net_id1} and {net_id2} cross at {p1}')
                    return True
        return False

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self

    def generate_recommend_area(self, net_id, pass_small_net=False):
        display_flag = False
        # 检测img下是否有net_id文件夹，没有则创建
        WIDTH = 640  # 模型需要的边长
        net = self.steiner_nets[net_id]
        old_index = net['old_index']
        pin0, pin1 = net['pins'][0], net['pins'][1]
        ori_position = (self.pins[pin0]['layers'][0], self.pins[pin0]['x'], self.pins[pin0]['y'])
        target_pos = (self.pins[pin1]['layers'][0], self.pins[pin1]['x'], self.pins[pin1]['y'])
        # 如果起始点和终止点欧式距离小于0.1WIDTH则返回None
        if pass_small_net and math.hypot(ori_position[1] - target_pos[1],
                                         ori_position[2] - target_pos[2]) < 0.1 * WIDTH:
            return None
        # 如果起始点和终止点横或纵距离大于384则返回None
        if abs(ori_position[1] - target_pos[1]) > WIDTH or abs(ori_position[2] - target_pos[2]) > WIDTH:
            return None
        if display_flag and not os.path.exists(f'img/{net_id}'):
            os.mkdir(f'img/{net_id}')
        width = self.max_x + 1 if self.max_x > WIDTH else WIDTH
        height = self.max_y + 1 if self.max_y > WIDTH else WIDTH
        feature_map = np.ones((self.layer_num + 1, width, height), dtype=float)
        obs_feature_map = feature_map[1:]
        obs_feature_map[np.where(self.obstacle == -1)] = 0
        obs_feature_map[np.where(self.obstacle == old_index)] = 0
        feature_map[0] = point_feature_generate(feature_map[0], ori_position, target_pos)
        obs_feature_map_generate(-1, feature_map, self.steiner_nets, add=1, old_index=old_index)
        if display_flag:
            for i in range(0, self.layer_num + 1):
                self.display(feature_map[i, :self.max_x + 1, :self.max_y + 1] * (3 if i == 0 else 1),
                             save_path=f'img/{net_id}/{i}.png')
        # 起始点和终止点的中点
        mid_point = ((ori_position[1] + target_pos[1]) // 2, (ori_position[2] + target_pos[2]) // 2)
        # 如果长或者宽大于384，以中点为中心，将feature_map的长或者宽裁剪到384
        x_min = y_min = 0
        x_max = y_max = WIDTH
        if width > WIDTH:
            x_min = mid_point[0] - WIDTH // 2
            x_max = mid_point[0] + WIDTH // 2
            if x_min < 0:
                x_min = 0
                x_max = WIDTH
            elif x_max >= width:
                x_min = width - WIDTH - 1
                x_max = width - 1
            feature_map = feature_map[:, x_min:x_max, :]
        if height > WIDTH:
            y_min = mid_point[1] - WIDTH // 2
            y_max = mid_point[1] + WIDTH // 2
            if y_min < 0:
                y_min = 0
                y_max = WIDTH
            elif y_max >= height:
                y_min = height - WIDTH - 1
                y_max = height - 1
            feature_map = feature_map[:, :, y_min:y_max]
        assert feature_map.shape == (self.layer_num + 1, WIDTH, WIDTH)
        # 将feature_map送入模型进行预测
        feature_map = torch.from_numpy(feature_map).unsqueeze(0).float()
        feature_map = feature_map.to(self.device)
        with torch.no_grad():
            s = time.time()
            pred = self.model(feature_map)
            logging.info(f'generate pred for net {net_id} cost {time.time() - s}s')
        pred = pred.squeeze(0).cpu().numpy()
        # 将预测结果通过sigmoid映射到0-1
        pred = 1 / (1 + np.exp(-pred))
        # pred = (pred - pred.min()) / (pred.max() - pred.min()) if pred.max() - pred.min() > 0 else pred
        # 将结果映射到原坐标系
        recommend_area = np.zeros((self.layer_num, self.max_x + 1, self.max_y + 1), dtype=float)
        recommend_area[:, x_min:x_max, y_min:y_max] = pred[:, :self.max_x + 1, :self.max_y + 1]
        if display_flag:
            for i in range(self.layer_num):
                self.display(recommend_area[i], layers=[i], save_path=f'img/{net_id}/pred_{i}.png')

        return recommend_area
