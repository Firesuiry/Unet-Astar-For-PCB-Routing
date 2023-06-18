import logging
import os.path
import uuid

import numpy as np

from problem import RandomProblem
from solver.jps_solver import JpsSolver
import cv2
from heapq import heappush, heappop
import json
import pickle

from solver.solver1 import Solver1
from utils.np_encoder import NpEncoder
import multiprocessing as mp
from network.utils.obs_feature_map_generate import obs_feature_map_generate

'''
1. 生成随机问题
2. 对问题进行求解
3. 随机选择一个net
4. 对net用lee算法进行求解，得到经过不同坐标点的最短距离，忽略距离大于（最优值+0.5线长+1）的点
5. 符合条件的点标1 不符合条件的点标0
'''
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]


def imwrite(name, img):
    # 逆时针旋转90度
    img = np.rot90(img, 3)
    # 左右翻转
    img = np.fliplr(img)
    if '/' in name or '\\' in name:
        cv2.imwrite(name, img)
    else:
        cv2.imwrite('img/' + name, img)


def generate_sample(w_max, w_min, h_max, h_min, l, pin_density, obs_density, sample_num, debug=True):
    if debug:
        assert sample_num == 1
    if debug:
        for i in range(sample_num):
            generate_1_sample(w_max, w_min, h_max, h_min, l, pin_density, obs_density, debug=debug)
    else:
        # multiprocess run
        pool = mp.Pool(processes=8)
        for i in range(sample_num):
            pool.apply_async(generate_1_sample, args=(w_max, w_min, h_max, h_min, l, pin_density, obs_density),
                             kwds={'debug': debug})
        pool.close()
        pool.join()


def generate_1_sample(w_max, w_min, h_max, h_min, l, pin_density, obs_density, debug=True, load_old=False):
    save_path = 'network/dataset/' + os.urandom(16).hex() + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists('problem.pkl') and load_old:
        problem = RandomProblem.load('problem.pkl')
        solver = Solver1.load('solver.pkl')
    else:
        problem = RandomProblem(w_max, w_min, h_max, h_min, l, pin_density, obs_density)
        if debug: problem.save('problem.pkl')
        solver = Solver1(problem, speed_test=True, hx_multi_rate=1.35, jps_search_rate=0.2)
        solver.resolution_solve(8, 100)
        if debug: solver.save('solver.pkl')
    nets = solver.steiner_nets
    # delete_net_id = np.random.randint(0, len(nets))
    h = problem.max_y
    w = problem.max_x
    generate_data(h, l, nets, problem, w, save_path, solver.obstacle, line_width=solver.line_width,
                  clearance=solver.clearance, via_radius=solver.via_radius)


def generate_data(h, l, nets, problem, w, save_path, obstacle, line_width, clearance, via_radius):
    # save problem
    problem.save(save_path + 'problem.pkl')
    # save nets
    with open(save_path + 'nets.json', 'w') as f:
        json.dump(nets, f, cls=NpEncoder)
    for delete_net_id in range(len(nets)):
        if nets[delete_net_id].get('path') is None:
            continue
        # generate feature map
        feature_map = np.ones((l, w // 8 + 1, h // 8 + 1), dtype=np.uint8)
        feature_map[np.where(obstacle == -1)] = 0
        feature_map[np.where(obstacle == nets[delete_net_id]['old_index'])] = 0
        obs_feature_map_generate(delete_net_id, feature_map, nets)
        if False:
            for layer in range(l):
                imwrite('feature_map_' + str(layer) + '.png', feature_map[layer])
        # compress and save feature map
        np.save(save_path + 'feature_map_' + str(delete_net_id) + '.npy', feature_map)
        solver = LeeSolver2(feature_map, line_width, clearance, via_radius)
        start_pin = problem.pins[nets[delete_net_id]['pins'][0]]
        start = (0, start_pin['x'] // 8, start_pin['y'] // 8)
        end_pin = problem.pins[nets[delete_net_id]['pins'][1]]
        goal = (0, end_pin['x'] // 8, end_pin['y'] // 8)
        result, points_map = solver.solve(start, goal)
        # compress and save result
        result_save = result.astype(bool)
        np.save(save_path + 'result_' + str(delete_net_id) + '.npy', result_save)
        data = {
            'start': start,
            'goal': goal,
        }
        # save data by json
        with open(save_path + 'data_' + str(delete_net_id) + '.json', 'w') as f:
            json.dump(data, f)
        if False:
            display(goal, l, nets, result, start, save_path=str(delete_net_id) + '_')


def display(goal, l, nets, result, start, save_path=''):
    normal_result = result / np.max(result) * 255 if np.max(result) != 0 else result
    img = np.uint8(normal_result)
    for layer in range(l):
        imwrite(save_path + 'result_' + str(layer) + '.png', img[layer])
        new_img = cv2.cvtColor(img[layer], cv2.COLOR_GRAY2BGR)

        # 画出path
        for net_id in range(len(nets)):
            if not nets[net_id].__contains__('path'):
                continue
            path = nets[net_id]['path']
            for i in range(len(nets[net_id]['path'])):
                if path[i][0] == layer and i > 0:
                    cv2.line(new_img, (path[i - 1][2], path[i - 1][1]), (path[i][2], path[i][1]), (255, 0, 0),
                             1)
        # 画出start和goal
        cv2.circle(new_img, (start[2], start[1]), 2, (0, 0, 255), -1)
        cv2.circle(new_img, (goal[2], goal[1]), 2, (0, 255, 0), -1)

        imwrite(save_path + 'result_' + str(layer) + '_with_path.png', new_img)
        logging.info(save_path + 'result_' + str(layer) + '_with_path.png saved')


sqrt2 = 2 ** 0.5


class LeeSolver2:

    def __init__(self, feature_map, line_width, clearance, via_radius):
        self.pass_cache = {}
        self.feature_map = feature_map
        self.layer_num = feature_map.shape[0]
        self.width = feature_map.shape[1]
        self.height = feature_map.shape[2]
        self.point1_map = np.zeros_like(feature_map, dtype=float) - 1
        self.point2_map = np.zeros_like(feature_map, dtype=float) - 1
        self.points_map = [self.point1_map, self.point2_map]
        self.path_length_map = np.zeros_like(feature_map, dtype=float) - 1
        self.log_flag = False
        self.line_width = line_width
        self.clearance = clearance
        self.via_radius = via_radius
        ...

    def is_pass(self, node, vertical_move=False):
        key = (node, vertical_move)
        if key not in self.pass_cache:
            self.pass_cache[key] = self._is_pass(node, vertical_move)
        return self.pass_cache[key]

    def _is_pass(self, node, vertical_move=False):
        layer, x, y = node
        if x < 0 or x >= self.width - 1 or y < 0 or y >= self.height - 1:
            return False
        if not vertical_move:
            line_clearance = self.clearance + self.line_width
            return not self.feature_map[layer, x - line_clearance:x + line_clearance + 1,
                       y - line_clearance:y + line_clearance + 1].any()
        else:
            line_clearance = self.via_radius + self.clearance
            return not self.feature_map[:, x - line_clearance:x + line_clearance + 1,
                       y - line_clearance:y + line_clearance + 1].any()

    def distance(self, point1, point2):
        if point1[0] != point2[0]:
            # assert point1[1] == point2[1] and point1[2] == point2[2]
            return 5
        if point1[1] != point2[1] and point1[2] != point2[2]:
            # assert point1[0] == point2[0]
            return sqrt2
        return 1

    def neighbors(self, point):
        layer, x, y = point
        _neighbors = []
        neighbors = []
        for d in DIRECTIONS:
            new_point = (layer, x + d[0], y + d[1])
            if self.is_pass(new_point):
                _neighbors.append(new_point)
        for l in range(self.layer_num):
            if l == layer:
                continue
            if self.is_pass((l, x, y), vertical_move=True):
                _neighbors.append((l, x, y))
        for point in _neighbors:
            neighbors.append(point)
        return neighbors

    def back_propagate(self, start_point, wave_index, path_length):
        points = [{'point': start_point, 'path_length': path_length}]
        self.path_length_map[start_point] = path_length
        while bool(points):
            data = points.pop()
            bp_points = self._back_propagate(data['point'], wave_index, data['path_length'])
            points.extend(bp_points)
            # print(len(points))

    def _back_propagate(self, start_point, wave_index, path_length):
        if self.path_length_map[start_point] != -1 and self.path_length_map[start_point] <= path_length:
            return []
        neighbors = self.neighbors(start_point)
        bp_points = []
        for ng in neighbors:
            if self.points_map[wave_index][ng] == -1:
                continue
            tentative_path_length = self.path_length_map[start_point] + self.distance(start_point, ng)
            if self.path_length_map[ng] == -1 or tentative_path_length < self.path_length_map[ng]:
                self.path_length_map[ng] = tentative_path_length
                bp_points.append({'point': ng, 'path_length': tentative_path_length})
        # print(f"current: {start_point} bp_points: {[data['point'] for data in bp_points]}")
        return bp_points

    def display(self):
        for layer in range(self.layer_num):
            logging.info('result write to result_' + str(layer) + '.png')
            for wave_id in range(2):
                normal_result = self.points_map[wave_id][layer]
                normal_result[np.where(normal_result == -1)] = 0
                for i in range(10):
                    maxr = np.max(normal_result) * i / 10
                    normal_result[np.where(normal_result < maxr)] = 0
                    normal_result = (normal_result - maxr)
                    normal_result[np.where(normal_result < 0)] = 0
                    normal_result = normal_result / np.max(normal_result) * 255 if np.max(
                        normal_result) != 0 else normal_result
                    normal_result = np.uint8(normal_result)
                    imwrite('points_map_layer' + str(layer) + '_wave' + str(wave_id) + str(i) + '.png', normal_result)

    def solve(self, start, goal):
        points = [start, goal]
        direct_path_length = abs(start[1] - goal[1]) + abs(start[2] - goal[2])
        current_wave = 0
        openset = [[], []]
        for layer in range(self.layer_num):
            start_point = (layer, start[1], start[2])
            goal_point = (layer, goal[1], goal[2])
            self.points_map[0][start_point] = 0
            self.points_map[1][goal_point] = 0
            heappush(openset[0], (0, start_point))
            heappush(openset[1], (0, goal_point))

        self.points_map[0][start] = 0
        self.points_map[1][goal] = 0
        min_path_length = np.inf
        run_flag = True
        explore_flag = True
        while run_flag:
            for wave_index in range(2):
                if len(openset[wave_index]) == 0:
                    continue
                point_data = heappop(openset[wave_index])
                point = point_data[1]
                point_power = point_data[0]
                neighbors = self.neighbors(point)
                for ng in neighbors:
                    if not explore_flag:
                        if self.points_map[1 - wave_index][ng] == -1:
                            continue
                    tentative_distance = self.points_map[wave_index][point] + self.distance(point, ng)
                    if self.points_map[wave_index][ng] == -1 or \
                            tentative_distance < self.points_map[wave_index][ng]:
                        self.points_map[wave_index][ng] = tentative_distance
                    else:
                        continue
                    if self.points_map[1 - wave_index][ng] != -1:
                        # 双方的波都到达了这个点
                        # path_length = self.back_propagate(ng, wave_index)
                        path_length = self.points_map[wave_index][ng] + self.points_map[1 - wave_index][ng]
                        if path_length < self.path_length_map[ng] or self.path_length_map[ng] == -1:
                            self.path_length_map[ng] = path_length
                        # self.back_propagate(ng, wave_index, path_length)
                        if path_length < min_path_length:
                            if self.log_flag: logging.info('min path length update: ' + str(path_length))
                            min_path_length = path_length
                        if explore_flag and path_length > min_path_length:
                            explore_flag = False
                        if current_wave > min_path_length: run_flag = False
                    heappush(openset[wave_index], (self.points_map[wave_index][ng], ng))

            if len(openset[0]) == 0 and len(openset[1]) == 0:
                run_flag = False

        self.path_length_map[np.where(self.path_length_map == -1)] = 0
        self.path_length_map[np.where(self.path_length_map > min_path_length + 0.1 * direct_path_length + 1)] = 0
        return self.path_length_map, self.points_map


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    # w_max, w_min, h_max, h_min = 2000, 1000, 2000, 1000
    # w_max, w_min, h_max, h_min = 5000, 1000, 5000, 1000
    w_max, w_min, h_max, h_min = 1000, 500, 1000, 500
    l, pin_density, obs_density = 2, 0.22, 0.11
    generate_1_sample(w_max, w_min, h_max, h_min, l, pin_density, obs_density, debug=True, load_old=False)
