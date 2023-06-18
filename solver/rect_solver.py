import random
import time

import cv2
import numpy as np

from solver.base_solver import BaseSolver
import logging

from solver.graph_solver.dijkstra_solver import DijkstraSolver
from solver.rect_detail_solver import RectDetailSolver
from utils.area_extract import area_extract
from utils.pin_group_divide import pin_group_divide
from utils.show_graph import show_graph


class RectSolver(BaseSolver):
    name = 'RECT_solver'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rect_crowded_rank = None
        self.rects, self.rect_flag = None, None

    def resolution_change(self, resolution=1):
        super().resolution_change(resolution)
        self.generate_rect_area()
        ...

    def solve(self):
        self.resolution_change(2)
        self.solve_graph()

    def rect_detail_route_generate(self):
        logging.info('开始生成rect的布线任务')
        for net_id in range(len(self.steiner_nets)):
            path = self.steiner_nets[net_id].get('rect_path')
            if not path:
                continue
            pin0_id = self.steiner_nets[net_id]['pins'][0]
            pin1_id = self.steiner_nets[net_id]['pins'][1]
            start_point = self.pins[pin0_id]['x'], self.pins[pin0_id]['y']
            start_range = [start_point[0], start_point[0], start_point[1], start_point[1]]
            start_layers = self.pins[pin0_id]['layers']
            end_point = self.pins[pin1_id]['x'], self.pins[pin1_id]['y']
            end_range = [end_point[0], end_point[0], end_point[1], end_point[1]]
            end_layers = self.pins[pin1_id]['layers']
            if len(path) == 1:
                data = {
                    'net_id': net_id,
                    'start_range': start_range,
                    'start_layers': start_layers,
                    'start_rect': None,
                    'end_range': end_range,
                    'end_layers': end_layers,
                    'end_rect': None,
                    'old_index': self.steiner_nets[net_id]['old_index'],
                }
                rect = self.rects[path[0]]
                if rect.get('route_net'):
                    rect['route_net'].append(data)
                else:
                    rect['route_net'] = [data]
            else:
                for i in range(len(path)):
                    if i < len(path) - 1:
                        er = self.rects[path[i]]['connect_range'].get(path[i + 1])
                        if er is None:
                            er = self.rects[path[i + 1]]['connect_range'][path[i]]
                        data = {
                            'net_id': net_id,
                            'start_range': start_range,
                            'start_layers': start_layers,
                            'start_rect': None if i == 0 else path[i - 1],
                            'end_range': er,
                            'end_layers': None,
                            'end_rect': path[i + 1],
                            'old_index': self.steiner_nets[net_id]['old_index'],
                        }
                        start_range = data['end_range']
                        start_layers = data['end_layers']
                    else:
                        data = {
                            'net_id': net_id,
                            'start_range': start_range,
                            'start_layers': start_layers,
                            'start_rect': path[i - 1],
                            'end_range': end_range,
                            'end_layers': end_layers,
                            'end_rect': None,
                            'old_index': self.steiner_nets[net_id]['old_index'],
                        }
                    rect = self.rects[path[i]]
                    if rect.get('route_net'):
                        rect['route_net'].append(data)
                    else:
                        rect['route_net'] = [data]

    def rect_crowded_sort(self):
        logging.info('开始对rect进行拥挤度排序')
        for i in range(len(self.rects)):
            rect = self.rects[i]
            x0, x1, y0, y1 = rect['x0'], rect['x1'], rect['y0'], rect['y1']
            blank_area = np.sum(self.obstacle[:, x0:x1 + 1, y0:y1 + 1] == -1)
            crowded = -blank_area ** 0.5
            if rect.get('route_net'):
                crowded += len(rect['route_net']) * (self.line_width + self.clearance)
            rect['crowded'] = crowded
        self.rect_crowded_rank = list(range(len(self.rects)))
        self.rect_crowded_rank.sort(key=lambda x: self.rects[x]['crowded'], reverse=True)

    def generate_rect_graph(self):
        logging.info('generate_rect_graph')
        width, height = self.rect_flag.shape

        for i in range(len(self.rects)):
            rect = self.rects[i]
            x0, x1, y0, y1 = rect['x0'], rect['x1'], rect['y0'], rect['y1']
            pos = [x0, x1, y0, y1]
            directions = [-1, 1, -1, 1]
            connects = {}
            connect_directions = [[], [], [], []]
            connect_range = {}
            for d in range(4):
                test_xy = pos[d]
                if d < 2:
                    test_flags = np.zeros((y1 - y0 + 1), dtype=int) - 2
                else:
                    test_flags = np.zeros((x1 - x0 + 1), dtype=int) - 2
                while np.logical_or((test_flags == -2), (test_flags == i)).any():
                    test_xy += directions[d]
                    poss = np.where(np.logical_or((test_flags == -2), (test_flags == i)))
                    if d < 2:
                        if test_xy < 0 or test_xy >= width:
                            break
                        test_flags[poss] = self.rect_flag[test_xy, pos[2]:pos[3] + 1][poss]
                    else:
                        if test_xy < 0 or test_xy >= height:
                            break
                        test_flags[poss] = self.rect_flag[pos[0]:pos[1] + 1, test_xy][poss]
                # 统计test_flags中非-2，-1的元素的个数
                connect_element_dict = {}
                for ii in range(len(test_flags)):
                    element = test_flags[ii]
                    if d < 2:
                        ele_x, ele_y = pos[d], pos[2] + ii
                    else:
                        ele_x, ele_y = pos[0] + ii, pos[d]
                    if element >= 0:
                        if element in connect_element_dict:
                            connect_element_dict[element] += 1
                        else:
                            connect_element_dict[element] = 1
                        if element not in connect_directions[d]:
                            connect_directions[d].append(element)
                        if element not in connect_range:
                            connect_range[element] = [ele_x, ele_x, ele_y, ele_y]
                        else:
                            if d > 1:
                                connect_range[element][0] = min(connect_range[element][0], ele_x)
                                connect_range[element][1] = max(connect_range[element][1], ele_x)
                            else:
                                connect_range[element][2] = min(connect_range[element][2], ele_y)
                                connect_range[element][3] = max(connect_range[element][3], ele_y)
                            # connect range是一个矩形，在rect相互重叠的情况下矩形会出现面积大于0的情况
                connects.update(connect_element_dict)
            self.rects[i]['connects'] = connects
            self.rects[i]['connect_directions'] = connect_directions
            self.rects[i]['connect_range'] = connect_range
            # 给rect中的pin添加rect_id属性
            if rect.get('pin_group'):
                pin_group = rect['pin_group']
                for pin in pin_group:
                    self.pins[pin]['rect_id'] = i
            rect['net'] = {}
        # 给rect添加相关的net属性
        for i in range(len(self.steiner_nets)):
            net = self.steiner_nets[i]
            rect_ids = []
            for pin in net['pins']:
                if 'rect_id' in self.pins[pin]:
                    rect_id = self.pins[pin]['rect_id']
                    if rect_id not in rect_ids:
                        rect_ids.append(rect_id)
            if len(rect_ids) > 1:
                assert len(rect_ids) == 2, 'len(rect_ids) > 2'
                rect_id1 = rect_ids[0]
                rect_id2 = rect_ids[1]
                if not self.rects[rect_id1]['net'].get(rect_id2):
                    self.rects[rect_id1]['net'][rect_id2] = []
                self.rects[rect_id1]['net'][rect_id2].append(i)
                if not self.rects[rect_id2]['net'].get(rect_id1):
                    self.rects[rect_id2]['net'][rect_id1] = []
                self.rects[rect_id2]['net'][rect_id1].append(i)
        edges = {}
        for i in range(len(self.rects)):
            rect = self.rects[i]
            for target_rect, limit in rect['connects'].items():
                if i < target_rect:
                    key = (i, target_rect)
                else:
                    key = (target_rect, i)
                if key not in edges:
                    edges[key] = limit

        show_graph(self.rects, edges)
        self.edges = edges
        self.net_path_data = {
            'edge_net_num': {},
            'rect_net_num': {},
        }
        ...

    def solve_graph(self):
        # generate net_path_data
        logging.info('start solve graph')
        s = time.time()
        for rect in self.rects:
            if rect.get('net'):
                num = 0
                for k, v in rect['net'].items():
                    num += len(v)
                self.net_path_data['rect_net_num'][rect['index']] = num / 2
            else:
                self.net_path_data['rect_net_num'][rect['index']] = 0

        no_change_num = 0
        search_num = 0
        net_ids = []
        while no_change_num < len(self.steiner_nets) and search_num < len(self.steiner_nets) * 5:
            search_num += 1
            if len(net_ids) == 0:
                net_ids = list(range(len(self.steiner_nets)))
            i = random.choice(net_ids)
            net_ids.remove(i)
            net = self.steiner_nets[i]
            if len(net['pins']) > 1:
                solver = DijkstraSolver(self.rects, self.edges, self.net_path_data, clearance=self.clearance,
                                        line_width=self.line_width)
                old_path = None
                if net.get('rect_path'):
                    self.update_net_path_data(net['rect_path'], add=-1)
                    old_path = net['rect_path']
                    net['rect_path'] = None
                rect_id1 = self.pins[net['pins'][0]]['rect_id']
                rect_id2 = self.pins[net['pins'][1]]['rect_id']
                start_exact_pos = (self.pins[net['pins'][0]]['x'], self.pins[net['pins'][0]]['y'])
                end_exact_pos = (self.pins[net['pins'][1]]['x'], self.pins[net['pins'][1]]['y'])
                path = solver.solve(rect_id1, rect_id2, start_exact_pos, end_exact_pos)
                path = list(path)
                logging.debug(f'net {i} path: {path}')
                if old_path and path == old_path:
                    no_change_num += 1
                else:
                    no_change_num = 0
                self.update_net_path_data(path)
                net['rect_path'] = path
        logging.info(
            f'finish graph solve use time:{time.time() - s:.2f}, no_change_num: {no_change_num}, search_num: {search_num}')
        self.get_rect_image('img/graph.png', self.rects, self.net_path_data['edge_net_num'], dispaly_net=True)
        ...

    def update_net_path_data(self, path, add=1):  # todo: 增加对线长度的考虑
        for i in range(len(path)):
            self.net_path_data['rect_net_num'][path[i]] = self.net_path_data['rect_net_num'].get(path[i], 0) + add
            if i > 0:
                if path[i] < path[i - 1]:
                    key = (path[i], path[i - 1])
                else:
                    key = (path[i - 1], path[i])
                self.net_path_data['edge_net_num'][key] = self.net_path_data['edge_net_num'].get(key, 0) + add

    def rect_route(self, rect_id):
        # 画个示意图
        x0, y0, x1, y1 = self.rects[rect_id]['x0'], self.rects[rect_id]['y0'], self.rects[rect_id]['x1'], \
            self.rects[rect_id]['y1']
        _solver = RectDetailSolver(rect_id=rect_id, rects=self.rects, clearance=self.clearance,
                                   line_width=self.line_width, obstacle=self.obstacle[:, x0:x1 + 1, y0:y1 + 1],
                                   layer_num=self.layer_num,
                                   pins=self.pins, via_radius=self.via_radius)
        _solver.solve()
        ...

    def generate_rect_area(self):
        logging.info('generate_rect_area')
        debug = True
        pin_groups = pin_group_divide(self.pins)
        obstacle_input = self.get_multilayer_obstacle()
        img = None
        if debug:
            img = obstacle_input.copy().astype(np.uint8) * 255
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img0 = img.copy()
            for pin in self.pins:
                cv2.circle(img0, (pin['y'], pin['x']), 7, (255, 0, 255))
        pin_group_rects = []
        for pin_group in pin_groups:
            boundary = self.get_pin_group_boundary(pin_group)
            x0 = boundary['x_min']
            x1 = boundary['x_max']
            y0 = boundary['y_min']
            y1 = boundary['y_max']
            rect_data = {
                'x0': x0,
                'x1': x1,
                'y0': y0,
                'y1': y1,
                'pin_group': pin_group,
                'width': x1 - x0,
                'height': y1 - y0,
                'width_line_limit': (x1 - x0) // (self.line_width + self.clearance),
                'height_line_limit': (y1 - y0) // (self.line_width + self.clearance),
            }
            pin_group_rects.append(rect_data)
            obstacle_input[x0:x1, y0:y1] = True
            if debug:
                cv2.rectangle(img, (y0, x0), (y1, x1), (0, 255, 0), 1)
                img1 = img0.copy()
                cv2.rectangle(img1, (y0, x0), (y1, x1), (0, 255, 0), 5)
                for pin_index in pin_group:
                    x, y = self.pins[pin_index]['x'], self.pins[pin_index]['y']
                    cv2.circle(img1, (y, x), 5, (0, 0, 255), -1)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_path = 'img/pin_group/area_extract_{}.png'.format(pin_group[0])
                cv2.imwrite(img_path, img1)
        img_path = 'img/area_extract.png'
        rects, rect_flag = area_extract(obstacle_input, self.line_width, self.clearance, path=img_path)

        for rect_data in pin_group_rects:
            rects.append(rect_data)
            x0, x1, y0, y1 = rect_data['x0'], rect_data['x1'], rect_data['y0'], rect_data['y1']
            rect_flag[x0:x1, y0:y1] = len(rects) - 1
            rect_data['index'] = len(rects) - 1
        for i in range(len(rects)):
            rect = rects[i]
            # 计算空白面积
            rect['blank_area'] = np.sum(self.obstacle[:, rect['x0']:rect['x1'] + 1, rect['y0']:rect['y1'] + 1] == -1)
            if rect.get('pin_group', None):
                rect['blank_area'] = 0
        img_path = 'img/area_extract_rect.png'
        if debug:
            self.get_rect_image(img_path, rects)

        self.rects = rects
        self.rect_flag = rect_flag
        self.generate_rect_graph()
        ...

    def get_multilayer_obstacle(self):
        obstacle_input = self.obstacle[0] != -1
        for layer in range(1, self.layer_num):
            obstacle_input = np.logical_or(obstacle_input, self.obstacle[layer] != -1)
        return obstacle_input

    def get_rect_image(self, img_path, rects, edge_net_num_dict=None, dispaly_net=False, highlight_rect=None):
        obstacle_input = self.get_multilayer_obstacle()
        img = obstacle_input.copy().astype(np.uint8) * 255
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if edge_net_num_dict is not None:
            for k, v in edge_net_num_dict.items():
                rect_id1, rect_id2 = k
                weight = v
                rect1 = rects[rect_id1]
                rect2 = rects[rect_id2]
                if rect_id2 not in list(rect1['connects'].keys()):
                    # 交换rect_id1和rect_id2
                    rect_id1, rect_id2 = rect_id2, rect_id1
                    rect1 = rects[rect_id1]
                    rect2 = rects[rect_id2]
                direction = -1
                for d in range(4):
                    if rect_id2 in rect1['connect_directions'][d]:
                        direction = d
                        break
                if direction == -1:
                    raise Exception('rect_id2 not in rect1')
                if direction < 2:
                    x = int((rect1['x0'], rect1['x1'])[direction])
                    y0 = max(rect1['y0'], rect2['y0'])
                    y1 = min(rect1['y1'], rect2['y1'])
                    y = int((y0 + y1) / 2)
                else:
                    y = int((rect1['y0'], rect1['y1'])[direction - 2])
                    x0 = max(rect1['x0'], rect2['x0'])
                    x1 = min(rect1['x1'], rect2['x1'])
                    x = int((x0 + x1) / 2)
                cv2.putText(img, str(weight), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if dispaly_net:
            for i in range(len(self.steiner_nets)):
                net = self.steiner_nets[i]
                if net.get('rect_path'):
                    path = net['rect_path']
                    start_pin = self.pins[net['pins'][0]]
                    end_pin = self.pins[net['pins'][1]]
                    mid_path = []
                    if len(path) > 2:
                        for j in range(0, len(path) - 1):
                            target_rect = self.rects[path[j]]['connect_range'].get(path[j + 1])
                            if target_rect is None:
                                target_rect = self.rects[path[j + 1]]['connect_range'].get(path[j])
                            x = (target_rect[0] + target_rect[1]) // 2
                            y = (target_rect[2] + target_rect[3]) // 2
                            mid_path.append((x, y))
                    display_path = [(start_pin['x'], start_pin['y']), *mid_path, (end_pin['x'], end_pin['y'])]
                    for j in range(len(display_path)):
                        display_path[j] = (display_path[j][1], display_path[j][0])
                    for j in range(len(display_path) - 1):
                        cv2.line(img, display_path[j], display_path[j + 1], (255, 0, 0), 1)

        for i in range(len(rects)):
            rect = rects[i]
            rect_pos = rect['x0'], rect['x1'], rect['y0'], rect['y1']
            cv2.rectangle(img, (rect_pos[2], rect_pos[0]), (rect_pos[3], rect_pos[1]), (0, 0, 255), 1)
            # write i in the center of rect
            cv2.putText(img, str(i), (int((rect_pos[2] + rect_pos[3]) / 2), int((rect_pos[0] + rect_pos[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_path, img)
