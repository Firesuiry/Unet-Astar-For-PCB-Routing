import cv2
import numpy as np

from solver.astar.rect_detail_aster import RectDetailAstar
from utils.circle_generate import circle_generate, put_circle
from utils.pin_group_divide import pin_group_divide
import logging


class RectDetailSolver:

    def __init__(self, rect_id, clearance, line_width, obstacle, rects, layer_num, pins, via_radius):
        self.rect_id = rect_id
        self.clearance = clearance
        self.line_width = line_width
        self.obstacle = obstacle
        self.rects = rects
        self.layer_num = layer_num
        self.pins = pins
        self.via_radius = via_radius

        self.rect = self.rects[self.rect_id]
        self.net_num = 0 if self.rect.get('route_net') is None else len(self.rect['route_net'])
        self.net_flag = np.zeros((self.net_num, self.layer_num, self.rect['width'] + 1, self.rect['height'] + 1),
                                 dtype=bool)

        self.x0 = self.rect['x0']
        self.x1 = self.rect['x1']
        self.y0 = self.rect['y0']
        self.y1 = self.rect['y1']

    def solve_with_pin_inside(self):
        logging.info('solve_with_pin_inside rect_id: %d' % self.rect_id)
        rect = self.rects[self.rect_id]
        # divide the net into n parts according to their start_rect or end_rect
        net_dict = {}
        for net_id in range(len(rect['route_net'])):
            net = rect['route_net'][net_id]
            start_rect = net['start_rect']
            end_rect = net['end_rect']
            for a_rect in [start_rect, end_rect]:
                if a_rect is not None:
                    if a_rect not in net_dict:
                        net_dict[a_rect] = []
                    if net_id not in net_dict[a_rect]:
                        net_dict[a_rect].append(net_id)
        # create a list to store the rect_id in net_dict, the order is descending of the net_num
        net_dict_list = []
        for rect_id in net_dict:
            net_dict_list.append((rect_id, len(net_dict[rect_id])))
        net_dict_list.sort(key=lambda x: x[1], reverse=True)
        # solve the rect in net_dict_list one by one
        for rect_id, _ in net_dict_list:
            net_ids = net_dict[rect_id]
            # exchange the start_range and end_range if the end_rect is None and the start_rect is not None
            pins = []
            for net_id in net_ids:
                net = rect['route_net'][net_id]
                if net['end_rect'] is None and net['start_rect'] is not None:
                    net['start_rect'], net['end_rect'] = net['end_rect'], net['start_rect']
                    net['start_range'], net['end_range'] = net['end_range'], net['start_range']
                    net['start_layers'], net['end_layers'] = net['end_layers'], net['start_layers']
                assert net['start_rect'] is None
                assert net['start_range'][0] == net['start_range'][1] and net['start_range'][2] == net['start_range'][3]
                pin = {
                    'x': net['start_range'][0],
                    'y': net['start_range'][2],
                    'index': len(pins)
                }
                pins.append(pin)
            pin_groups = pin_group_divide(pins, max_distance=0.3)
            logging.info(f'target_rect_id:{rect_id} pin_groups: {[len(pg) for pg in pin_groups]}')
            # sort the pin_group according to the length of the pin_group
            pin_groups.sort(key=lambda x: len(x), reverse=True)
            # solve the pin_group one by one
            for i in range(len(pin_groups)):
                pin_group = pin_groups[i]
                if len(pin_group) < 2:
                    continue
                self.display_pins([pins[pin_id] for pin_id in pin_group],
                                  save_path=f'img/rect/rect_{self.rect_id}_tg{rect_id}_pin_group_{i}.png')
                # 获得pin_group的外接矩形范围
                x0 = min([pins[pin_id]['x'] for pin_id in pin_group])
                x1 = max([pins[pin_id]['x'] for pin_id in pin_group])
                y0 = min([pins[pin_id]['y'] for pin_id in pin_group])
                y1 = max([pins[pin_id]['y'] for pin_id in pin_group])
                group_start_range = [x0, x1, y0, y1]
                group_end_range = rect['end_range']
                # calculate the max distance between the start_range and the end_range
                x_distance, y_distance = 0, 0
                if x0 < group_end_range[0]: x_distance = max(x_distance, group_end_range[0] - x0)
                if x1 > group_end_range[1]: x_distance = max(x_distance, x1 - group_end_range[1])
                if y0 < group_end_range[2]: y_distance = max(y_distance, group_end_range[2] - y0)
                if y1 > group_end_range[3]: y_distance = max(y_distance, y1 - group_end_range[3])
                target_directions = [0, 0]
                if x0 > group_end_range[1]: target_directions[0] = -1
                if x1 < group_end_range[0]: target_directions[0] = 1
                if y0 > group_end_range[3]: target_directions[1] = -1
                if y1 < group_end_range[2]: target_directions[1] = 1
                # 获取pin_group中pin的数量 计算线宽度
                pin_num = len(pin_group)
                line_width = self.line_width * pin_num + self.clearance * (pin_num - 1)
                width = x1 - x0
                height = y1 - y0
                directions = []
                if width > line_width - self.line_width:
                    directions.extend([[0, -1], [0, 1]])
                if height > line_width - self.line_width:
                    directions.extend([[-1, 0], [1, 0]])
                if len(directions) > 0:
                    # pin_group有单方向可出的，尝试直接出
                    ...

            ...

    def display_pins(self, pins, save_path):
        img = (self.obstacle != -1).astype(np.uint8) * 255
        img = np.max(img, axis=0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pin in pins:
            x = pin['x']
            y = pin['y']
            cv2.circle(img, (y - self.y0, x - self.x0), 5, (0, 0, 255), -1)
        cv2.imwrite(save_path, img)

    def solve_without_pin(self):
        ...

    def solve(self):
        rect = self.rects[self.rect_id]
        x0, y0, x1, y1 = rect['x0'], rect['y0'], rect['x1'], rect['y1']
        img = (self.obstacle != -1).astype(np.uint8) * 255
        # for layer in range(self.layer_num):
        #     cv2.imwrite(f'img/rect/rect_{self.rect_id}_layer_{layer}.png', img[layer])
        route_net = rect.get('route_net')
        circle = circle_generate(self.line_width)
        if route_net is None:
            return None
        # 按照区域内有没有pin分为两种情况
        if rect.get('pin_group') is None:
            self.solve_without_pin()
        else:
            self.solve_with_pin_inside()
        return
        # 用直线直接进行布线
        for net_id in range(len(route_net)):
            print(f'route net_id:{net_id}/{len(route_net)}')
            net = route_net[net_id]
            old_index = net['old_index']
            start_range = net['start_range']
            end_range = net['end_range']
            start_layers = net['start_layers']
            end_layers = net['end_layers']
            # 生成obstacle
            obstacle = np.logical_and((self.obstacle != -1), (self.obstacle != old_index))
            for net_id2 in range(len(route_net)):
                if net_id2 != net_id:
                    obstacle = np.logical_or(obstacle, self.net_flag[net_id2])
            # 生成obstacle的图片并保存
            # for layer in range(self.layer_num):
            #     cv2.imwrite(f'img/rect/rect_{self.rect_id}_layer_{layer}_obstacle.png', obstacle[layer].astype(np.uint8) * 255)
            _solver = RectDetailAstar(width=rect['width'],
                                      height=rect['height'], old_index=old_index, start_layers=start_layers,
                                      end_layers=end_layers, layer_max=self.layer_num, obstacle=obstacle,
                                      available_range=(self.x0, self.x1, self.y0, self.y1),
                                      start_range=start_range, end_range=end_range, clearance=self.clearance,
                                      line_width=self.line_width, via_radius=self.via_radius)
            path, searched_area = _solver.astar()
            if path is None:
                print(f'布线失败：rect_id={self.rect_id}, net_id={net_id}')
            else:
                # 更新net_flag
                for point in path:
                    layer, x, y = point
                    hlw = self.line_width // 2
                    x0 = self.x0
                    y0 = self.y0
                    area = self.net_flag[net_id, layer]
                    put_circle(area, circle, hlw, x, x0, y, y0)
                # 生成布线图片并保存
                for layer in range(self.layer_num):
                    img0 = cv2.cvtColor(img[layer], cv2.COLOR_GRAY2BGR)
                    img0[searched_area[layer]] = img0[searched_area[layer]] * 0.5 + [127, 0, 0]
                    img0[self.net_flag[net_id][layer]] = (0, 0, 255)
                    cv2.imwrite(f'img/rect/rect_{self.rect_id}_net{net_id}_layer{layer}.png', img0)

        print(f'布线成功：rect_id={self.rect_id}')
        for layer in range(self.layer_num):
            img0 = cv2.cvtColor(img[layer], cv2.COLOR_GRAY2BGR)
            for net in route_net:
                img0[self.net_flag[net['old_index']][layer]] = (0, 0, 255)
            cv2.imwrite(f'img/rect/rect_{self.rect_id}_layer{layer}_route.png', img0)
