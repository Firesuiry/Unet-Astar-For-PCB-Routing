import logging
import math
import random

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pickle


class RandomProblem:

    def __init__(self, w_max, w_min, h_max, h_min, l, pin_density, obs_density):
        self.pending_pins = None
        if l == 0:
            return
        self.max_x = random.randint(w_min, w_max)
        self.max_y = random.randint(h_min, h_max)
        area = self.max_x * self.max_y / 10000
        pin_num = int(area * pin_density)
        obs_num = int(area / 9 * obs_density)
        logging.info('RandomProblem: max_x: %d, max_y: %d, area: %d, pin_num: %d, obs_num: %d' % (
            self.max_x, self.max_y, area, pin_num, obs_num))
        assert pin_density + obs_density <= 0.5
        self.layer_num = l
        self.min_x = 0
        self.min_y = 0

        self.grid = 2
        self.clearance = 1
        self.line_width = 1

        self.padstacks = None
        self.generate_padstack()

        self.pins = None
        self.generate_pins(pin_num, obs_num)

        self.nets = None
        self.generate_nets()

    def generate_nets(self):
        self.nets = []
        # divide pins into nets
        index = 0
        logging.info('RandomProblem: generate_nets: pin_num: %d' % len(self.pending_pins))
        while True:
            if len(self.pending_pins) < 2:
                break
            net = {
                'index': index,
                'pins': [],
                'name': 'net' + str(index),
                'net_id': index
            }
            pin_num = 2
            for i in range(pin_num):
                pin_id = random.choice(self.pending_pins)
                net['pins'].append(pin_id)
                self.pending_pins.remove(pin_id)
                self.pins[pin_id]['net_id'] = net['index']
            self.nets.append(net)
            index += 1

    def generate_pins(self, pin_num, obs_num=10):
        self.pins = []
        self.pending_pins = []
        generate_num = 0
        generate_pin_num = 0
        generate_obs_num = 0
        i = 0
        while len(self.pins) < pin_num + obs_num:
            generate_num += 1
            if generate_pin_num < pin_num:
                t = 'PIN'
            else:
                t = 'OBS'
            pin = {
                'id': str(i),
                'shape': random.choice(['CIRCLE', 'RECT']) if t == 'PIN' else 'OBS_CIRCLE',
                'x': random.randint(0, self.max_x),
                'y': random.randint(0, self.max_y),
                'index': str(i),
                'type': t,
            }
            pin['x_int'] = round(pin['x'])
            pin['y_int'] = round(pin['y'])
            counter_flag = False
            for _pin in self.pins:
                min_distance = 100
                if _pin['type'] == 'OBS': min_distance += 200
                if t == 'OBS': min_distance += 200
                if math.hypot(_pin['x_int'] - pin['x_int'], _pin['y_int'] - pin['y_int']) < min_distance:
                    counter_flag = True
                    break
            if not counter_flag:
                self.pins.append(pin)
                if t == 'PIN':
                    self.pending_pins.append(i)
                    generate_pin_num += 1
                else:
                    generate_obs_num += 1
                i += 1
            if generate_num > 1000:
                logging.warning('generate pins failed in 1000 times')
                break

    def generate_padstack(self):
        self.padstacks = {
            'CIRCLE': {},
            'RECT': {},
            'OBS_CIRCLE': {},
        }
        for i in range(1, 1 + self.layer_num):
            self.padstacks['CIRCLE'][F'{i}'] = {'shape': 'circle', 'detail': [80]}
            self.padstacks['RECT'][F'{i}'] = {'shape': 'polygon', 'detail': [0, -28, 24, 28, 24, 28, -24, -28, -24]}
            self.padstacks['OBS_CIRCLE'][F'{i}'] = {'shape': 'circle', 'detail': [80 * 3]}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self


class Problem:

    def __init__(self, resolver):

        boundary = resolver.boundary
        pins = resolver.pins
        connections = resolver.connections
        self.padstacks = resolver.padstacks
        self.min_x = round(boundary['min_x'])
        self.min_y = round(boundary['min_y'])
        self.max_x = round(boundary['max_x'])
        self.max_y = round(boundary['max_y'])

        self.grid = resolver.grid
        self.clearance = resolver.clearance
        self.line_width = resolver.line_width

        self.x_offset = 0
        self.y_offset = 0

        if self.min_x != 0:
            self.x_offset = -self.min_x
            self.min_x = 0
            self.max_x = self.max_x + self.x_offset

        if self.min_y != 0:
            self.y_offset = -self.min_y
            self.min_y = 0
            self.max_y = self.max_y + self.y_offset

        self.layer_num = len(boundary['layer'])

        self.board = np.zeros((self.max_x + 1, self.max_y + 1, self.layer_num), dtype=int)

        self.pins = list(pins.values())
        for pin in self.pins:
            pin['x_int'] += self.x_offset
            pin['y_int'] += self.y_offset
        self.pins_dict = {}
        for i in range(len(self.pins)):
            self.pins[i]['index'] = i
            self.pins_dict[self.pins[i]['id']] = self.pins[i]

        self.nets = []
        i = 0
        for k, v in connections.items():
            _pins = v
            _pins = [self.pins_dict[ii]['index'] for ii in _pins]
            d = {
                'name': k,
                'pins': _pins,
                'index': i
            }
            for pin in _pins:
                self.pins[pin]['net_id'] = i
            self.nets.append(d)
            i += 1
        pass

    def get_location(self, x, y):
        return tuple([(x + self.x_offset), (y + self.y_offset)])

    def show(self, layer_id):
        img = np.zeros(((self.max_x + 1), (self.max_y + 1), 3), dtype=np.uint8)

        for pin in self.pins:
            x, y = self.get_location(pin['x_int'], pin['y_int'])
            print(pin['x_int'], pin['y_int'], x, y)
            img[x - 5:x + 5, y - 5:y + 5] = (0, 0, 255)
            # cv2.circle(img, (y, x), 10, (0, 0, 255), 10)

        for net in self.nets:
            pass

        cv2.imwrite('img.jpg', img)
        # cv2.imshow('img', img)

        # plt.figure("Image")  # 图像窗口名称
        # plt.imshow(img)
        # plt.axis('on')  # 关掉坐标轴为 off
        # plt.title('image')  # 图像题目
        #
        # # 必须有这个，要不然无法显示
        # plt.show()
