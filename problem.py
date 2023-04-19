import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import cv2


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
