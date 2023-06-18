import numpy as np


def pin_group_divide(pins, max_distance=0.15):
    """
    将pin分组，对于组内任意一pin，存在另一个pin与其距离小于max_distance
    :param pins:
    :return:
    """
    std = calculate_std(pins)
    max_distance = std * max_distance
    pin_num = len(pins)
    pin_groups = []
    pin_indexes = list(range(pin_num))
    while len(pin_indexes) > 0:
        pin_group = [pin_indexes.pop(0)]
        search_index = 0
        while search_index < len(pin_group):
            nearby_pins = find_nearby_pin(pins, pin_group[search_index], max_distance)
            for pin in nearby_pins:
                if pin not in pin_group: pin_group.append(pin)
                if pin in pin_indexes: pin_indexes.remove(pin)
            search_index += 1
        pin_groups.append(pin_group)
    # 如果输入的pins不是全局pins 更新index
    new_pin_groups = []
    for pin_group in pin_groups:
        new_pin_group = []
        for pin_index in pin_group:
            new_pin_group.append(pins[pin_index]['index'])
        new_pin_groups.append(new_pin_group)

    # 如果一个pin_group完全被另外一个pin_group包含，则将两者合并
    for i in range(len(new_pin_groups)):
        x0, x1, y0, y1 = 1e9, 0, 1e9, 0
        for pin_index in new_pin_groups[i]:
            pin = pins[pin_index]
            x0 = min(x0, pin['x'])
            x1 = max(x1, pin['x'])
            y0 = min(y0, pin['y'])
            y1 = max(y1, pin['y'])
        for j in range(i + 1, len(new_pin_groups)):
            x0_, x1_, y0_, y1_ = 1e9, 0, 1e9, 0
            for pin_index in new_pin_groups[j]:
                pin = pins[pin_index]
                x0_ = min(x0_, pin['x'])
                x1_ = max(x1_, pin['x'])
                y0_ = min(y0_, pin['y'])
                y1_ = max(y1_, pin['y'])
            if x0_ >= x0 and x1_ <= x1 and y0_ >= y0 and y1_ <= y1:
                new_pin_groups[i] += new_pin_groups[j]
                new_pin_groups[j] = []
    new_pin_groups = [pin_group for pin_group in new_pin_groups if pin_group != []]
    return new_pin_groups


def find_nearby_pin(pins, pin_index, distance):
    """
    找到与pin_index距离小于distance的pin
    :param pins:
    :param pin_index:
    :param distance:
    :return:
    """
    nearby_pins = []
    for i in range(len(pins)):
        if i != pin_index and distance_between_pins(pins, pin_index, i) < distance:
            nearby_pins.append(i)
    return nearby_pins


def distance_between_pins(pins, pin_index1, pin_index2):
    """
    计算两个pin之间的汉明距离
    :param pins:
    :param pin_index1:
    :param pin_index2:
    :return:
    """
    x1, y1 = pins[pin_index1]['x'], pins[pin_index1]['y']
    x2, y2 = pins[pin_index2]['x'], pins[pin_index2]['y']
    # return hamming distance
    return abs(x1 - x2) + abs(y1 - y2)


def calculate_std(pins):
    """
    计算标准差
    :param data:
    :return:
    """
    data = []
    for pin in pins:
        data.append((pin['x'], pin['y']))
    data = np.array(data)
    # 返回标准差的平方平均
    return np.mean(np.std(data, axis=0) ** 2) ** 0.5
