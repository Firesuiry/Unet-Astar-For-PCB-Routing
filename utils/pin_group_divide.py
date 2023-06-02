import numpy as np


def pin_group_divide(pins, max_distance=None):
    """
    将pin分组，对于组内任意一pin，存在另一个pin与其距离小于max_distance
    :param pins:
    :return:
    """

    if max_distance is None:
        std = calculate_std(pins)
        max_distance = std*0.2
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
    return pin_groups


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
