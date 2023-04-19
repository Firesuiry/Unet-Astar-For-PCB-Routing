import math

import numpy as np


def get_first_key(d: dict):
    return list(d.keys())[0]


def remove_empty_item(l: list):
    while '' in l:
        l.remove('')
    return l


def list_split_and_remove(s: str, 分隔符):
    return remove_empty_item(s.split(分隔符))


def 返回下一个括号的位置和中间的内容(text):
    # print(f'解析内容：{text}')
    data = {}
    第一个括号出现的位置 = -1
    当前括号的层数 = 0
    最后一个括号的位置 = -1
    for i in range(len(text)):
        single_word = text[i]
        if text[i] == "(":
            当前括号的层数 += 1
            if 第一个括号出现的位置 == -1:
                第一个括号出现的位置 = i
        elif text[i] == ")":
            当前括号的层数 -= 1
            if 当前括号的层数 == 0:
                最后一个括号的位置 = i
                break
    if 第一个括号出现的位置 == -1 or 最后一个括号的位置 == -1:
        return []
    括号中间的内容 = text[第一个括号出现的位置 + 1:最后一个括号的位置]
    key = ''
    for i in range(len(括号中间的内容)):
        if 括号中间的内容[i] == ' ':
            key = 括号中间的内容[0:i]
            下一段内容 = 括号中间的内容[i + 1:]
            if "(" in 下一段内容:
                下一段内容起始位置 = 下一段内容.index("(")
                data1 = {下一段内容[:下一段内容起始位置]: ''}
                data2 = 返回下一个括号的位置和中间的内容(下一段内容[下一段内容起始位置:])
                if (type(data2) == dict):
                    data[key] = [data1] + [data2]
                else:
                    data[key] = [data1] + data2
            else:
                data[key] = 括号中间的内容[i + 1:]
            break
        elif 括号中间的内容[i] == "(":
            key = 括号中间的内容[0:i]
            data[key] = 返回下一个括号的位置和中间的内容(括号中间的内容[i:])
            break
    other_data = 返回下一个括号的位置和中间的内容(text[最后一个括号的位置 + 1:])
    if type(other_data) == dict:
        other_data = [other_data]
    return_data = [data] + other_data
    if len(return_data) == 1:
        return_data = return_data[0]

    return return_data


def get_item(data, key):
    if type(data) == dict and data.get(key):
        return data.get(key)
    if type(data) == list:
        for d in data:
            if list(d.keys())[0].replace('\n', '') == key:
                return list(d.values())[0]
    raise BaseException('no exists')


class DsnResolver:

    def __init__(self, text):
        self.text = text
        self.tree = 返回下一个括号的位置和中间的内容(text)['PCB']
        # print(self.tree)
        self.structure_data = get_item(self.tree, 'structure')
        self.boundary_data = get_item(self.structure_data, 'boundary')
        self.boundary = {
            'min_x': 0,
            'max_x': 0,
            'min_y': 0,
            'max_y': 0,
        }
        self.layer = []
        self.pins = {}
        self.get_boundary()
        self.boundary['layer'] = self.layer

        self.library_data = get_item(self.tree, 'library')
        self.library_image_data = get_item(self.library_data, 'image')
        self.padstacks = {}
        self.get_padstack()
        self.get_pins()

        self.connections = {}
        self.network_data = get_item(self.tree, 'network')
        self.get_connections()

        self.grid = ...
        self.clearance = ...
        self.line_width = ...
        self.get_grid_clearance()

    def get_grid_clearance(self):
        for item in self.structure_data:
            if item.get('grid'):
                d = item['grid']
                self.grid = float(d.split(' ')[-1]) * 10
            elif item.get('rule'):
                sub_item = item['rule']
                if sub_item.get('clear'):
                    d = sub_item['clear']
                    if type(d) == str:
                        self.clearance = float(d) * 10
                    elif type(d) == list:
                        self.clearance = float(list(d[0].keys())[0].replace(' ', '')) * 10
                elif sub_item.get('width'):
                    self.line_width = float(sub_item['width']) * 10
        self.grid = int(self.grid)
        self.clearance = math.ceil(self.clearance)
        self.line_width = math.ceil(self.line_width)
        ...

    def get_connections(self):
        for d in self.network_data:
            if get_first_key(d) == 'net':
                d2 = d['net']
                name = get_first_key(d2[0]).replace('\n', '').replace(' ', '')
                pins = [pin.replace('u1-', '') for pin in d2[1]['pins'].split(' ')]
                self.connections[name] = pins

    def get_padstack(self):
        for item in self.library_data:
            is_padstack = False
            for k, v in item.items():
                if k == 'padstack':
                    is_padstack = True
            if is_padstack:
                data_list = item['padstack']
                padstack_name = ''
                new_data = {}
                for data in data_list:  # {'padstack': [{'p2316\n      ': ''}, {'shape': {'circle': '1 6.2992 0 0'}}, {'shape': {'circle': '2 6.2992 0 0'}}]}
                    if data.get('shape'):
                        # {'shape': {'circle': '1 2.4'}}
                        shape_data = data['shape']  # {'circle': '1 2.4'}
                        shape = list(shape_data.keys())[0]
                        shape_detail = shape_data[shape].split(' ')
                        new_data[shape_detail[0]] = {
                            'shape': shape,
                            'detail': [round(float(x) * 10) for x in shape_detail[1:]]
                        }
                    else:  # {'p2316\n      ': ''}
                        padstack_name = list(data.keys())[0].replace('\n', '')
                assert padstack_name
                self.padstacks[padstack_name.replace(' ', '')] = new_data

    def get_pins(self):
        for d in self.library_image_data:
            if get_first_key(d) == 'pin':
                contents = d['pin'].split(' ')
                pin_data = {
                    'id': str(contents[1]),
                    'shape': contents[0],
                    'x': float(contents[2]) * 10,
                    'y': float(contents[3]) * 10,
                }
                pin_data['x_int'] = round(pin_data['x'])
                pin_data['y_int'] = round(pin_data['y'])
                self.pins[pin_data['id']] = pin_data

    def get_boundary(self):
        path = self.boundary_data['path']
        datas = list_split_and_remove(path, ' ')
        boundary_layer_id = datas[0]
        aperture_width = datas[1]
        coordinate = []
        coordinate_num = int((len(datas) - 2) / 2)
        for i in range(coordinate_num):
            coordinate.append([float(datas[2 + i * 2]), float(datas[2 + i * 2 + 1])])
        self.boundary_coordinate = np.array(coordinate)
        self.boundary['min_x'] = round(np.min(self.boundary_coordinate[:, 0]) * 10)
        self.boundary['max_x'] = round(np.max(self.boundary_coordinate[:, 0]) * 10)
        self.boundary['min_y'] = round(np.min(self.boundary_coordinate[:, 1]) * 10)
        self.boundary['max_y'] = round(np.max(self.boundary_coordinate[:, 1]) * 10)
        layer_num = 0
        for d in self.structure_data:
            if list(d.keys())[0] == 'layer':
                d2 = d['layer']
                layer_id = list(d2[0].keys())[0].split('\n')[0]
                layer_type = d2[1]['type']
                self.layer.append({
                    'id': int(layer_id),
                    'type': layer_type,
                })
                pass
        pass

    def resolve(self):
        pass


if __name__ == '__main__':
    t = ''
    with open('dsn文件/Autorouter_PCB_2023-01-05.dsn', 'r') as f:
        t = f.read()
    resolver = DsnResolver(t)
