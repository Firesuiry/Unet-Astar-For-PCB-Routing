import logging
import pathlib
import pickle
import pandas as pd


def fine_tune():
    p = pathlib.Path('data\\nn')
    datas = []
    for file in p.glob('*.pickle'):
        filename = file.name
        fn = filename.replace('.pickle', '')
        # the format of fn is f'l{liner_nn_power}-m{multi_nn_power}'
        liner_nn_power = float(fn.split('-')[0][1:])
        multi_nn_power = float(fn.split('-')[1][1:])
        new_data = {}
        new_data['liner_nn_power'] = liner_nn_power
        new_data['multi_nn_power'] = multi_nn_power
        print(f'liner_nn_power: {liner_nn_power}, multi_nn_power: {multi_nn_power}')
        load_data(datas, file, new_data)
    # save the data as csv
    df = pd.DataFrame(datas)
    df.to_csv('data\\nn\\nn.csv', index=False)


class NewDict(dict):
    def __missing__(self, key):
        # create key automitically
        self[key] = []
        return self[key]


def compare():
    p = pathlib.Path('data\\nn2')
    datas_dict = NewDict()
    for file in p.glob('*.pickle'):
        filename = file.name
        fn = filename.replace('.pickle', '')
        # the format of fn is l{liner_nn_power}-{problem_path.name}
        liner_nn_power = float(fn.split('-')[0][1:])
        question_id = fn.split('-')[1]
        new_data = {
            '问题': question_id
        }
        load_data(datas_dict[liner_nn_power], file, new_data)
    # save the data as csv
    fail_to_save = []
    for k, v in datas_dict.items():
        try:
            df = pd.DataFrame(v)
            df.to_csv(f'data\\nn2\\l{k}.csv', index=False)
        except:
            fail_to_save.append(k)
    logging.error(f'fail to save: {fail_to_save}')


def load_data(datas, file, new_data):
    # load the data
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(data)
    # {'线路长度': 6689.869909008294, 'VIA数量': 85, '连通率': 1.0, '运行时间': 154.5764617919922, '设计规则违例': -1}
    new_data['线路长度'] = data['线路长度']
    new_data['VIA数量'] = data['VIA数量']
    new_data['连通率'] = data['连通率']
    new_data['运行时间'] = data['运行时间']
    new_data['设计规则违例'] = data['设计规则违例']
    new_data['新线路长度'] = data['新线路长度']
    new_data['新VIA数量'] = data['新VIA数量']
    new_data['新连通率'] = data['新连通率']
    route_detail = data['详细运行监控']
    # 到多少以后nn有显著优势 原始/nn版本展开格子数量/运行时间
    better_percent = 0
    net_num = len(list(route_detail.keys()))
    ori_run_time = 0
    ori_search_area = 0
    new_run_time = 0
    new_search_area = 0
    for net_id in range(net_num):
        if net_id < 0.25 * net_num:
            continue
        v = route_detail[net_id]
        # {'ori_usetime': -1, 'new_usetime': 0.029946327209472656, 'ori_search_area': 0, 'new_search_area': 723}
        ori_usetime = v['ori_usetime']
        new_usetime = v['new_usetime']
        if new_usetime < ori_usetime:
            better_percent = net_id / net_num
            print(f'better_percent: {better_percent}')
            break
    for net_id in range(net_num):
        if route_detail[net_id]['ori_search_area'] is None:
            continue
        ori_run_time += route_detail[net_id]['ori_usetime']
        ori_search_area += route_detail[net_id]['ori_search_area']
        if net_id <= better_percent:
            new_run_time += route_detail[net_id]['ori_usetime']
            new_search_area += route_detail[net_id]['ori_search_area']
        else:
            new_run_time += route_detail[net_id]['new_usetime']
            new_search_area += route_detail[net_id]['new_search_area']
    new_data['原始运行时间'] = ori_run_time
    new_data['原始展开格子数量'] = ori_search_area
    new_data['新运行时间'] = new_run_time
    new_data['新展开格子数量'] = new_search_area
    new_data['新运行时间/原始运行时间'] = new_run_time / ori_run_time
    new_data['新展开格子数量/原始展开格子数量'] = new_search_area / ori_search_area
    new_data['最优百分比'] = better_percent
    print(new_data)
    datas.append(new_data)


if __name__ == '__main__':
    compare()
# fine_tune()
