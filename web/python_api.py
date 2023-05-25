# 导入Flask类库
import json
import pickle
from pathlib import Path

import numpy as np
from flask import Flask
from multiprocessing import shared_memory
from flask_cors import CORS, cross_origin

from utils.critic_path import key_path2path

# 创建应用实例
app = Flask(__name__)
CORS(app, supports_credentials=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# 视图函数（路由）
@app.route("/")
def hello_world():
    data = get_data()
    return json.dumps(data, cls=NpEncoder)

@app.route('/search_path/<net_id>')
def search_path(net_id):
    p = Path('data/search_path')
    for file in p.glob('*.json'):
        file_id = file.name.split('_')[0]
        if str(file_id) == str(net_id):
            with open(file, 'r') as f:
                return f.read()
    return []


def get_data():
    try:
        shm = shared_memory.SharedMemory(name='solver_data')
    except FileNotFoundError:
        print('solver_data not found')
        return None
    data_len = np.frombuffer(shm.buf, dtype=np.int64, count=1)[0]
    binary_data = shm.buf[8:8 + data_len]
    data = pickle.loads(binary_data)
    # for net in data['steiner_nets']:
    #     if net.get('path'):
    #         net['all_path'] = key_path2path(net['path'])
    return data


# 启动服务
if __name__ == '__main__':
    app.run(debug=True)
    # test()
