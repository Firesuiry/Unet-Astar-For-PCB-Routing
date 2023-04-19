# 导入Flask类库
import json
import pickle

import numpy as np
from flask import Flask
from multiprocessing import shared_memory
from flask_cors import CORS, cross_origin
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
    try:
        shm = shared_memory.SharedMemory(name='solver_data')
    except FileNotFoundError:
        return None
    data_len = np.frombuffer(shm.buf, dtype=np.int64, count=1)[0]
    binary_data = shm.buf[8:8 + data_len]
    data = pickle.loads(binary_data)
    return json.dumps(data, cls=NpEncoder)


# 启动服务
if __name__ == '__main__':
    app.run(debug=True)
