# 论文 Unet-Astar: A Deep Learning-Based Fast Routing Algorithm for Unified PCB Routing 的开源代码
论文地址：https://ieeexplore.ieee.org/document/10274949

# 想说的话
如果对您有帮助，请点一个Star，引用本论文~~
如果有其他问题 欢迎提Issue

# 训练
```Python
import logging

from network.data_loader import data_copy
from network.train import train, eval

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    train(r'D:\dataset\\', model_type='ori', batch_size=16, dataset_max_num=4e4, epoch=30)
    train(r'D:\dataset\\', model_type='pp', batch_size=10, dataset_max_num=4e4, epoch=30)
    train(r'D:\dataset\\', model_type='', batch_size=16, dataset_max_num=4e4, epoch=30)
```


# 评估
```Python
import logging

from network.data_loader import data_copy
from network.train import train, eval

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    eval(r'D:\dataset\\', model_types=['pp', 'ori', ''], batch_size=20)
```

# 训练集生成
```Python
import logging
from multiprocessing import shared_memory

from network.sample_analysis import sample_analysis
from network.genrate_sample import generate_sample, generate_1_sample
from network.sample_display import sample_display
from network.train import train

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                    datefmt='%Y %b %d %H:%M:%S', )
logging.info('开始处理')



if __name__ == '__main__':
    shm = shared_memory.SharedMemory(create=True, size=2 ** 30, name='solver_data')
    w_max, w_min, h_max, h_min = 5000, 1000, 5000, 1000
    l, pin_density, obs_density = 2, 0.15, 0.1
    generate_sample(w_max, w_min, h_max, h_min, l, pin_density, obs_density, sample_num=int(999), debug=False)
```
