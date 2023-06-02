import logging

from network.sample_analysis import sample_analysis
# from network.genrate_sample import generate_sample
from network.train import train

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                    datefmt='%Y %b %d %H:%M:%S', )
logging.info('开始处理')

if __name__ == '__main__':
    w_max, w_min, h_max, h_min = 5000, 1000, 5000, 1000
    # w_max, w_min, h_max, h_min = 1000, 500, 1000, 500
    l, pin_density, obs_density = 2, 0.28, 0.11
    # generate_1_sample(3000, 3000, 2, 160/8, debug=False)
    # generate_sample(w_max, w_min, h_max, h_min, l, pin_density, obs_density, sample_num=int(1e6), debug=False)
    # solver = LeeSolver2(np.zeros((1, 9, 9), dtype=int))
    # solver.solve((0, 0, 0), (0, 8, 8))
    # solver.display()
    # train()

    sample_analysis(322)
