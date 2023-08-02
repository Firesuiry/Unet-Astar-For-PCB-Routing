import os
import pickle
from datetime import datetime
from multiprocessing import shared_memory
# import multiprocessing as mp
import torch.multiprocessing as mp
from pathlib import Path

import torch

from ant_solver import AntSolver
from dsn_resolve import DsnResolver
from network.unet import ResNetUNet
from problem import Problem, RandomProblem
from solver.astar_nn_solver import AstarNNSolver
from solver.foward_jps_solver import ForwardJPSSolver
from solver.jps_solver import JpsSolver
from solver.rect_solver import RectSolver
from solver.solver1 import Solver1
import logging
import time


def solve():
    s = time.time()
    # with open('dsn文件/Autorouter_PCB_2023-01-05.dsn', 'r') as f:
    with open('dsn文件/Autorouter_STM32F103VE_Board_JX V1.0_2023-03-18.dsn', 'r') as f:
        t = f.read()
    resolver = DsnResolver(t)
    problem = Problem(resolver)
    problem = pickle.load(open(R'D:\develop\PCB\network\dataset\d84205b3ab56bb5eeaf9729317b8eaa2\problem.pkl', 'rb'))
    # problem.show(0)
    # solver = Solver1(problem)
    # solver = JpsSolver(problem, hx_multi_rate=1.35, jps_search_rate=0.2, model_path='network/best_val_model.pth')
    solver = JpsSolver(problem, hx_multi_rate=1.35, jps_search_rate=0.2, model_path=None)
    solver.solve()
    # solver.net_display_init()
    # for i in range(len(solver.steiner_nets)):
    #     solver.assemble_net_display(i)
    # solver.save_data()
    logging.info(f'花费时间:{time.time() - s:.2f}')
    input('按任意键继续...')


def log_print(log):
    logging.info(log)
    with open('log.txt', 'a+', encoding='utf-8') as f:
        f.write(datetime.now().strftime("%Y-%m-%d-%H-%M-%S|") + log + '\n')


def compare():
    # logging.basicConfig(level=logging.INFO,
    #                     format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
    #                     datefmt='%Y %b %d %H:%M:%S',
    #                     filename=f'./logs/compare-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log',
    #                     filemode='a+')
    logging.info('开始处理')
    # with open('dsn文件/Autorouter_DC_5V_3V3_TWO Channel V1.0_2023-06-03.dsn', 'r', encoding='utf-8') as f:
    # with open('dsn文件/Autorouter_PCB_2023-01-05.dsn', 'r') as f:
    with open('dsn文件/Autorouter_STM32F103VE_Board_JX V1.0_2023-03-18.dsn', 'r') as f:
        t = f.read()
    resolver = DsnResolver(t)
    problem = Problem(resolver)
    # problem = pickle.load(open(R'D:\dataset\000b4560cc2a244ecd10c87feaa36621\problem.pkl', 'rb'))
    Solvers = [Solver1]
    Solver = AstarNNSolver
    # for Solver in Solvers:
    model_path = 'best_val_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = ResNetUNet(3, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    problems_path = Path(r'D:\dataset')
    for problem_path in list(problems_path.glob('*'))[15:]:
        problem = pickle.load(open(problem_path / 'problem.pkl', 'rb'))
        liner_nn_power = 800
        save_path = f'data\\nn2\\l{liner_nn_power}-{problem_path.name}.pickle'
        single_run(problem, model, liner_nn_power, 0, skip_percent=0.3, save_path=save_path)
        return 0
    problems_path = Path(r'Z:\\')
    root_save_path = f'data\\nn2'
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    pool = mp.Pool(24)
    liner_nn_power = 800
    skip_percent = 0.3
    for problem_path in list(problems_path.glob('*'))[:200]:
        problem = pickle.load(open(problem_path / 'problem.pkl', 'rb'))
        print(problem_path.name)
        save_path = f'{root_save_path}\\e-{problem_path.name}.pickle'
        pool.apply_async(single_run, args=(problem, model, liner_nn_power, 0, save_path, 0.2))
        save_path = f'{root_save_path}\\f-{problem_path.name}.pickle'
        # single_run(problem, None, liner_nn_power, 0, save_path, 0.2)
        pool.apply_async(single_run, args=(problem, None, liner_nn_power, 0, save_path, 0.2))
    pool.close()
    pool.join()


def skip_percent_test():
    model_path = 'best_val_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(3, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    problems_path = Path(r'Z:\\')
    root_save_path = f'data\\nn2'
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    pool = mp.Pool(8)
    liner_nn_power = 800
    for skip_percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:  # [100, 200, 300, 400, 500, 600, 700]:
        for problem_path in list(problems_path.glob('*'))[:20]:
            problem = pickle.load(open(problem_path / 'problem.pkl', 'rb'))
            print(problem_path.name)
            save_path = f'{root_save_path}\\s{skip_percent}-{problem_path.name}.pickle'
            pool.apply_async(single_run, args=(problem, model, liner_nn_power, 0, save_path, skip_percent))
    pool.close()
    pool.join()


def liner_power_test():
    model_path = 'best_val_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(3, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    problems_path = Path(r'Z:\\')
    root_save_path = f'data\\nn2'
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    pool = mp.Pool(8)
    for liner_nn_power in [800, 200, 400, 100, 1600, 0]:  # [100, 200, 300, 400, 500, 600, 700]:
        for problem_path in list(problems_path.glob('*'))[:20]:
            problem = pickle.load(open(problem_path / 'problem.pkl', 'rb'))
            print(problem_path.name)
            save_path = f'{root_save_path}\\l{liner_nn_power}-{problem_path.name}.pickle'
            pool.apply_async(single_run, args=(problem, model, liner_nn_power, 0, save_path, 0.2))
    pool.close()
    pool.join()


def single_run(problem, model, liner_nn_power, multi_nn_power, save_path=None, skip_percent=0.0):
    if save_path is None: save_path = f'data\\nn\\l{liner_nn_power}-m{multi_nn_power}.pickle'
    if os.path.exists(save_path):
        logging.info(f'l{liner_nn_power}-m{multi_nn_power}已存在')
        return
    kwargs = {
        'model': model,
        'liner_nn_power': liner_nn_power,
        'multi_nn_power': multi_nn_power,
        'skip_percent': skip_percent
    }
    solver = AstarNNSolver(problem, **kwargs)
    solver.solve()
    solver.save_data()
    run_result = solver.running_result()
    # save result
    with open(save_path, 'wb') as f:
        pickle.dump(run_result, f)
    log_print(f'l{liner_nn_power}-m{multi_nn_power}运行结果:{solver.running_result()}')
    ...


def cluster():
    s = time.time()
    with open('dsn文件/Autorouter_STM32F103VE_Board_JX V1.0_2023-03-18.dsn', 'r') as f:
        t = f.read()
    resolver = DsnResolver(t)
    problem = Problem(resolver)
    solver = RectSolver(problem)
    solver.resolution_change(2)
    solver.net_display_init()
    for i in range(len(solver.steiner_nets)):
        solver.assemble_net_display(i)
    solver.net_group_generate(0.1)
    for i in range(len(solver.net_groups)):
        solver.net_group_route(i)
    for net_id in solver.pending_nets:
        solver.net_route(net_id)
    solver.save_data()
    solver.cross_check()
    logging.info(f'花费时间:{time.time() - s:.2f}')
    ...


def test_problem():
    s = time.time()
    # problem = RandomProblem(w=3000, h=3000, l=2, pin_num=20)
    # problem.save('test.pickle')
    problem = RandomProblem.load('test.pickle')
    solver = JpsSolver(problem)
    solver.solve()
    logging.info(f'花费时间:{time.time() - s:.2f}')
    ...
    time.sleep(1)
    input('press any key to continue')


def test2():
    shm = shared_memory.SharedMemory(create=True, size=2 ** 30, name='solver_data')
    # solver = JpsSolver.load('data/fail_solver/net193.pickle')
    solver = RectSolver.load('data/rect_solver.pickle')
    for i in range(len(solver.rects)):
        solver.rect_route(solver.rect_crowded_rank[i])
    ...


def test3():
    with open('dsn文件/Autorouter_STM32F103VE_Board_JX V1.0_2023-03-18.dsn', 'r') as f:
        t = f.read()
    resolver = DsnResolver(t)
    problem = Problem(resolver)
    solver = RectSolver(problem, hx_multi_rate=1.35, jps_search_rate=0.2, model_path=None)
    solver.resolution_change(2)
    solver.save('data/rect_solver.pickle')


def test4():
    with open('dsn文件/Autorouter_STM32F103VE_Board_JX V1.0_2023-03-18.dsn', 'r') as f:
        t = f.read()
    resolver = DsnResolver(t)
    problem = Problem(resolver)
    solver = RectSolver(problem, hx_multi_rate=1.35, jps_search_rate=0.2, model_path=None)
    solver.resolution_change(1)
    solver.solve_graph()
    solver.rect_detail_route_generate()
    solver.rect_crowded_sort()
    solver.save('data/rect_solver.pickle')
    ...


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    torch.multiprocessing.set_start_method('spawn')
    # solve()
    # cluster()
    # test_problem()
    # test4()
    # test2()
    # test3()
    # skip_percent_test()
    compare()
