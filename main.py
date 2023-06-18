import pickle
from datetime import datetime
from multiprocessing import shared_memory

from ant_solver import AntSolver
from dsn_resolve import DsnResolver
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
    problem = pickle.load(open(R'D:\develop\PCB\network\dataset\d84205b3ab56bb5eeaf9729317b8eaa2\problem.pkl', 'rb'))
    Solvers = [Solver1]
    Solver = Solver1
    # for Solver in Solvers:
    for model in ['network/best_val_model.pth', None]:
        s = time.time()
        solver = Solver(problem, model_path=model)
        # solver = Solver(problem, model_path=None, hx_multi_rate=1.35, jps_search_rate=0.4)
        solver.solve()
        solver.save_data()
        log_print(f'{Solver.__name__}花费时间:{time.time() - s:.2f} 运行结果:{solver.running_result()}')
    # hx_params = [1, 1.35]
    # jps_params = [0.2]
    # for hx_param in hx_params:
    #     for jps_param in jps_params:
    #         s = time.time()
    #         solver = JpsSolver(problem, hx_multi_rate=hx_param, jps_search_rate=jps_param, speed_test=True)
    #         solver.solve()
    #         solver.save_data()
    #         log_print(f'JpsSolver hx_param:{hx_param} jps_param:{jps_param} '
    #                   f'花费时间:{time.time() - s:.2f} 运行结果:{solver.running_result()}')
    input('按任意键继续...')


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
    # solve()
    # cluster()
    # test_problem()
    # test4()
    # test2()
    # test3()
    compare()
