from ant_solver import AntSolver
from dsn_resolve import DsnResolver
from problem import Problem
from solver.solver1 import Solver1
import logging
import time
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                    datefmt='%Y %b %d %H:%M:%S', )
if __name__ == '__main__':
    logging.info('开始处理')
    t = ''
    s = time.time()
    with open('dsn文件/Autorouter_PCB_2023-01-05.dsn', 'r') as f:
    # with open('dsn文件/Autorouter_STM32F103VE_Board_JX V1.0_2023-03-18.dsn', 'r') as f:
        t = f.read()
    resolver = DsnResolver(t)
    problem = Problem(resolver)
    # problem.show(0)
    solver = Solver1(problem)
    solver.resolution_change(1)
    solver.save_data()
    solver.net_display_init()
    for i in range(len(solver.steiner_nets)):
        solver.assemble_net_display(i)
    solver.solve()
    solver.display()
    logging.info(f'花费时间:{time.time()-s:.2f}')
    pass

