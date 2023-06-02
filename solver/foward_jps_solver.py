from solver.astar.forward_jps import forwardJPSSolver
from solver.astar.multilayer_nn_astar import MazeNNSolver
# 实现论文名称
from solver.base_solver import BaseSolver
import logging


class ForwardJPSSolver(BaseSolver):
    name = 'ForwardJPSSolver'

    def get_solver(self, **kwargs):
        return forwardJPSSolver(**kwargs)
