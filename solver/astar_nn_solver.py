from solver.astar.jps3d0 import JPSSolver
from solver.astar.multilayer_nn_astar import MazeNNSolver
# 实现论文名称
from solver.base_solver import BaseSolver
import logging


class AstarNNSolver(BaseSolver):
    name = 'AstarNNSolver'

    def get_solver(self, **kwargs):
        return MazeNNSolver(**kwargs)
