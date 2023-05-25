from solver.astar.jps3d0 import JPSSolver
# 实现论文名称
from solver.base_solver import BaseSolver
import logging


class JpsSolver(BaseSolver):
    name = 'JPS_solver'

    def get_solver(self, **kwargs):
        return JPSSolver(**kwargs)