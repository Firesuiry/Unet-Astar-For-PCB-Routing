from solver.astar.jps3d0 import JPSSolver
# 实现论文名称
from solver.base_solver import BaseSolver
import logging


class JpsSolver(BaseSolver):

    def __init__(self, problem):
        super().__init__(problem)

    def get_solver(self, **kwargs):
        return JPSSolver(**kwargs)

    # def solve(self):
    #     super().solve()
