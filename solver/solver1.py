# A Unified Printed Circuit Board Routing Algorithm With Complicated Constraints and Differential Pairs
# 实现论文名称
from solver.base_solver import BaseSolver
import logging


class Solver1(BaseSolver):

    def __init__(self, problem):
        super().__init__(problem)

    def solve(self):
        super().solve()
