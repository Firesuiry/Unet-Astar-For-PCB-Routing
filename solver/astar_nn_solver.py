from solver.astar.jps3d0 import JPSSolver
from solver.astar.multilayer_nn_astar import MazeNNSolver
# 实现论文名称
from solver.base_solver import BaseSolver
import logging


class AstarNNSolver(BaseSolver):
    name = 'AstarNNSolver'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.liner_nn_power = kwargs.get('liner_nn_power', 3)
        self.multi_nn_power = kwargs.get('multi_nn_power', 3)
        self.skip_percent = kwargs.get('skip_percent', 0.3)

    def get_solver(self, **kwargs):
        kwargs['liner_nn_power'] = self.liner_nn_power
        kwargs['multi_nn_power'] = self.multi_nn_power
        return MazeNNSolver(**kwargs)

    def generate_recommend_area(self, net_id, pass_small_net=False):
        if net_id / len(self.steiner_nets) < self.skip_percent:
            return None
        return super().generate_recommend_area(net_id, pass_small_net)

    def running_result(self):
        data = super().running_result()
        data['liner_nn_power'] = self.liner_nn_power
        data['multi_nn_power'] = self.multi_nn_power
        data['skip_percent'] = self.skip_percent
        return data
