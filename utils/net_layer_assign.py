from functools import partial

import numpy as np
import logging
from sko.GA import GA
from solver.ga import GA as GA2

from solver.ga import Problem


def group_divide(cross_relation: np.ndarray):
    logging.info(F'group divide start')
    net_num = cross_relation.shape[0]
    groups = []
    pending_nets = list(range(net_num))
    while len(pending_nets) > 0:
        group_nets = []
        group_nets.append(pending_nets.pop())
        while True:
            flag = False
            for net in group_nets:
                remove_nets = []
                for net2 in pending_nets:
                    if cross_relation[net, net2]:
                        group_nets.append(net2)
                        remove_nets.append(net2)
                        flag = True
                for net2 in remove_nets:
                    pending_nets.remove(net2)
            if not flag:
                break
        group_nets.sort()
        groups.append(group_nets)
    return groups


def layer_assign(cross_relation: np.ndarray, layer_max=1):
    logging.info(F'layer assign start layer_max={layer_max}')
    net_num = cross_relation.shape[0]
    net_layers = np.zeros((net_num,), dtype=int)
    # 根据连接关系划分为多个分配任务
    groups = group_divide(cross_relation)
    # 求解各group的layer分配
    for i in range(len(groups)):
        group = groups[i]
        logging.info(F'layer assign group {i} start net_num={len(group)}')
        if len(group) == 1:
            net_layers[group[0]] = 0
            continue
        new_cross_relation = cross_relation[group][:, group]
        net_layers[group] = solve(new_cross_relation, layer_max=layer_max)
    return net_layers


def solve(cross_relation, layer_max=2):
    net_layers = heuristic_solver2(cross_relation, layer_max=layer_max)
    return net_layers


def greedy_solver(cross_relation, layer_max=2):
    logging.info(F'greedy_solver start layer_max={layer_max}')
    net_num = cross_relation.shape[0]
    net_layers = np.zeros((net_num,), dtype=int)
    pending_nets = list(range(net_num))

    while len(pending_nets) > 0:
        logging.debug(F'greedy_solver pending_net_num={len(pending_nets)}')
        current_process_net = [pending_nets.pop()]
        layer = 0
        net_layers[current_process_net[0]] = layer
        while True:
            flag = False
            for net in pending_nets:
                if np.sum((cross_relation[net] * (net_layers == layer))[current_process_net]) == 0:
                    net_layers[net] = layer
                    current_process_net.append(net)
                    flag = True
            if flag:
                for net in current_process_net:
                    if net in pending_nets:
                        pending_nets.remove(net)
            layer += 1
            if layer >= layer_max:
                break
            if not flag:
                break
    return net_layers


# 启发式求解
def heuristic_solver(cross_relation, layer_max=2):
    # demo_func = partial(evaluate, cross_relation=cross_relation)
    def demo_func(x):
        return evaluate(x, cross_relation)

    net_num = cross_relation.shape[0]
    ga = GA(func=demo_func, n_dim=net_num, max_iter=500, lb=[0] * net_num, ub=[layer_max - 1] * net_num, precision=1)
    best_x, best_y = ga.run()
    return best_x


def heuristic_solver2(cross_relation, layer_max=2):
    # demo_func = partial(evaluate, cross_relation=cross_relation)
    def demo_func(x):
        return evaluate(x, cross_relation)

    net_num = cross_relation.shape[0]
    problem = Problem(net_num, demo_func, range_max=layer_max)
    ga = GA2(population_size=100, mutation_rate=0.2, crossover_rate=0.2, tournament_size=2, generations=500,
             problem=problem)
    best_x, best_y = ga.run()
    return np.array(best_x)


def bhpsogwo_solver(cross_relation, layer_max=2):
    def demo_func(x):
        return evaluate(x, cross_relation)

    net_num = cross_relation.shape[0]
    ga = GA(func=demo_func, n_dim=net_num, max_iter=500, lb=[0] * net_num, ub=[layer_max - 1] * net_num, precision=1)
    best_x, best_y = ga.run()
    return best_x


def evaluate(net_layers, cross_relation):
    net_num = cross_relation.shape[0]
    same_layer_flag = np.zeros((net_num, net_num), dtype=bool)  # net之间是否同一层
    for i in range(net_num - 1):
        same_layer_flag[i] = net_layers == net_layers[i]
    same_layer_flag = np.tril(same_layer_flag, -1)
    cross_sum = np.sum(cross_relation * same_layer_flag)
    return cross_sum


def run():
    net_num = 100
    cross_relation = np.random.randint(0, 2, (net_num, net_num))
    cross_relation = cross_relation + cross_relation.T
    cross_relation = np.array(cross_relation, dtype=bool)
    for i in range(net_num):
        cross_relation[i, i] = False

    net_layers = heuristic_solver(cross_relation, layer_max=2)
    score = evaluate(net_layers, cross_relation)
    logging.info(F'heuristic_solver1 score {score}')
    net_layers = heuristic_solver2(cross_relation, layer_max=2)
    score = evaluate(net_layers, cross_relation)
    logging.info(F'heuristic_solver2 score {score}')
    net_layers = greedy_solver(cross_relation, layer_max=2)
    score = evaluate(net_layers, cross_relation)
    logging.info(F'greedy_solver score {score}')

    logging.info(F'net_layers {net_layers}')
    logging.info(F'layer assign score {score}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    logging.info('开始处理')
    run()
