import multiprocessing as mp
import random
import time

import numpy as np
from multiprocessing import shared_memory
import logging
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pygame

DIRECTIONS = np.array([
    [0, 0, 1],
    [0, 0, -1],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
])


def generate_single_prob(prob_id, net_id, location, direction, dest_locations, pheromone, max_location, former_cost):
    d = {
        'prob': 0,
        'cost_add': 0,
        'arrive': 0,
    }
    new_direction = DIRECTIONS[prob_id]
    single_prob = 0
    new_location = new_direction + location
    # 违规则返回0
    if (new_location > max_location).any() or (new_location < 0).any():
        return d
    # 如果同个方向再冲一波撞墙返回0
    if np.sum(new_direction) == 1:
        virtual_new_location = new_location + new_direction
        if (virtual_new_location > max_location).any() or (virtual_new_location < 0).any():
            return d
    if np.sum(new_direction) == 2:
        virtual_new_location = new_location + new_direction
        if (virtual_new_location[0] + new_direction[0] < 0 or
            virtual_new_location[0] + new_direction[0] > max_location[0]) and \
                (virtual_new_location[1] + new_direction[1] < 0 or
                 virtual_new_location[1] + new_direction[1] > max_location[1]):
            return d

    # 方向和当前direction不符返回0
    if direction[2] != 0:
        # 之前上下移动 则本次上下移动返回0
        if new_direction[2] != 0:
            return d
    else:
        # z方向为0
        if direction[0] + direction[1] > 0:
            # x y方向不为0 下次角度偏大则返回0
            if np.sum(np.abs(new_direction - direction)) > 1:
                return d
        else:
            # x y z方向都为0 禁止z方向运动
            if new_direction[2] != 0:
                return d

    single_prob = 100
    # 计算信息素加成
    pheromone_sum = np.sum(pheromone[tuple(new_location)])
    target_pheromone = pheromone[tuple(new_location)][net_id]
    other_pheromone = pheromone_sum - target_pheromone

    d['target_pheromone'] = target_pheromone
    d['other_pheromone'] = other_pheromone

    prob_add = target_pheromone
    if other_pheromone > 10:
        prob_add = prob_add * other_pheromone ** -0.1
    # single_prob += prob_add

    # pheromone[tuple(new_location)][net_id] += 1
    # 计算A*加成
    cost_add = 0
    if new_direction[2] != 0:
        cost_add += 5
    cost_add += np.sum(np.abs(new_direction[0:2]))
    d['cost_add'] = cost_add

    future_cost = np.min(np.sum(np.abs(new_location - dest_locations), axis=1))
    # all_cost = former_cost + cost_add + future_cost
    all_cost = future_cost
    d['future_cost'] = future_cost

    # single_prob = single_prob * all_cost ** -1
    arrive = False
    if future_cost == 0:
        single_prob = 1e300
        arrive = True
    d['prob'] = single_prob
    d['arrive'] = arrive
    return d


def generate_prob(prob, net_id, location, direction, dest_locations, pheromone, max_location, former_cost):
    prob[:] = 0
    costs = []
    arrive_flag = False
    future_costs = np.zeros((10,), dtype=float) - 1
    target_pheromones = np.zeros((10,), dtype=float)
    for i in range(10):
        d = generate_single_prob(i, net_id=net_id, location=location, direction=direction,
                                 dest_locations=dest_locations, pheromone=pheromone,
                                 max_location=max_location, former_cost=former_cost)
        prob[i] = d['prob']
        cost = d['cost_add']
        arrive = d['arrive']
        if d['prob'] != 0:
            target_pheromone = d['target_pheromone']
            other_pheromone = d['other_pheromone']
            future_costs[i] = future_cost = d['future_cost']
            target_pheromones[i] = target_pheromone
        costs.append(cost)
        arrive_flag = arrive_flag or arrive

    # 计算各方向概率
    fc_copy = future_costs.copy()
    max_cost = np.max(future_costs[np.where(future_costs != -1)])
    min_cost = np.min(future_costs[np.where(future_costs != -1)])
    if max_cost != min_cost:
        future_costs[np.where(future_costs == -1)] = np.max(future_costs)
        future_costs = future_costs - min_cost
        future_costs = future_costs / (max_cost - min_cost)
    else:
        future_costs[np.where(future_costs == max_cost)] = 0
        future_costs[np.where(future_costs == -1)] = 1

    prob += target_pheromones
    prob = prob * (1.2 - future_costs)
    prob = prob / np.sum(prob)

    assert (prob >= 0).all()

    if np.sum(prob) == 0:
        for i in range(10):
            prob[i], cost, arrive = generate_single_prob(i, net_id=net_id, location=location, direction=direction,
                                                         dest_locations=dest_locations, pheromone=pheromone,
                                                         max_location=max_location, former_cost=former_cost)

    return prob, costs, arrive_flag


def route(net_id, source_location, dest_locations, pheromone_shape, max_steps, max_location):
    pheromone_shm = shared_memory.SharedMemory(name='pheromone')
    pheromone = np.ndarray(pheromone_shape, dtype=np.float64,
                           buffer=pheromone_shm.buf)
    location = np.zeros((1, 3), dtype=int)
    direction = np.zeros((3,), dtype=int)
    prob = np.zeros((10,), dtype=float)

    route_locations = []

    # 归零
    location = source_location.copy()
    direction[:] = 0
    cost = 0

    display = True
    if display:
        img = np.zeros((pheromone.shape[0], pheromone.shape[1], 3), dtype=np.uint8)
        for dest_location in dest_locations:
            img[int(dest_location[0]), int(dest_location[1])] = [0, 255, 255]
        pygame.init()
        screen = pygame.display.set_mode((1200, 800))
    arrive = False
    for step in range(max_steps):
        # 概率计算
        prob, costs, arrive = generate_prob(prob, net_id=net_id, location=location, direction=direction,
                                            dest_locations=dest_locations, pheromone=pheromone,
                                            max_location=max_location, former_cost=cost)
        assert np.sum(prob) != 0

        indexs = list(range(len(DIRECTIONS)))
        index = np.random.choice(indexs, p=prob / np.sum(prob))

        cost += costs[index]
        logging.debug(f'step:{step} cost:{cost} arrive:{arrive}')
        if arrive:
            ...

        direction = DIRECTIONS[index]
        location = location + direction
        route_locations.append(location.copy())
        if display:
            img[location[0], location[1]] = [255, 0, 0]
            new_shape = (img.shape[1] * 4, img.shape[0] * 4)
            new_img = cv2.resize(img, new_shape)
            img_py = pygame.surfarray.make_surface(new_img)
            a = 0
            b = 0
            bl_color = (0, 0, 0)
            screen.fill(bl_color)
            screen.blit(img_py, [a, b])
            pygame.display.flip()
            pass

        if arrive or step == (max_steps-1):
            if display:
                # time.sleep(5)
                pygame.quit()
            break
    # 更新信息素
    if arrive:
        for route_location in route_locations:
            pheromone[tuple(route_location)] += 1
        pheromone[:, :, :, net_id] *= 0.99
    del pheromone
    pheromone_shm.close()


class AntSolver:
    def __init__(self, problem):
        self.problem = problem
        self.pins = problem.pins
        self.nets = problem.nets
        self.pin_num = len(self.pins)
        self.pheromone_shape = (problem.max_x, problem.max_y, problem.layer_num, len(problem.nets))
        pheromone = np.zeros(self.pheromone_shape, dtype=np.float64)
        self.pheromone_shm = shared_memory.SharedMemory(size=pheromone.nbytes, create=True, name='pheromone')
        del pheromone
        self.pheromone = np.ndarray(self.pheromone_shape, dtype=np.float64,
                                    buffer=self.pheromone_shm.buf)
        self.pheromone[:] = 0

        self.max_location = np.array([problem.max_x, problem.max_y, problem.layer_num]) - 1

    def solve(self):
        epoch = 0
        while True:
            epoch += 1
            logging.debug(f'epoch {epoch} start')
            if epoch > 9:
                break
            for net in self.nets[1:]:
                net_id = net['index']
                logging.debug(f"net {net['name']} start")

                pins = net['pins']
                for pin in pins:
                    source_location = np.array([self.pins[pin]['x_int'], -self.pins[pin]['y_int'], 0])
                    target_locations = self.target_locations(pins, pin)
                    route(net_id=net_id, source_location=source_location, dest_locations=target_locations,
                          pheromone_shape=self.pheromone_shape, max_steps=19999, max_location=self.max_location)

            pass

    def target_locations(self, pins, exclude_index):
        target_locations = np.zeros((len(pins) - 1, 3))
        index = 0
        for i in range(len(pins)):
            if self.pins[pins[i]]['index'] == exclude_index:
                continue
            target_locations[index] = [self.pins[pins[i]]['x_int'], -self.pins[pins[i]]['y_int'], 0]
            index += 1
        return target_locations
