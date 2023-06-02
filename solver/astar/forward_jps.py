import json
import logging
import math
from typing import Iterable, Union

from solver.astar.jps3d0 import JPSSolver
from solver.astar import AStar, T
from heapq import heappush, heappop

from utils.np_encoder import NpEncoder


class forwardJPSSolver(JPSSolver):

    def wave_node(self, node, prev_node, max_iter=50):
        direction = tuple([int(a != b and (a - b) / abs(a - b) or 0) for a, b in
                           zip(node[1:], prev_node[1:])])
        if direction == (0, 0):
            return []
        directions = []
        # generate directions near direction
        if direction[0] == 0:
            directions.append((1, direction[1]))
            directions.append((-1, direction[1]))
        elif direction[1] == 0:
            directions.append((direction[0], 1))
            directions.append((direction[0], -1))
        else:
            directions.append((direction[0], 0))
            directions.append((0, direction[1]))
        directions.insert(1, direction)
        last_wave = [node]
        for step in range(1, max_iter):
            # generate 3 key points in each direction
            key_points = []
            wave = []
            for d in directions:
                key_point = (node[0], node[1] + d[0] * step, node[2] + d[1] * step)
                key_points.append(key_point)
            # generate point between key point0 and key point1
            for i in range(1, len(key_points)):
                key_point0 = key_points[i - 1]
                key_point1 = key_points[i]
                if key_point0[1] == key_point1[1]:
                    for y in range(min(key_point0[2], key_point1[2]), max(key_point0[2], key_point1[2]) + 1):
                        wave.append((node[0], key_point0[1], y))
                elif key_point0[2] == key_point1[2]:
                    for x in range(min(key_point0[1], key_point1[1]), max(key_point0[1], key_point1[1]) + 1):
                        wave.append((node[0], x, key_point0[2]))
                else:
                    for x in range(min(key_point0[1], key_point1[1]), max(key_point0[1], key_point1[1]) + 1):
                        y = int((x - key_point0[1]) / (key_point1[1] - key_point0[1]) * (
                                key_point1[2] - key_point0[2]) + key_point0[2])
                        wave.append((node[0], x, y))
            # check if wave is passable
            for w in wave:
                if not self.is_pass(w):
                    return last_wave
                if self.is_goal_reached(w, self.goal):
                    return [w]
            last_wave = wave

    def jump_node(self, node, prev_node, jump_num=0, jump_max=1e10, **kwargs):
        debug = True
        layer, x, y = node
        direction = (x - prev_node[1], y - prev_node[2])
        if self.is_goal_reached(node, self.goal):
            if debug: logging.debug(f'jump_node: goal reached {node}')
            return node, {}
        pass_result = self.is_pass(node, direction == (0, 0))
        if not pass_result:
            if debug: logging.debug(f'jump_node: not pass {node}')
            return None, {}
        if jump_num > jump_max:
            if debug: logging.debug(f'jump_node: jump_max {node}')
            return node, {}
        jump_num += 1
        if self.is_pass((layer, x + direction[0], y + direction[1])):
            t, d = self.jump_node((layer, x + direction[0], y + direction[1]), node, jump_num, jump_max, **kwargs)
            if t:
                if debug: logging.debug(f'jump_node 继续向前搜索跳点: {node} {prev_node} {direction}')
                return t, d
        else:
            if jump_num > 1:
                return prev_node, {}
        if debug: logging.debug(f'jump_node: 啥也没发现 {node} {prev_node} {direction}')
        return None, ''

    def astar(
            self, start: T, goal: T, reversePath: bool = False
    ) -> Union[Iterable[T], None]:
        self.goal = goal
        if self.is_goal_reached(start, goal):
            return [start]
        jump_max = math.hypot(goal[1] - start[1], goal[2] - start[2]) * self.jps_search_rate
        jump_max = max(4 * int(self.line_width), int(jump_max))
        searchNodes = JPSSolver.SearchNodeDict()
        startNode = searchNodes[start] = JPSSolver.SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal)
        )
        openSet: list = []
        heappush(openSet, startNode)
        search_path = []
        while openSet:
            current = heappop(openSet)
            pre_direction = (0, 0)
            if current.came_from is not None:
                pre_direction = (
                    current.data[1] - current.came_from.data[1], current.data[2] - current.came_from.data[2])
            logging.debug(f'current:{current.data} f:{current.fscore}  target:{goal}')
            if self.is_goal_reached(current.data, goal):
                if self.save_search_path:
                    with open(f'data/search_path/{self.net_id}_search_path.json', 'w') as f:
                        json.dump(search_path, f, cls=NpEncoder)
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            neighbors = []
            current_neighbors = list(self.neighbors(current))
            if pre_direction != (0, 0):
                test_neighbor = (
                    current.data[0], current.data[1] + pre_direction[0], current.data[2] + pre_direction[1])
                jp_pos, jp_data = self.jump_node(test_neighbor, current.data,
                                                 jump_max=jump_max, jps=None, ori_direction=pre_direction)
                if jp_pos is not None:
                    current_neighbors.append(jp_pos)
            for neighbor in current_neighbors:
                logging.debug(f'neighbor:{neighbor}')
                jp_pos, jp_data = neighbor, {}
                if not self.is_pass(neighbor):
                    continue
                jp = searchNodes[jp_pos]
                if jp.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current.data, jp.data
                )
                if tentative_gscore >= jp.gscore:
                    continue
                self.point_check(current, goal, jp, jp_data, jp_pos, neighbor, openSet, tentative_gscore)
            if self.save_search_path:
                data = {'current': current.data, 'neighbors': current_neighbors}
                search_path.append(data)
        return None
