import json
import logging
import math
import time

import numpy as np

from solver.astar import AStar, T
from heapq import heappush, heappop
from typing import Iterable, Union, TypeVar, Generic
import logging


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


NET_FLAG_RESOLUTION = 10
Infinite = float("inf")
DIRECTIONS = np.array([
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
])

NINE_DIRECTIONS = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
    ]
)
VIA_COST = 5


class JPSSolver(AStar):
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    class SearchNode(AStar.SearchNode):
        """Representation of a search node"""

        __slots__ = ("data", "gscore", "fscore", "closed", "came_from", "out_openset", "jps")

        def __init__(
                self, data: T, gscore: float = Infinite, fscore: float = Infinite
        ) -> None:
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None
            self.jps = None

    class SearchNodeDict(AStar.SearchNodeDict):
        def __missing__(self, k):
            v = JPSSolver.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def __init__(self, width, height, solver, old_index, start_layers, end_layers, layer_max, obstacle, via_radius,
                 clearance,
                 available_range=None,
                 **kwargs):
        self.goal = None
        self.width = width
        self.height = height
        self.solver = solver
        self.old_index = old_index
        self.start_layers = start_layers
        self.end_layers = end_layers
        self.obstacle = (obstacle != self.old_index) * (obstacle != -1)
        self.via_radius = via_radius
        self.clearance = clearance

        self.net_flags = solver.net_flags
        self.net_flag_details = solver.net_flag_details
        self.layer_max = layer_max
        self.available_range = available_range
        self.cross_punish = kwargs.get('cross_punish', 10)
        self.line_width = max(kwargs.get('line_width', 2.5), 1)  # 为0时可能存在交叉问题
        self.passable_cache = {}
        self.viable_cache = {}
        self.speed_test = kwargs.get('speed_test', False)
        self.save_search_path = True
        if self.speed_test:
            self.save_search_path = False
        self.net_id = kwargs.get('net_id', None)
        # logging.info(f'开始布线 line_width:{self.line_width}')
        self.hx_multi_rate = kwargs.get('hx_multi_rate', 1)
        self.jps_search_rate = kwargs.get('jps_search_rate', 0.1)
        self.recommend_area = kwargs.get('recommend_area', None)



    def is_pass(self, node_data, vertical_move=False):
        key = (node_data, vertical_move)
        if self.passable_cache.__contains__(key):
            pass
        else:
            self.passable_cache[key] = self._is_pass(node_data, vertical_move)
        logging.debug(f'is_pass {node_data} {vertical_move} {self.passable_cache[key]}')
        return self.passable_cache[key]

    def _is_pass(self, node_data, vertical_move=False):
        (layer, x, y) = node_data
        log_flag = True
        if self.speed_test:
            log_flag = False
        if layer < 0 or layer >= self.layer_max:
            if log_flag: logging.debug(f'layer out of range, {node_data}')
            return False
        if x < 0 or x >= self.width:
            if log_flag: logging.debug(f'x out of range. {node_data}')
            return False
        if y < 0 or y >= self.height:
            if log_flag: logging.debug(f'y out of range. {node_data}')
            return False
        if self.available_range is not None and self.available_range[layer, x, y] == 0:
            if log_flag: logging.debug(f'not available range. {node_data}')
            return False
        line_clearance = int(self.clearance + self.line_width // 2)
        if vertical_move:
            if self.obstacle[:, x - self.via_radius:x + self.via_radius + 1,
               y - self.via_radius:y + self.via_radius + 1].any():
                if log_flag: logging.debug(f'vertical_move obstacle. {node_data}')
                return False
        elif self.obstacle[layer, x - line_clearance:x + line_clearance + 1,
             y - line_clearance:y + line_clearance + 1].any():
            if log_flag: logging.debug(f'obstacle. {node_data}')
            return False
        if log_flag: logging.debug(f'可通过. {node_data}')
        return 1

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (layer1, x1, y1) = n1
        (layer2, x2, y2) = n2
        distance = [abs(x1 - x2), abs(y1 - y2)]
        layer_distance = 0
        if layer1 not in self.end_layers:
            layer_distance = 1
        return (max(distance) + (math.sqrt(2) - 1) * min(distance) + layer_distance) * self.hx_multi_rate

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        # 欧式距离
        (layer1, x1, y1) = n1
        (layer2, x2, y2) = n2
        layer_distance = 0 if layer1 == layer2 else VIA_COST * int(self.line_width)
        gscore = math.hypot(x2 - x1, y2 - y1) + layer_distance
        return gscore

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        neighbors = []
        # 起点
        if node.came_from is None:
            for d in DIRECTIONS:
                for layer in self.start_layers:
                    neighbor = (layer, node.data[1] + d[0], node.data[2] + d[1])
                    if 0 <= neighbor[1] < self.width and 0 <= neighbor[2] < self.height:
                        neighbors.append(neighbor)
        else:
            # 同层内平移
            layer, x, y = node.data
            # direction = (node.data[1] - node.came_from.data[1], node.data[2] - node.came_from.data[2])
            direction = tuple([int(a != b and (a - b) / abs(a - b) or 0) for a, b in
                               zip(node.data[1:], node.came_from.data[1:])])
            if direction == (0, 0) and node.came_from and node.came_from.came_from:
                direction = tuple([int(a != b and (a - b) / abs(a - b) or 0) for a, b in
                                   zip(node.came_from.data[1:], node.came_from.came_from.data[1:])])
            if direction != (0, 0):
                n_neighbors = [(layer, node.data[1] + direction[0], node.data[2] + direction[1])]
                if direction[0] != 0 and direction[1] != 0:
                    n_neighbors.append((layer, node.data[1], node.data[2] + direction[1]))
                    n_neighbors.append((layer, node.data[1] + direction[0], node.data[2]))
                else:
                    if direction[0] == 0:
                        n_neighbors.append((layer, node.data[1] + 1, node.data[2] + direction[1]))
                        n_neighbors.append((layer, node.data[1] - 1, node.data[2] + direction[1]))
                    else:
                        n_neighbors.append((layer, node.data[1] + direction[0], node.data[2] + 1))
                        n_neighbors.append((layer, node.data[1] + direction[0], node.data[2] - 1))
                for neighbor in n_neighbors:
                    if 0 <= neighbor[1] < self.width and 0 <= neighbor[2] < self.height:
                        neighbors.append(neighbor)
            else:
                for d in DIRECTIONS:
                    neighbor = (layer, node.data[1] + d[0], node.data[2] + d[1])
                    if 0 <= neighbor[1] < self.width and 0 <= neighbor[2] < self.height:
                        neighbors.append(neighbor)
            # 通孔
            if self.is_viable(node.data):
                for layer in range(self.layer_max):
                    neighbor = (layer, x, y)
                    if 0 <= neighbor[1] < self.width and 0 <= neighbor[2] < self.height and layer != node.data[0]:
                        neighbors.append(neighbor)

        no_obs_neighbors = []
        for ng in neighbors:
            if self.solver.obstacle[ng] == -1:
                no_obs_neighbors.append(ng)

        logging.debug(
            f'ori_pos:{node.data} '
            f'from {node.came_from.data if node.came_from else None}'
            f'neighbors:{neighbors}')
        return neighbors

    def is_viable(self, node):
        layer, x, y = node
        node_key = (x, y)
        if self.viable_cache.__contains__(node_key):
            return self.viable_cache[node_key]
        else:
            result = self._is_viable(node)
            self.viable_cache[node_key] = result
            return result

    def _is_viable(self, node):
        layer, x, y = node
        if (self.solver.obstacle[:, x, y] != -1).any():
            return False
        flagx0, flagy0 = int(x / NET_FLAG_RESOLUTION), int(y / NET_FLAG_RESOLUTION)
        for i in range(9):
            direction_x = NINE_DIRECTIONS[i][0]
            direction_y = NINE_DIRECTIONS[i][1]
            gap = (NET_FLAG_RESOLUTION * 0.5 - self.line_width)
            if direction_x != 0:
                if direction_x * (x - (flagx0 + 0.5) * NET_FLAG_RESOLUTION) < gap:
                    continue
            if direction_y != 0:
                if direction_y * (y - (flagy0 + 0.5) * NET_FLAG_RESOLUTION) < gap:
                    continue
            flagx = flagx0 + direction_x
            flagy = flagy0 + direction_y
            if flagx < 0 or flagx >= self.net_flags.shape[1] or flagy < 0 or flagy >= self.net_flags.shape[2]:
                pass
            else:
                for layer in range(self.layer_max):
                    for neighbor_net in np.where(self.net_flags[layer, flagx, flagy, :])[0]:
                        for d in self.net_flag_details[neighbor_net][(layer, flagx, flagy)]:
                            point = d['point']
                            layer1, x1, y1 = point
                            if neighbor_net != self.old_index:
                                if math.hypot(x1 - x, y1 - y) <= self.line_width:
                                    return False
        return True

    def is_goal_reached(self, current: T, goal: T) -> bool:
        """
        Returns true when we can consider that 'current' is the goal.
        The default implementation simply compares `current == goal`, but this
        method can be overwritten in a subclass to provide more refined checks.
        """
        layer0, x0, y0 = current
        layer1, x1, y1 = goal
        if layer0 not in self.end_layers:
            return False
        return (x0, y0) == (x1, y1)

    def jump_node(self, node, prev_node, jump_num=0, jump_max=1e10, **kwargs):
        debug = True
        if jump_num > jump_max:
            if kwargs.get('no_max_return_node'):
                if debug: logging.debug(f'jump_node: jump_max no max return {node}')
                return None, {}
            if debug: logging.debug(f'jump_node: jump_max {node}')
            return node, {}
        jump_num += 1
        layer, x, y = node
        direction = (x - prev_node[1], y - prev_node[2])
        if self.is_goal_reached(node, self.goal):
            if debug: logging.debug(f'jump_node: goal reached {node}')
            return node, {}
        pass_result = self.is_pass(node, direction == (0, 0))
        if not pass_result:
            if debug: logging.debug(f'jump_node: not pass {node}')
            return None, {}
        if direction == (0, 0):  # 竖向迁移
            if debug: logging.debug(f'jump_node: vertical {node}')
            return node, {}
        else:
            if direction[0] == 0 or direction[1] == 0:  # 横向迁移
                if direction[0] != 0:
                    # 右下能走且下不能走， 或右上能走且上不能走
                    '''
                    * 1 0       0 0 0
                    0 → 0       0 0 0
                    * 1 0       0 0 0
                    '''
                    if (self.is_pass((layer, x + direction[0], y + 1)) and not self.is_pass((layer, x, y + 1))) or \
                            (self.is_pass((layer, x + direction[0], y - 1)) and not self.is_pass((layer, x, y - 1))):
                        if debug: logging.debug(f'jump_node 水平方向: {node} {prev_node} {direction}')
                        return node, {}
                else:  # 垂直方向
                    # 右下能走且右不能走，或坐下能走且左不能走
                    '''
                    0 0 0
                    1 ↓ 1
                    0 0 0

                    '''
                    if (self.is_pass((layer, x + 1, y + direction[1])) and not self.is_pass((layer, x + 1, y))) or (
                            self.is_pass((layer, x - 1, y + direction[1])) and not self.is_pass((layer, x - 1, y))):
                        if debug: logging.debug(f'jump_node 垂直方向: {node} {prev_node} {direction}')
                        return node, {}
            elif direction[0] != 0 and direction[1] != 0:  # 斜向迁移
                if (self.is_pass((layer, x - direction[0], y + direction[1])) and
                        not self.is_pass((layer, x - direction[0], y))):
                    if debug: logging.debug(f'jump_node 斜向方向: {node} {prev_node} {direction}')
                    return node, {}
                if (self.is_pass((layer, x + direction[0], y - direction[1])) and
                        not self.is_pass((layer, x, y - direction[1]))):
                    if debug: logging.debug(f'jump_node 斜向方向: {node} {prev_node} {direction}')
                    return node, {}
            else:
                raise Exception('Unknown direction')

        if direction[0] != 0 and direction[1] != 0:
            t1, data1 = self.jump_node((layer, x + direction[0], y), node, jump_num, jump_max, no_max_return_node=True,
                                       **kwargs)
            t2, data2 = self.jump_node((layer, x, y + direction[1]), node, jump_num, jump_max, no_max_return_node=True,
                                       **kwargs)
            if t1 or t2:
                if debug: logging.debug(f'jump_node 斜方向，水平方向有其他跳点: {node} {prev_node} {direction}')
                jps = []
                if t1:
                    jps.append(t1)
                if t2:
                    jps.append(t2)
                return_flag = True
                # 过滤连续的跳点 倾斜状态下
                if kwargs.get('jps'):
                    unique_jp_num = len(jps)
                    ori_jps = kwargs.get('jps')
                    for t in jps:
                        for ori_t in ori_jps:
                            if abs(t[1] - ori_t[1]) <= 1 and abs(t[2] - ori_t[2]) <= 1:
                                unique_jp_num -= 1
                                break
                    if unique_jp_num == 0:
                        kwargs['jps'] = jps
                        return_flag = False
                if return_flag:
                    return node, {'jps': jps}
        # if not self.is_viable((layer, x, y)) and self.is_viable((layer, x + direction[0], y + direction[1])):
        #     new_node = (layer, x + direction[0], y + direction[1])
        #     if debug: logging.debug(f'jump_node: 有via点 {node} {prev_node} {direction}')
        #     return new_node, {}
        if self.is_pass((layer, x + direction[0], y)) or self.is_pass((layer, x, y + direction[1])):
            t, _ = self.jump_node((layer, x + direction[0], y + direction[1]), node, jump_num, jump_max, **kwargs)
            if t:
                if debug: logging.debug(f'jump_node 继续向前搜索跳点: {node} {prev_node} {direction}')
                return t, {}
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
            logging.debug(f'current:{current.data} f:{current.fscore}  target:{goal}')
            if self.is_goal_reached(current.data, goal):
                if self.save_search_path:
                    with open(f'data/search_path/{self.net_id}_search_path.json', 'w') as f:
                        json.dump(search_path, f, cls=NpEncoder)
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            neighbors = []
            current_neighbors = self.neighbors(current)
            for neighbor in current_neighbors:
                logging.debug(f'neighbor:{neighbor}')
                direction = (neighbor[1] - current.data[1], neighbor[2] - current.data[2])
                jp_pos, jp_data = self.jump_node(neighbor, current.data,
                                                 jump_max=jump_max, jps=None, ori_direction=direction)
                if jp_pos is None:
                    logging.debug(f'jump_node is None: NG:{neighbor} CURRENT:{current.data}')
                    continue
                if self.save_search_path: neighbors.append(jp_pos)  # use for log
                jp = searchNodes[jp_pos]
                if jp.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current.data, jp.data
                )
                if tentative_gscore >= jp.gscore:
                    continue
                self.point_check(current, goal, jp, jp_data, jp_pos, neighbor, openSet, tentative_gscore)
            for neighbor in current_neighbors:
                if not self.is_pass(neighbor):
                    logging.debug(f'neighbor is not passable: NG:{neighbor} CURRENT:{current.data}')
                    continue
                jp_pos, jp_data = neighbor, {}
                if jp_pos is None:
                    logging.debug(f'jump_node is None: NG:{neighbor} CURRENT:{current.data}')
                    continue
                if self.save_search_path: neighbors.append(jp_pos)  # use for log
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
                data = {'current': current.data, 'neighbors': neighbors}
                search_path.append(data)
        return None

    def point_check(self, current, goal, jp, jp_data, jp_pos, neighbor, openSet, tentative_gscore):
        jp.came_from = current
        jp.gscore = tentative_gscore
        jp.fscore = tentative_gscore + self.heuristic_cost_estimate(
            jp.data, goal
        )
        if self.recommend_area is not None:
            jp.fscore = jp.fscore - self.line_width * VIA_COST * 3 * self.recommend_area[jp_pos]
        if jp_data.get('jps'):
            jp.jps = jp_data['jps']
        logging.debug(f'current:{current.data} neighbor:{neighbor} jp:{jp.data} f:{jp.fscore} target:{goal}')
        if jp.out_openset:
            jp.out_openset = False
            heappush(openSet, jp)
        else:
            # re-add the node in order to re-sort the heap
            openSet.remove(jp)
            heappush(openSet, jp)
        logging.debug('point_check')
