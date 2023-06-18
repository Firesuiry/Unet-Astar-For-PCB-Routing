import logging
import math

import numpy as np

from solver.astar import AStar, T

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

'''
1. 起点邻居 生成可行层的周围点
2. 终点只要是可行层的终点都可以
3. 有obs的层不能跨层
'''


class MazeSolver(AStar):
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    # def __init__(self, width, height, solver, old_index, start_layers, end_layers, layer_max, obstacle, via_radius,
    #              clearance,
    #              available_range=None,
    #              **kwargs):
    def __init__(self, width, height, solver, old_index, start_layers, end_layers, layer_max, obstacle, via_radius,
                 clearance,
                 available_range=None,
                 **kwargs):
        self.pass_cache = {}
        self.width = width
        self.height = height
        self.solver = solver
        self.old_index = old_index
        self.start_layers = start_layers
        self.end_layers = end_layers
        self.net_flags = solver.net_flags
        self.net_flag_details = solver.net_flag_details
        self.layer_max = layer_max
        self.available_range = available_range
        self.cross_punish = kwargs.get('cross_punish', 10)
        self.line_width = kwargs.get('line_width', 2.5)
        self.obstacle = (obstacle != self.old_index) * (obstacle != -1)
        self.via_radius = via_radius
        self.clearance = clearance
        self.recommend_area = kwargs.get('recommend_area', None)

        self.passable_cache = {}
        self.viable_cache = {}
        self.speed_test = kwargs.get('speed_test', False)

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (layer1, x1, y1) = n1
        (layer2, x2, y2) = n2
        distance = [abs(x1 - x2), abs(y1 - y2)]
        layer_distance = 0
        if layer1 not in self.end_layers:
            layer_distance = 1
        return max(distance) + (math.sqrt(2) - 1) * min(distance) + layer_distance

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        # 欧式距离
        (layer1, x1, y1) = n1.data
        (layer2, x2, y2) = n2.data
        layer_distance = 0 if layer1 == layer2 else VIA_COST
        gscore = math.hypot(layer_distance, x2 - x1, y2 - y1)
        # 拐弯
        if n1.came_from:
            (layer, x0, y0) = n1.came_from.data
            if (x2 + x0 == x1 and y2 + y0 == y1) or (layer1 != layer2):
                pass
            else:
                gscore += 3
        return gscore

    def is_viable(self, node):
        layer, x, y = node
        node_key = (x, y)
        if self.viable_cache.__contains__(node_key):
            return self.viable_cache[node_key]
        else:
            result = self.is_pass(node, vertical_move=True)
            self.viable_cache[node_key] = result
            return result

    def is_pass(self, node, vertical_move=False):
        key = (node, vertical_move)
        if key not in self.pass_cache:
            self.pass_cache[key] = self._is_pass(node, vertical_move)
        return self.pass_cache[key]

    def _is_pass(self, node, vertical_move=False):
        layer, x, y = node
        if x < 0 or x >= self.width - 1 or y < 0 or y >= self.height - 1:
            return False
        if not vertical_move:
            line_clearance = self.clearance + self.line_width
            return not self.obstacle[layer, x - line_clearance:x + line_clearance + 1,
                       y - line_clearance:y + line_clearance + 1].any()
        else:
            line_clearance = self.via_radius + self.clearance
            return not self.obstacle[:, x - line_clearance:x + line_clearance + 1,
                       y - line_clearance:y + line_clearance + 1].any()

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
            direction = (node.data[1] - node.came_from.data[1], node.data[2] - node.came_from.data[2])
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
                    if 0 <= neighbor[1] < self.width and 0 <= neighbor[2] < self.height and self.is_pass(neighbor):
                        neighbors.append(neighbor)
            else:
                for d in DIRECTIONS:
                    neighbor = (layer, node.data[1] + d[0], node.data[2] + d[1])
                    if 0 <= neighbor[1] < self.width and 0 <= neighbor[2] < self.height and self.is_pass(neighbor):
                        neighbors.append(neighbor)
            # 通孔
            if (self.solver.obstacle[:, x, y] != -1).any():
                ...
            else:
                for layer in range(self.layer_max):
                    neighbor = (layer, x, y)
                    if 0 <= neighbor[1] < self.width and 0 <= neighbor[2] < self.height and self.is_viable(neighbor):
                        neighbors.append(neighbor)

        logging.debug(
            f'ori_pos:{node.data} '
            f'from {node.came_from.data if node.came_from else None} '
            f'neighbors:{neighbors}')
        return neighbors

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
