import logging
import math

import numpy as np

from solver.astar import AStar, T
from typing import Iterable, Union, TypeVar, Generic
from heapq import heappush, heappop

from utils.circle_generate import circle_generate, put_circle

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


class RectDetailAstar:
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    class SearchNode:
        """Representation of a search node"""

        __slots__ = ("data", "gscore", "fscore", "closed", "came_from", "out_openset")

        def __init__(
                self, data: T, gscore: float = Infinite, fscore: float = Infinite
        ) -> None:
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b: "RectDetailAstar.SearchNode") -> bool:
            return self.fscore < b.fscore

        def __str__(self):
            return self.data.__str__()

    class SearchNodeDict(dict):
        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def __init__(self, width, height, old_index, start_layers, end_layers, layer_max,
                 available_range, start_range, end_range, obstacle, clearance, line_width,
                 via_radius):
        self.circle_cache = None
        self.pass_cache = {}
        self.width = width
        self.height = height
        self.old_index = old_index
        self.start_layers = start_layers
        self.end_layers = end_layers
        self.layer_max = layer_max
        self.available_range = available_range
        self.x0, self.x1, self.y0, self.y1 = available_range
        self.start_range = start_range
        self.end_range = end_range  # 终点的范围 x0 x1 y0 y1
        self.obstacle = obstacle
        self.clearance = clearance
        self.line_width = line_width
        self.via_radius = via_radius
        self.directions = DIRECTIONS
        for i in range(len(self.directions)):
            d = self.directions[i]
            for j in range(len(d)):
                d[j] = int(d[j] / np.linalg.norm(d) * self.line_width / 2)
            self.directions[i] = d
        self.zhi_step = self.directions[0][0]
        self.xie_step = self.directions[1][0]
        self.debug = True
        self.searched_area = np.zeros((self.layer_max, self.width + 1, self.height + 1), dtype=bool)

        if self.start_layers is None:
            self.start_layers = list(range(self.layer_max))
        if self.end_layers is None:
            self.end_layers = list(range(self.layer_max))

    def heuristic_cost_estimate(self, n1):
        """computes the 'direct' distance between two (x,y) tuples"""
        (layer1, n1x1, n1y1) = n1
        # calculate the distance between n1 and self.end_range
        x0, x1, y0, y1 = self.end_range
        x_distance = y_distance = 0
        if n1x1 < x0:
            x_distance += x0 - n1x1
        elif n1x1 > x1:
            x_distance += n1x1 - x1
        if n1y1 < y0:
            y_distance += y0 - n1y1
        elif n1y1 > y1:
            y_distance += n1y1 - y1

        distance = [x_distance, y_distance]
        layer_distance = 0
        if layer1 not in self.end_layers:
            layer_distance = 10
        return (max(distance) + (math.sqrt(2) - 1) * min(distance) + layer_distance) * 1.5

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        # 欧式距离
        (layer1, x1, y1) = n1.data
        (layer2, x2, y2) = n2.data
        layer_distance = 0 if layer1 == layer2 else VIA_COST
        gscore = math.hypot(x2 - x1, y2 - y1) + layer_distance

        # 拐弯
        if n1.came_from:
            (layer, x0, y0) = n1.came_from.data
            direction0 = (0 if x0 == x1 else (x0 - x1) / abs(x0 - x1),
                          0 if y0 == y1 else (y0 - y1) / abs(y0 - y1))
            direction1 = (0 if x1 == x2 else (x1 - x2) / abs(x1 - x2),
                          0 if y1 == y2 else (y1 - y2) / abs(y1 - y2))
            if direction0 == direction1 or (layer1 != layer2):
                pass
            else:
                gscore += 10

        # 交叉
        if not self.is_pass(n2.data, vertical_move=layer1 != layer2):
            gscore += 0

        return gscore

    def is_pass(self, node, vertical_move=False):
        key = (node, vertical_move)
        if key not in self.pass_cache:
            self.pass_cache[key] = self._is_pass(node, vertical_move)
        return self.pass_cache[key]

    def _is_pass(self, node, vertical_move=False):
        layer, x, y = node
        if x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1:
            return False
        if not vertical_move:
            line_clearance = self.clearance + self.line_width
            return not self.obstacle[layer, x - line_clearance - self.x0:x + line_clearance + 1 - self.x0,
                       y - line_clearance - self.y0:y + line_clearance + 1 - self.y0].any()
        else:
            line_clearance = self.via_radius + self.clearance
            return not self.obstacle[:, x - line_clearance - self.x0:x + line_clearance + 1 - self.x0,
                       y - line_clearance - self.y0:y + line_clearance + 1 - self.y0].any()

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        neighbors = []
        # 起点
        if node.came_from is None:
            for d in self.directions:
                for layer in self.start_layers:
                    neighbor = (layer, node.data[1] + d[0], node.data[2] + d[1])
                    if 0 <= neighbor[1] - self.x0 < self.width and 0 <= neighbor[2] - self.y0 < self.height:
                        neighbors.append(neighbor)
        else:
            # 同层内平移
            layer, x, y = node.data
            direction = [node.data[1] - node.came_from.data[1], node.data[2] - node.came_from.data[2]]
            if direction[0] != 0 and direction[1] != 0:
                direction = [direction[0] * self.xie_step / abs(direction[0]),
                             direction[1] * self.xie_step / abs(direction[1])]
            if direction[0] == 0 and direction[1] != 0: direction[1] = direction[1] * self.zhi_step / abs(direction[1])
            if direction[1] == 0 and direction[0] != 0: direction[0] = direction[0] * self.zhi_step / abs(direction[0])
            direction = (int(direction[0]), int(direction[1]))
            if direction != (0, 0):
                # 向前大步搜索
                point = node.data
                for _ in range(20):
                    point = (point[0], point[1] + direction[0], point[2] + direction[1])
                    if not self.is_pass(point):
                        break
                    neighbors.append(point)
                n_neighbors = [(layer, node.data[1] + direction[0], node.data[2] + direction[1])]
                if direction[0] != 0 and direction[1] != 0:
                    n_neighbors.append((layer, node.data[1], node.data[2] + direction[1]))
                    n_neighbors.append((layer, node.data[1] + direction[0], node.data[2]))
                else:
                    if direction[0] == 0:
                        n_neighbors.append((layer, node.data[1] + self.line_width, node.data[2] + direction[1]))
                        n_neighbors.append((layer, node.data[1] - self.line_width, node.data[2] + direction[1]))
                    else:
                        n_neighbors.append((layer, node.data[1] + direction[0], node.data[2] + self.line_width))
                        n_neighbors.append((layer, node.data[1] + direction[0], node.data[2] - self.line_width))
                for neighbor in n_neighbors:
                    neighbors.append(neighbor)
            else:
                for d in self.directions:
                    neighbor = (layer, node.data[1] + d[0], node.data[2] + d[1])
                    neighbors.append(neighbor)
            # 通孔
            if not self.is_pass(node.data, vertical_move=True):
                ...
            else:
                for layer in range(self.layer_max):
                    neighbor = (layer, x, y)
                    neighbors.append(neighbor)

        new_neighbors = []
        for neighbor in neighbors:
            if 0 <= neighbor[1] - self.x0 < self.width and 0 <= neighbor[2] - self.y0 < self.height:
                if not self.searched_area[neighbor[0], neighbor[1] - self.x0, neighbor[2] - self.y0]:
                    new_neighbors.append(neighbor)

        if self.debug: logging.info(
            f'ori_pos:{node.data} '
            f'from {node.came_from.data if node.came_from else None} '
            f'neighbors:{neighbors}')
        return neighbors

    def is_goal_reached(self, current: T) -> bool:
        """
        Returns true when we can consider that 'current' is the goal.
        The default implementation simply compares `current == goal`, but this
        method can be overwritten in a subclass to provide more refined checks.
        """
        layer0, x0, y0 = current
        if layer0 not in self.end_layers:
            return False
        if self.end_range[0] - self.line_width <= x0 < self.end_range[1] + self.line_width and \
                self.end_range[2] - self.line_width < y0 < self.end_range[3] + self.line_width:
            return True
        return False

    def astar(
            self, reversePath: bool = False
    ) -> Union[Iterable[T], None]:
        openSet: list = []
        展开节点数 = 0
        额外展开节点数 = 0
        for layer in self.start_layers:
            for x in range(self.start_range[0], self.start_range[1] + 1):
                for y in range(self.start_range[2], self.start_range[3] + 1):
                    start = (layer, x, y)
                    if self.is_goal_reached(start):
                        return [start]
                    searchNodes = AStar.SearchNodeDict()
                    startNode = searchNodes[start] = AStar.SearchNode(
                        start, gscore=0.0, fscore=self.heuristic_cost_estimate(start)
                    )
                    heappush(openSet, startNode)

        target_node = None
        while openSet:
            current = heappop(openSet)
            展开节点数 += 1
            if target_node is not None: 额外展开节点数 += 1
            if self.debug: logging.info(f'current:{current.data} f:{current.fscore}')
            if self.is_goal_reached(current.data):
                if target_node is None or target_node.fscore > current.fscore:
                    target_node = current
                    if self.debug: logging.info(f'找到目标:{target_node.data} f:{target_node.fscore}')
                    break
            current.out_openset = True
            current.closed = True
            if target_node is not None:
                continue
            neighbors = self.neighbors(current)
            if self.debug: logging.info(f'neighbors:{neighbors}')
            for neighbor in map(lambda n: searchNodes[n], neighbors):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current, neighbor
                )
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(neighbor.data)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
                self.update_searched_area(neighbor.data)
        if target_node is not None:
            logging.info(f'展开节点数:{展开节点数}, 额外展开节点数:{额外展开节点数}')
            return self.reconstruct_path(target_node, reversePath=reversePath), self.searched_area
        return None

    def reconstruct_path(self, last: SearchNode, reversePath=False) -> Iterable[T]:
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from

        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def update_searched_area(self, data):
        layer, x, y = data
        circle = self.get_circle()
        # set true in the searched area if the distance between the point in the searched area and the data smaller than line_width/2
        hlw = self.line_width // 2
        lw = self.line_width
        x0 = self.x0
        y0 = self.y0
        area = self.searched_area[layer]
        put_circle(area, circle, hlw - 1, x, x0, y, y0)
        # c = self.searched_area[layer][x - self.x0 - lw:x - self.x0 + lw + 1, y - self.y0 - lw:y - self.y0 + lw + 1]

    def get_circle(self):
        if hasattr(self, 'circle_cache') and self.circle_cache is not None:
            return self.circle_cache
        lw = self.line_width
        circle = circle_generate(lw - 2)
        self.circle_cache = circle
        return circle
