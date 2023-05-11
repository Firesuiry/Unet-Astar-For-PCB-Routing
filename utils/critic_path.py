import logging
import math

import numpy as np

from solver.astar import AStar


def path2key_path(path):
    direction = (0, 0, 0)
    key_path = [path[0], ]
    last_point = path[0]
    for i in range(1, len(path)):
        point = path[i]
        new_direction = (point[0] - last_point[0], point[1] - last_point[1], point[2] - last_point[2])
        # print(f'new_direction:{new_direction} direction:{direction} point:{point} last_point:{last_point}')
        if direction == (0, 0, 0):
            direction = new_direction
        if new_direction != direction:
            key_path.append(last_point)
            direction = new_direction
        last_point = point
    key_path.append(path[-1])
    return key_path


def key_path2path(key_path):
    path = []
    last_point = key_path[0]
    solver = MazeSolver()
    for i in range(1, len(key_path)):
        point = key_path[i]
        foundPath = list(solver.astar(last_point, point))
        logging.debug(f'last_point:{last_point} point:{point} foundPath:{foundPath}')
        if path and path[-1] == foundPath[0]:
            path += foundPath[1:]
        else:
            path += foundPath
        last_point = point
    return path


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


class MazeSolver(AStar):
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self):
        pass

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (layer1, x1, y1) = n1
        (layer2, x2, y2) = n2
        return math.hypot(bool(layer1 - layer2), x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        (layer1, x1, y1) = n1.data
        (layer2, x2, y2) = n2.data
        return math.hypot(bool(layer1 - layer2), x2 - x1, y2 - y1)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        layer, x, y = node.data
        neighbors = []
        for d in DIRECTIONS:
            neighbor = (layer, x + d[0], y + d[1])
            neighbors.append(neighbor)
        for z in [-1, 1]:
            neighbor = (layer + z, x, y)
            neighbors.append(neighbor)
        return neighbors

if __name__ == '__main__':
    a = [(0, 141, 16), (0, 140, 17), (0, 139, 18), (0, 138, 19), (0, 137, 20), (0, 136, 21), (0, 135, 22), (0, 134, 23), (0, 133, 24), (0, 132, 25), (0, 131, 26), (0, 130, 27), (0, 129, 28), (0, 128, 29), (0, 127, 30), (0, 126, 31), (1, 126, 31), (1, 127, 32)]
    b = path2key_path(a)
    ...
