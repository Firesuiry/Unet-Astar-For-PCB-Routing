import math

import numpy as np

from solver.astar import AStar


def path2key_path(path):
    direction = (0, 0)
    key_path = [path[0], ]
    last_point = path[0]
    for i in range(1, len(path)):
        point = path[i]
        new_direction = (point[0] - last_point[0], point[1] - last_point[1])
        if direction == (0, 0):
            direction = new_direction
        if new_direction != direction:
            key_path.append(point)
            direction = new_direction
        last_point = point
    key_path.append(path[-1])
    return key_path


def key_path2path(key_path):
    path = []
    last_point = key_path[0]
    for i in range(1, len(key_path)):
        point = key_path[i]
        foundPath = list(MazeSolver().astar(last_point, point))
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
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        (x1, y1) = n1.data
        (x2, y2) = n2.data
        return math.hypot(x2 - x1, y2 - y1)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node.data
        neighbors = []
        for d in DIRECTIONS:
            neighbor = (x + d[0], y + d[1])
            neighbors.append(neighbor)

        return neighbors
