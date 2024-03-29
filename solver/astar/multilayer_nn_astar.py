import math
from abc import ABC, abstractmethod
from heapq import heappush, heappop
from typing import Iterable, Union, TypeVar, Generic
import logging

from solver.astar import AStar

# infinity as a constant
Infinite = float("inf")

# introduce generic type
T = TypeVar("T")
from solver.astar.multilayer_astar import MazeSolver, VIA_COST


class MazeNNSolver(MazeSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.liner_nn_power = kwargs.get('liner_nn_power', 3)
        self.multi_nn_power = kwargs.get('multi_nn_power', 3)
        assert self.liner_nn_power >= 0
        assert 0 <= self.multi_nn_power <= 1

    def astar(
            self, start: T, goal: T, reversePath: bool = False
    ) -> Union[Iterable[T], None]:
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = AStar.SearchNodeDict()
        distance = max(math.hypot(start[1] - goal[1], start[2] - goal[2]) * 0.1, 5)
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal)
        )
        openSet: list = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            logging.debug(f'current:{current.data} f:{current.fscore}  target:{goal}')
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath), self.search_area(searchNodes)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current)):
                if neighbor.closed:
                    continue
                if not self.is_pass(neighbor.data):
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current, neighbor
                )
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(
                    neighbor.data, goal
                )
                if self.recommend_area is not None:
                    recommend_score = self.recommend_area[neighbor.data]
                    neighbor.fscore = (neighbor.fscore - self.liner_nn_power * recommend_score) * (
                            1 - self.multi_nn_power * recommend_score)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None, None
