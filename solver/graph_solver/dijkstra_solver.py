import logging
from heapq import heappush, heappop

Infinite = float("inf")


class DijkstraSolver:
    class SearchNode:
        """Representation of a search node"""

        __slots__ = ("data", "gscore", "closed", "came_from", "out_openset", "in_range")

        def __init__(
                self, data, gscore: float = Infinite
        ) -> None:
            self.data = data
            self.gscore = gscore
            self.closed = False
            self.out_openset = True
            self.came_from = None
            self.in_range = None

        def __lt__(self, b) -> bool:
            return self.gscore < b.gscore

        def __str__(self):
            return self.data.__str__()

    class SearchNodeDict(dict):
        def __missing__(self, k):
            v = DijkstraSolver.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def __init__(self, rects, edges, net_path_data, clearance, line_width):
        self.goal_exact_pos = None
        self.start_exact_pos = None
        self.rects = rects
        self.edges = edges
        self.net_path_data = net_path_data
        self.clearance = clearance
        self.line_width = line_width

    def is_goal_reached(self, current, goal):
        return current == goal

    def reconstruct_path(self, last: SearchNode, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from

        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def neighbors(self, node):
        return list(self.rects[node.data]['connects'].keys())

    def distance_between(self, node, next_node):
        distance = 0
        # 计算edge拥挤度
        key = (node.data, next_node.data) if node.data <= next_node.data else (next_node.data, node.data)
        edge_net_limit = self.edges[key] // (self.line_width + self.clearance)
        edge_net_num = self.net_path_data['edge_net_num'].get(key, 0) + 1
        if edge_net_num > edge_net_limit * 0.2:
            distance += (edge_net_num - edge_net_limit * 0.1) * 10
        if edge_net_num > edge_net_limit * 0.5:
            distance += (edge_net_num - edge_net_limit * 0.5) * 50
        if edge_net_num > edge_net_limit:
            distance += (edge_net_num - edge_net_limit) * 1000
        # 计算node拥挤度
        rect_area = self.rects[next_node.data]['blank_area']
        node_net_limit = rect_area // (self.line_width + self.clearance) ** 2
        node_net_num = self.net_path_data['rect_net_num'][next_node.data] + 1
        area_limit_rate = 0.1
        if node_net_num > node_net_limit * area_limit_rate:
            distance += (node_net_num - node_net_limit * area_limit_rate) * 10
        if node_net_num > node_net_limit:
            distance += (node_net_num - node_net_limit) * 1000
        # 计算在rect内的距离
        # 目标区域是node到next_node的链接矩形区域
        target_rect = self.rects[next_node.data]['connect_range'].get(node.data)
        if target_rect is None:  # 说明node在next_node内部
            target_rect = self.rects[node.data]['connect_range'][next_node.data]
        if node.came_from is None:
            # from start_exact_pos
            # calculate the min distance between start_exact_pos and target_rect
            # start_exact_pos is [x, y]
            start_exact_pos = self.start_exact_pos
            # target_rect is [x0, x1, y0, y1]
            x_distance = y_distance = 0
            if start_exact_pos[0] < target_rect[0]:
                x_distance += target_rect[0] - start_exact_pos[0]
            elif start_exact_pos[0] > target_rect[1]:
                x_distance += start_exact_pos[0] - target_rect[1]
            if start_exact_pos[1] < target_rect[2]:
                y_distance += target_rect[2] - start_exact_pos[1]
            elif start_exact_pos[1] > target_rect[3]:
                y_distance += start_exact_pos[1] - target_rect[3]
            # distance equal to max_distance + (sqrt(2) - 1) * min_distance
            distance += (max(x_distance, y_distance) + (1.414 - 1) * min(x_distance, y_distance)) * 3
            # 从首个rect尽快转移，降低拥挤度
        else:
            start_rect = self.rects[node.came_from.data]['connect_range'][node.data]
            # calculate the min distance between start_rect and target_rect
            x_distance = y_distance = 0
            if start_rect[1] < target_rect[0]:
                x_distance += target_rect[0] - start_rect[0]
            elif start_rect[0] > target_rect[1]:
                x_distance += start_rect[0] - target_rect[1]
            if start_rect[3] < target_rect[2]:
                y_distance += target_rect[2] - start_rect[3]
            elif start_rect[2] > target_rect[3]:
                y_distance += start_rect[2] - target_rect[3]
            # distance equal to max_distance + (sqrt(2) - 1) * min_distance
            distance += max(x_distance, y_distance) + (1.414 - 1) * min(x_distance, y_distance)
        return distance

    def solve(self, start, goal, start_exact_pos, goal_exact_pos):
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = DijkstraSolver.SearchNodeDict()
        startNode = searchNodes[start] = DijkstraSolver.SearchNode(
            start, gscore=0.0
        )
        goalNode = None
        self.start_exact_pos = start_exact_pos
        self.goal_exact_pos = goal_exact_pos
        openSet: list = []
        heappush(openSet, startNode)
        extra_search_num = 0
        while openSet:
            current = heappop(openSet)
            logging.debug(f'current:{current.data} f:{current.gscore}  target:{goal}')
            if goalNode is None and self.is_goal_reached(current.data, goal):
                goalNode = current
            if goalNode is not None:
                extra_search_num += 1
                if extra_search_num > 30:
                    break
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current, neighbor
                )
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        if goalNode is None:
            return None
        return self.reconstruct_path(goalNode, False)
