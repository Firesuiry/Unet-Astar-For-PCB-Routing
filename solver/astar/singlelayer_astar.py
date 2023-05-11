from solver.astar import AStar
from solver.base_solver import BaseSolver


class MazeSolver(AStar):
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, width, height, solver: BaseSolver, old_index, net_id, layer_id=0, available_range=None,
                 **kwargs):
        self.width = width
        self.height = height
        self.solver = solver
        self.old_index = old_index
        self.net_id = net_id
        self.net_flags = solver.net_flags
        self.net_flag_details = solver.net_flag_details
        self.layer_id = layer_id
        self.available_range = kwargs.get('available_range', None)
        self.cross_punish = kwargs.get('cross_punish', 10)

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        distance = [abs(x1 - x2), abs(y1 - y2)]
        return max(distance) + (math.sqrt(2) - 1) * min(distance)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        # 欧式距离
        (x1, y1) = n1.data
        (x2, y2) = n2.data
        gscore = math.hypot(x2 - x1, y2 - y1)

        # 障碍物
        l = [-1, self.old_index]
        if self.solver.obstacle[self.layer_id, x1, y1] not in l or self.solver.obstacle[self.layer_id, x2, y2] not in l:
            gscore += 100
        # 可行范围
        if self.available_range and self.available_range[self.layer_id, x2, y2] == 0:
            gscore += 100

        # 拐弯
        if n1.came_from:
            (x0, y0) = n1.came_from.data
            if x2 + x0 == x1 and y2 + y0 == y1:
                pass
            else:
                gscore += 0.5

        # 交叉
        min_distance = Infinite
        flagx0, flagy0 = int(x2 / NET_FLAG_RESOLUTION), int(y2 / NET_FLAG_RESOLUTION)
        for i in range(9):
            flagx = flagx0 + NINE_DIRECTIONS[i][0]
            flagy = flagy0 + NINE_DIRECTIONS[i][1]
            for neighbor_net in np.where(self.net_flags[self.layer_id, flagx, flagy, :])[0]:
                if neighbor_net == self.old_index:
                    continue
                for d in self.net_flag_details[neighbor_net][(self.layer_id, flagx, flagy)]:
                    point = d['point']
                    distance = math.hypot(x2 - point[0], y2 - point[1])
                    if distance < min_distance:
                        min_distance = distance
        line_width = 2.5
        if min_distance < line_width:
            gscore += self.cross_punish * (1 - (abs(min_distance) / line_width) ** 0.3)

        return gscore

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        neighbors = []
        if node.came_from is None:
            for d in DIRECTIONS:
                neighbor = (node.data[0] + d[0], node.data[1] + d[1])
                if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                    neighbors.append(neighbor)
        else:
            direction = (node.data[0] - node.came_from.data[0], node.data[1] - node.came_from.data[1])
            n_neighbors = [(node.data[0] + direction[0], node.data[1] + direction[1])]
            if direction[0] != 0 and direction[1] != 0:
                n_neighbors.append((node.data[0], node.data[1] + direction[1]))
                n_neighbors.append((node.data[0] + direction[0], node.data[1]))
            else:
                if direction[0] == 0:
                    n_neighbors.append((node.data[0] + 1, node.data[1] + direction[1]))
                    n_neighbors.append((node.data[0] - 1, node.data[1] + direction[1]))
                else:
                    n_neighbors.append((node.data[0] + direction[0], node.data[1] + 1))
                    n_neighbors.append((node.data[0] + direction[0], node.data[1] - 1))
            for neighbor in n_neighbors:
                if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                    neighbors.append(neighbor)
        logging.debug(f'ori_pos:{node.data} neighbors:{neighbors}')
        return neighbors