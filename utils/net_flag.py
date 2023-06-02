import numpy as np

from utils.critic_path import path2key_path


def update_net_flag(resolution, path, old_index, obstacle, line_width, via_radius):
    layer_num = obstacle.shape[0]
    for i in range(len(path)):
        point = path[i]
        flag_x = int(point[1] / resolution)
        flag_y = int(point[2] / resolution)
        layer_id = point[0]
        # obstacle is a np array, plot a circle on it, center is point[1:3], radius is line_width//2
        plot_circle(obstacle[layer_id], point, line_width, old_index)
        # plot via
        if i < len(path) - 1:
            next_point = path[i + 1]
            if next_point[0] != point[0]:
                for layer in range(layer_num):
                    plot_circle(obstacle[layer], point, via_radius, old_index)


def plot_circle(obstacle, point, line_width, old_index):
    '''
    draw a circle on obstacle,  radius is line_width//2
    :param obstacle: np array
    :param point: center of circle
    :param line_width:
    :return: obstacle
    '''
    center_x = point[1]
    center_y = point[2]
    for i in range(line_width + 1):
        for j in range(line_width + 1):
            if (i - line_width // 2) ** 2 + (j - line_width // 2) ** 2 <= line_width ** 2 // 4:
                obstacle[int(center_x) + i - line_width // 2, int(center_y) + j - line_width // 2] = old_index
    return obstacle


# test plot circle
def test_plot_circle():
    a = np.zeros((7, 7))
    a = plot_circle(a, [0, 3, 3], 2, 1)
    print(a)


if __name__ == '__main__':
    test_plot_circle()
