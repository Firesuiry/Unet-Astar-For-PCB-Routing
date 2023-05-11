from utils.critic_path import path2key_path


def update_net_flag(resolution, net_flags, net_flag_details, path, old_index, net_id):
    for point in path:
        flag_x = int(point[1] / resolution)
        flag_y = int(point[2] / resolution)
        layer_id = point[0]
        net_flags[layer_id, flag_x, flag_y, old_index] = True
        net_flag_details[old_index][(layer_id, flag_x, flag_y)].append({'point': point, 'net_id': net_id})
