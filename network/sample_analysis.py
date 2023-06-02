import numpy as np
import pickle


def sample_analysis(net_id=0):
    base_path = r'data/net_path'
    recommend_area_path = f'{base_path}/net{net_id}_recommend_area.pickle'
    net_path_0_path = f'{base_path}/net{net_id}_True_found_path.pickle'
    net_path_1_path = f'{base_path}/net{net_id}_False_found_path.pickle'
    # load recommend_area
    with open(recommend_area_path, 'rb') as f:
        recommend_area = pickle.load(f)
    # load net_path_0
    with open(net_path_0_path, 'rb') as f:
        net_path_0 = list(pickle.load(f))
    # load net_path_1
    with open(net_path_1_path, 'rb') as f:
        net_path_1 = list(pickle.load(f))
    # analysis
    # calculate the mean of recommend_area in path_0
    recommend_area_0 = []
    for point in net_path_0:
        recommend_area_0.append(recommend_area[point])
    recommend_area_0 = np.array(recommend_area_0)
    recommend_area_0_mean = np.mean(recommend_area_0)
    # calculate the mean of recommend_area in path_1
    recommend_area_1 = []
    for point in net_path_1:
        recommend_area_1.append(recommend_area[point])
    recommend_area_1 = np.array(recommend_area_1)
    recommend_area_1_mean = np.mean(recommend_area_1)

    # draw recommend_area in every layer and plot path_0 and path_1 on it
    layer_num = recommend_area.shape[0]
    net_path_array_0 = np.array(net_path_0)
    net_path_array_0_layer_0 = net_path_array_0[net_path_array_0[:, 0] == 0]
    net_path_array_0_layer_1 = net_path_array_0[net_path_array_0[:, 0] == 1]
    net_path_array_0_layers = [net_path_array_0_layer_0, net_path_array_0_layer_1]
    net_path_array_1 = np.array(net_path_1)
    net_path_array_1_layer_0 = net_path_array_1[net_path_array_1[:, 0] == 0]
    net_path_array_1_layer_1 = net_path_array_1[net_path_array_1[:, 0] == 1]
    net_path_array_1_layers = [net_path_array_1_layer_0, net_path_array_1_layer_1]
    for layer_id in range(layer_num):
        # draw recommend_area and plot path_0 and path_1 on it
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(f'net {net_id} layer {layer_id}')
        plt.imshow(recommend_area[layer_id])
        plt.scatter(net_path_array_0_layers[layer_id][:, 2], net_path_array_0_layers[layer_id][:, 1], s=1, c='r',alpha=0.3)
        plt.scatter(net_path_array_1_layers[layer_id][:, 2], net_path_array_1_layers[layer_id][:, 1], s=1, c='b',alpha=0.3)
        plt.show()
        # show the figure until press any key

    ...
