import cv2
def obs_feature_map_generate(delete_net_id, feature_map, nets, add=0, old_index=-1):
    for net_id in range(len(nets)):
        if net_id == delete_net_id:
            continue
        if old_index != -1 and nets[net_id]['old_index'] == old_index:
            continue
        if nets[net_id].__contains__('path') is False:
            continue
        for point_id in range(len(nets[net_id]['path'])):
            if point_id == 0:
                continue
            point0 = nets[net_id]['path'][point_id - 1]
            point1 = nets[net_id]['path'][point_id]
            if point0[0] != point1[0]:
                continue
            cv2.line(feature_map[point0[0] + add], (point0[2], point0[1]), (point1[2], point1[1]), 1, 2)