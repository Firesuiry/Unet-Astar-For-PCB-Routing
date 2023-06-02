import numpy as np


# feature_map是二维矩阵，障碍物标为0，其他标为1
# 需要在feature_map中找到最大的矩形，使得矩形内的元素都是1
# 返回矩形的左上角和右下角坐标
def find_rect_in_feature_map(feature_map):
    # 1. 生成一个新的矩阵，将feature_map的第一行和第一列都补上0
    # 2. 从第二行第二列开始遍历，如果当前元素为1，则将其值更新为左上角元素的值加1
    # 3. 遍历完后，找到最大值，即为最大矩形的右下角元素的值
    # 4. 从右下角元素开始，向左上角遍历，找到第一个值小于右下角元素的值的元素，即为左上角元素
    # 5. 返回左上角和右下角元素的坐标
    # ********* End *********#
    # 2. 从第二行第二列开始遍历，如果当前元素为1，则将其值更新为左上角元素的值加1
    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            if feature_map[i, j] == 1:
                feature_map[i, j] = feature_map[i - 1, j] + 1
    # 3. 遍历完后，找到最大值，即为最大矩形的右下角元素的值
    max_value = np.max(feature_map)
    # 4. 从右下角元素开始，向左上角遍历，找到第一个值小于右下角元素的值的元素，即为左上角元素
    for i in range(feature_map.shape[0] - 1, -1, -1):
        for j in range(feature_map.shape[1] - 1, -1, -1):
            if feature_map[i, j] < max_value:
                return i, j, i - max_value + 1, j - max_value + 1



if __name__ == '__main__':
    feature_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print(find_rect_in_feature_map(feature_map))
