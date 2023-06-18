import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img(img, convert=True, path=None):
    # copy img to img0
    if convert:
        img0 = img.copy().astype(np.uint8) * 255
    else:
        img0 = img.copy()
    # rotate img0
    img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(path, img0)


def is_point_inside_rects(point, rects):
    x, y = point
    for rect in rects:
        x0, x1, y0, y1 = rect
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def area_extract(img, line_width=1, clearance=1, path=None):
    # 通过plt展示原始图像
    # show_img(img)
    # get the height and width of img
    width, height = img.shape
    rects, new_rects = [], []
    # rect = [x0, x1, y0, y1]
    # calculate the area of img
    all_area = height * width
    rects_area = 0
    # -2 available -1 origin obs 0 and bigger is the index of rect
    rect_flag = np.zeros_like(img) - 2
    rect_flag[img == 1] = -1
    new_rect_flag = rect_flag.copy()
    img0 = img.copy().astype(np.uint8) * 255
    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    rects_area += np.sum(img == 1)
    g_num = 0
    未搜索到次数 = 0
    while rects_area < 1 * all_area and g_num < 1e3 and 未搜索到次数 < 1e2:
        g_num += 1
        未搜索到次数 += 1
        # random choose a point in img out of rects
        find_flag = False
        x, y = 0, 0
        for i in range(20):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            if rect_flag[x, y] == -2:
                find_flag = True
                break
        if not find_flag:
            points = np.where(rect_flag == -2)
            index = np.random.randint(0, len(points[0]))
            x = points[0][index]
            y = points[1][index]
        # base on the point, get the rect
        rect = [x, x, y, y]
        run_flag = True
        while run_flag:
            run_flag = False
            ds = [-1, 1, -1, 1]
            for i in range(4):
                test_xy = rect[i] + ds[i]
                if i < 2:
                    if test_xy < 0 or test_xy >= width:
                        continue
                    if (rect_flag[test_xy, rect[2]:rect[3] + 1] != -2).any():
                        continue
                    rect[i] = test_xy
                else:
                    if test_xy < 0 or test_xy >= height:
                        continue
                    if (rect_flag[rect[0]:rect[1] + 1, test_xy] != -2).any():
                        continue
                    rect[i] = test_xy
                run_flag = True
        rect_area = (rect[1] - rect[0]) * (rect[3] - rect[2])
        if (rect[1] - rect[0])/2 < line_width + clearance or (rect[3] - rect[2])/2 < line_width + clearance:
            continue
        rects_area += rect_area
        rect_flag[rect[0]:rect[1] + 1, rect[2]:rect[3] + 1] = len(rects)
        rect_data = {
            'x0': rect[0],
            'x1': rect[1],
            'y0': rect[2],
            'y1': rect[3],
            'width': rect[1] - rect[0],
            'height': rect[3] - rect[2],
            'width_line_limit': (rect[1] - rect[0]) // (line_width + clearance),
            'height_line_limit': (rect[3] - rect[2]) // (line_width + clearance),
            'index': len(rects),
        }
        rects.append(rect_data)
        未搜索到次数 = 0

    for rect in rects:
        rect_pos = rect['x0'], rect['x1'], rect['y0'], rect['y1']
        min_multi_rate = 3
        limit = (line_width + clearance)
        # if (rect_pos[1] - rect_pos[0]) / min_multi_rate < line_width + clearance or \
        #         (rect_pos[3] - rect_pos[2]) / min_multi_rate < line_width + clearance:
        #     continue
        new_rects.append(rect)
        new_rect_flag[rect_pos[0]:rect_pos[1] + 1, rect_pos[2]:rect_pos[3] + 1] = len(new_rects) - 1
        cv2.rectangle(img0, (rect_pos[2], rect_pos[0]), (rect_pos[3], rect_pos[1]), (0, 0, 255), 1)
    # show_img(img0, convert=False, path=path)
    return new_rects, new_rect_flag


if __name__ == "__main__":
    # read D:\develop\PCB\img\obstacle\0.npy
    obstacle = np.load(r'D:\develop\PCB\img\obstacle\0.npy')
    obstacle = obstacle[0]
    area_extract(obstacle, 4, 4)
