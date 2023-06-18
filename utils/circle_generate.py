import math

import numpy as np


def circle_generate(lw):
    circle = np.zeros((lw // 2 * 2 + 1, lw // 2 * 2 + 1), dtype=np.bool)
    for x0 in range(lw // 2 * 2 + 1):
        for y0 in range(lw // 2 * 2 + 1):
            if math.hypot(x0 - lw // 2, y0 - lw // 2) <= lw / 2:
                circle[x0][y0] = True
    return circle


def put_circle(area, circle, hlw, x, x0, y, y0):
    cx0, cy0 = 0, 0
    cx1, cy1 = circle.shape[0] + 1, circle.shape[1] + 1
    ax0 = x - x0 - hlw
    ax1 = x - x0 + hlw + 1
    ay0 = y - y0 - hlw
    ay1 = y - y0 + hlw + 1
    ay1 = max(ay1, 0)
    ax1 = max(ax1, 0)
    ax0 = min(ax0, area.shape[0])
    ay0 = min(ay0, area.shape[1])
    if ax0 < 0:
        cx0 = -ax0
        ax0 = 0
    if ax1 > area.shape[0]:
        cx1 = circle.shape[0] - (ax1 - area.shape[0])
        ax1 = area.shape[0]
    if ay0 < 0:
        cy0 = -ay0
        ay0 = 0
    if ay1 > area.shape[1]:
        cy1 = circle.shape[1] - (ay1 - area.shape[1])
        ay1 = area.shape[1]
    cx1 = max(cx1, 0)
    cy1 = max(cy1, 0)
    area[ax0:ax1, ay0:ay1] = np.logical_or(area[ax0:ax1, ay0:ay1], circle[cx0:cx1, cy0:cy1])
