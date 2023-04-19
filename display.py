import sys

import pygame
from pygame import QUIT
from multiprocessing import shared_memory
import numpy as np
import pickle

pygame.init()

main_surface = pygame.display.set_mode((1000, 1000), 0, 32)
pygame.display.set_caption("我的pygame游戏")

# 底部surface
bottom_surface = pygame.Surface((1000, 1000))
# 一个高亮surface 用于显示数据
highlight_surface = pygame.Surface((1000, 1000))
# 每层一个surface
layer_surface = []
layer_bottom_surface = []


def get_data():
    global data
    try:
        shm = shared_memory.SharedMemory(name='solver_data')
    except FileNotFoundError:
        return None
    data_len = np.frombuffer(shm.buf, dtype=np.int64, count=1)[0]
    binary_data = shm.buf[8:8 + data_len]
    data = pickle.loads(binary_data)
    try:
        shm.close()
        shm.unlink()
    except:
        pass


def data_display():
    global data
    layer_surface.clear()
    for i in range(data['layer_num']):
        surface = pygame.Surface((1000, 1000))
        layer_surface.append(surface)
        surface.fill((100, 100, 100))
        main_surface.blit(surface, (0, 0))
    pygame.display.flip()


get_data()
data_display()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
