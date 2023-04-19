import sys, random

import pygame
from pygame.color import THECOLORS as COLORS
from pygame.locals import *


def draw_background():
    # white background
    screen.fill(COLORS['white'])
    pygame.draw.rect(screen, COLORS['black'], (-100, GAME_SIZE[1], 3000, 200), 0)


def draw_wall():
    for xy in wall_list:
        pygame.draw.rect(screen, COLORS['darkgray'],
                         (xy[0] - WALL_WIDTH / 2, xy[1] - WALL_WIDTH / 2, WALL_WIDTH, WALL_HEIGHT), 0)


def draw_snake():
    head = snake_list[0]
    pygame.draw.circle(screen, COLORS['darkred'], (head[0], head[1]), int(SNAKE_WIDTH / 2), 0)
    for xy in snake_list[1:]:
        pygame.draw.rect(screen, COLORS['darkred'],
                         (xy[0] - SNAKE_WIDTH / 2, xy[1] - SNAKE_WIDTH / 2, SNAKE_WIDTH, SNAKE_HEIGHT), 2)


def draw_food():
    for xyz in food_list:
        pygame.draw.rect(screen, FOOD_COLORS[xyz[2] - 1],
                         (xyz[0] - FOOD_WIDTH / 2, xyz[1] - FOOD_WIDTH / 2, FOOD_WIDTH, FOOD_HEIGHT), 0)


def draw_context():
    txt = FONT_M.render('Snake length: ' + str(len(snake_list) - 1), True, COLORS['lightblue'])
    x, y = 10, GAME_SIZE[1] + (int((SIZE[1] - GAME_SIZE[1]) / 2))
    y = int(y - FONT_M.size('Count')[1] / 2)
    screen.blit(txt, (x, y))


def draw_pause():
    s = pygame.Surface(SIZE, pygame.SRCALPHA)
    s.fill((255, 255, 255, 220))
    screen.blit(s, (0, 0))
    txt = FONT_M.render('PAUSE', True, COLORS['darkgray'])
    x, y = SIZE[0] / 2, SIZE[1] / 2
    x, y = int(x - FONT_M.size('PAUSE')[0] / 2), int(y - FONT_M.size('PAUSE')[1] / 2)
    screen.blit(txt, (x, y))


def draw_dead():
    s = pygame.Surface(SIZE, pygame.SRCALPHA)
    s.fill((255, 255, 255, 240))
    screen.blit(s, (0, 0))
    txt = FONT_M.render('YOU DEAD', True, COLORS['black'])
    x, y = SIZE[0] / 2, SIZE[1] / 2
    x, y = int(x - FONT_M.size('YOU DEAD')[0] / 2), int(y - FONT_M.size('YOU DEAD')[1] / 2)
    screen.blit(txt, (x, y))


def rect_cover(rect1, rect2):
    left1 = int(rect1[0])
    right1 = int(rect1[0] + rect1[2])
    up1 = int(rect1[1])
    down1 = int(rect1[1] + rect1[3])
    left2 = int(rect2[0])
    right2 = int(rect2[0] + rect2[2])
    up2 = int(rect2[1])
    down2 = int(rect2[1] + rect2[3])

    if not (right2 <= left1 or left2 >= right1 or down2 <= up1 or up2 >= down1):
        return True
    return False


def add_food():
    while (True):
        xyz = [random.choice(X_LIST), random.choice(Y_LIST), random.choice([1, 2, 3, 4])]
        if xyz not in wall_list:
            food_list.append(xyz)
            break


def add_body(length=1):
    for c in range(length):
        # 尾巴加一节
        last2, last1 = snake_list[-2], snake_list[-1]
        if last2[0] == last1[0]:  # 竖着的两段
            if last2[1] > last1[1]:  # 朝下
                snake_list.append([last1[0], last1[1] - SNAKE_WIDTH])
            else:
                snake_list.append([last1[0], last1[1] + SNAKE_WIDTH])
        else:  # 横着的两段
            if last2[0] > last1[0]:  # 朝右
                snake_list.append([last1[0] - SNAKE_WIDTH, last1[1]])
            else:
                snake_list.append([last1[0] + SNAKE_WIDTH, last1[1]])


def check_food():
    # 头与食物
    first = snake_list[0]
    snake_head_rect = (first[0] - SNAKE_WIDTH / 2, first[1] - SNAKE_WIDTH / 2, SNAKE_WIDTH, SNAKE_HEIGHT)
    for i in range(len(food_list)):
        xyz = food_list[i]
        food_rect = (xyz[0] - FOOD_WIDTH / 2, xyz[1] - FOOD_WIDTH / 2, FOOD_WIDTH, FOOD_HEIGHT)
        if rect_cover(snake_head_rect, food_rect):
            add_body(xyz[2])
            del food_list[i]
            return True
    return False


def check_dead():
    first = snake_list[0]
    snake_head_rect = (first[0] - SNAKE_WIDTH / 2, first[1] - SNAKE_WIDTH / 2, SNAKE_WIDTH, SNAKE_HEIGHT)
    # 头与边缘
    if first[0] < 0 or first[0] > GAME_SIZE[0] or first[1] < 0 or first[1] > GAME_SIZE[1]:
        return True
    # 头与墙壁
    for xy in wall_list:
        wall_rect = (xy[0] - WALL_WIDTH / 2, xy[1] - WALL_WIDTH / 2, WALL_WIDTH, WALL_HEIGHT)
        if rect_cover(snake_head_rect, wall_rect):
            return True
    # 头与自身
    for xy in snake_list[1:]:
        body_rect = (xy[0] - SNAKE_WIDTH / 2, xy[1] - SNAKE_WIDTH / 2, SNAKE_WIDTH, SNAKE_HEIGHT)
        if rect_cover(snake_head_rect, body_rect):
            return True
    return False


if __name__ == "__main__":
    # init pygame
    pygame.init()

    # contant
    GAME_SIZE = [900, 900]
    SIZE = [GAME_SIZE[0], GAME_SIZE[1] + 100]
    FONT_S = pygame.font.SysFont('Times', 50)
    FONT_M = pygame.font.SysFont('Times', 90)
    DIRECTION = ['up', 'right', 'down', 'left']
    X_LIST = [x for x in range(GAME_SIZE[0])]
    Y_LIST = [y for y in range(GAME_SIZE[1])]
    FOOD_COLORS = ((46, 139, 87), (199, 21, 133), (25, 25, 112), (255, 215, 0))

    # wall
    wall_list = [[100, 200], [600, 500], [350, 200], [500, 800]]
    WALL_WIDTH, WALL_HEIGHT = 30, 30

    # food
    food_list = [(150, 200, 1), (300, 500, 1), (740, 542, 1), (300, 600, 1), (700, 600, 1)]
    FOOD_WIDTH, FOOD_HEIGHT = 14, 14

    # create screen 500*500
    screen = pygame.display.set_mode(SIZE)

    # variable parameter
    snake_list = [[100 + 12 * 4, 100], [100 + 12 * 3, 100], [100 + 12 * 2, 100], [100 + 12 * 1, 100], [100, 100]]
    SNAKE_WIDTH, SNAKE_HEIGHT = 12, 12
    snake_v = 0
    count_time = 0

    # level
    frame = 0.05
    level = 1

    # main loop
    running = True
    pause = False
    dead = False
    head = 'right'
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pause = not pause
            elif event.type == pygame.KEYUP:
                if event.key == K_LEFT:
                    if head in ['up', 'down']:
                        head = 'left'
                elif event.key == K_RIGHT:
                    if head in ['up', 'down']:
                        head = 'right'
                elif event.key == K_UP:
                    if head in ['left', 'right']:
                        head = 'up'
                elif event.key == K_DOWN:
                    if head in ['left', 'right']:
                        head = 'down'

        # update data
        if not pause and not dead:
            count_time += frame * level
            first = snake_list[0]
            snake_list[1:] = snake_list[:-1]
            if head == 'up':
                snake_list[0] = [first[0], first[1] - SNAKE_WIDTH]
            elif head == 'down':
                snake_list[0] = [first[0], first[1] + SNAKE_WIDTH]
            elif head == 'left':
                snake_list[0] = [first[0] - SNAKE_WIDTH, first[1]]
            elif head == 'right':
                snake_list[0] = [first[0] + SNAKE_WIDTH, first[1]]

        # background
        draw_background()
        # tunnel
        draw_wall()
        # choose item
        draw_snake()
        # food
        draw_food()
        # point
        draw_context()
        # pause
        if not dead and pause:
            draw_pause()
        # dead
        if dead:
            draw_dead()
        # flip
        pygame.display.flip()

        # pause 20ms
        pygame.time.delay(int(frame / level * 1000))

        # check win or not
        dead = check_dead()

        if check_food():
            add_food()

    pygame.quit()