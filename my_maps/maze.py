import gym
from pogema import GridConfig
from pogema.animation import AnimationMonitor
from enum import Enum
import random
import numpy as np
import math
import copy

# random.seed(42)
# np.random.seed(42)

class Cell(str, Enum):
    EMPTY = '.'
    BLOCKED = '#'


def own_grid(variant, size=64):
    grid_np = np.zeros((size, size), dtype=np.uint8)

    if 1 in variant:
        n_row = np.random.randint(1, size-1, random.randint(1, size//10))
        n_col = np.random.randint(1, size-1, random.randint(1, size//10))


        grid_np[n_row, :] = 1
        grid_np[:, n_col] = 1

        for row in n_row:
            grid_np[row, np.random.randint(1, size-1, random.randint(1, size//16))] = 0
        for col in n_col:
            grid_np[np.random.randint(1, size-1, random.randint(1, size//16)), col] = 0

        grid = ''
        for r in range(size):
            grid += ''.join([Cell.BLOCKED if x == 1 else Cell.EMPTY for x in grid_np[r, :].tolist()]) + '\n'

    if 2 in variant:
        n_lines1 = np.random.randint(1, size-1, random.randint(1, size//10+1))
        n_lines1.sort()
        # copy_1 = []
        # for i in range(len(n_lines1)-1):
        #     if abs(n_lines1[i] - n_lines1[i+1]) > 5:
        #         copy_1.append(n_lines1[i])

        n_lines2 = np.random.randint(1, size-1, random.randint(1, size//10+1))
        # n_lines2.sort()
        # copy_2 = []
        # for i in range(len(n_lines2)-1):
        #     if abs(n_lines2[i] - n_lines2[i+1]) > 5:
        #         copy_2.append(n_lines2[i])

        n_fi1 = np.random.uniform(size=len(n_lines1))
        n_fi2 = np.random.uniform(size=len(n_lines2))

        for i, line in enumerate(n_lines1):
            n_dirok = random.randint(1, size//16)
            x_dirki = random.choices(list(range(1, size-1)), k=n_dirok)

            fi = -math.pi/10 + 2*math.pi/10 * n_fi1[i]
            k = math.tan(fi)
            b = line
            for x in range(size):
                y_point = int(k * x + b)
                if y_point >= size or y_point < 0:
                    continue
                grid_np[x, y_point] = 1
                if x in x_dirki:
                    grid_np[x, y_point] = 0

            for y in range(size):
                x_point = int((y - b) / k)
                if x_point >= size or x_point < 0:
                    continue
                # if random.random() < 0.95:
                grid_np[x_point, y] = 1

        for i, line in enumerate(n_lines2):
            n_dirok = random.randint(1, size//16)
            y_dirki = random.choices(list(range(1, size-1)), k=n_dirok)


            fi = math.pi/2 - math.pi/10 + 2*math.pi/10 * n_fi2[i]
            k = math.tan(fi)
            b = -line * k
            for x in range(size):
                y_point = int(k * x + b)
                if y_point >= size or y_point < 0:
                    continue
                # if random.random() < 0.95:
                grid_np[x, y_point] = 1
            for y in range(size):
                x_point = int((y - b) / k)
                if x_point >= size or x_point < 0:
                    continue
                grid_np[x_point, y] = 1
                if y in y_dirki:
                    grid_np[x_point, y] = 0

        grid = ''
        for r in range(size):
            grid += ''.join([Cell.BLOCKED if x == 1 else Cell.EMPTY for x in grid_np[r, :].tolist()]) + '\n'

    if 3 in variant:
        n_kvadrat = random.randint(2, size//6)
        xs = np.random.randint(1, size-1, n_kvadrat)
        ys = np.random.randint(1, size-1, n_kvadrat)
        ds = np.random.randint(5, max(6, size-10), n_kvadrat)

        for i in range(n_kvadrat):
            n_dirok = random.randint(0, size//16)
            x_dirki = random.choices(list(range(xs[i] - ds[i]//2, xs[i] + ds[i]//2)), k=n_dirok)
            for x in range(xs[i] - ds[i]//2, xs[i] + ds[i]//2):
                if 0 <= x < size:
                    if x not in x_dirki:
                        y1 = ys[i] - ds[i] // 2
                        if 0 <= y1 < size:
                            grid_np[x, y1] = 1
                        y2 = ys[i] + ds[i] // 2
                        if 0 <= y2 < size:
                            grid_np[x, y2] = 1
                    else:
                        if random.random() > 0.5:
                            y1 = ys[i] - ds[i] // 2
                            if 0 <= y1 < size:
                                grid_np[x, y1] = 1
                        else:
                            y2 = ys[i] + ds[i] // 2
                            if 0 <= y2 < size:
                                grid_np[x, y2] = 1

            n_dirok = random.randint(0, size//16)
            y_dirki = random.choices(list(range(ys[i] - ds[i]//2, ys[i] + ds[i]//2)), k=n_dirok)
            for y in range(ys[i] - ds[i]//2, ys[i] + ds[i]//2):
                if 0 <= y < size:
                    if y not in y_dirki:
                        x1 = xs[i] - ds[i] // 2
                        if 0 <= x1 < size:
                            grid_np[x1, y] = 1
                        x2 = xs[i] + ds[i] // 2
                        if 0 <= x2 < size:
                            grid_np[x2, y] = 1
                    else:
                        if random.random() > 0.5:
                            x1 = xs[i] - ds[i] // 2
                            if 0 <= x1 < size:
                                grid_np[x1, y] = 1
                        else:
                            x2 = xs[i] + ds[i] // 2
                            if 0 <= x2 < size:
                                grid_np[x2, y] = 1

        grid = ''
        for r in range(size):
            grid += ''.join([Cell.BLOCKED if x == 1 else Cell.EMPTY for x in grid_np[r, :].tolist()]) + '\n'

    return grid

if __name__ == '__main__':

    grid = own_grid([3], 32)
    # Define new configuration with 8 randomly placed agents
    grid_config = GridConfig(map=grid, num_agents=16)

    # Create custom Pogema environment
    env = gym.make('Pogema-v0', grid_config=grid_config)
    env = AnimationMonitor(env)

    obs = env.reset()

    done = [False, ...]

    while not all(done):
        # Use random policy to make actions
        obs, reward, done, info = env.step([env.action_space.sample() for _ in range(len(obs))])

    env.save_animation('render.svg')
