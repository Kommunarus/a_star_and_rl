import torch
import random
from dataset import Model
import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from ppo_net import PPOActor
import collections

size_look = 15

def play_a_star():
    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver_astar = Model()
    steps = 0
    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver_astar.act(obs,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative())[0])
        steps += 1
        #print(steps, np.sum(done))

    # сохраняем анимацию и рисуем ее
    env.save_animation("../renders/render_A.svg", egocentric_idx=None)
    return steps

def play_deep_a_star():
    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = PPOActor(hidden_dim=128, n_actions=5)
    solver.load_state_dict(torch.load('model_best.pth'))

    solver_astar = Model()
    zabor_map = [np.zeros((2 * (64 + 2 * size_look) + 1, 2 * (64 + 2 * size_look) + 1), dtype=np.uint8), ]
    history_dict = {}
    step = 0

    while not all(done):
        current_xy = env.get_agents_xy_relative()
        targets_xy = env.get_targets_xy_relative()

        action, dist_agents = solver_astar.act(obs,
                                         current_xy,
                                         targets_xy)

        # Используем deep AStar
        newobs = update_obstacles(obs, current_xy, targets_xy, zabor_map, history_dict, dist_agents, step)
        newobs = np.float32(newobs[0])
        nz = np.nonzero(newobs[8])
        max_e = np.max(newobs[8])
        min_e = np.min(newobs[8][nz])
        mask = newobs[8] != 0
        for i in range(31):
            for j in range(31):
                if max_e - min_e != 0:
                    if mask[i, j]:
                        newobs[8][i, j] = (max_e - newobs[8][i, j]) / (max_e - min_e)
                else:
                    newobs[8][i, j] = 1

        data_s = torch.from_numpy(newobs)


        pred = solver(torch.unsqueeze(data_s, 0).to(torch.float32))
        # action_bot = np.argmax(pred.detach().numpy(), 1)[0]
        act_bot_probs = torch.nn.Softmax(1)(pred)
        action_bot = torch.multinomial(act_bot_probs, 1).item()

        obs, reward, done, info = env.step([action_bot])
        step += 1
        # print(steps, np.sum(done))

    # сохраняем анимацию и рисуем ее
    env.save_animation("../renders/render_deep_A.svg", egocentric_idx=None)

    return step

def update_obstacles(obs, currentxy, targetsxy, zabormap, historydict, distagents, step):
    newobs = []
    for i, (o, current_xy, targets_xy, zabor_map, dist_agents) in \
                                            enumerate(zip(obs, currentxy, targetsxy, zabormap, distagents)):

        centr = 64 + 2*size_look

        map_target = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_target_local = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_start = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_all_agents = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)

        d0 = centr + current_xy[0] - 5
        d1 = centr + current_xy[1] - 5
        zabor_map[d0: d0 + 11, d1: d1 + 11] = o[0]

        map_all_agents[d0: d0 + 11, d1: d1 + 11] = o[1]
        map_all_agents[centr + current_xy[0],  centr + current_xy[1]] = 0

        svobodno_map = 1 - (zabor_map + map_all_agents)

        historydict[step] = (centr + current_xy[0], centr + current_xy[1])
        hist = []
        for j in range(8):
            hhh = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
            if step - j >= 0:
                coord_hist = historydict[step - j]
                hhh[coord_hist[0], coord_hist[1]] = 1
            hist.append(hhh)


        map_target[centr + targets_xy[0], centr + targets_xy[1]] = 1

        map_target_local[d0: d0 + 11, d1: d1 + 11] = o[2]

        map_start[centr, centr] = 1

        x1 = centr + current_xy[0] - size_look
        x2 = x1 + 2*size_look + 1
        y1 = centr + current_xy[1] - size_look
        y2 = y1 + 2*size_look + 1

        map_1 = zabor_map[x1: x2, y1: y2]
        map_2 = map_all_agents[x1: x2, y1: y2]
        map_3 = svobodno_map[x1: x2, y1: y2]
        map_4 = np.zeros((size_look*2+1, size_look*2+1), dtype=np.uint8)
        map_5 = np.ones((size_look*2+1, size_look*2+1), dtype=np.uint8)
        map_6 = map_target[x1: x2, y1: y2]
        map_7 = map_target_local[x1: x2, y1: y2]
        map_8 = map_start[x1: x2, y1: y2]
        map_9 = dist_agents[x1: x2, y1: y2]

        hist_map = []
        for mmm in hist:
            hist_map.append(mmm[x1: x2, y1: y2])

        newobs.append(np.stack([map_1, map_2, map_3, map_4, map_5, map_6, map_7, map_8, map_9] + hist_map))

    return newobs


if __name__ == '__main__':
    counter = collections.Counter()
    li = []
    for episod in range(1):
        grid_config = GridConfig(num_agents=1,  # количество агентов на карте
                                 size=64,  # размеры карты
                                 density=0.4,  # плотность препятствий
                                 seed=random.randint(0, 922337203685),  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )

        step_deepA = play_deep_a_star()
        step_A = play_a_star()

        li.append(step_deepA - step_A)

    counter.update(li)
    for value, count in counter.most_common():
        print(value, count)