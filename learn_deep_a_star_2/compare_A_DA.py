import torch
import random
from zero.model import Model
import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from ppo_net import PPOActor
import collections

def play_a_star():
    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = Model()
    steps = 0
    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
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
    solver = PPOActor(input_shape=3, dop_input_shape=5, rnn_hidden_dim=64, n_actions=5)
    solver.load_state_dict(torch.load('model_best.pth'))
    hidden_state = torch.zeros((1, 1, 64))
    a_old = 0
    steps = 0
    while not all(done):
        # Используем deep AStar
        ten_a_old = torch.zeros(5)
        ten_a_old[a_old] = 1
        pred, hidden_state = solver(torch.unsqueeze(torch.from_numpy(obs[0]), 0).to(torch.float32),
                                    torch.unsqueeze(ten_a_old, 0),
                                    hidden_state)
        y2 = np.argmax(pred.detach().numpy(), 1)
        obs, reward, done, info = env.step([y2[0]])
        a_old = y2[0]
        steps += 1
        # print(steps, np.sum(done))

    # сохраняем анимацию и рисуем ее
    env.save_animation("../renders/render_deep_A.svg", egocentric_idx=None)

    return steps

if __name__ == '__main__':
    counter = collections.Counter()
    li = []
    for episod in range(1_000):
        grid_config = GridConfig(num_agents=1,  # количество агентов на карте
                                 size=random.randint(20,64),  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=random.randint(0, 922337203685),  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )

        step_deepA = play_deep_a_star()
        step_A = play_a_star()

        if step_deepA - step_A < -5:
            break
        li.append(step_deepA - step_A)

    counter.update(li)
    for value, count in counter.most_common():
        print(value, count)