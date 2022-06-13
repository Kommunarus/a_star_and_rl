import torch
import random
from zero.model import Model
import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from learn_deep_rl_IV.ppo.ppo import PPO
import collections

def play_a_star():
    env = gym.make("Pogema-v0", grid_config=grid_config)
    # env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()
    num_as = len(obs)

    done = [False for k in range(num_as)]
    solver = Model()
    rewards_game = [[] for _ in range(num_as)]
    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
        #print(steps, np.sum(done))
        for robot in range(num_as):
            rewards_game[robot].append(reward[robot])

    # сохраняем анимацию и рисуем ее
    # env.save_animation("../renders/render_A.svg", egocentric_idx=None)
    target = [sum(x) for x in rewards_game]
    win = sum(target)
    csr = 1 if win == num_as else 0
    return win/num_as, csr

def play_deep_a_star(path):
    env = gym.make("Pogema-v0", grid_config=grid_config)
    obs = env.reset()

    num_as = len(obs)
    our_agent = PPO(env, num_as, path_to_actor=path)
    our_agent.actor.eval()
    our_agent.init_hidden(1)

    # Rewards this episode
    done = [False] * num_as
    rewards_game = [[] for _ in range(num_as)]
    batch_acts_old = []
    ep_t = 0
    finish = [False] * len(obs)

    while not all(done):
        if ep_t == 0:
            batch_acts_old.append([0] * len(obs))
        else:
            batch_acts_old.append(action)
        action, _ = our_agent.get_action(obs, batch_acts_old[-1], finish)
        obs, rew, done, _ = env.step(action)

        for robot in range(num_as):
            rewards_game[robot].append(rew[robot])

        for y, a_done in enumerate(done):
            if a_done == True:
                finish[y] = True

    target = [sum(x) for x in rewards_game]
    win = sum(target)
    csr = 1 if win == num_as else 0
    return win/num_as, csr

if __name__ == '__main__':
    counter = collections.Counter()
    n_games = 100
    for size in [15, 30, 60]:
        isr_dn = []
        csr_dn = []
        isr_do = []
        csr_do = []
        isr_a = []
        csr_a = []
        print(size)
        for episod in range(n_games):
            grid_config = GridConfig(num_agents=size,  # количество агентов на карте
                                         size=size,  # размеры карты
                                         density=0.3,  # плотность препятствий
                                         seed=random.randint(0, 922337203685),  # сид генерации задания
                                         max_episode_steps=256,  # максимальная длина эпизода
                                         obs_radius=5,  # радиус обзора
                                         )

            winDO, csrDO = play_deep_a_star('ppo/model_50.pth')
            winDN, csrDN = play_deep_a_star('ppo/ppo_actor.pth')
            winA, csrA = play_a_star()

            isr_dn.append(winDN)
            csr_dn.append(csrDN)
            isr_do.append(winDO)
            csr_do.append(csrDO)
            isr_a.append(winA)
            csr_a.append(csrA)

        # if step_deepA - step_A < -5:
        #     break
        # li.append(step_deepA - step_A)

        print('A star. csr {:.01f}%, isr {:.01f}%'.format(sum(csr_a)/n_games*100, sum(isr_a)/n_games*100))
        print('Deep 1. csr {:.01f}%, isr {:.01f}%'.format(sum(csr_do)/n_games*100, sum(isr_do)/n_games*100))
        print('Deep 2. csr {:.01f}%, isr {:.01f}%'.format(sum(csr_dn)/n_games*100, sum(isr_dn)/n_games*100))
