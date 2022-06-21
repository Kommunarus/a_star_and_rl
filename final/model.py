import random

import numpy as np
import torch

from model_astar import Model as astarmodel
from pogema.animation import AnimationMonitor

from pogema import GridConfig
from ppo import PPO
from my_maps.maze import own_grid
import gym
from collections import Counter

class Model:

    def __init__(self, p=0.5, device=torch.device('cpu')):
        self.our_agent = PPO(path_to_actor='ppo_actor_IV.pth')
        self.our_agent.actor.eval()
        self.our_agent.actor.to(device)
        self.our_agent.init_hidden(1)
        self.batch_acts_old = []

        self.solver = astarmodel()
        self.ep_t = 0
        self.steps = 0
        self.p = p
        self.device = device


    def act(self, obs, dones, positions_xy, targets_xy) -> list:

        if self.ep_t == 0:
            self.batch_acts_old.append([0] * len(obs))
        else:
            self.batch_acts_old.append(self.action)


        action_deep, _ = self.our_agent.get_action(obs, self.batch_acts_old[-1], len(obs))
        # action_classic = self.solver.act(obs, dones, positions_xy, targets_xy)
        #
        # actions = []
        #
        # for ag in range(len(obs)):
        #     if self.ep_t < 220:
        #         if self.ep_t % 10 != 0:
        #             if random.random() > self.p:
        #                 actions.append(action_deep[ag])
        #             else:
        #                 # actions.append(action_deep[ag])
        #                 actions.append(action_classic[ag])
        #         else:
        #             actions.append(random.randint(1, 4))
        #     else:
        #         actions.append(action_classic[ag])

        self.action = action_deep
        self.ep_t += 1

        self.steps += 1

        return action_deep

if __name__ == '__main__':
    n_games = 10
    for episod in range(n_games):
        classs = Model(0.5)
        isr_do = []
        csr_do = []
        # grid_config = GridConfig(num_agents=64,  # количество агентов на карте
        #                              size=64,  # размеры карты
        #                              density=0.3,  # плотность препятствий
        #                              seed=None,  # сид генерации задания
        #                              max_episode_steps=256,  # максимальная длина эпизода
        #                              obs_radius=5,  # радиус обзора
        #                              )
        #
        # env = gym.make("Pogema-v0", grid_config=grid_config)
        size = 32
        tt = random.choice([[1], [2], [3], [1, 3], [2, 3]])

        grid = own_grid(tt, size)
        grid_config = GridConfig(map=grid, num_agents=16, size=size, max_episode_steps=256, obs_radius=5)
        env = gym.make('Pogema-v0', grid_config=grid_config)
        env = AnimationMonitor(env)
        obs = env.reset()
        done = [False for k in range(len(obs))]
        rewards_game = [[] for _ in range(len(obs))]


        while not all(done):
            act = classs.act(obs, done, env.get_agents_xy_relative(), env.get_targets_xy_relative())
            obs, rew, done, _ = env.step(act)
            for robot in range(len(obs)):
                rewards_game[robot].append(rew[robot])


        target = [sum(x) for x in rewards_game]
        win = sum(target) / len(obs)
        # csr = 1 if win == len(obs) else 0
        print('Игра {}. Результат isr {}. {}'.format(episod+1, win, tt))
        env.save_animation('render/game{}.svg'.format(episod+1))

