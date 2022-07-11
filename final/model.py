import random

import numpy as np
import torch

from model_astar import Model as astarmodel
from pogema.animation import AnimationMonitor

from pogema import GridConfig
from ppo import PPO, PPO_small
# from my_maps.maze import own_grid
import gym
from collections import Counter

class Model:

    def __init__(self, p=0.5, alg=1, device=torch.device('cpu'), ):
        if alg == 1:
            self.our_agen = PPO(path_to_actor='ppo_actor_7000a.pth')
            self.our_agen.actor.eval()
            self.our_agen.actor.to(device)
            self.our_agen.init_hidden(1)
        if alg == 2:
            self.our_agen = PPO(path_to_actor='ppo_actor_998.pth')
            self.our_agen.actor.eval()
            self.our_agen.actor.to(device)
            self.our_agen.init_hidden(1)
        if alg == 3:
            self.our_agen = PPO_small(path_to_actor='ppo_actor_IV.pth')
            self.our_agen.actor.eval()
            self.our_agen.actor.to(device)
            self.our_agen.init_hidden(1)

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
            self.batch_acts_old.append(self.action_old)

        action_deep, _ = self.our_agen.get_action(obs, self.batch_acts_old[-1], len(obs))
        # action_classic = self.solver.act(obs, dones, positions_xy, targets_xy)

        self.action_old = action_deep
        self.ep_t += 1

        self.steps += 1

        return action_deep

if __name__ == '__main__':
    n_games = 10
    for episod in range(n_games):
        seed = random.randint(0, 2**10)
        isr_do = []
        csr_do = []
        grid_config = GridConfig(num_agents=128,  # количество агентов на карте
                                     size=64,  # размеры карты
                                     density=0.3,  # плотность препятствий
                                     seed=seed,  # сид генерации задания
                                     max_episode_steps=256,  # максимальная длина эпизода
                                     obs_radius=5,  # радиус обзора
                                     )
        #
        # env = gym.make("Pogema-v0", grid_config=grid_config)
        # size = 16
        # tt = random.choice([[1], [2], [3], [1, 3], [2, 3]])
        print('')
        for alg in [1, 3]:

            classs = Model(0.5, alg=alg)

            # grid = own_grid(tt, size)
            # grid_config = GridConfig(map=grid, num_agents=32, size=size, max_episode_steps=256, obs_radius=5)
            env = gym.make('Pogema-v0', grid_config=grid_config)
            # env = AnimationMonitor(env)
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
            if alg == 1:
                print('Игра new {}. Результат isr {} / {}'.format(episod+1, win, int(sum(target))))
            if alg == 2:
                print('Игра NEW BAD {}. Результат isr {} / {}'.format(episod+1, win, int(sum(target))))
            if alg == 3:
                print('Игра old {}. Результат isr {} / {}'.format(episod+1, win, int(sum(target))))

            # csr = 1 if win == len(obs) else 0
            # env.save_animation('render/game{}.svg'.format(episod+1))

