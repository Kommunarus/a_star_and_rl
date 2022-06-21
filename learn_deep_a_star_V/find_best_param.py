import random
import numpy as np
import torch

from final.model import Model
from model_astar import Model as astarmodel
import gym
from pogema import GridConfig



random.seed(42)
np.random.seed(42)

MAX_SEED = 2**32-1
SEEDS_PER_WORKER = 10

seeds = [np.random.randint(MAX_SEED) for _ in range(SEEDS_PER_WORKER)]

# all_map = [(4, 8), (8, 8),
#            (8, 16), (16, 16), (32, 16),
#            (8, 32), (16, 32), (32, 32), (64, 32), (128, 32),
#            (8, 64), (16, 64), (32, 64), (64, 64), (128, 64),
#            ]
all_map = [(128, 32), (128, 64),
           ]


for p in [0.05*i for i in range(1, 20)]:
    isr_do = []
    csr_do = []
    for i in range(SEEDS_PER_WORKER):
        map = random.choice(all_map)
        classs = Model(p, device=torch.device('cuda:1'))
        grid_config = GridConfig(num_agents=map[0],  # количество агентов на карте
                                     size=map[1],  # размеры карты
                                     density=0.3,  # плотность препятствий
                                     seed=seeds[i],  # сид генерации задания
                                     max_episode_steps=256,  # максимальная длина эпизода
                                     obs_radius=5,  # радиус обзора
                                     )

        env = gym.make("Pogema-v0", grid_config=grid_config)
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
        csr = 1 if win == 1 else 0
        isr_do.append(win)
        csr_do.append(csr)
    print('{} {:.03f} {:.03f}'.format(p, sum(isr_do)/len(isr_do), sum(csr_do)/len(csr_do)))

