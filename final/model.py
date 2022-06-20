import random

import numpy as np

from model_astar import Model as astarmodel
from pogema.animation import AnimationMonitor

from pogema import GridConfig
from ppo import PPO
import gym
from collections import Counter

class Model:

    def __init__(self):
        self.our_agent = PPO(path_to_actor='ppo_actor_IV.pth')
        self.our_agent.actor.eval()
        self.our_agent.init_hidden(1)
        self.batch_acts_old = []

        self.solver = astarmodel()
        self.ep_t = 0
        self.history = None
        self.history2 = None
        self.n_future = 3
        self.steps = 0
        self.dist = [[] for _ in range(200)]
        # будет поворот или нет, сколько ходов крутиться, угол, сколько раз полный цикл, когда закончилось
        self.rotate = [[False, 0, 0, 0, 0] for _ in range(200)]
        self.positions_xy = [[] for _ in range(200)]
        self.step_out = 15


    def rotate_target(self, angle, obj):
        obj[2] = np.rot90(obj[2], angle)
        return obj


    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        # for i in range(len(obs)):
        #     self.positions_xy[i].append(positions_xy[i])
        #
        #
        # for i, ds in enumerate(self.dist):
        #     test_list = ds[-self.step_out:]
        #     if len(test_list) == self.step_out:
        #         # mean = sum(test_list) / len(test_list)
        #         # variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
        #         # res = variance ** 0.5
        #
        #         if not dones[i] and Counter(self.positions_xy[i]).most_common()[0][1] > 4 and \
        #                 not self.rotate[i][0] and \
        #                 (self.rotate[i][2] == 0 or (self.rotate[i][2] == 5 and self.rotate[i][3] <2)):
        #             self.rotate[i] = [True, 0, 2, self.rotate[i][3], self.ep_t]
        #
        #
        # for i, r in enumerate(self.rotate):
        #     if r[0]:
        #         if r[1] >= self.step_out:
        #             self.rotate[i][1] = 0
        #             if r[2] == 2:
        #                 self.rotate[i][2] = (1 if random.random() > 0.5 else 3)
        #             elif r[2] == 1 or r[2] == 3:
        #                 self.rotate[i] = [False, 0, 5, self.rotate[i][3] + 1, self.ep_t]


        # for i, (r, o) in enumerate(zip(self.rotate[:len(obs)], obs)):
        #     if r[0]:
        #         self.rotate_target(r[2], o)
        #         self.rotate[i][1] = r[1] + 1


        if self.ep_t == 0:
            self.batch_acts_old.append([0] * len(obs))
            # print(len(obs), min([min(t) for t in targets_xy]), max([max(t) for t in targets_xy]))
        else:
            self.batch_acts_old.append(self.action)
        # all_env = env.get_obstacles()
        # all_target = env.get_targets_xy()
        # all_agents = env.get_agents_xy()


        action_deep, _ = self.our_agent.get_action(obs, self.batch_acts_old[-1], len(obs))
        action_classic = self.solver.act(obs, dones, positions_xy, targets_xy)

        actions = []
        for ag in range(len(obs)):
            if random.random() > 0.85:
                actions.append(action_deep[ag])
            else:
                actions.append(action_classic[ag])

        # for ag in range(len(obs)):
        #     if self.rotate[ag][3] == 2:
        #         actions.append(action_classic[ag])
        #     else:
        #         actions.append(action_deep[ag])
            # a = obs[ag][1]
            # if np.sum(a) <= 1:
            #     actions.append(action_classic[ag])
            # else:
            #     actions.append(action_deep[ag])
        # print(self.rotate[:8])

        # actions = action_deep
        self.action = actions
        self.ep_t += 1

        # for agent_id in range(200,  len(obs)):
        #     actions.append(0)

        for i, d in enumerate(targets_xy):
            h_new = abs(positions_xy[i][0] - targets_xy[i][0]) + abs(positions_xy[i][1] - targets_xy[i][1])

            self.dist[i].append(h_new)

        if self.steps == 255:
            # print(self.rotate[:len(obs)])
            pass
        #     print(sum(dones), len(obs))
        #     if len(obs) - sum(dones) < 10:
        #         for a in range(len(obs)):
        #             if not dones[a]:
        #                 mm = obs[a][0].astype(int)
        #                 aa = obs[a][1].astype(int)
        #                 cc = obs[a][2].astype(int)
        #
        #                 mm[np.nonzero(aa)] = 9
        #                 mm[np.nonzero(cc)] = 5
        #                 print(mm)

        self.steps += 1

        return actions

if __name__ == '__main__':
    n_games = 10
    for episod in range(n_games):
        classs = Model()
        isr_do = []
        csr_do = []
        grid_config = GridConfig(num_agents=64,  # количество агентов на карте
                                     size=64,  # размеры карты
                                     density=0.3,  # плотность препятствий
                                     seed=None,  # сид генерации задания
                                     max_episode_steps=256,  # максимальная длина эпизода
                                     obs_radius=5,  # радиус обзора
                                     )

        env = gym.make("Pogema-v0", grid_config=grid_config)
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
        print('Игра {}. Результат isr {}'.format(episod+1, win))
        env.save_animation('render/game{}.svg'.format(episod+1))

