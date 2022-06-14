#!/usr/bin/env python3
import copy
import numpy as np
import random
import gym
from pogema import GridConfig
import torch
import torch.nn as nn
from final.model_astar import Model as astarmodel
from final.ppo import PPO

device = torch.device('cuda:1')


POPULATION_SIZE = 50
PARENTS_COUNT = 10


class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + 2*action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

    def select_action(self, s, a1, a2):
        st = torch.from_numpy(np.array([s])).squeeze(0).to(device)
        a1t = torch.nn.functional.one_hot(torch.tensor(a1), 5).to(device)
        a2t = torch.nn.functional.one_hot(torch.tensor(a2), 5).to(device)
        st = torch.nn.Flatten()(st)
        input = torch.hstack([st, a1t, a2t]).to(torch.float)
        return self.forward(input)


class Model:

    def __init__(self):
        self.our_agent = PPO(path_to_actor='ppo_actor_IV.pth')
        self.our_agent.actor.eval()
        self.our_agent.init_hidden(1)
        self.batch_acts_old = []

        self.solver = astarmodel()
        self.ep_t = 0

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.ep_t == 0:
            self.batch_acts_old.append([0] * len(obs))
        else:
            self.batch_acts_old.append(self.action)

        action_deep, _ = self.our_agent.get_action(obs, self.batch_acts_old[-1], len(obs))
        action_astar = self.solver.act(obs, dones, positions_xy, targets_xy)

        self.ep_t += 1

        return (action_deep, action_astar)


def evaluate(nets, seed):
    grid_config = GridConfig(num_agents=n_agent,  # количество агентов на карте
                             size=n_size,  # размеры карты
                             density=0.4,  # плотность препятствий
                             seed=seed,  # сид генерации задания
                             max_episode_steps=256,  # максимальная длина эпизода
                             obs_radius=5,  # радиус обзора
                             )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    classs = Model()

    obs = env.reset()
    done = [False for k in range(len(obs))]
    rewards_game = [[] for _ in range(len(obs))]

    while not all(done):
        action_deep, action_astar = classs.act(obs, done, env.get_agents_xy_relative(),
                                               env.get_targets_xy_relative())



        out_net = nets.select_action(obs, action_deep, action_astar)


        act = [x if z[0] > 0.5 else y for x, y, z in zip(action_deep, action_astar, out_net)]
        classs.action = act


        obs, rew, done, _ = env.step(act)
        for robot in range(len(obs)):
            rewards_game[robot].append(rew[robot])


    target = [sum(x) for x in rewards_game]
    win = sum(target)
    csr = 1 if win == len(obs) else 0
    return win

def get_action(nets, indx, s):
    act = {}
    for i in range(n_agent):
        act[i] = torch.argmax(nets[i][indx[i]].select_action(s[i])).item()

    return act


def mutate_parent(nets):
    new_net = copy.deepcopy(nets)
    for p in new_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise).to(device)
        p.data += NOISE_STD * noise_t
    return new_net

def get_nets(nets, indx):
    out = []
    for i in range(n_agent):
        out.append(nets[i][indx[i]])
    return out

def get_random(n):
    return [random.randint(0, POPULATION_SIZE-1) for i in range(n)]

if __name__ == "__main__":
    n_agent = 100
    n_size = 100

    n_state = 3*11*11
    n_action = 5

    gen_idx = 0
    nets = [
            Net(n_state, n_action).to(device)
            for _ in range(POPULATION_SIZE)
        ]

    seedi = random.randint(0, 10**10)

    population = [
        (nets[i], evaluate(nets[i], seedi)) for i in range(POPULATION_SIZE)
    ]

    NOISE_STD = 0.5
    alfa = 0.999
    good_step = 0
    max_good_steps = 20

    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)

        print("%d: reward_mean=%.2f, reward_max=%.2f, "
              "reward_std=%.2f" % (
            gen_idx, reward_mean, reward_max, reward_std))
        if reward_mean >= 95:
            good_step += 1
            if good_step > max_good_steps:
                print("Solved in %d steps" % gen_idx)
                break

        prev_population = population
        torch.save(population[0][0].state_dict(), 'model/agent.pkl')

        # generate next population
        # population = [population[0]]
        seedi = random.randint(0, 10**10)

        for _ in range(POPULATION_SIZE):
            parent_idx = np.random.randint(0, PARENTS_COUNT)
            parent = prev_population[parent_idx][0]
            nets = mutate_parent(parent)
            population.append((nets, evaluate(nets, seedi)))
        gen_idx += 1
        NOISE_STD = alfa * NOISE_STD