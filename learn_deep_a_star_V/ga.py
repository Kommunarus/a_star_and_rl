#!/usr/bin/env python3
import copy
import numpy as np
import random
import gym
from pogema import GridConfig
import torch
import torch.nn as nn
from model_astar import Model as astarmodel
from ppo import PPO
import collections
import torch.multiprocessing as mp
import time

device = torch.device('cuda:0')
NOISE_STD = 0.01
POPULATION_SIZE = 100
PARENTS_COUNT = 10
WORKERS_COUNT = 10
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2 ** 32 - 1
all_map = [(4, 8), (8, 8),
           (8, 16), (16, 16), (32, 16),
           (8, 32), (16, 32), (32, 32), (64, 32),# (128, 32),
           (8, 64), (16, 64), (32, 64), (64, 64), #(128, 64), #(256, 64),
           ]

seeds_for_game = [np.random.randint(MAX_SEED) for _ in range(200)]


class Net(nn.Module):
    def __init__(self, obs_size, action_size, dim_step):
        super(Net, self).__init__()
        self.dim_step = dim_step
        self.action_size = action_size
        self.net = nn.Sequential(
            nn.Linear(obs_size + 2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        # self.net.state_dict(torch.load('model/agent_4.pth'))


    def forward(self, x):
        return self.net(x)

    def select_action(self, s, a1, a2, step):
        st = torch.from_numpy(np.array([s])).squeeze(0).to(device)
        # a1t = torch.nn.functional.one_hot(torch.tensor(a1), self.action_size).to(device)
        # a2t = torch.nn.functional.one_hot(torch.tensor(a2), self.action_size).to(device)
        # a3t = torch.nn.functional.one_hot(torch.tensor(step), self.dim_step).to(device)
        a1t = torch.unsqueeze(torch.tensor(a1).to(device)/self.action_size, 1)
        a2t = torch.unsqueeze(torch.tensor(a2).to(device)/self.action_size, 1)
        a3t = torch.unsqueeze(torch.tensor(step).to(device)/self.dim_step, 1)
        st = torch.nn.Flatten()(st)
        input = torch.hstack([st, a1t, a2t, a3t]).to(torch.float)
        return self.forward(input)


class Model:

    def __init__(self):
        self.our_agent = PPO(path_to_actor='ppo_actor_IV.pth')
        self.our_agent.actor.eval()
        self.our_agent.init_hidden(1)
        self.batch_acts_old = []

        self.solver = astarmodel()
        self.ep_t = 0
        self.action = None

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.ep_t == 0:
            self.batch_acts_old.append([0] * len(obs))
        else:
            self.batch_acts_old.append(self.action)

        action_deep, _ = self.our_agent.get_action(obs, self.batch_acts_old[-1], len(obs))
        action_astar = self.solver.act(obs, dones, positions_xy, targets_xy)

        self.ep_t += 1

        return [action_deep, action_astar]


def evaluate(env, net):
    classs = Model()

    obs = env.reset()
    done = [False for k in range(len(obs))]
    rewards_game = [[] for _ in range(len(obs))]
    step = 0
    while not all(done):
        action_deep, action_astar = classs.act(obs, done, env.get_agents_xy_relative(),
                                               env.get_targets_xy_relative())

        out_net = net.select_action(obs, action_deep, action_astar, [step, ]*len(obs))

        act = [(x if z[0] > 0.5 else y) for x, y, z in zip(action_deep, action_astar, out_net)]
        classs.action = act
        step += 1

        obs, rew, done, _ = env.step(act)
        for robot in range(len(obs)):
            rewards_game[robot].append(rew[robot])


    target = [sum(x) for x in rewards_game]
    isr = sum(target)
    # csr = 1 if win == len(obs) else 0
    return isr, step


def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net).to(device) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(device)
        p.data += NOISE_STD * noise_t
    return new_net


def build_net(seeds):
    torch.manual_seed(seeds[0])
    net = Net(3*11*11, 5, 256).to(device)
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'isr', 'steps'])

def savemodel(seed_elite):
    torch.manual_seed(seed_elite[0])
    net = Net(3 * 11 * 11, 5, 256)
    for seed in seed_elite[1:]:
        np.random.seed(seed)
        for p in net.parameters():
            noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
            p.data += NOISE_STD * noise_t
    torch.save(net.state_dict(), 'model/agent_best.pth')


def worker_func(input_queue, output_queue):
    p = mp.current_process()
    cache = {}

    while True:
        sfg, parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1])
                else:
                    net = build_net(net_seeds)
            else:
                net = build_net(net_seeds)
            new_cache[net_seeds] = net
            net = net.to(device)
            isr, steps = 0, 0
            for map in all_map:
                grid_config = GridConfig(num_agents=map[0],
                                         size=map[1],
                                         density=0.3,
                                         seed=sfg,
                                         max_episode_steps=256,
                                         obs_radius=5,
                                         )

                env = gym.make("Pogema-v0", grid_config=grid_config)

                isr_one, steps_one = evaluate(env, net)
                isr += isr_one
                steps += steps_one
            # print(p.name, isr, steps)
            output_queue.put(OutputItem(seeds=net_seeds, isr=isr, steps=steps))
        cache = new_cache


if __name__ == "__main__":

    mp.set_start_method('spawn')

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)

    workers = []
    sfg = seeds_for_game[0]
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put([sfg, seeds])

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.isr))
            batch_steps += out_item.steps
        # if elite is not None:
        #     population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        speed = batch_steps / (time.time() - t_start)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s, seed_map %d" % (
            gen_idx+1, reward_mean, reward_max, reward_std, speed, sfg))

        elite = population[0]
        with open('elite_score.txt', 'a') as f:
            f.write('{} {} {}\n'.format(elite[0], elite[1], sfg))
        # print(elite)
        seed_elite = elite[0]
        savemodel(seed_elite)
        sfg = seeds_for_game[gen_idx+1]

        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
            worker_queue.put([sfg, seeds])
        gen_idx += 1
