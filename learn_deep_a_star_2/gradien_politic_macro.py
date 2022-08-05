import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import random
# from zero.model import Model
import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from ppo_net import PPOActor2
import collections
import os
import shutil
from dataset import Model as astar
import datetime
import torch.nn.functional as F

device = torch.device('cuda:1')

class Agent:
    def __init__(self):
        self.lenmap = max_size + 2*size_look
        self.zabor_map = np.zeros((2 *self.lenmap + 1, 2 * self.lenmap + 1), dtype=np.uint8)


class PolicyNetwork():
    def __init__(self, path_to_model=None):
        self.maxagent = 128
        self.solver = PPOActor2().to(device)
        if path_to_model is not None:
            self.solver.load_state_dict(torch.load(path_to_model))


        self.optimizer = torch.optim.SGD(self.solver.parameters(), lr)


    def predict(self, s):
        eps = 1e-4
        x = self.solver(s)
        act_bot_probs = nn.Softmax(1)(torch.squeeze(x))
        move_probs = torch.clip(act_bot_probs, eps, 1 - eps)
        move_probs = move_probs / torch.sum(move_probs, 1).unsqueeze(1)
        # action_bot = torch.multinomial(act_bot_probs, 1).item()
        return move_probs


    def update(self, batch_size, dataloader_train):
        self.solver.train()

        for i, (ts, ta, tt, nt) in enumerate(dataloader_train):
            s, a, target, n = ts.to(device).to(torch.float32), \
                          ta.to(device), \
                          tt.to(device), \
                          nt.to(device)

            # if len(s) != batch_size:
            #     continue

            pred_all = self.solver(s)
            pred = torch.zeros((pred_all.shape[0], pred_all.shape[2])).to(device)
            for i in range(pred_all.shape[0]):
                pred[i] = pred_all[i, n[i]]
            prob = nn.Softmax(1)(pred)
            prob_a = prob.gather(1, torch.unsqueeze(a, 1)).squeeze(1)
            loss1 = (-(prob_a + 1e-5).log() * target).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss1.backward()
            self.optimizer.step()


    def get_action(self, list_s):
        with torch.no_grad():
            actins = []
            tensor_s = torch.from_numpy(np.array(list_s)).to(torch.float32).to(device)
            new_tensor = torch.zeros((max_agent, 17, 31, 31)).to(torch.float32).to(device)
            numis = random.sample(range(max_agent), k=len(list_s))
            for i, num in enumerate(numis):
                new_tensor[num] = tensor_s[i]
            new_tensor = torch.unsqueeze(new_tensor, 0)
            probs = self.predict(new_tensor)
            for i, num in enumerate(numis):
                action = torch.multinomial(probs[num], 1).item()
                actins.append(action)
        return actins, numis

def update_obstacles(obs, currentxy, targetsxy, historydict, dist_agents_without, dist_agents_with, step, agents):
    newobs = []
    for i, (o, current_xy, targets_xy, distagentswithout, distagentswith) in \
            enumerate(zip(obs, currentxy, targetsxy, dist_agents_without, dist_agents_with)):

        centr = max_size + 2 * size_look
        zabor_map = agents[i].zabor_map

        map_target = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_target_local = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_start = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_all_agents = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)

        d0 = centr + current_xy[0] - 5
        d1 = centr + current_xy[1] - 5
        zabor_map[d0: d0 + 11, d1: d1 + 11] = o[0]

        map_all_agents[d0: d0 + 11, d1: d1 + 11] = o[1]
        map_all_agents[centr + current_xy[0], centr + current_xy[1]] = 0

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
        x2 = x1 + 2 * size_look + 1
        y1 = centr + current_xy[1] - size_look
        y2 = y1 + 2 * size_look + 1

        map_0 = zabor_map[x1: x2, y1: y2]
        map_1 = map_all_agents[x1: x2, y1: y2]
        map_2 = svobodno_map[x1: x2, y1: y2]
        map_3 = np.zeros((size_look * 2 + 1, size_look * 2 + 1), dtype=np.uint8)
        map_4 = np.ones((size_look * 2 + 1, size_look * 2 + 1), dtype=np.uint8)
        map_5 = map_target[x1: x2, y1: y2]
        map_6 = map_target_local[x1: x2, y1: y2]
        map_7 = map_start[x1: x2, y1: y2]
        # map_8 = dist_agentswith[x1: x2, y1: y2]
        map_8 = distagentswithout[x1: x2, y1: y2]

        hist_map = []
        for mmm in hist:
            hist_map.append(mmm[x1: x2, y1: y2])

        # map_16 = [dist_agents_wout[x1: x2, y1: y2]]
        # map_16 = [distagentswith[x1: x2, y1: y2]]

        end_numpy = np.stack([map_0, map_1, map_2, map_3, map_4, map_5, map_6, map_7, map_8] + hist_map)

        end_numpy = end_numpy.astype(np.float32)

        for numlayer in [8]:
            nz = np.nonzero(end_numpy[numlayer])

            new8 = np.zeros_like(end_numpy[numlayer])
            if len(nz[0]) != 0:
                max_e = np.max(end_numpy[numlayer])
                min_e = np.min(end_numpy[numlayer][nz])
                if min_e == max_e:
                    new8[nz] = 1
                else:
                    new8[nz] = (max_e - end_numpy[numlayer][nz]) / (max_e - min_e)


            end_numpy[numlayer] = new8
        newobs.append(end_numpy)

    return newobs


def reinforce(bs, current_policy=None):
    states_dataset = []
    numers_dataset = []
    listnumers_dataset = []
    actions_dataset = []
    rewards_dataset = []
    # all_map = [(32, 16), (32, 32), (32, 64)]
    all_map = [(8, 16), (16, 16), #(32, 16),
               (8, 32), (16, 32), #(32, 32),#(64, 32),# (128, 32),
               (8, 64), (16, 64), #(32, 64),#(64, 64),# (128, 64),
               ]
    cooperativ_score = []
    cooperativ_numagent = []
    all_fin = 0
    for map in all_map:
        agents = []

        # map = random.choice(all_map)
        num_agents = map[0]
        for i in range(num_agents):
            agents.append(Agent())

        # num_agents = random.randint(min_agent, max_agent)
        grid_config = GridConfig(num_agents=map[0],
                                 size=map[1],
                                 density=random.uniform(0.2, 0.4),
                                 seed=None,
                                 max_episode_steps=256,
                                 obs_radius=5,
                                 )
        env = gym.make("Pogema-v0", grid_config=grid_config)

        states = [[] for _ in range(num_agents)]
        numers = [[] for _ in range(num_agents)]
        listnumers = [[] for _ in range(num_agents)]
        actions = [[] for _ in range(num_agents)]
        rewards_game = [[] for _ in range(num_agents)]
        state = env.reset()

        history_dict = {}
        step = 0
        solverastar_withoutagent = astar(False)
        # solverastar_withagent = astar(True)

        while True:
            targets_xy = env.get_targets_xy_relative()
            agents_xy = env.get_agents_xy_relative()
            _, dist_agents_without = solverastar_withoutagent.act(state, agents_xy, targets_xy)
            # _, dist_agents_with = solverastar_withagent.act(state, agents_xy, targets_xy)

            newobs = update_obstacles(state, agents_xy, targets_xy, history_dict,
                                      dist_agents_without, dist_agents_without, step, agents)


            target = [sum(x) for x in rewards_game]
            actionrl, numis = current_policy.get_action(newobs)

            all_action = []
            for robot in range(num_agents):
                if target[robot] == 0:
                    if random.random() > 0.05:
                        action_rl = actionrl[robot]
                    else:
                        action_rl = random.choice([0, 1, 2, 3, 4])

                    states[robot].append(newobs)
                    numers[robot].append(numis[robot])
                    listnumers[robot].append(numis)
                    actions[robot].append(action_rl)
                else:
                    action_rl = 0
                all_action.append(action_rl)



            next_state, reward, done, _ = env.step(all_action)

            for robot in range(num_agents):
                if target[robot] == 0:
                    rewards_game[robot].append(reward[robot])

            if all(done):
                break

            state = next_state
            step += 1

        target = [sum(x) for x in rewards_game]
        all_fin += sum(target)
        rewards = []
        for numa, t in enumerate(target):
            states_agent = states[numa]
            if t == 1:
                rewards.append([1]*len(states_agent))
            else:
                rewards.append([-1] * len(states_agent))


        states_dataset.append(copy.deepcopy(states))
        numers_dataset.append(copy.deepcopy(numers))
        listnumers_dataset.append(copy.deepcopy(listnumers))
        actions_dataset.append(copy.deepcopy(actions))
        rewards_dataset.append(copy.deepcopy(rewards))
        cooperativ_score.append(round(sum(target) / len(target), 2))
        cooperativ_numagent.append(num_agents)

    dataset = Dataset_games(states_dataset, actions_dataset, rewards_dataset, numers_dataset, listnumers_dataset)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    current_policy.update(bs, dataloader)
    torch.save(current_policy.solver.state_dict(), 'model_gp.pth')
    if episode % 250 == 0:
        torch.save(current_policy.solver.state_dict(),
                   './history_model/model_gp_{}_{}.pth'.format(datetime.datetime.now(),
                                                               round(all_fin/sum([x[0] for x in all_map]), 2)))

    print(episode, int(all_fin), round(all_fin/sum([x[0] for x in all_map]), 2),
          round(sum(cooperativ_score)/len(cooperativ_score), 2), [x for x, y in zip(cooperativ_score, all_map)],
          sep='\t\t')
    # print(cooperativ_numagent)


class Dataset_games(Dataset):
    def __init__(self, states, actions, rewards, numers, listnumers):
        self.states = [item for play in states for agent in play for item in agent]
        self.actions = [item for play in actions for agent in play for item in agent]
        self.rewards = [item for play in rewards for agent in play for item in agent]
        self.numers = [item for play in numers for agent in play for item in agent]
        self.listnumers = [item for play in listnumers for agent in play for item in agent]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        list_s = self.states[idx]
        numis = self.listnumers[idx]

        tensor_s = torch.from_numpy(np.array(list_s)).to(torch.float32).to(device)
        new_tensor = torch.zeros((max_agent, 17, 31, 31)).to(torch.float32).to(device)
        for i, num in enumerate(numis):
            new_tensor[num] = tensor_s[i]
        # new_tensor = torch.unsqueeze(new_tensor, 0)

        a = torch.tensor(self.actions[idx])
        t = torch.tensor(self.rewards[idx])
        n = torch.tensor(self.numers[idx])
        return new_tensor, a, t, n


if __name__ == '__main__':
    max_agent = 32

    lr = 1e-5
    batchsize = 32
    max_size = 64
    size_look = 15

    pathagent = 'model_end_macro3d_32.pth'
    # pathagent = 'model_gp.pth'
    episode = 0
    current_policy = PolicyNetwork(path_to_model=pathagent)
    while True:
        reinforce(batchsize, current_policy)
        episode += 1


