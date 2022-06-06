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
from learn_deep_a_star.ppo_net import PPOActor
import collections
import os
import shutil
from learn_deep_a_star.dataset import Model

device = torch.device('cuda:1')

class PolicyNetwork():
    def __init__(self, path_to_model=None):

        self.solver = PPOActor(input_shape=3, dop_input_shape=5, rnn_hidden_dim=64, n_actions=5).to(device)
        if path_to_model is not None:
            self.solver.load_state_dict(torch.load(path_to_model))

        self.hidden_state = torch.zeros((1, 1, 64)).to(device)
        self.a_old = torch.zeros((1, 5)).to(device)

        self.optimizer = torch.optim.SGD(self.solver.parameters(), lr)


    def predict(self, s):
        x, h_new = self.solver(s, self.a_old, self.hidden_state)
        out = nn.Softmax(1)(x)
        self.hidden_state = h_new
        return torch.squeeze(out)


    def update(self, dataloader_train):
        self.solver.train()
        self.hidden_state = torch.zeros((batch_size, 1, 64)).to(device)

        for i, (ts, ta, tao, tt, taa) in enumerate(dataloader_train):
            s, a, a_old, target, a_star = ts.to(device).to(torch.float32), \
                          ta.to(device), \
                          tao.to(device), tt.to(device), taa.to(device)

            if len(s) != batch_size:
                continue

            pred, self.hidden_state = self.solver(s, a_old, self.hidden_state)
            prob = nn.Softmax(1)(pred)
            prob_a = prob.gather(1, torch.unsqueeze(a,1)).squeeze(1)
            loss1 = (-(prob_a+1e-5).log() * target).mean()

            loss2 = torch.nn.CrossEntropyLoss()(pred, a_star.to(torch.float))

            loss = loss1 + loss2
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def get_action(self, s):
        tensor_s = torch.unsqueeze(torch.from_numpy(s).to(torch.float32), 0).to(device)

        probs = self.predict(tensor_s)

        action = torch.multinomial(probs, 1).item()

        self.a_old = torch.zeros((1, 5)).to(device)
        self.a_old[0, action] = 1

        return action

def reinforce(n_episode, path_new_agent=None):
    states_dataset = []
    actions_dataset = []
    actions_star_dataset = []
    actions_old_dataset = []
    rewards_dataset = []
    for episode in range(n_episode):
        solver = Model()

        num_agents = 64
        our_agents = []
        for robot in range(num_agents):
            our_agent = PolicyNetwork(path_new_agent)
            our_agent.hidden_state = torch.zeros((1, 1, 64)).to(device)
            our_agents.append(our_agent)

        # num_agents = random.randint(min_agent, max_agent)
        grid_config = GridConfig(num_agents=num_agents,  # количество агентов на карте
                                 size=random.randint(min_agent, max_agent),  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=None,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
        env = gym.make("Pogema-v0", grid_config=grid_config)

        states = [[] for _ in range(num_agents)]
        actions = [[] for _ in range(num_agents)]
        actions_star = [[] for _ in range(num_agents)]
        actions_old = [[] for _ in range(num_agents)]

        rewards_game = [[] for _ in range(num_agents)]
        state = env.reset()

        a_old = [0] * num_agents

        while True:
            target = [sum(x) for x in rewards_game]
            action_astar = solver.act(state,
                                env.get_agents_xy_relative(),
                                env.get_targets_xy_relative())
            all_action = []
            for robot in range(num_agents):
                states[robot].append(state[robot].astype(np.uint8))
                if target[robot] == 0:
                    action_rl = our_agents[robot].get_action(state[robot])
                else:
                    action_rl = 0
                all_action.append(action_rl)
                actions[robot].append(action_rl)
                actions_old[robot].append(a_old[robot])
                a_old[robot] = action_rl
                actions_star[robot].append(action_astar[robot])



            next_state, reward, done, _ = env.step(all_action)

            for robot in range(num_agents):
                rewards_game[robot].append(reward[robot])

            if all(done):
                break

            state = next_state
        target = [sum(x) for x in rewards_game]
        rewards = [([1]*256 if x == 1 else [-1]*256) for x in target]

        states_dataset.append(copy.deepcopy(states))
        actions_dataset.append(copy.deepcopy(actions))
        actions_star_dataset.append(copy.deepcopy(actions_star))
        actions_old_dataset.append(copy.deepcopy(actions_old))
        rewards_dataset.append(copy.deepcopy(rewards))

    dataset = Dataset_games(states_dataset, actions_dataset, actions_old_dataset, rewards_dataset, actions_star_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    our_agent = PolicyNetwork(path_new_agent)
    our_agent.update(dataloader)
    torch.save(our_agent.solver.state_dict(), path_new_agent)


class Dataset_games(Dataset):
    def __init__(self, states, actions, actions_old, rewards, action_star):
        self.states = [item for play in states for agent in play for item in agent]
        self.actions = [item for play in actions for agent in play for item in agent]
        self.actions_star = [item for play in action_star for agent in play for item in agent]
        self.actions_old = [item for play in actions_old for agent in play for item in agent]
        self.rewards = [item for play in rewards  for agent in play for item in agent]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = torch.from_numpy(self.states[idx])

        a = torch.tensor(self.actions[idx])

        a_old = torch.nn.functional.one_hot(torch.tensor(self.actions_old[idx]), 5)

        t = torch.tensor(self.rewards[idx])
        a_star= torch.nn.functional.one_hot(torch.tensor(self.actions_star[idx]), 5)

        return s, a, a_old, t, a_star


def play_game(config, path_new_agent):
    env = gym.make("Pogema-v0", grid_config=config)
    obs = env.reset()

    our_agent = PolicyNetwork(path_new_agent)

    num_agents = len(obs)

    done = [False for _ in range(len(obs))]

    rewards_game = [[] for _ in range(num_agents)]

    while not all(done):
        all_action = []
        for robot in range(num_agents):
            action_rl = our_agent.get_action(obs[robot])
            all_action.append(action_rl)

        obs, reward, done, info = env.step(all_action)

        for robot in range(num_agents):
            rewards_game[robot].append(reward[robot])

    target = [sum(x) for x in rewards_game]
    win = sum(target)
    return win

def compare_two_agent(path_new_agent):
    # policy_old = PolicyNetwork(old_agent)
    # policy_new = PolicyNetwork()
    win_new, time_new = 0, 0
    for ep in range(n_episode_val):
        seed = random.randint(0, 922337203685)
        random.seed(seed)
        grid_config = GridConfig(num_agents=max_agent,  # количество агентов на карте
                                 size=64,  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
        res_new = play_game(grid_config, path_new_agent)

        win_new += res_new
        time_new += 64

    k = win_new / time_new
    return k, win_new, time_new

if __name__ == '__main__':

    n_action = 5
    min_agent = 30
    max_agent = 64
    lr = 0.0001
    n_episode = 2
    n_episode_val = 2
    batch_size = 64
    gamma = 0.99
    n_generation = 100
    path_old_agent = 'model_III_v1.pth'
    while True:

        path_new_agent = 'model_III_v2.pth'
        if not os.path.exists(path_new_agent):
            shutil.copyfile(path_old_agent, path_new_agent)

        reinforce(n_episode, path_new_agent)

        score, win_new, time_new = compare_two_agent(path_new_agent)
        print('\t дошло {:d} из {} ({:.03f})'.format(win_new, time_new, score))



    # counter = collections.Counter()
    # counter.update(total_reward_episode)
    # print(counter)
