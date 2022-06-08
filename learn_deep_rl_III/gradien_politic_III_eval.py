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

device = torch.device('cuda:0')

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

        for i, (ts, ta, tao, tt) in enumerate(dataloader_train):
            s, a, a_old, target = ts.to(device).to(torch.float32), \
                          ta.to(device), \
                          tao.to(device), tt.to(device)

            if len(s) != batch_size:
                continue

            pred, self.hidden_state = self.solver(s, a_old, self.hidden_state)
            prob = nn.Softmax(1)(pred)
            prob_a = prob.gather(1, torch.unsqueeze(a,1)).squeeze(1)
            loss = (-(prob_a+1e-5).log() * target).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def get_action(self, s):
        tensor_s = torch.unsqueeze(torch.from_numpy(s).to(torch.float32), 0).to(device)

        probs = self.predict(tensor_s)

        action = torch.multinomial(probs, 1).item()
        # action = torch.argmax(probs).item()

        log_prob = torch.log(probs[action])

        self.a_old = torch.zeros((1, 5)).to(device)
        self.a_old[0, action] = 1

        return action, log_prob

def reinforce(n_episode, gamma=1.0, path_new_agent=None, path_old_agent=None, ):
    our_agent = PolicyNetwork(path_new_agent)
    states_dataset = []
    actions_dataset = []
    actions_old_dataset = []
    rewards_dataset = []
    for episode in range(n_episode):
        our_agent.hidden_state = torch.zeros((1, 1, 64)).to(device)

        num_agents = random.randint(min_agent, max_agent)
        grid_config = GridConfig(num_agents=num_agents,  # количество агентов на карте
                                 size=random.randint(min_agent, max_agent),  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=None,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
        env = gym.make("Pogema-v0", grid_config=grid_config)

        states = []
        actions = []
        actions_old = []

        rewards_game = []
        state = env.reset()

        n_rl_agent = random.randint(0, num_agents-1)
        a_old = 0

        while True:
            states.append(state[n_rl_agent].astype(np.uint8))

            all_action = []
            for robot in range(num_agents):
                if robot == n_rl_agent:
                    action_rl, log_prob = our_agent.get_action(state[robot])

                    all_action.append(action_rl)

                    actions.append(action_rl)
                    actions_old.append(a_old)

                    a_old = action_rl

                else:
                    all_action.append(random.randint(0, 4))


            next_state, reward, done, _ = env.step(all_action)

            rewards_game.append(reward[n_rl_agent])

            if done[n_rl_agent] or all(done):
                break

            state = next_state
        target = sum(rewards_game)
        rewards = [(1 if target == 1 else -1) for _ in rewards_game]

        states_dataset.append(copy.deepcopy(states))
        actions_dataset.append(copy.deepcopy(actions))
        actions_old_dataset.append(copy.deepcopy(actions_old))
        rewards_dataset.append(copy.deepcopy(rewards))

    dataset = Dataset_games(states_dataset, actions_dataset, actions_old_dataset, rewards_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    our_agent.update(dataloader)
    torch.save(our_agent.solver.state_dict(), path_new_agent)


class Dataset_games(Dataset):
    def __init__(self, states, actions, actions_old, rewards):
        self.states = [item for sublist in states for item in sublist]
        self.actions = [item for sublist in actions for item in sublist]
        self.actions_old = [item for sublist in actions_old for item in sublist]
        self.rewards = [item for sublist in rewards for item in sublist]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = torch.from_numpy(self.states[idx])

        a = torch.tensor(self.actions[idx])

        a_old = torch.nn.functional.one_hot(torch.tensor(self.actions_old[idx]), 5)

        t = torch.tensor(self.rewards[idx])

        return s, a, a_old, t


def play_game(config, path_new_agent, n_game):
    env = gym.make("Pogema-v0", grid_config=config)
    env = AnimationMonitor(env)
    obs = env.reset()

    our_agent = PolicyNetwork(path_new_agent)

    num_agents = len(obs)

    done = [False for _ in range(len(obs))]

    rewards_game = [[] for _ in range(num_agents)]

    while not all(done):
        all_action = []
        for robot in range(num_agents):
            action_rl = our_agent.get_action(obs[robot])
            all_action.append(action_rl[0])

        obs, reward, done, info = env.step(all_action)

        for robot in range(num_agents):
            rewards_game[robot].append(reward[robot])

    target = [sum(x) for x in rewards_game]
    win = sum(target)
    csr = 1 if win == num_agents else 0
    env.save_animation(f'../renders/my_animation_{n_game}.svg')
    return win, csr

def compare_two_agent( path_old_agent):
    # policy_old = PolicyNetwork(old_agent)
    # policy_new = PolicyNetwork()
    win_old, time_old, csr_old  = 0, 0, 0
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
        res_old = play_game(grid_config, path_old_agent, ep)

        win_old += res_old[0]
        csr_old += res_old[1]
        time_old += 64
        print(res_old)

    return win_old, time_old, csr_old/n_episode_val

if __name__ == '__main__':

    n_action = 5
    min_agent = 30
    max_agent = 60
    lr = 0.00005
    n_episode = 5000
    n_episode_val = 10
    batch_size = 2000
    gamma = 0.99
    n_generation = 100
    i_generation = 2
    path_old_agent = 'model_III_v2.pth'
    print(f'generation {i_generation}')
    out = compare_two_agent(path_old_agent)
    print(out)


