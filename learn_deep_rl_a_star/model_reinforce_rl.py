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

device = torch.device('cuda:1')

class PolicyNetwork():
    def __init__(self, path_to_model=None):

        self.solver = PPOActor(input_shape=3, dop_input_shape=5, rnn_hidden_dim=64, n_actions=5).to(device)
        if path_to_model is not None:
            self.solver.load_state_dict(torch.load(path_to_model))

        self.hidden_state = torch.zeros((1, 1, 64)).to(device)
        self.a_old = torch.zeros((1, 5)).to(device)

        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr)


    def predict(self, s):
        x, h_new = self.solver(s, self.a_old, self.hidden_state)
        out = nn.Softmax(1)(x)
        self.hidden_state = h_new
        return torch.squeeze(out)


    def update(self, returns, log_probs):
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(-log_prob * Gt)

        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, s):
        tensor_s = torch.unsqueeze(torch.from_numpy(s).to(torch.float32), 0).to(device)

        probs = self.predict(tensor_s)

        action = torch.multinomial(probs, 1).item()

        log_prob = torch.log(probs[action])

        self.a_old = torch.zeros((1, 5)).to(device)
        self.a_old[0, action] = 1

        return action, log_prob

def reinforce(n_episode, gamma=1.0, path_new_agent=None, path_old_agent=None, ):
    for episode in range(n_episode):
        # if episode % 100 == 0:
        #     print('episode {}'.format(episode))

        num_agents = random.randint(min_agent, max_agent)
        grid_config = GridConfig(num_agents=num_agents,  # количество агентов на карте
                                 size=random.randint(30, 64),  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=None,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
        env = gym.make("Pogema-v0", grid_config=grid_config)

        log_probs = []
        rewards = []
        state = env.reset()

        n_rl_agent = random.randint(0, num_agents-1)

        robots = []
        for robot in range(num_agents):
            if robot == n_rl_agent:
                robots.append(PolicyNetwork(path_new_agent))
            else:
                robots.append(PolicyNetwork(path_old_agent))

        while True:
            all_action = []
            for robot in range(num_agents):
                action_rl, log_prob = robots[robot].get_action(state[robot])
                all_action.append(action_rl)

                if robot == n_rl_agent:
                    log_probs.append(log_prob)

            next_state, reward, done, _ = env.step(all_action)

            rewards.append(reward[n_rl_agent])

            if done[n_rl_agent] or all(done):
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                robots[n_rl_agent].update(returns, log_probs)
                # print(sum(rewards))
                break

            state = next_state

        torch.save(robots[n_rl_agent].solver.state_dict(), path_new_agent)

def play_game(config, path_new_agent, path_old_agent, n_rl_agent):
    env = gym.make("Pogema-v0", grid_config=config)
    obs = env.reset()

    num_agents = len(obs)
    robots = []
    for robot in range(num_agents):
        if robot == n_rl_agent:
            robots.append(PolicyNetwork(path_new_agent))
        else:
            robots.append(PolicyNetwork(path_old_agent))

    done = [False for _ in range(len(obs))]
    steps = 0
    win = 0

    while not all(done):
        all_action = []
        for robot in range(num_agents):
            action_rl, log_prob = robots[robot].get_action(obs[robot])
            all_action.append(action_rl)

        obs, reward, done, info = env.step(all_action)

        if done[n_rl_agent] and steps <= 250:
            win = 1
            break

        steps += 1


    return win, steps

def compare_two_agent(path_new_agent, path_old_agent):
    # policy_old = PolicyNetwork(old_agent)
    # policy_new = PolicyNetwork()
    win_old, time_old = 0, 0
    win_new, time_new = 0, 0
    for ep in range(n_episode_val):
        seed = random.randint(0, 922337203685)
        grid_config = GridConfig(num_agents=max_agent,  # количество агентов на карте
                                 size=64,  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
        n_rl_agent = random.randint(0, max_agent-1)
        res_old = play_game(grid_config, path_old_agent, path_old_agent, n_rl_agent)
        res_new = play_game(grid_config, path_new_agent, path_old_agent, n_rl_agent)

        win_old += res_old[0]
        time_old += res_old[1]
        win_new += res_new[0]
        time_new += res_new[1]

    k = 0
    if win_old <= win_new and time_old >= time_new:
        k = time_old / time_new
    return k, win_old, time_old, win_new, time_new

if __name__ == '__main__':

    n_action = 5
    min_agent = 2
    max_agent = 10
    lr = 0.0001
    n_episode = 50
    n_episode_val = 5
    gamma = 0.99
    n_generation = 100
    i_generation = 1
    path_old_agent = 'model_best.pth'
    print(f'generation {i_generation}')
    while True:

        path_new_agent = f'model_rl_{(i_generation)}.pth'
        if not os.path.exists(path_new_agent):
            shutil.copyfile(path_old_agent, path_new_agent)

        reinforce(n_episode, gamma, path_new_agent, path_old_agent)

        score, win_old, time_old, win_new, time_new = compare_two_agent(path_new_agent, path_old_agent)
        if score > 1.1:
            i_generation += 1
            path_old_agent = path_new_agent
            print(f'to born new star with score {score:.03f}')
            print('statistic', 'old:', win_old, time_old, 'new', win_new, time_new)
            print(f'generation {i_generation}')
        else:
            print(f'score {score:.03f}')
            print('statistic', 'old:', win_old, time_old, 'new', win_new, time_new)



        if i_generation == n_generation:
            break

    # counter = collections.Counter()
    # counter.update(total_reward_episode)
    # print(counter)
