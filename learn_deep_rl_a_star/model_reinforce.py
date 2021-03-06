import torch.nn as nn
import torch
import random
from zero.model import Model
import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from learn_deep_a_star.ppo_net import PPOActor
import collections

device = torch.device('cuda:1')

class PolicyNetwork():
    def __init__(self, path_to_model='temp.pth', lr=0.001):

        self.solver = PPOActor(input_shape=3, dop_input_shape=5, rnn_hidden_dim=64, n_actions=5).to(device)
        self.solver.load_state_dict(torch.load(path_to_model))

        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr)


    def predict(self, s, a_old, h):
        a = torch.unsqueeze(a_old, 0).to(device)
        x, h_new = self.solver(s, a, h)
        out = nn.Softmax(1)(x)
        return torch.squeeze(out), h_new


    def update(self, returns, log_probs):
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(-log_prob * Gt)

        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, s, a_old, h):
        tensor_s = torch.unsqueeze(torch.from_numpy(s).to(torch.float32), 0)
        probs, h_new = self.predict(tensor_s.to(device), a_old, h.to(device))
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob, h_new

def reinforce(estimator, n_episode, gamma=1.0):
    for episode in range(n_episode):
        # if episode % 100 == 0:
        #     print('episode {}'.format(episode))

        num_agents = random.randint(10, 64)
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
        hidden_state = torch.zeros((1, 1, 64))
        a_old = 0

        done = [False for _ in range(len(state))]
        n_rl_agent = random.randint(0, num_agents-1)

        solver_star_A = Model()

        while True:
            ten_a_old = torch.zeros(5)
            ten_a_old[a_old] = 1

            action_rl, log_prob, hidden_state = estimator.get_action(state[n_rl_agent],
                                                                     ten_a_old,
                                                                     hidden_state)
            all_action = solver_star_A.act(state, done,
                               env.get_agents_xy_relative(),
                               env.get_targets_xy_relative())
            all_action[n_rl_agent] = action_rl

            next_state, reward, done, _ = env.step(all_action)

            # total_reward_episode[episode] += int(reward[n_rl_agent])
            log_probs.append(log_prob)
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
                estimator.update(returns, log_probs)
                break

            state = next_state
            a_old = action_rl

    torch.save(estimator.solver.state_dict(), 'temp.pth')

def play_game(config, policy, n_rl_agent):
    env = gym.make("Pogema-v0", grid_config=config)
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver_a_star = Model()
    steps = 0

    hidden_state = torch.zeros((1, 1, 64)).to(device)
    a_old = 0
    win = 0

    while not all(done):
        action = solver_a_star.act(obs, done,
                          env.get_agents_xy_relative(),
                          env.get_targets_xy_relative())

        ten_a_old = torch.zeros(5)
        ten_a_old[a_old] = 1
        pred, hidden_state = policy.solver(torch.unsqueeze(torch.from_numpy(obs[0]), 0).to(torch.float32).to(device),
                                    torch.unsqueeze(ten_a_old, 0).to(device),
                                    hidden_state)

        y2 = np.argmax(pred.detach().cpu().numpy(), 1)
        action[n_rl_agent] = y2[0]
        obs, reward, done, info = env.step(action)

        if done[n_rl_agent]:
            win = 1
            break

        steps += 1
        a_old = y2[0]


    return win, steps

def compare_two_agent(old_agent):
    policy_old = PolicyNetwork(old_agent)
    policy_new = PolicyNetwork()
    win_old, time_old = 0, 0
    win_new, time_new = 0, 0
    for ep in range(n_episode_val):
        seed = random.randint(0, 922337203685)
        grid_config = GridConfig(num_agents=64,  # количество агентов на карте
                                 size=64,  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
        n_rl_agent = random.randint(0, 63)
        res_old = play_game(grid_config, policy_old, n_rl_agent)
        res_new = play_game(grid_config, policy_new, n_rl_agent)

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
    lr = 0.001
    n_episode = 50
    n_episode_val = 5
    gamma = 0.99
    n_generation = 100
    i_generation = 0
    path_old_agent = 'model_best.pth'
    while True:
        print(f'generation {i_generation+1}')

        path_new_agent = f'model_rl_{(i_generation+1)}.pth'
        policy_net = PolicyNetwork(path_old_agent, lr)
        reinforce(policy_net, n_episode, gamma)

        score, win_old, time_old, win_new, time_new = compare_two_agent(path_old_agent)
        if score > 1.1:
            i_generation += 1
            torch.save(policy_net.solver.state_dict(), path_new_agent)

            path_old_agent = path_new_agent
            print(f'to born new star with score {score:.03f}')
        else:
            print(f'score {score:.03f}')
        print('statistic', 'old:', win_old, time_old, 'new', win_new, time_new)



        if i_generation == n_generation:
            break

    # counter = collections.Counter()
    # counter.update(total_reward_episode)
    # print(counter)
