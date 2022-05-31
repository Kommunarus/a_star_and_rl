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

class PolicyNetwork():
    def __init__(self, lr=0.001):

        self.solver = PPOActor(input_shape=3, dop_input_shape=5, rnn_hidden_dim=64, n_actions=5)
        self.solver.load_state_dict(torch.load('../learn_deep_a_star/model_best.pth'))

        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr)


    def predict(self, s, a_old, h):
        a = torch.unsqueeze(a_old, 0)
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
        probs, h_new = self.predict(tensor_s, a_old, h)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob, h_new

def reinforce(estimator, n_episode, gamma=1.0):
    for episode in range(n_episode):
        if episode % 100 == 0:
            print('episode {}'.format(episode))

        num_agents = 10 #random.randint(10, 64)
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

            total_reward_episode[episode] += int(reward[n_rl_agent])
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


if __name__ == '__main__':

    n_train = 1
    n_state = (3, 11, 11)
    n_action = 5
    n_hidden = 128
    lr = 0.0001
    n_episode = 500
    gamma = 0.99

    policy_net = PolicyNetwork(lr)
    total_reward_episode = [0] * n_episode
    reinforce(policy_net, n_episode, gamma)

    torch.save(policy_net.solver.state_dict(), 'model_rl.pth')

    counter = collections.Counter()
    counter.update(total_reward_episode)
    print(counter)
