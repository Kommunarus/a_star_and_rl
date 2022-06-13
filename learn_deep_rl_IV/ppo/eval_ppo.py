import torch
import numpy as np
from torch.optim import SGD, Adam, RMSprop
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal
import gym
from pogema import GridConfig
from ppo import PPO
import random

device = torch.device('cuda:0')

def play_game(config, path_new_agent):
    env = gym.make("Pogema-v0", grid_config=config)
    obs = env.reset()

    num_as = len(obs)
    our_agent = PPO(env, num_as, path_to_actor=path_new_agent)
    our_agent.actor.eval()
    our_agent.init_hidden(1)

    # Rewards this episode
    done = [False] * num_as
    rewards_game = [[] for _ in range(num_as)]
    batch_acts_old = []
    ep_t = 0
    finish = [False] * len(obs)

    while not all(done):
        if ep_t == 0:
            batch_acts_old.append([0] * len(obs))
        else:
            batch_acts_old.append(action)
        action, _ = our_agent.get_action(obs, batch_acts_old[-1], finish)
        obs, rew, done, _ = env.step(action)

        for robot in range(num_as):
            rewards_game[robot].append(rew[robot])

        for y, a_done in enumerate(done):
            if a_done == True:
                finish[y] = True

    target = [sum(x) for x in rewards_game]
    win = sum(target)
    csr = 1 if win == num_as else 0
    return win, csr

if __name__ == '__main__':

    n_agents = 60
    grid_config = GridConfig(num_agents=n_agents,
                             size=60,
                             density=0.3,
                             seed=None,
                             max_episode_steps=256,
                             obs_radius=5,
                             )
    win, csr = play_game(grid_config, 'ppo_actor.pth')
    print(win)