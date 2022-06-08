import torch
import numpy as np
from torch.optim import SGD, Adam, RMSprop
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal
import gym
from pogema import GridConfig

device = torch.device('cuda:1')

class PPOActor(nn.Module):
    def __init__(self, input_shape, dop_input_shape, rnn_hidden_dim, n_actions):
        super(PPOActor, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.feachextractor = nn.Sequential(
            nn.Conv2d(input_shape, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(1)
        )

        self.fc1 = nn.Linear(11*11*16 + dop_input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, n_actions)

    def forward(self, s, dopobs, hidden_state):
        aa_flat = self.feachextractor(s)
        input = torch.hstack([aa_flat, dopobs])
        x = F.relu(self.fc1(input))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, torch.unsqueeze(h.detach(), 1)

class PPOCritic(nn.Module):
    def __init__(self, input_shape, dop_input_shape, rnn_hidden_dim):
        super(PPOCritic, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.feachextractor = nn.Sequential(
            nn.Conv2d(input_shape, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(1)
        )

        self.fc1 = nn.Linear(11*11*16 + dop_input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, 1)

    def forward(self, s, dopobs, hidden_state):
        aa_flat = self.feachextractor(s)
        input = torch.hstack([aa_flat, dopobs])
        x = F.relu(self.fc1(input))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, torch.unsqueeze(h.detach(), 1)

class Agent:
    def __init__(self):
        self.critic_rnn_hidden_dim = torch.zeros((1, 64), dtype=torch.float).to(device)
        self.actor_rnn_hidden_dim = torch.zeros((1, 64), dtype=torch.float).to(device)
        self.targets_xy = -1000
        self.agents_xy = []

class PPO:
    def __init__(self, env, num_agents=None, path_to_actor=None, path_to_critic=None):
        self.env = env

        self.n_updates_per_iteration = 10
        self.gamma = 0.99
        self.clip = 0.2
        self.lr = 0.0001
        self.obs_dim = (3, 11, 11)
        self.act_dim = 5
        self.num_agents = num_agents

        self.critic = PPOCritic(3, 5, 64).to(device)
        if path_to_critic is not None:
            self.critic.load_state_dict(torch.load(path_to_critic))

        self.actor = PPOActor(3, 5, 64, 5).to(device)
        if path_to_actor is not None:
            self.actor.load_state_dict(torch.load(path_to_actor))

        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(Agent())

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)


    def learn(self, max_time_steps):
        t_so_far = 0
        mm = 0
        while t_so_far < max_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_acts_old = self.rollout()
            t_so_far += np.sum(batch_lens)

            self.init_hidden(1)
            for i, agent in enumerate(self.agents):

                V, _ = self.evaluate(batch_obs[:, i,:], batch_acts[:, i], batch_acts_old[:, i], i)
                A_k = batch_rtgs[:, i] - V.detach()
                # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                for _ in range(self.n_updates_per_iteration):
                    self.init_hidden(1)
                    V, curr_log_probs = self.evaluate(batch_obs[:, i,:], batch_acts[:, i], batch_acts_old[:, i], i)
                    ratios = torch.exp(curr_log_probs - batch_log_probs[:, i])
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = torch.nn.MSELoss()(V, batch_rtgs[:, i])

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
            mm += 1
            if mm % 1 == 0:
                torch.save(self.actor.state_dict(), 'ppo_actor.pth')
                torch.save(self.critic.state_dict(), 'ppo_critic.pth')
                grid_config = GridConfig(num_agents=64,  # количество агентов на карте
                                         size=64,  # размеры карты
                                         density=0.3,  # плотность препятствий
                                         seed=None,  # сид генерации задания
                                         max_episode_steps=256,  # максимальная длина эпизода
                                         obs_radius=5,  # радиус обзора
                                         )

                out = play_game(grid_config, 'ppo_actor.pth')
                print(out)


    def get_action(self, obs, a_old):
        actions = []
        log_probs = []
        for i, agent in enumerate(self.agents):
            inp = torch.unsqueeze(torch.tensor(obs[i], dtype=torch.float), 0).to(device)
            aold_tensor = torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(a_old[i]), 5), 0).to(device)

            act_bot, agent.actor_rnn_hidden_dim = self.actor(inp, aold_tensor, agent.actor_rnn_hidden_dim)

            act_bot_probs = torch.nn.Softmax(1)(act_bot)
            action_bot = torch.multinomial(act_bot_probs, 1).item()
            log_prob = torch.log(act_bot_probs[0, action_bot])

            actions.append(action_bot)
            log_probs.append(log_prob.item())
        return np.array(actions), np.array(log_probs)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_acts_old = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []


        t = 0
        obs = self.env.reset()
        self.init_hidden(1)

        # Rewards this episode
        ep_rews = []
        done = [False] * len(obs)
        finish = [False] * len(obs)
        ep_t = 0

        targets_xy = self.env.get_targets_xy_relative()
        agents_xy = self.env.get_agents_xy_relative()
        self.update_xy(targets_xy, agents_xy)

        while not all(done):
            if ep_t == 0:
                batch_acts_old.append([0] * len(obs))
            else:
                batch_acts_old.append(batch_acts[-1])
            batch_obs.append(obs)
            with torch.no_grad():
                action, log_prob = self.get_action(obs, batch_acts_old[-1])
            obs, rew, done, _ = self.env.step(action)


            for y, a_done in enumerate(done):
                if a_done == True:
                    finish[y] = True

            ep_rews.append(self.get_reward(rew, finish))
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            ep_t += 1
            t += 1
            targets_xy = self.env.get_targets_xy_relative()
            agents_xy = self.env.get_agents_xy_relative()
            self.update_xy(targets_xy, agents_xy)


        batch_lens.append(ep_t)
        batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long).to(device)
        batch_acts_old = torch.tensor(np.array(batch_acts_old), dtype=torch.long).to(device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(device)
        batch_rtgs = self.compute_rtgs(batch_rews).to(device)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_acts_old

    def get_reward(self, step_reward, finish):
        dist_agents = self.env.get_agents_xy_relative()
        rew = []
        for da, sr, f, agent in zip(dist_agents, step_reward, finish, self.agents):
            if sr == 1.0:
                rew.append(100)
            elif da in agent.agents_xy:
                rew.append(-5)
            # elif abs(dt[0]) + abs(dt[1]) > abs(agent.targets_xy[0]) + abs(agent.targets_xy[1]):
            #     rew.append(-5)
            elif f == True:
                rew.append(0)
            else:
                rew.append(-1)

        return rew

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = [0] * len(ep_rews[0]) # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = [x + y * self.gamma for x, y in zip(rew, discounted_reward)]
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts, batch_acts_old, num_agent):
        V = []
        for i in range(batch_obs.shape[0]):
            s = batch_obs[i].unsqueeze(0)
            a_o = torch.nn.functional.one_hot(torch.unsqueeze(batch_acts_old[i], 0), 5)
            h_rnn = self.agents[num_agent].critic_rnn_hidden_dim

            v_step, self.agents[num_agent].critic_rnn_hidden_dim = self.critic(s, a_o, h_rnn)
            V.append(v_step.squeeze(0))

        V = torch.stack(V, 0)

        Lo = []
        for i in range(batch_obs.shape[0]):
            s = batch_obs[i].unsqueeze(0)
            a_o = torch.nn.functional.one_hot(torch.unsqueeze(batch_acts_old[i], 0), 5)
            h_rnn = self.agents[num_agent].actor_rnn_hidden_dim

            act_bot, self.agents[num_agent].actor_rnn_hidden_dim = self.actor(s, a_o, h_rnn)
            act_bot_probs = torch.nn.Softmax(1)(act_bot)
            log_prob = torch.log(act_bot_probs[0, batch_acts[i]])
            # A.append(act_bot)
            Lo.append(log_prob)

        log_prob = torch.stack(Lo, 0)

        return V.squeeze(1), log_prob

    def init_hidden(self, episode_num):
        for i in range(self.num_agents):
            self.agents[i].actor_rnn_hidden_dim = torch.zeros((episode_num, 64)).to(device)
            self.agents[i].critic_rnn_hidden_dim = torch.zeros((episode_num, 64)).to(device)
            self.agents[i].targets_xy = -1000
            self.agents[i].agents_xy = []

    def update_xy(self, targets_xy, agents_xy):
        for i in range(self.num_agents):
            self.agents[i].targets_xy = targets_xy[i]
            self.agents[i].agents_xy.append(agents_xy[i])


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
    while not all(done):
        if ep_t == 0:
            batch_acts_old.append([0] * len(obs))
        else:
            batch_acts_old.append(action)
        action, _ = our_agent.get_action(obs, batch_acts_old[-1])
        obs, rew, done, _ = env.step(action)

        for robot in range(num_as):
            rewards_game[robot].append(rew[robot])

    target = [sum(x) for x in rewards_game]
    win = sum(target)
    csr = 1 if win == num_as else 0
    return win, csr


if __name__ == '__main__':

    n_agents = 4
    grid_config = GridConfig(num_agents=n_agents,
                             size=64,
                             density=0.3,
                             seed=None,
                             max_episode_steps=256,
                             obs_radius=5,
                             )
    env = gym.make("Pogema-v0", grid_config=grid_config)

    model = PPO(env, n_agents, path_to_actor='model_50.pth')
    model.learn(1_000_000)




