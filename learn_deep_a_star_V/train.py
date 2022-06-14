import torch
import numpy as np
from torch.optim import SGD, Adam, RMSprop
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal
import gym
from pogema import GridConfig
import random
from zero.model import Model as modelastar
from learn_deep_rl_IV.ppo.ppo import PPO as ppo_deepa



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

    def forward(self, s, action_astar, action_deepa, hidden_state):
        aa_flat = self.feachextractor(s)
        input = torch.hstack([aa_flat, action_astar, action_deepa])
        x = F.relu(self.fc1(input))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, torch.unsqueeze(h.detach(), 1)

class PPOCritic(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim):
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

        self.fc1 = nn.Linear(11*11*16, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, 1)

    def forward(self, s, hidden_state):
        aa_flat = self.feachextractor(s)
        # input = torch.hstack([aa_flat, dopobs])
        x = F.relu(self.fc1(aa_flat))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, torch.unsqueeze(h.detach(), 1)

class Agent:
    def __init__(self):
        self.critic_rnn_hidden_dim = torch.zeros((1, 64), dtype=torch.float).to(device)
        self.actor_rnn_hidden_dim = torch.zeros((1, 64), dtype=torch.float).to(device)
        self.targets_xy = (0, 0)
        self.agents_xy = []

class PPO:
    def __init__(self, env, num_agents=None, path_to_actor=None, path_to_critic=None):
        self.env = env

        self.n_updates_per_iteration = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.beta = 0.005
        self.clip = 0.05
        self.lr_actor = 0.5e-5
        self.lr_critic = 0.5e-4
        self.num_agents = num_agents

        self.critic = PPOCritic(3, 64).to(device)
        if path_to_critic is not None:
            self.critic.load_state_dict(torch.load(path_to_critic))

        self.actor = PPOActor(3, 10, 64, 2).to(device)
        if path_to_actor is not None:
            self.actor.load_state_dict(torch.load(path_to_actor))

        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(Agent())

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)

        self.solver_astar = modelastar()

        self.our_agent = ppo_deepa(self.env, n_agents, path_to_actor='ppo_actor_IV.pth')
        self.our_agent.actor.eval()
        self.our_agent.init_hidden(1)



    def learn(self, max_time_steps):
        t_so_far = 0
        mm = 0
        while t_so_far < max_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, \
            batch_lens, batch_acts_old, batch_exps, win, \
            batch_acts_astar, batch_acts_deepa, batch_act_choice = self.rollout()

            t_so_far += np.sum(batch_lens)

            self.init_hidden(1)
            l1, l2, l3 = 0, 0, 0
            for i, agent in enumerate(self.agents):

                V, _, _ = self.evaluate(batch_obs[:, i, :], i,
                                        batch_acts_astar[:, i],
                                        batch_acts_deepa[:, i],
                                        batch_act_choice[:, i])
                traj_adv_v, traj_ref_v = self.calc_adv_ref(batch_rtgs[:, i], V, batch_exps[:, i])
                # A_k = batch_rtgs[:, i] - V.detach()
                # traj_adv_v = (traj_adv_v - traj_adv_v.mean()) / (traj_adv_v.std() + 1e-10)
                len_traj = len(traj_adv_v)

                for _ in range(self.n_updates_per_iteration):
                    self.init_hidden(1)
                    V, curr_log_probs, entropy = self.evaluate(batch_obs[:len_traj, i, :],
                                                      i,
                                                               batch_acts_astar[:len_traj, i],
                                                               batch_acts_deepa[:len_traj, i],
                                                               batch_act_choice[:len_traj, i])
                    ratios = torch.exp(curr_log_probs - batch_log_probs[:len_traj, i])
                    surr1 = ratios * traj_adv_v
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * traj_adv_v

                    a1 = (-torch.min(surr1, surr2)).mean()
                    a2 = self.beta*torch.mean(entropy)
                    actor_loss = a1 - a2
                    critic_loss = torch.nn.MSELoss()(V, traj_ref_v)

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                    l1 += a1.item()
                    l2 += a2.item()
                    l3 += critic_loss.item()
            mm += 1
            if mm % 1 == 0:
                torch.save(self.actor.state_dict(), 'ppo_actor_V.pth')
                torch.save(self.critic.state_dict(), 'ppo_critic_V.pth')
                print('\tloss actor: {:.03f}, entropy: {:.03f}, loss critic: {:.03f},'
                      ' win in game-train {}'.format(l1/self.n_updates_per_iteration,
                                                     l2/self.n_updates_per_iteration,
                                                     l3/self.n_updates_per_iteration, win))


    def get_action(self, obs, a_old, finish):
        actions = []
        actions_astar = []
        actions_deepa = []
        choices = []
        log_probs = []
        action_astar = self.solver_astar.act(obs, None, self.env.get_agents_xy_relative(),
                                             self.env.get_targets_xy_relative())

        action_deepa, _ = self.our_agent.get_action(obs, a_old, finish)


        for i, agent in enumerate(self.agents):
            if finish[i] == True:
                action_bot = 0
                astar = 0
                deepa = 0
                choice = 0
                log_prob = 0.9
            else:
                inp = torch.unsqueeze(torch.tensor(obs[i], dtype=torch.float), 0).to(device)

                astar_tensor = torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(action_astar[i]), 5), 0).to(device)
                deepa_tensor = torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(action_deepa[i]), 5), 0).to(device)

                act_bot, agent.actor_rnn_hidden_dim = self.actor(inp, astar_tensor, deepa_tensor, agent.actor_rnn_hidden_dim)

                act_bot_probs = torch.nn.Softmax(1)(act_bot)
                action_bot_choice = torch.multinomial(act_bot_probs, 1).item()
                log_prob = torch.log(act_bot_probs[0, action_bot_choice]).item()

                if action_bot_choice == 0:
                    action_bot = action_astar[i]
                    choice = 0
                else:
                    action_bot = action_deepa[i]
                    choice = 1

                astar = action_astar[i]
                deepa = action_deepa[i]



            actions.append(action_bot)
            actions_astar.append(astar)
            actions_deepa.append(deepa)
            choices.append(choice)
            log_probs.append(log_prob)
        return np.array(actions), np.array(log_probs), np.array(actions_astar), np.array(actions_deepa), np.array(choices)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_acts_astar = []
        batch_acts_deepa = []
        batch_acts_old = []
        batch_acts_choice = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_exp = []


        t = 0
        obs = self.env.reset()
        self.init_hidden(1)
        n_agents = len(obs)
        # Rewards this episode
        ep_rews = []
        done = [False] * n_agents
        finish = [False] * n_agents
        ep_t = 0
        rewards_game = [[] for _ in range(n_agents)]

        targets_xy = self.env.get_targets_xy_relative()
        agents_xy = self.env.get_agents_xy_relative()
        self.update_xy(targets_xy, agents_xy)

        while not all(done):
            if ep_t == 0:
                batch_acts_old.append([0] * n_agents)
            else:
                batch_acts_old.append(batch_acts[-1])
            batch_obs.append(obs)
            with torch.no_grad():
                action, log_prob, act_astar, act_deepa, act_choice = self.get_action(obs, batch_acts_old[-1], finish)
            obs, rew, done, _ = self.env.step(action)

            exp = [0] * n_agents
            for y, a_done in enumerate(done):
                if a_done == True:
                    if finish[y] == False:
                        exp[y] = 1
                        finish[y] = True
                    else:
                        exp[y] = 2

            batch_exp.append(exp)

            ep_rews.append(self.get_reward(rew, finish))
            batch_acts.append(action)
            batch_acts_astar.append(act_astar)
            batch_acts_deepa.append(act_deepa)
            batch_acts_choice.append(act_choice)
            batch_log_probs.append(log_prob)
            ep_t += 1
            t += 1
            targets_xy = self.env.get_targets_xy_relative()
            agents_xy = self.env.get_agents_xy_relative()
            self.update_xy(targets_xy, agents_xy)

            for robot in range(n_agents):
                rewards_game[robot].append(rew[robot])

        batch_lens.append(ep_t)
        batch_rews.append(ep_rews)

        target = [sum(x) for x in rewards_game]
        win = sum(target)

        # Reshape data as tensors in the shape specified before returning

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long).to(device)
        batch_acts_astar = torch.tensor(np.array(batch_acts_astar), dtype=torch.long).to(device)
        batch_acts_deepa = torch.tensor(np.array(batch_acts_deepa), dtype=torch.long).to(device)
        batch_acts_old = torch.tensor(np.array(batch_acts_old), dtype=torch.long).to(device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(device)
        batch_rtgs = self.compute_rtgs(batch_rews).to(device)
        batch_exps = torch.tensor(batch_exp, dtype=torch.long).to(device)
        batch_act_choices = torch.tensor(np.array(batch_acts_choice), dtype=torch.long).to(device)
        return (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_acts_old, batch_exps, win,
                batch_acts_astar, batch_acts_deepa, batch_act_choices)

    def get_reward(self, step_reward, finish):
        dist_agents = self.env.get_agents_xy_relative()
        rew = []
        for da, sr, f, agent in zip(dist_agents, step_reward, finish, self.agents):
            h_new = abs(da[0] - agent.targets_xy[0]) + abs(da[1] - agent.targets_xy[1])  # Manhattan distance as a heuristic function
            h_old = abs(agent.agents_xy[-1][0] - agent.targets_xy[0]) + \
                    abs(agent.agents_xy[-1][1] - agent.targets_xy[1])  # Manhattan distance as a heuristic function

            if sr == 1.0:
                rew.append(1)
            elif f == True:
                rew.append(0)
            elif h_new < h_old:
                rew.append(0.02)
            elif da in agent.agents_xy:
                rew.append(-0.02)
            else:
                rew.append(-0.01)

        return rew

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in batch_rews:
            for rew in ep_rews:
                batch_rtgs.append(rew)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def calc_adv_ref(self, rews, values_v, exps_v):
        values = values_v.squeeze().data.cpu().numpy()
        rewards = rews.squeeze().data.cpu().numpy()
        exps = exps_v.squeeze().data.cpu().numpy()
        last_gae = 0.0
        result_adv = []
        result_ref = []
        v = reversed(values[:-1])
        v_p1 = reversed(values[1:])
        rew = reversed(rewards[:-1])
        ex = reversed(exps[:-1])
        for val, next_val, reward, exp in zip(v, v_p1, rew, ex):
            if exp == 2:
                continue
            if exp == 1:
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + self.gamma * next_val - val
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
        ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
        return adv_v, ref_v

    def evaluate(self, batch_obs, num_agent, batch_astar, batch_deepa, batch_act_ch):
        V = []
        for i in range(batch_obs.shape[0]):
            s = batch_obs[i].unsqueeze(0)
            h_rnn = self.agents[num_agent].critic_rnn_hidden_dim
            v_step, self.agents[num_agent].critic_rnn_hidden_dim = self.critic(s, h_rnn)
            V.append(v_step.squeeze(0))

        V = torch.stack(V, 0)

        Lo = []
        entropy = []
        for i in range(batch_obs.shape[0]):
            s = batch_obs[i].unsqueeze(0)
            a_star = torch.nn.functional.one_hot(torch.unsqueeze(batch_astar[i], 0), 5)
            a_deep = torch.nn.functional.one_hot(torch.unsqueeze(batch_deepa[i], 0), 5)
            h_rnn = self.agents[num_agent].actor_rnn_hidden_dim

            act_bot, self.agents[num_agent].actor_rnn_hidden_dim = self.actor(s, a_star, a_deep, h_rnn)
            act_bot_probs = torch.nn.Softmax(1)(act_bot)
            log_prob = torch.log(act_bot_probs[0, batch_act_ch[i]])
            # A.append(act_bot)
            Lo.append(log_prob)
            entropy.append(-torch.sum(act_bot_probs*torch.log(act_bot_probs)))

        log_prob = torch.stack(Lo, 0)
        entr = torch.stack(entropy, 0)

        return V.squeeze(1), log_prob, entr

    def init_hidden(self, episode_num):
        for i in range(self.num_agents):
            self.agents[i].actor_rnn_hidden_dim = torch.zeros((episode_num, 64)).to(device)
            self.agents[i].critic_rnn_hidden_dim = torch.zeros((episode_num, 64)).to(device)
            self.agents[i].targets_xy = (0, 0)
            self.agents[i].agents_xy = []

    def update_xy(self, targets_xy, agents_xy):
        for i in range(self.num_agents):
            self.agents[i].targets_xy = targets_xy[i]
            self.agents[i].agents_xy.append(agents_xy[i])




if __name__ == '__main__':

    n_agents = 60
    grid_config = GridConfig(num_agents=n_agents,
                             size=60,
                             density=0.3,
                             seed=None,
                             max_episode_steps=256,
                             obs_radius=5,
                             )
    env = gym.make("Pogema-v0", grid_config=grid_config)

    # model = PPO(env, n_agents, path_to_actor='ppo_actor.pth', path_to_critic='ppo_critic.pth')
    model = PPO(env, n_agents)
    model.learn(10_000_000)

