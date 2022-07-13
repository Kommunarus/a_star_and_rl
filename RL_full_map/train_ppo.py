import torch
import numpy as np
from torch.optim import SGD, Adam, RMSprop
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal
import gym
from pogema import GridConfig
import random
from model_astar import Model as astarmodel
import argparse


class PPOActor(nn.Module):
    def __init__(self, input_shape, dop_input_shape, rnn_hidden_dim, n_actions):
        super(PPOActor, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.feachextractor = nn.Sequential(
            nn.Conv2d(input_shape, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.Flatten(1)
        )

        self.fc1 = nn.Linear(3*3*32 + dop_input_shape, self.rnn_hidden_dim)
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
    def __init__(self, input_shape, rnn_hidden_dim):
        super(PPOCritic, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.feachextractor = nn.Sequential(
            nn.Conv2d(input_shape, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.Flatten(1)
        )

        self.fc1 = nn.Linear(3*3*32, self.rnn_hidden_dim)
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
        self.lenmap = 70
        self.fullmap = np.zeros((2 * self.lenmap + 1, 2 * self.lenmap + 1), dtype=np.uint8)


class PPO:
    def __init__(self, path_to_actor=None, path_to_critic=None, rew_list=None, id_save=''):
        # self.env = env

        self.n_updates_per_iteration = 5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.beta = 0.005
        self.clip = 0.15
        self.lr_actor = 1e-5
        self.lr_critic = 1e-4
        # self.num_agents = num_agents

        self.critic = PPOCritic(3, 64).to(device)
        if path_to_critic is not None:
            self.critic.load_state_dict(torch.load(path_to_critic))

        self.actor = PPOActor(3, 5, 64, 5).to(device)
        if path_to_actor is not None:
            self.actor.load_state_dict(torch.load(path_to_actor))

        # self.agents = []
        # for i in range(self.num_agents):
        #     self.agents.append(Agent())

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)

        self.rew_list = rew_list
        self.id_save = id_save

    def learn(self, max_time_steps):
        # all_map = [(8, 32), (16, 32), (32, 32),(64, 32),(16, 64), (32, 64), (64, 64), (128, 64)]
        all_map = [(8, 16), (16, 16), (32, 16),
                   (8, 32), (16, 32), (32, 32), #(64, 32), (128, 32),
                   (8, 64), (16, 64), (32, 64), #(64, 64), (128, 64),
                   ]

        t_so_far = 0
        mm = 0
        last_isr = 0
        count_repeat = 0
        first = True
        while mm < max_time_steps:
            if last_isr != 1 and count_repeat <= -11 and not first:
                # >AB02;O5< :0@BC
                repeat = True
                count_repeat += 1
                pass
            else:
                repeat = False
                first = False
                count_repeat = 0
                # print('-'.join(['']*100))

                map = random.choice(all_map)
                # map = random.choices(all_map, weights=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2], k=1)[0]
                n_agents = map[0]
                self.size_map = map[1]
                self.num_agents = n_agents
                grid_config = GridConfig(num_agents=n_agents,
                                         size=map[1],
                                         density=0.3,
                                         seed=None,
                                         max_episode_steps=256,
                                         obs_radius=5,
                                         )
                env = gym.make("Pogema-v0", grid_config=grid_config)

                self.env = env
            self.agents = []
            for i in range(self.num_agents):
                self.agents.append(Agent())
            self.solver = astarmodel()

            self.actor.train()

            batch_obs_np, batch_acts, batch_log_probs, batch_rtgs, \
            batch_lens, batch_acts_old, batch_exps, win = self.rollout()
            t_so_far += np.sum(batch_lens)

            self.init_hidden(1)
            l1, l2, l3 = 0, 0, 0
            for i, agent in enumerate(self.agents):
                batch_obs = torch.tensor(batch_obs_np[:, i, :], dtype=torch.float).to(device)
                V, _, _ = self.evaluate(batch_obs[:, :], batch_acts[:, i], batch_acts_old[:, i], i)
                traj_adv_v, traj_ref_v = self.calc_adv_ref(batch_rtgs[:, i], V, batch_exps[:, i])
                len_traj = len(traj_adv_v)
                # traj_adv_v = traj_adv_v  / torch.std(traj_adv_v)

                for _ in range(self.n_updates_per_iteration):
                    self.init_hidden(1)
                    V, curr_log_probs, entropy = self.evaluate(batch_obs[:len_traj, :],
                                                               batch_acts[:len_traj, i],
                                                               batch_acts_old[:len_traj, i], i)
                    ratios = torch.exp(curr_log_probs - batch_log_probs[:len_traj, i])
                    surr1 = ratios * traj_adv_v
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * traj_adv_v

                    a1 = (-torch.min(surr1, surr2)).mean()
                    a2 = self.beta * torch.mean(entropy)
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
            last_isr = win / n_agents
            if mm % 1 == 0:
                torch.save(self.actor.state_dict(), f'ppo_actor_{self.id_save}.pth')
                torch.save(self.critic.state_dict(), f'ppo_critic_{self.id_save}.pth')
                print(
                    '{}. isr {:.03f} / ({:d}, {:d}) size {}\t\t loss actor: {:.03f}, entropy: {:.03f}, loss critic: {:.03f},'
                    ''.format(mm, win / n_agents, int(win), int(n_agents), map[1],
                              l1 / self.n_updates_per_iteration,
                              l2 / self.n_updates_per_iteration,
                              l3 / self.n_updates_per_iteration,
                              ))

    def get_action(self, obs, a_old, finish):
        actions = []
        log_probs = []
        for i, agent in enumerate(self.agents):
            if finish[i] == True:
                action_bot = 0
                log_prob = 0.9
            else:
                inp = torch.unsqueeze(torch.tensor(obs[i], dtype=torch.float), 0).to(device)
                aold_tensor = torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(a_old[i]), 5), 0).to(device)

                act_bot, agent.actor_rnn_hidden_dim = self.actor(inp, aold_tensor, agent.actor_rnn_hidden_dim)

                act_bot_probs = torch.nn.Softmax(1)(act_bot)
                action_bot = torch.multinomial(act_bot_probs, 1).item()
                log_prob = torch.log(act_bot_probs[0, action_bot]).item()

            actions.append(action_bot)
            log_probs.append(log_prob)
        return np.array(actions), np.array(log_probs)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_acts_old = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_exp = []

        t = 0
        obs = self.env.reset()
        # print(len(obs))
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

            newobs = self.update_obstacles(obs)
            batch_obs.append(newobs)
            all_env = self.env.get_obstacles()
            all_agents = self.env.get_agents_xy()

            action_classic = self.solver.act(obs, done, agents_xy, targets_xy, all_env, all_agents, self.size_map)

            with torch.no_grad():
                action, log_prob = self.get_action(newobs, batch_acts_old[-1], finish, )
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
            ep_rews.append(self.get_reward(rew, finish, action, action_classic))
            batch_acts.append(action)
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

        # batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(device)
        batch_obs = np.array(batch_obs, dtype=np.uint8)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long).to(device)
        batch_acts_old = torch.tensor(np.array(batch_acts_old), dtype=torch.long).to(device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(device)
        batch_rtgs = self.compute_rtgs(batch_rews).to(device)
        batch_exps = torch.tensor(batch_exp, dtype=torch.long).to(device)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_acts_old, batch_exps, win

    def update_obstacles(self, obs):
        newobs = []
        for i, o in enumerate(obs):
            current_map = self.agents[i].fullmap
            current_xy = self.agents[i].agents_xy[-1]
            d0 = self.agents[i].lenmap + current_xy[0] - 5
            d1 = self.agents[i].lenmap + current_xy[1] - 5
            current_map[d0: d0 + 11, d1: d1 + 11] = o[0]

            map_agent = np.zeros((2 * self.agents[i].lenmap + 1, 2 * self.agents[i].lenmap + 1), dtype=np.uint8)
            map_agent[d0: d0 + 11, d1: d1 + 11] = o[1]

            map_target = np.zeros((2 * self.agents[i].lenmap + 1, 2 * self.agents[i].lenmap + 1), dtype=np.uint8)
            # map_target[d0: d0 + 11, d1: d1 + 11] = o[2]
            map_target[self.agents[i].lenmap + self.agents[i].targets_xy[0],
                       self.agents[i].lenmap + self.agents[i].targets_xy[1]] = 1

            newobs.append(np.stack([current_map, map_agent, map_target]))

        return newobs

    def get_reward(self, step_reward, finish, action, action_classic):
        dist_agents = self.env.get_agents_xy_relative()
        rew = []
        have_good_f = False
        step = 0
        for da, sr, f, agent, act, act_class in zip(dist_agents, step_reward, finish, self.agents, action,
                                                    action_classic):
            # for da, sr, f, agent, act, act_class in zip(dist_agents, step_reward, finish, self.agents, action,
            #                                             action_classic):
            # h_new = abs(da[0] - agent.targets_xy[0]) + abs(
            #     da[1] - agent.targets_xy[1])  # Manhattan distance as a heuristic function
            # h_old = abs(agent.agents_xy[-1][0] - agent.targets_xy[0]) + \
            #         abs(agent.agents_xy[-1][1] - agent.targets_xy[1])
            #     koeff = 2 * step / 255 + 1
            if sr == 1.0:
                rew.append(self.rew_list[0])
                # have_good_f = True
            # else:
            #     rew.append(0)
            elif f == True:
                rew.append(0)
            # elif da in agent.agents_xy:
            #     rew.append(-self.rew_list[1])
            # rew.append(-0.02 * sum([1 for x in agent.agents_xy if x == da]))
            elif act == act_class:
                rew.append(self.rew_list[1])
            # elif (h_new < h_old):  # and (not (da in agent.agents_xy)):
            #     rew.append(self.rew_list[3])
            else:
                rew.append(-self.rew_list[2])

            step += 1
        # if not have_good_f:
        #     rew[-1] = -1
        return rew

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in batch_rews:
            for rew in ep_rews:
                batch_rtgs.append(rew)
        # for ep_rews in reversed(batch_rews):
        #     discounted_reward = [0] * len(ep_rews[0]) # The discounted reward so far
        #     for rew in reversed(ep_rews):
        #         discounted_reward = [x + y * self.gamma for x, y in zip(rew, discounted_reward)]
        #         batch_rtgs.insert(0, discounted_reward)
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

    def evaluate(self, batch_obs, batch_acts, batch_acts_old, num_agent):
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
            a_o = torch.nn.functional.one_hot(torch.unsqueeze(batch_acts_old[i], 0), 5)
            h_rnn = self.agents[num_agent].actor_rnn_hidden_dim

            act_bot, self.agents[num_agent].actor_rnn_hidden_dim = self.actor(s, a_o, h_rnn)
            act_bot_probs = torch.nn.Softmax(1)(act_bot)
            log_prob = torch.log(act_bot_probs[0, batch_acts[i]])
            # A.append(act_bot)
            Lo.append(log_prob)
            entropy.append(-torch.sum(act_bot_probs * torch.log(act_bot_probs)))

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-game', type=int, default=100_000)
    parser.add_argument('--id', default='second')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--reward', default='1,0.01,0.01')  # total, -xy, astar, new<old, -other

    arg = parser.parse_args()
    device = torch.device(f'cuda:{arg.cuda}')
    # print(arg.reward)

    rew = [float(x) for x in arg.reward.split(',')]

    model = PPO(path_to_actor='ppo_actor_first.pth', path_to_critic='ppo_critic_first.pth',
                rew_list=rew, id_save=arg.id)
    # model = PPO(env, n_agents, path_to_actor='model_50.pth')
    model.learn(arg.n_game)

