import torch
import numpy as np
from torch.optim import SGD, Adam, RMSprop
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal
import gym
from pogema import GridConfig
import random
import argparse
import copy
from dataset import Model


max_size = 64
size_look = 15

class PPOActor(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super(PPOActor, self).__init__()
        flayer = [nn.Conv2d(17, 192, 3, 1, 1), nn.ReLU(), ]
        slayer = []
        for i in range(6):
            slayer = slayer + [nn.Conv2d(192, 192, 3, 1, 1), nn.ReLU(), ]

        ml = flayer + slayer + [nn.Conv2d(192, 1, 1, 1, 0), nn.ReLU(), nn.Flatten(1), ]

        fa_actor = nn.ModuleList(ml)

        self.hidden_dim = hidden_dim
        self.feachextractor = fa_actor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

        self.fc1 = nn.Linear(31 * 31 * 1, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, n_actions)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, s):
        x = self.feachextractor[0](s)
        for l in self.feachextractor[1:]:
            x = l(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PPOCritic(nn.Module):
    def __init__(self, hidden_dim):
        super(PPOCritic, self).__init__()
        flayer = [nn.Conv2d(17, 192, 3, 1, 1), nn.ReLU(), ]
        slayer = []
        for i in range(6):
            slayer = slayer + [nn.Conv2d(192, 192, 3, 1, 1), nn.ReLU(), ]

        ml = flayer + slayer + [nn.Conv2d(192, 1, 1, 1, 0), nn.ReLU(), nn.Flatten(1), ]

        fa_actor = nn.ModuleList(ml)

        self.hidden_dim = hidden_dim
        self.feachextractor = fa_actor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

        self.fc1 = nn.Linear(31 * 31 * 1, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, s):
        x = self.feachextractor[0](s)
        for l in self.feachextractor[1:]:
            x = l(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent:
    def __init__(self):
        self.lenmap = max_size + 2*size_look
        self.zabor_map = np.zeros((2 *self.lenmap + 1, 2 * self.lenmap + 1), dtype=np.uint8)


class PPO:
    def __init__(self, path_to_actor=None, path_to_critic=None, id_save=''):
        # self.env = env

        self.n_updates_per_iteration = 5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.beta = 0.005
        self.clip = 0.1
        self.lr_actor = 1e-5
        self.lr_critic = 1e-4
        # self.num_agents = num_agents

        self.critic = PPOCritic(128).to(device)
        if path_to_critic is not None:
            self.critic.load_state_dict(torch.load(path_to_critic))

        self.actor = PPOActor(128, 5).to(device)
        if path_to_actor is not None:
            self.actor.load_state_dict(torch.load(path_to_actor))


        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)

        self.id_save = id_save

    def learn(self, max_time_steps):
        # all_map = [(8, 32), (16, 32), (32, 32),(64, 32),(16, 64), (32, 64), (64, 64), (128, 64)]
        all_map = [(8, 16), (16, 16), (32, 16),
                   (8, 32), (16, 32), (32, 32), (64, 32), (128, 32)
                   (8, 64), (16, 64), (32, 64), (64, 64), (128, 64), (256, 64),
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

                # map = (64, max_size)
                # map = (random.randint(8, 32), random.randint(16, max_size))
                map = random.choice(all_map)
                # map = random.choices(all_map, weights=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2], k=1)[0]
                n_agents = map[0]
                self.size_map = map[1]
                self.num_agents = n_agents
                grid_config = GridConfig(num_agents=n_agents,
                                         size=map[1],
                                         density=random.uniform(0.2, 0.4),
                                         seed=None,
                                         max_episode_steps=256,
                                         obs_radius=5,
                                         )
                env = gym.make("Pogema-v0", grid_config=grid_config)

                self.env = env
            self.agents = []
            for i in range(self.num_agents):
                self.agents.append(Agent())

            self.actor.train()

            batch_obs_np, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_exps, win = self.rollout()
            t_so_far += np.sum(batch_lens)

            l1, l2, l3 = 0, 0, 0
            for i, agent in enumerate(self.agents):
                batch_obs = torch.tensor(batch_obs_np[:, i, :], dtype=torch.float).to(device)
                V, _, _ = self.evaluate(batch_obs[:, :], batch_acts[:, i], i)
                traj_adv_v, traj_ref_v = self.calc_adv_ref(batch_rtgs[:, i], V, batch_exps[:, i])
                len_traj = len(traj_adv_v)
                # traj_adv_v = traj_adv_v  / torch.std(traj_adv_v)

                for _ in range(self.n_updates_per_iteration):
                    V, curr_log_probs, entropy = self.evaluate(batch_obs[:len_traj, :],
                                                               batch_acts[:len_traj, i], i)
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

    def get_action(self, obs, finish):
        actions = []
        log_probs = []
        for i, agent in enumerate(self.agents):
            if finish[i] == True:
                action_bot = 0
                log_prob = 0.9
            else:
                inp = torch.unsqueeze(torch.tensor(obs[i], dtype=torch.float), 0).to(device)

                act_bot = self.actor(inp)

                act_bot_probs = torch.nn.Softmax(1)(act_bot)
                action_bot = torch.multinomial(act_bot_probs, 1).item()
                log_prob = torch.log(act_bot_probs[0, action_bot]).item()

            actions.append(action_bot)
            log_probs.append(log_prob)
        return np.array(actions), np.array(log_probs)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_exp = []

        t = 0
        obs = self.env.reset()
        # print(len(obs))
        n_agents = len(obs)
        # Rewards this episode
        ep_rews = []
        done = [False] * n_agents
        finish = [False] * n_agents
        ep_t = 0
        rewards_game = [[] for _ in range(n_agents)]

        history_dict = {}
        solver = Model()
        step = 0

        while not all(done):

            targets_xy = self.env.get_targets_xy_relative()
            agents_xy = self.env.get_agents_xy_relative()
            action, dist_agents = solver.act(obs, agents_xy, targets_xy)

            newobs = self.update_obstacles(obs, agents_xy, targets_xy, history_dict, dist_agents, step)
            batch_obs.append(newobs)

            with torch.no_grad():
                action, log_prob = self.get_action(newobs, finish, )
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
            batch_log_probs.append(log_prob)
            ep_t += 1
            t += 1
            step += 1


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
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(device)
        batch_rtgs = self.compute_rtgs(batch_rews).to(device)
        batch_exps = torch.tensor(batch_exp, dtype=torch.long).to(device)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_exps, win

    def update_obstacles(self, obs, currentxy, targetsxy, historydict, distagents, step):
        newobs = []
        for i, (o, current_xy, targets_xy, dist_agents) in \
                enumerate(zip(obs, currentxy, targetsxy, distagents)):

            centr = max_size + 2 * size_look
            zabor_map = self.agents[i].zabor_map

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

            map_1 = zabor_map[x1: x2, y1: y2]
            map_2 = map_all_agents[x1: x2, y1: y2]
            map_3 = svobodno_map[x1: x2, y1: y2]
            map_4 = np.zeros((size_look * 2 + 1, size_look * 2 + 1), dtype=np.uint8)
            map_5 = np.ones((size_look * 2 + 1, size_look * 2 + 1), dtype=np.uint8)
            map_6 = map_target[x1: x2, y1: y2]
            map_7 = map_target_local[x1: x2, y1: y2]
            map_8 = map_start[x1: x2, y1: y2]
            map_9 = dist_agents[x1: x2, y1: y2]



            hist_map = []
            for mmm in hist:
                hist_map.append(mmm[x1: x2, y1: y2])
            end_numpy = np.stack([map_1, map_2, map_3, map_4, map_5, map_6, map_7, map_8, map_9] + hist_map)

            end_numpy = end_numpy.astype(np.float64)
            nz = np.nonzero(end_numpy[8])
            if len(nz[0]) != 0:
                max_e = np.max(end_numpy[8])
                min_e = np.min(end_numpy[8][nz])
                mask = end_numpy[8] != 0
                for i in range(2 * size_look + 1):
                    for j in range(2 * size_look + 1):
                        if max_e - min_e != 0:
                            if mask[i, j]:
                                end_numpy[8][i, j] = (max_e - end_numpy[8][i, j]) / (max_e - min_e)
                        else:
                            end_numpy[8][i, j] = 1


            newobs.append(end_numpy)

        return newobs

    def get_reward(self, step_reward, finish):
        rew = []
        for sr, f in zip(step_reward, finish):
            if sr == 1.0:
                rew.append(1)
            # elif f == True:
            #     rew.append(0)
            else:
                rew.append(0)

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

    def evaluate(self, batch_obs, batch_acts, num_agent):
        V = []
        for i in range(batch_obs.shape[0]):
            s = batch_obs[i].unsqueeze(0)
            v_step = self.critic(s)
            V.append(v_step.squeeze(0))

        V = torch.stack(V, 0)

        Lo = []
        entropy = []
        for i in range(batch_obs.shape[0]):
            s = batch_obs[i].unsqueeze(0)

            act_bot = self.actor(s)
            act_bot_probs = torch.nn.Softmax(1)(act_bot)
            log_prob = torch.log(act_bot_probs[0, batch_acts[i]])
            # A.append(act_bot)
            Lo.append(log_prob)
            entropy.append(-torch.sum(act_bot_probs * torch.log(act_bot_probs)))

        log_prob = torch.stack(Lo, 0)
        entr = torch.stack(entropy, 0)

        return V.squeeze(1), log_prob, entr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-game', type=int, default=100_000)
    parser.add_argument('--id', default='first')
    parser.add_argument('--cuda', type=int, default=1)

    arg = parser.parse_args()
    device = torch.device(f'cuda:{arg.cuda}')
    # print(arg.reward)


    model = PPO(id_save=arg.id, path_to_actor='model_best.pth')
    # model = PPO(path_to_actor='ppo_actor_second.pth', path_to_critic='ppo_critic_second.pth',
    #             rew_list=rew, id_save=arg.id)
    # model = PPO(env, n_agents, path_to_actor='model_50.pth')
    model.learn(arg.n_game)

