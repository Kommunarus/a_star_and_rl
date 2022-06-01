from network import PPOActor, PPOCritic
import torch
from torch.optim import Adam
import numpy as np

device = torch.device('cuda:1')

class PPO:
    def __init__(self, env, num_agents, path_to_actor_bot=None, path_to_actor=None, path_to_critic=None):
        self.env = env
        self.num_agents = num_agents

        self.timesteps_per_batch = 4000
        self.max_timesteps_per_episode = 256
        self.n_updates_per_iteration = 10
        self.gamma = 0.97
        self.clip = 0.2
        self.lr = 0.0001

        self.actor = PPOActor(64).to(device)
        self.critic = PPOCritic(64).to(device)


        if path_to_actor is not None:
            self.actor.load_state_dict(torch.load(path_to_actor))
        if path_to_critic is not None:
            self.critic.load_state_dict(torch.load(path_to_critic))

        self.actors = [self.actor, ]
        for i in range(self.num_agents-1):
            actor = PPOActor(64).to(device)
            if path_to_actor_bot is not None:
                actor.load_state_dict(torch.load(path_to_actor_bot))
            self.actors.append(actor)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def learn(self, max_time_steps):
        t_so_far = 0
        while t_so_far < max_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            self.init_hidden(len(batch_obs))
            
            t_so_far += np.sum(batch_lens)
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                self.init_hidden(len(batch_obs))
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            out = self.play_val_games()
            print(out)


    def get_action(self, obs):
        actions = []
        log_probs = []
        for i in range(self.num_agents):
            inp = torch.unsqueeze(torch.tensor(obs[i], dtype=torch.float), 0).to(device)

            act_bot = self.actors[i](inp)

            act_bot_probs = torch.nn.Softmax(1)(act_bot)
            action_bot = torch.multinomial(act_bot_probs, 1).item()
            log_prob = torch.log(act_bot_probs[0, action_bot])

            actions.append(action_bot)
            log_probs.append(log_prob.item())
        return actions, log_probs

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []


        t = 0
        while t < self.timesteps_per_batch:
            obs = self.env.reset()
            self.init_hidden(1)

            # Rewards this episode
            ep_rews = []
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs[0].astype(np.uint8))
                with torch.no_grad():
                    action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                ep_rews.append(int(rew[0]))
                batch_acts.append(action[0])
                batch_log_probs.append(log_prob[0])
                if done[0]:
                    break
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).to(device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(device)
        batch_rtgs = self.compute_rtgs(batch_rews).to(device)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        act_bot = self.actor(batch_obs)
        act_bot_probs = torch.nn.Softmax(1)(act_bot)
        log_prob = torch.log(act_bot_probs.gather(1, batch_acts.to(torch.long).unsqueeze(1)))

        return V, log_prob.squeeze(1)

    def init_hidden(self, episode_num):
        for agent in self.actors:
            agent.hidden_state = torch.zeros((episode_num, 64)).to(device)
        self.critic.hidden_state = torch.zeros((episode_num, 64)).to(device)


    def play_val_games(self):
        with torch.no_grad():
            out = []
            for i in range(10):
                steps = 0
                win = 0
                obs = self.env.reset()
                self.init_hidden(1)

                done = [False for _ in range(len(obs))]

                while not all(done):
                    all_action, _ = self.get_action(obs)

                    obs, reward, done, info = env.step(all_action)
                    if done[0] and steps <= 254:
                        win += 1
                        break
                    steps += 1

                out.append([win, steps])

        return out


if __name__ == '__main__':
    import gym
    from pogema import GridConfig

    n_agents = 1
    grid_config = GridConfig(num_agents=n_agents,
                             size=10,
                             density=0.3,
                             seed=None,
                             max_episode_steps=256,
                             obs_radius=5,
                             )
    env = gym.make("Pogema-v0", grid_config=grid_config)

    model = PPO(env, n_agents)
    model.learn(10_000_000)

    torch.save(model.actor.state_dict(), 'ppo_actor.pth')



