import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cpu')

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


class Agent:
    def __init__(self):
        self.actor_rnn_hidden_dim = torch.zeros((1, 64), dtype=torch.float).to(device)

class PPO:
    def __init__(self,  path_to_actor=None):

        self.num_max_agents = 200

        self.actor = PPOActor(3, 5, 64, 5).to(device)
        if path_to_actor is not None:
            self.actor.load_state_dict(torch.load(path_to_actor, map_location=torch.device('cpu')))

        self.agents = []
        for i in range(self.num_max_agents):
            self.agents.append(Agent())


    def get_action(self, obs, a_old, n_agent=1):
        actions = []
        log_probs = []
        for i, agent in enumerate(self.agents):
            inp = torch.unsqueeze(torch.tensor(obs[i], dtype=torch.float), 0).to(device)
            aold_tensor = torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(a_old[i]), 5), 0).to(device)

            act_bot, agent.actor_rnn_hidden_dim = self.actor(inp, aold_tensor, agent.actor_rnn_hidden_dim)

            act_bot_probs = torch.nn.Softmax(1)(act_bot)
            action_bot = torch.multinomial(act_bot_probs, 1).item()
            log_prob = torch.log(act_bot_probs[0, action_bot]).item()

            actions.append(action_bot)
            log_probs.append(log_prob)
            if i + 1 == n_agent:
                break
        return np.array(actions), np.array(log_probs)



    def init_hidden(self, episode_num):
        for i in range(self.num_max_agents):
            self.agents[i].actor_rnn_hidden_dim = torch.zeros((episode_num, 64)).to(device)


