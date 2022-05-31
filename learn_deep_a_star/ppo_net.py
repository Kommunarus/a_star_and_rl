import torch.nn as nn
import torch
import torch.nn.functional as F

class PPOCritic(nn.Module):
    def __init__(self, critic_input_shape, args, layer_norm=True):
        super(PPOCritic, self).__init__()
        self.args = args
        self.cv = nn.Conv2d(3, 8, 3, 2)
        self.cv2 = nn.Conv2d(8, 16, 3, 2)
        self.fc1 = nn.Linear(2*2*16 + critic_input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

        if layer_norm:
            self.layer_norm(self.fc1, std=1.0)
            self.layer_norm(self.fc2, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs, dop, hidden):

        aa = self.cv(inputs)
        aa = F.relu(aa)
        aa = self.cv2(aa)
        aa_flat = torch.flatten(aa, start_dim=1)

        input = torch.hstack([aa_flat, dop])

        x = F.relu(self.fc1(input))
        h_in = hidden.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        v = self.fc2(h)
        return v, h

class PPOActor(nn.Module):
    def __init__(self, input_shape, dop_input_shape, rnn_hidden_dim, n_actions):
        super(PPOActor, self).__init__()
        # self.args = args
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

if __name__ == '__main__':
    input = torch.randn(32, 3, 11, 11)
    input_dop = torch.randn(32, 5)
    policy_hidden = torch.zeros((32, 1, 64))
    model = PPOActor(3, 5, 64, 5)
    output = model(input, input_dop, policy_hidden)
    print(output[0].size())