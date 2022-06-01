import torch.nn as nn
import torch
import torch.nn.functional as F

class PPOCritic(nn.Module):
    def __init__(self, rnn_hidden_dim, layer_norm=True):
        super(PPOCritic, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.feachextractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
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

        self.hidden_state = None


        if layer_norm:
            self.layer_norm(self.fc1, std=1.0)
            self.layer_norm(self.fc2, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs):
        aa_flat = self.feachextractor(inputs)
        x = F.relu(self.fc1(aa_flat))
        # if self.hidden_state is None:
        #     self.hidden_state = torch.zeros((len(inputs), self.rnn_hidden_dim))

        self.hidden_state = self.rnn(x, self.hidden_state)
        v = self.fc2(self.hidden_state)
        return v

class PPOActor(nn.Module):
    def __init__(self, rnn_hidden_dim):
        super(PPOActor, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.feachextractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
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
        self.fc2 = nn.Linear(self.rnn_hidden_dim, 5)

        self.hidden_state = None


    def forward(self, s):
        aa_flat = self.feachextractor(s)
        x = F.relu(self.fc1(aa_flat))
        # if self.hidden_state is None:
        #     self.hidden_state = torch.zeros((len(s), self.rnn_hidden_dim))

        self.hidden_state = self.rnn(x, self.hidden_state)
        q = self.fc2(self.hidden_state)
        return q



if __name__ == '__main__':
    input = torch.randn(32, 3, 11, 11)
    input_dop = torch.randn(32, 5)
    policy_hidden = torch.zeros((32, 1, 64))
    model = PPOActor(3, 5, 64, 5)
    output = model(input, input_dop, policy_hidden)
    print(output[0].size())