import torch.nn as nn
import torch
import torch.nn.functional as F


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

if __name__ == '__main__':
    input = torch.randn(32, 17, 31, 31)
    input_dop = torch.randn(32, 5)
    policy_hidden = torch.zeros((32, 1, 64))
    model = PPOActor(64, 5)
    output = model(input, input_dop, policy_hidden)
    print(output[0].size())