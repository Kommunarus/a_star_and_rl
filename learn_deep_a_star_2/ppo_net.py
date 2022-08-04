import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import random



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


class PPOActor_macro(nn.Module):
    def __init__(self):
        super(PPOActor_macro, self).__init__()
        self.maxagent = 128
        self.fdim = 5
        flayer = [nn.Conv2d(17, 128, 3, 1, 1), nn.ReLU(), ]
        slayer = []
        for i in range(6):
            slayer = slayer + [nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(), ]

        ml = flayer + slayer + [nn.Conv2d(128, 1, 1, 1, 0), nn.ReLU(), nn.Flatten(1), ]

        fa_actor = nn.ModuleList(ml)

        self.feachextractor = fa_actor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

        self.fc1 = nn.Linear(31 * 31 * 1, self.fdim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)

        hdim = self.fdim * self.maxagent + self.maxagent
        self.fc3 = nn.Linear(hdim, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 5)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)


    def forward(self, s, n):
        batch = s.shape[0]
        # n_agent = s.shape[1]
        x = self.feachextractor[0](s)
        for l in self.feachextractor[1:]:
            x = l(x)
        f_a = F.relu(self.fc1(x))
        num_seat = random.choices(list(range(self.maxagent)), k=batch)
        x_1 = []
        k = 0
        for b in range(batch):
            x_2 = []
            for n_a in range(self.maxagent):
                if n_a == num_seat[b]:
                    x_2.append(f_a[k])
                    k += 1
                else:
                    x_2.append(torch.zeros((self.fdim, ), dtype=torch.float32).to(device))
            x_2.append(F.one_hot(torch.tensor(num_seat[b]), self.maxagent).to(device))
            x_1.append(torch.concat(x_2))

        x = torch.stack(x_1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        prob = self.fc5(x)
        prob = nn.Softmax(1)(prob)

        return prob, num_seat


if __name__ == '__main__':
    input = torch.randn(32, 1, 17, 31, 31)
    model = PPOActor_macro()
    output = model(input)
    print(output[1])