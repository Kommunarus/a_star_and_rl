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


    def get_feach(self, s, num_seat, device):
        f1 = []
        for state in s:
            tensor_s = torch.unsqueeze(torch.from_numpy(state).to(torch.float32), 0).to(device)

            # n_agent = s.shape[1]
            x = self.feachextractor[0](tensor_s)
            for l in self.feachextractor[1:]:
                x = l(x)
            f_a = F.relu(self.fc1(x))
            f1.append(f_a.squeeze())

        k = 0
        x_2 = []
        for n_a in range(self.maxagent):
            if n_a <= num_seat[-1] and n_a == num_seat[k]:
                x_2.append(f1[k])
                k += 1
            else:
                x_2.append(torch.zeros((self.fdim, ), dtype=torch.float32).to(device))
        x = torch.concat(x_2)

        return x

    def learning(self, newobs, n, num_seat, device):
        s = self.get_feach(newobs, num_seat, device)
        return self.forward(s, n, device)


    def forward(self, s, n, device):

        code = F.one_hot(torch.tensor(n), self.maxagent).to(device)
        x = torch.hstack([s, code])

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        prob = self.fc5(x)
        # prob = nn.Softmax(1)(prob)

        return prob


class PPOActor2(nn.Module):
    def __init__(self):
        super(PPOActor2, self).__init__()
        self.layer1 = nn.Conv3d(in_channels=17, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv3d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv3d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.layer4 = nn.Conv3d(in_channels=96, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.layer5 = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.flat = nn.Flatten(2)
        self.fc1 = nn.Linear(31 * 31 * 1, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 5)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        # torch.nn.init.xavier_uniform_(self.layer5.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, s):
        #B*128*17*31*31
        input = torch.moveaxis(s, 1, 2)
        #B*17*128*31*31
        x = F.relu(self.layer1(input))
        #B*128*128*31*31
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        # x = F.relu(self.layer5(x))
        #B*1*128*31*31
        x = torch.moveaxis(x, 1, 2)
        #B*128*1*31*31
        x = self.flat(x)
        #B*128*961
        x = F.relu(self.fc1(x))
        #B*128*1024
        x = F.relu(self.fc2(x))
        #B*128*512
        x = self.fc3(x)
        #B*128*5
        return x

if __name__ == '__main__':
    input = torch.randn(32, 32, 17, 31, 31)
    model = PPOActor2()
    output = model(input)
    print(output.shape)