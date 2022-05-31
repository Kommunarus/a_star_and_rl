import numpy as np
from heapq import heappop, heappush
import gym
from pogema import GridConfig
import h5py
from torch.utils.data import Dataset, DataLoader
import torch

grid_config = GridConfig(num_agents=1,  # количество агентов на карте
                         size=64,  # размеры карты
                         density=0.3,  # плотность препятствий
                         seed=None,  # сид генерации задания
                         max_episode_steps=256,  # максимальная длина эпизода
                         obs_radius=5,  # радиус обзора
                         )

env = gym.make("Pogema-v0", grid_config=grid_config)

class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


class AStar:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 10000  # due to the absence of information about the map size we need some other stop criterion
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n = (u.i+d[0], u.j + d[1])
                if n not in self.obstacles and n not in self.CLOSED and n not in self.other_agents:
                    h = abs(n[0] - self.goal[0]) + abs(n[1] - self.goal[1])  # Manhattan distance as a heuristic function
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)  # store information about the predecessor

    def get_next_node(self):
        next_node = self.start  # if path not found, current start position is returned
        if self.goal in self.CLOSED:  # if path found
            next_node = self.goal
            while self.CLOSED[next_node] != self.start:  # get node in the path with start node as a predecessor
                next_node = self.CLOSED[next_node]
        return next_node

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(np.nonzero(obs))  # get the coordinates of all obstacles in current observation
        for obstacle in obstacles:
            self.obstacles.add((n[0] + obstacle[0], n[1] + obstacle[1]))  # save them with correct coordinates
        self.other_agents.clear()  # forget previously seen agents as they move
        agents = np.transpose(np.nonzero(other_agents))  # get the coordinates of all agents that are seen
        for agent in agents:
            self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))  # save them with correct coordinates


class Model:
    def __init__(self):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.agents is None:
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue
            self.agents[k].update_obstacles(obs[k][0], obs[k][1], (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.agents[k].get_next_node()
            actions.append(self.actions[(next_node[0] - positions_xy[k][0], next_node[1] - positions_xy[k][1])])
        return actions


def write_dataset():
    N = 100_000

    list_s = []
    list_a = []
    list_a_old = []
    n_save = 0

    with h5py.File('./test.hdf5', 'w') as f:

        for game in range(N):
            if game % 1000 == 0 and game > 0:
                print(game, len(list_s))
                f.create_dataset(f"s{n_save}", data=np.array(list_s, dtype=np.uint8))
                f.create_dataset(f"a{n_save}", data=np.array(list_a, dtype=np.uint8))
                f.create_dataset(f"a_old{n_save}", data=np.array(list_a_old, dtype=np.uint8))

                list_s = []
                list_a = []
                list_a_old = []
                n_save += 1

            obs = env.reset()
            a_old = 0
            solver = Model()


            done = [False for k in range(len(obs))]
            while not all(done):
                action = solver.act(obs, done,
                                      env.get_agents_xy_relative(),
                                      env.get_targets_xy_relative())
                obs_new, reward, done, info = env.step(action)

                list_s.append(obs[0])
                list_a_old.append(a_old)
                list_a.append(action[0])

                obs = obs_new
                a_old = action[0]

        print(game, len(list_s))
        f.create_dataset(f"s{n_save}", data=np.array(list_s, dtype=np.uint8))
        f.create_dataset(f"a{n_save}", data=np.array(list_a, dtype=np.uint8))
        f.create_dataset(f"a_old{n_save}", data=np.array(list_a_old, dtype=np.uint8))


def read_dataset():
    with h5py.File('./test.hdf5', 'r') as f:
        data_s = f['s']
        data_a = f['a']

        print(data_s[5])
        print(data_a[5])


class H5Dataset(Dataset):
    def __init__(self, h5_path='./test.hdf5', num_chunk=50, limit=-1):
        self.limit = limit
        self.h5_path = h5_path
        self._archives = h5py.File(h5_path, "r")
        self.indices = {}
        idx = 0
        for c in range(num_chunk):
            for i in range(len(self._archives[f'a{c}'])):
                self.indices[idx] = (c, i)
                idx += 1

        self._archives = None

    @property
    def archive(self):
        if self._archives is None: # lazy loading here!
            self._archives = h5py.File(self.h5_path, "r")
        return self._archives

    def __getitem__(self, index):
        c, i = self.indices[index]
        state = self.archive[f"s{c}"][i]
        a = self.archive[f"a{c}"][i]
        a_old = self.archive[f"a_old{c}"][i]

        data_s = torch.from_numpy(state)
        data_a_old =  torch.zeros(5)
        data_a_old[a_old] = 1

        return {"s": data_s, "a": a, "a_old": data_a_old}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)


if __name__ == '__main__':
    # pass
    write_dataset()
    # read_dataset()

    loader = torch.utils.data.DataLoader(H5Dataset(num_chunk=9), shuffle=True, batch_size=16)
    print(len(loader.dataset))
    batch = next(iter(loader))
    print(batch['s'].size())
    print(batch['a'])
    print(batch['a_old'])