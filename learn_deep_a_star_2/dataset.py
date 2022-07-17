import numpy as np
from heapq import heappop, heappush
import gym
from pogema import GridConfig
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import argparse

grid_config = GridConfig(num_agents=1,  # количество агентов на карте
                         size=64,  # размеры карты
                         density=0.3,  # плотность препятствий
                         seed=None,  # сид генерации задания
                         max_episode_steps=256,  # максимальная длина эпизода
                         obs_radius=5,  # радиус обзора
                         )

env = gym.make("Pogema-v0", grid_config=grid_config)
# name_ds = 'train_2'
size_look = 15


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
        self.f = []

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        self.f = []
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
                    self.f.append({'i': u.i, 'j': u.j, 'f': u.g + 1 + h})
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

    def act(self, obs, positions_xy, targets_xy) -> list:
        if self.agents is None:
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        dist_agents = []
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue
            self.agents[k].update_obstacles(obs[k][0], obs[k][1], (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.agents[k].get_next_node()

            actions.append(self.actions[(next_node[0] - positions_xy[k][0], next_node[1] - positions_xy[k][1])])

            dist_agents.append(dict_to_np(self.agents[k].f))
        return actions, dist_agents

def dict_to_np(dict_open_cell):
    size_look = 15
    centr = 64 + 2 * size_look
    map = np.zeros((2*centr+1, 2*centr + 1), dtype=np.int32)
    for unit in dict_open_cell:
        x1 = centr + unit['i']
        y1 = centr + unit['j']
        map[x1, y1] = unit['f']
    return map

def write_dataset(name_ds, N):
    # N = 500_000

    list_s = []
    list_a = []
    n_save = 0

    with h5py.File(f'./{name_ds}.hdf5', 'w') as f:

        for game in range(N):
            if game % 1000 == 0 and game > 0:
                print(game, len(list_s))
                f.create_dataset(f"s{n_save}", data=np.array(list_s, dtype=np.int8))
                f.create_dataset(f"a{n_save}", data=np.array(list_a, dtype=np.int8))

                list_s = []
                list_a = []
                n_save += 1

            obs = env.reset()
            solver = Model()

            zabor_map = [np.zeros((2 * (64 + 2*size_look) + 1, 2 * (64 + 2*size_look) + 1), dtype=np.uint8), ]
            history_dict = {}
            step = 0

            done = [False for k in range(len(obs))]
            while not all(done):
                current_xy = env.get_agents_xy_relative()
                targets_xy = env.get_targets_xy_relative()

                action, dist_agents = solver.act(obs,
                                      current_xy,
                                      targets_xy)
                obs_new, reward, done, info = env.step(action)

                newobs = update_obstacles(obs, current_xy, targets_xy, zabor_map, history_dict, step, dist_agents)
                list_s.append(newobs[0])
                list_a.append(action[0])

                obs = obs_new
                step += 1

        print(game, len(list_s))
        f.create_dataset(f"s{n_save}", data=np.array(list_s, dtype=np.uint8))
        f.create_dataset(f"a{n_save}", data=np.array(list_a, dtype=np.uint8))

def update_obstacles(obs, currentxy, targetsxy, zabormap, historydict, step, distagents):
    newobs = []
    for i, (o, current_xy, targets_xy, zabor_map, dist_agents) in \
                                            enumerate(zip(obs, currentxy, targetsxy, zabormap, distagents)):

        centr = 64 + 2*size_look
        d0 = centr + current_xy[0] - 5
        d1 = centr + current_xy[1] - 5
        zabor_map[d0: d0 + 11, d1: d1 + 11] = o[0]

        map_all_agents = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_all_agents[d0: d0 + 11, d1: d1 + 11] = o[1]
        map_all_agents[centr + current_xy[0],  centr + current_xy[1]] = 0

        svobodno_map = 1 - (zabor_map + map_all_agents)

        historydict[step] = (centr + current_xy[0], centr + current_xy[1])
        hist = []
        for j in range(8):
            hhh = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
            if step - j >= 0:
                coord_hist = historydict[step - j]
                hhh[coord_hist[0], coord_hist[1]] = 1
            hist.append(hhh)


        map_target = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_target[centr + targets_xy[0], centr + targets_xy[1]] = 1

        map_target_local = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_target_local[d0: d0 + 11, d1: d1 + 11] = o[2]

        map_start = np.zeros((2 * centr + 1, 2 * centr + 1), dtype=np.uint8)
        map_start[centr, centr] = 1

        x1 = centr + current_xy[0] - size_look
        x2 = centr + current_xy[0] + size_look + 1
        y1 = centr + current_xy[1] - size_look
        y2 = centr + current_xy[1] + size_look + 1

        map_1 = zabor_map[x1: x2, y1: y2]
        map_2 = map_all_agents[x1: x2, y1: y2]
        map_3 = svobodno_map[x1: x2, y1: y2]
        map_4 = np.zeros((size_look*2+1, size_look*2+1), dtype=np.uint8)
        map_5 = np.ones((size_look*2+1, size_look*2+1), dtype=np.uint8)
        map_6 = map_target[x1: x2, y1: y2]
        map_7 = map_target_local[x1: x2, y1: y2]
        map_8 = map_start[x1: x2, y1: y2]
        map_9 = dist_agents[x1: x2, y1: y2]

        hist_map = []
        for mmm in hist:
            hist_map.append(mmm[x1: x2, y1: y2])

        newobs.append(np.stack([map_1, map_2, map_3, map_4, map_5, map_6, map_7, map_8, map_9] + hist_map))

    return newobs



def read_dataset(name_ds):
    with h5py.File(f'./{name_ds}.hdf5', 'r') as f:
        data_s = f['s']
        data_a = f['a']

        print(data_s[5])
        print(data_a[5])


class H5Dataset(Dataset):
    def __init__(self, h5_path='', num_chunk=50, limit=-1):
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

        data_s = torch.from_numpy(state)

        return {"s": data_s, "a": a}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pups')
    parser.add_argument('-n', type=int, default=1000)

    arg = parser.parse_args()
    # pass
    write_dataset(arg.name, arg.n)
    # read_dataset()

    loader = torch.utils.data.DataLoader(H5Dataset(h5_path=f'./{arg.name}.hdf5', num_chunk=1), shuffle=True, batch_size=7)
    print(len(loader.dataset))
    batch = next(iter(loader))
    print(batch['s'].size())
    print(batch['a'])
