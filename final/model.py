import random
from model_astar import Model as astarmodel
from pogema.animation import AnimationMonitor

from pogema import GridConfig
from ppo import PPO
import gym

class Model:

    def __init__(self):
        self.our_agent = PPO(path_to_actor='ppo_actor_IV.pth')
        self.our_agent.actor.eval()
        self.our_agent.init_hidden(1)
        self.batch_acts_old = []

        self.solver = astarmodel()
        self.ep_t = 0

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.ep_t == 0:
            self.batch_acts_old.append([0] * len(obs))
        else:
            self.batch_acts_old.append(self.action)

        action_deep, _ = self.our_agent.get_action(obs, self.batch_acts_old[-1], len(obs))
        action_class = self.solver.act(obs, dones, positions_xy, targets_xy)

        actions = [x if random.random() > 0.5 else y for x, y in zip(action_deep, action_class)]
        self.action = actions
        self.ep_t += 1

        for agent_id in range(200,  len(obs)):
            actions.append(0)


        return actions

if __name__ == '__main__':
    n_games = 10
    for episod in range(n_games):
        classs = Model()
        isr_do = []
        csr_do = []
        grid_config = GridConfig(num_agents=60,  # количество агентов на карте
                                     size=60,  # размеры карты
                                     density=0.3,  # плотность препятствий
                                     seed=None,  # сид генерации задания
                                     max_episode_steps=256,  # максимальная длина эпизода
                                     obs_radius=5,  # радиус обзора
                                     )

        env = gym.make("Pogema-v0", grid_config=grid_config)
        env = AnimationMonitor(env)
        obs = env.reset()
        done = [False for k in range(len(obs))]
        rewards_game = [[] for _ in range(len(obs))]


        while not all(done):
            act = classs.act(obs, done, env.get_agents_xy_relative(), env.get_targets_xy_relative())
            obs, rew, done, _ = env.step(act)
            for robot in range(len(obs)):
                rewards_game[robot].append(rew[robot])


        target = [sum(x) for x in rewards_game]
        win = sum(target)
        csr = 1 if win == len(obs) else 0
        print('Игра {}. Результат: csr {}, isr {}'.format(episod+1, csr, win))
        env.save_animation('render/game{}.svg'.format(episod+1))

