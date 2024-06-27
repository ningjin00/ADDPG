import numpy as np
import gym
from gym import spaces
import math


class DroneFlightEnv(gym.Env):
    """A simple drone flight environment for reinforcement learning."""

    def __init__(self):
        super(DroneFlightEnv, self).__init__()

        # 定义空间大小，例如10x10的格子
        self.grid_size = 1000
        # 定义起点和终点
        self.start_point = (0, 0)
        # 定义连续动作空间：x和y方向上的位移，每个方向的位移范围可以根据需要设定，这里假设为-1到1
        self.action_space = spaces.Box(low=np.array([-50, -50]), high=np.array([50, 50]), dtype=np.float32)
        # 定义状态空间，包括无人机在格子中的x和y坐标，即x和y的取值范围
        self.observation_space = spaces.Box(low=-self.grid_size, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        # 初始状态
        self.state = None

    def reset(self):
        # 在每个episode开始时重置状态
        self.state = np.array(self.start_point, dtype=np.int32)
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = action[0][0], action[0][1]

        # 确保位移量dx和dy在合理范围内，这里将它们限制在-1到1之间
        dx = np.clip(dx, -50, 50)
        dy = np.clip(dy, -50, 50)

        # 更新状态，考虑到边界条件
        x = np.clip(x + dx, -self.grid_size, self.grid_size - 1)
        y = np.clip(y + dy, -self.grid_size, self.grid_size - 1)
        self.state = np.array([x, y])

        # 计算到当前状态的奖励，你来补充
        # reward = log(x**2+y**2)
        reward = (math.log(1+(x-200) ** 2 + (y-50) ** 2)+math.log(1+(x+200) ** 2 + (y-40) ** 2)+math.log(1+(x-800) ** 2 +(y-100) ** 2)+math.log(1+(x+100)**2+(y-100)**2))

        return self.state, reward

    def close(self):
        pass

    def seed(self, seed=None):
        # 设置随机种子
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
