from typing import Tuple, Optional
import gym
from gym.core import ActType, ObsType
import numpy as np

from src.custom_envs.base_env import BaseEnv

class BanditEnv(BaseEnv):

    def __init__(self,
                 means: list = [-1, 0, 1],
                 variance: float = 1,
                 maxtime: int = 1):
        self._means = means
        self._variance = variance

        self._timestep = 0
        self._maxtime = maxtime

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(1, start=1)

    def render(self, mode="human"):
        print(f"""
--- BanditEnv ---
means:{self._means}
variance:{self._variance}
---  ======== ---
""")
        print()
        print()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsType:
        return 1

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self._timestep += 1
        done = self._timestep >= self._maxtime
        if done:
            self._timestep = 0

        reward = np.random.normal(self._means[action], np.sqrt(self._variance))
        return 1, reward, done, {}


class NonStationaryBanditEnv(BanditEnv):

    def __init__(self,
                 means: list = [[-1, 0, 1], [2, 0, -2]],
                 variance: float = 1,
                 maxtime: int = 100):
        self._means = means
        self._variance = variance

        self._timestep = 0
        self._maxtime = maxtime
        self._state = 0

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(len(means))

    def render(self, mode="human"):
        print(f"""
--- BanditEnv ---
means:{self._means}
variance:{self._variance}
---  ======== ---
""")
        print()
        print()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsType:
        return 1

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self._timestep += 1
        done = self._timestep > self._maxtime
        if done:
            self._timestep = 0

        reward = np.random.normal(self._means[self._state][action], np.sqrt(self._variance))
        self._state = self.observation_space.sample()
        return 1, reward, done, {}
