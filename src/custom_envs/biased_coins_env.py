from typing import Tuple, Optional
import gym
from gym.core import ActType, ObsType
import numpy as np

from src.custom_envs.base_env import BaseEnv

class CoinsEnv(BaseEnv):

    def __init__(self,
                 ps: list = np.random.uniform(0, 1, size=(2, 3)),
                 maxtime: int = 10):
        self._ps = ps

        self._timestep = 0
        self._maxtime = maxtime

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(1)

        self._state = 0

    def render(self, mode="human"):
        print(f"""
        
--- BiasedCoins ---
ps:{self._ps}
---  ======== ---

""")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsType:
        return 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self._timestep += 1
        done = self._timestep > self._maxtime
        if done:
            self._timestep = 0
        p = self._ps[self._state][action]
        reward = np.random.choice([0, 1], p=[1 - p, p])
        return 0, reward, done, {}
