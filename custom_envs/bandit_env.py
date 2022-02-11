from typing import Tuple, Optional

import gym
from gym.core import ActType, ObsType
import numpy as np

class BanditEnv(gym.Env):

    def __init__(self,
                 means: list = [-1, 0, 1],
                 variance: float = 0.5,
                 maxtime: int = 100):

        self._means = means
        self._variance = variance

        self._timestep = 0
        self._maxtime = maxtime

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(1)

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
        return 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self._timestep += 1
        done = self._timestep > self._maxtime
        if done:
            self._timestep = 0

        reward = np.random.normal(self._means[action], np.sqrt(self._variance))
        return 0, reward, done, {}
