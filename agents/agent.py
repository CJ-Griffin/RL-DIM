from abc import ABC, abstractmethod, abstractproperty

import numpy as np

import gym
from running.run_parameters import TrainParams
from utils import is_space_finite

CHARS = list("QWERTYUIOPASDFGHJKLZXCVBNM1234567890")


class Agent(ABC):

    def __init__(self,
                 action_space: gym.Space,
                 state_space: gym.Space,
                 params: TrainParams):
        self.check_compatibility(action_space=action_space, state_space=state_space)
        self._action_space = action_space
        self._state_space = state_space
        self._should_debug = params.should_debug
        self._unique_ID = "".join(list(np.random.choice(CHARS, size=20)))

    @abstractmethod
    def act(self, state):
        raise NotImplementedError

    @abstractmethod
    def step(self, state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state: gym.core.ObsType,
             done: bool):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def REQUIRES_FINITE_STATE_SPACE(self) -> bool:
        pass

    @property
    @abstractmethod
    def REQUIRES_FINITE_ACTION_SPACE(self) -> bool:
        pass

    @property
    @abstractmethod
    def REQUIRES_TRAINING(self) -> bool:
        pass

    def render(self):
        print(str(self))

    def check_compatibility(self, action_space: gym.Space, state_space: gym.Space):
        if self.REQUIRES_FINITE_ACTION_SPACE and not is_space_finite(action_space):
            raise Exception(f"Model ({self}) is not compatible with infinite action space ({action_space})" )
        elif self.REQUIRES_FINITE_STATE_SPACE and not is_space_finite(state_space):
            raise Exception(f"Model ({self}) is not compatible with infinite state space ({state_space})" )
