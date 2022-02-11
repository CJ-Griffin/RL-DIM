from abc import ABC, abstractmethod, abstractproperty
import gym


class Agent(ABC):

    NEEDS_TRAINING = False

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        self._action_space = action_space
        self._state_space = state_space

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

    def render(self):
        print(str(self))
