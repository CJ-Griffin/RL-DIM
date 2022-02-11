from random import random

import numpy as np

from agents.agent import Agent
import gym
from utils import is_space_finite, get_action_list
from abc import abstractmethod
from agents.QsaLearners.replay_buffer import ReplayBuffer


class BlackjackAgent(Agent):
    REQUIRES_TRAINING = False
    REQUIRES_FINITE_STATE_SPACE = True
    REQUIRES_FINITE_ACTION_SPACE = True

    def act(self, state):
        if state[0] < 16:
            return 1
        else:
            return 0

    def step(self, state: gym.core.ObsType, action: gym.core.ActType, reward: float, next_state, done: bool):
        pass

    def update(self):
        pass
