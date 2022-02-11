from random import random

import numpy as np

from agents.agent import Agent
import gym
from utils import is_space_finite, get_action_list
from abc import abstractmethod
from agents.QsaLearners.replay_buffer import ReplayBuffer


class QsaLearner(Agent):
    REQUIRES_TRAINING = True
    REQUIRES_FINITE_ACTION_SPACE = True

    # A generalisation of Q-learners (which explicitly learn off-policy)
    def __init__(self,
                 action_space: gym.Space,
                 state_space: gym.Space,
                 epsilon: float = 0.05,
                 buffer_size: int = 100,
                 batch_size: int = 100,
                 update_freq: int = 100,
                 gamma: float = 0.99):
        super().__init__(action_space, state_space)
        assert (is_space_finite(action_space)), action_space
        assert (is_space_finite(state_space)), state_space
        self._epsilon = epsilon
        self._update_freq = update_freq
        self._gamma = gamma
        self._allowed_actions = get_action_list(self._action_space)
        self._memory = ReplayBuffer(buffer_size=buffer_size,
                                    batch_size=batch_size)

    def act(self, state):
        # is_epsilon_greedy: bool = True, # Want to keep act method generic
        # is_greedy: bool = False):
        assert state in self._state_space, (state, self._state_space)
        # assert not is_greedy and is_epsilon_greedy, "Can't be both greedy and epsilon greedy!"
        # if is_epsilon_greedy:
        #     return self._epsilon_greedy_act(state)
        self._init_Q_s(state)
        if 1 == np.random.binomial(1, self._epsilon):
            return self._action_space.sample()
        else:
            return self.get_greedy_action(state)

    @abstractmethod
    def get_greedy_action(self, state):
        pass

    @abstractmethod
    def _init_Q_s(self, state):
        pass
    # def step(self, state, action, reward: float, next_state, done: bool):
    #     self._memory.add(state, action, reward, next_state, done)
    #     if self.t % self._update_freq == 0 and len(self._memory) > self.batch_size:
    #         self.update()

    @abstractmethod
    def update(self):
        pass

    def render(self):
        str_out = "\n------------------------ \n"
        str_out += f"{self.__class__.__name__} - Q(s,a)=... \n"
        Q_string = self._Q_to_string()
        if len(Q_string.split("\n")) >= 10:
            Q_string = "\n".join(Q_string.split("\n")[:5])+"\n ... \n"
        str_out += Q_string
        str_out += "------------------------\n"
        print(str_out)

    def _Q_to_string(self):
        str_out = ""
        for state in self._Q.keys():
            str_out += f" | state = {state}\n"
            for action in self._allowed_actions:
                str_out += f" |    | {action} : {self._Q[state][action]}\n"
        return str_out
