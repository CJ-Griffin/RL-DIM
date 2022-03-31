import numpy as np

import gym
from agents import Agent
from utils import get_action_list


class BayesianCoinAgent(Agent):
    REQUIRES_TRAINING = True
    REQUIRES_FINITE_STATE_SPACE = True
    REQUIRES_FINITE_ACTION_SPACE = True

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        super().__init__(action_space, state_space)
        assert (isinstance(state_space, gym.spaces.Discrete))
        self._state_list = list(range(state_space.n))
        self._action_list = get_action_list(action_space)
        self._data = {}
        self._epsilon = 0.9
        for state in self._state_list:
            self._data[state] = {}
            for action in self._action_list:
                self._data[state][action] = []

    def act(self, state):
        if 1 == np.random.binomial(1, self._epsilon):
            return self._action_space.sample()
        else:
            return self.get_greedy_action(state)

    def get_greedy_action(self, state):
        ps = {}
        for action in self._action_list:
            hist = self._data[state][action]
            heads = np.sum(hist)
            tails = len(hist) - heads
            ps[action] = (heads + 1) / (heads + tails + 2)
        print(ps)
        chosen_action = max(ps, key=ps.get)
        print(chosen_action)
        return chosen_action

    def step(self, state: gym.core.ObsType, action: gym.core.ActType, reward: float, next_state: gym.core.ObsType,
             done: bool):
        self._data[state][action].append(reward)
        pass

    def update(self):
        pass
