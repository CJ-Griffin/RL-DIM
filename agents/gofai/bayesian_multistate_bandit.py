import gym
import utils
from agents.QsaLearners import QsaLearner, QLearner
import pyro
import torch
import numpy as np
import pyro.distributions as dist


class BayesianMultistateBandit(QsaLearner):
    REQUIRES_FINITE_STATE_SPACE = True

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        super().__init__(action_space, state_space)
        self._unique_id = utils.generate_random_string(10)
        self._means_mean_init = 0.0
        self._means_var_init = 3.0
        self._data = {}

    def get_greedy_action(self, state):
        a_vals = [self.get_reward_dist(state, action) for action in self._allowed_actions]
        print(a_vals)
        return 0

    def _init_Q_s(self, state):
        if state not in self._data:
            self._data[state] = {}
            for action in self._allowed_actions:
                self._data[state][action] = []

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True)
        assert int(dones[-1]) == 1, dones
        assert not np.any(dones[:-2]), dones
        T = len(rewards)

        # This agent only works on multistate-bandit environments

        for s, a, r in zip(states, actions, rewards):
            s = self.get_hashable_state(s)
            self._data[s][a].append(r)

    def get_reward_dist(self,
             s: gym.core.ObsType,
             a: gym.core.ActType):
        r_s_a_mean = pyro.sample(f"r_{s}_{a}_mean", dist.Normal(self._means_mean_init, self._means_var_init))
        r_s_a = pyro.sample(f"r_{s}_{a}", dist.Normal(r_s_a_mean, 1.0), obs=self._data[s][a])
        print(r_s_a)
        return r_s_a

    def step(self,
             state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state: gym.core.ObsType,
             done: bool):
        self._memory.add(state, action, reward, next_state, done)
        if done:
            self.update()
