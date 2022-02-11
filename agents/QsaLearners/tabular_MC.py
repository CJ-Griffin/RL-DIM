import numpy as np
import torch

from agents.agent import Agent
import gym
from utils import is_space_finite
from agents.QsaLearners.qsalearner import QsaLearner

# TODO Consider reworking s.t. there is a TabularQsa class
#       containing all learners that use dictionaries (_Q)


class TabularMC(QsaLearner):

    def __init__(self,
                 action_space: gym.Space,
                 state_space: gym.Space,
                 epsilon: float = 0.05,
                 gamma: float = 0.9):
        super().__init__(action_space=action_space,
                         state_space=state_space,
                         epsilon=epsilon,
                         gamma=gamma,
                         buffer_size=int(1e6))
        self._Q = {}
        self._Q_count = {}

    def _init_Q_s(self, state):
        # state = torch.tensor(state)
        if state not in self._Q:
            self._Q[state] = {action: 0.0 for action in self._allowed_actions}
            self._Q_count[state] = {action: 0 for action in self._allowed_actions}

    def get_greedy_action(self, state):
        # state = torch.tensor(state)
        action_dict = self._Q[state]
        action = max(action_dict, key=action_dict.get)
        return action

    def step(self, state, action, reward: float, next_state, done: bool):
        self._memory.add(state, action, reward, next_state, done)
        if done:
            self.update()
        # if self.t % self._update_freq == 0 and len(self._memory) > self.batch_size:
        #     self.update()

    # To be called only when self._memory has > self._batch_size items
    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True)
        assert int(dones[-1]) == 1, dones
        assert not np.any(dones[:-2]), dones
        T = len(rewards)
        returns = np.zeros(T)
        returns[-1] = rewards[-1]
        # sets rewards[-2], rewards[-3], ...., rewards[1], rewards[0]
        for t in range(T - 2, -1, -1):
            returns[t] = rewards[t] + (self._gamma * returns[t + 1])
        # for a,b,c in zip(rewards, returns,actions):
        #     print("|", a,b,c)
        assert len(returns) == len(rewards), (returns, rewards)
        # First visit
        seen = []

        for t in range(T):
            state = (states[t])
            action = (actions[t])
            if True:  # (state, action) not in seen
                seen.append((state, action))
                ret = (returns[t])
                # print(f"{state:1.0f}, {action:1.0f}, " + f"{ret:2.2f}".zfill(5))
                n = self._Q_count[state][action]
                self._Q_count[state][action] += 1
                prev = float(self._Q[state][action])
                self._Q[state][action] += (ret - prev) / (n + 1)
