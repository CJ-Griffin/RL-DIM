import matplotlib.pyplot as plt
import numpy as np

import gym
from agents.QsaLearners.Qsa_Lookups.lookup_learner import LookupLearner
from utils import imshow_grid

class QLearner(LookupLearner):

    def should_update(self, done: bool) -> bool:
        return len(self._memory) >= self._batch_size

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True)

        for i, (s, a, r, sn, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
            if not d and i != self._batch_size -1:
                if self._is_debug_mode:
                    try:
                        next_qs = self._Q[sn]
                    except KeyError as e:
                        imshow_grid(self._Q_hash_dict[s])
                        raise KeyError(self._Q, sn, a, i, e)
                else:
                    next_qs = self._Q[sn]
                an = max(next_qs, key=next_qs.get)
                next_q = next_qs[an]
            else:
                next_q = 0.0

            td_error = r + (self._gamma * next_q) - self._Q[s][a]
            self._Q[s][a] += self._alpha * td_error
