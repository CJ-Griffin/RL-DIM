import collections

import gym
from src.agents.action_value_learners.tabular_AV_learners.tabular_AV import TabularAV
from src.utils.generic_utils import imshow_grid
from src.run_parameters import TrainParams


class QLearner(TabularAV):

    def __init__(self, action_space: gym.Space, state_space: gym.Space, params: TrainParams):
        super().__init__(action_space, state_space, params)
        self.td_error_log = collections.deque(maxlen=1000)

    def should_update(self, done: bool) -> bool:
        return done

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True,
                                                                           as_torch=False)

        for i, (s, a, r, sn, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
            if not d: # and i != self._batch_size - 1:
                if self._should_debug:
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
            # if r==0.4:
            #     print(f"{r} + ({self._gamma} * {next_q}) - {self._Q[s][a]}")
            #     print("r + (self._gamma * next_q) - self._Q[s][a]")
            #     print(self._batch_size, i)
            #     print(s, sn)
            #     print(td_error)
            self.td_error_log.append(td_error)
            self._Q[s][a] += self._alpha * td_error
