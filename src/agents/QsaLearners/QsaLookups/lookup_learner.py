import gym
from src.agents.QsaLearners.qsalearner import QsaLearner, LinearMemory
from src.run_parameters import TrainParams
import xxhash
import collections
import numpy as np


class LookupLearner(QsaLearner):
    REQUIRES_FINITE_STATE_SPACE = True

    def __init__(self, action_space: gym.Space, state_space: gym.Space, params: TrainParams):
        super().__init__(action_space, state_space, params)
        self._alpha: float = params.alpha
        self._q_init: float = params.q_init
        self._Q = {}
        self._Q_count = {}
        self._Q_hash_dict = {}
        self._state_hasher = xxhash.xxh64()

    def _init_Q_s(self, state):
        new_state = self.get_hashable_state(state)
        if self._should_debug:
            self._Q_hash_dict[new_state] = state
        state = new_state
        if state not in self._Q:
            self._Q[state] = {action: self._q_init for action in self._allowed_actions}
            self._Q_count[state] = {action: 0 for action in self._allowed_actions}

    def restore_after_load(self):
        import xxhash
        self._state_hasher = xxhash.xxh64()
        self._memory = LinearMemory(buffer_size=self._buffer_size,
                                    batch_size=self._batch_size)

    def get_greedy_action(self, state):
        init_state = state
        state = self.get_hashable_state(state)
        # state = torch.tensor(state)
        try:
            action_dict = self._Q[state]
        except:
            raise Exception(init_state, state)
        action = max(action_dict, key=action_dict.get)
        return action

    def step(self, state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state,
             done: bool):
        # print(self._Q, self._unique_ID)
        state = self.get_hashable_state(state)
        next_state = self.get_hashable_state(next_state)
        self._memory.add(state, action, reward, next_state, done)
        if self.should_update(done):
            self.update()

    def update(self):
        raise NotImplementedError()

    def should_update(self, done: bool) -> bool:
        raise NotImplementedError

    def get_hashable_state(self, state):
        if not isinstance(state, collections.Hashable):
            # If the state is mutable, hash it instead
            if isinstance(state, np.ndarray):
                # check whether it is a 1D, small shape
                # if state.size in list(range(10)):
                #     print(state)
                #     print(state.shape)
                #     state = tuple(list(state))
                #     print(state)
                # else:
                self._state_hasher.reset()
                self._state_hasher.update(state)
                state = self._state_hasher.digest()
                self._state_hasher.reset()
            else:
                raise NotImplementedError(state)
        return state
