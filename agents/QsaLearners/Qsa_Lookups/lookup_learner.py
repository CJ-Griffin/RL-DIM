import gym
from agents.agent import Agent
from agents.QsaLearners.qsalearner import QsaLearner


class LookupLearner(QsaLearner):
    REQUIRES_FINITE_STATE_SPACE = True
    _Q = {}
    _Q_count = {}
    _Q_hash_dict = {}

    def __init__(self,
                 action_space: gym.Space,
                 state_space: gym.Space,
                 epsilon: float = 0.05,
                 buffer_size: int = 100,
                 batch_size: int = 100,
                 update_freq: int = 100,
                 gamma: float = 0.99,
                 q_init: float = 2.0,
                 alpha: float = 0.1,
                 debug_mode: bool = False):

        super().__init__(action_space,
                         state_space,
                         epsilon,
                         buffer_size,
                         batch_size,
                         update_freq,
                         gamma,
                         debug_mode=debug_mode)
        self._alpha: float = alpha
        self._q_init: float = q_init

    def _init_Q_s(self, state):
        if self._debug_mode:
            new_state = self.get_hashable_state(state)
            self._Q_hash_dict[new_state] = state
            state = new_state
        else:
            state = self.get_hashable_state(state)
        if state not in self._Q:
            self._Q[state] = {action: self._q_init for action in self._allowed_actions}
            self._Q_count[state] = {action: 0 for action in self._allowed_actions}

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
        state = self.get_hashable_state(state)
        next_state = self.get_hashable_state(next_state)
        self._memory.add(state, action, reward, next_state, done)
        if self.should_update(done):
            self.update()

    def update(self):
        raise NotImplementedError()

    def should_update(self, done: bool) -> bool:
        raise NotImplementedError

