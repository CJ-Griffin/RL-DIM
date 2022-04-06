import numpy as np
from src.agents.agent import Agent
import gym
from src.run_parameters import TrainParams
from src.generic_utils import get_action_list
from abc import abstractmethod
from src.agents.QsaLearners.replay_buffer import LinearMemory


class QsaLearner(Agent):
    REQUIRES_TRAINING = True
    REQUIRES_FINITE_ACTION_SPACE = True

    # A generalisation of Q-learners (which explicitly learn off-policy)

    def __init__(self, action_space: gym.Space, state_space: gym.Space, params: TrainParams):
        super().__init__(action_space, state_space, params)
        self._epsilon: float = params.epsilon
        self._update_freq: int = params.update_freq
        self._gamma: float = params.gamma
        self._buffer_size: int = params.buffer_size
        self._batch_size: int = params.batch_size

        self._allowed_actions = get_action_list(self._action_space)
        self._memory = LinearMemory(buffer_size=self._buffer_size,
                                    batch_size=self._batch_size)

    def act(self, state):
        assert state in self._state_space, (state, self._state_space)

        self._init_Q_s(state)
        if not self.is_eval_mode and 1 == np.random.binomial(1, self._epsilon):
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
        # print(self._unique_ID)
        # print(self._memory.memory)
        str_out = "\n------------------------ \n"
        str_out += f"{self.__class__.__name__} - Q(s,a)=... \n"
        Q_string = self._Q_to_string()
        if len(Q_string.split("\n")) >= 10:
            Q_string = "\n".join(Q_string.split("\n")[:5]) + "\n ... \n"
        str_out += Q_string
        str_out += f"\nID = {self._unique_ID}\n"
        str_out += "------------------------\n"
        print(str_out)

    def _Q_to_string(self):
        str_out = ""
        for state in self._Q.keys():
            str_out += f" | state = {state}\n"
            for action in self._allowed_actions:
                str_out += f" |    | {action} : {self._Q[state][action]}\n"
        return str_out
