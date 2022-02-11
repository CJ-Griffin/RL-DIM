import gym
from agents.agent import Agent
from agents.QsaLearners.qsalearner import QsaLearner


class QLearner(QsaLearner):
    def __init__(self,
                 action_space: gym.Space,
                 state_space: gym.Space,
                 q_init: float = 0.0,
                 alpha: float = 0.1,
                 epsilon: float = 0.05,
                 buffer_size: int = 1000,
                 batch_size: int = 100,
                 update_freq: int = 100,
                 gamma: float = 0.99):
        super().__init__(action_space, state_space, epsilon, buffer_size, batch_size, update_freq, gamma)
        self._Q = {}
        self._alpha = alpha
        self._q_init = q_init
        self._batch_size = batch_size
        self._buffer_size = buffer_size

    def _init_Q_s(self, state):
        if state not in self._Q:
            self._Q[state] = {action: self._q_init for action in self._allowed_actions}

    def get_greedy_action(self, state):
        # state = torch.tensor(state)
        action_dict = self._Q[state]
        action = max(action_dict, key=action_dict.get)
        return action

    def step(self, state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state,
             done: bool):
        self._memory.add(state, action, reward, next_state, done)
        if len(self._memory) >= self._batch_size:
            self.update()

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True)

        for s, a, r, sn, d in zip(states, actions, rewards, next_states, dones):
            next_qs = self._Q[sn]
            an = max(next_qs, key=next_qs.get)
            next_q = next_qs[an]
            td_error = r + (self._gamma * next_q) - self._Q[s][a]
            self._Q[s][a] += self._alpha * td_error

    def render(self):
        str_out = "\n------------------------ \n"
        str_out += "QLearner - Q(s,a)=... \n"
        for state in self._Q.keys():
            str_out += f" | state = {state}\n"
            for action in self._allowed_actions:
                str_out += f" |    | {action} : {self._Q[state][action]}\n"
        str_out += "------------------------\n"
        print(str_out)
