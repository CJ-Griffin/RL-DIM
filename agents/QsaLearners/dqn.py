import torch

import gym
from agents.agent import Agent
from agents.QsaLearners.qsalearner import QsaLearner
from utils import *


class DNN(torch.nn.Module):
    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        super().__init__()
        # Draw a sample from state space, turn it into a vector
        # its dimension will be the shape of the state space
        self.in_size = len(vectorise_state(state_space.sample()))
        self.action_list = get_action_list(action_space)
        self.out_size = len(self.action_list)
        print(self.in_size, self.out_size)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, self.out_size),
            torch.nn.ReLU()
        )

    def forward(self, state: gym.Space):
        return self.forward(torch.tensor(state))

    def forward(self, x: torch.Tensor):
        return self.dense(x)


class DQN(QsaLearner):
    # TODO - address categorical spaces and one-hot encodings
    def __init__(self,
                 action_space: gym.Space,
                 state_space: gym.Space,
                 alpha: float = 0.1,
                 epsilon: float = 0.05,
                 buffer_size: int = 1000,
                 batch_size: int = 100,
                 update_freq: int = 100,
                 gamma: float = 0.99,
                 # is_state_one_hot: bool = False,
                 # is_action_one_hot: bool = False,
                 learning_rate: float = 1e-4
                 ):
        super().__init__(action_space, state_space, epsilon, buffer_size, batch_size, update_freq, gamma)
        self._alpha = alpha
        self._batch_size = batch_size
        self._buffer_size = buffer_size

        self._Q_net = DNN(state_space=state_space, action_space=action_space)
        self.optimiser = torch.optim.Adam(self._Q_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.MSELoss()

    def _init_Q_s(self, state):
        pass

    def get_greedy_action(self, state):
        x = (torch.tensor(state)).float()
        if x.shape == torch.Size([]):
            x = x.reshape(1)
        Qs = self._Q_net(x)
        a = Qs.argmax()
        a = int(a)
        return a

    def step(self, state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state,
             done: bool):
        self._memory.add(state, action, reward, next_state, done)
        if len(self._memory) >= self._batch_size:
            self.update()

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True, as_torch=True)
        Q_next_stars, _ = self._Q_net(next_states).max(dim=1)
        targets = (self._gamma * Q_next_stars) + rewards.flatten()
        prevs = self._Q_net(states)[torch.arange(len(states)), actions.flatten()]
        loss = self.loss(targets, prevs)
        loss.backward()
        self.optimiser.step()

    def _Q_to_string(self):
        return str(self._Q_net)
