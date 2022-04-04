import neptune.new

import gym
import torch
import numpy as np

from src.agents.QsaLearners.qsalearner import QsaLearner
from src.agents.networks import DNN, CNN, CNN2, CNN3
from src.agents.QsaLearners.replay_buffer import ReplayBuffer
from src.run_parameters import TrainParams
from src.generic_utils import vectorise_state, imageify_state


class DQN(QsaLearner):
    REQUIRES_FINITE_STATE_SPACE = False
    NETWORK_CLASS = DNN

    @staticmethod
    def preprocess_state(state):
        return vectorise_state(state)

    def __init__(self, action_space: gym.Space, state_space: gym.Space, params: TrainParams, existing_model=None):
        super().__init__(action_space, state_space, params)
        self._params = params # Keep this around for saving the model
        self._update_freq = params.update_freq
        self._num_steps_since_update = 0
        self._memory = ReplayBuffer(self._buffer_size)

        self._Q_net = self.NETWORK_CLASS(state_space=self._state_space, action_space=self._action_space)
        self.optimiser = torch.optim.Adam(self._Q_net.parameters(), lr=params.learning_rate)
        self.loss = torch.nn.MSELoss()

    def _init_Q_s(self, state):
        pass

    def get_greedy_action(self, state):
        state = self.preprocess_state(state)
        x = state
        Qs = self._Q_net(x.unsqueeze(0))
        a = int(Qs.argmax())
        if self._should_debug:
            p = 1e-1
            if np.random.choice([True, False], p=[p, 1.0 - p]):
                if self.NETWORK_CLASS == CNN:
                    print(f"Q({state[0, :, :4, :4]}) = {Qs}")
                else:
                    print(f"Q({state[:4]}) = {Qs}")
                # print(self._Q_net)
                print(a)
        return a

    def step(self, state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state,
             done: bool):

        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        # print(state, next_state)
        self._memory.add(state, action, reward, next_state, done)
        self._num_steps_since_update += 1
        if self._num_steps_since_update % self._update_freq == 0:
            self.update()

    def update(self):
        self._num_steps_since_update = 0
        sample = self._memory.sample(self._batch_size)
        states, actions, rewards, next_states, dones = sample

        # I'm not detaching here - this could mess with backprop!
        with torch.no_grad():
            Q_vals = self._Q_net(next_states).detach()
            Q_next_states, _ = Q_vals.max(dim=1)

        Q_next_states[dones.flatten()] = 0.0
        targets = (self._gamma * Q_next_states) + rewards.flatten()
        prev_qs = self._Q_net(states)
        prevs = prev_qs[torch.arange(len(states)), actions.squeeze()]
        loss = self.loss(prevs, targets)
        p = 1e-1
        if self._should_debug and np.random.choice([True, False], p=[p, 1.0 - p]):
            self.debug_print(Q_next_states, actions, dones, prevs, rewards, states, targets, loss)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def debug_print(self, Q_next_states, actions, dones, prevs, rewards, states, targets, loss):
        for i in range(len(states)):
            if states[i].shape[0] >= 3:
                print(states[i].view(3, -1))
            else:
                print(states[i])
            print(f"Action {actions[i]} gave rewards {rewards[i]}")
            print(f"target = {targets[i]} = (gamma * {Q_next_states[i]}) + {rewards[i]}")
            print(f"Previous Q-value: {prevs[i]}")
            # print(Q_next_states[i])
            # print(targets[i])
            # print(prevs[i])
            # print(prevs_0[i])
            print()
            if dones[i]: print("DONE\n")
            print()
        # print(Qs)
        # print("State-action pairs from update")
        # print([(int(a), float(tar)) for (a, tar) in zip(list(actions), list(targets))])
        print("loss ", loss, "=" * 80)

    def _Q_to_string(self):
        return str(self._Q_net)


class DQN_CNN(DQN):
    NETWORK_CLASS = CNN

    @staticmethod
    def preprocess_state(state):
        return imageify_state(state)


class DQN_CNN2(DQN):
    NETWORK_CLASS = CNN2

    @staticmethod
    def preprocess_state(state):
        return imageify_state(state)


class DQN_CNN3(DQN):
    NETWORK_CLASS = CNN3

    @staticmethod
    def preprocess_state(state):
        return imageify_state(state)
