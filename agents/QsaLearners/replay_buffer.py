import collections

import numpy as np
import torch

device = torch.device("cpu")


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        field_names = ["state", "action", "reward", "next_state", "done"]
        self.ExperienceType = collections.namedtuple("Experience", field_names=field_names)
        self.memory = collections.deque(maxlen=self.buffer_size)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.ExperienceType(state, action, reward, next_state, done))

    def sample(self, sample_all: bool = False, as_torch=False):
        if sample_all:
            experiences = list(self.memory)
        else:
            experiences = list(np.random.sample(self.memory, k=self.batch_size))
        self.memory.clear()

        # TODO - if not broken - remove this
        # Clear empty experiences
        # experiences = [e for e in experiences if e is not None]
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_states = [e.next_state for e in experiences]
        dones = [e.done for e in experiences]
        if not as_torch:
            return states, actions, rewards, next_states, dones
        else:
            states = torch.from_numpy(np.vstack(states)).float().to(device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
            next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
            dones = torch.from_numpy(np.vstack(dones)).bool().to(device)
            return states, actions, rewards, next_states, dones
