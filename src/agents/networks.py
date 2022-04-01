import torch

import gym
from src.generic_utils import vectorise_state, get_action_list, imageify_state


class DNN(torch.nn.Module):
    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        super().__init__()
        # Draw a sample from state space, turn it into a vector
        # its dimension will be the shape of the state space
        self.in_size = len(vectorise_state(state_space.sample()))
        self.action_list = get_action_list(action_space)
        self.out_size = len(self.action_list)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, self.out_size)
        )

    def forward(self, state_vec: torch.Tensor):
        return self.dense(state_vec)


class CNN(torch.nn.Module):
    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        super().__init__()
        # Draw a sample from state space, turn it into a vector
        # its dimension will be the shape of the state space
        self.in_size = imageify_state(state_space.sample()).shape[1:]
        print(self.in_size, "="*100)

        self.action_list = get_action_list(action_space)
        self.out_size = len(self.action_list)

        if self.in_size[0] <= 6 or self.in_size[0] <= 6:
            self.convs = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Flatten()
            )
        else:
            self.convs = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(in_channels=4, out_channels=5, kernel_size=2),
                # torch.nn.MaxPool2d(2, 2),
                torch.nn.Flatten()
            )

        self.dense = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, self.out_size),
        )

    def forward(self, state_img: torch.Tensor):
        vec_rep = self.convs(state_img)
        out = self.dense(vec_rep)
        return out
