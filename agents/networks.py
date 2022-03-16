import torch
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import gym
from utils import vectorise_state, get_action_list, imageify_state


class BDNN(torch.nn.Module):
    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        super().__init__()
        # Draw a sample from state space, turn it into a vector
        # its dimension will be the shape of the state space
        self._m = DNN(state_space=state_space, action_space=action_space)

        # Taken from github
        const_bnn_prior_parameters = {
                "prior_mu": 0,
                "prior_sigma": 2.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": 1.0,
                "type": "Reparameterization",  # Flipout or Reparameterization
                "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
                "moped_delta": 0.5,
        }

        dnn_to_bnn(self._m, const_bnn_prior_parameters)

        # Redefine via DNN
        self.in_size = len(vectorise_state(state_space.sample()))
        self.action_list = get_action_list(action_space)
        self.out_size = len(self.action_list)

    def forward(self, state_vec: torch.Tensor):
        return self._m(state_vec)


class DNN(torch.nn.Module):
    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        super().__init__()
        # Draw a sample from state space, turn it into a vector
        # its dimension will be the shape of the state space
        self.in_size = len(vectorise_state(state_space.sample()))
        self.action_list = get_action_list(action_space)
        self.out_size = len(self.action_list)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, self.out_size),
            torch.nn.ReLU()
        )

    def forward(self, state_vec: torch.Tensor):
        return self.dense(state_vec)


class CNN(torch.nn.Module):
    def  __init__(self, state_space: gym.Space, action_space: gym.Space):
        super().__init__()
        # Draw a sample from state space, turn it into a vector
        # its dimension will be the shape of the state space
        self.in_size = imageify_state(state_space.sample())
        self.action_list = get_action_list(action_space)
        self.out_size = len(self.action_list)
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(in_channels=4, out_channels=5, kernel_size=2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten()
        )
        self.dense = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, self.out_size),
            torch.nn.ReLU()
        )

    def forward(self, state_img: torch.Tensor):
        vec_rep = self.convs(state_img)
        out = self.dense(vec_rep)
        return out
