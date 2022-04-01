import os
import numpy as np
import gym
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

from src.custom_envs.base_env import BaseEnv


# Adapted from https://github.com/jbinas/gym-mnist/blob/master/mnist.py
class MnistEnv(BaseEnv):
    def __init__(self, use_training_set=False, seed=1337):
        self.shape = (3, 28, 28)
        self.seed(seed=seed)

        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
        transform = transforms.Normalize((0.5,), (1.0,))
        data = dset.MNIST(root=root, train=use_training_set, transform=transform, download=True)
        if use_training_set:
            x, y = data.train_data, data.train_labels
        else:
            x, y = data.test_data, data.test_labels
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = x.float() / 255
        self.data = [x[y == i] for i in range(10)]
        self.num_samples = [x_.shape[0] for x_ in self.data]

        # the first action is the null action
        self.action_space = gym.spaces.Discrete(10)
        lowest = np.zeros((3, 28, 28))
        highest = np.ones((3, 28, 28))
        self.observation_space = gym.spaces.Box(
            low=lowest,
            high=highest,
            shape=self.shape
        )
        self.reward_range = (-1, 1)
        self.end_of_episode_state = torch.zeros((3, 28, 28)).float()
        assert self.end_of_episode_state in self.observation_space
        self.state = None
        self.ans = None

    def step(self, action):
        if self.ans == action:
            reward = 1.0
        else:
            reward = 0
        obs = self.end_of_episode_state
        return obs, reward, True, {}

    def reset(self):
        self.ans = np.random.choice(list(range(10)))
        options = self.data[self.ans]
        ind = np.random.randint(0, options.shape[0])
        self.state = self.data[self.ans][ind]
        return self.state

    def render(self, mode="human"):
        print()
        print("--- MNIST_ENV ---")
        if self.state is None:
            print("No image")
        else:
            im_bin = (self.state[0, :, :] * 3).round()
            conv_dict = {3: " #",
                         2: " *",
                         1: " -",
                         0: "  "}
            for i in range(28):
                im_str = ""
                for j in range(28):
                    im_str += conv_dict[int(im_bin[i, j])]
                print(im_str)
        print("---  ======== ---")
        print()
