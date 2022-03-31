from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt

import gym


def imshow_grid(grid: np.ndarray):
    if len(grid.shape) == 3 and 3 in grid.shape:
        if grid.shape[0] == 3:
            grid2 = np.swapaxes(grid * 255, 0, 2)
            grid3 = np.swapaxes(grid2, 0, 1)
            plt.imshow(grid3)
            plt.show()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def is_space_finite(space: gym.Space) -> bool:
    known_simple_finite_spaces = [
        gym.spaces.Discrete,
        gym.spaces.MultiDiscrete,
        gym.spaces.MultiBinary
    ]

    known_simple_infinite_spaces = [
        # gym.spaces.Box  # Finite when bounded and integer
    ]

    for space_type in known_simple_finite_spaces:
        if isinstance(space, space_type):
            return True
    for space_type in known_simple_infinite_spaces:
        if isinstance(space, space_type):
            return False
    if isinstance(space, gym.spaces.Box):
        if not np.all(space.bounded_below) or not np.all(space.bounded_above):
            return False
        else:
            if space.dtype is np.uint8:
                return True
            else:
                return NotImplementedError, space
    if hasattr(space, "spaces"):
        spaces = space.spaces
        if isinstance(spaces, list):
            spaces = spaces
        elif isinstance(spaces, dict):
            spaces = list(spaces.values())
        elif isinstance(spaces, tuple):
            spaces = list(spaces)
        else:
            raise NotImplementedError(space, spaces)
        print(spaces, "--")
        are_spaces_finite = [is_space_finite(sub_space) for sub_space in spaces]
        return np.all(are_spaces_finite)

    raise NotImplementedError("I don't know how to handle this yet", space)


def get_action_list(action_space: gym.Space):
    assert is_space_finite(action_space), \
        f"Cannot get finite list of actions from an infinite space: {action_space}"
    if isinstance(action_space, gym.spaces.Discrete):
        start = action_space.start
        return list(range(start, start + action_space.n))
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        raise NotImplementedError(action_space)
        # import itertools
        # limits = action_space.nvec
        # ind_lists = [range(lim) for lim in limits]
        # actions = itertools.product(ind_lists)
        # return actions

    else:
        raise NotImplementedError


def vectorise_state(state: gym.core.ActType) -> torch.Tensor:
    if isinstance(state, int):
        return torch.tensor([state]).float()
    elif isinstance(state, tuple):
        return torch.tensor(state).float()
    elif isinstance(state, np.ndarray):
        return torch.tensor(state).flatten().float()
    elif isinstance(state, dict):
        raise Exception("This probably shouldnt happen!")
        # if list(state.keys()) == ['image']:
        #     return vectorise_state(state['image'])
        # else:
        #     raise NotImplementedError(state)
    elif isinstance(state, torch.Tensor):
        return state.flatten().float()
    else:
        raise NotImplementedError(state)


def imageify_state(state) -> torch.Tensor:
    if isinstance(state, int):
        return np.ones((1, 3, 10, 10)) * state
    elif isinstance(state, list) or isinstance(state, np.ndarray):
        state = torch.tensor(state).float()
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        if state.shape[1] == 3:
            return state
        else:
            raise NotImplementedError
        # print(torch.tensor(state).shape)
        # image = torch.tensor(state).transpose(0,2).float().unsqueeze(0)
        # assert(image.shape[1] == 3), image.shape
        # return image
    # elif isinstance(state, np.ndarray):
    #     raise NotImplementedError(state)
    elif isinstance(state, torch.Tensor):
        if len(state.shape) == 3 and state.shape[0] == 3:
            return state.unsqueeze(0)
        if len(state.shape) == 4 and state.shape[1] == 3:
            return state
    else:
        raise NotImplementedError(state)


def generate_random_string(n: int) -> str:
    letters = list("abcdefghijklmnopqrstuvwxyz")
    chosen = np.random.choice(letters, n)
    return "".join(list(chosen))


def get_datetime_string() -> str:
    dt = datetime.now().strftime("%Ym%d_%H%M%S")


def are_sets_independent(sets: set[set]) -> bool:
    for s1 in sets:
        for s2 in sets - {s1}:
            if not len(s1 & s2) == 0:
                return False
    return True
