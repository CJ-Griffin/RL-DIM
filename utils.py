import numpy as np
import torch

import gym


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
    else:
        raise NotImplementedError


def vectorise_state(state: gym.core.ActType) -> torch.Tensor:
    if isinstance(state, int):
        return torch.tensor([state])
    elif isinstance(state, tuple):
        return torch.tensor(state)
    elif isinstance(state, dict):
        if list(state.keys()) == ['image']:
            return vectorise_state(state['image'])
        else:
            raise NotImplementedError(state)
    else:
        raise NotImplementedError(state)


