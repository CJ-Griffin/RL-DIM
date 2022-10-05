import numpy as np

from src.agents.agent import Agent


class HumanAgent(Agent):
    REQUIRES_TRAINING = False
    REQUIRES_FINITE_STATE_SPACE = False
    REQUIRES_FINITE_ACTION_SPACE = False

    def act(self, state):
        # assert (state in self._state_space), (state, self._state_space)
        key_dict = {
            "w": 0,
            "d": 1,
            "s": 2,
            "a": 3,
            "e": 4,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9
        }
        key = None
        while key not in key_dict:
            key = input()
        action = key_dict[key]
        assert action in self._action_space, (action, self._action_space)
        return action

    def step(self, state, action, reward: float, next_state, done: bool):
        print(f"Action {action} got reward {reward}")
        pass

    def update(self):
        pass

    def render(self):
        print("RandomAgent over space: " + str(self._action_space))


class HumanAgent(Agent):
    REQUIRES_TRAINING = False
    REQUIRES_FINITE_STATE_SPACE = False
    REQUIRES_FINITE_ACTION_SPACE = False

    def act(self, state):
        # assert (state in self._state_space), (state, self._state_space)
        key_dict = {
            "w": 0,
            "d": 1,
            "s": 2,
            "a": 3,
            "e": 4,
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
            "5": 4,
            "6": 5,
            "7": 6,
            "8": 7,
            "9": 8
        }
        key = None
        while key not in key_dict:
            key = input()
        action = key_dict[key]
        print(self._action_space.sample())
        assert action in self._action_space, (action, self._action_space)
        return action

    def step(self, state, action, reward: float, next_state, done: bool):
        print(f"Action {action} got reward {reward}")
        pass

    def update(self):
        pass

    def render(self):
        print("RandomAgent over space: " + str(self._action_space))
