import gym
from agents.agent import Agent


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
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3
        }
        key = None
        while key not in key_dict:
            key = input()
        action = key_dict[key]
        assert action in self._action_space, (action, self._action_space)
        return action

    def step(self, state, action, reward: float, next_state, done: bool):
        pass

    def update(self):
        pass

    def render(self):
        print("RandomAgent over space: " + str(self._action_space))