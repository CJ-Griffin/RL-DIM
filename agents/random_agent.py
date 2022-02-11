import gym
from agents.agent import Agent


class RandomAgent(Agent):

    def act(self, state):
        # assert (state in self._state_space), (state, self._state_space)
        return self._action_space.sample()

    def step(self, state, action, reward: float, next_state, done: bool):
        pass

    def update(self):
        pass

    def render(self):
        print("RandomAgent over space: " + str(self._action_space))
