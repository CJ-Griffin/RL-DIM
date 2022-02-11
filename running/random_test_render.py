import numpy as np
import gym
import run
from agents import *

def load_random_env():
    env_names = list(gym.envs.registry.env_specs)
    print(env_names)
    loaded = False
    i=0
    while not loaded and i < 1000:
        i += 1
        env_name = np.random.choice(env_names)
        try:
            env = gym.make(env_name)
            loaded = True
            print(f"Loaded {env_name}")
        except Exception as e:
            pass
    return env


if __name__ == "__main__":
    env = load_random_env()
    agent = RandomAgent(env.action_space, env.observation_space)
    run.run_episode(agent, env)
