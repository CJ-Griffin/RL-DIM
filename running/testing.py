import os
from datetime import datetime
import numpy as np

import custom_envs.bandit_env
import gym
from running import run
from agents import *
from torch.utils.tensorboard import SummaryWriter

env_to_train_ep_num = {
    "bandit": (custom_envs.bandit_env.BanditEnv(), int(1e1)),
    "blackjack": (gym.make("Blackjack-v1"), int(1e5)),
    "frozen": (gym.make("FrozenLake-v1"), int(1e6)),
    "taxi": (gym.make("Taxi-v3"), int(1e5))
}

if __name__ == "__main__":
    print("Available evs: " + str(list(gym.envs.registry.env_specs)))
    env_names = list(env_to_train_ep_num.keys())
    env_name = "0" # input(f"Which env to run? {list(enumerate(env_names))}")
    if env_name.isdigit():
        env_name = env_names[int(env_name)]
    print(f"Running {env_name}")
    env, num_train_eps = env_to_train_ep_num[env_name]

    agent_dict = {
        "rand": RandomAgent(env.action_space, env.observation_space),
        "MC": TabularMC(env.action_space, env.observation_space, gamma=1.0),
        "SARSA": SARSA(env.action_space, env.observation_space),
        "QLearner": QLearner(env.action_space, env.observation_space),
        # "DQN": DQN(env.action_space, env.observation_space)
    }
    if env_name == "blackjack":
        agent_dict["GOFAI"] = BlackjackAgent(env.action_space, env.observation_space)

    base_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"For tensorboard, run: \n tensorboard --logdir={base_dir} \n")
    scores = {}
    writers = {}

    for agent_name in agent_dict.keys():
        writers[agent_name] = SummaryWriter(os.path.join(base_dir, agent_name))
        agent = agent_dict[agent_name]
        if agent.NEEDS_TRAINING:
            run.run_episodic(agent, env, num_train_eps, writer=writers[agent_name])
        else:
            print(f"skipping training for {agent_name}")
        scores[agent_name] = run.run_eval(agent, env, 100)
        agent.render()

    print(scores)
