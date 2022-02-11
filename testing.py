import os
from argparse import ArgumentParser
from datetime import datetime

from custom_envs.bandit_env import BanditEnv
from running import run
from agents import *
from torch.utils.tensorboard import SummaryWriter
# from gooey import Gooey
from gym_minigrid.wrappers import *
import run_params

env_to_train_ep_num = {
    "bandit": (BanditEnv(), int(1e3)),
    "blackjack": (gym.make("Blackjack-v1"), int(1e7)),
    "frozen": (gym.make("FrozenLake-v1"), int(1e5)),
    "taxi": (gym.make("Taxi-v3"), int(1e5)),
    "empty_grid": (OneHotPartialObsWrapper(gym.make("MiniGrid-Empty-5x5-v0")), int(1e4))
    # "grid": (gym.make('Gridworld-v0'), int(1e5))
}

# @Gooey
def main():
    parser = ArgumentParser(description="Run RL experiments")

    parser.add_argument("--env_name", type=str,
                        default=run_params.ENV_NAME,
                        choices=env_to_train_ep_num.keys(),
                        help="The name of the game to train on")

    parser.add_argument("-e", "--episodes", type=int,
                        default=run_params.EPISODES,
                        help="How many episodes to train for")
    args = parser.parse_args()

    print("Available envs: " + str(list(gym.envs.registry.env_specs)))
    env_names = list(env_to_train_ep_num.keys())
    env_name = args.env_name # input(f"Which env to run? {list(enumerate(env_names))}")
    if env_name.isdigit():
        env_name = env_names[int(env_name)]
    print(f"Running {env_name}")
    env, num_train_eps = env_to_train_ep_num[env_name]
    print(env.__class__)
    if args.episodes is not None:
        num_train_eps = args.episodes

    print(env_name)
    print(env.observation_space, "-----")

    agent_dict = {
        "rand": RandomAgent(env.action_space, env.observation_space),
        "MC": TabularMC(env.action_space, env.observation_space, gamma=1.0),
        # "SARSA": SARSA(env.action_space, env.observation_space),
        # "QLearner": QLearner(env.action_space, env.observation_space),
        "DQN": DQN(env.action_space, env.observation_space)
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
        if agent.REQUIRES_TRAINING:
            run.run_episodic(agent, env, num_train_eps, writer=writers[agent_name])
        else:
            print(f"skipping training for {agent_name}")
        scores[agent_name] = run.run_eval(agent, env, 100)
        agent.render()

    print(scores)


if __name__ == "__main__":
    main()
