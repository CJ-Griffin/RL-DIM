import gym

from src.agents import Agent
import src.custom_envs
from src.run_parameters import TrainParams


def get_env(experiment_params: TrainParams) -> gym.Env:
    env_name = experiment_params.env_name
    env_class = getattr(src.custom_envs, env_name)
    env = env_class()
    return env


def get_agent(env: gym.Env, experiment_params: TrainParams) -> Agent:
    agent_name = experiment_params.agent_name
    agent_class = getattr(src.agents, agent_name)
    agent = agent_class(action_space=env.action_space,
                        state_space=env.observation_space,
                        params=experiment_params)
    return agent
