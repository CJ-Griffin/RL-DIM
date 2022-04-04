import neptune.new as neptune
import numpy as np
import torch

import gym

from src.agents import Agent
import src.custom_envs
from src.custom_envs import Grid, BaseEnv
from src.run_parameters import TrainParams


def get_env(experiment_params: TrainParams) -> BaseEnv:
    env_name = experiment_params.env_name
    env_class = getattr(src.custom_envs, env_name)
    env = env_class()
    if issubclass(env_class, Grid):
        env.init_dist_measure(mu=experiment_params.mu,
                              dist_measure_name=experiment_params.dist_measure_name)
    return env


def get_agent(env: BaseEnv, experiment_params: TrainParams) -> Agent:
    agent_name = experiment_params.agent_name
    if agent_name[:5] != "save.":
        agent_class = getattr(src.agents, agent_name)
        agent = agent_class(action_space=env.action_space,
                            state_space=env.observation_space,
                            params=experiment_params)
    else:
        run_name, ep_no = (agent_name[5:]).split("_")
        agent = load_agent_from_neptune(run_name, int(ep_no))
        # We do this to avoid a weird np generator error
        agent.update_state_action_spaces(state_space=env.observation_space,
                                         action_space=env.action_space)
    return agent


def save_agent_to_neptune(agent: Agent, nept_log: neptune.Run, episode_num: int):
    an = agent.get_unique_name()
    path = f"models/temp/{an}.pt"
    agent.save(path)
    nept_log[f"model_saves/q_net-{episode_num}"].upload(path)


def load_agent_from_neptune(run_name: str, ep_no: int) -> (torch.nn.Module, neptune.Run):
    destination_path = f"models/temp/download_{run_name}_{ep_no}.pt"

    nept_log = neptune.init(project="cj.griffin/RL4YP",
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjE4NGNlOC0wMmFjLTQxZTEtODg1ZC0xMDRhMTg3YjI2ZjAifQ==",
                            run=run_name)
    nept_log[f"model_saves/q_net-{ep_no}"].download(destination_path)
    nept_log.stop()

    if torch.cuda.is_available():
        agent = torch.load(destination_path)
    else:
        agent = torch.load(destination_path, map_location=torch.device('cpu'))
    return agent
