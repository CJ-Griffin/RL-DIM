import argparse
import datetime
import math
import os

import numpy as np
from matplotlib import pyplot as plt

import custom_envs
from running import run
import agents
from def_params import skein_dict
import gym
from running.run_parameters import TrainParams
import neptune.new as neptune
from neptune.new.exceptions import CannotResolveHostname
from array2gif import write_gif


def get_env(experiment_params: TrainParams) -> gym.Env:
    env_name = experiment_params.env_name
    env_class = getattr(custom_envs, env_name)
    env = env_class()
    return env


def get_agent(env: gym.Env, experiment_params: TrainParams) -> agents.Agent:
    agent_name = experiment_params.agent_name
    agent_class = getattr(agents, agent_name)
    agent = agent_class(action_space=env.action_space,
                        state_space=env.observation_space,
                        params=experiment_params)
    return agent


def init_neptune_log(params: TrainParams, skein_id: str, experiment_name: str):
    if params.should_skip_neptune:
        return None
    else:
        try:
            nept_log = neptune.init(project="cj.griffin/RL4YP",
                                    api_token=os.getenv('NEPTUNE_API_TOKEN'), )
            nept_log["parameters"] = params.get_dict()
            nept_log["skein_id"] = skein_id
            nept_log["experiment_name"] = experiment_name
            return nept_log
        except CannotResolveHostname as connect_error:
            if params.is_test:
                print("FAILED TO CONNECT TO NEPTUNE")
                return None
            else:
                raise connect_error


def save_recordings(nept_log: neptune.Run, skein_id: str, recordings: dict[list[np.array]]):
    # dir_name = f"logs/episode_gifs/{skein_id}/{np.random.randint(1000)}"
    # os.mkdir(dir_name)
    for ep_no, recording in recordings.items():
        _, h, w = recording[0].shape
        for frame in recording:
            assert frame.shape == (3, h, w), frame.shape
        recording_scaled = [frame * 255 for frame in recording]
        # path = f"{dir_name}/ep{ep_no}.gif"
        path = f"logs/episode_gifs/temp.gif"
        write_gif(recording_scaled, path, fps=5)
        nept_log[f"ep_gifs/ep{ep_no}"].upload(path)


def run_experiment(params: TrainParams, skein_id: str, experiment_name: str):
    nept_log = init_neptune_log(params, skein_id, experiment_name)

    print(params)
    env = get_env(params)
    agent = get_agent(env, params)
    env.render()
    agent.render()
    if agent.REQUIRES_TRAINING:
        episode_scores = run.run_episodic(agent=agent,
                                          env=env,
                                          num_episodes=params.num_episodes,
                                          should_render=params.should_render)
        for score in episode_scores:
            if nept_log is not None:
                nept_log["ep_scores"].log(score)
    else:
        episode_scores = []
        print(f"skipping training for {params.agent_name}")

    num_eval_episodes = 1 if params.agent_name == "HumanAgent" else 200
    eval_score = run.run_eval(agent, env, num_eval_episodes, should_render=(params.agent_name == "HumanAgent"))
    if nept_log is not None:
        nept_log["eval_score"] = eval_score
    if params.should_render and not params.agent_name == "HumanAgent":
        run.run_eval(agent, env, 5, should_render=params.should_render)

    agent.render()

    print(params.agent_name, eval_score)
    if nept_log is not None:
        if len(env.recordings) > 0:
            save_recordings(nept_log, skein_id, env.recordings)
        nept_log.stop()
    else:
        resolution = 1000
        n = math.floor(len(episode_scores) / resolution)
        episode_scores_averaged = [np.mean(episode_scores[i * resolution: (i + 1) * resolution]) for i in range(n)]
        plt.plot(episode_scores_averaged)
        plt.show()


def run_skein(params_list: list[TrainParams], skein_id: str, experiment_name: str):
    for exp_num, exp_params in enumerate(params_list):
        print()
        print()
        print("=" * 214)
        print(("-" * 100) + f" Experiment {exp_num} " + ("-" * 100))
        run_experiment(params=exp_params,
                       skein_id=skein_id,
                       experiment_name=experiment_name)
        print("-" * 214)
        print("=" * 214)
        print()
        print()


if __name__ == '__main__':
    prsr = argparse.ArgumentParser()
    prsr.add_argument("-e", "--experiment_name",
                      choices=list(skein_dict.keys()) + [None],
                      default=None)
    args = prsr.parse_args()
    if args.experiment_name is not None:
        choice = args.experiment_name
    else:
        import enquiries

        choice = enquiries.choose('Choose an experiment to run: ', skein_dict.keys())
        print(f"CHOSEN: {choice}")
    skein = skein_dict[choice]
    time_id = datetime.datetime.now().strftime('%Ym%d_%H%M%S')
    skein_id = f"{choice}_{time_id}"
    if not skein[0].should_profile:
        run_skein(skein, skein_id, choice)
    else:
        import cProfile, pstats

        profiler = cProfile.Profile()

        profiler.enable()
        run_skein(skein, skein_id, choice)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
