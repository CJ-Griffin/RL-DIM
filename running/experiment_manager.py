import argparse

import custom_envs
from running import run
import agents
from def_params import skein_dict
import gym
from running.run_parameters import TrainParams


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


def run_experiment(params: TrainParams):
    print(params)
    env = get_env(params)
    agent = get_agent(env, params)

    if agent.REQUIRES_TRAINING:
        run.run_episodic(agent=agent,
                         env=env,
                         num_episodes=params.num_episodes,
                         should_render=params.should_render)
    else:
        print(f"skipping training for {params.agent_name}")

    score = run.run_eval(agent, env, 100, should_render=(params.agent_name == "human"))
    if params.should_render:
        run.run_eval(agent, env, 5, should_render=params.should_render)

    agent.render()

    print(params.agent_name, score)


def run_skein(params_list: list[TrainParams]):
    for exp_num, exp_params in enumerate(params_list):
        print()
        print(("="*50) + f" Experiment {exp_num} " + ("="*50))
        run_experiment(params=exp_params)


if __name__ == '__main__':
    prsr = argparse.ArgumentParser()
    prsr.add_argument("experiment_name", choices=skein_dict.keys())
    args = prsr.parse_args()
    skein = skein_dict[args.experiment_name]
    if not skein[0].should_profile:
        run_skein(skein)
    else:
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        run_skein(skein)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
