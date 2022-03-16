import custom_envs
from running import run
import agents
from def_params import def_params
import gym
from running.run_parameters import TrainParams


def get_env(experiment_params: TrainParams):
    env_name = experiment_params.env_name
    env_class = getattr(custom_envs, env_name)
    env = env_class()
    return env


def get_agent(env: gym.Env, experiment_params: TrainParams):
    agent_name = experiment_params.agent_name
    agent_class = getattr(agents, agent_name)
    agent = agent_class(action_space=env.action_space,
                        state_space=env.observation_space)
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


if __name__ == '__main__':
    if not def_params.should_profile:
        run_experiment(def_params)
    else:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()
        run_experiment(def_params)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
