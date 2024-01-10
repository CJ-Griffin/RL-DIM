# Useful python libraries
import argparse
import datetime

from tqdm import tqdm

import sys

gym_path = '/home/catherine/RL-DIM/gym'  # Path to the parent of the inner 'gym' folder
sys.path.insert(0, gym_path)
# print("Gym module path:", gym.__file__)
import numpy as np
import gym
import copy

# My imports
from src.agents import Agent, QLearner
from src.custom_envs import Grid, Repulsion
from src.custom_envs.grid_envs import *
from src.skein_definitions import SKEIN_DICT
from src.run_parameters import TrainParams
from src.utils.generic_utils import init_neptune_log, reduce_res_freq, save_recordings
from src.utils.rl_utils import get_env, get_agent, save_agent_to_neptune

def run_skein(params_list: list[TrainParams], skein_id: str,
              experiment_name: str, is_parallel: bool = True):
    if is_parallel:
        from multiprocessing import Pool, cpu_count
        from functools import partial

        f = partial(run_experiment, skein_id=skein_id, experiment_name=experiment_name)
        n = int(min(cpu_count(), 4))
        with Pool(n) as p:
            print(f"Pooling with {n}")
            p.map(f, params_list)
    else:
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


def run_experiment(params: TrainParams, skein_id: str, experiment_name: str):
    env = get_env(params)
    if env is None:
        raise ValueError(f"Environment could not be created with parameters: {params}")
    agent = get_agent(env, params)
    if agent is None:
        raise ValueError(f"Agent could not be created with parameters: {params}")

    # if we're doing FTR, then create a counterfactual env and agent
    if params.should_future_task_reward:
        env_cf = get_env(params)  # Create a new environment instance
        agent_cf = get_agent(env_cf, params)  # Create a new agent instance for the counterfactual environment

    # TODO sort what happens when you render cont_env
    env.render(mode="")
    agent.render()
    if agent.REQUIRES_TRAINING:
        episode_scores, total_info = run_episodic(agent=agent,
                                                  agent_cf=agent_cf,
                                                  env=env,
                                                  env_cf=env_cf,
                                                  num_episodes=params.num_episodes,
                                                  should_tqdm=params.is_test,
                                                  should_future_task_reward=params.should_future_task_reward)
    else:
        episode_scores = []
        total_info = {}
        print(f"skipping training for {params.agent_name}")

    # if not params.agent_name == "HumanAgent":
    #     eval_score, eval_info = run_episodic(agent, env, 100, is_eval=True)
    #     print(params.agent_name, eval_score)
    # else:
    #     eval_score = 0
    #     eval_info = 0

    if params.should_render and not params.agent_name == "HumanAgent":
        for ep_no in range(5):
            print("=" * 20 + f" Sample Render {ep_no} " + "=" * 20)
            run_episode(agent, env, should_render=True)

    if params.agent_name == "HumanAgent":
        env.start_recording()
        info = run_episode(agent, env, should_render=True)
        print(info)
        env.stop_and_log_recording(0)
        env.start_recording()
        run_episode(agent, env, should_render=True)
        env.stop_and_log_recording(0)
        print(info)
        env.start_recording()
        run_episode(agent, env, should_render=True)
        env.stop_and_log_recording(0)
        print(info)

    recordings = env.get_recordings()

    if not params.should_skip_neptune:
        nept_log = init_neptune_log(params, skein_id, experiment_name)
        for (key, val) in total_info.items():
            if isinstance(val, list):
                if len(val) > 1000:
                    val = reduce_res_freq(val)
                for elem in val:
                    nept_log[key].log(elem)
            else:
                nept_log[key] = val

        save_agent_to_neptune(agent, nept_log, -1)
        if len(recordings) > 0:
            save_recordings(nept_log, recordings)
        nept_log.stop()


def update_total_info(total_info, info, eval_info):
    for key, val in info.items():
        # if isinstance(val, np.ndarray):
        #     for i in range(len(val)):
        #         new_key = f"{key}_{i}"
        #         if new_key not in total_info:
        #             total_info[new_key] = [info[key][i]]
        #         else:
        #             total_info[new_key].append([info[key][i]])
        # else:
        if key not in total_info:
            total_info[key] = [val]
        else:
            total_info[key].append(val)

    for (key, val) in eval_info.items():
        ekey = f"eval/{key}"
        if ekey not in total_info:
            total_info[ekey] = [val]
        else:
            total_info[ekey].append(val)


def run_episodic(agent: Agent,
                 agent_cf: Agent,
                 env: gym.Env,
                 env_cf: gym.Env,
                 num_episodes: int,
                 is_eval: bool = False,
                 eval_freq: int = None,
                 should_tqdm: bool = False,
                 should_future_task_reward: bool = False):
    if eval_freq is None:
        eval_freq = min(int(1e3), num_episodes // 10)
    episode_scores = []

    if should_tqdm:
        ep_iter = tqdm(range(num_episodes))
    else:
        ep_iter = range(num_episodes)

    total_info = {}

    for ep_num in ep_iter:
        should_record_this_ep = str(ep_num + 1)[1:] in "0" * 20 and str(ep_num + 1)[0] in ["5", "1"]
        if should_record_this_ep:
            env.start_recording()

        if should_future_task_reward:
            info = run_episode(agent, env, is_eval=is_eval, agent_cf=agent_cf, env_cf=env_cf,
                               should_future_task_reward=should_future_task_reward)
        else:
            info = run_episode(agent, env, is_eval=is_eval)

        episode_scores.append(info["ep_score"])

        if should_record_this_ep:
            env.stop_and_log_recording((-ep_num if is_eval else ep_num))

        if (ep_num + 1) % eval_freq == 0:
            env.start_recording()
            eval_info = run_episode(agent=agent,
                                    env=env,
                                    is_eval=True,
                                    agent_cf=agent_cf,
                                    env_cf=env_cf)
            env.stop_and_log_recording(-ep_num)
            if should_tqdm:
                ep_iter.set_description(f"AS = {eval_info['ep_score']}")
        else:
            eval_info = {}
        update_total_info(total_info, info, eval_info)

    return episode_scores, total_info


# handles stepping an agent through an environment
def step_agent_in_env(agent, env, state, should_render=False, future_goal=False):
    if should_render:
        env.render()

    # TODO figure out how to handle future_goal_state
    action = agent.act(state)
    next_state, reward, done, info = env.step(action, future_goal)
    agent.step(state, action, reward, next_state, done)

    return next_state, reward, done, info


def run_episode(agent, env, should_render=False, max_steps=int(1e4), is_eval=False, agent_cf=None, env_cf=None,
                should_future_task_reward=False):
    state = env.reset()
    num_steps = 0
    done = False
    total_rewards = 0
    aux_reward = 0
    spec_rewards = []
    dist_rewards = []
    aux_rewards = []

    agent.is_eval_mode = is_eval

    while not done and num_steps < max_steps:
        num_steps += 1

        # evaluate performance on future goal state
        # TODO make probabilistic
        if should_future_task_reward:
            aux_reward = simulate_future_task(env, agent, env_cf, agent_cf, max_steps, env.future_goal_state())
            total_rewards += aux_reward
            aux_rewards.append(aux_reward)

        next_state, reward, done, info = step_agent_in_env(agent, env, state, should_render)
        state = next_state
        total_rewards += reward

        if isinstance(env, Grid):
            spec_rewards.append(info["spec_reward"])
            dist_rewards.append(info["dist_reward"])

    # Post episode processing
    if isinstance(env, Grid):
        vases_smashed = env.get_vases_smashed()
        doors_left_open = env.get_doors_left_open()
        sushi_eaten = env.get_num_sushi_eaten()
    else:
        vases_smashed = 0
        doors_left_open = 0
        sushi_eaten = 0
    env.close()

    spec_score = sum(spec_rewards) if spec_rewards else 0
    dist_score = sum(dist_rewards) if dist_rewards else 0
    aux_score = sum(aux_rewards) if aux_rewards else 0

    episode_info = {
        "ep_score": total_rewards,
        "spec_score": spec_score,
        "dist_score": dist_score,
        "aux_score": aux_score,
        "vases_smashed": vases_smashed,
        "doors_left_open": doors_left_open,
        "sushi_eaten": sushi_eaten,
        "num_steps": num_steps,
        "td_error": float(sum(agent.td_error_log)) / len(agent.td_error_log) if isinstance(agent, QLearner) else 0
    }
    return episode_info


def simulate_future_task(env, agent, env_cf, agent_cf, max_steps, future_goal_state):
    # takes some parameters as input, returns the auxiliary reward

    # Save the current state of the actual environment, so we can restore it
    saved_state = env.save_state()
    saved_state_cf = env_cf.save_state()

    state = env.get_state()
    state_cf = env_cf.get_state()

    for s in range(max_steps):
        state_actual, _, done_actual, _ = step_agent_in_env(agent, env, state, future_goal=True)
        state_cf, _, done_cf, _ = step_agent_in_env(agent_cf, env_cf, state_cf, future_goal=True)

        # Check if either agent has reached the future goal state
        if np.array_equal(state_actual, future_goal_state) and np.array_equal(state_cf, future_goal_state):
            reward = 1
            break
        elif np.array_equal(state_cf, future_goal_state) and not np.array_equal(state_actual, future_goal_state):
            reward = -1
            break
    else:
        # If neither reached the goal state, return 0
        reward = 0

    # Restore the original states
    env.load_state(saved_state)
    env_cf.load_state(saved_state_cf)

    return reward


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '/home/catherine/RL-DIM')
    g_prsr = argparse.ArgumentParser()
    g_prsr.add_argument("-e", "--experiment_name",
                        choices=list(SKEIN_DICT.keys()) + [None],
                        default=None)
    g_prsr.add_argument('-p', "--parallel", action='store_true')
    g_prsr.add_argument('-t', "--trial_run", action='store_true')
    g_args = g_prsr.parse_args()
    if g_args.experiment_name is not None:
        g_choice = g_args.experiment_name
    else:
        import enquiries

        g_choice = enquiries.choose('Choose an experiment to run: ', SKEIN_DICT.keys())
        print(f"CHOSEN: {g_choice}")
    g_skein = SKEIN_DICT[g_choice]
    g_time_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    g_skein_id = f"{g_choice}_{g_time_id}"
    if g_args.trial_run:
        for g_exp in g_skein:
            g_exp.num_episodes = 1000
            g_exp.is_test = True
    if not g_skein[0].should_profile:
        run_skein(g_skein, g_skein_id, g_choice, g_args.parallel)
    else:
        import cProfile, pstats

        g_profiler = cProfile.Profile()

        g_profiler.enable()
        run_skein(g_skein, g_skein_id, g_choice)
        g_profiler.disable()
        g_stats = pstats.Stats(g_profiler).sort_stats('cumtime')
        g_stats.print_stats()
