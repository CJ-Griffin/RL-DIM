# Useful python libraries
import argparse
import datetime

from tqdm import tqdm

import numpy as np
import gym

# My imports
from src.agents import Agent, QLearner
from src.custom_envs import Grid, Repulsion
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
    print(params)
    env = get_env(params)
    agent = get_agent(env, params)
    # TODO sort what happens when you render cont_env
    env.render(mode="")
    agent.render()
    if agent.REQUIRES_TRAINING:
        episode_scores, total_info = run_episodic(agent=agent,
                                                  env=env,
                                                  num_episodes=params.num_episodes,
                                                  should_tqdm=params.is_test)
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
                 env: gym.Env,
                 num_episodes: int,
                 is_eval: bool = False,
                 eval_freq: int = None,
                 should_tqdm: bool = False):
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

        info = run_episode(agent, env, is_eval=is_eval)
        episode_scores.append(info["ep_score"])

        if should_record_this_ep:
            env.stop_and_log_recording((-ep_num if is_eval else ep_num))

        if (ep_num + 1) % eval_freq == 0:
            env.start_recording()
            eval_info = run_episode(agent=agent,
                                    env=env,
                                    is_eval=True)
            env.stop_and_log_recording(-ep_num)
            if should_tqdm:
                ep_iter.set_description(f"AS = {eval_info['ep_score']}")
        else:
            eval_info = {}
        update_total_info(total_info, info, eval_info)

    return episode_scores, total_info


def run_episode(agent: Agent,
                env: gym.Env,
                max_steps: int = int(1e4),
                should_render: bool = False,
                is_eval=False) -> dict:
    state = env.reset()
    num_steps = 0
    done = False
    rewards = []
    spec_rewards = []
    dist_rewards = []
    action_freqs = np.zeros(9)

    agent.is_eval_mode = is_eval

    while not done and num_steps <= max_steps:
        num_steps += 1
        if should_render:
            env.render()
        # if is_eval:
        #     env.render()
        #     if agent.__class__.__name__ == "QLearner":
        #         print(agent._Q[agent.get_hashable_state(state)])
        action = agent.act(state)
        action_freqs[action] += 1
        next_state, reward, done, info = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        rewards.append(reward)
        if isinstance(env, Grid):
            spec_rewards.append(info["spec_reward"])
            dist_rewards.append(info["dist_reward"])

    agent.is_eval_mode = False

    if isinstance(env, Grid):
        vases_smashed = env.get_vases_smashed()
        doors_left_open = env.get_doors_left_open()
        sushi_eaten = env.get_num_sushi_eaten()
    else:
        vases_smashed = 0
        doors_left_open = 0
        sushi_eaten = 0
    env.close()
    score = sum(rewards)  # TODO consider different episode scores
    if isinstance(env, Grid) or isinstance(env, Repulsion):
        spec_score = sum(spec_rewards)
        dist_score = sum(dist_rewards)
    if isinstance(agent, QLearner):
        td_error = float(sum(agent.td_error_log))/len(agent.td_error_log)
    else:
        td_error = 0
    return {
        # "num_steps": num_steps,
        "ep_score": score,
        "spec_score": spec_score,
        "dist_score": dist_score,
        # "action_freqs": action_freqs / (len(action_freqs) + 1),
        "vases_smashed": vases_smashed,
        "doors_left_open": doors_left_open,
        "sushi_eaten": sushi_eaten,
        "num_steps": num_steps,
        "td_error:": td_error
    }


if __name__ == '__main__':
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
