import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import gym
from agents import Agent
from torch.utils.tensorboard import SummaryWriter


def run_episode(agent: Agent,
                env: gym.Env,
                max_steps: int = int(1e4),
                # writer: SummaryWriter = None,
                prev_steps: int = 0,
                # uncertainty_writers = None,
                should_render: bool = False) -> (int, float):
    state = env.reset()
    num_steps = 0
    done = False
    rewards = []

    while not done and num_steps <= max_steps:
        num_steps += 1
        if should_render:
            env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        rewards.append(reward)

    env.close()
    # TODO - Question whether this is really the score
    score = np.sum(rewards)
    # print(agent)
    return num_steps, score


def run_episodic(agent: Agent,
                 env: gym.Env,
                 num_episodes: int,
                 should_render: bool = False,
                 should_show_prog_bar: bool = True,
                 should_eval=True,
                 episode_record_interval: int = 1000):
    prev_steps = 0
    if not should_show_prog_bar:
        ep_iter = range(num_episodes)
    else:
        ep_iter = tqdm(range(num_episodes))
    # eval_scores = []
    episode_scores = []
    for ep_num in tqdm(range(num_episodes)):
        # num_steps, _ = run_episode(agent, env, writer=writer, prev_steps=prev_steps, should_render=should_render, uncertainty_writers=uncertainty_writers)
        if ep_num % episode_record_interval == 0:
            env.start_recording()

        num_steps, score = run_episode(agent, env,
                                       prev_steps=prev_steps,
                                       should_render=should_render)
        prev_steps += num_steps
        episode_scores.append(score)
        if ep_num % episode_record_interval == 0:
            env.stop_and_log_recording(ep_num)
        if should_show_prog_bar and ep_num % 10000 == 0:
            ep_iter.set_description(f"AS = {np.mean(episode_scores[-1000:])}")
        # if should_eval and ep_num-1%1000 == 0:
    #     #     eval_scores.append(run_eval(agent, env, 100))
    # if should_eval:
    #     plt.plot(eval_scores)
    #     plt.show()
    return episode_scores


def run_eval(agent: Agent,
             env: gym.Env,
             num_episodes: int,
             should_render: bool = False,
             num_eval_recordings: int = 10) -> float:
    scores = []
    for ep_num in range(num_episodes):
        if ep_num <= num_eval_recordings:
            env.start_recording()
        scores.append(run_episode(agent, env, should_render=should_render)[1])
        if ep_num <= num_eval_recordings:
            env.stop_and_log_recording(-ep_num)
    return np.mean(scores)


def run_steps(agent: Agent,
              env: gym.Env,
              max_interacts: int,
              writer: SummaryWriter = None):
    # Currently, it will run up to an episode over (no max_steps within run_episode)
    # TODO Consider changing this
    total_interacts = 0
    while total_interacts < max_interacts:
        episode_steps = run_episode(agent, env, writer=writer)
        total_interacts += episode_steps
