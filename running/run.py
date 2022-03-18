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
        # if writer is not None:
        #     writer.add_scalar("Reward", reward, prev_steps+num_steps)
            # if hasattr(agent, "get_mean_and_uncertainty"):
            #     if (prev_steps+num_steps) % 100 == 0:
            #         means, uncertainties = agent.get_mean_and_uncertainty(state)
            #         n = uncertainties.shape[0]
            #         for i in range(0, n):
            #             uncertainty_writers[i].add_scalar(f"ActionUncertainty{state}", uncertainties[i], prev_steps+num_steps)
            #             uncertainty_writers[i].add_scalar(f"ActionPred{state}", means[i], prev_steps+num_steps)
    env.close()
    score = np.sum(rewards)
    # print(agent)
    return num_steps, score


def run_episodic(agent: Agent,
                 env: gym.Env,
                 num_episodes: int,
                 # writer: SummaryWriter = None,
                 should_render: bool = False,
                 should_show_prog_bar: bool = True,
                 # uncertainty_writers=None,
                 should_eval=True):
    env.render()
    agent.render()
    prev_steps = 0
    if not should_show_prog_bar:
        ep_iter = range(num_episodes)
    else:
        ep_iter = tqdm(range(num_episodes))
    # eval_scores = []
    for ep_num in ep_iter:
        # num_steps, _ = run_episode(agent, env, writer=writer, prev_steps=prev_steps, should_render=should_render, uncertainty_writers=uncertainty_writers)
        num_steps, _ = run_episode(agent, env, prev_steps=prev_steps, should_render=should_render)
        prev_steps += num_steps
        # if should_eval and ep_num-1%1000 == 0:
    #     #     eval_scores.append(run_eval(agent, env, 100))
    # if should_eval:
    #     plt.plot(eval_scores)
    #     plt.show()


def run_eval(agent: Agent,
             env: gym.Env,
             num_episodes: int,
             should_render: bool = False) -> float:
    scores = [run_episode(agent, env, should_render=should_render)[1] for i in range(num_episodes)]
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

