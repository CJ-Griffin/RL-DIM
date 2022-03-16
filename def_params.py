from running.run_parameters import TrainParams

def_params = TrainParams(
    env_name="BanditEnv",
    agent_name="RandomAgent",
    num_episodes=10,
    gamma=None,
    alpha=None,
    epsilon=None,
    buffer_size=None,
    batch_size=None,
    update_freq=None,
    q_init=None,
    learning_rate=None,
    should_debug=False,
    should_render=False,
    should_profile=False
)

# ENV_NAME = "coins"
# EPISODES = None  # Make None to use default from env
# SHOULD_RENDER = True
# SHOULD_PROFILE = False
# GAMMA = 0.9
# SHOULD_DEBUG = False
# AGENTS = [
#     # "human",
#     "rand",
#     # "MC",
#     # "SARSA",
#     # "QLearner",
#     # "DQN",
#     # "DQN_CNN",
#     # "BDQN",
#     # "bayes_coin",
# ]
