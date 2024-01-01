from src.custom_envs import Grid, BaseEnv
from src.run_parameters import TrainParams
import src.custom_envs

ALL_ENVS = []
ALL_GRIDS = []
for thing_name in dir(src.custom_envs):
    thing = getattr(src.custom_envs, thing_name)
    try:
        if issubclass(thing, BaseEnv):
            ALL_ENVS.append(thing.__name__)
            if issubclass(thing, Grid):
                ALL_GRIDS.append(thing.__name__)
    except Exception as e:
        pass
ALL_ENVS.remove("Grid")
ALL_GRIDS.remove("Grid")

SOME_AGENT_NAMES = [
    "RandomAgent",
    "HumanAgent",
    "MonteCarlo",
    "SARSA",
    "QLearner",
    "DQN",
    "DQN_CNN"
]

SOME_ENV_NAMES = [
    "BanditEnv",
    "EmptyGrid1D",
    "SmallEmptyGrid",
    "SimpleGrid",
    "DirtGrid",
    "RandDirtGrid",
    "WallGrid",
    "SemiRandWallGrid",
    "Repulsion"
    # "SemiRandDirtWallGrid"
]

SKEIN_DICT = {
    "temp": [
        TrainParams(
            env_name="Repulsion",
            agent_name="HumanAgent",
            num_episodes=1,
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True
        )
    ],

    "future-task-test": [
        TrainParams(
            env_name="SimpleGrid",
            agent_name="QLearner",
            num_episodes=int(10),
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True,
            should_future_task_reward=True
        )
    ],

    "DQN-cont": [
        TrainParams(
            env_name="Repulsion",
            agent_name="DQN",
            num_episodes=int(1e5),
            gamma=0.95,
            is_test=True,
            # should_render=True,ßßß
            # should_debug=True
        )
    ],

    "grid-test": [
        TrainParams(
            env_name="SimpleGrid",
            agent_name="QLearner",
            num_episodes=int(10),
            gamma=0.95,
            is_test=True,
            should_render=True,
            should_debug=True
        )
    ],

    # "temp": [
    #     TrainParams(
    #         env_name="DoorGrid",
    #         agent_name="QLearner",
    #         num_episodes=100,
    #         gamma=0.9,
    #         is_test=True,
    #         should_render=True,
    #         should_debug=True
    #     )
    # ],
    #
    # "SmallMuseumGridTrain": [
    #     TrainParams(
    #         env_name="SmallMuseumGrid",
    #         agent_name=an,
    #         num_episodes=int(2e5),
    #         gamma=0.9,
    #         is_test=True,
    #         # should_render=True,
    #         # should_debug=True,
    #         # should_skip_neptune=True
    #     ) for an in ["DQN_CNN"]
    #     # ["SARSA",
    #     #  "QLearner",
    #     #  "DQN",
    #     #  "DQN_CNN"]  # SOME_AGENT_NAMES if an != "HumanAgent"
    # ],
    #
    # "SimpleWallTrainDQN_CNN": [
    #     TrainParams(
    #         env_name="SimpleWallGrid",
    #         agent_name=an,
    #         num_episodes=int(1e5),
    #         gamma=0.9,
    #         is_test=True,
    #         # should_render=True,
    #         # should_debug=True,
    #         # should_skip_neptune=True
    #     ) for an in ["DQN_CNN"]
    #     # ["SARSA",
    #     #  "QLearner",
    #     #  "DQN",
    #     #  "DQN_CNN"]  # SOME_AGENT_NAMES if an != "HumanAgent"
    # ],
    #
    # "MuseumGridTestTraining": [
    #     TrainParams(
    #         env_name="MuseumGrid",
    #         agent_name=an,
    #         num_episodes=int(2e3),
    #         gamma=0.9,
    #         is_test=True,
    #     ) for an in ["DQN", "RandomAgent"]
    # ],
    #
    # "test_all_envs_and_agents": [
    #     TrainParams(
    #         env_name=env_name,
    #         agent_name=agent_name,
    #         num_episodes=100,
    #         gamma=0.9,
    #         should_debug=False,
    #         is_test=True,
    #         should_skip_neptune=True
    #     )
    #     for env_name in ALL_ENVS if env_name != "BaseEnv"
    #     for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    # ],
    #
    # "human_test_dist_measures": [
    #     TrainParams(
    #         env_name="SmallMuseumGrid",
    #         agent_name="HumanAgent",
    #         num_episodes=100,
    #         gamma=0.9,
    #         should_debug=False,
    #         is_test=True,
    #         dist_measure_name=dmn,
    #         mu=mu
    #     )
    #     for dmn in ["null", "simple", "vase"]
    #     for mu in [-0.1, 10.0]
    # ],
    #
    # "human_test_all_grids": [
    #     TrainParams(
    #         env_name=env_name,
    #         agent_name="HumanAgent",
    #         num_episodes=100,
    #         gamma=0.9,
    #         should_debug=False,
    #         is_test=True
    #     ) for env_name in ALL_GRIDS[2:]
    # ],
    #
    # "human_test_museum_grid": [
    #     TrainParams(
    #         env_name="MuseumGrid",
    #         agent_name="HumanAgent",
    #         num_episodes=100,
    #         gamma=0.9,
    #         should_debug=False,
    #         is_test=True
    #     )
    # ],
    #
    # "human_test_easy_museum_grid": [
    #     TrainParams(
    #         env_name="EasyMuseumGrid",
    #         agent_name="HumanAgent",
    #         num_episodes=100,
    #         gamma=0.9,
    #         should_debug=False,
    #         is_test=True
    #     )
    # ],
    #
    "test_all_agents": list(reversed([
        TrainParams(
            env_name="MuseumRush",
            agent_name=agent_name,
            num_episodes=1000,
            gamma=0.9,
            should_debug=False,
            is_test=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ])),
    #
    # "train_empty_grid": list(reversed([
    #     TrainParams(
    #         env_name="SmallEmptyGrid",
    #         agent_name=agent_name,
    #         num_episodes=1e5,
    #         gamma=0.9,
    #         should_debug=False,
    #         # is_test=True
    #     ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    # ])),
    #
    # "test_DQN_CNN": [
    #     TrainParams(
    #         env_name="TinyEmptyGrid",
    #         agent_name="DQN",
    #         num_episodes=int(1e4),
    #         gamma=0.9,
    #         epsilon=epsilon,
    #         # should_debug=True,
    #         is_test=True
    #     )
    #     for epsilon in [0.05]  # [0.01, 0.05, 0.1]
    # ],
    #
    # "test_loading": [
    #     TrainParams(
    #         env_name="TinyEmptyGrid",
    #         agent_name="save.RLYP-742_9999",
    #         num_episodes=int(1e4),
    #         gamma=0.9,
    #         epsilon=epsilon,
    #         # should_debug=True,
    #         is_test=True
    #     )
    #     for epsilon in [0.05]  # [0.01, 0.05, 0.1]
    # ],
    #
    # "DQN_on_MNIST": [
    #     TrainParams(
    #         env_name="MnistEnv",
    #         agent_name=a,
    #         num_episodes=1e6,
    #         gamma=0.9,
    #         should_debug=False,
    #         should_skip_neptune=False
    #     ) for a in ["DQN_CNN"]
    # ],
}

import src.more_skein_definitions

more_skein_dicts = []
for x in dir(src.more_skein_definitions):
    mod = getattr(src.more_skein_definitions, x)
    if hasattr(mod, "SKEIN_DICT"):
        sd = getattr(mod, "SKEIN_DICT")
        for key in sd.keys():
            SKEIN_DICT[key] = sd[key]
