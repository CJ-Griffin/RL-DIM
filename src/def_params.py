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
    "TabularMC",
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
    # "SemiRandDirtWallGrid"
]

SKEIN_DICT = {

    "temp": [
        TrainParams(
            env_name="DoorGrid",
            agent_name="QLearner",
            num_episodes=100,
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True
        )
    ],

    "test_simple_wall_grid": [
        TrainParams(
            env_name="SimpleWallGrid",
            agent_name="HumanAgent",
            num_episodes=100,
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True
        )
    ],

    "DoorGridTest": [
        TrainParams(
            env_name="DoorGrid",
            agent_name="HumanAgent",
            num_episodes=100,
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True
        )
    ],

    "VaseGridTest": [
        TrainParams(
            env_name="VaseGrid",
            agent_name="HumanAgent",
            num_episodes=100,
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True
        )
    ],

    "MuseumGridTest": [
        TrainParams(
            env_name="MuseumGrid",
            agent_name="HumanAgent",
            num_episodes=100,
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True,
            should_skip_neptune=True
        )
    ],

    "SmallMuseumGridTest": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="HumanAgent",
            num_episodes=1,
            gamma=0.9,
            is_test=True,
            should_render=True,
            should_debug=True,
            # should_skip_neptune=True
        )
    ],

    "SmallMuseumGridTrain": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name=an,
            num_episodes=int(2e5),
            gamma=0.9,
            is_test=True,
            # should_render=True,
            # should_debug=True,
            # should_skip_neptune=True
        ) for an in ["DQN_CNN"]
        # ["SARSA",
        #  "QLearner",
        #  "DQN",
        #  "DQN_CNN"]  # SOME_AGENT_NAMES if an != "HumanAgent"
    ],

    "SimpleWallTrainDQN_CNN": [
        TrainParams(
            env_name="SimpleWallGrid",
            agent_name=an,
            num_episodes=int(1e5),
            gamma=0.9,
            is_test=True,
            # should_render=True,
            # should_debug=True,
            # should_skip_neptune=True
        ) for an in ["DQN_CNN"]
        # ["SARSA",
        #  "QLearner",
        #  "DQN",
        #  "DQN_CNN"]  # SOME_AGENT_NAMES if an != "HumanAgent"
    ],

    "MuseumGridTrain": [
        TrainParams(
            env_name="MuseumGrid",
            agent_name=an,
            num_episodes=int(2e3),
            gamma=0.9,
            is_test=True,
        ) for an in ["DQN"]
    ],

    "test_all_envs": [
        TrainParams(
            env_name=env_name,
            agent_name="TabularMC",
            num_episodes=100,
            gamma=0.9,
            should_debug=False,
            is_test=True
        ) for env_name in SOME_ENV_NAMES
    ],

    "human_test_all_grids": [
        TrainParams(
            env_name=env_name,
            agent_name="HumanAgent",
            num_episodes=100,
            gamma=0.9,
            should_debug=False,
            is_test=True
        ) for env_name in ALL_GRIDS[2:]
    ],

    "test_all_agents": list(reversed([
        TrainParams(
            env_name="SmallEmptyGrid",
            agent_name=agent_name,
            num_episodes=1000,
            gamma=0.9,
            should_debug=False,
            is_test=True
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ])),

    "train_empty_grid": list(reversed([
        TrainParams(
            env_name="SmallEmptyGrid",
            agent_name=agent_name,
            num_episodes=1e5,
            gamma=0.9,
            should_debug=False,
            # is_test=True
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ])),

    "test_DQN_CNN": [
        TrainParams(
            env_name="TinyEmptyGrid",
            agent_name="DQN",
            num_episodes=int(1e4),
            gamma=0.9,
            epsilon=epsilon,
            # should_debug=True
        )
        for epsilon in [0.05]  # [0.01, 0.05, 0.1]
    ],

    "TinyEmptyGrid_DQN_HyperParamSearch": [
        TrainParams(
            env_name="TinyEmptyGrid",
            agent_name="DQN",
            num_episodes=int(1e3),
            gamma=0.9,
            batch_size=batch_size,
            buffer_size=buffer_size,
            update_freq=update_freq
        )
        for batch_size in [4, 16, 64]
        for buffer_size in [int(1e3), int(1e4), int(1e5)]
        for update_freq in [int(1e1), int(1e2), int(1e3)]
    ],

    "DQN_on_MNIST": [
        TrainParams(
            env_name="MnistEnv",
            agent_name=a,
            num_episodes=1e6,
            gamma=0.9,
            should_debug=False,
            is_test=False,
            should_skip_neptune=False
        ) for a in ["DQN_CNN"]
    ],

    "find_mnist_lr": [
        TrainParams(
            env_name="MnistEnv",
            agent_name="DQN_CNN",
            num_episodes=1e5,
            gamma=0.9,
            should_debug=False,
            is_test=False,
            # should_skip_neptune=True,
            learning_rate=lr
        ) for lr in [0.0001, 0.0005, 0.001, 0.005]
    ],

    "short_compare_on_empty_grid": [
        TrainParams(
            env_name="SimpleGrid",
            agent_name=agent_name,
            num_episodes=10000,
            gamma=0.9,
            should_debug=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],

    "short_compare_on_simple_wall_grid": [
        TrainParams(
            env_name="SimpleWallGrid",
            agent_name=agent_name,
            num_episodes=10000,
            gamma=0.9,
            should_debug=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],

    "short_compare_on_small_empty_grid": [
        TrainParams(
            env_name="SmallEmptyGrid",
            agent_name=agent_name,
            num_episodes=10000,
            gamma=0.9,
            should_debug=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],

    "short_compare_on_tiny_empty_grid": [
        TrainParams(
            env_name="TinyEmptyGrid",
            agent_name=agent_name,
            num_episodes=10000,
            gamma=0.9,
            should_debug=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],

    "short_compare_on_1d_grid": [
        TrainParams(
            env_name="EmptyGrid1D",
            agent_name=agent_name,
            num_episodes=10000,
            gamma=0.9,
            should_debug=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],

    "short_compare_on_bandit": [
        TrainParams(
            env_name="BanditEnv",
            agent_name=agent_name,
            num_episodes=1e3,
            gamma=0.9,
            # should_debug=True
        ) for agent_name in ["DQN"] for i in range(20)  # , "SARSA"]  # SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],
}
