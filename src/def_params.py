from src.run_parameters import TrainParams

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
    "EmptyGrid",
    "DirtGrid",
    "RandDirtGrid",
    "WallGrid",
    "SemiRandWallGrid",
    # "SemiRandDirtWallGrid"
]

skein_dict = {

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
            should_skip_neptune=True
        ) for an in ["DQN"]
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
            env_name="SmallRandDirtGrid",
            agent_name="DQN_CNN",
            num_episodes=int(1e5),
            gamma=0.9,
            epsilon=epsilon
        )
        for epsilon in [0.01, 0.05, 0.1]
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
            env_name="EmptyGrid",
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
            num_episodes=1000,
            gamma=0.9,
            should_debug=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],
}
