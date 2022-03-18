from running.run_parameters import TrainParams

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
    "CoinGrid",
    "RandCoinGrid",
    "WallGrid",
    "SemiRandWallGrid",
    "SemiRandCoinWallGrid"
]

skein_dict = {

    "default": [
        TrainParams(
            env_name="WallGrid",
            agent_name="SARSA",
            num_episodes=100,
            gamma=1.0
        )
    ],

    "test_all_envs": [
        TrainParams(
            env_name=env_name,
            agent_name="TabularMC",
            num_episodes=1000,
            gamma=1.0,
            should_debug=False
        ) for env_name in SOME_ENV_NAMES
    ],

    "test_all_agents": [
        TrainParams(
            env_name="SmallEmptyGrid",
            agent_name=agent_name,
            num_episodes=1000,
            gamma=1.0,
            should_debug=False
        ) for agent_name in SOME_AGENT_NAMES if agent_name != "HumanAgent"
    ],


    "test_DQN_CNN": [
        TrainParams(
            env_name="SmallEmptyGrid",
            agent_name="DQN_CNN",
            num_episodes=1000,
            gamma=1.0,
            should_debug=False
        )
    ],
}
