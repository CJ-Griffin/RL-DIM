from src.run_parameters import TrainParams

SKEIN_DICT = {

    "SmallMuseumGridTest": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=0.9,
            epsilon=0.05,
            dist_measure_name="simple",
            is_test=True,
        )
    ],

    "HumanSmallMuseumGridTest": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="HumanAgent",
            num_episodes=int(1e5),
            gamma=0.9,
            epsilon=0.05,
            dist_measure_name="rgb",
            is_test=True
        )
    ],

    "Comparing_Algorithms": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name=an,
            num_episodes=int(1e6),
            gamma=0.9,
            epsilon=0.05,
            dist_measure_name="simple"
        ) for an in ["QLearner", "SARSA", "TabularMC"]
    ],

    "Mu_Finding": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(1e6),
            gamma=0.9,
            epsilon=0.05,
            dist_measure_name="vase_door",
            mu=mu
        ) for mu in [0.0, 0.25, 0.5, 1.0, 2.0, 0.0, 0.25, 0.5, 1.0, 2.0, 0.0, 0.25, 0.5, 1.0, 2.0,
                     0.0, 0.25, 0.5, 1.0, 2.0, 0.0, 0.25, 0.5, 1.0, 2.0]
    ],

    "Mu_Finding_RGB": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=0.9,
            epsilon=0.05,
            dist_measure_name="rgb",
            mu=mu
        ) for mu in [0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0,
                     2.0, ]
    ],

    "Mu_Finding_Simple": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=0.9,
            epsilon=0.05,
            dist_measure_name="simple",
            mu=mu
        ) for mu in [0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0,
                     2.0, ]
    ],
}
