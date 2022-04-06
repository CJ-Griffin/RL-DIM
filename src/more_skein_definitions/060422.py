from src.run_parameters import TrainParams

SKEIN_DICT = {

    "HumanEasyMuseumGridTest": [
        TrainParams(
            env_name="EasyMuseumGrid",
            agent_name="HumanAgent",
            num_episodes=int(1e3),
            gamma=1.0,
            epsilon=0.05,
            dist_measure_name="rgb_dist",
            is_test=True
        )
    ],

    "Mu_Finding": [
        TrainParams(
            env_name="EasyMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(5e5),
            gamma=1.0,
            epsilon=0.05,
            dist_measure_name="vase_door",
            mu=mu
        ) for mu in [0.0, 0.25, 0.5, 1.0, 2.0, 0.0, 0.25, 0.5, 1.0, 2.0, 0.0, 0.25, 0.5, 1.0, 2.0,
                     0.0, 0.25, 0.5, 1.0, 2.0, 0.0, 0.25, 0.5, 1.0, 2.0]
    ],

    "Mu_Finding_RGB": [
        TrainParams(
            env_name="EasyMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(5e5),
            gamma=1.0,
            epsilon=0.05,
            dist_measure_name="rgb",
            mu=mu
        ) for mu in [0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0,
                     2.0, ]
    ],

    "Mu_Finding_Simple": [
        TrainParams(
            env_name="EasyMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(5e5),
            gamma=1.0,
            epsilon=0.05,
            dist_measure_name="simple",
            mu=mu
        ) for mu in [0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0,
                     2.0, ]
    ],
}
