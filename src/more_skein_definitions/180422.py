from src.run_parameters import TrainParams

SKEIN_DICT = {
    "MuseumRush": [
        TrainParams(
            env_name="MuseumRush",
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=1.0,
            epsilon=0.05,
            mu=mu,
            dist_measure_name=D,
        )
        for D in ['perf', 'simple', "rgb"]
        for mu in [0.0, 0.2, 0.4, 0.6, 0.8,
                   1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    ],

    "EmptyDirtyRoom": [
        TrainParams(
            env_name="EmptyDirtyRoom",
            agent_name="QLearner",
            num_episodes=int(2e5),
            gamma=0.9,
            epsilon=0.05,
            mu=mu,
            dist_measure_name=D,
        )
        for D in ['perf', 'simple', "rgb"]
        for mu in [0.0, 0.2, 0.4, 0.6, 0.8,
                   1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    ],

    "EasyDoorGrid": [
        TrainParams(
            env_name="EasyDoorGrid",
            agent_name="QLearner",
            num_episodes=int(2e5),
            gamma=0.9,
            epsilon=0.05,
            mu=mu,
            dist_measure_name=D,
        )
        for D in ['perf', 'simple', "rgb"]
        for mu in [0.0, 0.2, 0.4, 0.6, 0.8,
                   1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    ],

    "SmallMuseumGrid": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(1e6),
            gamma=0.9,
            epsilon=0.05,
            mu=mu,
            dist_measure_name=D,
        )
        for D in ['perf', 'simple', "rgb"]
        for mu in [0.0, 0.2, 0.4, 0.6, 0.8,
                   1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    ],

}
