from src.run_parameters import TrainParams

SKEIN_DICT = {
    "MuseumRush": [
        TrainParams(
            env_name="EasyDoorGrid",
            agent_name="QLearner",
            num_episodes=int(1e4),
            gamma=1.0,
            mu=float(mu),
            dist_measure_name=D,
            is_test=True
        )
        for D in (['perf', 'simple', "rgb", "rev"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "EasyDoorGrid": [
        TrainParams(
            env_name="EasyDoorGrid",
            agent_name="QLearner",
            num_episodes=int(1e4),
            gamma=1.0,
            mu=float(mu),
            is_test=True,
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb", "rev"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "EmptyDirtyRoom": [
        TrainParams(
            env_name="EmptyDirtyRoom",
            agent_name="QLearner",
            num_episodes=int(1e4),
            gamma=0.9,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb", "rev"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "SmallMuseumGrid": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="QLearner",
            num_episodes=int(1e6),
            gamma=0.9,
            epsilon=0.05,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb", "rev"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "SushiGrid": [
        TrainParams(
            env_name="SushiGrid",
            agent_name="QLearner",
            num_episodes=int(1e4),
            gamma=0.9,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb", "rev"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "RandomMuseumRoom": [
        TrainParams(
            env_name="RandomMuseumRoom",
            agent_name="DQN",
            num_episodes=int(1e6),
            gamma=0.9,
            epsilon=0.1,
            mu=float(mu),
            is_test=True,
            dist_measure_name=D,
        )
        for D in ['perf']
        for mu in ([8, 0, 64])
    ],

}
