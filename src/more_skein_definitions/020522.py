from src.run_parameters import TrainParams

SKEIN_DICT = {

    "NewEasyDoorGridShort": [
        TrainParams(
            env_name="EasyDoorGrid",
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "NewEasyDoorGridLong": [
        TrainParams(
            env_name="EasyDoorGrid",
            agent_name="QLearner",
            num_episodes=int(1e6),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "NewSushiShort": [
        TrainParams(
            env_name="SushiGrid",
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "NewSushiLong": [
        TrainParams(
            env_name="SushiGrid",
            agent_name="QLearner",
            num_episodes=int(1e6),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

}
