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
            gamma=0.8,
            epsilon=0.1,
            alpha=0.1,
            mu=float(8 if D == "perf" else 1),
            dist_measure_name=D,
        )
        for D in (['null', 'perf', "rev", "RR"])
    ],

    "NewSushiLong": [
        TrainParams(
            env_name="SushiGrid",
            agent_name="QLearner",
            num_episodes=int(1e6),
            gamma=0.9,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (['perf', 'simple', "rgb"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
    ],

    "rr_comp": [
        TrainParams(
            env_name=env,
            agent_name="QLearner",
            num_episodes=int(1e6),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (["RR"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
        for env in (["MuseumRush", "EasyDoorGrid", 'SmallMuseumGrid', "EmptyDirtyRoom"])
    ],

    "hooman": [
        TrainParams(
            env_name="EasyDoorGrid",
            agent_name="HumanAgent",
            num_episodes=int(1e4),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (["swAU"])  # 'rev', "RR"])
        for mu in [8]  # ([0, 1, 2, 4, 8, 16, 32, 64])
        for env in (['SmallMuseumGrid', "EmptyDirtyRoom"])
    ],

}
