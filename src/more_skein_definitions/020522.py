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
            should_render=True, #modified
            should_skip_neptune=True #modified
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
            num_episodes=int(1e4),
            gamma=0.9,
            epsilon=0.1,
            alpha=0.1,
            mu=float(8),
            dist_measure_name=D,
        )
        for D in (['null', 'perf', "rev"]) #, "RR"])
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

    "rr_comp1": [
        TrainParams(
            env_name=env,
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (["RR"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
        for env in (["MuseumRush"])
    ],

    "rr_comp2": [
        TrainParams(
            env_name=env,
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (["RR"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
        for env in (["EasyDoorGrid"])
    ],

    "rr_comp3": [
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
        for env in (['SmallMuseumGrid'])
    ],

    "rr_comp4": [
        TrainParams(
            env_name=env,
            agent_name="QLearner",
            num_episodes=int(1e5),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (["RR"])
        for mu in ([0, 1, 2, 4, 8, 16, 32, 64])
        for env in (["EmptyDirtyRoom"])
    ],

    "hooman": [
        TrainParams(
            env_name="SushiGrid",
            agent_name="HumanAgent",
            num_episodes=int(1e4),
            gamma=1.0,
            epsilon=0.1,
            alpha=0.1,
            mu=float(mu),
            dist_measure_name=D,
        )
        for D in (["rev"])  # 'rev', "RR"])
        for mu in [8]  # ([0, 1, 2, 4, 8, 16, 32, 64])
        for env in (['SmallMuseumGrid', "EmptyDirtyRoom"])
    ],

}
