from src.run_parameters import TrainParams

SKEIN_DICT = {
    # "MuseumRush": [
    #     TrainParams(
    #         env_name="MuseumRush",
    #         agent_name="QLearner",
    #         num_episodes=int(1e6),
    #         gamma=0.9,
    #         epsilon=0.05,
    #         mu=mu,
    #         dist_measure_name=D,
    #     )
    #     for D in ['vase_door', 'simple', "rgb"]
    #     for mu in [0.0, 0.2, 0.4, 0.6, 0.8,
    #                1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # ],
    #
    # "Experiment5": [
    #     TrainParams(
    #         env_name="MuseumRush",
    #         agent_name="QLearner",
    #         num_episodes=int(1e5),
    #         gamma=0.9,
    #         epsilon=0.05,
    #         mu=mu,
    #         dist_measure_name=D,
    #     ) for (D, mu) in
    #     [
    #         ('null', 0.0),
    #         ('vase_door', 0.2), ('vase_door', 0.4), ('vase_door', 0.6),
    #         ('vase_door', 0.8),
    #         ('vase_door', 1.0), ('vase_door', 1.2), ('vase_door', 1.4), ('vase_door', 1.6),
    #         ('vase_door', 1.8), ('vase_door', 2.0),
    #         ('simple', 0.2), ('simple', 0.4), ('simple', 0.6), ('simple', 0.8), ('simple', 1.0),
    #         ('simple', 1.2), ('simple', 1.4), ('simple', 1.6), ('simple', 1.8),
    #         ('simple', 2.0),
    #         ('rgb', 0.2), ('rgb', 0.4), ('rgb', 0.6), ('rgb', 0.8), ('rgb', 1.0),
    #         ('rgb', 1.2), ('rgb', 1.4), ('rgb', 1.6), ('rgb', 1.8), ('rgb', 2.0)
    #     ]
    # ],
    #
    # "Experiment4": [
    #     TrainParams(
    #         env_name="EasyDoorGrid",
    #         agent_name="QLearner",
    #         num_episodes=int(1e5),
    #         gamma=0.9,
    #         epsilon=0.05,
    #         mu=mu,
    #         dist_measure_name=D,
    #     ) for (D, mu) in
    #     [
    #         # ('null', 0.0), ('vase_door', 0.2), ('vase_door', 0.4),
    #         ('vase_door', 0.6),
    #         # ('vase_door', 0.8),
    #         # ('vase_door', 1.0), ('vase_door', 1.2), ('vase_door', 1.4),
    #         ('vase_door', 1.6),
    #         # ('vase_door', 1.8), ('vase_door', 2.0), ('simple', 0.2), ('simple', 0.4),
    #         # ('simple', 0.6), ('simple', 0.8), ('simple', 1.0), ('simple', 1.2), ('simple', 1.4), ('simple', 1.6),
    #         # ('simple', 1.8), ('simple', 2.0), ('rgb', 0.2), ('rgb', 0.4), ('rgb', 0.6), ('rgb', 0.8), ('rgb', 1.0),
    #         # ('rgb', 1.2), ('rgb', 1.4), ('rgb', 1.6), ('rgb', 1.8), ('rgb', 2.0)
    #     ]
    # ],
    #
    # "HumanRush": [
    #     TrainParams(
    #         env_name="MuseumRush",
    #         agent_name="HumanAgent",
    #         num_episodes=int(1e2),
    #         gamma=0.9,
    #         epsilon=0.05,
    #         mu=mu,
    #         dist_measure_name=D,
    #         is_test=True,
    #     ) for (D, mu) in
    #     [
    #         ('simple', 2.0)
    #     ]
    # ],
    #
    # "HumanEmptyDirty": [
    #     TrainParams(
    #         env_name="EmptyDirtyRoom",
    #         agent_name="HumanAgent",
    #         num_episodes=int(1e2),
    #         gamma=0.9,
    #         epsilon=0.05,
    #         mu=mu,
    #         dist_measure_name=D,
    #         is_test=True,
    #     ) for (D, mu) in
    #     [
    #         ('simple', 2.0)
    #     ]
    # ]
}
