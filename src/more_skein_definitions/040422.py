from src.run_parameters import TrainParams

SKEIN_DICT = {
    #
    # "epsilon_tweaking": [
    #     TrainParams(
    #         env_name="MuseumGrid",
    #         agent_name="DQN_CNN",
    #         num_episodes=int(2e5),
    #         gamma=0.9,
    #         epsilon=e
    #     ) for e in [0.01, 0.05, 0.1]
    # ],
    #
    # "gamma_tweaking": [
    #     TrainParams(
    #         env_name="MuseumGrid",
    #         agent_name="DQN_CNN",
    #         num_episodes=int(2e5),
    #         gamma=0.9,
    #         epsilon=0.05
    #     ) for e in [0.75, 0.9, 0.95]
    # ],
    #
    # "replay_param_tweaking": [
    #     TrainParams(
    #         env_name="MuseumGrid",
    #         agent_name="DQN_CNN",
    #         num_episodes=int(5e4),
    #         gamma=0.9,
    #         update_freq=uf,
    #         batch_size=bas,
    #         buffer_size=bus,
    #     )
    #     for uf in [4, 16, 64]
    #     for bas in [4, 16, 64]
    #     for bus in [1000, 10000]
    # ],
}
