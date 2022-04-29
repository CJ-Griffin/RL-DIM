from src.run_parameters import TrainParams

SKEIN_DICT = {
    "template": [
        TrainParams(
            env_name="SmallMuseumGrid",
            agent_name="DQN_CNN",
            num_episodes=1e5,
            gamma=0.9
        )
    ],
}
