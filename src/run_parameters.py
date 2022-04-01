from dataclasses import dataclass, asdict
from src import custom_envs


@dataclass
class TrainParams:
    env_name: str
    agent_name: str
    num_episodes: int
    gamma: float
    alpha: float = 0.05
    epsilon: float = 0.05
    buffer_size: int = int(5e4)
    batch_size: int = 32
    update_freq: int = 100
    q_init: float = 0.0
    learning_rate: float = 1e-3
    should_debug: bool = False
    should_render: bool = False
    should_profile: bool = False
    is_test: bool = False
    should_skip_neptune: bool = False

    def __post_init__(self):
        self.num_episodes = force_integer(self.num_episodes)
        self.validate()

    def validate(self):
        # pass
        # Imperfect but works
        # import src.custom_envs
        # envs = custom_envs
        assert (self.env_name in dir(custom_envs)), f"{self.env_name} not in {dir(custom_envs)}"
        # assert(self.agent_name in dir(agents)), f"{self.agent_name} not in {dir(agents)}"

    def __str__(self):
        return str(self.get_dict())

    def get_dict(self):
        return asdict(self)


def force_integer(n):
    if isinstance(n, int):
        return n
    elif isinstance(n, float) and n.is_integer():
        return int(n)
    else:
        raise TypeError(n, "is not an integer")


if __name__ == "__main__":
    ps = TrainParams("BanditEnv", "RandomAgent", 100, 1.0)
    print(ps)

    # ps = TrainParamsFromCLI()
    # print(ps)
