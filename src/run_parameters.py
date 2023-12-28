from dataclasses import dataclass, asdict
from datetime import datetime

from src import custom_envs, agents


@dataclass
class TrainParams:
    # Non-default arguments
    env_name: str  # MDP
    gamma: float  # MDP
    agent_name: str  # Agent
    num_episodes: int  # Agent

    # Default arguments
    dist_measure_name: str = "null"  # MDP
    mu: float = 1.0  # MDP
    alpha: float = 0.05  # Agent
    epsilon: float = 0.05  # Agent
    buffer_size: int = int(5e4)  # Agent
    batch_size: int = 32  # Agent
    update_freq: int = 100  # Agent
    q_init: float = 0.0  # Agent
    learning_rate: float = 1e-3  # Agent
    should_debug: bool = False  # Experiment management
    should_render: bool = False  # Experiment management
    should_profile: bool = False  # Experiment management
    is_test: bool = False  # Experiment management (will not do tqdm if not test)
    should_skip_neptune: bool = False  # Experiment management

    def __post_init__(self):
        self.num_episodes = force_integer(self.num_episodes)
        self.validate()

    def validate(self):
        # pass
        # Imperfect but works
        # import src.custom_envs
        # envs = custom_envs
        assert (self.env_name in dir(custom_envs)), f"{self.env_name} not in {dir(custom_envs)}"
        if self.agent_name[:5] == "save.":
            rest = self.agent_name[5:]
            assert "_" in rest, self.agent_name
            run_name, ep_no = rest.split("_")
            assert len(run_name) > 0 and len(ep_no) > 0, (run_name, ep_no, rest)
        else:
            assert(self.agent_name in dir(agents)), f"{self.agent_name} not in {dir(agents)}"

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
