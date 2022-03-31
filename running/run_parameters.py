from argparse import ArgumentParser
from dataclasses import dataclass, asdict
import sys

import agents
import custom_envs


@dataclass
class TrainParams:
    env_name: str
    agent_name: str
    num_episodes: int
    gamma: float
    alpha: float = 0.05
    epsilon: float = 0.05
    buffer_size: int = int(1e4)
    batch_size: int = 100
    update_freq: int = 10
    q_init: float = 0.0
    learning_rate: float = int(1e-3)
    should_debug: bool = False
    should_render: bool = False
    should_profile: bool = False
    is_test: bool = False
    should_skip_neptune: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        # Imperfect but works
        assert (self.env_name in dir(custom_envs)), f"{self.env_name} not in {dir(custom_envs)}"
        # assert(self.agent_name in dir(agents)), f"{self.agent_name} not in {dir(agents)}"

    def __str__(self):
        return str(self.get_dict())

    def get_dict(self):
        return asdict(self)


# class TrainParamsFromCLI(TrainParams):
#     def __init__(self):
#         super().__init__(None)
#     def __post_init__(self):
#
#         parser = ArgumentParser(description="Run RL experiments")
#         print(self.__dict__)
#         for attr_name in dict(self).keys():
#             parser.add_argument(attr_name, default=None)
#
#         args = parser.parse_args()
#         for attr_name in dict(self).keys():
#             setattr(self, attr_name, getattr(args, attr_name))
#
#         super().__post_init__()
#
#         # args = sys.argv[1:]
#         # arg_str = str(args)
#         # while len(args) > 0:
#         #     arg_name = args.pop()
#         #     arg_val = args.pop()
#         #     if not hasattr(self, "arg_name"):
#         #         raise Exception(f"{self} has no attribute {arg_name}. Args: {arg_str}")
#         #     else:
#         #         try:
#         #             setattr(self, arg_name, arg_val)
#         #         except TypeError as te:
#         #             raise TypeError(te, f"{self}'s attribute' {arg_name} does not accept type of {arg_val}. Args: {arg_str}")
#         #         except Exception as e:
#         #             raise e


if __name__ == "__main__":
    ps = TrainParams("BanditEnv", "RandomAgent", 100, 1.0)
    print(ps)

    # ps = TrainParamsFromCLI()
    # print(ps)
