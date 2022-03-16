from argparse import ArgumentParser
from dataclasses import dataclass, asdict
import sys

import agents
import def_params
import custom_envs


@dataclass
class TrainParams:
    env_name: str
    agent_name: str
    num_episodes: int
    gamma: int
    alpha: float = None
    epsilon: float = None
    buffer_size: int = None
    batch_size: int = None
    update_freq: int = None
    q_init: float = None
    learning_rate: float = None
    should_debug: bool = False
    should_render: bool = False
    should_profile: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        # Imperfect but works
        assert(self.env_name in dir(custom_envs)), f"{self.env_name} not in {dir(custom_envs)}"
        assert(self.agent_name in dir(agents)), f"{self.agent_name} not in {dir(agents)}"

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
