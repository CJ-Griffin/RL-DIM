import gym
from numpy import array as nparray


class BaseEnv(gym.Env):

    def start_recording(self):
        pass

    def stop_and_log_recording(self, ep_num: int):
        pass

    def get_recordings(self) -> dict[list[nparray]]:
        return {}
