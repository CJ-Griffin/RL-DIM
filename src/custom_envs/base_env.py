import gym
from numpy import array as nparray
from typing import Dict, List
# to fix the dictionary issues

class BaseEnv(gym.Env):

    def start_recording(self):
        pass

    def stop_and_log_recording(self, ep_num: int):
        pass

    def get_recordings(self) -> Dict[str, List[nparray]]:
        return {}
