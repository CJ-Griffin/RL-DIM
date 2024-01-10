from src.custom_envs import BaseEnv

class CounterfactualEnvWrapper(BaseEnv):
    def __init__(self, EnvClass):
        #print("this works")
        self.actual_env = EnvClass()
        self.counterfactual_env = EnvClass()
        self.action_space = self.actual_env.action_space
        self.observation_space = self.actual_env.observation_space

    def step(self, action):
        actual_state, actual_reward, actual_done, actual_info = self.actual_env.step(action)
        return actual_state, actual_reward, actual_done, actual_info

    def init_dist_measure(self, *args, **kwargs):
        # TODO get rid of
        self.actual_env.init_dist_measure(*args, **kwargs)
        self.counterfactual_env.init_dist_measure(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.actual_env.render(*args, **kwargs)

    def reset(self, *args, **kwargs):
        actual_s0 = self.actual_env.reset(*args, **kwargs)
        counterfactual_s0 = self.counterfactual_env.reset(*args, **kwargs)
        return actual_s0