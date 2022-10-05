import copy
from typing import Tuple, Optional, Union
import gym
import numpy as np
import emoji
from gym.core import ObsType, ActType
from termcolor import colored
from colorama import Back

from gym.error import DependencyNotInstalled
from src.custom_envs.base_env import BaseEnv


class Repulsion(BaseEnv):
    action_vec_dict = {
        0: np.array([-0.7071, -0.7071], dtype=np.float32),
        1: np.array([-1.0, 0.0], dtype=np.float32),
        2: np.array([-0.7071, 0.7071], dtype=np.float32),
        3: np.array([0.0, -1.0], dtype=np.float32),
        4: np.array([0.0, 0.0], dtype=np.float32),
        5: np.array([0.0, 1.0], dtype=np.float32),
        6: np.array([0.7071, -0.7071], dtype=np.float32),
        7: np.array([1.0, 0.0], dtype=np.float32),
        8: np.array([0.7071, 0.7071], dtype=np.float32)
    }

    def __init__(self, num_obstacles=10):
        self.state_shape = (2, num_obstacles + 1)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=self.state_shape,
                                                dtype=np.float32)

        self.state = np.zeros(self.state_shape, dtype=np.float32)
        self.init_state = None

        self.state_width, self.state_height = (self.observation_space.high - self.observation_space.low)[:, 0]

        self.goal_loc = np.array((0.8, 0.5)).reshape((2, 1))
        self.goal_radius = 0.07

        self.player_loc = np.array((0.05, 0.5)).reshape((2, 1))
        self.player_move_speed = 0.05

        self.repulsion_radius = 0.3
        self.repulsion_constant = 0.005
        self.repulsion_max = 0.1

        self.object_spawn_y_min = 0.35
        self.object_spawn_y_max = 0.95

        self.screen = None
        self.clock = None
        self.isopen = False

        self.max_steps = 50
        self.cur_steps = 0

        # TODO - make gamma more central to the environment rather than the agent
        self.gamma = Exception("This should have been changed")

    def init_gamma(self, gamma: float):
        self.gamma = gamma

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        prev_state = self.state.copy()
        action_vec = self.action_vec_dict[action] * self.player_move_speed
        next_state = self.state
        next_state[:, 0] += action_vec

        #### repulsion ####

        obstacles = next_state[:, 1:]
        player = next_state[:, 0].reshape(2, 1)
        diffs = obstacles - player
        dist = np.linalg.norm(diffs, axis=0)

        norm_diffs = diffs / dist
        should_ignore = (dist > self.repulsion_radius)

        magnitudes = self.repulsion_constant / dist
        magnitudes[should_ignore] = 0

        should_cap = (magnitudes > self.repulsion_max)
        magnitudes[should_cap] = self.repulsion_max

        displacement = norm_diffs * magnitudes

        next_state[:, 1:] += displacement

        #### repulsion ####

        next_state[next_state > 1.0] = 1.0
        next_state[next_state < 0.0] = 0.0

        self.state = next_state
        is_win = (np.linalg.norm(player - self.goal_loc) < self.goal_radius)

        # use reward shaping!
        # rewards moving closer to the goal (0,0)
        spec_reward = np.linalg.norm(prev_state[:, 0] - self.goal_loc) - \
                      (self.gamma * np.linalg.norm(next_state[:, 0] - self.goal_loc))
        dist_reward = self._get_dist_term(prev_state, next_state)

        done = is_win or (self.cur_steps >= 50)
        self.cur_steps += 1

        return next_state, spec_reward - dist_reward, done, {
            "dist_reward": dist_reward,
            "spec_reward": spec_reward
        }

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, tuple[ObsType, dict]]:
        self.state = np.zeros(self.state_shape)
        self.state[1, :] = np.random.uniform(low=0.1, high=0.9, size=self.state_shape[1]).astype(np.float32)
        self.state[0, :] = np.random.uniform(low=self.object_spawn_y_min, high=self.object_spawn_y_max,
                                             size=self.state_shape[1]).astype(np.float32)
        self.state[:, 0] = self.player_loc[:, 0]
        self.init_state = self.state.copy()
        return self.state

    def render(self, mode="human"):
        if mode != "human":
            return None
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        ratio = 200
        screen_width = ratio
        screen_height = ratio

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        def state_coord_to_screen_coord(state_coord: np.array) -> np.array:
            return np.array(state_coord * ratio, dtype=int)

        # ############## ############### DRAW STUFF ############### ############## #

        screen_coord = state_coord_to_screen_coord(x)

        player_screen_coord = screen_coord[:, 0]
        goal_screen_coord = state_coord_to_screen_coord(self.goal_loc)
        obs_screen_coord = screen_coord[:, 1:]

        gfxdraw.filled_circle(
            self.surf,
            int(goal_screen_coord[1]),
            int(goal_screen_coord[0]),
            int(self.goal_radius * ratio),
            (150, 255, 150),
        )

        gfxdraw.filled_circle(
            self.surf,
            int(player_screen_coord[1]),
            int(player_screen_coord[0]),
            int(ratio * 0.01),
            (0, 0, 0),
        )

        num_obs = obs_screen_coord.shape[1]
        for obs_ind in range(num_obs):
            colour = (
                int(255 * (obs_ind) / (num_obs - 1)),
                int(255 * 2 * (np.abs(((num_obs - 1.0) / 2) - obs_ind)) / (num_obs - 1)),
                int(255 * ((num_obs - 1) - obs_ind) / (num_obs - 1)),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(obs_screen_coord[1, obs_ind]),
                int(obs_screen_coord[0, obs_ind]),
                int(ratio * 0.01),
                colour,
            )

        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        # l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        # axleoffset = cartheight / 4.0
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # carty = 100  # TOP OF CART
        # cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        # cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        # gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        # gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        #
        # l, r, t, b = (
        #     -polewidth / 2,
        #     polewidth / 2,
        #     polelen - polewidth / 2,
        #     -polewidth / 2,
        # )
        #
        # pole_coords = []
        # for coord in [(l, b), (l, t), (r, t), (r, b)]:
        #     coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
        #     coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
        #     pole_coords.append(coord)
        # gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        # gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
        #
        # gfxdraw.aacircle(
        #     self.surf,
        #     int(cartx),
        #     int(carty + axleoffset),
        #     int(polewidth / 2),
        #     (129, 132, 203),
        # )
        # gfxdraw.filled_circle(
        #     self.surf,
        #     int(cartx),
        #     int(carty + axleoffset),
        #     int(polewidth / 2),
        #     (129, 132, 203),
        # )
        #
        # gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))
        # gfxdraw.filled_circle(
        #     self.surf,
        #     int(cartx),
        #     int(carty + axleoffset),
        #     int(polewidth / 2),
        #     (129, 132, 203),
        # )
        # ############## ############### DONE ############### ############## #

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _get_dist_term(self, s_t: np.ndarray, s_tp1: np.ndarray) -> float:
        # print(s_tp1[:, 1:] - self.init_state[:, 1:])
        d_tp1 = np.linalg.norm(s_tp1[:, 1:] - self.init_state[:, 1:])
        d_t = np.linalg.norm(s_t[:, 1:] - self.init_state[:, 1:])
        print(d_tp1, d_t)
        return (self.gamma * d_tp1) - d_t
