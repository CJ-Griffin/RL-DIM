from typing import Tuple, Optional

import gym
import numpy as np

from gym.core import ObsType, ActType

CHAR_TO_PIXEL = {
    '.': (0, 0, 0),
    'R': (1, 0, 0),
    'G': (0, 1, 0),
    '#': (0, 0, 1),
    'C': (1, 1, 0)
}


# '': np.array([0,0,0]),
# 'R': np.array([1,0,0]),
# '#': np.array([0,0,1]),
# 'G': np.array([0,1,0]),


def char_to_pixel(char):
    return CHAR_TO_PIXEL[char]


vchar_to_pixel = np.vectorize(char_to_pixel)


class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# TODO - make random seed system


class Grid(gym.Env):
    def __init__(self, height, width, player_init,
                 goal_loc=None, max_steps=100, coin_value=1.0):
        if goal_loc is None:
            goal_loc = (height - 1, width - 1)
        self.goal_loc = goal_loc
        self.max_steps = max_steps
        self.coin_value = coin_value

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=[3, height, width],
            dtype=np.int
        )
        self.player_init = player_init
        self.height = height
        self.width = width

        self.rob = Robot(*player_init)
        self.grid = self.get_init_grid()
        self._last_reward = None

    def get_new_coin_locations(self):
        raise NotImplementedError()

    def get_new_wall_locations(self):
        raise NotImplementedError()

    def get_init_grid(self):
        arr = np.empty((self.height, self.width), dtype=np.unicode_)
        arr[:, :] = '.'

        wall_locations = self.get_new_wall_locations()
        coin_locations = self.get_new_coin_locations()

        # Define conflicts where two objects exist
        conflicts = set(wall_locations) & set(coin_locations)
        if conflicts:  # If the set of conflicts is non empty we have a problem
            raise Exception(conflicts)

        for wall_loc in wall_locations:
            arr[wall_loc] = '#'

        for coin_loc in coin_locations:
            arr[coin_loc] = 'C'

        arr[self.goal_loc] = 'G'
        return arr

    def np_grid_to_string(self, grid):
        row_strings = [" ".join(list(row)) for row in grid]
        return "\n".join(row_strings)

    def render(self, mode="human"):
        grid = self.get_grid_with_rob()
        print("\n--- GridEnv ---")
        print(self.np_grid_to_string(grid))
        print(f"Last reward: {self._last_reward}")
        print("---  ======== ---")
        print()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsType:
        self.rob.y, self.rob.x = self.player_init
        self.grid = self.get_init_grid()
        self.elapsed_steps = 0
        return self.get_obs()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        dest = self.get_dest(action)
        self.elapsed_steps += 1

        if dest is not None:
            gets_coin = (self.grid[dest] == 'C')
            has_won = (self.grid[dest] == 'G')
            self.rob.y = dest[0]
            self.rob.x = dest[1]
            if gets_coin:
                self.grid[dest] = '.'
        else:
            gets_coin = False
            has_won = False

        obs = self.get_obs()
        reward = (int(gets_coin) * self.coin_value) + \
                 (-0.5 * (dest is None)) + \
                 (10 * has_won)
        self._last_reward = reward
        done = (self.elapsed_steps >= self.max_steps) or has_won
        info = {}
        # print(reward)
        return obs, reward, done, info

    def get_grid_with_rob(self):
        grid = self.grid.copy()
        grid[self.rob.y, self.rob.x] = 'R'
        return grid

    def get_obs(self):
        grid = self.get_grid_with_rob()
        grid2 = vchar_to_pixel(grid)
        grid3 = np.stack(grid2)
        return grid3

    def get_dest(self, action):
        displacements = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1])
        }
        (y, x) = np.array([self.rob.y, self.rob.x]) + displacements[action]
        if self.is_loc_in_bounds(y, x) and self.grid[y, x] in ['.', 'C', 'G']:
            return (y, x)
        else:
            return None

    def is_loc_in_bounds(self, y: int, x: int) -> bool:
        return (0 <= x < self.width) and (0 <= y < self.height)


class EmptyGrid(Grid):
    def get_new_coin_locations(self):
        return []

    def get_new_wall_locations(self):
        return []

    def __init__(self, height=10, width=16, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)


class EmptyGrid1D(EmptyGrid):
    def __init__(self):
        super().__init__(height=1, width=5, player_init=(0, 0), goal_loc=None)


class SmallEmptyGrid(EmptyGrid):
    def __init__(self, height=5, width=8, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)


class CoinGrid(Grid):
    def __init__(self, height=5, width=8, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)

    def get_new_wall_locations(self):
        return []

    def get_new_coin_locations(self):
        h = self.height
        w = self.width
        locs = [
            (h - 1, 0),
            (0, w - 1),
            (int(h / 2), int(w / 2)),
            (int(h - 1), int(w / 2)),
            (int(h / 2), int(w - 1))
        ]
        return locs


class RandCoinGrid(CoinGrid):

    def get_new_coin_locations(self):
        h = self.height
        w = self.width
        ys = np.random.randint(low=0, high=h, size=5)
        xs = np.random.randint(low=0, high=w, size=5)

        locs = list(zip(ys, xs))
        return locs


class WallGrid(Grid):
    def __init__(self, height=10, width=16, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)

    def get_new_coin_locations(self):
        return []

    def get_new_wall_locations(self):
        h = self.height
        w = self.width
        wall_xs = range(1, w - 2, 2)
        gap_locs = [h - 1 if j % 2 == 0 else 0 for j in range(len(wall_xs))]
        locs = []
        for i, x in enumerate(wall_xs):
            # Fill all spaces except for the gap
            gap = gap_locs[i]
            locs += [(y, x) for y in range(h) if y != gap]
        return locs


class SemiRandWallGrid(WallGrid):

    def get_new_wall_locations(self):
        h = self.height
        w = self.width
        wall_xs = range(1, w - 2, 2)
        gap_locs = np.random.randint(low=0, high=h, size=len(wall_xs))
        locs = []
        for i, x in enumerate(wall_xs):
            # Fill all spaces except for the gap
            gap = gap_locs[i]
            locs += [(y, x) for y in range(h) if y != gap]
        return locs


class SemiRandCoinWallGrid(SemiRandWallGrid):

    def get_new_coin_locations(self):
        h = self.height
        w = self.width
        xs = range(0, w, 2)
        num_coins = len(xs)
        ys = np.random.randint(low=1, high=h - 1, size=num_coins)
        locs = list(zip(ys, xs))
        return locs
