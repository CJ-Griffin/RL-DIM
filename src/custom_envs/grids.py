from typing import Tuple, Optional
import gym
import numpy as np
import emoji
from gym.core import ObsType, ActType

from src.custom_envs.base_env import BaseEnv
# from src.utils import are_sets_independent

CHAR_TO_PIXEL = {
    ' ': (0, 0, 0),  # Empty space
    'R': (1, 1, 1),  # Robot
    'G': (0, 1, 0),  # Goal
    '#': (0, 0, 1),  # Wall
    '.': (1, 0, 0),  # Dirt
    '|': (0, 0, 1),  # Closed door
    '/': (0, 1, 1),  # Open door
    'V': (1, 0, 1)  # Vase
}

CHAR_TO_EMOJI = {
    ' ': " ",  # Empty space
    'R': ":robot:",  # Robot
    'G': ":star:",  # Goal
    '#': ":white_large_square:",  # Wall
    '.': ":brown_circle:",  # Dirt
    '|': ":door:",  # Closed door
    '/': ":window:",  # Open door
    'V': ":amphora:"  # Vase
}


# '': np.array([0,0,0]),
# 'R': np.array([1,0,0]),
# '#': np.array([0,0,1]),
# 'G': np.array([0,1,0]),


def char_to_pixel(char):
    return CHAR_TO_PIXEL[char]


def char_to_emoji(char):
    return CHAR_TO_EMOJI[char]


vchar_to_pixel = np.vectorize(char_to_pixel)


class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# TODO - make random seed system


class Grid(BaseEnv):
    def __init__(self, height, width, player_init,
                 goal_loc=None, max_steps=100, dirt_value=1.0):
        if goal_loc is None:
            goal_loc = (height - 1, width - 1)
        self.goal_loc = goal_loc
        self.max_steps = max_steps
        self.dirt_value = dirt_value
        self.elapsed_steps = 0

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=[3, height, width],
            dtype=np.int
        )
        self.player_init = player_init
        self.height = height
        self.width = width

        self._is_recording = False
        self._im_history: list[np.array] = []
        self._recordings = {}

        self.rob = Robot(*player_init)
        self.grid = self.get_init_grid()
        self.rob_loc_history = [(self.rob.y, self.rob.x)]
        self.grid_s0_history = self.grid.copy()
        self._last_reward = None
        self.passable_objects = [' ', '.', 'G', '/', 'V']

    # WDDV = Wall, Door, Dirt, Vase
    def get_object_locations_WDDV(self) -> (list, list, list, list):
        walls, doors = self.get_new_wall_and_door_locations()
        return walls, doors, self.get_new_dirt_locations(), self.get_vase_locations()

    def get_new_dirt_locations(self):
        raise NotImplementedError()

    def get_new_wall_and_door_locations(self) -> (list, list):
        raise NotImplementedError()

    def get_vase_locations(self) -> list:
        raise NotImplementedError()

    def get_init_grid(self):
        arr = np.empty((self.height, self.width), dtype=np.unicode_)
        arr[:, :] = ' '

        wall_locations, door_locations, dirt_locations, vase_locations = self.get_object_locations_WDDV()

        # Define conflicts where two objects exist
        object_sets = [
            frozenset(wall_locations),
            frozenset(dirt_locations),
            frozenset(vase_locations),
            frozenset({self.player_init}),
            frozenset({self.goal_loc})
        ]
        from src.generic_utils import are_sets_independent
        assert are_sets_independent(set(object_sets)), object_sets

        for wall_loc in wall_locations:
            arr[wall_loc] = '#'

        for wall_loc in door_locations:
            arr[wall_loc] = '|'

        for dirt_loc in dirt_locations:
            arr[dirt_loc] = '.'

        for vase_loc in vase_locations:
            arr[vase_loc] = 'V'

        arr[self.goal_loc] = 'G'
        return arr

    def np_grid_to_string(self, grid, should_emojify=False):
        if should_emojify:
            row_strings = [" ".join(list(map(char_to_emoji, row))) for row in grid]
        else:
            row_strings = [" ".join(list(row)) for row in grid]
        return emoji.emojize("\n".join(row_strings))

    def render(self, mode="human"):
        grid = self.get_grid_with_rob()
        print("\n--- GridEnv ---")
        print(self.np_grid_to_string(grid, should_emojify=False))
        print(f"Last reward: {self._last_reward}")
        print("---  ======== ---")
        print()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsType:
        self.rob.y, self.rob.x = self.player_init
        self.grid = self.get_init_grid()
        self.rob_loc_history = [(self.rob.y, self.rob.x)]
        self.elapsed_steps = 0
        if self._is_recording:
            self._record_step()
        return self.get_obs()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.elapsed_steps += 1
        if action != 4:
            dest = self.get_dest(action)
            is_interact = False
        else:
            dest = None
            is_interact = True

        if dest is not None:
            gets_dirt = (self.grid[dest] == '.')
            has_won = (self.grid[dest] == 'G')
            if self.grid[dest] == 'V':
                destroys_vase = True
                self.grid[dest] = '.'
            else:
                destroys_vase = False
            self.rob.y = dest[0]
            self.rob.x = dest[1]
            if gets_dirt:
                self.grid[dest] = ' '
        else:
            gets_dirt = False
            has_won = False
            destroys_vase = False
            if is_interact:
                self.do_interactions()

        obs = self.get_obs()
        reward = (int(gets_dirt) * self.dirt_value) + \
                 (-0.1 * (dest is None and not is_interact)) + \
                 (10 * has_won)
        self._last_reward = reward
        done = (self.elapsed_steps >= self.max_steps) or has_won
        info = {}
        if self._is_recording:
            self._record_step()
        return obs, reward, done, info

    def do_interactions(self):
        y = self.rob.y
        x = self.rob.x
        adjacent = [
            (y + 1, x),
            (y - 1, x),
            (y, x + 1),
            (y, x - 1)
        ]
        adjacent = filter(self.is_loc_in_bounds_tuple, adjacent)
        for (y1, x1) in adjacent:
            self.interact(y1, x1)

    def interact(self, y, x):
        if self.grid[y, x] == '|':
            self.grid[y, x] = '/'
        elif self.grid[y, x] == '/':
            self.grid[y, x] = '|'

    def get_grid_with_rob(self):
        grid = self.grid.copy()
        grid[self.rob.y, self.rob.x] = 'R'
        return grid

    def get_obs(self) -> np.array:
        grid = self.get_grid_with_rob()
        grid2 = vchar_to_pixel(grid)
        grid3 = np.stack(grid2)
        return grid3

    def start_recording(self):
        assert not self._is_recording
        self._is_recording = True

    def _record_step(self):
        self._im_history.append(self.get_obs())

    def stop_and_log_recording(self, episode_number: int) -> None:
        self._is_recording = False
        self._recordings[episode_number] = self._im_history
        self._im_history = []

    def get_recordings(self) -> dict[list[np.array]]:
        return self._recordings

    def get_dest(self, action):
        assert action in [0, 1, 2, 3], f"Action {action} not valid"
        displacements = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1])
        }
        (y, x) = np.array([self.rob.y, self.rob.x]) + displacements[action]
        if self.is_loc_in_bounds(y, x) and self.grid[y, x] in self.passable_objects:
            return (y, x)
        else:
            return None

    def is_loc_in_bounds_tuple(self, loc: (int, int)) -> bool:
        return self.is_loc_in_bounds(*loc)

    def is_loc_in_bounds(self, y: int, x: int) -> bool:
        return (0 <= x < self.width) and (0 <= y < self.height)


# TODO: Redo non-museum grids
class EmptyGrid(Grid):
    def get_vase_locations(self) -> list:
        return []

    def get_new_dirt_locations(self):
        return []

    def get_new_wall_and_door_locations(self):
        return [], []

    def __init__(self, height=10, width=16, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)


class EmptyGrid1D(EmptyGrid):
    def __init__(self):
        super().__init__(height=1, width=5, player_init=(0, 0), goal_loc=None)


class TinyEmptyGrid(EmptyGrid):
    def __init__(self, height=3, width=4, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)


class SmallEmptyGrid(EmptyGrid):
    def __init__(self, height=5, width=8, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)


class DirtGrid(Grid):
    def get_vase_locations(self) -> list:
        return []

    def __init__(self, height=5, width=8, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)

    def get_new_wall_and_door_locations(self):
        return [], []

    def get_new_dirt_locations(self):
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


class RandDirtGrid(DirtGrid):

    def get_object_locations_WDDV(self) -> (list, list, list, list):
        return [], [], self.get_new_dirt_locations(), []

    def get_new_dirt_locations(self, num_dirts=5):
        h = self.height
        w = self.width
        locs = []
        while locs == [] or self.player_init in locs or self.goal_loc in locs:
            ys = np.random.randint(low=0, high=h, size=num_dirts)
            xs = np.random.randint(low=0, high=w, size=num_dirts)
            locs = list(zip(ys, xs))
        return locs


class SmallRandDirtGrid(RandDirtGrid):

    def __init__(self, height=3, width=5, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)


class WallGrid(Grid):
    def get_vase_locations(self) -> list:
        return []

    def __init__(self, height=10, width=16, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)

    def get_new_dirt_locations(self):
        return []

    def get_new_wall_and_door_locations(self):
        h = self.height
        w = self.width
        wall_xs = range(1, w - 2, 2)
        gap_locs = [h - 1 if j % 2 == 0 else 0 for j in range(len(wall_xs))]
        locs = []
        for i, x in enumerate(wall_xs):
            # Fill all spaces except for the gap
            gap = gap_locs[i]
            locs += [(y, x) for y in range(h) if y != gap]
        return locs, []


class SemiRandWallGrid(WallGrid):

    def get_new_wall_and_door_locations(self):
        h = self.height
        w = self.width
        wall_xs = range(1, w - 2, 2)
        gap_locs = np.random.randint(low=0, high=h, size=len(wall_xs))
        locs = []
        for i, x in enumerate(wall_xs):
            # Fill all spaces except for the gap
            gap = gap_locs[i]
            locs += [(y, x) for y in range(h) if y != gap]
        return locs, []


class DoorGrid(SemiRandWallGrid):

    def __init__(self, height=10, width=16, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)

    def get_new_dirt_locations(self):
        return []

    def get_new_wall_and_door_locations(self):
        h = self.height
        w = self.width
        wall_xs = range(1, w - 2, 2)
        door_ys = [h - 1 if j % 2 == 0 else 0 for j in range(len(wall_xs))]
        door_xs = range(1, w - 2, 2)
        door_locs = list(zip(door_ys, door_xs))
        locs = []
        for i, x in enumerate(wall_xs):
            # Fill all spaces except for the gap
            gap = door_ys[i]
            locs += [(y, x) for y in range(h) if y != gap]
        return locs, door_locs


class VaseGrid(EmptyGrid):

    def __init__(self, height=10, width=16, player_init=(0, 0), goal_loc=None, max_steps=100):
        super().__init__(height, width, player_init, goal_loc, max_steps)

    def get_vase_locations(self):
        num_vases = int(self.height * self.width / 10)
        ys = np.random.choice(list(range(self.height)), size=num_vases)
        xs = np.random.choice(list(range(self.width)), size=num_vases)
        locs = set(zip(ys, xs))
        locs = locs - {self.player_init, self.goal_loc}
        return list(locs)


class MuseumGrid(Grid):

    def __init__(self, room_size=3, num_rooms_wide=2):
        self.room_size = room_size
        self.num_rooms_wide = num_rooms_wide
        width = ((self.room_size + 1) * (self.num_rooms_wide)) + 1
        height = width
        player_init = (1, 1)  # (int(height/2), int(width/2))

        super().__init__(height,
                         width,
                         player_init,
                         goal_loc=(height - 2, width - 2),
                         max_steps=100,
                         dirt_value=1.0)

    def get_new_wall_and_door_locations(self) -> (list, list):
        outer_boundaries = \
            [(0, x) for x in range(self.width)] + \
            [(self.height - 1, x) for x in range(self.width)] + \
            [(y, 0) for y in range(self.height)] + \
            [(y, self.width - 1) for y in range(self.height)]

        horizontal_boundary_ys = [(self.room_size + 1) * i for i in range(1, self.num_rooms_wide)]
        vertical_boundary_xs = [(self.room_size + 1) * i for i in range(1, self.num_rooms_wide)]
        door_locations = []
        wall_locations = outer_boundaries
        for y in horizontal_boundary_ys:
            for x in range(self.width):
                if x % (self.room_size + 1) == int((self.room_size + 1) / 2):
                    door_locations.append((y, x))
                else:
                    wall_locations.append((y, x))
        for x in vertical_boundary_xs:
            for y in range(self.height):
                if y % (self.room_size + 1) == int((self.room_size + 1) / 2):
                    door_locations.append((y, x))
                else:
                    wall_locations.append((y, x))
        # wall_locations += [(y, x) for x in range(self.width) for y in horizontal_boundary_ys]
        # wall_locations += [(y, x) for y in range(self.height) for x in vertical_boundary_xs]
        return wall_locations, door_locations

    def get_object_locations_WDDV(self) -> (list, list, list, list):
        wall_locations, door_locations = self.get_new_wall_and_door_locations()
        dirt_locations, vase_locations = self.get_dirt_vase_locations()
        return wall_locations, door_locations, dirt_locations, vase_locations

    def get_dirt_vase_locations(self) -> (list, list):
        room_corners = [
            (1 + (room_y * (self.room_size + 1)), 1 + (room_x * (self.room_size + 1)))
            for room_x in range(self.num_rooms_wide)
            for room_y in range(self.num_rooms_wide)
        ]
        room_locations = [
            [(y + y1, x + x1) for y1 in range(self.room_size) for x1 in range(self.room_size)]
            for (y, x) in room_corners
        ]
        room_locations = [[loc for loc in room if loc not in [self.player_init, self.goal_loc]]
                          for room in room_locations]
        vases_per_room = 1
        dirt_per_room = 2
        num_locs = vases_per_room + dirt_per_room
        vases = []
        dirts = []
        for room in room_locations:
            loc_inds = np.random.choice(len(room), size=num_locs, replace=False)
            locs = [room[loc_ind] for loc_ind in loc_inds]
            vases += locs[0:vases_per_room]
            dirts += locs[vases_per_room:]
        return dirts, vases


class SmallMuseumGrid(MuseumGrid):
    def __init__(self):
        super().__init__(room_size=3, num_rooms_wide=1)
