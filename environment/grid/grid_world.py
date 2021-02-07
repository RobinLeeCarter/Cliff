import numpy as np

import common
from environment.grid import grid


class GridWorld:
    def __init__(self, grid_: grid.Grid, rng: np.random.Generator):
        self.grid: grid.Grid = grid_
        self.rng: np.random.Generator = rng
        self.max_y: int = self.grid.track.shape[0] - 1
        self.max_x: int = self.grid.track.shape[1] - 1
        self.starts: np.ndarray = (self.grid.track[:, :] == common.Square.START)
        self.starts_flat: np.ndarray = np.flatnonzero(self.starts)

    def get_a_start_position(self) -> common.XY:
        start_flat = self.rng.choice(self.starts_flat)
        iy, ix = np.unravel_index(start_flat, shape=self.grid.track.shape)
        return common.XY(x=ix, y=self.max_y - iy)

    def is_at_goal(self, position: common.XY) -> bool:
        return self.get_square(position) == common.Square.END

    def get_square(self, position: common.XY) -> common.Square:
        value: int = self.grid.track[self.max_y - position.y, position.x]
        # noinspection PyArgumentList
        return common.Square(value)  # pycharm inspection bug
