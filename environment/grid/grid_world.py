import numpy as np

import common
from environment.grid import grid


class GridWorld:
    def __init__(self, grid_: grid.Grid, rng: np.random.Generator):
        self.grid: grid.Grid = grid_
        self.rng: np.random.Generator = rng
        self.max_y: int = self.grid.track.shape[0] - 1
        self.max_x: int = self.grid.track.shape[1] - 1

    def get_a_start_position(self) -> common.XY:
        return self.grid.start

    def is_at_goal(self, grid_position: common.XY) -> bool:
        return grid_position == self.grid.goal

    def get_square(self, position: common.XY) -> common.Square:
        value: int = self.grid.track[self.max_y - position.y, position.x]
        # noinspection PyArgumentList
        return common.Square(value)  # pycharm inspection bug
