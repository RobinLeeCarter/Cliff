import numpy as np

import common
from environment.grid import grid


class GridWorld:
    def __init__(self, grid_: grid.Grid, rng: np.random.Generator):
        self.grid: grid.Grid = grid_
        self.rng: np.random.Generator = rng
        self.random_wind_choices = np.array([-1, 0, 1], dtype=int)

    def get_a_start_position(self) -> common.XY:
        return self.grid.start

    def is_at_goal(self, grid_position: common.XY) -> bool:
        return grid_position == self.grid.goal

    # noinspection PyUnusedLocal
    def get_wind(self, position: common.XY) -> common.XY:
        grid_wind = self.grid.upward_wind[position.x]
        random_wind = self.rng.choice(self.random_wind_choices)
        total_wind = grid_wind + random_wind
        return common.XY(x=0, y=total_wind)

    def get_square(self, position: common.XY) -> common.Square:
        if position == self.grid.start:
            return common.Square.START
        elif position == self.grid.goal:
            return common.Square.END
        else:
            return common.Square.NORMAL
