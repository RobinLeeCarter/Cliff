import numpy as np

import common
from environment import grid
import data

rng: np.random.Generator = np.random.default_rng()


def grid_test() -> bool:
    grid_world_ = grid.GridWorld(data.GRID_1, rng)
    grid_: grid.Grid = grid_world_.grid
    shape = grid_.max_y + 1, grid_.max_x + 1
    wind_grid = np.empty(shape=shape, dtype=int)
    for iy, ix in np.ndindex(wind_grid.shape):
        position: common.XY = common.XY(
            x=ix,
            y=grid_.max_y - iy
        )
        wind: common.XY = grid_world_.get_wind(position)
        wind_grid[iy, ix] = wind.y

    print(wind_grid)
    return True


if __name__ == '__main__':
    if grid_test():
        print("Passed")
