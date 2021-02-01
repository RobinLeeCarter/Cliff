import numpy as np

import common
from environment import grid
import data

rng: np.random.Generator = np.random.default_rng()


def grid_test() -> bool:
    grid_world_ = grid.GridWorld(data.GRID_1, rng)
    grid_: grid.Grid = grid_world_.grid
    shape = grid_world_.max_y + 1, grid_world_.max_x + 1
    cartesian_grid = np.empty(shape=shape, dtype=common.Square)
    for y, x in np.ndindex(cartesian_grid.shape):
        position: common.XY = common.XY(x, y)
        square: common.Square = grid_world_.get_square(position)
        cartesian_grid[y, x] = square

    print(cartesian_grid)
    return True


if __name__ == '__main__':
    if grid_test():
        print("Passed")
