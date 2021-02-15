from __future__ import annotations

import numpy as np

import common
import environments


def grid_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    cliff = environments.factory(environment_parameters)
    grid_world_ = cliff.grid_world
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
