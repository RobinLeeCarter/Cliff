from __future__ import annotations

import numpy as np

from mdp import common
from mdp.task.cliff.model.environment_parameters import EnvironmentParameters
from mdp.task.cliff.model.environment import Environment


def grid_test() -> bool:
    environment_parameters = EnvironmentParameters(
        actions_list=common.ActionsList.FOUR_MOVES
    )
    environment = Environment(environment_parameters)
    grid_world_ = environment.grid_world
    shape = grid_world_.max_y + 1, grid_world_.max_x + 1
    cartesian_grid = np.empty(shape=shape, dtype=int)
    # noinspection PyTypeChecker
    for y, x in np.ndindex(cartesian_grid.shape):
        position: common.XY = common.XY(x, y)
        square: int = grid_world_.get_square(position)
        cartesian_grid[y, x] = square

    print(cartesian_grid)
    return True


if __name__ == '__main__':
    if grid_test():
        print("Passed")
