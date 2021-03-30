from __future__ import annotations

import numpy as np

from mdp import common
from mdp.scenarios.factory import environment_factory


def grid_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.ScenarioType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    cliff = environment_factory.environment_factory(environment_parameters)
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
