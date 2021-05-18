from __future__ import annotations

import numpy as np

import utils
from mdp import common
from mdp.scenarios.cliff.model.environment_parameters import default
from mdp.scenarios.cliff.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.cliff.model.environment import Environment


def grid_test() -> bool:
    environment_parameters = EnvironmentParameters(
        environment_type=common.ScenarioType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    utils.set_none_to_default(environment_parameters, default)
    environment = Environment(environment_parameters)
    grid_world_ = environment.grid_world
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
