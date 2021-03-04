from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model import environment
from mdp.scenarios.common.model import position_move
from mdp.scenarios.windy import grid_world


class Environment(position_move.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int)
        upward_wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=np.int)
        random_wind = environment_parameters.random_wind

        grid_world_ = grid_world.GridWorld(grid, upward_wind, random_wind)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> environment.Response:
        return environment.Response(
            reward=-1.0,
            state=self._new_state
        )
