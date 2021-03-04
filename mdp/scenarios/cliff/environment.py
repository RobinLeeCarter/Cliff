from __future__ import annotations

import numpy as np

from mdp import common
from mdp.model import environment
from mdp.scenarios.common.model import position_move


class Environment(position_move.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
        ], dtype=np.int)
        grid_world_ = position_move.GridWorld(grid)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> environment.Response:
        if self._square == common.Square.CLIFF:
            return environment.Response(
                reward=-100.0,
                state=self._get_a_start_state()
            )
        else:
            return environment.Response(
                reward=-1.0,
                state=self._new_state
            )

