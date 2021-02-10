from __future__ import annotations

import numpy as np

import environment
from environments.windy import actions, grid_world


class Windy(environment.Environment):
    def __init__(self, random_wind: bool = False, verbose: bool = False):
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int)
        upward_wind = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        ], dtype=np.int)

        grid_world_ = grid_world.GridWorld(grid, upward_wind, random_wind)
        actions_ = actions.Actions()
        super().__init__(grid_world_, actions_, verbose)

    def _get_response(self) -> environment.Response:
        return environment.Response(
            reward=-1.0,
            state=self._projected_state
        )
