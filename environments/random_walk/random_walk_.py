from __future__ import annotations

import numpy as np

import common
import environment


class RandomWalk(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid = np.array([
            [3, 0, 0, 2, 0, 0, 3]
        ], dtype=np.int)
        grid_world_ = environment.GridWorld(grid)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> environment.Response:
        reward: float
        if self._projected_state.position == common.XY(x=self.grid_world.max_x, y=0):
            reward = 1.0
        else:
            reward = 0.0

        return environment.Response(
            reward=reward,
            state=self._projected_state
        )
