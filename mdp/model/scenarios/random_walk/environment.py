from __future__ import annotations

import numpy as np

from mdp import common
from mdp.model import environment
from mdp.model.scenarios.common import environment_state_position, state_position
from mdp.model.scenarios.random_walk import grid_world


class Environment(environment_state_position.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid = np.array([
            [3, 0, 0, 2, 0, 0, 3]
        ], dtype=np.int)
        v_optimal = np.array([
            [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
        ], dtype=np.float)
        grid_world_ = grid_world.GridWorld(grid, v_optimal)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> environment.Response:
        reward: float
        self._projected_state: state_position.State
        if self._projected_state.position == common.XY(x=self.grid_world.max_x, y=0):
            reward = 1.0
        else:
            reward = 0.0

        return environment.Response(
            reward=reward,
            state=self._projected_state
        )

    def get_optimum(self, state: environment.State) -> float:
        self.grid_world: grid_world.GridWorld
        state: state_position.State
        return self.grid_world.get_optimum(state.position)
