from __future__ import annotations

from typing import Optional

import numpy as np

from mdp import common
from mdp.model.environment import grid_world
from mdp.scenarios.jacks import environment_parameters


class GridWorld(grid_world.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        environment_parameters_.grid = self._create_grid(environment_parameters_)
        super().__init__(environment_parameters_)
        self._max_cars: int = environment_parameters_.max_cars
        self._max_transfers: int = environment_parameters_.max_transfers

    @property
    def policy_min(self) -> int:
        return -self._max_transfers

    @property
    def policy_max(self) -> int:
        return self._max_transfers

    def _create_grid(self, environment_parameters_: environment_parameters.EnvironmentParameters) -> np.ndarray:
        max_cars = environment_parameters_.max_cars
        grid: np.ndarray = np.zeros(shape=(max_cars+1, max_cars+1), dtype=int)
        return grid

    def change_request(self, current_position: common.XY, move: Optional[common.XY]) -> common.XY:
        raise NotImplementedError
        # requested_position: common.XY = common.XY(
        #     x=current_position.x,
        #     y=current_position.y
        # )
        # # project back to grid if outside
        # new_position: common.XY = self._project_back_to_grid(requested_position)
        # return new_position

