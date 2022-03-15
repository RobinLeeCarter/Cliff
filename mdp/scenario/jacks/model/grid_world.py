from __future__ import annotations

from typing import Optional

import numpy as np

from mdp import common
from mdp.model.tabular.environment import grid_world
from mdp.scenario.jacks.comparison.environment_parameters import EnvironmentParameters


class GridWorld(grid_world.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, environment_parameters: EnvironmentParameters):
        environment_parameters.grid = self._create_grid(environment_parameters)
        super().__init__(environment_parameters)
        self._max_cars: int = environment_parameters.max_cars
        self._max_transfers: int = environment_parameters.max_transfers

    @property
    def policy_min(self) -> int:
        return -self._max_transfers

    @property
    def policy_max(self) -> int:
        return self._max_transfers

    def _create_grid(self, environment_parameters_: EnvironmentParameters) -> np.ndarray:
        max_cars = environment_parameters_.max_cars
        grid: np.ndarray = np.zeros(shape=(max_cars+1, max_cars+1), dtype=int)
        return grid

    def change_request(self, current_position: common.XY, move: Optional[common.XY]) -> common.XY:
        raise NotImplementedError
