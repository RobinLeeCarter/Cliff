from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenarios.blackjack.model.environment_parameters import EnvironmentParameters

import numpy as np

from mdp import common
from mdp.model.environment.tabular import grid_world


class GridWorld(grid_world.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, environment_parameters: EnvironmentParameters, grid_shape: tuple[int, int]):
        environment_parameters.grid = self._create_grid(grid_shape)
        super().__init__(environment_parameters)

    @property
    def policy_min(self) -> int:
        return 0

    @property
    def policy_max(self) -> int:
        return 1

    def _create_grid(self, grid_shape: tuple[int, int]) -> np.ndarray:
        grid: np.ndarray = np.zeros(shape=grid_shape, dtype=int)
        return grid

    def change_request(self, current_position: common.XY, move: Optional[common.XY]) -> common.XY:
        raise NotImplementedError
