from __future__ import annotations

from typing import Optional

import numpy as np

from mdp import common
from mdp.scenarios.position_move import grid_world
from mdp.scenarios.random_walk import environment_parameters


class GridWorld(grid_world.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        super().__init__(environment_parameters_)
        self._v_optimal: np.ndarray = environment_parameters_.v_optimal
        self._random_move_choices: np.ndarray = environment_parameters_.random_move_choices

    def change_request(self, current_position: common.XY, move: Optional[common.XY]) -> common.XY:
        random = self._get_random_movement()
        requested_position: common.XY = common.XY(
            x=current_position.x + random.x,
            y=current_position.y + random.y
        )
        # project back to grid if outside
        new_position: common.XY = self._project_back_to_grid(requested_position)
        return new_position

    def _get_random_movement(self) -> common.XY:
        x_random: int = common.rng.choice(self._random_move_choices)
        return common.XY(
            x=x_random,
            y=0
        )

    def get_optimum(self, position: common.XY) -> float:
        index = self._position_to_index(position)
        return self._v_optimal[index]
