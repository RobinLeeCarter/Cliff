from __future__ import annotations
from typing import Optional

import numpy as np

from mdp import common
from mdp.task.position_move.model import grid_world

from mdp.task.random_walk.model.environment_parameters import EnvironmentParameters


class GridWorld(grid_world.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, environment_parameters: EnvironmentParameters):
        super().__init__(environment_parameters)
        self._v_optimal: np.ndarray = environment_parameters.v_optimal
        # noinspection PyTypeChecker
        random_move_choices: list[int] = environment_parameters.random_move_choices
        self._random_move_distribution: common.UniformMultinoulli[int] =\
            common.UniformMultinoulli[int](random_move_choices)

    def change_request(self, current_position: common.XY, move: Optional[common.XY]) -> common.XY:
        move = self._get_random_movement()
        requested_position: common.XY = common.XY(
            x=current_position.x + move.x,
            y=current_position.y + move.y
        )
        # project back to grid if outside
        new_position: common.XY = self.project_back_to_grid(requested_position)
        return new_position

    def _get_random_movement(self) -> common.XY:
        x_random: int = self._random_move_distribution.draw_one()
        return common.XY(
            x=x_random,
            y=0
        )

    def get_optimum(self, position: common.XY) -> float:
        index = self._position_to_index(position)
        return self._v_optimal[index]
