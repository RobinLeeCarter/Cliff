from __future__ import annotations

import numpy as np

import common
from mdp.model import environment


class GridWorld(environment.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, grid_array: np.ndarray):
        super().__init__(grid_array)
        self._random_move_choices = np.array([-1, 1], dtype=int)

    def change_request(self, current_position: common.XY, move: common.XY) -> common.XY:
        random = self._get_random_movement()
        requested_position: common.XY = common.XY(
            x=current_position.x + move.x + random.x,
            y=current_position.y + move.y + random.y
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
