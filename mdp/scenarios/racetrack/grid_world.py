from __future__ import annotations

import numpy as np

from mdp import common
from mdp.model import environment


class RacetrackGridWorld(environment.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, grid_array: np.ndarray):
        super().__init__(grid_array)
        self._end_y: np.ndarray = (self.grid_array[:, self.max_x] == common.enums.Square.END)

    def change_request(self, position: common.XY, velocity: common.XY, acceleration: common.XY) -> common.XY:
        requested_position: common.XY = common.XY(
            x=position.x + velocity.x + wind.x,
            y=position.y + velocity.y + wind.y
        )
        # project back to grid if outside
        new_position: common.XY = self._project_back_to_grid(requested_position)
        return new_position

