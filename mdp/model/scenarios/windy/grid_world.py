from __future__ import annotations

import numpy as np

import common
from mdp.model import environment


class GridWorld(environment.GridWorld):
    """GridWorld doesn'_t know about states and actions it just deals in the rules of the grid"""
    def __init__(self, grid_array: np.ndarray, upward_wind: np.ndarray, random_wind: bool = False):
        super().__init__(grid_array)
        self.upward_wind: np.ndarray = upward_wind
        self.random_wind: bool = random_wind

        self._random_wind_choices = np.array([-1, 0, 1], dtype=int)

    def change_request(self, current_position: common.XY, move: common.XY) -> common.XY:
        wind = self._get_wind(current_position)
        requested_position: common.XY = common.XY(
            x=current_position.x + move.x + wind.x,
            y=current_position.y + move.y + wind.y
        )
        # project back to grid if outside
        new_position: common.XY = self._project_back_to_grid(requested_position)
        return new_position

    def _get_wind(self, current_position: common.XY) -> common.XY:
        extra_wind: int
        if self.random_wind:
            extra_wind = common.rng.choice(self._random_wind_choices)
        else:
            extra_wind = 0

        wind = common.XY(
            x=0,
            y=self.upward_wind[current_position.x] + extra_wind
        )
        return wind
