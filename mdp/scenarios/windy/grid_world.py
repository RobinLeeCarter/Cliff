from __future__ import annotations

import numpy as np

from mdp import common
from mdp.scenarios.position_move import grid_world
from mdp.scenarios.windy import environment_parameters


class GridWorld(grid_world.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        super().__init__(environment_parameters_)
        self.upward_wind: np.ndarray = environment_parameters_.upward_wind
        self.random_wind: bool = environment_parameters_.random_wind
        self._random_wind_choices: np.ndarray = environment_parameters_.random_wind_choices

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
