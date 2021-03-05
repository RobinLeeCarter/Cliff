from __future__ import annotations

import numpy as np

from mdp import common
from mdp.model import environment
from mdp.scenarios.racetrack import environment_parameters


class GridWorld(environment.GridWorld):
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        super().__init__(environment_parameters_)
        self._end_y: np.ndarray = (self._grid_array[:, self.max_x] == common.enums.Square.END)
        self.skid_probability: float = environment_parameters_.skid_probability

    def get_square(self, position: common.XY) -> common.Square:
        x_inside = (0 <= position.x <= self.max_x)
        y_inside = (0 <= position.y <= self.max_y)

        if y_inside and self._end_y[self.max_y - position.y] and position.x > self.max_x:
            # 'over' finish line (to the right of it)
            return common.Square.END
        elif not x_inside or not y_inside:
            # crash outside of track and not over finish line
            return common.Square.CLIFF
        else:
            # just whatever the track value is
            value: int = self._grid_array[self.max_y - position.y, position.x]
            # noinspection PyArgumentList
            return common.Square(value=value)   # docs say this is fine

    def change_request(self, position: common.XY, velocity: common.XY, acceleration: common.XY)\
            -> tuple[common.XY, common.XY]:
        if common.rng.uniform() > self.skid_probability:   # not skidding
            new_velocity = common.XY(
                x=velocity.x + acceleration.x,
                y=velocity.y + acceleration.y
            )
        else:  # skid
            new_velocity = velocity

        new_position: common.XY = common.XY(
            x=position.x + new_velocity.x,
            y=position.y + new_velocity.y
        )
        # project back to grid if outside
        # new_position: common.XY = self._project_back_to_grid(expected_position)
        return new_position, new_velocity

