from __future__ import annotations
from typing import Optional

import numpy as np

from mdp import common


class GridWorld:
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, grid_array: np.ndarray):
        self.grid_array: np.ndarray = grid_array
        self.max_y: int = self.grid_array.shape[0] - 1
        self.max_x: int = self.grid_array.shape[1] - 1
        starts: np.ndarray = (self.grid_array[:, :] == common.Square.START)
        self._starts_flat: np.ndarray = np.flatnonzero(starts)
        self._single_start: Optional[common.XY] = None
        self._test_single_start()

        self.output_squares: np.ndarray = np.empty(shape=self.grid_array.shape, dtype=common.OutputSquare)
        # set_gridworld output_squares so don't have to test for existance.
        for index in np.ndindex(self.output_squares.shape):
            self.output_squares[index] = common.OutputSquare()

    def _test_single_start(self):
        if len(self._starts_flat) == 1:
            iy, ix = np.unravel_index(self._starts_flat[0], shape=self.grid_array.shape)
            self._single_start = self._position_flip(common.XY(ix, iy))

    def get_a_start_position(self) -> common.XY:
        if self._single_start:
            return self._single_start
        else:
            start_flat = common.rng.choice(self._starts_flat)
            iy, ix = np.unravel_index(start_flat, shape=self.grid_array.shape)
            return self._position_flip(common.XY(ix, iy))

    def change_request(self, current_position: common.XY, move: Optional[common.XY]) -> common.XY:
        if move is None:
            requested_position: common.XY = current_position
        else:
            requested_position: common.XY = common.XY(
                x=current_position.x + move.x,
                y=current_position.y + move.y
            )
        # project back to grid if outside
        new_position: common.XY = self._project_back_to_grid(requested_position)
        return new_position

    def is_at_goal(self, position: common.XY) -> bool:
        return self.get_square(position) == common.Square.END

    def get_square(self, position: common.XY) -> common.Square:
        value: int = self.grid_array[self.max_y - position.y, position.x]
        # noinspection PyArgumentList
        return common.Square(value)  # pycharm inspection bug

    def _project_back_to_grid(self, requested_position: common.XY) -> common.XY:
        x = requested_position.x
        y = requested_position.y
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.max_x:
            x = self.max_x
        if y > self.max_y:
            y = self.max_y
        return common.XY(x=x, y=y)

    # convert position (lower-left orgin) to/from numpy index (same in reverse)
    def _position_flip(self, xy_in: common.XY) -> common.XY:
        return common.XY(x=xy_in.x, y=self.max_y - xy_in.y)

    def _move_flip(self, xy_in: common.XY) -> common.XY:
        return common.XY(x=xy_in.x, y=-xy_in.y)

    def set_state_function(self, position: common.XY, v_value: Optional[float]):
        output_square: common.OutputSquare = self._get_output_square(position)
        output_square.v_value = v_value

    def set_state_action_function(self,
                                  position: common.XY,
                                  move: common.XY,
                                  q_value: Optional[float],
                                  is_policy: bool = False):
        output_square: common.OutputSquare = self._get_output_square(position)
        move_value: common.MoveValue = common.MoveValue(move, q_value, is_policy)
        output_square.move_values[move] = move_value

    def _get_output_square(self, position: common.XY) -> common.OutputSquare:
        np_index: common.XY = self._position_flip(position)
        output_square: common.OutputSquare = self.output_squares[np_index.y, np_index.x]
        return output_square
