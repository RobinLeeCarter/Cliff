from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environment import grid
import common


class GridWorld:
    """GridWorld doesn't know about states and actions it just deals in the rules of the grid"""
    def __init__(self, grid_: grid.Grid):
        self.grid: grid.Grid = grid_
        self.max_y: int = self.grid.grid_array.shape[0] - 1
        self.max_x: int = self.grid.grid_array.shape[1] - 1
        starts: np.ndarray = (self.grid.grid_array[:, :] == common.Square.START)
        self._starts_flat: np.ndarray = np.flatnonzero(starts)
        self._single_start: Optional[common.XY] = None
        self._test_single_start()

    def _test_single_start(self):
        if len(self._starts_flat) == 1:
            iy, ix = np.unravel_index(self._starts_flat[0], shape=self.grid.grid_array.shape)
            self._single_start = self._position_flip(common.XY(ix, iy))

    def get_a_start_position(self) -> common.XY:
        if self._single_start:
            return self._single_start
        else:
            start_flat = common.rng.choice(self._starts_flat)
            iy, ix = np.unravel_index(start_flat, shape=self.grid.grid_array.shape)
            return self._position_flip(common.XY(ix, iy))

    def change_request(self, current: common.XY, request: common.XY) -> common.XY:
        current_index = self._position_flip(current)  # position
        request_index = self._move_flip(request)      # move

        requested_index: common.XY = common.XY(
            x=current_index.x + request_index.x,
            y=current_index.y + request_index.y
        )
        # project back to grid if outside
        projected_index: common.XY = self._project_back_to_grid(requested_index)
        return self._position_flip(projected_index)

    def is_at_goal(self, position: common.XY) -> bool:
        return self.get_square(position) == common.Square.END

    def get_square(self, position: common.XY) -> common.Square:
        value: int = self.grid.grid_array[self.max_y - position.y, position.x]
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