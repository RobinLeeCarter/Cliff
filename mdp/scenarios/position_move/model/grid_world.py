from __future__ import annotations
from typing import Optional

from mdp import common
from mdp.model.environment import grid_world


class GridWorld(grid_world.GridWorld):
    def change_request(self, current_position: common.XY, move: Optional[common.XY]) -> common.XY:
        if move is None:
            requested_position: common.XY = current_position
        else:
            requested_position: common.XY = common.XY(
                x=current_position.x + move.x,
                y=current_position.y + move.y
            )
        # project back to grid if outside
        new_position: common.XY = self.project_back_to_grid(requested_position)
        return new_position
