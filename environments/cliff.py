from __future__ import annotations

import numpy as np

import common
import environment


class Cliff(environment.Environment):
    def __init__(self, verbose: bool = False):
        grid_array = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
        ], dtype=np.int)
        grid_world = environment.GridWorld(grid_array)
        actions = CliffActions()
        super().__init__(grid_world, actions, verbose)

    def _get_response(self) -> environment.Response:
        if self._square == common.Square.CLIFF:
            return environment.Response(
                reward=-100.0,
                state=self.get_a_start_state()
            )
        else:
            return environment.Response(
                reward=-1.0,
                state=self._projected_state
            )


class CliffActions(environment.Actions):
    def _build_action_list(self):
        self._four_actions()
        # self._four_friendly_actions()
        # self._kings_moves()
        # self._kings_moves(include_center=True)
