from __future__ import annotations

import numpy as np

import common
import environment


class RandomWalk(environment.Environment):
    def __init__(self, verbose: bool = False):
        grid_array = np.array([
            [3, 0, 0, 2, 0, 0, 3]
        ], dtype=np.int)
        grid_world = environment.GridWorld(grid_array)
        actions = RandomWalkActions()
        super().__init__(grid_world, actions, verbose)

    def _get_response(self) -> environment.Response:
        reward: float
        if self._projected_state.position == common.XY(x=self.grid_world.max_x, y=0):
            reward = 1.0
        else:
            reward = 0.0

        return environment.Response(
            reward=reward,
            state=self._projected_state
        )


class RandomWalkActions(environment.Actions):
    def _build_action_list(self):
        # no actions
        pass
