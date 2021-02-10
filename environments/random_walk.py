from __future__ import annotations

import numpy as np

import environment


class RandomWalk(environment.Environment):
    def __init__(self, verbose: bool = False):
        grid_array = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
        ], dtype=np.int32)
        grid = environment.Grid(
            grid_array=grid_array
        )
        grid_world = environment.GridWorld(grid)
        super().__init__(grid_world_=grid_world, verbose=verbose)
