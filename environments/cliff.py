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
        ], dtype=np.int32)
        grid = common.Grid(
            grid_array=grid_array
        )
        super().__init__(grid_=grid, verbose=verbose)
