from __future__ import annotations

import data
import environment


Cliff = environment.Environment(grid_=data.CLIFF_GRID)

# class Cliff(environment.Environment):
#     def __init__(self, verbose: bool = False):
#         super().__init__(grid_=data.CLIFF_GRID, verbose=verbose)
