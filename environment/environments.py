from data import grids
from environment import environment


Cliff = environment.Environment(grid_=grids.CLIFF_GRID)

# class Cliff(environment.Environment):
#     def __init__(self, verbose: bool = False):
#         super().__init__(grid_=data.CLIFF_GRID, verbose=verbose)
