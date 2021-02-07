import numpy as np

from environment import grid

_CLIFF_ARRAY = np.array([
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
], dtype=np.int32)

CLIFF_GRID = grid.Grid(
  grid_array=_CLIFF_ARRAY
)
