import numpy as np

from environment import grid

_TRACK_1 = np.array([
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
], dtype=np.int32)

GRID_1 = grid.Grid(
  track=_TRACK_1
)
