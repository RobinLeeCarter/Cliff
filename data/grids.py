import numpy as np

import common
from environment import grid

_TRACK_1 = np.array([
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
], dtype=np.int32)

GRID_1 = grid.Grid(
  start=common.XY(0, 0),
  goal=common.XY(10, 0),
  track=_TRACK_1
)
