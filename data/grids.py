import numpy as np

import common
from environment import grid


GRID_1 = grid.Grid(
  max_x=9,
  max_y=6,
  start=common.XY(x=0, y=3),
  goal=common.XY(x=7, y=3),
  upward_wind=np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
)
