from __future__ import annotations

import numpy as np

import environment

_CLIFF_ARRAY = np.array([
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
], dtype=np.int32)

CLIFF_GRID = environment.Grid(
  grid_array=_CLIFF_ARRAY
)
