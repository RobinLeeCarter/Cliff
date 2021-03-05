from __future__ import annotations

import numpy as np

GRID_1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int)

UPWARD_WIND_1 = np.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    dtype=np.int)

RANDOM_WIND_CHOICES_1 = np.array(
    [-1, 0, 1]
    , dtype=int)
