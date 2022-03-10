from __future__ import annotations

import numpy as np

GRID = np.array([
    [3, 0, 0, 2, 0, 0, 3]
], dtype=np.int)

V_OPTIMAL = np.array([
    [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0]
], dtype=np.float)

RANDOM_MOVE_CHOICES = np.array(
    [-1, 1]
    , dtype=int)
