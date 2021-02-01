from dataclasses import dataclass

import numpy as np

import common


@dataclass
class Grid:
    max_x: int
    max_y: int
    start: common.XY      # x, y from bottom left
    goal: common.XY       # x, y from bottom left
    upward_wind: np.ndarray
