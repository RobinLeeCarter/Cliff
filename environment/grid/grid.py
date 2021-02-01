from dataclasses import dataclass

import numpy as np

import common


# unnecesary but maintains some consistency with Windy GridWorld
@dataclass
class Grid:
    start: common.XY      # x, y from bottom left
    goal: common.XY       # x, y from bottom left
    track: np.ndarray
