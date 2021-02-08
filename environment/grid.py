from __future__ import annotations
from dataclasses import dataclass

import numpy as np


# maintains consistency with Windy GridWorld enabling extensions to Grid such as wind
@dataclass
class Grid:
    grid_array: np.ndarray
