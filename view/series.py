from dataclasses import dataclass

import numpy as np


@dataclass
class Series:
    title: str
    values: np.ndarray
    algorithm_type: type = None
