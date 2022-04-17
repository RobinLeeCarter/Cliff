from __future__ import annotations
# from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np


@dataclass
class Vector:
    vector: np.ndarray

    def dot_product(self, full_vector: np.ndarray) -> float:
        return float(np.dot(self.vector, full_vector))
