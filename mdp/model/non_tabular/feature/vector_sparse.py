from __future__ import annotations
# from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from mdp.model.non_tabular.feature.vector import Vector


@dataclass
class VectorSparse(Vector):
    def dot_product(self, full_vector: np.ndarray) -> float:
        return float(np.sum(full_vector[self.vector]))
