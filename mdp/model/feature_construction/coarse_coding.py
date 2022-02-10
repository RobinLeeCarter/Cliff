from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.state import State
from mdp.model.feature_construction.feature import Feature


class CoarseCoding(Feature, ABC):
    def __init__(self, max_size: Optional[int] = None):
        self._max_size: Optional[int] = max_size

    def get_x(self, state: State) -> np.ndarray:
        """return the full x vector (unlikely to be used)"""
        if self._max_size:
            x_sparse: np.ndarray = self.get_x_sparse(state)
            x = np.zeros(shape=self._max_size, dtype=np.int)
            x[x_sparse] = 1
            return x
        else:
            raise Exception("Size of x not specified")

    @abstractmethod
    def get_x_sparse(self, state: State) -> np.ndarray:
        """return just the indexes of x which are 1 (rest are 0)"""
        pass
