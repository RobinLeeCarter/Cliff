from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from mdp import common

if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.base_feature import BaseFeature


class FeatureWeights:
    def __init__(self,
                 feature: BaseFeature,
                 initial_value: float = 0.0
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param initial_value: initial value for weights
        """
        self._initial_value: float = initial_value
        self._size: int = feature.max_size
        # weights
        self._w: np.ndarray = np.full(shape=self._size, fill_value=self._initial_value, dtype=float)

    # @property
    # def size(self) -> int:
    #     return self._size

    @property
    def w(self) -> np.ndarray:
        return self._w

    def reset(self):
        self._w: np.ndarray = self._w.fill(self._initial_value)

    def update_weights(self, delta_w: np.ndarray):
        self._w += delta_w

    def update_weights_sparse(self, indices: np.ndarray, delta_w: float):
        self._w[indices] += delta_w
