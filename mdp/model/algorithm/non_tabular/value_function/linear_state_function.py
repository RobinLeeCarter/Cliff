from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.feature.feature import Feature
from mdp.model.algorithm.non_tabular.value_function.state_function import StateFunction


class LinearStateFunction(StateFunction):
    def __init__(self,
                 feature: Feature,
                 initial_value: float
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param initial_value: initial value for the weights (else zero)
        """
        self.feature: Feature = feature
        self.initial_value: float = initial_value

        self.size: int = self.feature.max_size
        # weights
        self.w: np.ndarray = np.full(shape=self.size, fill_value=initial_value, dtype=float)
        self.is_sparse: bool = self.feature.is_sparse

    def __getitem__(self, state: NonTabularState) -> float:
        x = self.feature[state]
        if self.is_sparse:
            result: float = self.w[x]
        else:
            result: float = float(np.dot(self.w, x))
        return result