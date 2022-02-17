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
        return self._w_dot_product(x)

    # TODO: DRY: function is repeated in state_action function
    def _w_dot_product(self, x: np.ndarray) -> float:
        if self.feature.is_sparse:
            return self.w[x]
        else:
            return float(np.dot(self.w, x))
