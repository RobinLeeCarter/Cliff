from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction
    from mdp.model.feature.feature import Feature
from mdp.model.algorithm.non_tabular.value_function.state_action_function import StateActionFunction


class LinearStateActionFunction(StateActionFunction):
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

    def __getitem__(self, state: NonTabularState, action: NonTabularAction) -> float:
        x = self.feature[state, action]
        return self._w_dot_product(x)

    def set_state(self, state: NonTabularState):
        self.feature.state = state

    def get_action_values(self, actions: list[NonTabularAction]) -> np.ndarray:
        values: list[float] = []
        for action in actions:
            self.feature.action = action
            x = self.feature.x
            value = self._w_dot_product(x)
            values.append(value)
        return np.array(values)

    def _w_dot_product(self, x: np.ndarray) -> float:
        if self.feature.is_sparse:
            return self.w[x]
        else:
            return float(np.dot(self.w, x))
