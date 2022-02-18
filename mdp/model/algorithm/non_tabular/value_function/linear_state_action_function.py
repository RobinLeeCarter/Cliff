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
        self.feature.state = state
        self.feature.action = action
        return self.feature.dot_product_full_vector(self.w)

    def get_action_values(self, state: NonTabularState, actions: list[NonTabularAction]) -> np.ndarray:
        values: list[float] = []
        # set state just once
        self.feature.state = state
        for action in actions:
            self.feature.action = action
            # will calculate vector and then use it
            value = self.feature.dot_product_full_vector(self.w)
            values.append(value)
        return np.array(values)
