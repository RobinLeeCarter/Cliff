from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from mdp import common

if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.feature import Feature
from mdp.model.non_tabular.value_function.state_function import StateFunction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.placeholder_action import PlaceholderAction

State = TypeVar('State', bound=NonTabularState)


class LinearStateFunction(StateFunction[State]):
    def __init__(self,
                 feature: Feature[State, PlaceholderAction],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        super().__init__(feature, value_function_parameters)

        self.size: int = self.feature.max_size
        # weights
        self.w: np.ndarray = np.full(shape=self.size, fill_value=self.initial_value, dtype=float)

    def __getitem__(self, state: State) -> float:
        self.feature.set_state(state)
        return self.feature.dot_product_full_vector(self.w)

