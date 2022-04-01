from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from mdp import common

if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.value_function.state.state_function import StateFunction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.placeholder_action import PlaceholderAction

State = TypeVar('State', bound=NonTabularState)


class LinearStateFunction(StateFunction[State],
                          value_function_type=common.ValueFunctionType.LINEAR_STATE):
    def __init__(self,
                 feature: BaseFeature[State, PlaceholderAction],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        super().__init__(feature, value_function_parameters)

        self.size: int = self._feature.max_size
        # weights
        self.w: np.ndarray = np.full(shape=self.size, fill_value=self._initial_value, dtype=float)

    def __getitem__(self, state: State) -> float:
        self._feature.set_state(state)
        return self._feature.dot_product_full_vector(self.w)

