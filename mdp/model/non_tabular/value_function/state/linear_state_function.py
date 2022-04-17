from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Optional

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
        self._feature_vector: Optional[np.ndarray] = None

    def __getitem__(self, state: State) -> float:
        value: float
        if state.is_terminal:
            value = 0.0
        else:
            self._feature_vector = self._feature[state]
            value = self.calc_value(self._feature_vector)
        return value

    def get_gradient(self, state: State) -> np.ndarray:
        self._feature_vector = self._feature[state]
        gradient = self.calc_gradient(self._feature_vector)
        return gradient

    def calc_value(self, feature_vector: np.ndarray) -> float:
        value: float = self._feature.dot_product(feature_vector, self.w)
        return value

    def calc_gradient(self, feature_vector: np.ndarray) -> np.ndarray:
        return feature_vector
