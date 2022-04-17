from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

import numpy as np

from mdp import common
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.placeholder_action import PlaceholderAction
from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.value_function.base_value_function import BaseValueFunction

State = TypeVar('State', bound=NonTabularState)


class StateFunction(Generic[State], BaseValueFunction, ABC):
    def __init__(self,
                 feature: Optional[BaseFeature[State, PlaceholderAction]],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        super().__init__(feature, value_function_parameters)
        self._feature: Optional[BaseFeature[State, PlaceholderAction]] = feature

    @abstractmethod
    def __getitem__(self, state: State) -> float:
        if state.is_terminal:
            return 0.0
        else:
            raise Exception("Not implemented")

    @abstractmethod
    def get_gradient(self, state: State) -> np.ndarray:
        pass

    def calc_value(self, feature_vector: np.ndarray) -> float:
        raise Exception("Not implemented")

    def calc_gradient(self, feature_vector: np.ndarray) -> np.ndarray:
        raise Exception("Not implemented")
